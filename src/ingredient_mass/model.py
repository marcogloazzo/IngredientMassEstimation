from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IngredientMassHeads(nn.Module):
    """Small heads trained on frozen image features."""

    def __init__(
        self,
        input_dim: int,
        ingredient_count: int,
        hidden_dim: int = 384,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ingredient_count = ingredient_count
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.presence_head = nn.Linear(hidden_dim, ingredient_count)
        self.amount_head = nn.Linear(hidden_dim, ingredient_count)
        self.mass_head = nn.Linear(hidden_dim, 1)
        # Start from a realistic positive plate mass instead of softplus(0) ~= 0.69 g.
        nn.init.zeros_(self.mass_head.weight)
        nn.init.constant_(self.mass_head.bias, 250.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(features)
        presence_logits = self.presence_head(hidden)
        amount_logits = self.amount_head(hidden)
        gates = torch.sigmoid(presence_logits)
        positive_amounts = F.softplus(amount_logits)
        unnormalized = gates * positive_amounts + 1e-8
        shares = unnormalized / unnormalized.sum(dim=-1, keepdim=True)
        total_mass = F.softplus(self.mass_head(hidden)).squeeze(-1)
        return {
            "presence_logits": presence_logits,
            "presence_probabilities": gates,
            "shares": shares,
            "total_mass": total_mass,
        }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    presence_target: torch.Tensor,
    shares_target: torch.Tensor,
    total_mass_target: torch.Tensor,
    nutrients_target: torch.Tensor,
    nutrient_matrix: torch.Tensor,
    positive_weights: torch.Tensor,
    loss_weights: dict[str, float],
    *,
    presence_loss_name: str = "bce",
    asl_config: dict | None = None,
    masses_target: torch.Tensor | None = None,
    ingredient_mass_scale: float = 300.0,
    mass_scale: float = 300.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if presence_loss_name == "bce":
        presence_loss = F.binary_cross_entropy_with_logits(
            outputs["presence_logits"].float(), presence_target, pos_weight=positive_weights
        )
    elif presence_loss_name == "asl":
        presence_loss = asymmetric_loss(outputs["presence_logits"], presence_target, **(asl_config or {}))
    else:
        raise ValueError("presence_loss_name must be 'bce' or 'asl'")

    share_errors = torch.abs(outputs["shares"] - shares_target)
    share_element_weights = 1.0 + 4.0 * presence_target
    share_loss = (share_errors * share_element_weights).sum() / share_element_weights.sum()

    if not math.isfinite(mass_scale) or mass_scale <= 0:
        raise ValueError("mass_scale must be finite and positive")
    mass_loss = torch.mean(torch.abs(outputs["total_mass"] - total_mass_target)) / mass_scale

    predicted_masses = outputs["shares"] * outputs["total_mass"].unsqueeze(1)
    ingredient_mass_loss = predicted_masses.new_zeros(())
    if loss_weights.get("ingredient_mass", 0.0) != 0:
        if masses_target is None:
            raise ValueError("Per-ingredient gram loss requires actual per-ingredient masses")
        if ingredient_mass_scale <= 0:
            raise ValueError("ingredient_mass_scale must be positive")
        # Per-plate sum includes present AND absent ingredients, then averages plates.
        ingredient_mass_loss = (predicted_masses - masses_target).abs().sum(dim=1).mean() / ingredient_mass_scale
    predicted_nutrients = predicted_masses @ nutrient_matrix
    nutrient_scales = torch.tensor(
        [500.0, 30.0, 80.0, 40.0],
        dtype=predicted_nutrients.dtype,
        device=predicted_nutrients.device,
    )
    nutrition_loss = torch.mean(
        torch.abs(predicted_nutrients - nutrients_target) / nutrient_scales
    )

    total = (
        loss_weights["presence"] * presence_loss
        + loss_weights["shares"] * share_loss
        + loss_weights["mass"] * mass_loss
        + loss_weights["nutrition"] * nutrition_loss
        + loss_weights.get("ingredient_mass", 0.0) * ingredient_mass_loss
    )
    parts = {
        "total": float(total.detach().item()),
        "presence": float(presence_loss.detach().item()),
        "shares": float(share_loss.detach().item()),
        "mass": float(mass_loss.detach().item()),
        "nutrition": float(nutrition_loss.detach().item()),
        "ingredient_mass": float(ingredient_mass_loss.detach().item()),
    }
    return total, parts


def mean_training_mass(masses) -> float:
    """Compute one fixed denominator from training targets, never per minibatch."""
    values = [float(value) for value in masses]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Training masses must be non-empty, finite and positive")
    return math.fsum(values) / len(values)


def asymmetric_loss(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05,
                    detach_focal_weight=True, eps=1e-8):
    """ASL multilabel formula, mean reduction to match our BCE loss scale.

    Reference: https://github.com/Alibaba-MIIL/ASL (ICCV 2021).
    Unlike a global grad-mode toggle, detach preserves validation/no-grad state.
    """
    if not 0 <= clip < 1 or gamma_neg < 0 or gamma_pos < 0:
        raise ValueError("Invalid ASL clipping or focal exponents")
    positive = logits.float().sigmoid()
    negative = (1.0 - positive + clip).clamp(max=1.0)
    log_probability = targets * positive.clamp(min=eps).log() + (1 - targets) * negative.clamp(min=eps).log()
    correct_probability = positive * targets + negative * (1 - targets)
    gamma = gamma_pos * targets + gamma_neg * (1 - targets)
    weight = (1 - correct_probability).clamp(min=0).pow(gamma)
    if detach_focal_weight:
        weight = weight.detach()
    return -(weight * log_probability).mean()


def load_checkpoint(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
