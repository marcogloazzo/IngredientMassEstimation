"""V4/V5: optional specialized MLPs and an auxiliary visual total-mass predictor.

The deployed masses remain p*c. The auxiliary head is supervision, NOT a
post-hoc rescaling rule. Flags permit separate, controlled ablations.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .conditional_model import ConditionalDepthHeads, conditional_mass_loss, training_statistics
from .multiscale_model import ConvNeXtMultiscaleDepth


def task_mlp(dropout):
    return nn.Sequential(
        nn.LayerNorm(768), nn.Linear(768, 384), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(384, 192), nn.GELU(), nn.Dropout(dropout),
    )


class SpecializedDepthHeads(ConditionalDepthHeads):
    def __init__(self, ingredient_count, config):
        super().__init__(ingredient_count, config)
        self.split_task_mlps = bool(config.get("split_task_mlps", True))
        self.has_auxiliary_mass = bool(config.get("auxiliary_total_mass", True))
        dropout = float(config.get("dropout", .3))
        if self.split_task_mlps:
            del self.shared
            self.presence_mlp = task_mlp(dropout)
            self.quantity_mlp = task_mlp(dropout)
        if self.has_auxiliary_mass:
            dimension = int(config["attention_dim"])
            hidden = int(config.get("mass_hidden_dim", 128))
            self.auxiliary_mass_head = nn.Sequential(
                nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )
            scale = float(config.get("mass_scale", 300.))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Auxiliary mass scale must be a finite positive train mean")
            self.register_buffer("auxiliary_gram_scale", torch.tensor(scale))
            nn.init.zeros_(self.auxiliary_mass_head[-1].weight)
            nn.init.constant_(self.auxiliary_mass_head[-1].bias, math.log(math.expm1(1.)))

    def forward_memory(self, memory, positions, return_attention=False):
        queries = self.queries.weight[None].expand(memory.shape[0], -1, -1)
        labels = torch.zeros_like(queries)
        attention = None
        for layer in self.decoder:
            labels, attention = layer(labels, memory, queries, positions, return_attention)
        projected = self.output_projection(labels)
        if self.split_task_mlps:
            presence_hidden, quantity_hidden = self.presence_mlp(projected), self.quantity_mlp(projected)
        else:
            presence_hidden = quantity_hidden = self.shared(projected)
        logits = self.presence_head(presence_hidden).float()
        probability = torch.sigmoid(logits - self.positive_weights.clamp(min=1).log())
        conditional = F.softplus(self.conditional_head(quantity_hidden).float()) * self.gram_scale
        masses = probability * conditional + 1e-8
        total = masses.sum(-1)
        outputs = {
            "presence_logits": logits, "presence_probabilities": probability,
            "raw_presence_probabilities": logits.sigmoid(), "conditional_grams": conditional,
            "conditional_class_weights": self.conditional_weights,
            "ingredient_masses": masses, "total_mass": total, "shares": masses / total[:, None],
        }
        if self.has_auxiliary_mass:
            # Pool spatial memory directly: not probabilities or predicted grams.
            auxiliary = F.softplus(self.auxiliary_mass_head(memory.mean(1)).float()).squeeze(-1)
            auxiliary = auxiliary * self.auxiliary_gram_scale
            outputs["auxiliary_total_mass"] = auxiliary
        if return_attention:
            outputs.update(label_features=labels, projected_label_features=projected,
                           presence_features=presence_hidden, quantity_features=quantity_hidden)
        return outputs, attention


class ConvNeXtSpecializedMass(ConvNeXtMultiscaleDepth):
    def __init__(self, ingredient_count, config, pretrained=True, torch_cache=None):
        if config.get("presence_loss") != "bce":
            raise ValueError("V4/V5 preserve V3 weighted BCE")
        if config.get("presence_gate") != "pos_weight_corrected_sigmoid" or int(config.get("image_size", 0)) != 320:
            raise ValueError("V4/V5 preserve prior-corrected presence gates and 320x320 geometry")
        weights = config["loss_weights"]
        if not config.get("auxiliary_total_mass", True) and any(float(weights.get(key, 0)) != 0 for key in ("aux_mass", "mass_consistency")):
            raise ValueError("Disable auxiliary loss weights when disabling its head")
        for key in ("aux_mass", "mass_consistency"):
            if not math.isfinite(float(weights.get(key, 0))) or float(weights.get(key, 0)) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        super().__init__(ingredient_count, config, pretrained, torch_cache, heads_class=SpecializedDepthHeads)

    def initialize_from_train(self, records, vocabulary):
        self.heads.initialize_from_train(training_statistics(records, vocabulary))
        if self.heads.has_auxiliary_mass:
            mean = sum(float(r["total_mass"]) for r in records if r["split"] == "train") / sum(r["split"] == "train" for r in records)
            self.heads.auxiliary_gram_scale.fill_(mean)


def specialized_mass_loss(outputs, presence, masses, total_mass, nutrients, matrix, positive_weights, config):
    base_config = {**config, "loss_weights": {
        key: value for key, value in config["loss_weights"].items()
        if key not in ("aux_mass", "mass_consistency")
    }}
    loss, parts = conditional_mass_loss(outputs, presence, masses, total_mass, nutrients,
                                        matrix, positive_weights, base_config)
    auxiliary = outputs.get("auxiliary_total_mass")
    if auxiliary is None:
        aux_loss = consistency = loss.new_zeros(())
    else:
        scale = float(config["mass_scale"])
        aux_loss = (auxiliary.float() - total_mass).abs().mean() / scale
        # Both paths receive gradients; no ground-truth mass enters forward().
        consistency = (auxiliary.float() - outputs["total_mass"].float()).abs().mean() / scale
    loss = loss + float(config["loss_weights"].get("aux_mass", 0)) * aux_loss
    loss = loss + float(config["loss_weights"].get("mass_consistency", 0)) * consistency
    parts.update(total=float(loss.detach()), aux_mass=float(aux_loss.detach()), mass_consistency=float(consistency.detach()))
    return loss, parts
