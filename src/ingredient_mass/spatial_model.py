"""Compact Query2Label adaptation with ingredient amount and total-mass tasks.

References (architecture, not compatible pretrained decoder checkpoints):
https://github.com/SlongLiu/query2labels
https://arxiv.org/abs/2107.10834
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


MODEL_TYPE = "convnext_small_query2label"
MODEL_TYPE_V2 = "convnext_small_query2label_multiscale_depth_v2"
MODEL_TYPE_V3 = "convnext_small_query2label_conditional_mass_v3"
MODEL_TYPE_V4 = "convnext_small_query2label_specialized_mass_v4"
MODEL_TYPE_V5 = "convnext_small_query2label_auxiliary_mass_v5"
MODEL_TYPE_V6 = "convnext_small_query2label_spatial_clip_asl_v6"
SPECIALIZED_MODEL_TYPES = (MODEL_TYPE_V4, MODEL_TYPE_V5, MODEL_TYPE_V6)
CONDITIONAL_MODEL_TYPES = (MODEL_TYPE_V3, *SPECIALIZED_MODEL_TYPES)
DEPTH_MODEL_TYPES = (MODEL_TYPE_V2, *CONDITIONAL_MODEL_TYPES)
SUPPORTED_MODEL_TYPES = (MODEL_TYPE, *DEPTH_MODEL_TYPES)


def protected_result_namespaces(model_type):
    """Older Drive namespaces are read-only inputs for a newer experiment."""
    return {MODEL_TYPE_V3: {"v2"}, MODEL_TYPE_V4: {"v2", "v3"},
            MODEL_TYPE_V5: {"v2", "v3", "v4"}, MODEL_TYPE_V6: {"v2", "v3", "v4", "v5"}}.get(model_type, set())


def position_encoding(height: int, width: int, dimension: int, device, dtype):
    """Deterministic 2-D sine/cosine positions; no learned image size."""
    if dimension % 4:
        raise ValueError("Attention dimension must be divisible by four")
    frequency = 10000 ** (-torch.arange(dimension // 4, device=device).float() / (dimension // 4))
    y, x = torch.meshgrid(
        torch.linspace(0, 2 * math.pi, height, device=device),
        torch.linspace(0, 2 * math.pi, width, device=device), indexing="ij",
    )
    x = x.flatten()[:, None] * frequency
    y = y.flatten()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).to(dtype)[None]


class LabelDecoderLayer(nn.Module):
    """Q2L label queries cross-attend to image regions, followed by an FFN.

    As in the Q2L no-self-attention configuration, labels do not self-attend.
    The reduced width/depth is specific to this low-compute adaptation.
    """

    def __init__(self, dimension, heads, feedforward, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, labels, memory, queries, positions, return_attention=False):
        attended, weights = self.attention(
            labels + queries, memory + positions, memory,
            need_weights=return_attention, average_attn_weights=True,
        )
        labels = self.norm1(labels + self.dropout(attended))
        return self.norm2(labels + self.dropout(self.ffn(labels))), weights


class GroupWiseLinear(nn.Module):
    """Independent output weights for each ingredient's representation."""

    def __init__(self, classes, dimension):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        self.bias = nn.Parameter(torch.zeros(classes))
        nn.init.uniform_(self.weight, -1 / math.sqrt(dimension), 1 / math.sqrt(dimension))

    def forward(self, labels):
        return (labels * self.weight).sum(dim=-1) + self.bias


class SpatialIngredientHeads(nn.Module):
    def __init__(self, ingredient_count, config):
        super().__init__()
        dimension = int(config.get("attention_dim", 256))
        heads = int(config.get("attention_heads", 4))
        layers = int(config.get("attention_layers", 1))
        if layers < 1 or dimension % heads or dimension % 4:
            raise ValueError("Invalid attention dimensions/heads/layers")
        self.input_projection = nn.Conv2d(768, dimension, 1)
        self.queries = nn.Embedding(ingredient_count, dimension)
        self.decoder = nn.ModuleList([
            LabelDecoderLayer(dimension, heads, int(config.get("attention_feedforward", 1024)),
                              float(config.get("attention_dropout", 0.1)))
            for _ in range(layers)
        ])
        # Restore 768 channels so the agreed shared MLP is exactly 768 -> 384 -> 192.
        self.output_projection = nn.Linear(dimension, 768)
        dropout = float(config.get("dropout", 0.3))
        self.shared = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 192), nn.GELU(), nn.Dropout(dropout),
        )
        self.presence_head = GroupWiseLinear(ingredient_count, 192)
        self.amount_head = GroupWiseLinear(ingredient_count, 192)
        self.mass_head = nn.Sequential(
            nn.Linear(192, int(config.get("mass_hidden_dim", 128))), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(int(config.get("mass_hidden_dim", 128)), 1),
        )
        nn.init.normal_(self.mass_head[-1].weight, std=0.001)
        nn.init.constant_(self.mass_head[-1].bias, 250.0)

    def forward(self, spatial, return_attention=False):
        batch, _, height, width = spatial.shape
        memory = self.input_projection(spatial).flatten(2).transpose(1, 2)
        positions = position_encoding(height, width, memory.shape[-1], memory.device, memory.dtype)
        outputs, weights = self.forward_memory(memory, positions, return_attention)
        if return_attention:
            outputs.update({"attention": weights.reshape(batch, -1, height, width),
                            "spatial_features": spatial})
        return outputs

    def forward_memory(self, memory, positions, return_attention=False):
        """Decode any spatial memory; preserve v1 head names/checkpoint layout."""
        batch = memory.shape[0]
        queries = self.queries.weight[None].expand(batch, -1, -1)
        labels = torch.zeros_like(queries)
        weights = None
        for layer in self.decoder:
            labels, weights = layer(labels, memory, queries, positions, return_attention)
        projected = self.output_projection(labels)
        hidden = self.shared(projected)
        # Compute gates, proportions, and mass in FP32 even with autocast enabled.
        logits = self.presence_head(hidden).float()
        gates = logits.sigmoid()
        amounts = F.softplus(self.amount_head(hidden).float())
        unnormalized = gates * amounts + 1e-8
        shares = unnormalized / unnormalized.sum(dim=-1, keepdim=True)
        total_mass = F.softplus(self.mass_head(hidden.mean(dim=1)).float()).squeeze(-1)
        outputs = {
            "presence_logits": logits, "presence_probabilities": gates,
            "shares": shares, "total_mass": total_mass,
            "ingredient_masses": shares * total_mass[:, None],
        }
        if return_attention:
            outputs.update({
                "label_features": labels,
                "projected_label_features": projected, "shared_features": hidden,
            })
        return outputs, weights


class ConvNeXtQuery2Label(nn.Module):
    def __init__(self, ingredient_count, config, pretrained=True, torch_cache: Path | None = None):
        super().__init__()
        from torchvision.models import ConvNeXt_Small_Weights, convnext_small

        if torch_cache is not None:
            torch_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(torch_cache.resolve())
        backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features
        self.final_norm = backbone.classifier[0]
        self.heads = SpatialIngredientHeads(ingredient_count, config)
        self.encoder_trainable = False
        self.set_encoder_trainable(False)

    def encoder_parameters(self):
        # Include the downsampling immediately preceding the final ConvNeXt stage.
        for module in (self.features[6:], self.final_norm):
            yield from module.parameters()

    def set_encoder_trainable(self, enabled):
        self.encoder_trainable = bool(enabled)
        self.features.requires_grad_(False)
        self.final_norm.requires_grad_(False)
        for parameter in self.encoder_parameters():
            parameter.requires_grad_(self.encoder_trainable)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in self.features[:6]:
            module.eval()
        if not self.encoder_trainable:
            for module in self.features[6:]:
                module.eval()
            self.final_norm.eval()
        return self

    def forward(self, images, return_attention=False):
        with torch.no_grad():
            frozen = self.features[:6](images)
        spatial = self.final_norm(self.features[6:](frozen))
        return self.heads(spatial, return_attention)


def image_transforms(training=False):
    """Match torchvision ConvNeXt Small geometry; augment training only."""
    from torchvision.models import ConvNeXt_Small_Weights
    from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter

    deterministic = ConvNeXt_Small_Weights.DEFAULT.transforms()
    if not training:
        return deterministic
    return Compose([
        RandomHorizontalFlip(), ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        deterministic,
    ])


def model_from_checkpoint(checkpoint, device):
    if checkpoint.get("model_type") not in SUPPORTED_MODEL_TYPES:
        raise ValueError("This script requires a spatial Query2Label checkpoint, not cached-feature heads.")
    # All encoder weights are in the checkpoint; inference must not download ImageNet again.
    config = dict(checkpoint["config"])
    config["model_type"] = checkpoint["model_type"]
    model = create_image_model(len(checkpoint["vocabulary"]["entries"]), config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if checkpoint["model_type"] == MODEL_TYPE_V6 and not bool(model.heads.clip_initialized.item()):
        raise ValueError("V6 checkpoint has not initialized its CLIP queries")
    return model.to(device).eval()


def create_image_model(ingredient_count, config, pretrained=True, torch_cache=None):
    model_type = config.get("model_type", MODEL_TYPE)
    if model_type == MODEL_TYPE:
        return ConvNeXtQuery2Label(ingredient_count, config, pretrained, torch_cache)
    if model_type == MODEL_TYPE_V2:
        from .multiscale_model import ConvNeXtMultiscaleDepth
        return ConvNeXtMultiscaleDepth(ingredient_count, config, pretrained, torch_cache)
    if model_type == MODEL_TYPE_V3:
        from .conditional_model import ConvNeXtConditionalMass
        return ConvNeXtConditionalMass(ingredient_count, config, pretrained, torch_cache)
    if model_type == MODEL_TYPE_V6:
        from .robust_model import ConvNeXtRobustMass
        return ConvNeXtRobustMass(ingredient_count, config, pretrained, torch_cache)
    if model_type in SPECIALIZED_MODEL_TYPES:
        if model_type == MODEL_TYPE_V5 and (config.get("split_task_mlps") is not False or config.get("auxiliary_total_mass") is not True):
            raise ValueError("V5 requires the shared MLP and active auxiliary total-mass head")
        from .specialized_model import ConvNeXtSpecializedMass
        return ConvNeXtSpecializedMass(ingredient_count, config, pretrained, torch_cache)
    raise ValueError(f"Unsupported model type: {model_type}")
