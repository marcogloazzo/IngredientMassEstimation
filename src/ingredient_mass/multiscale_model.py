"""V2: RGB stage 3/4 + a trainable relative-depth CNN + two Q2L layers.

Depth Anything V2 is a frozen RGB preprocessing component, outside this module.
The CLI accepts RGB only; training reuses aligned, precomputed depth maps.
"""
import os

import torch
from torch import nn
from torch.nn import functional as F

from .spatial_model import SpatialIngredientHeads, position_encoding


class SmallDepthEncoder(nn.Module):
    """One-channel 224x224 depth to 64x14x14 and 128x7x7 features."""
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                                 nn.GroupNorm(8, cout), nn.GELU())

        self.stage3 = nn.Sequential(block(1, 16), block(16, 32),
                                    block(32, 64), block(64, 64))
        self.stage4 = block(64, 128)

    def forward(self, depth):
        high = self.stage3(depth)
        return high, self.stage4(high)


class MultiscaleDepthHeads(SpatialIngredientHeads):
    def __init__(self, ingredient_count, config):
        super().__init__(ingredient_count, config)
        dim = int(config.get("attention_dim", 256))
        self.rgb_stage3_projection = nn.Conv2d(384, dim, 1)
        # input_projection inherited from v1 maps RGB stage 4 from 768 channels.
        self.depth_encoder = SmallDepthEncoder()
        self.depth_projections = nn.ModuleList([nn.Conv2d(64, dim, 1), nn.Conv2d(128, dim, 1)])
        self.fusions = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU()) for _ in range(2)
        ])
        self.scale_embeddings = nn.Embedding(2, dim)

    def forward(self, rgb_stage3, rgb_stage4, depth, return_attention=False):
        batch = rgb_stage3.shape[0]
        d3, d4 = self.depth_encoder(depth)
        rgb = [self.rgb_stage3_projection(rgb_stage3), self.input_projection(rgb_stage4)]
        depth_features = [d3, d4]
        memories, positions, fused_features = [], [], []
        for scale in range(2):
            d = self.depth_projections[scale](depth_features[scale])
            if d.shape[-2:] != rgb[scale].shape[-2:]:
                raise ValueError("RGB/depth scales are not aligned; use the same 224x224 crop")
            fused = self.fusions[scale](torch.cat([rgb[scale], d], dim=1))
            h, w = fused.shape[-2:]
            memories.append(fused.flatten(2).transpose(1, 2))
            position = position_encoding(h, w, fused.shape[1], fused.device, fused.dtype)
            positions.append(position + self.scale_embeddings.weight[scale].to(fused.dtype)[None, None])
            fused_features.append(fused)
        memory = torch.cat(memories, dim=1)
        position = torch.cat(positions, dim=1)
        outputs, weights = self.forward_memory(memory, position, return_attention)
        if return_attention:
            h3, w3 = rgb_stage3.shape[-2:]
            h4, w4 = rgb_stage4.shape[-2:]
            attention3 = weights[:, :, :h3*w3].reshape(batch, -1, h3, w3)
            attention4 = weights[:, :, h3*w3:].reshape(batch, -1, h4, w4)
            # Project low-resolution token mass onto the high-resolution grid.
            # Keep the two native maps and the 245-token attention for inspection.
            combined = attention3 + F.interpolate(attention4, (h3, w3), mode="nearest") * (h4*w4)/(h3*w3)
            outputs.update({
                "attention": combined, "attention_stage3": attention3,
                "attention_stage4": attention4, "attention_tokens": weights,
                "rgb_features_stage3": rgb_stage3, "rgb_features_stage4": rgb_stage4,
                "depth_input": depth, "depth_features_stage3": d3, "depth_features_stage4": d4,
                "fused_stage3": fused_features[0], "fused_stage4": fused_features[1],
                "spatial_memory": memory,
            })
        return outputs


class ConvNeXtMultiscaleDepth(nn.Module):
    def __init__(self, ingredient_count, config, pretrained=True, torch_cache=None):
        super().__init__()
        from torchvision.models import ConvNeXt_Small_Weights, convnext_small
        if config.get("encoder_training") != "stages_2_4":
            raise ValueError("V2 freezes stem/stage 1 and adjusts stages 2-4")
        if int(config.get("attention_layers", 2)) != 2:
            raise ValueError("V2 uses two cross-attention layers")
        if torch_cache is not None:
            torch_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(torch_cache.resolve())
        backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features
        self.final_norm = backbone.classifier[0]
        self.heads = MultiscaleDepthHeads(ingredient_count, config)
        self.encoder_trainable = False
        self.set_encoder_trainable(False)

    def encoder_parameters(self):
        for module in (self.features[2:], self.final_norm):
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
        for module in self.features[:2]:
            module.eval()
        if not self.encoder_trainable:
            for module in self.features[2:]:
                module.eval()
            self.final_norm.eval()
        return self

    def forward(self, images, depth=None, return_attention=False):
        if depth is None:
            raise ValueError("V2 requires estimated depth generated internally from RGB, not sensor depth")
        if depth.shape != (images.shape[0], 1, *images.shape[-2:]):
            raise ValueError("Expected aligned depth of shape B x 1 x H x W")
        with torch.no_grad():
            frozen = self.features[:2](images)
        high = self.features[2:6](frozen)
        low = self.final_norm(self.features[6:](high))
        return self.heads(high, low, depth, return_attention)
