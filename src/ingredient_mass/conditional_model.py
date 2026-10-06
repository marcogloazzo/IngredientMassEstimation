"""V3: RGB-preserving residual depth fusion and positive-only gram supervision."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import vectorize_record
from .multiscale_model import ConvNeXtMultiscaleDepth, MultiscaleDepthHeads
from .spatial_model import GroupWiseLinear, position_encoding


def training_statistics(records, vocabulary):
    train = [vectorize_record(r, vocabulary) for r in records if r["split"] == "train"]
    if not train:
        raise ValueError("V3 needs training records for initialization/class balance")
    masses = np.stack([r["masses"] for r in train])
    count = (masses > 0).sum(0)
    means = masses.sum(0) / np.maximum(count, 1)
    weights = np.clip((len(train) - count) / np.maximum(count, 1), 1, 20)
    balance = np.sqrt(len(train) / np.maximum(count, 1)).astype(np.float32)
    supported = count > 0
    balance /= balance[supported].mean() if supported.any() else 1
    balance = np.clip(balance, .25, 3)
    balance[~supported] = 0
    return {"prevalence": count / len(train), "conditional_mean_grams": means,
            "positive_weights": weights, "conditional_weights": balance}


class ConditionalDepthHeads(MultiscaleDepthHeads):
    def __init__(self, ingredient_count, config):
        super().__init__(ingredient_count, config)
        # V2's ratio/global-mass heads are not part of the V3 state dict.
        del self.amount_head
        del self.mass_head
        del self.fusions
        self.conditional_head = GroupWiseLinear(ingredient_count, 192)
        self.gram_scale = float(config.get("conditional_gram_scale", 300))
        if self.gram_scale <= 0:
            raise ValueError("conditional_gram_scale must be positive")
        dim = int(config["attention_dim"])
        self.rgb_norms = nn.ModuleList([nn.GroupNorm(1, dim) for _ in range(2)])
        self.depth_norms = nn.ModuleList([nn.GroupNorm(1, dim) for _ in range(2)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        initial = float(config.get("depth_gate_initial", .1))
        if not 0 < initial < 1:
            raise ValueError("depth_gate_initial must be in (0,1)")
        self.depth_gate_logits = nn.Parameter(torch.full((2,), math.log(initial / (1 - initial))))
        self.register_buffer("positive_weights", torch.ones(ingredient_count))
        self.register_buffer("conditional_weights", torch.ones(ingredient_count))
        nn.init.zeros_(self.conditional_head.weight)
        nn.init.constant_(self.conditional_head.bias, -2.)

    @torch.no_grad()
    def initialize_from_train(self, statistics):
        device = self.positive_weights.device
        weight = torch.as_tensor(statistics["positive_weights"], device=device).float()
        prior = torch.as_tensor(statistics["prevalence"], device=device).float().clamp(.001, .999)
        mean = torch.as_tensor(statistics["conditional_mean_grams"], device=device).float().clamp(min=.1)
        self.positive_weights.copy_(weight)
        self.conditional_weights.copy_(torch.as_tensor(statistics["conditional_weights"], device=device).float())
        # Correct weighted-BCE logits back toward the unweighted prior for the
        # quantity gate. This is a prior correction, NOT guaranteed calibration.
        self.presence_head.bias.copy_(torch.logit(prior) + weight.log())
        x = mean / self.gram_scale
        self.conditional_head.bias.copy_(x + torch.log(-torch.expm1(-x)))

    def forward_memory(self, memory, positions, return_attention=False):
        queries = self.queries.weight[None].expand(memory.shape[0], -1, -1)
        labels = torch.zeros_like(queries)
        attention = None
        for layer in self.decoder:
            labels, attention = layer(labels, memory, queries, positions, return_attention)
        projected = self.output_projection(labels)
        hidden = self.shared(projected)
        logits = self.presence_head(hidden).float()
        probability = torch.sigmoid(logits - self.positive_weights.clamp(min=1).log())
        conditional = F.softplus(self.conditional_head(hidden).float()) * self.gram_scale
        masses = probability * conditional + 1e-8
        total = masses.sum(-1)
        outputs = {"presence_logits": logits, "presence_probabilities": probability,
                   "raw_presence_probabilities": logits.sigmoid(), "conditional_grams": conditional,
                   "conditional_class_weights": self.conditional_weights,
                   "ingredient_masses": masses, "total_mass": total, "shares": masses / total[:, None]}
        if return_attention:
            outputs.update(label_features=labels, projected_label_features=projected, shared_features=hidden)
        return outputs, attention

    def forward(self, rgb_stage3, rgb_stage4, depth, return_attention=False):
        d3, d4 = self.depth_encoder(depth)
        rgb = [self.rgb_stage3_projection(rgb_stage3), self.input_projection(rgb_stage4)]
        depths = [d3, d4]
        memory, position, fused_features = [], [], []
        for scale in range(2):
            d = self.depth_projections[scale](depths[scale])
            if d.shape[-2:] != rgb[scale].shape[-2:]:
                raise ValueError("V3 RGB/depth scales must be aligned")
            fused = self.rgb_norms[scale](rgb[scale]) + self.depth_gate_logits[scale].sigmoid() * self.depth_norms[scale](d)
            tokens = self.memory_norms[scale](fused.flatten(2).transpose(1, 2))
            h, w = fused.shape[-2:]
            memory.append(tokens)
            position.append(position_encoding(h, w, tokens.shape[-1], tokens.device, tokens.dtype) +
                            self.scale_embeddings.weight[scale].to(tokens.dtype)[None, None])
            fused_features.append(fused)
        outputs, attention = self.forward_memory(torch.cat(memory, 1), torch.cat(position, 1), return_attention)
        if return_attention:
            h3, w3 = rgb[0].shape[-2:]; h4, w4 = rgb[1].shape[-2:]
            a3 = attention[:, :, :h3*w3].reshape(depth.shape[0], -1, h3, w3)
            a4 = attention[:, :, h3*w3:].reshape(depth.shape[0], -1, h4, w4)
            outputs.update(attention=a3 + F.interpolate(a4, (h3, w3), mode="nearest") * (h4*w4)/(h3*w3),
                           attention_stage3=a3, attention_stage4=a4, attention_tokens=attention,
                           rgb_features_stage3=rgb_stage3, rgb_features_stage4=rgb_stage4,
                           depth_input=depth, depth_features_stage3=d3, depth_features_stage4=d4,
                           fused_stage3=fused_features[0], fused_stage4=fused_features[1],
                           spatial_memory=torch.cat(memory, 1), depth_fusion_gates=self.depth_gate_logits.sigmoid())
        return outputs


class ConvNeXtConditionalMass(ConvNeXtMultiscaleDepth):
    def __init__(self, ingredient_count, config, pretrained=True, torch_cache=None):
        if config.get("presence_loss") != "bce":
            raise ValueError("V3 requires weighted BCE")
        if config.get("presence_gate") != "pos_weight_corrected_sigmoid" or int(config.get("image_size", 0)) != 320:
            raise ValueError("V3 requires prior-corrected presence gates and 320x320 geometry")
        super().__init__(ingredient_count, config, pretrained, torch_cache, heads_class=ConditionalDepthHeads)

    def initialize_from_train(self, records, vocabulary):
        self.heads.initialize_from_train(training_statistics(records, vocabulary))


def conditional_mass_loss(outputs, presence, masses, total_mass, nutrients, matrix, positive_weights, config):
    weights = config["loss_weights"]
    gram_scale = float(config["ingredient_mass_scale"])
    mass_scale = float(config["mass_scale"])
    if gram_scale <= 0 or mass_scale <= 0:
        raise ValueError("Mass scales must be positive")
    logits = outputs["presence_logits"].float()
    bce = F.binary_cross_entropy_with_logits(logits, presence, pos_weight=positive_weights)
    predicted = outputs["ingredient_masses"].float()
    # Direct conditional supervision survives even when the presence gate is low.
    errors = F.smooth_l1_loss(outputs["conditional_grams"] / gram_scale, masses / gram_scale,
                            reduction="none", beta=float(config.get("conditional_huber_beta", .1)))
    positive_element_weights = presence * outputs["conditional_class_weights"][None]
    conditional = (errors * positive_element_weights).sum() / positive_element_weights.sum().clamp(min=1e-8)
    false_mass = (predicted * (1 - presence)).sum(-1).mean() / gram_scale
    ingredient = (predicted - masses).abs().sum(-1).mean() / gram_scale
    mass = (outputs["total_mass"] - total_mass).abs().mean() / mass_scale
    scales = predicted.new_tensor([500., 30., 80., 40.])
    nutrition = ((predicted @ matrix - nutrients).abs() / scales).mean()
    parts = {"presence": bce, "conditional": conditional, "false_mass": false_mass,
             "ingredient_mass": ingredient, "mass": mass, "nutrition": nutrition}
    total = sum(float(weights[key]) * value for key, value in parts.items())
    return total, {"total": float(total.detach()), **{key: float(value.detach()) for key, value in parts.items()}}
