"""V6: ASL, class scales, spatial depth gates and CLIP-initialized queries."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .conditional_model import training_statistics
from .data import vectorize_record
from .model import asymmetric_loss
from .multiscale_model import ConvNeXtMultiscaleDepth
from .specialized_model import SpecializedDepthHeads
from .spatial_model import position_encoding


def class_mass_statistics(records, vocabulary, config):
    stats = training_statistics(records, vocabulary)
    train = [vectorize_record(r, vocabulary)["masses"] for r in records if r["split"] == "train"]
    masses = np.asarray(train, dtype=np.float64)
    if not np.isfinite(masses).all() or (masses < 0).any():
        raise ValueError("Invalid training ingredient masses")
    settings = config["conditional_scale"]
    quantile, minimum, maximum = (float(settings[k]) for k in ("quantile", "min_grams", "max_grams"))
    min_support = int(settings["min_support"])
    if not 0 < quantile <= 1 or not 0 < minimum <= maximum or min_support < 1:
        raise ValueError("Invalid per-class mass-scale settings")
    positive = masses[masses > 0]
    if not positive.size:
        raise ValueError("No positive training ingredient masses")
    fallback = float(np.clip(np.quantile(positive, quantile), minimum, maximum))
    scales, supports, details = [], [], []
    for index, entry in enumerate(vocabulary["entries"]):
        values = masses[:, index][masses[:, index] > 0]
        support = int(values.size)
        estimate = float(np.quantile(values, quantile)) if support >= min_support else fallback
        scale = float(np.clip(estimate, minimum, maximum))
        scales.append(scale); supports.append(support)
        details.append({"key": entry["key"], "name": entry["name"], "positive_train_records": support,
                        "scale_grams": scale, "conditional_mean_grams": float(stats["conditional_mean_grams"][index]),
                        "source": "class_positive_quantile" if support >= min_support else "pooled_train_fallback"})
    stats.update(class_gram_scales=np.asarray(scales, dtype=np.float32),
                 class_support=np.asarray(supports, dtype=np.int64),
                 scale_details=details, fallback_scale_grams=fallback)
    return stats


class RobustDepthHeads(SpecializedDepthHeads):
    def __init__(self, ingredient_count, config):
        super().__init__(ingredient_count, config)
        del self.positive_weights
        del self.depth_gate_logits
        self.register_buffer("class_gram_scales", torch.full((ingredient_count,), 300.))
        self.register_buffer("class_support", torch.zeros(ingredient_count, dtype=torch.long))
        self.register_buffer("clip_initialized", torch.tensor(False))
        dimension = int(config["attention_dim"])
        clip_dimension = int(config["clip_queries"].get("embedding_dim", 512))
        if clip_dimension < dimension:
            raise ValueError("CLIP embedding dimension must be >= attention dimension")
        self.queries = nn.Embedding(ingredient_count, clip_dimension)
        self.query_projection = nn.Linear(clip_dimension, dimension, bias=False)
        generator = torch.Generator().manual_seed(int(config["clip_queries"]["projection_seed"]))
        orthogonal, _ = torch.linalg.qr(torch.randn(clip_dimension, dimension, generator=generator), mode="reduced")
        with torch.no_grad(): self.query_projection.weight.copy_(orthogonal.T)
        self.query_dimension = dimension
        settings = config["depth_spatial_gate"]
        hidden, initial = int(settings["hidden_dim"]), float(settings["initial"])
        if hidden < 1 or not 0 < initial < 1:
            raise ValueError("Invalid spatial depth-gate settings")
        self.spatial_gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2*dimension, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
            for _ in range(2)])
        for gate in self.spatial_gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, math.log(initial/(1-initial)))
        self.mass_filter_enabled = bool(config["mass_filter"]["enabled"])
        self.mass_threshold = float(config["mass_filter"]["threshold"])
        if not 0 <= self.mass_threshold <= 1:
            raise ValueError("Mass threshold must be inside [0,1]")
        if config["mass_filter"].get("apply_during_training", False):
            raise ValueError("V6 hard mass filtering is evaluation/inference-only to preserve training gradients")

    @torch.no_grad()
    def initialize_from_train(self, statistics):
        device = self.class_gram_scales.device
        scales = torch.as_tensor(statistics["class_gram_scales"], device=device)
        self.class_gram_scales.copy_(scales)
        self.class_support.copy_(torch.as_tensor(statistics["class_support"], device=device))
        self.conditional_weights.copy_(torch.as_tensor(statistics["conditional_weights"], device=device))
        prior = torch.as_tensor(statistics["prevalence"], device=device).float().clamp(.001, .999)
        # ASL does NOT use weighted-BCE logit offsets, neither in bias nor gate.
        self.presence_head.bias.copy_(torch.logit(prior))
        mean = torch.as_tensor(statistics["conditional_mean_grams"], device=device).float().clamp(min=.1)
        x = mean/scales
        self.conditional_head.bias.copy_(x + torch.log(-torch.expm1(-x)))

    @torch.no_grad()
    def initialize_clip_queries(self, embeddings, initialized_mask):
        values = torch.as_tensor(embeddings, device=self.queries.weight.device, dtype=torch.float32)
        mask = torch.as_tensor(initialized_mask, device=values.device, dtype=torch.bool)
        if values.shape != self.queries.weight.shape or mask.shape != (values.shape[0],):
            raise ValueError("CLIP cache dimensions differ from model queries")
        if not torch.isfinite(values).all() or not mask.any():
            raise ValueError("Invalid CLIP query values")
        self.queries.weight[mask] = F.normalize(values[mask], dim=-1) * math.sqrt(values.shape[1])
        # 'other' retains its random, trainable embedding rather than a fake semantic label.
        self.clip_initialized.fill_(True)

    def forward_memory(self, memory, positions, return_attention=False):
        query = F.normalize(self.query_projection(self.queries.weight).float(), dim=-1) * math.sqrt(self.query_dimension)
        queries = query.to(memory.dtype)[None].expand(memory.shape[0], -1, -1)
        labels = torch.zeros_like(queries)
        attention = None
        for layer in self.decoder:
            labels, attention = layer(labels, memory, queries, positions, return_attention)
        projected = self.output_projection(labels)
        hidden = self.shared(projected)
        logits = self.presence_head(hidden).float()
        probability = logits.sigmoid()
        conditional = F.softplus(self.conditional_head(hidden).float()) * self.class_gram_scales[None]
        raw = probability * conditional
        retained = probability >= self.mass_threshold if self.mass_filter_enabled else torch.ones_like(probability, dtype=torch.bool)
        masses = raw if self.training else raw * retained
        total = masses.sum(-1)
        outputs = {"presence_logits": logits, "presence_probabilities": probability,
                   "raw_presence_probabilities": probability, "conditional_grams": conditional,
                   "class_gram_scales": self.class_gram_scales, "conditional_class_weights": self.conditional_weights,
                   "unfiltered_ingredient_masses": raw, "mass_retained": retained,
                   "ingredient_masses": masses, "total_mass": total,
                   "shares": masses/total.clamp(min=1e-8)[:, None]}
        if self.has_auxiliary_mass:
            auxiliary = F.softplus(self.auxiliary_mass_head(memory.mean(1)).float()).squeeze(-1)
            outputs["auxiliary_total_mass"] = auxiliary * self.auxiliary_gram_scale
        if return_attention:
            outputs.update(label_features=labels, projected_label_features=projected, shared_features=hidden,
                           projected_query_features=query, clip_initialized=self.clip_initialized)
        return outputs, attention

    def forward(self, rgb_stage3, rgb_stage4, depth, return_attention=False):
        d3, d4 = self.depth_encoder(depth)
        rgb = [self.rgb_stage3_projection(rgb_stage3), self.input_projection(rgb_stage4)]
        depths, memory, position, fused_features, gates = [d3, d4], [], [], [], []
        for scale in range(2):
            d = self.depth_projections[scale](depths[scale])
            if d.shape[-2:] != rgb[scale].shape[-2:]:
                raise ValueError("V6 RGB and depth must use matching 320 geometry")
            r, d = self.rgb_norms[scale](rgb[scale]), self.depth_norms[scale](d)
            gate = self.spatial_gates[scale](torch.cat([r,d], 1)).sigmoid()
            fused = r + gate*d
            tokens = self.memory_norms[scale](fused.flatten(2).transpose(1,2))
            h,w = fused.shape[-2:]
            memory.append(tokens); gates.append(gate); fused_features.append(fused)
            position.append(position_encoding(h,w,tokens.shape[-1],tokens.device,tokens.dtype) +
                            self.scale_embeddings.weight[scale].to(tokens.dtype)[None,None])
        outputs, attention = self.forward_memory(torch.cat(memory,1),torch.cat(position,1),return_attention)
        outputs["depth_gate_means"] = torch.stack([g.mean((1,2,3)) for g in gates],dim=1)
        if return_attention:
            h3,w3 = rgb[0].shape[-2:]; h4,w4 = rgb[1].shape[-2:]
            a3 = attention[:,:,:h3*w3].reshape(depth.shape[0],-1,h3,w3)
            a4 = attention[:,:,h3*w3:].reshape(depth.shape[0],-1,h4,w4)
            outputs.update(attention=a3+F.interpolate(a4,(h3,w3),mode="nearest")*(h4*w4)/(h3*w3),
                attention_stage3=a3,attention_stage4=a4,attention_tokens=attention,
                rgb_features_stage3=rgb_stage3,rgb_features_stage4=rgb_stage4,
                depth_input=depth,depth_features_stage3=d3,depth_features_stage4=d4,
                fused_stage3=fused_features[0],fused_stage4=fused_features[1],spatial_memory=torch.cat(memory,1),
                depth_spatial_gate_stage3=gates[0],depth_spatial_gate_stage4=gates[1])
        return outputs


class ConvNeXtRobustMass(ConvNeXtMultiscaleDepth):
    def __init__(self, ingredient_count, config, pretrained=True, torch_cache=None):
        if config.get("presence_loss") != "asl" or config.get("presence_gate") != "sigmoid":
            raise ValueError("V6 requires ASL and raw sigmoid gates, without BCE prior correction")
        if config.get("split_task_mlps") is not False or config.get("auxiliary_total_mass") is not True:
            raise ValueError("V6 preserves the shared MLP and V5 auxiliary mass head")
        if int(config.get("image_size",0)) != 320:
            raise ValueError("V6 reuses V3's 320 RGB/depth geometry")
        super().__init__(ingredient_count,config,pretrained,torch_cache,heads_class=RobustDepthHeads)
        self._v6_statistics_config = {"conditional_scale":dict(config["conditional_scale"])}

    def initialize_from_train(self, records, vocabulary):
        stats = class_mass_statistics(records,vocabulary,self.config_for_statistics)
        self.heads.initialize_from_train(stats)
        mean = sum(float(r["total_mass"]) for r in records if r["split"] == "train")/sum(r["split"] == "train" for r in records)
        self.heads.auxiliary_gram_scale.fill_(mean)
        return stats

    @property
    def config_for_statistics(self):
        return self._v6_statistics_config


def robust_mass_loss(outputs, presence, masses, total_mass, nutrients, matrix, positive_weights, config):
    weights = config["loss_weights"]
    gram_scale, mass_scale = float(config["ingredient_mass_scale"]), float(config["mass_scale"])
    if gram_scale <= 0 or mass_scale <= 0:
        raise ValueError("Loss normalizers must be positive")
    asl = asymmetric_loss(outputs["presence_logits"],presence,**config["asl"])
    scales = outputs["class_gram_scales"].float()[None]
    errors = F.smooth_l1_loss(outputs["conditional_grams"]/scales,masses/scales,
                             reduction="none",beta=float(config["conditional_huber_beta"]))
    positive_elements = presence * outputs["conditional_class_weights"][None]
    conditional = (errors*positive_elements).sum()/positive_elements.sum().clamp(min=1e-8)
    # Dense objectives preserve gradients even where the inference cutoff removes a mass.
    predicted = outputs["unfiltered_ingredient_masses"].float()
    summed = predicted.sum(-1)
    false_mass = (predicted*(1-presence)).sum(-1).mean()/gram_scale
    ingredient = (predicted-masses).abs().sum(-1).mean()/gram_scale
    mass = (summed-total_mass).abs().mean()/mass_scale
    nutrition = ((predicted@matrix-nutrients).abs()/predicted.new_tensor([500.,30.,80.,40.])).mean()
    auxiliary = outputs["auxiliary_total_mass"].float()
    aux_mass = (auxiliary-total_mass).abs().mean()/mass_scale
    consistency = (auxiliary-summed).abs().mean()/mass_scale
    parts = dict(presence=asl,conditional=conditional,false_mass=false_mass,ingredient_mass=ingredient,
                 mass=mass,nutrition=nutrition,aux_mass=aux_mass,mass_consistency=consistency)
    total = sum(float(weights[key])*value for key,value in parts.items())
    return total,{"total":float(total.detach()),**{key:float(value.detach()) for key,value in parts.items()}}
