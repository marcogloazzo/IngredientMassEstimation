"""Image-based training/evaluation utilities, independent of old pooled caches."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import read_json, read_jsonl, vectorize_record
from .features import PreparedImageDataset
from .model import compute_loss
from .spatial_model import SUPPORTED_MODEL_TYPES, DEPTH_MODEL_TYPES, CONDITIONAL_MODEL_TYPES, SPECIALIZED_MODEL_TYPES, image_transforms


def prepared_fingerprint(processed_dir):
    from .depth import file_sha256
    return {name: file_sha256(Path(processed_dir) / name) for name in ("records.jsonl", "vocabulary.json")}


def load_prepared(processed_dir: Path, dataset_root: Path, config: dict):
    vocabulary = read_json(processed_dir / "vocabulary.json")
    records = read_jsonl(processed_dir / "records.jsonl")
    if config.get("model_type") not in SUPPORTED_MODEL_TYPES or config.get("encoder") != "convnext_small":
        raise ValueError("Use the spatial ConvNeXt Small configuration")
    if len(vocabulary["entries"]) != int(config["top_k"]):
        raise ValueError("Vocabulary size differs from config top_k; prepare labels with this config")
    if config.get("model_type") in CONDITIONAL_MODEL_TYPES or config.get("split_policy") == "grouped_provisional":
        summary = read_json(Path(processed_dir) / "summary.json")
        if summary.get("mode") != "provisional" or config.get("accept_provisional_split") is not True:
            raise ValueError("This configuration requires the explicitly accepted grouped provisional dataset")
        if summary.get("official_test_preserved") is not True:
            raise ValueError("Grouped dataset must preserve official test")
        by_group = {}
        if len({r["dish_id"] for r in records}) != len(records):
            raise ValueError("Duplicate prepared dish IDs")
        for record in records:
            group = record.get("scan_group_id")
            if not group:
                raise ValueError("V3 records require scan_group_id")
            if group in by_group and by_group[group] != record["split"]:
                raise ValueError("Group leakage in V3 prepared records")
            by_group[group] = record["split"]
    if config.get("prepared_fingerprint") and prepared_fingerprint(processed_dir) != config["prepared_fingerprint"]:
        raise ValueError("Prepared split/vocabulary changed since this checkpoint")
    missing = [record["image"] for record in records if not (dataset_root / record["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prepared RGB images missing; first: {missing[0]}")
    return records, vocabulary


def make_loader(records, vocabulary, dataset_root, split, batch_size, workers, device,
                training=False, seed=42, config=None, depth_cache=None):
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"No records for split {split}")
    if config and config.get("model_type") in DEPTH_MODEL_TYPES:
        from .depth import PairedDepthDataset
        if depth_cache is None:
            raise ValueError("V2 loader requires --depth-cache")
        dataset = PairedDepthDataset(selected, dataset_root, vocabulary, depth_cache, training, config)
    else:
        dataset = PreparedImageDataset(selected, dataset_root, vocabulary, image_transforms(training))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training, num_workers=workers,
        pin_memory=device.type == "cuda", generator=generator,
        persistent_workers=False,
    )


def positive_class_weights(records, vocabulary):
    targets = np.stack([vectorize_record(record, vocabulary)["presence"]
                        for record in records if record["split"] == "train"])
    positives = targets.sum(axis=0)
    return torch.from_numpy(np.clip((len(targets) - positives) / np.maximum(positives, 1), 1, 20).astype(np.float32))


def autocast_context(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def run_epoch(model, loader, matrix, positive_weights, config, device,
              optimizer=None, scaler=None, description="Evaluating", collect=True):
    training = optimizer is not None
    model.train(training)
    accumulation = int(config.get("accumulation_steps", 1)) if training else 1
    if accumulation < 1:
        raise ValueError("accumulation_steps must be positive")
    if training:
        optimizer.zero_grad(set_to_none=True)
    collected = {key: [] for key in ("presence", "shares", "mass")}
    if config.get("model_type") in CONDITIONAL_MODEL_TYPES:
        collected["conditional_grams"] = []
        collected["ingredient_masses"] = []
    if config.get("model_type") in SPECIALIZED_MODEL_TYPES and config.get("auxiliary_total_mass", True):
        collected["auxiliary_total_mass"] = []
    targets = {key: [] for key in ("presence", "shares", "masses", "total_mass", "nutrients", "dish_ids")}
    totals, sample_count = {}, 0
    progress = tqdm(loader, desc=description, unit="batch", file=sys.stdout, mininterval=1)
    for index, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        target = {key: batch[key].to(device, non_blocking=True)
                  for key in ("presence", "shares", "masses", "total_mass", "nutrients")}
        with torch.set_grad_enabled(training):
            with autocast_context(device, config.get("amp", True)):
                if "depth" in batch:
                    outputs = model(images, depth=batch["depth"].to(device, non_blocking=True))
                else:
                    outputs = model(images)
            # All loss arithmetic outside autocast, using FP32 model predictions.
            if config.get("model_type") in CONDITIONAL_MODEL_TYPES:
                if config.get("model_type") in SPECIALIZED_MODEL_TYPES:
                    from .specialized_model import specialized_mass_loss as mass_loss
                else:
                    from .conditional_model import conditional_mass_loss as mass_loss
                loss, parts = mass_loss(outputs, target["presence"], target["masses"],
                                        target["total_mass"], target["nutrients"], matrix, positive_weights, config)
            else:
                loss, parts = compute_loss(
                    outputs, target["presence"], target["shares"], target["total_mass"],
                    target["nutrients"], matrix, positive_weights, config["loss_weights"],
                    presence_loss_name=config.get("presence_loss", "asl"),
                    asl_config=config.get("asl"), masses_target=target["masses"],
                    ingredient_mass_scale=float(config.get("ingredient_mass_scale", 300)),
                    mass_scale=float(config.get("mass_scale", 300)),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {description}, batch {index}")
            if training:
                # Exact sample-weighted normalization, including a short final accumulation group.
                group_start = (index // accumulation) * accumulation * loader.batch_size
                group_samples = min(accumulation * loader.batch_size, len(loader.dataset) - group_start)
                scaler.scale(loss * len(images) / group_samples).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        sample_count += len(images)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * len(images)
        if collect:
            for key, output_name in (("presence", "presence_probabilities"), ("shares", "shares"), ("mass", "total_mass")):
                collected[key].append(outputs[output_name].detach().float().cpu().numpy())
            for key in ("conditional_grams", "ingredient_masses", "auxiliary_total_mass"):
                if key in collected:
                    collected[key].append(outputs[key].detach().float().cpu().numpy())
            for key in targets:
                targets[key].append(np.asarray(batch["dish_id"], dtype=str) if key == "dish_ids" else batch[key].numpy())
        progress.set_postfix(loss=f"{totals['total'] / sample_count:.4f}", samples=f"{sample_count}/{len(loader.dataset)}")
    losses = {key: value / sample_count for key, value in totals.items()}
    if not collect:
        return losses, None, None
    return (losses, {key: np.concatenate(value) for key, value in collected.items()},
            {key: np.concatenate(value) for key, value in targets.items()})


def save_predictions(path, predictions, targets, vocabulary):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, probabilities=predictions["presence"], predicted_shares=predictions["shares"],
        predicted_total_mass=predictions["mass"], **targets,
        vocabulary_keys=np.asarray([entry["key"] for entry in vocabulary["entries"]]),
        **{key: predictions[key] for key in ("conditional_grams", "ingredient_masses", "auxiliary_total_mass") if key in predictions},
    )


def auxiliary_mass_metrics(predictions, targets):
    if "auxiliary_total_mass" not in predictions:
        return {}
    auxiliary = predictions["auxiliary_total_mass"]
    return {"auxiliary_total_mass": {
        "mae_grams": float(np.abs(auxiliary - targets["total_mass"]).mean()),
        "bias_grams": float((auxiliary - targets["total_mass"]).mean()),
        "consistency_gap_mae_grams": float(np.abs(auxiliary - predictions["mass"]).mean()),
        "used_for_deployed_masses": False,
    }}
