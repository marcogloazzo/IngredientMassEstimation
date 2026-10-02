"""Image-based training/evaluation utilities, independent of old pooled caches."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import read_json, read_jsonl, vectorize_record
from .features import PreparedImageDataset
from .model import compute_loss
from .spatial_model import MODEL_TYPE, image_transforms


def load_prepared(processed_dir: Path, dataset_root: Path, config: dict):
    vocabulary = read_json(processed_dir / "vocabulary.json")
    records = read_jsonl(processed_dir / "records.jsonl")
    if config.get("model_type") != MODEL_TYPE or config.get("encoder") != "convnext_small":
        raise ValueError("Use the spatial ConvNeXt Small configuration")
    if len(vocabulary["entries"]) != int(config["top_k"]):
        raise ValueError("Vocabulary size differs from config top_k; prepare labels with this config")
    missing = [record["image"] for record in records if not (dataset_root / record["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prepared RGB images missing; first: {missing[0]}")
    return records, vocabulary


def make_loader(records, vocabulary, dataset_root, split, batch_size, workers, device,
                training=False, seed=42):
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"No records for split {split}")
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
    targets = {key: [] for key in ("presence", "shares", "masses", "total_mass", "nutrients", "dish_ids")}
    totals, sample_count = {}, 0
    progress = tqdm(loader, desc=description, unit="batch")
    for index, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        target = {key: batch[key].to(device, non_blocking=True)
                  for key in ("presence", "shares", "masses", "total_mass", "nutrients")}
        with torch.set_grad_enabled(training):
            with autocast_context(device, config.get("amp", True)):
                outputs = model(images)
            # All loss arithmetic outside autocast, using FP32 model predictions.
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
    )
