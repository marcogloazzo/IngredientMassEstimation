from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import read_jsonl, vectorize_record


ENCODER_NAME = "efficientnet_b0"
FEATURE_DIMENSION = 1280
ENCODER_DIMENSIONS = {"efficientnet_b0": 1280, "convnext_small": 768}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def create_frozen_encoder(
    torch_cache: Path | None = None, encoder_name: str = ENCODER_NAME,
):
    if encoder_name not in ENCODER_DIMENSIONS:
        raise ValueError(f"Unsupported encoder: {encoder_name!r}")
    if torch_cache is not None:
        torch_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_cache.resolve())

    from torchvision.models import (
        ConvNeXt_Small_Weights, EfficientNet_B0_Weights,
        convnext_small, efficientnet_b0,
    )

    if encoder_name == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT
        encoder = convnext_small(weights=weights)
        # Keep pretrained LayerNorm and flatten; replace only the ImageNet logits.
        encoder.classifier[-1] = torch.nn.Identity()
    else:
        weights = EfficientNet_B0_Weights.DEFAULT
        encoder = efficientnet_b0(weights=weights)
        encoder.classifier = torch.nn.Identity()
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, weights.transforms()


class PreparedImageDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        dataset_root: Path,
        vocabulary: dict,
        transform,
    ):
        self.records = records
        self.dataset_root = dataset_root
        self.vocabulary = vocabulary
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_path = self.dataset_root / Path(record["image"])
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        target = vectorize_record(record, self.vocabulary)
        return {
            "image": self.transform(image),
            "dish_id": record["dish_id"],
            "presence": torch.from_numpy(target["presence"]),
            "shares": torch.from_numpy(target["shares"]),
            "masses": torch.from_numpy(target["masses"]),
            "total_mass": torch.tensor(target["total_mass"], dtype=torch.float32),
            "nutrients": torch.from_numpy(target["nutrients"]),
        }


def _empty_cache(feature_dimension: int, ingredient_count: int) -> dict[str, np.ndarray]:
    return {
        "features": np.empty((0, feature_dimension), dtype=np.float32),
        "presence": np.empty((0, ingredient_count), dtype=np.float32),
        "shares": np.empty((0, ingredient_count), dtype=np.float32),
        "masses": np.empty((0, ingredient_count), dtype=np.float32),
        "total_mass": np.empty((0,), dtype=np.float32),
        "nutrients": np.empty((0, 4), dtype=np.float32),
        "dish_ids": np.empty((0,), dtype="U1"),
    }


def extract_split_features(
    records_path: Path,
    dataset_root: Path,
    vocabulary: dict,
    split: str,
    output_path: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    torch_cache: Path,
    encoder_name: str = ENCODER_NAME,
) -> None:
    records = [record for record in read_jsonl(records_path) if record["split"] == split]
    if not records:
        raise ValueError(f"No records found for split {split!r}")

    encoder, transform = create_frozen_encoder(torch_cache, encoder_name)
    encoder = encoder.to(device)
    dataset = PreparedImageDataset(records, dataset_root, vocabulary, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in _empty_cache(ENCODER_DIMENSIONS[encoder_name], len(vocabulary["entries"]))
    }
    chunks["dish_ids"] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Extracting {split}", unit="batch"):
            features = encoder(batch["image"].to(device)).float().cpu().numpy()
            chunks["features"].append(features)
            for key in ("presence", "shares", "masses", "total_mass", "nutrients"):
                chunks[key].append(batch[key].numpy())
            chunks["dish_ids"].append(np.asarray(batch["dish_id"], dtype=str))

    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    arrays["encoder_name"] = np.asarray(encoder_name)
    arrays["vocabulary_keys"] = np.asarray([entry["key"] for entry in vocabulary["entries"]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def encode_single_image(
    image_path: Path,
    device: torch.device,
    torch_cache: Path,
    encoder_name: str = ENCODER_NAME,
) -> torch.Tensor:
    encoder, transform = create_frozen_encoder(torch_cache, encoder_name)
    encoder = encoder.to(device)
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        return encoder(tensor).float()


def validate_feature_cache(cache: dict, vocabulary: dict, encoder_name: str) -> None:
    """Reject old or incompatible cached inputs before training/evaluation."""
    if encoder_name not in ENCODER_DIMENSIONS:
        raise ValueError(f"Unsupported encoder: {encoder_name!r}")
    if cache["features"].shape[1] != ENCODER_DIMENSIONS[encoder_name]:
        raise ValueError("Cached feature dimension does not match the selected encoder; re-extract features.")
    if cache["presence"].shape[1] != len(vocabulary["entries"]):
        raise ValueError("Cached labels do not match the vocabulary; re-extract features.")
    if "encoder_name" in cache:
        if str(cache["encoder_name"].item()) != encoder_name:
            raise ValueError("Cached encoder does not match the selected encoder.")
    elif encoder_name != ENCODER_NAME:
        raise ValueError("New encoders require cache provenance; re-extract features.")
    if "vocabulary_keys" in cache:
        if cache["vocabulary_keys"].tolist() != [entry["key"] for entry in vocabulary["entries"]]:
            raise ValueError("Cached ingredient order does not match the vocabulary.")
