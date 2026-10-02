"""Frozen Depth Anything V2 integration and aligned RGB/depth data handling.

Only RGB is an external input. Cached maps contain relative depth inferred from
the deterministic RGB crop, never Nutrition5k sensor depth or nutrition labels.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import vectorize_record


DEPTH_MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_depth_config(config):
    depth = config.get("depth", {})
    if depth.get("backend") != "depth_anything_v2" or depth.get("frozen") is not True:
        raise ValueError("V2 requires frozen Depth Anything V2 relative-depth preprocessing")
    if depth.get("encoder") not in DEPTH_MODEL_CONFIGS:
        raise ValueError("Supported Depth Anything V2 encoders: vits, vitb, vitl")
    if depth.get("normalization") != "percentile_2_98" or depth.get("geometry") != "rgb_crop_230_224_v1":
        raise ValueError("Unsupported depth normalization or RGB/depth geometry")
    if int(depth.get("input_size", 0)) < 14:
        raise ValueError("Invalid Depth Anything input size")
    return depth


def rgb_crop(image):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    return TF.center_crop(TF.resize(image, 230, InterpolationMode.BILINEAR, antialias=True), [224, 224])


def open_rgb_crop(path):
    with Image.open(path) as source:
        return rgb_crop(ImageOps.exif_transpose(source).convert("RGB"))


def paired_tensors(crop, depth, training=False):
    """Shared flip for RGB/depth; photometric augmentation applies only to RGB."""
    from torchvision.transforms import ColorJitter
    from torchvision.transforms import functional as TF
    values = np.asarray(depth, dtype=np.float32)
    if values.shape != (224, 224) or not np.isfinite(values).all():
        raise ValueError("Expected a finite aligned 224x224 relative-depth map")
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Relative-depth cache values must be normalized to [0,1]")
    tensor = torch.from_numpy(values.copy()).unsqueeze(0)
    if training:
        if torch.rand(()).item() < 0.5:
            crop = TF.hflip(crop)
            tensor = tensor.flip(-1)
        crop = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(crop)
    image = TF.normalize(TF.to_tensor(crop), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, tensor


def normalize_relative_depth(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (224, 224) or not np.isfinite(values).all():
        raise ValueError("Depth Anything produced invalid depth")
    low, high = np.percentile(values, [2, 98])
    return np.clip((values - low) / max(float(high - low), 1e-8), 0, 1).astype(np.float32)


class FrozenDepthAnything:
    """Official relative-depth model; no gradients or pretrained downloads here."""
    def __init__(self, config, repo, checkpoint, device):
        depth = validate_depth_config(config)
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if not (repo / "depth_anything_v2/dpt.py").is_file():
            raise FileNotFoundError("Clone the official Depth-Anything-V2 repo; set --depth-anything-repo")
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        if revision != depth["repo_revision"]:
            raise ValueError("Depth Anything source revision differs from config; checkout the pinned commit")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing Depth Anything weights: {checkpoint}")
        self.checkpoint_sha256 = file_sha256(checkpoint)
        expected = config.get("depth_checkpoint_sha256")
        if expected and expected != self.checkpoint_sha256:
            raise ValueError("Depth Anything weights differ from those used for training")
        sys.path.insert(0, str(repo))
        from depth_anything_v2.dpt import DepthAnythingV2
        self.device = torch.device(device)
        self.input_size = int(depth["input_size"])
        self.model = DepthAnythingV2(encoder=depth["encoder"], **DEPTH_MODEL_CONFIGS[depth["encoder"]])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, crop):
        # Match the official OpenCV preprocessing, but honor the requested device
        # instead of infer_image's automatic CUDA choice.
        import cv2
        rgb = np.asarray(crop, dtype=np.uint8)
        if rgb.shape != (224, 224, 3):
            raise ValueError("Depth estimation must receive the same 224x224 RGB crop as ConvNeXt")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tensor, _ = self.model.image2tensor(bgr, self.input_size)
        values = self.model(tensor.to(self.device))
        values = F.interpolate(values[:, None], (224, 224), mode="bilinear", align_corners=True)[0, 0]
        return normalize_relative_depth(values.cpu().numpy())


def depth_cache_path(cache, record):
    identifier = record["dish_id"]
    if not identifier or any(not (c.isalnum() or c == "_") for c in identifier):
        raise ValueError("Invalid dish ID for depth cache")
    return Path(cache) / f"{identifier}.npy"


def validate_depth_cache(cache, records, config, dataset_root=None, verify_sources=False):
    validate_depth_config(config)
    if cache is None:
        raise ValueError("V2 requires --depth-cache; first run scripts/prepare_depth.py")
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing depth manifest: {manifest_path}; run prepare_depth.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("depth_config") != config["depth"]:
        raise ValueError("Depth cache configuration differs; generate a new cache")
    digest = manifest.get("checkpoint_sha256", "")
    if len(digest) != 64 or (config.get("depth_checkpoint_sha256") and digest != config["depth_checkpoint_sha256"]):
        raise ValueError("Depth cache weights differ from the trained model")
    for record in records:
        entry = manifest.get("entries", {}).get(record["dish_id"], {})
        path = depth_cache_path(cache, record)
        if entry.get("image") != record["image"] or not path.is_file():
            raise FileNotFoundError(f"Missing/mismatched cached depth for {record['dish_id']}")
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        if values.shape != (224, 224) or values.dtype != np.float16 or not np.isfinite(values).all():
            raise ValueError(f"Invalid depth cache: {path}")
        if values.min() < 0 or values.max() > 1:
            raise ValueError(f"Unnormalized depth cache: {path}")
        if verify_sources:
            if dataset_root is None or file_sha256(Path(dataset_root) / record["image"]) != entry.get("image_sha256"):
                raise ValueError(f"Cached RGB source changed: {record['dish_id']}")
    return manifest


class PairedDepthDataset(Dataset):
    def __init__(self, records, dataset_root, vocabulary, cache, training=False):
        self.records, self.dataset_root = records, Path(dataset_root)
        self.vocabulary, self.cache, self.training = vocabulary, Path(cache), training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        crop = open_rgb_crop(self.dataset_root / record["image"])
        depth = np.load(depth_cache_path(self.cache, record), allow_pickle=False)
        image, depth_tensor = paired_tensors(crop, depth, self.training)
        target = vectorize_record(record, self.vocabulary)
        return {
            "image": image, "depth": depth_tensor, "dish_id": record["dish_id"],
            "presence": torch.from_numpy(target["presence"]), "shares": torch.from_numpy(target["shares"]),
            "masses": torch.from_numpy(target["masses"]),
            "total_mass": torch.tensor(target["total_mass"], dtype=torch.float32),
            "nutrients": torch.from_numpy(target["nutrients"]),
        }
