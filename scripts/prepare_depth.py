"""Precompute frozen relative depth from RGB only; this script does not train."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ingredient_mass.data import write_json
from ingredient_mass.depth import (FrozenDepthAnything, depth_cache_path, file_sha256,
                                   open_rgb_crop, validate_depth_cache, validate_depth_config, image_size)
from ingredient_mass.features import resolve_device
from ingredient_mass.image_training import load_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/query2label_100_v2.json")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/raw/nutrition5k_dataset")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed_100")
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--depth-anything-repo", type=Path)
    parser.add_argument("--depth-checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--check-cache", action="store_true", help="Verify maps/sources only; no depth model loading")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("model_type") == "convnext_small_query2label_conditional_mass_v3" and "v2" in {part.lower() for part in args.depth_cache.parts}:
        raise ValueError("V3 needs a separate 320x320 depth cache, never the v2 namespace")
    validate_depth_config(config)
    records, _ = load_prepared(args.processed_dir, args.dataset_root, config)
    if args.check_cache:
        validate_depth_cache(args.depth_cache, records, config, args.dataset_root, verify_sources=True)
        print(f"Verified aligned depth cache: {len(records)} RGB images")
        return
    if args.depth_anything_repo is None or args.depth_checkpoint is None:
        parser.error("Depth generation requires --depth-anything-repo and --depth-checkpoint")
    predictor = FrozenDepthAnything(config, args.depth_anything_repo, args.depth_checkpoint, resolve_device(args.device))
    cache = args.depth_cache
    manifest_path = cache / "manifest.json"
    expected = {"schema_version": 1, "depth_config": config["depth"],
                "checkpoint_sha256": predictor.checkpoint_sha256}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing depth cache differs; choose a new --depth-cache")
    else:
        if cache.exists() and any(cache.iterdir()):
            raise FileExistsError("Non-empty cache without manifest; choose a new cache directory")
        manifest = {**expected, "entries": {}}
    cache.mkdir(parents=True, exist_ok=True)

    def save_manifest():
        temporary = cache / "manifest.tmp.json"
        write_json(temporary, manifest)
        temporary.replace(manifest_path)

    save_manifest()
    for record in tqdm(records, desc="Frozen Depth Anything V2", unit="image", file=sys.stdout):
        rgb_path = args.dataset_root / record["image"]
        fingerprint = file_sha256(rgb_path)
        path = depth_cache_path(cache, record)
        entry = manifest["entries"].get(record["dish_id"], {})
        if path.is_file() and entry.get("image_sha256") == fingerprint and entry.get("image") == record["image"]:
            # Do not silently reuse a corrupt or incompatible map.
            try:
                values = np.load(path, allow_pickle=False)
                valid = values.shape == (image_size(config), image_size(config)) and values.dtype == np.float16 and np.isfinite(values).all()
                valid = valid and values.min() >= 0 and values.max() <= 1
                if image_size(config) == 320:
                    valid = valid and entry.get("map_sha256") == file_sha256(path)
            except (ValueError, OSError):
                valid = False
            if valid:
                continue
        values = predictor.predict(open_rgb_crop(rgb_path, config))
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, values.astype(np.float16), allow_pickle=False)
        temporary.replace(path)
        manifest["entries"][record["dish_id"]] = {"image": record["image"], "image_sha256": fingerprint}
        if image_size(config) == 320:
            manifest["entries"][record["dish_id"]]["map_sha256"] = file_sha256(path)
        save_manifest()
    validate_depth_cache(cache, records, config)
    print(f"Complete depth cache: {cache}; {len(records)} maps, float16, {image_size(config)}x{image_size(config)}")


if __name__ == "__main__":
    main()
