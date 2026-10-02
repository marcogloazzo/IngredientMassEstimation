import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import read_json  # noqa: E402
from ingredient_mass.features import extract_split_features, resolve_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache frozen image encoder features for low-compute training."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "nutrition5k_dataset",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "cache"
    )
    parser.add_argument(
        "--torch-cache", type=Path, default=PROJECT_ROOT / "cache" / "torch"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "default.json"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "test"),
        default=("train", "validation", "test")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    encoder_name = config.get("encoder", "efficientnet_b0")
    device = resolve_device(args.device)
    vocabulary = read_json(args.processed_dir / "vocabulary.json")
    if len(vocabulary["entries"]) != int(config["top_k"]):
        raise ValueError("Vocabulary size does not match config top_k; prepare the new vocabulary first.")
    print(f"Device: {device}")
    print(f"Encoder: {encoder_name}; classes: {len(vocabulary['entries'])}")
    print("The ImageNet encoder is frozen; only feature extraction is performed.")
    for split in args.splits:
        output_path = args.cache_dir.resolve() / f"{split}_features.npz"
        extract_split_features(
            records_path=args.processed_dir.resolve() / "records.jsonl",
            dataset_root=args.dataset_root.resolve(),
            vocabulary=vocabulary,
            split=split,
            output_path=output_path,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            torch_cache=args.torch_cache,
            encoder_name=encoder_name,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
