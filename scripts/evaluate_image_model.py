import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import nutrient_matrix, write_json
from ingredient_mass.features import resolve_device
from ingredient_mass.image_training import load_prepared, make_loader, positive_class_weights, run_epoch, save_predictions
from ingredient_mass.metrics import find_best_global_threshold, full_metrics
from ingredient_mass.model import load_checkpoint
from ingredient_mass.spatial_model import MODEL_TYPE_V2, model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate spatial model on original RGB images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/raw/nutrition5k_dataset")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed_100")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/query2label_evaluation")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--depth-cache", type=Path, help="Required for v2; reuse the training depth cache")
    parser.add_argument("--calibrate", action="store_true", help="Validation only; also save calibrated.pt")
    args = parser.parse_args()
    if args.calibrate and args.split != "validation":
        parser.error("Calibrate on validation only, never on test")
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    config, vocabulary = dict(checkpoint["config"]), checkpoint["vocabulary"]
    records, prepared_vocabulary = load_prepared(args.processed_dir, args.dataset_root, config)
    if prepared_vocabulary != vocabulary:
        raise ValueError("Prepared vocabulary/nutrient profiles differ from the trained checkpoint")
    if checkpoint["model_type"] == MODEL_TYPE_V2:
        from ingredient_mass.depth import validate_depth_cache
        validate_depth_cache(args.depth_cache, records, config)
    device = resolve_device(args.device)
    model = model_from_checkpoint(checkpoint, device)
    loader = make_loader(records, vocabulary, args.dataset_root, args.split, args.batch_size, args.workers, device,
                         config=config, depth_cache=args.depth_cache)
    matrix_np = nutrient_matrix(vocabulary)
    losses, predictions, targets = run_epoch(
        model, loader, torch.from_numpy(matrix_np).to(device),
        positive_class_weights(records, vocabulary).to(device), config, device,
        description=f"Evaluating {args.split}",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        threshold, recognition = find_best_global_threshold(predictions["presence"], targets["presence"])
        previous = config["presence_threshold"]
        config["presence_threshold"] = threshold
        checkpoint["config"] = config
        checkpoint["calibration"] = {"split": "validation", "previous_threshold": previous,
                                      "threshold": threshold, "metrics": recognition}
        torch.save(checkpoint, args.output_dir / "calibrated.pt")
    metrics = full_metrics(predictions["presence"], predictions["shares"], predictions["mass"], targets,
                           matrix_np, float(config["presence_threshold"]))
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "evaluation_details.json", {"split": args.split, "samples": len(loader.dataset),
                                                            "losses": losses, "threshold": config["presence_threshold"]})
    save_predictions(args.output_dir / "predictions.npz", predictions, targets, vocabulary)
    print(json.dumps(metrics, indent=2))
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
