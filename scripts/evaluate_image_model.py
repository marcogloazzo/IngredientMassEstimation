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
from ingredient_mass.image_training import load_prepared, make_loader, positive_class_weights, run_epoch, save_predictions, prepared_fingerprint, auxiliary_mass_metrics
from ingredient_mass.metrics import find_best_global_threshold, full_metrics
from ingredient_mass.model import load_checkpoint
from ingredient_mass.spatial_model import DEPTH_MODEL_TYPES, CONDITIONAL_MODEL_TYPES, MODEL_TYPE_V3, MODEL_TYPE_V4, MODEL_TYPE_V5, protected_result_namespaces, model_from_checkpoint
from ingredient_mass.depth import file_sha256


def main():
    parser = argparse.ArgumentParser(description="Evaluate spatial model on original RGB images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/raw/nutrition5k_dataset")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed_100")
    parser.add_argument("--output-dir", type=Path)
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
    default_report = {MODEL_TYPE_V3: "query2label_v3_evaluation", MODEL_TYPE_V4: "query2label_v4_evaluation", MODEL_TYPE_V5: "query2label_v5_evaluation"}.get(checkpoint["model_type"], "query2label_evaluation")
    args.output_dir = args.output_dir or PROJECT_ROOT / "outputs" / default_report
    if checkpoint["model_type"] in CONDITIONAL_MODEL_TYPES:
        forbidden = protected_result_namespaces(checkpoint["model_type"])
        if forbidden & {part.lower() for part in args.output_dir.parts}:
            raise ValueError("Evaluation must not write inside an earlier version's namespace")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Versioned report directory is not empty; choose a NEW directory to preserve previous results")
    config, vocabulary = dict(checkpoint["config"]), checkpoint["vocabulary"]
    records, prepared_vocabulary = load_prepared(args.processed_dir, args.dataset_root, config)
    if prepared_vocabulary != vocabulary:
        raise ValueError("Prepared vocabulary/nutrient profiles differ from the trained checkpoint")
    if checkpoint["model_type"] in DEPTH_MODEL_TYPES:
        from ingredient_mass.depth import validate_depth_cache
        validate_depth_cache(args.depth_cache, records, config, args.dataset_root,
                             verify_sources=checkpoint["model_type"] in CONDITIONAL_MODEL_TYPES)
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
                                      "threshold": threshold, "metrics": recognition,
                                      "method": "global_presence_threshold_only", "changes_ingredient_grams": False}
        torch.save(checkpoint, args.output_dir / "calibrated.pt")
    metrics = full_metrics(predictions["presence"], predictions["shares"], predictions["mass"], targets,
                           matrix_np, float(config["presence_threshold"]))
    metrics.update(auxiliary_mass_metrics(predictions, targets))
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "evaluation_details.json", {"split": args.split, "samples": len(loader.dataset),
                                                            "losses": losses, "threshold": config["presence_threshold"],
                                                            "epoch": checkpoint.get("epoch"), "model_type": checkpoint["model_type"],
                                                            "prepared_fingerprint": prepared_fingerprint(args.processed_dir),
                                                            "vocabulary_keys": [entry["key"] for entry in vocabulary["entries"]],
                                                            "checkpoint_sha256": file_sha256(args.checkpoint)})
    save_predictions(args.output_dir / "predictions.npz", predictions, targets, vocabulary)
    print(json.dumps(metrics, indent=2))
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
