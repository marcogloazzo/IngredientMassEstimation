import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.features import resolve_device  # noqa: E402
from ingredient_mass.metrics import find_best_global_threshold  # noqa: E402
from ingredient_mass.model import IngredientMassHeads, load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate the ingredient-presence threshold on validation data."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ingredient_heads" / "best_heads.pt",
    )
    parser.add_argument(
        "--validation-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "validation_features.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "ingredient_heads"
        / "best_heads_calibrated.pt",
    )
    parser.add_argument("--objective", choices=("micro_f1", "macro_f1"), default="micro_f1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    model = IngredientMassHeads(
        input_dim=int(checkpoint["input_dim"]),
        ingredient_count=int(checkpoint["ingredient_count"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    with np.load(args.validation_features) as archive:
        features = archive["features"].astype(np.float32)
        targets = archive["presence"].astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    probability_batches = []
    with torch.inference_mode():
        for (batch,) in loader:
            outputs = model(batch.to(device))
            probability_batches.append(outputs["presence_probabilities"].cpu().numpy())
    probabilities = np.concatenate(probability_batches, axis=0)
    threshold, metrics = find_best_global_threshold(
        probabilities, targets, objective=args.objective
    )

    checkpoint["config"] = dict(checkpoint["config"])
    previous_threshold = float(checkpoint["config"].get("presence_threshold", 0.5))
    checkpoint["config"]["presence_threshold"] = threshold
    checkpoint["calibration"] = {
        "split": "validation",
        "objective": args.objective,
        "previous_threshold": previous_threshold,
        "selected_threshold": threshold,
        "validation_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    print(json.dumps(checkpoint["calibration"], indent=2))
    print(f"Calibrated checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
