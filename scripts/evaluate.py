import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import nutrient_matrix  # noqa: E402
from ingredient_mass.features import resolve_device, validate_feature_cache  # noqa: E402
from ingredient_mass.metrics import full_metrics  # noqa: E402
from ingredient_mass.model import IngredientMassHeads, load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ingredient and mass heads.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ingredient_heads" / "best_heads.pt",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "test_features.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    vocabulary = checkpoint["vocabulary"]
    config = checkpoint["config"]
    model = IngredientMassHeads(
        input_dim=int(checkpoint["input_dim"]),
        ingredient_count=int(checkpoint["ingredient_count"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    with np.load(args.features) as archive:
        cache = {key: archive[key] for key in archive.files}
    validate_feature_cache(cache, vocabulary, config.get("encoder", "efficientnet_b0"))
    dataset = TensorDataset(torch.from_numpy(cache["features"]).float())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probabilities, shares, masses = [], [], []
    with torch.inference_mode():
        for (features,) in loader:
            outputs = model(features.to(device))
            probabilities.append(outputs["presence_probabilities"].cpu().numpy())
            shares.append(outputs["shares"].cpu().numpy())
            masses.append(outputs["total_mass"].cpu().numpy())
    probabilities_np = np.concatenate(probabilities)
    shares_np = np.concatenate(shares)
    masses_np = np.concatenate(masses)

    matrix = nutrient_matrix(vocabulary)
    metrics = full_metrics(
        probabilities_np,
        shares_np,
        masses_np,
        cache,
        matrix,
        float(config["presence_threshold"]),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    names = [entry["name"] for entry in vocabulary["entries"]]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dish_id",
                "true_total_mass",
                "predicted_total_mass",
                "predicted_ingredients",
            ),
        )
        writer.writeheader()
        for index, dish_id in enumerate(cache["dish_ids"]):
            order = np.argsort(shares_np[index])[::-1][:8]
            predicted = [
                f"{names[item]}:{100 * shares_np[index, item]:.1f}%"
                for item in order
                if probabilities_np[index, item] >= float(config["presence_threshold"])
                or shares_np[index, item] >= 0.05
            ]
            writer.writerow(
                {
                    "dish_id": str(dish_id),
                    "true_total_mass": float(cache["total_mass"][index]),
                    "predicted_total_mass": float(masses_np[index]),
                    "predicted_ingredients": "; ".join(predicted),
                }
            )

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
