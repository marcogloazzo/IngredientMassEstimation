import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import nutrient_matrix  # noqa: E402
from ingredient_mass.features import encode_single_image, resolve_device  # noqa: E402
from ingredient_mass.model import IngredientMassHeads, load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict ingredients, their masses, and nutrition from one image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ingredient_heads" / "best_heads.pt",
    )
    mass_group = parser.add_mutually_exclusive_group()
    mass_group.add_argument(
        "--total-mass-grams",
        type=float,
        help="Known total mass; useful for isolating ingredient-composition performance.",
    )
    mass_group.add_argument(
        "--omnifood-json",
        type=Path,
        help="prediction.json produced by OmniFood8K/Code/infer_single.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "single"
    )
    parser.add_argument(
        "--torch-cache", type=Path, default=PROJECT_ROOT / "cache" / "torch"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-ingredients", type=int, default=10)
    parser.add_argument("--minimum-share", type=float, default=0.01)
    return parser.parse_args()


def mass_from_omnifood_json(path: Path) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    mass = report["predictions"]["mass"]
    return float(mass["value"] if isinstance(mass, dict) else mass)


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    vocabulary = checkpoint["vocabulary"]
    model = IngredientMassHeads(
        input_dim=int(checkpoint["input_dim"]),
        ingredient_count=int(checkpoint["ingredient_count"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    encoder_name = checkpoint["config"].get("encoder", "efficientnet_b0")
    features = encode_single_image(args.image, device, args.torch_cache, encoder_name)
    with torch.inference_mode():
        outputs = model(features)
    probabilities = outputs["presence_probabilities"][0].cpu().numpy()
    shares = outputs["shares"][0].cpu().numpy()
    lightweight_mass = float(outputs["total_mass"][0].item())

    if args.total_mass_grams is not None:
        total_mass = float(args.total_mass_grams)
        mass_source = "user"
    elif args.omnifood_json is not None:
        total_mass = mass_from_omnifood_json(args.omnifood_json)
        mass_source = "omnifood"
    else:
        total_mass = lightweight_mass
        mass_source = "lightweight_head"

    matrix = nutrient_matrix(vocabulary)
    predicted_masses = shares * total_mass
    predicted_nutrients = predicted_masses @ matrix
    threshold = float(checkpoint["config"]["presence_threshold"])
    order = np.argsort(predicted_masses)[::-1]
    ingredients = []
    for index in order:
        if len(ingredients) >= args.max_ingredients:
            break
        if shares[index] < args.minimum_share and probabilities[index] < threshold:
            continue
        entry = vocabulary["entries"][int(index)]
        ingredients.append(
            {
                "name": entry["name"],
                "presence_probability": float(probabilities[index]),
                "share": float(shares[index]),
                "grams": float(predicted_masses[index]),
                "is_aggregate_other": entry["key"] == "__other__",
            }
        )

    report = {
        "image": str(args.image.resolve()),
        "mass_source": mass_source,
        "total_mass_grams": total_mass,
        "lightweight_mass_grams": lightweight_mass,
        "ingredients": ingredients,
        "nutrition": {
            name: float(value)
            for name, value in zip(vocabulary["nutrient_order"], predicted_nutrients)
        },
        "warning": (
            "The 'other' class uses an average nutrient density and is less interpretable. "
            "Invisible ingredients are inferred from correlations, not observed directly."
        ),
    }

    output_dir = args.output_dir.resolve() / args.image.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "ingredients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("name", "presence_probability", "share", "grams", "is_aggregate_other"),
        )
        writer.writeheader()
        writer.writerows(ingredients)

    print(f"Total mass ({mass_source}): {total_mass:.1f} g")
    for ingredient in ingredients:
        print(
            f"  {ingredient['name']:<30} {ingredient['grams']:7.1f} g "
            f"({100 * ingredient['share']:5.1f}%)"
        )
    print("Nutrition:")
    for name, value in report["nutrition"].items():
        unit = "kcal" if name == "calories" else "g"
        print(f"  {name:<15} {value:8.2f} {unit}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
