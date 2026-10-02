import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import nutrient_matrix, read_json  # noqa: E402
from ingredient_mass.features import resolve_device, validate_feature_cache  # noqa: E402
from ingredient_mass.metrics import full_metrics  # noqa: E402
from ingredient_mass.model import IngredientMassHeads, compute_loss, mean_training_mass  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train only the small ingredient heads.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "default.json"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "cache"
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "vocabulary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "ingredient_heads"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def create_loader(cache: dict[str, np.ndarray], batch_size: int, shuffle: bool):
    dataset = TensorDataset(
        torch.from_numpy(cache["features"]).float(),
        torch.from_numpy(cache["presence"]).float(),
        torch.from_numpy(cache["shares"]).float(),
        torch.from_numpy(cache["total_mass"]).float(),
        torch.from_numpy(cache["nutrients"]).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_model(model, loader, nutrient_tensor, positive_weights, config, device):
    model.eval()
    loss_totals = {key: 0.0 for key in ("total", "presence", "shares", "mass", "nutrition")}
    batches = 0
    collected = {key: [] for key in ("presence", "shares", "mass")}
    with torch.inference_mode():
        for features, presence, shares, total_mass, nutrients in loader:
            features = features.to(device)
            presence = presence.to(device)
            shares = shares.to(device)
            total_mass = total_mass.to(device)
            nutrients = nutrients.to(device)
            outputs = model(features)
            _, parts = compute_loss(
                outputs,
                presence,
                shares,
                total_mass,
                nutrients,
                nutrient_tensor,
                positive_weights,
                config["loss_weights"],
                mass_scale=float(config.get("mass_scale", 300)),
            )
            for key in loss_totals:
                loss_totals[key] += parts[key]
            batches += 1
            collected["presence"].append(outputs["presence_probabilities"].cpu().numpy())
            collected["shares"].append(outputs["shares"].cpu().numpy())
            collected["mass"].append(outputs["total_mass"].cpu().numpy())
    return (
        {key: value / max(batches, 1) for key, value in loss_totals.items()},
        {key: np.concatenate(value, axis=0) for key, value in collected.items()},
    )


def checkpoint_payload(
    model,
    input_dim,
    ingredient_count,
    config,
    vocabulary,
    epoch,
    validation_loss,
    metrics,
    selected_by,
):
    return {
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "ingredient_count": ingredient_count,
        "hidden_dim": int(config["hidden_dim"]),
        "dropout": float(config["dropout"]),
        "config": config,
        "vocabulary": vocabulary,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_metrics": metrics,
        "selected_by": selected_by,
    }


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(args.device)
    train_cache = load_cache(args.cache_dir / "train_features.npz")
    validation_cache = load_cache(args.cache_dir / "validation_features.npz")
    config["mass_normalization"] = "train_mean"
    config["mass_scale"] = mean_training_mass(train_cache["total_mass"])
    print(f"Total-mass loss scale (training mean): {config['mass_scale']:.6f} g")
    vocabulary = read_json(args.vocabulary)
    if len(vocabulary["entries"]) != int(config["top_k"]):
        raise ValueError("Vocabulary size does not match config top_k.")
    for cache in (train_cache, validation_cache):
        validate_feature_cache(cache, vocabulary, config.get("encoder", "efficientnet_b0"))
    matrix_np = nutrient_matrix(vocabulary)
    nutrient_tensor = torch.from_numpy(matrix_np).float().to(device)

    ingredient_count = train_cache["presence"].shape[1]
    input_dim = train_cache["features"].shape[1]
    model = IngredientMassHeads(
        input_dim=input_dim,
        ingredient_count=ingredient_count,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)

    positives = train_cache["presence"].sum(axis=0)
    negatives = len(train_cache["presence"]) - positives
    positive_weights = torch.from_numpy(
        np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0).astype(np.float32)
    ).to(device)

    train_loader = create_loader(train_cache, int(config["batch_size"]), shuffle=True)
    validation_loader = create_loader(
        validation_cache, int(config["batch_size"]), shuffle=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_heads.pt"
    checkpoint_paths = {
        "validation_loss": output_dir / "best_loss_heads.pt",
        "micro_f1": output_dir / "best_f1_heads.pt",
        "ingredient_mass_mae": output_dir / "best_ingredient_mass_heads.pt",
        "total_mass_mae": output_dir / "best_total_mass_heads.pt",
    }
    best_values = {
        "validation_loss": float("inf"),
        "micro_f1": -float("inf"),
        "ingredient_mass_mae": float("inf"),
        "total_mass_mae": float("inf"),
    }
    selection_metric = config.get("selection_metric", "validation_loss")
    if selection_metric not in best_values:
        raise ValueError(
            f"Unsupported selection_metric {selection_metric!r}; "
            f"choose one of {tuple(best_values)}"
        )
    epochs_without_improvement = 0

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=(
                "epoch",
                "train_loss",
                "validation_loss",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "ingredient_mass_mae",
                "total_mass_mae",
                "calorie_mae",
            ),
        )
        writer.writeheader()

        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
            for features, presence, shares, total_mass, nutrients in progress:
                features = features.to(device)
                presence = presence.to(device)
                shares = shares.to(device)
                total_mass = total_mass.to(device)
                nutrients = nutrients.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(features)
                loss, parts = compute_loss(
                    outputs,
                    presence,
                    shares,
                    total_mass,
                    nutrients,
                    nutrient_tensor,
                    positive_weights,
                    config["loss_weights"],
                    mass_scale=float(config["mass_scale"]),
                )
                loss.backward()
                optimizer.step()
                running_loss += parts["total"]
                progress.set_postfix(loss=f"{parts['total']:.4f}")

            train_loss = running_loss / max(len(train_loader), 1)
            validation_losses, predictions = evaluate_model(
                model,
                validation_loader,
                nutrient_tensor,
                positive_weights,
                config,
                device,
            )
            metrics = full_metrics(
                predictions["presence"],
                predictions["shares"],
                predictions["mass"],
                validation_cache,
                matrix_np,
                float(config["presence_threshold"]),
            )
            micro_f1 = metrics["ingredient_recognition"]["micro_f1"]
            micro_precision = metrics["ingredient_recognition"]["micro_precision"]
            micro_recall = metrics["ingredient_recognition"]["micro_recall"]
            quantity_metrics = metrics["end_to_end_lightweight_mass"]
            ingredient_mass_mae = quantity_metrics[
                "ingredient_mass_mae_present_grams"
            ]
            total_mass_mae = quantity_metrics["total_mass_mae_grams"]
            calorie_mae = quantity_metrics["nutrients"]["calories"]["mae"]
            validation_loss = validation_losses["total"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall,
                    "micro_f1": micro_f1,
                    "ingredient_mass_mae": ingredient_mass_mae,
                    "total_mass_mae": total_mass_mae,
                    "calorie_mae": calorie_mae,
                }
            )
            history_file.flush()
            print(
                f"Epoch {epoch}: train={train_loss:.4f}, val={validation_loss:.4f}, "
                f"micro-F1={micro_f1:.3f}, ingredient-MAE={ingredient_mass_mae:.1f} g, "
                f"total-mass-MAE={total_mass_mae:.1f} g"
            )

            current_values = {
                "validation_loss": validation_loss,
                "micro_f1": micro_f1,
                "ingredient_mass_mae": ingredient_mass_mae,
                "total_mass_mae": total_mass_mae,
            }
            improved_metrics = set()
            for metric_name, current_value in current_values.items():
                maximize = metric_name == "micro_f1"
                improved = (
                    current_value > best_values[metric_name]
                    if maximize
                    else current_value < best_values[metric_name]
                )
                if improved:
                    best_values[metric_name] = current_value
                    improved_metrics.add(metric_name)
                    torch.save(
                        checkpoint_payload(
                            model,
                            input_dim,
                            ingredient_count,
                            config,
                            vocabulary,
                            epoch,
                            validation_loss,
                            metrics,
                            metric_name,
                        ),
                        checkpoint_paths[metric_name],
                    )

            if selection_metric in improved_metrics:
                torch.save(
                    checkpoint_payload(
                        model,
                        input_dim,
                        ingredient_count,
                        config,
                        vocabulary,
                        epoch,
                        validation_loss,
                        metrics,
                        selection_metric,
                    ),
                    best_path,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(config["patience"]):
                    print(f"Early stopping on {selection_metric}.")
                    break

    print(f"Primary checkpoint ({selection_metric}): {best_path}")
    for metric_name, checkpoint_path in checkpoint_paths.items():
        print(f"Best {metric_name}: {checkpoint_path}")


if __name__ == "__main__":
    main()
