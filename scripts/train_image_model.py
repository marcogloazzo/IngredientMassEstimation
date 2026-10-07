"""Train spatial Q2L v1-v5 with partial ConvNeXt adjustment; never train on test."""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import nutrient_matrix, write_json
from ingredient_mass.features import resolve_device
from ingredient_mass.image_training import load_prepared, make_loader, positive_class_weights, run_epoch, prepared_fingerprint, save_predictions, auxiliary_mass_metrics
from ingredient_mass.metrics import full_metrics
from ingredient_mass.model import load_checkpoint, mean_training_mass
from ingredient_mass.spatial_model import DEPTH_MODEL_TYPES, CONDITIONAL_MODEL_TYPES, SPECIALIZED_MODEL_TYPES, MODEL_TYPE_V3, MODEL_TYPE_V4, MODEL_TYPE_V5, protected_result_namespaces, create_image_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/query2label_100.json")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/raw/nutrition5k_dataset")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed_100")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--torch-cache", type=Path, default=PROJECT_ROOT / "cache/torch")
    parser.add_argument("--depth-cache", type=Path, help="V2/V3/V4 aligned RGB-estimated depth cache; do not mix 224/320 geometry")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--presence-loss", choices=("asl", "bce"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-data", action="store_true", help="Check config/data only; no model loading or downloads")
    return parser.parse_args()


def capture_rng(loader):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
                  int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader.generator.get_state(),
    }


def restore_rng(state, loader):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])


def restore_scaler(scaler, state):
    # CPU checkpoints have an empty scaler state. When moving to CUDA,
    # keep the new scaler's initial state instead of loading an empty dict.
    if state:
        scaler.load_state_dict(state)


def make_scheduler(optimizer, config):
    settings = config.get("scheduler")
    if not settings:
        return None
    if settings["name"] != "warmup_cosine":
        raise ValueError("Unknown LR scheduler")
    import math
    warmup, horizon = int(settings["warmup_epochs"]), int(settings["total_epochs"])
    minimum = float(settings["min_factor"])
    if not 0 <= warmup < horizon or not 0 < minimum <= 1:
        raise ValueError("Invalid warmup/cosine settings")

    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(1., (step - warmup) / (horizon - warmup))
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("presence_loss", "batch_size", "epochs"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    expected_training = "stages_2_4" if config.get("model_type") in DEPTH_MODEL_TYPES else "last_stage"
    if config.get("encoder_training") != expected_training:
        raise ValueError(f"This model requires encoder_training={expected_training}")
    if config.get("shared_dims") != [384, 192]:
        raise ValueError("This implementation uses the agreed shared_dims [384, 192]")
    if config["presence_loss"] not in ("asl", "bce"):
        raise ValueError("Unknown ingredient presence loss")
    for key in ("epochs", "batch_size", "accumulation_steps", "patience"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    records, vocabulary = load_prepared(args.processed_dir, args.dataset_root, config)
    config["mass_normalization"] = "train_mean"
    config["mass_scale"] = mean_training_mass(
        record["total_mass"] for record in records if record["split"] == "train"
    )
    if config.get("model_type") in CONDITIONAL_MODEL_TYPES or config.get("split_policy") == "grouped_provisional":
        config["prepared_fingerprint"] = prepared_fingerprint(args.processed_dir)
    if config.get("model_type") in DEPTH_MODEL_TYPES:
        from ingredient_mass.depth import validate_depth_cache
        manifest = validate_depth_cache(args.depth_cache, records, config, args.dataset_root,
                                        verify_sources=config.get("model_type") in CONDITIONAL_MODEL_TYPES)
        config["depth_checkpoint_sha256"] = manifest["checkpoint_sha256"]
        if config.get("model_type") in CONDITIONAL_MODEL_TYPES:
            from ingredient_mass.depth import file_sha256
            config["depth_manifest_sha256"] = file_sha256(args.depth_cache / "manifest.json")
    if args.check_data:
        print(json.dumps({"config": config, "classes": len(vocabulary["entries"]),
                          "splits": {split: sum(r["split"] == split for r in records)
                                     for split in ("train", "validation", "test")}}, indent=2))
        return
    default_run = {MODEL_TYPE_V3: "query2label_100_v3_bce", MODEL_TYPE_V4: "query2label_100_v4_specialized", MODEL_TYPE_V5: "query2label_100_v5_auxiliary"}.get(config["model_type"], "query2label_100_asl")
    output_dir = (args.output_dir or PROJECT_ROOT / "artifacts" / default_run).resolve()
    if protected_result_namespaces(config["model_type"]) & {part.lower() for part in output_dir.parts}:
        raise ValueError("Training must use a new namespace, never an earlier version's results")
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None:
        raise FileExistsError("Output directory is not empty; use a new directory or --resume last.pt")
    device = resolve_device(args.device)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    resumed = load_checkpoint(args.resume, "cpu") if args.resume else None
    if resumed is not None:
        if resumed.get("model_type") != config["model_type"] or resumed["vocabulary"] != vocabulary:
            raise ValueError("Resume checkpoint architecture/vocabulary differs")
        for key, value in config.items():
            if key != "epochs" and resumed["config"].get(key) != value:
                raise ValueError(f"Resume config differs at {key}; use a new experiment")
    model = create_image_model(len(vocabulary["entries"]), config,
                               pretrained=resumed is None, torch_cache=args.torch_cache).to(device)
    if resumed:
        model.load_state_dict(resumed["model_state"], strict=True)
    elif config["model_type"] in CONDITIONAL_MODEL_TYPES:
        model.initialize_from_train(records, vocabulary)
    train_loader = make_loader(records, vocabulary, args.dataset_root, "train", int(config["batch_size"]),
                               args.workers, device, training=True, seed=seed,
                               config=config, depth_cache=args.depth_cache)
    validation_loader = make_loader(records, vocabulary, args.dataset_root, "validation", int(config["batch_size"]),
                                    args.workers, device, seed=seed, config=config, depth_cache=args.depth_cache)
    matrix_np = nutrient_matrix(vocabulary)
    matrix = torch.from_numpy(matrix_np).to(device)
    positive_weights = positive_class_weights(records, vocabulary).to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.heads.parameters()), "lr": float(config["learning_rate"])},
        {"params": list(model.encoder_parameters()), "lr": float(config["encoder_learning_rate"])},
    ], weight_decay=float(config["weight_decay"]))
    scheduler = make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config.get("amp", True)))
    start_epoch, stale = 1, 0
    selection = config.get("selection_metric", "ingredient_mass_l1_per_plate")
    if selection not in ("ingredient_mass_l1_per_plate", "total_mass_mae", "calorie_mae", "micro_f1", "validation_loss"):
        raise ValueError("Unsupported selection_metric")
    best = -float("inf") if selection == "micro_f1" else float("inf")
    if resumed:
        optimizer.load_state_dict(resumed["optimizer_state"])
        if scheduler is not None:
            if not resumed.get("scheduler_state"):
                raise ValueError("V3 resume requires the saved LR scheduler state")
            scheduler.load_state_dict(resumed["scheduler_state"])
        restore_scaler(scaler, resumed["scaler_state"])
        start_epoch, stale, best = resumed["epoch"] + 1, resumed["stale_epochs"], resumed["best_value"]
        restore_rng(resumed["rng_state"], train_loader)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    print(f"Device: {device}; presence loss: {config['presence_loss']}; classes: {len(vocabulary['entries'])}")
    print(f"Model: {config['model_type']}; encoder adjustment: {config['encoder_training']}", flush=True)
    print(f"Total-mass loss scale (training mean): {config['mass_scale']:.6f} g")
    print(f"Effective batch: {config['batch_size']} x {config['accumulation_steps']}; selection: {selection}")
    history_path = output_dir / "history.csv"
    has_history = history_path.is_file()
    with history_path.open("a" if resumed else "w", newline="", encoding="utf-8") as history:
        fields = ["epoch", "encoder_trainable", "seconds", "train_loss", "validation_loss", "micro_f1",
                  "ingredient_mass_l1_per_plate", "total_mass_mae", "calorie_mae"]
        if config["model_type"] in CONDITIONAL_MODEL_TYPES:
            fields += ["macro_f1", "false_ingredient_mass", "lr_heads", "lr_encoder"]
            fields += [f"{split}_{key}" for split in ("train", "validation") for key in config["loss_weights"]]
        if config["model_type"] in SPECIALIZED_MODEL_TYPES and config.get("auxiliary_total_mass", True):
            fields += ["auxiliary_total_mass_mae", "mass_consistency_gap_mae"]
        writer = csv.DictWriter(history, fieldnames=fields)
        if not resumed or not has_history:
            writer.writeheader()
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            begin = time.perf_counter()
            encoder_enabled = epoch > int(config.get("warmup_epochs", 3))
            model.set_encoder_trainable(encoder_enabled)
            print(f"Epoch {epoch}: encoder {config['encoder_training']} "
                  f"{'trainable' if encoder_enabled else 'frozen (warmup)'}", flush=True)
            train_losses, _, _ = run_epoch(model, train_loader, matrix, positive_weights, config, device,
                                           optimizer, scaler, f"Train {epoch}", collect=False)
            validation_losses, predictions, targets = run_epoch(
                model, validation_loader, matrix, positive_weights, config, device, description=f"Validation {epoch}")
            metrics = full_metrics(predictions["presence"], predictions["shares"], predictions["mass"],
                                   targets, matrix_np, float(config["presence_threshold"]))
            metrics.update(auxiliary_mass_metrics(predictions, targets))
            quantities = metrics["end_to_end_lightweight_mass"]
            row = {
                "epoch": epoch, "encoder_trainable": encoder_enabled,
                "seconds": time.perf_counter() - begin,
                "train_loss": train_losses["total"], "validation_loss": validation_losses["total"],
                "micro_f1": metrics["ingredient_recognition"]["micro_f1"],
                "ingredient_mass_l1_per_plate": quantities["ingredient_mass_l1_per_plate_grams"],
                "total_mass_mae": quantities["total_mass_mae_grams"],
                "calorie_mae": quantities["nutrients"]["calories"]["mae"],
            }
            if config["model_type"] in CONDITIONAL_MODEL_TYPES:
                row.update(macro_f1=metrics["ingredient_recognition"]["macro_f1"],
                           false_ingredient_mass=quantities["false_ingredient_mass_per_plate_grams"],
                           lr_heads=optimizer.param_groups[0]["lr"], lr_encoder=optimizer.param_groups[1]["lr"])
                row.update({f"{split}_{key}": parts[key] for split, parts in (("train", train_losses), ("validation", validation_losses))
                            for key in config["loss_weights"]})
            if "auxiliary_total_mass" in metrics:
                row.update(auxiliary_total_mass_mae=metrics["auxiliary_total_mass"]["mae_grams"],
                           mass_consistency_gap_mae=metrics["auxiliary_total_mass"]["consistency_gap_mae_grams"])
            writer.writerow(row)
            history.flush()
            improved = row[selection] > best if selection == "micro_f1" else row[selection] < best
            if improved:
                best, stale = row[selection], 0
            else:
                stale += 1
            payload = {
                "model_type": config["model_type"], "model_state": model.state_dict(), "config": config,
                "vocabulary": vocabulary, "epoch": epoch, "validation_metrics": metrics,
                "validation_losses": validation_losses, "selected_by": selection, "best_value": best,
            }
            if improved:
                torch.save(payload, output_dir / "best.pt")
                if config["model_type"] in SPECIALIZED_MODEL_TYPES:
                    save_predictions(output_dir / "best_validation_predictions.npz", predictions, targets, vocabulary)
                    write_json(output_dir / "best_validation_details.json", {
                        "split": "validation", "epoch": epoch, "model_type": config["model_type"],
                        "threshold": config["presence_threshold"], "prepared_fingerprint": config["prepared_fingerprint"],
                        "selected_by": selection, "metrics": metrics,
                    })
            if scheduler is not None:
                scheduler.step()
            # Save complete state every epoch, suitable for Colab reconnect/resume.
            payload.update({"optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                            "stale_epochs": stale, "rng_state": capture_rng(train_loader)})
            if scheduler is not None:
                payload["scheduler_state"] = scheduler.state_dict()
            temporary = output_dir / "last.tmp.pt"
            torch.save(payload, temporary)
            temporary.replace(output_dir / "last.pt")
            write_json(output_dir / "latest_validation.json", {"epoch": epoch, "metrics": metrics,
                                                               "losses": validation_losses})
            print(f"F1={row['micro_f1']:.3f}; ingredient L1/plate={row['ingredient_mass_l1_per_plate']:.1f} g; "
                  f"mass MAE={row['total_mass_mae']:.1f} g; calories MAE={row['calorie_mae']:.1f} kcal; "
                  f"epoch={row['seconds']:.0f} s")
            if stale >= int(config["patience"]) and encoder_enabled:
                print(f"Early stopping on validation {selection}")
                break
    print(f"Best model: {output_dir / 'best.pt'}; resumable model: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
