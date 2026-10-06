"""Single RGB inference plus numeric intermediate tensors and attention overlays."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import nutrient_matrix, write_json
from ingredient_mass.features import resolve_device
from ingredient_mass.model import load_checkpoint
from ingredient_mass.spatial_model import MODEL_TYPE_V2, MODEL_TYPE_V3, image_transforms, model_from_checkpoint


def save_intermediates(output_dir, tensor, outputs, vocabulary, top_k):
    """Align overlays with the actual resized/cropped network input, not the full photo."""
    directory = output_dir / "intermediates"
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.detach().float().cpu().numpy() for key, value in outputs.items()}
    arrays["normalized_input"] = tensor.cpu().numpy()
    np.savez_compressed(directory / "tensors.npz", **arrays,
                        ingredient_names=np.asarray([entry["name"] for entry in vocabulary["entries"]]))
    summaries = {key: {"shape": list(value.shape), "min": float(value.min()),
                       "max": float(value.max()), "mean": float(value.mean())}
                 for key, value in arrays.items()}
    write_json(directory / "tensor_summary.json", summaries)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = np.clip((arrays["normalized_input"][0].transpose(1, 2, 0) * std + mean) * 255, 0, 255).astype(np.uint8)
    preview = Image.fromarray(rgb)
    preview.save(directory / "input_crop.png")
    if "depth_input" in arrays:
        Image.fromarray((arrays["depth_input"][0, 0] * 255).clip(0, 255).astype(np.uint8)).save(
            directory / "estimated_relative_depth.png")
    order = np.argsort(arrays["ingredient_masses"][0])[::-1][:top_k]
    for rank, index in enumerate(order, 1):
        name = vocabulary["entries"][int(index)]["name"]
        safe_name = "".join(character if character.isalnum() else "_" for character in name)
        for key in ("attention", "attention_stage3", "attention_stage4"):
            if key not in arrays:
                continue
            attention = arrays[key][0, index]
            relative = (attention - attention.min()) / max(float(np.ptp(attention)), 1e-8)
            heatmap = Image.fromarray((relative * 255).astype(np.uint8)).resize(preview.size, Image.Resampling.BILINEAR)
            strength = np.asarray(heatmap, dtype=np.float32) / 255
            alpha = 0.55 * strength[:, :, None]
            red = np.zeros_like(rgb)
            red[:, :, 0] = 255
            overlay = ((1 - alpha) * rgb + alpha * red).astype(np.uint8)
            target = directory if key == "attention" else directory / key
            target.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(target / f"{rank:02d}_{safe_name}_attention.png")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/query2label_single")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--save-attention", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--depth-anything-repo", type=Path, help="V2/V3: official pinned Depth Anything source")
    parser.add_argument("--depth-checkpoint", type=Path, help="V2/V3: same frozen Depth Anything weights used for training")
    parser.add_argument("--total-mass-grams", type=float, help="Optional known mass for composition-only analysis")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.total_mass_grams is not None and args.total_mass_grams <= 0:
        parser.error("Known mass must be positive")
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image missing: {args.image}")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    vocabulary = checkpoint["vocabulary"]
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    depth = None
    if checkpoint["model_type"] in (MODEL_TYPE_V2, MODEL_TYPE_V3):
        if args.depth_anything_repo is None or args.depth_checkpoint is None:
            parser.error("Multiscale RGB inference requires --depth-anything-repo and --depth-checkpoint")
        from ingredient_mass.depth import FrozenDepthAnything, rgb_crop, paired_tensors
        crop = rgb_crop(image, checkpoint["config"])
        predictor = FrozenDepthAnything(checkpoint["config"], args.depth_anything_repo, args.depth_checkpoint, device)
        # Match the float16 map precision used by the training cache.
        tensor, depth = paired_tensors(crop, predictor.predict(crop).astype(np.float16), config=checkpoint["config"])
        tensor, depth = tensor.unsqueeze(0), depth.unsqueeze(0)
        # Free the frozen predictor before moving ConvNeXt to CUDA.
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        tensor = image_transforms()(image).unsqueeze(0)
    model = model_from_checkpoint(checkpoint, device)
    with torch.inference_mode():
        if depth is None:
            outputs = model(tensor.to(device), return_attention=args.save_attention)
        else:
            outputs = model(tensor.to(device), depth=depth.to(device), return_attention=args.save_attention)
    predicted_mass = float(outputs["total_mass"].item())
    total_mass = float(args.total_mass_grams) if args.total_mass_grams is not None else predicted_mass
    shares = outputs["shares"][0].cpu().numpy()
    probabilities = outputs["presence_probabilities"][0].cpu().numpy()
    masses = shares * total_mass
    nutrients = masses @ nutrient_matrix(vocabulary)
    order = np.argsort(masses)[::-1]
    threshold = float(checkpoint["config"]["presence_threshold"])
    ingredients = [{"name": vocabulary["entries"][int(index)]["name"],
                    "key": vocabulary["entries"][int(index)]["key"],
                    "probability": float(probabilities[index]), "recognized": bool(probabilities[index] >= threshold),
                    "share": float(shares[index]), "grams": float(masses[index])}
                   for index in order]
    report = {
        "image": str(args.image.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "predicted_total_mass_grams": predicted_mass, "used_total_mass_grams": total_mass,
        "mass_source": "user" if args.total_mass_grams is not None else "model",
        "presence_threshold": threshold, "ingredients": ingredients,
        "model_type": checkpoint["model_type"],
        "nutrition": {name: float(value) for name, value in zip(vocabulary["nutrient_order"], nutrients)},
        "warning": "Nutrition uses all classes, not just the printed top-k. Attention is not segmentation or proof of ingredient presence; invisible ingredients may be inferred from context.",
    }
    if depth is not None:
        report["depth"] = {"source": "estimated_from_rgb", "relative_not_metric": True,
                           "encoder": checkpoint["config"]["depth"]["encoder"],
                           "checkpoint_sha256": checkpoint["config"]["depth_checkpoint_sha256"]}
    output_dir = args.output_dir / args.image.stem
    if checkpoint["model_type"] == MODEL_TYPE_V3 and "v2" in {part.lower() for part in output_dir.parts}:
        raise ValueError("V3 predictions must not write inside the v2 namespace")
    if checkpoint["model_type"] == MODEL_TYPE_V3 and (output_dir / "prediction.json").is_file():
        previous = json.loads((output_dir / "prediction.json").read_text(encoding="utf-8"))
        if previous.get("model_type") != MODEL_TYPE_V3:
            raise ValueError("Prediction directory belongs to another model; choose a new output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_attention:
        report["intermediate_shapes"] = save_intermediates(output_dir, tensor, outputs, vocabulary, args.top_k)
    write_json(output_dir / "prediction.json", report)
    print(f"Total mass: {total_mass:.1f} g (model: {predicted_mass:.1f} g)")
    for ingredient in ingredients[:args.top_k]:
        print(f"{ingredient['name']:<25} {ingredient['grams']:7.1f} g; share={ingredient['share']:.1%}; p={ingredient['probability']:.3f}")
    print(json.dumps(report["nutrition"], indent=2))
    print(f"Reports: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
