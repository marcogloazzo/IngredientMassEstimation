"""Inspect shapes/parameter counts with an untrained synthetic image; no downloads."""
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ingredient_mass.spatial_model import MODEL_TYPE_V2, MODEL_TYPE_V3, create_image_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/query2label_100.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(2)
    model = create_image_model(int(config["top_k"]), config, pretrained=False).eval()
    size = int(config.get("image_size", 224))
    with torch.inference_mode():
        kwargs = {"depth": torch.zeros(1, 1, size, size)} if config["model_type"] in (MODEL_TYPE_V2, MODEL_TYPE_V3) else {}
        outputs = model(torch.zeros(1, 3, size, size), return_attention=True, **kwargs)
    model.set_encoder_trainable(True)
    print(json.dumps({
        "synthetic_untrained": True,
        "shapes": {key: list(value.shape) for key, value in outputs.items()},
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable_partial": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters_encoder_partial": sum(p.numel() for p in model.encoder_parameters()),
        "parameters_heads": sum(p.numel() for p in model.heads.parameters()),
    }, indent=2))


if __name__ == "__main__":
    main()
