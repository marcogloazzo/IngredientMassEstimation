import argparse
import importlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Verify files without loading any model.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "nutrition5k_dataset",
    )
    parser.add_argument(
        "--omnifood-root", type=Path, default=PROJECT_ROOT.parent / "OmniFood8K" / "Code"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checks = {
        "Nutrition5k metadata": args.dataset_root / "metadata",
        "Nutrition5k splits": args.dataset_root / "dish_ids" / "splits",
        "Nutrition5k overhead RGB": args.dataset_root / "imagery" / "realsense_overhead",
        "OmniFood inference script": args.omnifood_root / "infer_single.py",
        "OmniFood nutrition checkpoint": args.omnifood_root / "pth" / "ckpt_best.pth",
        "Depth Anything checkpoint": args.omnifood_root
        / "checkpoints"
        / "depth_anything_v2_vitl.pth",
    }
    results = {name: path.exists() for name, path in checks.items()}
    packages = {}
    for package in ("torch", "torchvision", "numpy", "PIL", "tqdm"):
        try:
            importlib.import_module(package)
            packages[package] = True
        except ImportError:
            packages[package] = False

    rgb_count = 0
    rgb_root = checks["Nutrition5k overhead RGB"]
    if rgb_root.is_dir():
        rgb_count = sum(1 for _ in rgb_root.glob("*/rgb.png"))
    report = {
        "files": {name: {"exists": results[name], "path": str(path)} for name, path in checks.items()},
        "packages": packages,
        "overhead_rgb_count": rgb_count,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
