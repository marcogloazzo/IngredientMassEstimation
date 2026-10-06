"""Compare saved reports only, flagging non-controlled V2/V3 comparisons."""
import argparse
import json
from html import escape
from pathlib import Path


def comparison(left, right):
    a = json.loads((Path(left) / "metrics.json").read_text(encoding="utf-8"))
    b = json.loads((Path(right) / "metrics.json").read_text(encoding="utf-8"))
    details = []
    for directory in (left, right):
        path = Path(directory) / "evaluation_details.json"
        details.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {})
    if details[0].get("split") and details[1].get("split") and details[0]["split"] != details[1]["split"]:
        raise ValueError("Compare the same split: do not mix validation and test")
    controlled = (bool(details[0].get("prepared_fingerprint")) and
                  details[0].get("prepared_fingerprint") == details[1].get("prepared_fingerprint") and
                  details[0].get("split") == details[1].get("split"))
    rows = []
    for label, section, key, factor, higher in (
        ("F1 micro (%)", "ingredient_recognition", "micro_f1", 100, True),
        ("F1 macro (%)", "ingredient_recognition", "macro_f1", 100, True),
        ("L1 ingredientes (g/plato)", "end_to_end_lightweight_mass", "ingredient_mass_l1_per_plate_grams", 1, False),
        ("MAE masa total (g)", "end_to_end_lightweight_mass", "total_mass_mae_grams", 1, False),
        ("Gramos falsos/plato", "end_to_end_lightweight_mass", "false_ingredient_mass_per_plate_grams", 1, False)):
        x, y = factor * a[section][key], factor * b[section][key]
        rows.append({"metric": label, "left": x, "right": y, "right_minus_left": y - x, "higher_is_better": higher})
    for name in ("calories", "fat", "carbohydrates", "protein"):
        x = a["end_to_end_lightweight_mass"]["nutrients"][name]["mae"]
        y = b["end_to_end_lightweight_mass"]["nutrients"][name]["mae"]
        rows.append({"metric": f"MAE {name} ({'kcal' if name == 'calories' else 'g'})", "left": x, "right": y,
                     "right_minus_left": y - x, "higher_is_better": False})
    return {"controlled_data_comparison": controlled, "left": str(left), "right": str(right), "details": details,
            "warning": "Same prepared data and split" if controlled else "Historical/descriptive only: data/vocabulary fingerprints differ or are unavailable; do not attribute changes only to architecture.", "metrics": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = comparison(args.left, args.right)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty/new comparison directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = "".join(f"<tr><td>{escape(row['metric'])}</td><td>{row['left']:.2f}</td><td>{row['right']:.2f}</td><td>{row['right_minus_left']:+.2f}</td></tr>" for row in result["metrics"])
    html = f"""<!doctype html><html lang="es"><meta charset="utf-8"><style>body{{font:16px Arial;color:#172b45;padding:24px}}table{{border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #dce5ef}}th{{background:#e5edff}}p{{line-height:1.5}}</style>
    <h2>Comparación de experimentos</h2><p>{escape(result['warning'])}</p><p>Izquierda: {escape(str(args.left))}<br>Derecha: {escape(str(args.right))}</p>
    <table><tr><th>Métrica</th><th>Izquierda</th><th>Derecha</th><th>Δ derecha − izquierda</th></tr>{rows}</table></html>"""
    (args.output_dir / "comparison.html").write_text(html, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
