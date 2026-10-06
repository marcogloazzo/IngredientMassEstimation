"""Diagnose p*c using saved validation NPZ only. No weights, GPU or training."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingredient_mass.gate_diagnostics import diagnose_gate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value):
    return "—" if value is None else f"{value:,.2f}"


def render_report(report):
    baseline = report["scenarios"][0]
    def table(rows, columns):
        heading = "".join(f"<th>{escape(label)}</th>" for key, label in columns)
        body = []
        for row in rows:
            def formatted(value):
                if isinstance(value, bool):
                    return "Sí" if value else "No"
                return number(value) if isinstance(value, (int, float)) or value is None else str(value)
            cells = "".join(f"<td>{escape(formatted(row.get(key)))}</td>" for key, label in columns)
            body.append(f"<tr>{cells}</tr>")
        return f"<div class='scroll'><table><thead><tr>{heading}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    cards = "".join(f"<div class='card'><small>{escape(label)}</small><strong>{number(value)} {unit}</strong></div>" for label, value, unit in (
        ("L1 ingredientes", baseline["ingredient_l1_g_per_plate"], "g/plato"),
        ("Masa fantasma", baseline["false_g_per_plate"], "g/plato"),
        ("MAE presente · c", report["global_present"]["conditional_mae_g"], "g/ingrediente"),
        ("MAE presente · p×c", report["global_present"]["final_mae_g"], "g/ingrediente"),
    ))
    scenarios = table(report["scenarios"] + list(report["best_validation_thresholds"].values()), [
        ("name", "Escenario"), ("oracle_not_deployable", "Oracle · no desplegable"), ("threshold", "τ"),
        ("ingredient_l1_g_per_plate", "L1 g/plato"), ("present_l1_g_per_plate", "L1 presentes g/plato"),
        ("false_g_per_plate", "Falsos g/plato"), ("total_mass_mae_g", "MAE total g"),
    ])
    classes = table(report["classes_by_l1"][:30], [
        ("name", "Ingrediente"), ("support", "Soporte"), ("conditional_mae_present_g", "MAE c presente g"),
        ("final_mae_present_g", "MAE p×c presente g"), ("gate_effect_mae_g", "Δ MAE por gate"),
        ("conditional_bias_present_g", "Sesgo c g"), ("final_bias_present_g", "Sesgo p×c g"),
        ("mean_gate_present", "p medio presente"), ("l1_contribution_g_per_plate", "L1 g/plato"), ("false_g_per_plate", "Falsos g/plato"),
    ])
    buckets = table(report["recognition_buckets"], [("group", "Grupo"), ("ingredient_entries", "Entradas"),
        ("mean_probability", "p medio"), ("conditional_mae_g", "MAE c g"), ("final_mae_g", "MAE p×c g"), ("final_bias_g", "Sesgo final g")])
    warnings = "".join(f"<li>{escape(item)}</li>" for item in report["warnings"])
    auxiliary = ""
    if "auxiliary_total_mass" in report:
        auxiliary = "<h2>Masa total auxiliar · no usada para reescalar</h2>" + table([report["auxiliary_total_mass"]], [
            ("mae_g", "MAE auxiliar g"), ("bias_g", "Sesgo auxiliar g"),
            ("gap_to_ingredient_sum_mae_g", "Discrepancia con suma de ingredientes g")])
    return f"""<!doctype html><html lang="es"><meta charset="utf-8"><title>Diagnóstico p × c</title>
    <style>body{{font:15px system-ui;background:#f4f7fb;color:#19334b;margin:28px}}h1,h2{{color:#153f67}}.cards{{display:flex;gap:14px;flex-wrap:wrap}}.card{{background:white;border:1px solid #d7e4ef;border-radius:22px;padding:18px 24px;box-shadow:0 5px 15px #17334b0a}}small,strong{{display:block}}strong{{font-size:25px;margin-top:8px}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dce5ef;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}th{{background:#e8effa}}.scroll{{overflow:auto;border-radius:14px}}p,li{{line-height:1.6}}.notice{{background:#fff3d5;border-radius:16px;padding:14px 20px}}</style>
    <h1>Diagnóstico del producto p × c</h1><p>Validación · {report['samples']} imágenes · época {escape(str(report['epoch']))} · solo predicciones guardadas.</p>
    <div class="cards">{cards}</div><p>MAE presente se promedia por ingrediente real; L1 se suma por plato. No son la misma métrica.</p>
    <h2>Contrafactuales con c fijo</h2>{scenarios}<p>Los mejores umbrales fueron elegidos sobre esta misma validación: son exploratorios. No se modificó el modelo.</p>
    <h2>Ingredientes que dominan el error</h2>{classes}<p>Δ MAE positivo: multiplicar por p empeora el error presente; negativo: lo mejora. Sesgo negativo: subestimación. Soporte bajo: resultado incierto.</p>
    <h2>Presencia correcta vs errores de reconocimiento</h2>{buckets}<p>TP/FN corresponden a ingredientes reales; FP/TN a ausentes. El umbral usado es {report['recognition_threshold']:.3f}.</p>
    {auxiliary}<div class="notice"><ul>{warnings}</ul></div></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--details", type=Path, help="Defaults to sibling evaluation_details.json")
    parser.add_argument("--vocabulary", type=Path, help="Optional matching train vocabulary for names/nutrient errors")
    parser.add_argument("--output-dir", type=Path, required=True, help="New/empty directory; never overwrites source reports")
    args = parser.parse_args()
    details_path = args.details or args.predictions.with_name("evaluation_details.json")
    if args.output_dir.resolve() in (args.predictions.parent.resolve(), details_path.parent.resolve()):
        raise ValueError("Use a separate output directory, not the source evaluation directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Diagnostic output is not empty; choose a new folder")
    details = json.loads(details_path.read_text(encoding="utf-8"))
    vocabulary = None
    if args.vocabulary:
        expected = details.get("prepared_fingerprint", {}).get("vocabulary.json")
        if expected and sha256(args.vocabulary) != expected:
            raise ValueError("Vocabulary file fingerprint differs from evaluation")
        vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    with np.load(args.predictions, allow_pickle=False) as source:
        report = diagnose_gate({key: source[key] for key in source.files}, details, vocabulary)
    report["source_files"] = {"predictions": str(args.predictions.resolve()), "predictions_sha256": sha256(args.predictions),
                              "details": str(details_path.resolve()), "details_sha256": sha256(details_path)}
    if args.vocabulary:
        report["source_files"].update(vocabulary=str(args.vocabulary.resolve()), vocabulary_sha256=sha256(args.vocabulary))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (args.output_dir / "diagnostic.html").write_text(render_report(report), encoding="utf-8")
    print(f"Validation epoch {report['epoch']}; samples={report['samples']}; no model loaded.")
    for row in report["scenarios"] + list(report["best_validation_thresholds"].values()):
        threshold = f" tau={row['threshold']:.3f}" if "threshold" in row else ""
        print(f"{row['name']}{threshold}: L1={row['ingredient_l1_g_per_plate']:.2f} g/plate; false={row['false_g_per_plate']:.2f} g/plate; total MAE={row['total_mass_mae_g']:.2f} g")
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
