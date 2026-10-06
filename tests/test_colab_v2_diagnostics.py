"""Offline tests for section 11: synthetic arrays only, no model or training."""
import ast
import csv
import json
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from unittest.mock import patch

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "COLAB_QUERY2LABEL_V2.ipynb").read_text(encoding="utf-8"))
SOURCE = next("".join(cell["source"]) for cell in NOTEBOOK["cells"]
              if cell["cell_type"] == "code" and "def run_v2_diagnostics" in "".join(cell["source"]))


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.outputs = []
        self.namespace = dict(np=np, plt=plt, json=json, Counter=Counter, datetime=datetime,
                              csv=csv,
                              escape=escape, Path=Path, display=self.outputs.append, HTML=lambda value: value,
                              TOP_DIAGNOSTIC=12, MIN_CLASS_SUPPORT=1, SHOW_TEST_SUMMARY=True)
        tree = ast.parse(SOURCE)
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
                               type_ignores=[])
        exec(compile(functions, "colab_v2_diagnostics", "exec"), self.namespace)
        self.arrays = {
            "probabilities": np.array([[0.9, 0.1], [0.8, 0.8], [0.1, 0.9]]),
            "predicted_shares": np.array([[0.8, 0.2], [0.6, 0.4], [0.2, 0.8]]),
            "predicted_total_mass": np.array([100., 200., 100.]),
            "masses": np.array([[100., 0.], [100., 100.], [0., 100.]]),
            "total_mass": np.array([100., 200., 100.]),
            "presence": np.array([[1., 0.], [1., 1.], [0., 1.]]),
            "nutrients": np.array([[100., 0., 100., 0.], [200., 0., 200., 0.], [100., 0., 100., 0.]]),
            "dish_ids": np.array(["a", "b", "c"]), "vocabulary_keys": np.array(["rice", "__other__"]),
        }
        self.vocabulary = {
            "nutrient_order": ["calories", "fat", "carbohydrates", "protein"],
            "entries": [{"key": key, "name": name,
                         "nutrients_per_gram": dict(calories=1., fat=0., carbohydrates=1., protein=0.)}
                        for key, name in (("rice", "Rice <script>"), ("__other__", "other"))],
        }

    def test_notebook_syntax_and_section_read_only(self):
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        section = next(c for c in NOTEBOOK["cells"] if c["cell_type"] == "code"
                       and "def run_v2_diagnostics" in "".join(c["source"]))
        # The user saved an executed V2 dashboard. Its outputs are evidence and
        # must not be cleared merely to satisfy a template-only assertion.
        self.assertTrue(section["execution_count"] is None or isinstance(section["execution_count"], int))
        self.assertIsInstance(section["outputs"], list)
        for forbidden in ("subprocess", "torch.load", "run_stream(", "savefig", "write_text", "torch.save"):
            self.assertNotIn(forbidden, SOURCE)
        headings = ["".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "markdown"]
        self.assertTrue(any(s.startswith("## 11.") for s in headings))

    def test_statistics_l1_and_false_grams(self):
        stats = self.namespace["diag_statistics"](self.arrays, 0.5)
        np.testing.assert_allclose(stats["l1"], [40, 40, 40])
        np.testing.assert_allclose(stats["false_mass"].sum(axis=1), [20, 0, 20])
        np.testing.assert_allclose(stats["oracle_l1"], stats["l1"])
        np.testing.assert_allclose(stats["f1"], [1, 1])
        self.assertTrue(stats["exact"].all())

    def test_missing_threshold_omits_binary_diagnostics(self):
        stats = self.namespace["diag_statistics"](self.arrays, None)
        self.assertNotIn("f1", stats)
        self.assertNotIn("detected", stats)
        with self.assertRaises(ValueError):
            self.namespace["diag_statistics"](self.arrays, 2)

    def test_presence_mass_consistency(self):
        arrays = dict(self.arrays, presence=np.zeros((3, 2)))
        with self.assertRaisesRegex(ValueError, "presencia"):
            self.namespace["diag_statistics"](arrays, 0.5)

    def test_class_contributions_support_and_other(self):
        records = [{"split": "train", "ingredients": [
            {"ingredient_id": "rice", "grams": 1}, {"ingredient_id": "rice", "grams": 2},
            {"ingredient_id": "unknown1", "grams": 1}, {"ingredient_id": "unknown2", "grams": 1}]}]
        stats = self.namespace["diag_statistics"](self.arrays, 0.5)
        rows = self.namespace["diag_class_rows"](self.arrays, stats, self.vocabulary, records)
        self.assertEqual([r["Train platos"] for r in rows], [1, 1])
        self.assertEqual([r["Val platos"] for r in rows], [2, 2])
        self.assertAlmostEqual(sum(r["L1 aportado g/plato"] for r in rows), stats["l1"].mean())
        self.assertAlmostEqual(sum(r["Gramos falsos g/plato"] for r in rows), 40 / 3)

    def test_nutrient_reference_and_zero_denominator(self):
        stats = self.namespace["diag_statistics"](self.arrays, 0.5)
        rows = self.namespace["diag_nutrition_rows"](self.arrays, stats, self.vocabulary)
        reference = [r for r in rows if r["Escenario"].startswith("Gramos reales")]
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["MAE"] == 0 for r in reference))
        self.assertIsNone(next(r["PMAE %"] for r in rows if r["Nutriente"] == "fat"))
        changed = dict(self.vocabulary, entries=self.vocabulary["entries"][::-1])
        with self.assertRaisesRegex(ValueError, "vocabulario"):
            self.namespace["diag_nutrition_rows"](self.arrays, stats, changed)

    def test_npz_loading_and_invalid_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            np.savez(path, **self.arrays)
            arrays = self.namespace["diag_prediction_arrays"](path)
            self.assertEqual(arrays["masses"].shape, (3, 2))
            for replacement in ({"probabilities": np.full((3, 2), np.nan)},
                                {"predicted_shares": np.zeros((3, 2))},
                                {"dish_ids": np.array(["a", "a", "c"])},
                                {"predicted_total_mass": np.array([-1, 200, 100])}):
                np.savez(path, **dict(self.arrays, **replacement))
                with self.assertRaises(ValueError):
                    self.namespace["diag_prediction_arrays"](path)

    def test_html_escaping_and_limit(self):
        self.namespace["diag_table"]([{"<header>": "<script>"}, {"<header>": "hidden"}], 1)
        self.assertIn("&lt;header&gt;", self.outputs[-1])
        self.assertIn("&lt;script&gt;", self.outputs[-1])
        self.assertNotIn("hidden", self.outputs[-1])

    def test_dashboard_tolerates_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.namespace.update(RUN=base / "run", VALIDATION=base / "validation", TEST=base / "test",
                                  PROCESSED=base / "processed", CONFIG=base / "config.json")
            self.namespace["run_v2_diagnostics"]()
        combined = "\n".join(self.outputs)
        self.assertIn("Falta history.csv", combined)
        self.assertIn("predictions.npz", combined)
        self.assertIn("NO DISPONIBLE", combined)

    def test_dashboard_with_synthetic_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("run", "validation", "processed"):
                (base / name).mkdir()
            config = {"selection_metric": "ingredient_mass_l1_per_plate", "patience": 8, "mass_scale": 200,
                      "loss_weights": {"mass": 1}, "batch_size": 4, "accumulation_steps": 8}
            (base / "run/config.json").write_text(json.dumps(config), encoding="utf-8")
            (base / "processed/vocabulary.json").write_text(json.dumps(self.vocabulary), encoding="utf-8")
            records = [{"dish_id": name, "split": "validation", "total_mass": float(mass), "ingredients": []}
                       for name, mass in zip(self.arrays["dish_ids"], self.arrays["total_mass"])]
            (base / "processed/records.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            np.savez(base / "validation/predictions.npz", **self.arrays)
            details = {"split": "validation", "samples": 3, "threshold": 0.5}
            (base / "validation/evaluation_details.json").write_text(json.dumps(details), encoding="utf-8")
            recognition = {key: 1.0 for key in ("micro_f1", "macro_f1", "micro_precision", "micro_recall", "exact_match")}
            quantities = {"ingredient_mass_l1_per_plate_grams": 40., "total_mass_mae_grams": 0., "total_mass_pmae": 0.,
                          "false_ingredient_mass_per_plate_grams": 40 / 3, "nutrients": {}}
            report = {"ingredient_recognition": recognition, "end_to_end_lightweight_mass": quantities,
                      "composition_with_true_total_mass": quantities}
            (base / "validation/metrics.json").write_text(json.dumps(report), encoding="utf-8")
            self.namespace.update(RUN=base / "run", VALIDATION=base / "validation", TEST=base / "test",
                                  PROCESSED=base / "processed", CONFIG=base / "config.json")
            self.namespace["run_v2_diagnostics"]()
            combined = "\n".join(self.outputs)
            self.assertNotIn("No se pudo completar", combined)
            self.assertIn("PLATE_INDEX (validation)", combined)
            self.assertIn("Rice &lt;script&gt;", combined)
            self.outputs.clear()
            report["end_to_end_lightweight_mass"]["ingredient_mass_l1_per_plate_grams"] = 999
            (base / "validation/metrics.json").write_text(json.dumps(report), encoding="utf-8")
            self.namespace["run_v2_diagnostics"]()
            self.assertIn("posiblemente son evaluaciones diferentes", "\n".join(self.outputs))

    def test_history_best_epoch_and_duplicate_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            fields = ["epoch", "train_loss", "validation_loss", "micro_f1",
                      "ingredient_mass_l1_per_plate", "total_mass_mae", "calorie_mae"]
            rows = [dict(zip(fields, values)) for values in (
                (1, 1.0, 1.2, 0.5, 100, 20, 30), (2, 0.9, 1.0, 0.6, 80, 15, 25),
                (3, 0.8, 0.95, 0.6, 85, 12, 22))]
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            with patch.object(plt, "show"):
                self.namespace["diag_history"](path, {"selection_metric": "ingredient_mass_l1_per_plate", "patience": 8})
            combined = "\n".join(self.outputs)
            self.assertIn("Mejor época registrada", combined)
            self.assertIn("<strong>2</strong>", combined)
            self.assertIn("<strong>1</strong>", combined)
            with path.open("a", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=fields).writerow(rows[-1])
            self.namespace["diag_history"](path, {})
            self.assertIn("épocas duplicadas", self.outputs[-1])

    def test_attention_uniform_baseline_and_null_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "intermediates").mkdir()
            path = folder / "intermediates/tensors.npz"
            tensors = dict(attention_tokens=np.ones((1, 2, 245)),
                           ingredient_names=np.array(["rice", "other"]),
                           ingredient_masses=np.array([[80., 20.]]),
                           presence_probabilities=np.array([[0.9, 0.1]]))
            np.savez(path, **tensors)
            self.namespace["diag_inference_tensors"](folder)
            combined = "\n".join(self.outputs)
            self.assertIn("80,00", combined)
            self.assertIn("20,00", combined)
            self.assertIn("1,000", combined)
            np.savez(path, **dict(tensors, attention_tokens=np.zeros((1, 2, 245))))
            self.namespace["diag_inference_tensors"](folder)
            self.assertIn("atención nula", self.outputs[-1])


if __name__ == "__main__":
    unittest.main()
