"""Native notebook visualization using synthetic saved reports; no model runs."""
import ast
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ingredient_mass.notebook_reports import _number, show_metrics, show_diagnostic, show_comparison, show_saved_report


def metrics():
    return {"ingredient_recognition":{"micro_f1":.6,"macro_f1":.5},
            "end_to_end_lightweight_mass":{"ingredient_mass_l1_per_plate_grams":150.,
                "ingredient_mass_mae_present_grams":15.,"false_ingredient_mass_per_plate_grams":30.,
                "total_mass_mae_grams":60.,"nutrients":{"calories":{"mae":80.}}},
            "auxiliary_total_mass":{"mae_grams":40.,"bias_grams":-10.,"consistency_gap_mae_grams":20.}}


def diagnostic():
    return {"split":"validation","samples":2,"epoch":48,
            "global_present":{"conditional_mae_g":14.,"final_mae_g":15.,"conditional_bias_g":-7.,"final_bias_g":-9.},
            "scenarios":[{"name":"baseline","ingredient_l1_g_per_plate":150.,"false_g_per_plate":30.,"total_mass_mae_g":60.}],
            "best_validation_thresholds":{},
            "classes_by_l1":[{"name":"chicken apple sausage with a long ingredient name","support":12,
                "conditional_mae_present_g":20.,"final_mae_present_g":25.,"gate_effect_mae_g":5.,
                "final_bias_present_g":-10.,"l1_contribution_g_per_plate":8.,"false_g_per_plate":2.},
                {"name":"absent class","support":0,"conditional_mae_present_g":None}],
            "recognition_buckets":[{"group":"TP","ingredient_entries":3,"mean_probability":.8,
                "conditional_mae_g":10.,"final_mae_g":12.,"final_bias_g":-5.}],
            "auxiliary_total_mass":{"mae_g":40.,"gap_to_ingredient_sum_mae_g":20.}}


def comparison():
    return {"warning":"Same prepared data and split","metrics":[
        {"metric":"L1 ingredientes (g/plato)","left":160.,"right":150.,"right_minus_left":-10.,"higher_is_better":False},
        {"metric":"MAE masa total (g)","left":60.,"right":65.,"right_minus_left":5.,"higher_is_better":False},
        {"metric":"Gramos falsos/plato","left":30.,"right":25.,"right_minus_left":-5.,"higher_is_better":False},
        {"metric":"F1 micro (%)","left":60.,"right":62.,"right_minus_left":2.,"higher_is_better":True}]}


class NativeReportTests(unittest.TestCase):
    def setUp(self):
        self.show = patch("matplotlib.pyplot.show").start()
        self.addCleanup(patch.stopall)

    def test_missing_values_and_counts(self):
        self.assertEqual(_number(None),"—")
        self.assertEqual(_number(float("nan")),"—")
        self.assertEqual(_number(12),"12")
        self.assertEqual(_number(12.345),"12.35")

    def test_metrics_use_percent_and_preserve_source(self):
        source = metrics(); before = copy.deepcopy(source)
        figures = show_metrics(source)
        table = figures[0].axes[0].tables[0]
        self.assertEqual(table[(1,1)].get_text().get_text(),"60.00")
        self.assertEqual(table[(2,1)].get_text().get_text(),"50.00")
        self.assertEqual(source,before)
        self.assertEqual(self.show.call_count,1)

    def test_diagnostic_handles_missing_class_support_without_html(self):
        source = diagnostic(); before = copy.deepcopy(source)
        figures = show_diagnostic(source)
        self.assertEqual(len(figures),4)
        class_table = figures[2].axes[0].tables[0]
        self.assertEqual(class_table[(2,2)].get_text().get_text(),"—")
        self.assertEqual(class_table[(1,1)].get_text().get_text(),"12")
        self.assertEqual(source,before)

    def test_comparison_directions_and_chart_values(self):
        figures = show_comparison(comparison())
        table = figures[0].axes[0].tables[0]
        self.assertEqual(table[(1,4)].get_text().get_text(),"Mejora")
        self.assertEqual(table[(2,4)].get_text().get_text(),"Empeora")
        self.assertEqual(table[(4,4)].get_text().get_text(),"Mejora")
        self.assertEqual([p.get_height() for p in figures[1].axes[0].patches],[160.,60.,30.,150.,65.,25.])

    def test_saved_reports_read_only_and_missing_report_prints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metrics.json"
            path.write_text(json.dumps(metrics()),encoding="utf-8")
            before = path.read_bytes()
            self.assertEqual(len(show_saved_report(path,"metrics")),1)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(len(list(Path(directory).iterdir())),1)
            with patch("sys.stdout",new_callable=io.StringIO) as output:
                self.assertEqual(show_saved_report(Path(directory)/"missing.json","diagnostic"),[])
            self.assertIn("Todavía no está disponible",output.getvalue())

    def test_v5_notebook_has_no_html_display_and_safe_flags(self):
        nb = json.loads((ROOT/"COLAB_QUERY2LABEL_V5.ipynb").read_text(encoding="utf-8"))
        all_code = ""
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            source = "".join(cell["source"]); all_code += source
            ast.parse(source)
        self.assertNotIn("HTML",all_code)
        self.assertNotIn("diagnostic.html",all_code)
        self.assertNotIn("comparison.html",all_code)
        self.assertIn("show_saved_report",all_code)
        self.assertIn('"matplotlib"',all_code)
        self.assertIn("RUN_TRAINING = False",all_code)
        self.assertIn("RUN_TEST = False",all_code)


if __name__ == "__main__": unittest.main()
