"""Historical V2 ASL with grouped labels, isolated outputs; synthetic-only tests."""
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"tests"))
from test_v3 import fake_backbone
from ingredient_mass.depth import image_size,validate_depth_config
from ingredient_mass.image_training import check_experiment_output
from ingredient_mass.notebook_reports import show_comparison
from ingredient_mass.spatial_model import create_image_model,MODEL_TYPE_V2

CONFIG = json.loads((ROOT/"config/query2label_100_v2_asl_grouped.json").read_text(encoding="utf-8"))
ORIGINAL = json.loads((ROOT/"config/query2label_100_v2.json").read_text(encoding="utf-8"))


class V2GroupedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_original_v2_configuration_preserved_except_split_and_protection(self):
        extras = {"split_policy","accept_provisional_split","experiment_namespace","reference_v2_commit"}
        self.assertEqual({k:v for k,v in CONFIG.items() if k not in extras},ORIGINAL)
        self.assertEqual(CONFIG["presence_loss"],"asl")
        self.assertEqual(CONFIG["model_type"],MODEL_TYPE_V2)
        self.assertEqual(CONFIG["experiment_namespace"],"v2_grouped_asl")

    def test_v2_depth_geometry_224_not_320(self):
        self.assertEqual(image_size(CONFIG),224)
        self.assertEqual(validate_depth_config(CONFIG),validate_depth_config(ORIGINAL))
        v3 = json.loads((ROOT/"config/query2label_100_v3.json").read_text())
        self.assertNotEqual(validate_depth_config(CONFIG),validate_depth_config(v3))

    def test_same_model_parameters_and_forward_with_identical_seed(self):
        small = {"attention_dim":16,"attention_feedforward":32,"attention_dropout":0.,"dropout":0.}
        with patch("torchvision.models.convnext_small",side_effect=fake_backbone):
            torch.manual_seed(42); original = create_image_model(3,{**ORIGINAL,**small},pretrained=False).eval()
            torch.manual_seed(42); grouped = create_image_model(3,{**CONFIG,**small},pretrained=False).eval()
        self.assertEqual(original.state_dict().keys(),grouped.state_dict().keys())
        self.assertTrue(all(torch.equal(value,grouped.state_dict()[key]) for key,value in original.state_dict().items()))
        rgb,depth = torch.rand(1,3,224,224),torch.rand(1,1,224,224)
        with torch.no_grad(): a,b = original(rgb,depth=depth,return_attention=True),grouped(rgb,depth=depth,return_attention=True)
        self.assertEqual(b["attention_tokens"].shape,(1,3,245))
        for key in ("presence_probabilities","ingredient_masses","total_mass"):
            self.assertTrue(torch.equal(a[key],b[key]))

    def test_historical_namespaces_rejected_even_when_nested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            check_experiment_output(root/"v2_grouped_asl/runs/new",CONFIG)
            for path in (root/"v2/runs/new",root/"v6/reports/new",root/"outputs/new",
                         root/"v2/v2_grouped_asl/runs/new"):
                with self.assertRaises(ValueError): check_experiment_output(path,CONFIG)
            check_experiment_output(root/"v2/runs/old",ORIGINAL)

    def test_notebook_new_output_rejects_overwrite(self):
        nb = json.loads((ROOT/"COLAB_QUERY2LABEL_V2_GROUPED.ipynb").read_text(encoding="utf-8"))
        setup = ast.parse("".join(nb["cells"][6]["source"]))
        function = next(node for node in setup.body if isinstance(node,ast.FunctionDef) and node.name=="new_output")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); control = root/"v2_grouped_asl"
            namespace = {"CONTROL":control}
            exec(compile(ast.Module(body=[function],type_ignores=[]),"output_guard","exec"),namespace)
            namespace["new_output"](control/"runs/new")
            old = control/"reports/old"; old.mkdir(parents=True)
            file = old/"metrics.json"; file.write_text("original")
            with self.assertRaises(AssertionError): namespace["new_output"](old)
            with self.assertRaises(AssertionError): namespace["new_output"](root/"v2/runs")
            self.assertEqual(file.read_text(),"original")

    def test_notebook_safe_flags_accepted_labels_and_old_cache_read_only(self):
        nb = json.loads((ROOT/"COLAB_QUERY2LABEL_V2_GROUPED.ipynb").read_text(encoding="utf-8"))
        flags = {}; code = ""
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            self.assertIsNone(cell["execution_count"]); self.assertEqual(cell["outputs"],[])
            source = "".join(cell["source"]); code += source
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,bool):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id.startswith("RUN_"): flags[target.id]=node.value.value
        self.assertTrue(flags and all(value is False for value in flags.values()))
        self.assertIn('CONTROL = DRIVE / "v2_grouped_asl"',code)
        self.assertIn('BRANCH = "v2-grouped-asl"',code)
        self.assertIn("v3/data/processed_100_grouped_provisional.zip",code)
        self.assertIn("shutil.copytree(roots[0], DEPTH_CACHE)",code)
        self.assertNotIn("scripts/prepare_nutrition5k.py",code)
        self.assertNotIn("--clip-queries",code)
        self.assertNotIn("scripts/diagnose_mass_gate.py",code)
        self.assertNotIn("display(HTML",code)

    def test_native_comparison_label_identifies_v2_grouped(self):
        result = {"right":"/drive/v2_grouped_asl/reports","metrics":[
            {"metric":"F1 micro (%)","left":60.,"right":62.,"right_minus_left":2.,"higher_is_better":True}]}
        with patch("matplotlib.pyplot.show"),patch("sys.stdout",new_callable=io.StringIO):
            figures = show_comparison(result,"V3 vs V2 agrupada")
        self.assertEqual(figures[0].axes[0].tables[0][(0,2)].get_text().get_text(),"V2 agrupada")


if __name__ == "__main__": unittest.main()
