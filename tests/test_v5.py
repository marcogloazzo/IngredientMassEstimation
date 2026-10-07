"""V5 shared-MLP auxiliary-mass checks; synthetic tensors, no dataset training."""
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_v3 import fake_backbone, vocabulary, record, SyntheticPairs
from ingredient_mass.depth import image_size, validate_depth_config
from ingredient_mass.gate_diagnostics import diagnose_gate
from ingredient_mass.image_training import run_epoch, save_predictions, auxiliary_mass_metrics
from ingredient_mass.model import load_checkpoint
from ingredient_mass.spatial_model import MODEL_TYPE_V4, MODEL_TYPE_V5, create_image_model, model_from_checkpoint, protected_result_namespaces
from ingredient_mass.specialized_model import specialized_mass_loss

CONFIG = json.loads((ROOT / "config/query2label_100_v5.json").read_text(encoding="utf-8"))
SMALL = {**CONFIG, "attention_dim":16, "attention_feedforward":32,
         "attention_dropout":0., "dropout":0., "amp":False, "mass_scale":200.}


def model(config=None):
    with patch("torchvision.models.convnext_small", side_effect=fake_backbone):
        return create_image_model(3, config or SMALL, pretrained=False)


class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_exact_aux_only_ablation_new_identity(self):
        previous = json.loads((ROOT / "config/query2label_100_v4_aux_only.json").read_text())
        self.assertNotEqual(CONFIG["model_type"], previous["model_type"])
        self.assertEqual({k:v for k,v in CONFIG.items() if k != "model_type"},
                         {k:v for k,v in previous.items() if k != "model_type"})
        self.assertEqual(CONFIG["split_task_mlps"], False)
        self.assertEqual(CONFIG["auxiliary_total_mass"], True)

    def test_only_shared_mlp_auxiliary_and_conservation(self):
        m = model().eval()
        with torch.no_grad():
            out = m(torch.rand(1,3,320,320), depth=torch.rand(1,1,320,320), return_attention=True)
        self.assertTrue(hasattr(m.heads, "shared"))
        self.assertFalse(hasattr(m.heads, "presence_mlp"))
        self.assertFalse(hasattr(m.heads, "quantity_mlp"))
        self.assertEqual(out["attention_tokens"].shape, (1,3,500))
        self.assertTrue(torch.allclose(out["ingredient_masses"], out["presence_probabilities"]*out["conditional_grams"], atol=1e-6))
        self.assertTrue(torch.allclose(out["ingredient_masses"].sum(-1), out["total_mass"]))
        self.assertAlmostEqual(float(out["auxiliary_total_mass"]), 200., places=4)

    def test_auxiliary_initialization_excludes_validation(self):
        m = model({**SMALL,"mass_scale":75.}).eval()
        records = [record("a", "train", {"a":100}), record("b", "train", {"b":50}), record("v", "validation", {"a":10000})]
        m.initialize_from_train(records, vocabulary())
        self.assertAlmostEqual(float(m.heads.auxiliary_gram_scale), 75.)

    def test_nonzero_auxiliary_losses_and_gradients(self):
        m = model()
        out = m(torch.rand(2,3,64,64), depth=torch.rand(2,1,64,64))
        loss, parts = specialized_mass_loss(out, torch.tensor([[1.,0.,0.],[0.,1.,0.]]),
            torch.tensor([[100.,0.,0.],[0.,50.,0.]]), torch.tensor([100.,50.]),
            torch.ones(2,4), torch.ones(3,4), torch.ones(3), SMALL)
        loss.backward()
        self.assertGreater(parts["aux_mass"], 0.)
        self.assertGreater(parts["mass_consistency"], 0.)
        self.assertTrue(all(np.isfinite(v) for v in parts.values()))
        self.assertGreater(float(m.heads.auxiliary_mass_head[-1].bias.grad.abs().sum()), 0.)
        self.assertGreater(float(m.heads.conditional_head.bias.grad.abs().sum()), 0.)
        self.assertAlmostEqual(parts["total"], sum(CONFIG["loss_weights"][key]*parts[key] for key in CONFIG["loss_weights"]), places=5)

    def test_v5_rejects_wrong_architecture_flags(self):
        for changes in ({"split_task_mlps":True}, {"auxiliary_total_mass":False}):
            with self.assertRaisesRegex(ValueError,"V5 requires"):
                model({**SMALL,**changes})

    def test_saved_predictions_keep_auxiliary_and_support_diagnostics(self):
        losses, predictions, targets = run_epoch(model().eval(), DataLoader(SyntheticPairs(), batch_size=2),
            torch.ones(3,4), torch.ones(3), SMALL, torch.device("cpu"), description="Synthetic V5 validation")
        self.assertGreater(losses["aux_mass"], 0.)
        metrics = auxiliary_mass_metrics(predictions, targets)
        self.assertFalse(metrics["auxiliary_total_mass"]["used_for_deployed_masses"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"predictions.npz"
            save_predictions(path, predictions, targets, vocabulary())
            with np.load(path, allow_pickle=False) as source:
                report = diagnose_gate(dict(source), {"split":"validation","model_type":MODEL_TYPE_V5}, vocabulary())
        self.assertIn("auxiliary_total_mass", report)
        self.assertEqual(report["model_type"], MODEL_TYPE_V5)

    def test_checkpoint_and_cache_compatibility(self):
        m = model().eval()
        buffer = io.BytesIO()
        torch.save(dict(model_type=MODEL_TYPE_V5,config=SMALL,vocabulary=vocabulary(),model_state=m.state_dict()),buffer)
        buffer.seek(0)
        with patch("torchvision.models.convnext_small", side_effect=fake_backbone) as builder:
            restored = model_from_checkpoint(load_checkpoint(buffer), torch.device("cpu"))
        self.assertIsNone(builder.call_args.kwargs["weights"])
        image, depth = torch.rand(1,3,64,64),torch.rand(1,1,64,64)
        with torch.no_grad(): before, after = m(image,depth=depth),restored(image,depth=depth)
        for key in ("ingredient_masses","auxiliary_total_mass"):
            self.assertTrue(torch.allclose(before[key],after[key]))
        v3 = json.loads((ROOT/"config/query2label_100_v3.json").read_text())
        self.assertEqual(validate_depth_config(CONFIG),validate_depth_config(v3))
        self.assertEqual(image_size(CONFIG),320)
        self.assertEqual(protected_result_namespaces(MODEL_TYPE_V5),{"v2","v3","v4"})
        self.assertEqual(protected_result_namespaces(MODEL_TYPE_V4),{"v2","v3"})

    def test_notebook_clean_flags_and_isolated_paths(self):
        nb = json.loads((ROOT/"COLAB_QUERY2LABEL_V5.ipynb").read_text(encoding="utf-8"))
        flags = {}
        all_code = ""
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"],[])
            code = "".join(cell["source"]); all_code += code
            for node in ast.parse(code).body:
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,bool):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id.startswith("RUN_"):
                            flags[target.id] = node.value.value
        self.assertTrue(flags and all(v is False for v in flags.values()))
        self.assertIn('V5 = DRIVE / "v5"',all_code)
        self.assertIn('BRANCH = "v5-auxiliary-mass"',all_code)
        self.assertIn('config/query2label_100_v5.json',all_code)
        self.assertIn('("v3", V3_VALIDATION)',all_code)
        self.assertIn('("v4_split_only", V4_VALIDATION)',all_code)
        self.assertNotIn("VARIANT =",all_code)
        self.assertIn('MODE = "new"',all_code)


if __name__ == "__main__": unittest.main()
