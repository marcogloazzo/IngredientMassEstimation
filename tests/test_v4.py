"""V4 synthetic verification and offline diagnostics; no Nutrition5k training."""
import ast
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_v3 import fake_backbone, vocabulary, record, SyntheticPairs
from ingredient_mass.depth import image_size, validate_depth_config
from ingredient_mass.gate_diagnostics import diagnose_gate
from ingredient_mass.image_training import run_epoch, save_predictions, auxiliary_mass_metrics
from ingredient_mass.model import load_checkpoint
from ingredient_mass.specialized_model import specialized_mass_loss
from ingredient_mass.spatial_model import MODEL_TYPE_V4, create_image_model, model_from_checkpoint
from torch.utils.data import DataLoader

CONFIG = json.loads((ROOT / "config/query2label_100_v4.json").read_text(encoding="utf-8"))
SMALL = {**CONFIG, "attention_dim": 16, "attention_feedforward": 32,
         "attention_dropout": 0., "dropout": 0., "amp": False, "mass_scale": 200.}


def model(config=None, classes=3):
    with patch("torchvision.models.convnext_small", side_effect=fake_backbone):
        return create_image_model(classes, config or SMALL, pretrained=False)


def arrays():
    p = np.array([[.5, .2], [.1, .8]])
    c = np.array([[100., 40.], [20., 60.]])
    truth = np.array([[100., 0.], [0., 50.]])
    masses = p * c
    total = masses.sum(1)
    return dict(probabilities=p, conditional_grams=c, masses=truth,
                predicted_shares=masses / total[:, None], predicted_total_mass=total,
                total_mass=truth.sum(1), dish_ids=np.array(["dish_a", "dish_b"]), vocabulary_keys=np.array(["a", "b"]))


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_independent_mlps_and_unchanged_mass_rule(self):
        m = model(classes=100).eval()
        with torch.no_grad():
            out = m(torch.rand(1, 3, 320, 320), depth=torch.rand(1, 1, 320, 320), return_attention=True)
        self.assertFalse(hasattr(m.heads, "shared"))
        self.assertNotEqual(m.heads.presence_mlp[1].weight.data_ptr(), m.heads.quantity_mlp[1].weight.data_ptr())
        self.assertEqual(out["attention_tokens"].shape, (1, 100, 500))
        self.assertTrue(torch.allclose(out["ingredient_masses"], out["presence_probabilities"] * out["conditional_grams"], atol=1e-6))
        self.assertTrue(torch.allclose(out["ingredient_masses"].sum(-1), out["total_mass"]))
        self.assertEqual(out["auxiliary_total_mass"].shape, (1,))
        self.assertTrue(torch.allclose(out["auxiliary_total_mass"], torch.tensor([200.])))

    def test_all_ablation_configs(self):
        for name, split, auxiliary in (("v4", True, True), ("v4_split_only", True, False), ("v4_aux_only", False, True)):
            config = json.loads((ROOT / f"config/query2label_100_{name}.json").read_text())
            config.update({key: SMALL[key] for key in ("attention_dim", "attention_feedforward", "dropout", "attention_dropout", "mass_scale")})
            m = model(config).eval()
            out = m(torch.rand(1, 3, 64, 64), depth=torch.rand(1, 1, 64, 64))
            self.assertEqual(hasattr(m.heads, "presence_mlp"), split)
            self.assertEqual("auxiliary_total_mass" in out, auxiliary)
            self.assertEqual(hasattr(m.heads, "shared"), not split)

    def test_initialization_uses_train_only(self):
        m = model().eval()
        records = [record("dish_1", "train", {"a": 100}), record("dish_2", "train", {"b": 50}), record("dish_v", "validation", {"a": 10000})]
        m.initialize_from_train(records, vocabulary())
        self.assertAlmostEqual(float(m.heads.auxiliary_gram_scale), 75.)
        with torch.no_grad(): out = m(torch.rand(1,3,64,64), depth=torch.rand(1,1,64,64))
        self.assertAlmostEqual(float(out["auxiliary_total_mass"]), 75., places=4)

    def test_losses_have_finite_gradients(self):
        m = model(); m.set_encoder_trainable(True)
        # Nonzero test weights verify downstream paths; production cold start remains zero-initialized.
        with torch.no_grad(): m.heads.conditional_head.weight.normal_(std=.01)
        out = m(torch.rand(2,3,64,64), depth=torch.rand(2,1,64,64))
        loss, parts = specialized_mass_loss(out, torch.tensor([[1.,0.,0.],[0.,1.,0.]]),
            torch.tensor([[100.,0.,0.],[0.,50.,0.]]), torch.tensor([100.,50.]),
            torch.ones(2,4), torch.ones(3,4), torch.ones(3), SMALL)
        loss.backward()
        self.assertTrue(all(np.isfinite(value) for value in parts.values()))
        self.assertAlmostEqual(parts["total"], sum(SMALL["loss_weights"][key] * parts[key] for key in SMALL["loss_weights"]), places=5)
        for parameter in (m.heads.presence_mlp[1].weight, m.heads.quantity_mlp[1].weight,
                          m.heads.auxiliary_mass_head[-1].bias, m.heads.depth_gate_logits):
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(m.features[0].weight.grad)

    def test_auxiliary_consistency_gradient_and_weighting(self):
        auxiliary = torch.tensor([120.], requires_grad=True)
        deployed = torch.tensor([80.], requires_grad=True)
        out = dict(presence_logits=torch.zeros(1,3), conditional_grams=torch.zeros(1,3),
                   conditional_class_weights=torch.ones(3), ingredient_masses=torch.zeros(1,3),
                   total_mass=deployed, auxiliary_total_mass=auxiliary)
        cfg = {**SMALL, "loss_weights": {key: float(key == "mass_consistency") for key in SMALL["loss_weights"]}}
        loss, _ = specialized_mass_loss(out, torch.ones(1,3), torch.ones(1,3), torch.tensor([100.]),
                                        torch.ones(1,4), torch.ones(3,4), torch.ones(3), cfg)
        loss.backward()
        self.assertAlmostEqual(float(auxiliary.grad), 1/200, places=6)
        self.assertAlmostEqual(float(deployed.grad), -1/200, places=6)

    def test_checkpoint_roundtrip_no_download(self):
        m = model().eval()
        buffer = io.BytesIO()
        torch.save(dict(model_type=MODEL_TYPE_V4, config=SMALL, vocabulary=vocabulary(), model_state=m.state_dict()), buffer)
        buffer.seek(0)
        with patch("torchvision.models.convnext_small", side_effect=fake_backbone) as builder:
            restored = model_from_checkpoint(load_checkpoint(buffer), torch.device("cpu"))
        self.assertIsNone(builder.call_args.kwargs["weights"])
        image, depth = torch.rand(1,3,64,64), torch.rand(1,1,64,64)
        with torch.no_grad(): before, after = m(image, depth=depth), restored(image, depth=depth)
        for key in ("ingredient_masses", "auxiliary_total_mass", "presence_probabilities"):
            self.assertTrue(torch.allclose(before[key], after[key]))

    def test_v3_depth_cache_geometry_reused(self):
        v3 = json.loads((ROOT / "config/query2label_100_v3.json").read_text())
        self.assertEqual(validate_depth_config(CONFIG), validate_depth_config(v3))
        self.assertEqual(image_size(CONFIG), 320)

    def test_saved_v4_arrays_are_diagnosable(self):
        losses, predictions, targets = run_epoch(model().eval(), DataLoader(SyntheticPairs(), batch_size=2),
            torch.ones(3,4), torch.ones(3), SMALL, torch.device("cpu"), description="Synthetic V4 validation")
        self.assertIn("aux_mass", losses)
        self.assertIn("auxiliary_total_mass", auxiliary_mass_metrics(predictions, targets))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            save_predictions(path, predictions, targets, vocabulary())
            with np.load(path, allow_pickle=False) as source:
                report = diagnose_gate(dict(source), {"split":"validation", "threshold":.5}, vocabulary())
            self.assertEqual(report["samples"], 3)
            self.assertIn("auxiliary_total_mass", report)

    def test_disabled_auxiliary_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Disable auxiliary"):
            model({**SMALL, "auxiliary_total_mass":False})

    def test_counterfactuals_and_exact_l1_partition(self):
        result = diagnose_gate(arrays(), {"split":"validation", "threshold":.35})
        baseline, remove_false, oracle = result["scenarios"][:3]
        self.assertAlmostEqual(baseline["ingredient_l1_g_per_plate"], 31.)
        self.assertAlmostEqual(baseline["false_g_per_plate"], 5.)
        self.assertAlmostEqual(baseline["present_l1_g_per_plate"], 26.)
        self.assertAlmostEqual(remove_false["ingredient_l1_g_per_plate"], 26.)
        self.assertAlmostEqual(oracle["ingredient_l1_g_per_plate"], 5.)
        self.assertAlmostEqual(result["global_present"]["conditional_mae_g"], 5.)
        self.assertAlmostEqual(result["global_present"]["final_mae_g"], 26.)

    def test_diagnostic_rejects_test_and_bad_products(self):
        with self.assertRaisesRegex(ValueError, "validation-only"):
            diagnose_gate(arrays(), {"split":"test"})
        bad = arrays(); bad["conditional_grams"][0,0] = 200
        with self.assertRaisesRegex(ValueError, "match p\\*c"):
            diagnose_gate(bad, {"split":"validation"})

    def test_diagnostic_rejects_corruption_and_wrong_metadata(self):
        for key in ("conditional_grams", "probabilities", "masses"):
            bad = arrays(); bad[key][0,0] = np.nan
            with self.assertRaises(ValueError): diagnose_gate(bad, {"split":"validation"})
        with self.assertRaises(ValueError): diagnose_gate(arrays(), {"split":"validation", "samples":3})
        with self.assertRaises(ValueError): diagnose_gate(arrays(), {"split":"validation", "vocabulary_keys":["b","a"]})

    def test_v4_notebook_is_safe_and_old_versions_separate(self):
        nb = json.loads((ROOT / "COLAB_QUERY2LABEL_V4.ipynb").read_text(encoding="utf-8"))
        flags = {}
        code = ""
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            self.assertIsNone(cell["execution_count"]); self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"]); code += source
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.startswith("RUN_"): flags[target.id] = node.value.value
        self.assertTrue(flags and all(value is False for value in flags.values()))
        self.assertIn('V4 = DRIVE / "v4"', code)
        self.assertIn('SOURCE_V3 = DRIVE / "v3"', code)
        self.assertIn("diagnose_mass_gate.py", code)
        self.assertIn("RUN_TRAINING = False", code)
        self.assertIn('VARIANT = "split_only"', code)


if __name__ == "__main__": unittest.main()
