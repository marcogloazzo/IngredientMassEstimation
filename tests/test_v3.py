"""V3 verification with synthetic tensors only; no dataset training/downloads."""
import ast
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ingredient_mass.conditional_model import conditional_mass_loss, training_statistics
from ingredient_mass.depth import rgb_crop, paired_tensors, normalize_relative_depth, validate_depth_cache, file_sha256
from ingredient_mass.image_training import run_epoch, save_predictions, load_prepared, prepared_fingerprint
from ingredient_mass.model import load_checkpoint
from ingredient_mass.spatial_model import MODEL_TYPE_V2, MODEL_TYPE_V3, create_image_model, model_from_checkpoint
from train_image_model import make_scheduler
from archive_experiment import archive_experiment
from infer_image_model import save_intermediates
from compare_image_experiments import comparison

CONFIG = json.loads((ROOT / "config/query2label_100_v3.json").read_text(encoding="utf-8"))
SMALL = {**CONFIG, "attention_dim": 16, "attention_feedforward": 32,
         "attention_dropout": 0., "dropout": 0., "amp": False, "mass_scale": 200.}


def fake_backbone(*args, **kwargs):
    net = nn.Module()
    net.features = nn.Sequential(nn.Conv2d(3, 96, 4, stride=4), nn.Conv2d(96, 96, 1),
                                 nn.Conv2d(96, 192, 2, stride=2), nn.Conv2d(192, 192, 1),
                                 nn.Conv2d(192, 384, 2, stride=2), nn.Conv2d(384, 384, 1),
                                 nn.Conv2d(384, 768, 2, stride=2), nn.Conv2d(768, 768, 1))
    net.classifier = nn.Sequential(nn.GroupNorm(1, 768), nn.Flatten(), nn.Linear(768, 1))
    return net


def model(classes=3):
    with patch("torchvision.models.convnext_small", side_effect=fake_backbone):
        return create_image_model(classes, SMALL, pretrained=False)


def vocabulary():
    return {"nutrient_order": ["calories", "fat", "carbohydrates", "protein"],
            "entries": [{"key": key, "name": key, "nutrients_per_gram": dict(calories=2., fat=.1, carbohydrates=.2, protein=.3)}
                        for key in ("a", "b", "__other__")]}


def record(identifier, split, grams):
    return {"dish_id": identifier, "split": split, "image": f"{identifier}.png", "scan_group_id": identifier,
            "total_mass": sum(grams.values()), "nutrients": dict(calories=200., fat=10., carbohydrates=20., protein=30.),
            "ingredients": [{"ingredient_id": key, "name": key, "grams": value, "calories": 2*value,
                             "fat": .1*value, "carbohydrates": .2*value, "protein": .3*value} for key, value in grams.items()]}


class SyntheticPairs(Dataset):
    def __len__(self): return 3
    def __getitem__(self, index):
        return {"image": torch.rand(3, 64, 64), "depth": torch.rand(1, 64, 64), "dish_id": f"dish_{index}",
                "presence": torch.tensor([1., 0., 0.]), "shares": torch.tensor([1., 0., 0.]),
                "masses": torch.tensor([100., 0., 0.]), "total_mass": torch.tensor(100.), "nutrients": torch.ones(4)}


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_500_tokens_residual_gates_and_mass_conservation(self):
        m = model(100).eval()
        with torch.no_grad(): out = m(torch.rand(1, 3, 320, 320), depth=torch.rand(1, 1, 320, 320), return_attention=True)
        self.assertEqual(out["attention_tokens"].shape, (1, 100, 500))
        self.assertEqual(out["attention_stage3"].shape, (1, 100, 20, 20))
        self.assertEqual(out["attention_stage4"].shape, (1, 100, 10, 10))
        self.assertEqual(out["conditional_grams"].shape, (1, 100))
        self.assertTrue(torch.allclose(out["ingredient_masses"].sum(-1), out["total_mass"]))
        self.assertTrue(torch.allclose(out["shares"].sum(-1), torch.ones(1)))
        self.assertTrue(torch.allclose(out["attention"].sum((-1, -2)), torch.ones(1, 100), atol=1e-6))
        self.assertTrue(torch.allclose(out["depth_fusion_gates"], torch.full((2,), .1)))
        self.assertFalse(any("mass_head" in key or "amount_head" in key for key in m.state_dict()))

    def test_conditional_supervision_survives_zero_presence(self):
        conditional = torch.tensor([[30., 40., 50.]], requires_grad=True)
        logits = torch.full((1, 3), -100., requires_grad=True)
        outputs = {"presence_logits": logits, "conditional_grams": conditional,
                   "conditional_class_weights": torch.ones(3), "ingredient_masses": logits.sigmoid()*conditional,
                   "total_mass": (logits.sigmoid()*conditional).sum(-1)}
        config = {**SMALL, "loss_weights": {key: float(key == "conditional") for key in SMALL["loss_weights"]}}
        loss, _ = conditional_mass_loss(outputs, torch.tensor([[1., 0., 0.]]), torch.tensor([[100., 0., 0.]]),
                                         torch.tensor([100.]), torch.zeros(1, 4), torch.zeros(3, 4), torch.ones(3), config)
        loss.backward()
        self.assertGreater(float(conditional.grad[0, 0].abs()), 0)
        self.assertEqual(float(conditional.grad[0, 1:].abs().sum()), 0)

    def test_training_initialization_and_statistics_exclude_validation(self):
        records = [record("dish_1", "train", {"a": 100.}), record("dish_2", "train", {"b": 50.}),
                   record("dish_3", "validation", {"a": 10000.})]
        stats = training_statistics(records, vocabulary())
        np.testing.assert_allclose(stats["conditional_mean_grams"], [100., 50., 0.])
        np.testing.assert_allclose(stats["prevalence"], [.5, .5, 0.])
        m = model(); m.initialize_from_train(records, vocabulary())
        expected = torch.nn.functional.softplus(m.heads.conditional_head.bias) * 300
        self.assertTrue(torch.allclose(expected[:2], torch.tensor([100., 50.]), atol=1e-5))

    def test_finite_losses_and_gradients_both_scales_and_heads(self):
        m = model(); m.set_encoder_trainable(True)
        out = m(torch.rand(2, 3, 64, 64), depth=torch.rand(2, 1, 64, 64))
        loss, parts = conditional_mass_loss(out, torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
                                           torch.tensor([[100., 0., 0.], [0., 50., 0.]]), torch.tensor([100., 50.]),
                                           torch.ones(2, 4), torch.ones(3, 4), torch.ones(3), SMALL)
        loss.backward()
        self.assertTrue(all(np.isfinite(value) for value in parts.values()))
        self.assertGreater(float(m.heads.conditional_head.bias.grad.abs().sum()), 0)
        self.assertGreater(float(m.heads.presence_head.bias.grad.abs().sum()), 0)
        self.assertGreater(float(m.heads.depth_gate_logits.grad.abs().sum()), 0)
        self.assertIsNone(m.features[0].weight.grad)
        for index in (2, 4, 6): self.assertGreater(float(m.features[index].weight.grad.abs().sum()), 0)

    def test_letterbox_preserves_edges_and_depth_padding(self):
        image = Image.new("RGB", (600, 200), (0, 0, 0))
        image.paste((255, 0, 0), (0, 0, 60, 200)); image.paste((0, 255, 0), (540, 0, 600, 200))
        crop = rgb_crop(image, CONFIG)
        self.assertEqual(crop.size, (320, 320))
        rgb = np.asarray(crop)
        self.assertGreater(int(rgb[:, 0, 0].max()), 240)
        self.assertGreater(int(rgb[:, -1, 1].max()), 240)
        depth = normalize_relative_depth(np.arange(320*320).reshape(320, 320), 320, crop.info["valid_box"])
        self.assertTrue(np.all(depth[:100] == 0))
        self.assertTrue(np.isfinite(depth).all())
        img, d = paired_tensors(crop, depth, training=False, config=CONFIG)
        self.assertEqual(img.shape, (3, 320, 320)); self.assertEqual(d.shape, (1, 320, 320))

    def test_shared_rotation_flip_and_no_crop(self):
        gradient = np.tile(np.arange(320) / 319, (320, 1))
        crop = Image.fromarray(np.repeat((gradient*255).astype(np.uint8)[:, :, None], 3, -1))
        config = {**CONFIG, "augmentation": {"rotation_degrees": 0}}
        with patch("torch.rand", return_value=torch.tensor(0.)), patch("torchvision.transforms.ColorJitter") as jitter:
            jitter.return_value.side_effect = lambda x: x
            rgb, d = paired_tensors(crop, gradient.astype(np.float32), True, config)
        restored = rgb[0] * .229 + .485
        self.assertTrue(torch.allclose(restored, d[0], atol=1/255 + 1e-6))

    def test_nonzero_rotation_uses_identical_rgb_depth_grid(self):
        from torchvision.transforms import functional as TF, InterpolationMode
        gradient = np.tile(np.arange(320) / 319, (320, 1))
        crop = Image.fromarray(np.repeat((gradient*255).astype(np.uint8)[:, :, None], 3, -1))
        with patch("torch.rand", return_value=torch.tensor(.75)), patch("torchvision.transforms.ColorJitter") as jitter:
            jitter.return_value.side_effect = lambda x: x
            rgb, d = paired_tensors(crop, gradient.astype(np.float32), True, CONFIG)
        valid = TF.rotate(torch.ones(1,320,320), 7.5, InterpolationMode.BILINEAR, expand=True, fill=0)
        valid = TF.resize(valid, [320,320], InterpolationMode.BILINEAR, antialias=True)[0] > .999
        restored = rgb[0] * .229 + .485
        self.assertTrue(torch.allclose(restored[valid], d[0][valid], atol=1/255 + 1e-6))

    def test_checkpoint_roundtrip_and_intermediate_exports(self):
        m = model().eval(); v = vocabulary()
        buffer = io.BytesIO(); torch.save({"model_type": MODEL_TYPE_V3, "config": SMALL, "vocabulary": v, "model_state": m.state_dict()}, buffer); buffer.seek(0)
        with patch("torchvision.models.convnext_small", side_effect=fake_backbone) as builder:
            loaded = model_from_checkpoint(load_checkpoint(buffer), torch.device("cpu"))
        self.assertIsNone(builder.call_args.kwargs["weights"])
        image, depth = torch.rand(1, 3, 320, 320), torch.rand(1, 1, 320, 320)
        with torch.no_grad():
            expected = m(image, depth=depth); out = loaded(image, depth=depth, return_attention=True)
        self.assertTrue(torch.allclose(expected["ingredient_masses"], out["ingredient_masses"], atol=1e-4))
        with tempfile.TemporaryDirectory() as directory:
            save_intermediates(Path(directory), image, out, v, 2)
            self.assertTrue((Path(directory) / "intermediates/tensors.npz").is_file())

    def test_v3_cache_rejects_v2_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory); r = {"dish_id": "dish_1", "image": "a.png"}
            np.save(cache / "dish_1.npy", np.zeros((320, 320), dtype=np.float16))
            manifest = {"schema_version": 1, "depth_config": CONFIG["depth"], "checkpoint_sha256": "a"*64,
                        "entries": {"dish_1": {"image": "a.png", "map_sha256": file_sha256(cache / "dish_1.npy")}}}
            (cache / "manifest.json").write_text(json.dumps(manifest))
            validate_depth_cache(cache, [r], CONFIG)
            old = json.loads((ROOT / "config/query2label_100_v2.json").read_text())
            with self.assertRaises(ValueError): validate_depth_cache(cache, [r], old)
            np.save(cache / "dish_1.npy", np.zeros((224, 224), dtype=np.float16))
            with self.assertRaises(ValueError): validate_depth_cache(cache, [r], CONFIG)

    def test_evaluation_collects_conditional_grams(self):
        losses, predictions, targets = run_epoch(model().eval(), DataLoader(SyntheticPairs(), batch_size=2),
                                                 torch.ones(3, 4), torch.ones(3), SMALL, torch.device("cpu"), description="Synthetic V3 validation")
        self.assertTrue(np.isfinite(losses["total"]))
        self.assertEqual(predictions["conditional_grams"].shape, (3, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            save_predictions(path, predictions, targets, vocabulary())
            with np.load(path) as source: self.assertIn("conditional_grams", source.files)

    def test_scheduler_resume_same_trajectory(self):
        parameter = nn.Parameter(torch.ones(1)); optimizer = torch.optim.AdamW([parameter], lr=.001)
        scheduler = make_scheduler(optimizer, CONFIG)
        for _ in range(7): optimizer.step(); scheduler.step()
        optimizer_state, scheduler_state = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        optimizer.step(); scheduler.step(); expected = optimizer.param_groups[0]["lr"]
        restored_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=.001)
        restored_scheduler = make_scheduler(restored_optimizer, CONFIG)
        restored_optimizer.load_state_dict(optimizer_state); restored_scheduler.load_state_dict(scheduler_state)
        restored_optimizer.step(); restored_scheduler.step()
        self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], expected)
        self.assertIsNone(make_scheduler(restored_optimizer, {}))

    def test_data_integrity_and_group_leakage_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prepared = root / "prepared"; prepared.mkdir()
            records = [record("dish_1", "train", {"a": 100}), record("dish_2", "validation", {"b": 50}), record("dish_3", "test", {"a": 50})]
            for r in records: Image.new("RGB", (10, 10)).save(root / r["image"])
            (prepared / "vocabulary.json").write_text(json.dumps(vocabulary()))
            (prepared / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            (prepared / "summary.json").write_text(json.dumps({"mode": "provisional", "official_test_preserved": True}))
            cfg = {**CONFIG, "top_k": 3}
            load_prepared(prepared, root, cfg)
            cfg["prepared_fingerprint"] = prepared_fingerprint(prepared)
            records[1]["total_mass"] += 1
            (prepared / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError, "changed since"):
                load_prepared(prepared, root, cfg)
            records[1]["total_mass"] -= 1
            records[1]["scan_group_id"] = records[0]["scan_group_id"]
            (prepared / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError, "Group leakage"): load_prepared(prepared, root, cfg)

    def test_snapshot_preserves_source_and_rejects_overwrite(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "v2"; (source / "runs").mkdir(parents=True)
            path = source / "runs/best.pt"; path.write_bytes(b"original-v2-results")
            output = root / "snapshot.zip"
            archive_experiment(source, output)
            self.assertEqual(path.read_bytes(), b"original-v2-results")
            with zipfile.ZipFile(output) as archive: self.assertEqual(archive.read("runs/best.pt"), path.read_bytes())
            with self.assertRaises(FileExistsError): archive_experiment(source, output)
            with self.assertRaises(ValueError): archive_experiment(source, source / "inside.zip")

    def test_comparison_marks_historical_and_rejects_different_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); left = root / "left"; right = root / "right"
            left.mkdir(); right.mkdir()
            source = {"ingredient_recognition": {"micro_f1": .5, "macro_f1": .4},
                      "end_to_end_lightweight_mass": {"ingredient_mass_l1_per_plate_grams": 100.,
                          "total_mass_mae_grams": 20., "false_ingredient_mass_per_plate_grams": 10.,
                          "nutrients": {key: {"mae": 1.} for key in ("calories", "fat", "carbohydrates", "protein")}}}
            for folder in (left, right): (folder / "metrics.json").write_text(json.dumps(source))
            self.assertFalse(comparison(left, right)["controlled_data_comparison"])
            details = {"split": "test", "prepared_fingerprint": {"records.jsonl": "same", "vocabulary.json": "same"}}
            for folder in (left, right): (folder / "evaluation_details.json").write_text(json.dumps(details))
            self.assertTrue(comparison(left, right)["controlled_data_comparison"])
            details["split"] = "validation"
            (right / "evaluation_details.json").write_text(json.dumps(details))
            with self.assertRaises(ValueError): comparison(left, right)

    def test_new_notebook_is_safe_and_v3_geometry(self):
        notebook = json.loads((ROOT / "COLAB_QUERY2LABEL_V3.ipynb").read_text(encoding="utf-8"))
        flags = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"]); self.assertEqual(cell["outputs"], [])
                tree = ast.parse("".join(cell["source"]))
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.startswith("RUN_"): flags[target.id] = node.value.value
        self.assertTrue(flags and all(value is False for value in flags.values()))
        diagnostics = next("".join(c["source"]) for c in notebook["cells"] if "def run_v3_diagnostics" in "".join(c["source"]))
        self.assertIn("attention.shape[1] != 500", diagnostics)
        self.assertIn("MAE condicional presente g", diagnostics)
        self.assertNotIn("attention[i, :196]", diagnostics)


if __name__ == "__main__": unittest.main()
