"""V2 synthetic checks; no pretrained downloads or dataset training."""
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
from ingredient_mass.depth import (FrozenDepthAnything, normalize_relative_depth, open_rgb_crop,
                                   paired_tensors, validate_depth_cache)
from ingredient_mass.image_training import run_epoch
from ingredient_mass.model import load_checkpoint
from ingredient_mass.spatial_model import MODEL_TYPE_V2, create_image_model, model_from_checkpoint, image_transforms
from infer_image_model import save_intermediates


CONFIG = json.loads((ROOT / "config/query2label_100_v2.json").read_text(encoding="utf-8"))
SMALL = {**CONFIG, "attention_dim": 16, "attention_feedforward": 32,
         "attention_dropout": 0.0, "dropout": 0.0, "mass_hidden_dim": 8, "amp": False}


def fake_backbone(*args, **kwargs):
    net = nn.Module()
    net.features = nn.Sequential(
        nn.Conv2d(3, 96, 4, stride=4), nn.Conv2d(96, 96, 1),
        nn.Conv2d(96, 192, 2, stride=2), nn.Conv2d(192, 192, 1),
        nn.Conv2d(192, 384, 2, stride=2), nn.Conv2d(384, 384, 1),
        nn.Conv2d(384, 768, 2, stride=2), nn.Conv2d(768, 768, 1),
    )
    net.classifier = nn.Sequential(nn.GroupNorm(1, 768), nn.Flatten(), nn.Linear(768, 1))
    return net


def make_model(classes=100):
    with patch("torchvision.models.convnext_small", side_effect=fake_backbone):
        return create_image_model(classes, SMALL, pretrained=False)


class PairedSynthetic(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {"image": torch.ones(3, 64, 64) * index / 3,
                "depth": torch.linspace(0, 1, 64).repeat(1, 64, 1),
                "dish_id": f"dish_{index}", "presence": torch.tensor([1., 0, 0]),
                "shares": torch.tensor([1., 0, 0]), "masses": torch.tensor([100., 0, 0]),
                "total_mass": torch.tensor(100.), "nutrients": torch.ones(4)}


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_245_tokens_and_mass_conservation(self):
        model = make_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 224, 224), depth=torch.rand(2, 1, 224, 224), return_attention=True)
        self.assertEqual(outputs["attention_tokens"].shape, (2, 100, 245))
        self.assertEqual(outputs["attention_stage3"].shape, (2, 100, 14, 14))
        self.assertEqual(outputs["attention_stage4"].shape, (2, 100, 7, 7))
        self.assertEqual(outputs["shared_features"].shape, (2, 100, 192))
        self.assertEqual(outputs["projected_label_features"].shape, (2, 100, 768))
        self.assertEqual(len(model.heads.decoder), 2)
        self.assertTrue(torch.allclose(outputs["attention_tokens"].sum(-1), torch.ones(2, 100), atol=1e-6))
        self.assertTrue(torch.allclose(outputs["attention"].sum((-1, -2)), torch.ones(2, 100), atol=1e-6))
        self.assertTrue(torch.allclose(outputs["ingredient_masses"].sum(1), outputs["total_mass"], atol=1e-4))

    def test_freezing_and_gradients_through_both_scales_and_depth(self):
        model = make_model(3)
        self.assertFalse(any(p.requires_grad for p in model.encoder_parameters()))
        model.set_encoder_trainable(True)
        model.train()
        images = torch.randn(1, 3, 64, 64)
        outputs = model(images, depth=torch.rand(1, 1, 64, 64))
        outputs["presence_logits"].square().sum().backward()
        for i in (0, 1):
            self.assertIsNone(model.features[i].weight.grad)
            self.assertFalse(model.features[i].training)
        for i in (2, 3, 4, 5, 6, 7):
            self.assertGreater(float(model.features[i].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.heads.depth_encoder.stage3[0][0].weight.grad.abs().sum()), 0)
        for layer in model.heads.decoder:
            self.assertGreater(float(layer.attention.in_proj_weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.heads.scale_embeddings.weight.grad.abs().sum()), 0)
        model.set_encoder_trainable(False)
        self.assertFalse(any(p.requires_grad for p in model.encoder_parameters()))
        self.assertTrue(model.heads.depth_encoder.training)

    def test_rgb_depth_flip_alignment_and_v1_geometry(self):
        gradient = np.tile(np.arange(224, dtype=np.uint8), (224, 1))
        crop = Image.fromarray(np.stack([gradient] * 3, -1))
        depth = gradient.astype(np.float32) / 255
        with patch("torch.rand", return_value=torch.tensor(0.0)), patch("torchvision.transforms.ColorJitter") as jitter:
            jitter.return_value.side_effect = lambda x: x
            rgb, d = paired_tensors(crop, depth, training=True)
        unnormalized = rgb[0] * 0.229 + 0.485
        self.assertTrue(torch.allclose(unnormalized, d[0], atol=1e-6))
        with tempfile.TemporaryDirectory() as directory:
            photo = Path(directory) / "photo.png"
            original = Image.fromarray(np.random.default_rng(42).integers(0, 255, (301, 400, 3), dtype=np.uint8))
            original.save(photo)
            paired, _ = paired_tensors(open_rgb_crop(photo), np.zeros((224, 224)))
            self.assertTrue(torch.allclose(paired, image_transforms()(original), atol=1e-6))

    def test_depth_normalization_handles_flat_and_invalid_maps(self):
        self.assertTrue(np.array_equal(normalize_relative_depth(np.ones((224, 224))), np.zeros((224, 224))))
        normalized = normalize_relative_depth(np.arange(224*224).reshape(224, 224))
        self.assertEqual(float(normalized.min()), 0)
        self.assertEqual(float(normalized.max()), 1)
        with self.assertRaises(ValueError):
            normalize_relative_depth(np.full((224, 224), np.nan))

    def test_checkpoint_and_attention_exports(self):
        model = make_model(3).eval()
        vocabulary = {"entries": [{"key": str(i), "name": name} for i, name in enumerate(("rice", "beef", "other"))]}
        payload = {"model_type": MODEL_TYPE_V2, "config": SMALL, "vocabulary": vocabulary, "model_state": model.state_dict()}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        with patch("torchvision.models.convnext_small", side_effect=fake_backbone) as builder:
            restored = model_from_checkpoint(load_checkpoint(buffer), torch.device("cpu"))
            self.assertIsNone(builder.call_args.kwargs["weights"])
        images, depth = torch.randn(1, 3, 224, 224), torch.rand(1, 1, 224, 224)
        with torch.no_grad():
            expected = model(images, depth=depth)
            outputs = restored(images, depth=depth, return_attention=True)
        # Exporting weights selects a different attention kernel; tiny round-off is expected.
        self.assertTrue(torch.allclose(expected["shares"], outputs["shares"], atol=1e-6, rtol=1e-5))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            save_intermediates(directory, images, outputs, vocabulary, 2)
            for relative in ("estimated_relative_depth.png", "input_crop.png", "tensor_summary.json"):
                self.assertTrue((directory / "intermediates" / relative).is_file())
            for relative in ("", "attention_stage3", "attention_stage4"):
                self.assertEqual(len(list((directory / "intermediates" / relative).glob("*_attention.png"))), 2)

    def test_evaluation_consumes_paired_depth_and_has_finite_losses(self):
        model = make_model(3).eval()
        losses, predictions, targets = run_epoch(model, DataLoader(PairedSynthetic(), batch_size=2),
                                                 torch.ones(3, 4), torch.ones(3), {**SMALL, "mass_scale": 200},
                                                 torch.device("cpu"), description="Synthetic v2 evaluation")
        self.assertTrue(np.isfinite(losses["total"]))
        self.assertEqual(predictions["presence"].shape, (3, 3))
        self.assertEqual(targets["masses"].shape, (3, 3))

    def test_cache_validates_config_hash_and_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            record = {"dish_id": "dish_1", "image": "rgb.png"}
            np.save(cache / "dish_1.npy", np.zeros((224, 224), dtype=np.float16))
            manifest = {"schema_version": 1, "depth_config": CONFIG["depth"], "checkpoint_sha256": "a" * 64,
                        "entries": {"dish_1": {"image": "rgb.png", "image_sha256": "b" * 64}}}
            (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            validate_depth_cache(cache, [record], CONFIG)
            with self.assertRaises(ValueError):
                validate_depth_cache(cache, [record], {**CONFIG, "depth_checkpoint_sha256": "c" * 64})
            with self.assertRaises(ValueError):
                validate_depth_cache(cache, [record], {**CONFIG, "depth": {**CONFIG["depth"], "encoder": "vitb"}})
            np.save(cache / "dish_1.npy", np.full((224, 224), np.nan, dtype=np.float16))
            with self.assertRaises(ValueError):
                validate_depth_cache(cache, [record], CONFIG)

    def test_frozen_depth_wrapper_device_and_normalization(self):
        fake = nn.Conv2d(3, 1, 1)
        fake.image2tensor = lambda bgr, size: (torch.from_numpy(bgr.copy()).permute(2, 0, 1).float()[None], bgr.shape[:2])
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "depth_anything_v2").mkdir()
            (repo / "depth_anything_v2/dpt.py").touch()
            checkpoint = repo / "weights.pth"
            torch.save(fake.state_dict(), checkpoint)
            module = type(sys)("depth_anything_v2.dpt")
            module.DepthAnythingV2 = lambda **kwargs: fake
            with patch.dict(sys.modules, {"depth_anything_v2.dpt": module}), patch("subprocess.run") as run:
                run.return_value.stdout = CONFIG["depth"]["repo_revision"]
                predictor = FrozenDepthAnything(CONFIG, repo, checkpoint, "cpu")
                self.assertFalse(any(p.requires_grad for p in predictor.model.parameters()))
                # Official output is BxHxW; adapt this small mock's output to that contract.
                original = fake.forward
                fake.forward = lambda x: original(x).squeeze(1)
                depth = predictor.predict(Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8)))
                self.assertEqual(depth.shape, (224, 224))
                self.assertTrue(np.isfinite(depth).all())


if __name__ == "__main__":
    unittest.main()
