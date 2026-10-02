"""Synthetic CPU checks with a mocked backbone; no downloads or dataset training."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ingredient_mass.image_training import run_epoch
from ingredient_mass.model import asymmetric_loss, compute_loss, load_checkpoint
from ingredient_mass.spatial_model import ConvNeXtQuery2Label, MODEL_TYPE, SpatialIngredientHeads, model_from_checkpoint
from infer_image_model import save_intermediates
from train_image_model import capture_rng, restore_rng


CONFIG = {
    "attention_dim": 16, "attention_heads": 4, "attention_layers": 1,
    "attention_feedforward": 32, "attention_dropout": 0.0, "dropout": 0.0,
    "mass_hidden_dim": 8, "presence_loss": "asl", "accumulation_steps": 2,
    "amp": False, "ingredient_mass_scale": 300.0,
    "loss_weights": {"presence": 1, "shares": 5, "mass": 1, "nutrition": 0.25, "ingredient_mass": 1},
}


def fake_backbone(*args, **kwargs):
    model = nn.Module()
    model.features = nn.Sequential(
        nn.Conv2d(3, 768, 1, stride=4), nn.GELU(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(),
        nn.Conv2d(768, 768, 1), nn.GELU(),
    )
    model.classifier = nn.Sequential(nn.Identity(), nn.Flatten(1), nn.Linear(768, 1000))
    return model


class TinyImages(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"image": torch.ones(3, 8, 8) * index / 5, "dish_id": f"dish_{index}",
                "presence": torch.tensor([1.0, 0, 0]), "shares": torch.tensor([1.0, 0, 0]),
                "masses": torch.tensor([100.0, 0, 0]), "total_mass": torch.tensor(100.0),
                "nutrients": torch.tensor([100.0, 10, 20, 30])}


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_asl_matches_bce_without_focusing_or_clipping(self):
        logits = torch.tensor([[-2.0, 0.5, 1.3]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0, 1.0]])
        expected = nn.functional.binary_cross_entropy_with_logits(logits, target)
        actual = asymmetric_loss(logits, target, gamma_pos=0, gamma_neg=0, clip=0)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with torch.no_grad():
            asymmetric_loss(logits, target)
            self.assertFalse(torch.is_grad_enabled())

    def test_gram_loss_updates_proportions_and_total_and_includes_absent_classes(self):
        raw_share = torch.tensor([[1.4, 0.0]], requires_grad=True)
        mass = torch.tensor([200.0], requires_grad=True)
        shares = raw_share.softmax(dim=1)
        outputs = {"presence_logits": torch.zeros(1, 2), "shares": shares, "total_mass": mass}
        weights = {"presence": 0, "shares": 0, "mass": 0, "nutrition": 0, "ingredient_mass": 1}
        loss, parts = compute_loss(
            outputs, torch.tensor([[1.0, 0]]), torch.tensor([[1.0, 0]]), torch.tensor([200.0]),
            torch.zeros(1, 4), torch.zeros(2, 4), torch.ones(2), weights,
            masses_target=torch.tensor([[200.0, 0]]),
        )
        expected = ((shares * 200 - torch.tensor([[200.0, 0]])).abs().sum() / 300).item()
        self.assertAlmostEqual(parts["ingredient_mass"], expected, places=6)
        loss.backward()
        self.assertGreater(float(raw_share.grad.abs().sum()), 0)
        self.assertGreater(float(mass.grad.abs().sum()), 0)

    def test_shapes_attention_conservation_and_partial_freeze(self):
        with patch("torchvision.models.convnext_small", side_effect=fake_backbone):
            model = ConvNeXtQuery2Label(100, CONFIG, pretrained=False)
        model.eval()
        outputs = model(torch.randn(2, 3, 8, 8), return_attention=True)
        self.assertEqual(outputs["presence_logits"].shape, (2, 100))
        self.assertEqual(outputs["shared_features"].shape, (2, 100, 192))
        self.assertEqual(outputs["projected_label_features"].shape, (2, 100, 768))
        self.assertTrue(torch.allclose(outputs["attention"].sum(dim=(-2, -1)), torch.ones(2, 100), atol=1e-6))
        self.assertTrue(torch.allclose(outputs["ingredient_masses"].sum(dim=1), outputs["total_mass"], atol=1e-4))
        model.set_encoder_trainable(True)
        model.train()
        outputs = model(torch.randn(1, 3, 8, 8))
        outputs["presence_logits"].sum().backward()
        self.assertIsNone(model.features[0].weight.grad)
        self.assertGreater(float(model.features[6].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.heads.queries.weight.grad.abs().sum()), 0)
        self.assertFalse(model.features[0].training)
        self.assertTrue(model.features[6].training)
        model.set_encoder_trainable(False)
        self.assertFalse(any(p.requires_grad for p in model.encoder_parameters()))

    def test_checkpoint_roundtrip_requires_no_pretrained_download(self):
        with patch("torchvision.models.convnext_small", side_effect=fake_backbone) as builder:
            model = ConvNeXtQuery2Label(3, CONFIG, pretrained=False).eval()
            payload = {"model_type": MODEL_TYPE, "model_state": model.state_dict(), "config": CONFIG,
                       "vocabulary": {"entries": [{"key": str(i)} for i in range(3)]}}
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            buffer.seek(0)
            loaded = load_checkpoint(buffer)
            restored = model_from_checkpoint(loaded, torch.device("cpu"))
            self.assertIsNone(builder.call_args.kwargs["weights"])
            images = torch.randn(1, 3, 8, 8)
            self.assertTrue(torch.equal(model(images)["shares"], restored(images)["shares"]))

    def test_training_and_evaluation_loops_with_short_accumulation_group(self):
        with patch("torchvision.models.convnext_small", side_effect=fake_backbone):
            model = ConvNeXtQuery2Label(3, CONFIG, pretrained=False)
        model.set_encoder_trainable(True)
        loader = DataLoader(TinyImages(), batch_size=2)
        matrix = torch.ones(3, 4)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        initial = model.heads.presence_head.weight.detach().clone()
        for loss_name in ("asl", "bce"):
            losses, predictions, targets = run_epoch(
                model, loader, matrix, torch.ones(3), {**CONFIG, "presence_loss": loss_name}, torch.device("cpu"),
                optimizer, scaler, "Synthetic step",
            )
            self.assertTrue(np.isfinite(losses["total"]))
            self.assertEqual(predictions["presence"].shape, (5, 3))
            self.assertEqual(targets["masses"].shape, (5, 3))
        self.assertFalse(torch.equal(initial, model.heads.presence_head.weight))
        run_epoch(model, loader, matrix, torch.ones(3), CONFIG, torch.device("cpu"), description="Synthetic validation")

    def test_save_attention_artifacts(self):
        model = SpatialIngredientHeads(3, CONFIG).eval()
        outputs = model(torch.randn(1, 768, 2, 2), return_attention=True)
        vocabulary = {"entries": [{"name": name} for name in ("rice", "pollo", "other")]}
        # tempfile is contained by the test runner's chosen temporary root.
        with tempfile.TemporaryDirectory() as directory:
            summaries = save_intermediates(Path(directory), torch.zeros(1, 3, 224, 224), outputs, vocabulary, 2)
            self.assertEqual(summaries["attention"]["shape"], [1, 3, 2, 2])
            self.assertEqual(len(list((Path(directory) / "intermediates").glob("*_attention.png"))), 2)
            self.assertTrue((Path(directory) / "intermediates/tensors.npz").is_file())

    def test_resumable_rng_state_is_safe_to_load(self):
        loader = DataLoader(TinyImages(), batch_size=2, generator=torch.Generator().manual_seed(42))
        state = capture_rng(loader)
        buffer = io.BytesIO()
        torch.save({"rng_state": state}, buffer)
        expected_torch = torch.rand(3)
        expected_numpy = np.random.rand(3)
        buffer.seek(0)
        loaded = load_checkpoint(buffer)
        restore_rng(loaded["rng_state"], loader)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        self.assertTrue(np.array_equal(np.random.rand(3), expected_numpy))


if __name__ == "__main__":
    unittest.main()
