import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ingredient_mass.model import compute_loss, mean_training_mass


class MassNormalizationTests(unittest.TestCase):
    def test_mean_uses_training_only(self):
        records = [{"split": "train", "total_mass": 100},
                   {"split": "train", "total_mass": 300},
                   {"split": "validation", "total_mass": 9000},
                   {"split": "test", "total_mass": 10000}]
        self.assertEqual(mean_training_mass(r["total_mass"] for r in records if r["split"] == "train"), 200)

    def test_invalid_masses_rejected(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                mean_training_mass(values)

    def test_fixed_scale_loss_and_gradient(self):
        mass = torch.tensor([130.0, 240.0], requires_grad=True)
        outputs = {"presence_logits": torch.zeros(2, 1), "shares": torch.ones(2, 1), "total_mass": mass}
        loss, parts = compute_loss(
            outputs, torch.ones(2, 1), torch.ones(2, 1), torch.tensor([100.0, 300.0]),
            torch.zeros(2, 4), torch.zeros(1, 4), torch.ones(1),
            {"presence": 0, "shares": 0, "mass": 1, "nutrition": 0}, mass_scale=200,
        )
        self.assertAlmostEqual(parts["mass"], 45 / 200, places=6)
        loss.backward()
        self.assertTrue(torch.allclose(mass.grad, torch.tensor([1 / 400, -1 / 400])))


if __name__ == "__main__":
    unittest.main()
