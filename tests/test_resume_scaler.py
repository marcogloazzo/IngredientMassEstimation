"""Check CPU-to-CUDA AMP resume without allocating CUDA tensors or training."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_image_model import restore_scaler


class ResumeScalerTests(unittest.TestCase):
    def test_cpu_state_keeps_fresh_cuda_scaler(self):
        cpu_state = torch.amp.GradScaler("cuda", enabled=False).state_dict()
        self.assertEqual(cpu_state, {})
        with patch("torch.cuda.is_available", return_value=True):
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            restore_scaler(scaler, cpu_state)
            self.assertEqual(scaler.get_scale(), 1024)

    def test_cuda_state_is_restored(self):
        with patch("torch.cuda.is_available", return_value=True):
            source = torch.amp.GradScaler("cuda", init_scale=2048)
            target = torch.amp.GradScaler("cuda", init_scale=1024)
            restore_scaler(target, source.state_dict())
            self.assertEqual(target.get_scale(), 2048)

    def test_cuda_state_can_be_loaded_into_disabled_cpu_scaler(self):
        with patch("torch.cuda.is_available", return_value=True):
            state = torch.amp.GradScaler("cuda").state_dict()
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        restore_scaler(scaler, state)
        self.assertFalse(scaler.is_enabled())


if __name__ == "__main__":
    unittest.main()
