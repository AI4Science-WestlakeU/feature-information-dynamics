"""Condition dispatch and resume checks using explicitly synthetic test data."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch not installed")
class SharedTests(unittest.TestCase):
    """Check shared U-Net mechanics without interpreting synthetic risks."""

    def test_null_and_clean_endpoint(self) -> None:
        import torch
        from feature_information_dynamics.mnist_calibrated import SharedDenoiser
        torch.set_num_threads(1)
        torch.manual_seed(3)
        model = SharedDenoiser(8).eval()
        torch.nn.init.normal_(model.final.weight, std=.01)
        x = torch.randn(4, 784)
        y = torch.arange(4)
        t = torch.full((4, 1), .2)
        with torch.no_grad():
            torch.testing.assert_close(model(x, t, y, False), model(x, t, y+4, False), rtol=0, atol=0)
            torch.testing.assert_close(model(x, torch.ones_like(t), y, True), x, rtol=0, atol=0)
            self.assertFalse(torch.equal(model(x, t, y, False), model(x, t, y, True)))

    def test_shared_resume(self) -> None:
        import torch
        from feature_information_dynamics import mnist_calibrated as module
        rng = np.random.default_rng(4)
        arrays = {"train-images-idx3-ubyte.gz": rng.normal(size=(200, 784)).astype(np.float32),
                  "train-labels-idx1-ubyte.gz": np.tile(np.arange(10), 20)}
        config = {"seed": 21, "train_samples": 100, "selection_samples": 100,
                  "diagnostic_samples": 100, "steps": 4, "validate_every": 2,
                  "batch_size": 4, "eval_batch_size": 25, "channels": 8,
                  "lr": .0003, "log10_snr": [-2, 0], "loss_weighting": "edm",
                  "ema_decay": .99, "lr_schedule": "cosine", "min_lr": .00001}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root/"config.json"
            config_path.write_text(json.dumps(config))
            with patch.object(module, "load_mnist", return_value=(arrays, {"synthetic": "test"})):
                module.train(config_path, root, root/"full", "cpu")
                original = module.atomic_save

                def interrupt(path: Path, payload: dict) -> None:
                    original(path, payload)
                    if path.name == "last.pt" and payload["step"] == 2:
                        raise InterruptedError("Test interruption after atomic persistence")

                with patch.object(module, "atomic_save", side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        module.train(config_path, root, root/"resumed", "cpu")
                module.train(config_path, root, root/"resumed", "cpu", resume=True)
            a = torch.load(root/"full"/"last.pt", weights_only=True)
            b = torch.load(root/"resumed"/"last.pt", weights_only=True)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(a["rng"], b["rng"], rtol=0, atol=0)
            for name in a["ema"]:
                torch.testing.assert_close(a["ema"][name], b["ema"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
