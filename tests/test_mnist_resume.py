"""Small synthetic mechanics tests; these are not scientific MNIST evidence."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch not installed")
class ResumeTests(unittest.TestCase):
    """Interrupt after an atomic snapshot and compare to uninterrupted training."""

    def test_resume_preserves_updates(self) -> None:
        import torch
        from feature_information_dynamics import mnist

        torch.set_num_threads(1)
        config = {"seed": 21, "hidden": 16, "lr": .001, "steps": 8,
                  "validate_every": 4, "batch_size": 10, "eval_batch_size": 7,
                  "log10_snr": [-2, 0]}
        rng = np.random.default_rng(4)
        images = rng.normal(size=(20, 784)).astype(np.float32)
        labels = np.tile(np.arange(10), 2)
        selection = (images[:10], labels[:10], [f"synthetic:{i}" for i in range(10)])
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temporary:
            full, resumed = Path(temporary)/"full", Path(temporary)/"resumed"
            full.mkdir()
            resumed.mkdir()
            mnist.train_arm(config, full, True, (images, labels), selection, device, False)
            original_save = mnist.atomic_save

            def interrupt_after_save(path: Path, payload: dict) -> None:
                original_save(path, payload)
                if path.name.endswith("last.pt") and payload["step"] == 4:
                    raise InterruptedError("Injected interruption after persisted snapshot")

            with patch.object(mnist, "atomic_save", side_effect=interrupt_after_save):
                with self.assertRaises(InterruptedError):
                    mnist.train_arm(config, resumed, True, (images, labels), selection, device, False)
            mnist.train_arm(config, resumed, True, (images, labels), selection, device, True)
            a = torch.load(full/"class_last.pt", weights_only=True)
            b = torch.load(resumed/"class_last.pt", weights_only=True)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(a["stream"], b["stream"], rtol=0, atol=0)
            self.assertEqual(a["history"], b["history"])


if __name__ == "__main__":
    unittest.main()
