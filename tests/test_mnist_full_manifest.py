"""Check that full-test acceptance uses every ID and the actual label entropy."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch not installed")
class FullManifestTests(unittest.TestCase):
    def test_all_ids_and_nonuniform_entropy(self):
        from feature_information_dynamics import mnist_confirmation as module
        labels = np.array([0, 0, 0, 1], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "manifest.json"
            with patch.object(module, "load_mnist", return_value=(
                    {"t10k-labels-idx1-ubyte.gz": labels}, {"synthetic": "test"})):
                module.prepare_full(Path(tmp), target, .2)
            manifest = json.loads(target.read_text())
            self.assertEqual([s["sample_id"] for s in manifest["samples"]],
                             ["test:0", "test:1", "test:2", "test:3"])
            self.assertAlmostEqual(manifest["reference_nats"],
                                   -.75*np.log(.75)-.25*np.log(.25))
            self.assertEqual(manifest["relative_error_limit"], .2)


if __name__ == "__main__":
    unittest.main()
