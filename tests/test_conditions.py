"""Condition preparation preserves stored empty masks; missing data is an error."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from feature_information_dynamics.conditions import prepare
from feature_information_dynamics.noise import rank_noise_seed


class PreparationTests(unittest.TestCase):
    def test_stored_empty_mask_remains_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.full((256, 256, 3), 255, np.uint8)).save(root/"rgb.png")
            np.savez(root/"mask.npz", mask=np.zeros((256, 256), np.uint8))
            manifest = root/"input.json"
            manifest.write_text(json.dumps([{"sample_id": "a", "class_id": 1,
                "rgb_path": str(root/"rgb.png"), "mask_path": str(root/"mask.npz")}]))
            prepare(manifest, root/"out")
            with np.load(root/"out/sample_000000.npz") as result:
                self.assertFalse(result["mask"].any())
                self.assertFalse(result["masked_canny"].any())
            rows = json.loads((root/"out/samples.json").read_text())
            self.assertTrue(rows[0]["mask_empty"])

    def test_missing_mask_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(root/"rgb.png")
            manifest = root/"input.json"
            manifest.write_text(json.dumps([{"sample_id": "a", "class_id": 1,
                "rgb_path": str(root/"rgb.png"), "mask_path": str(root/"missing.npz")}]))
            with self.assertRaises(FileNotFoundError):
                prepare(manifest, root/"out")

    def test_old_rank_seed_rules(self):
        self.assertEqual(rank_noise_seed(42, 3, "torch_rank_seed_plus_100003_v1"), 300051)
        self.assertEqual(rank_noise_seed(42, 3, "torch_rank_polynomial_seed_v1"),
                         (42*2654435761+3*40503) & ((1 << 63)-1))
        with self.assertRaises(ValueError):
            rank_noise_seed(42, -1, "torch_rank_polynomial_seed_v1")
