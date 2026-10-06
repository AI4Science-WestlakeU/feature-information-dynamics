"""Scientific regression checks independent of any training experiment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_information_dynamics.analysis import cumulative_integral, weights
from feature_information_dynamics.bundle import load_bundle, validate, write_bundle
from feature_information_dynamics.noise import stable_noise


class ScientificTests(unittest.TestCase):
    """Check units, interaction allocation, pairing, and artifact integrity."""

    def test_interaction_shapley(self) -> None:
        features = ["class", "mask", "canny"]
        subsets = [[f for i, f in enumerate(features) if b & (1 << i)] for b in range(8)]
        # Independent gains 1, 2, 3 plus a class/mask interaction of 4.
        risk = np.array([20-sum([1, 2, 3][i] for i, f in enumerate(features) if f in s)
                         - (4 if "class" in s and "mask" in s else 0) for s in subsets])
        chain, shapley = weights(features, subsets)
        np.testing.assert_allclose(chain @ risk, [1, 6, 3])
        np.testing.assert_allclose(shapley @ risk, [3, 4, 3])
        np.testing.assert_allclose((shapley @ risk).sum(), risk[0]-risk[-1])

    def test_signed_gap_and_log_units(self) -> None:
        chain, _ = weights(["class"], [[], ["class"]])
        self.assertLess(float((chain @ np.array([1., 2.]))[0]), 0)
        actual = cumulative_integral(np.ones((1, 3)), np.array([-1., 0., 1.]))
        np.testing.assert_allclose(actual, [[0, np.log(10), 2*np.log(10)]])

    def test_noise_is_id_not_order_dependent(self) -> None:
        whole = stable_noise(["test:3", "test:7"], 784, 42)
        np.testing.assert_array_equal(whole[1], stable_noise(["test:7"], 784, 42)[0])
        self.assertFalse(np.array_equal(whole[0], whole[1]))

    def test_bundle_hash_and_axes(self) -> None:
        meta = {"features": ["class"], "subsets": [[], ["class"]],
                "log10_snr": [-1, 1], "error_reduction": "sum", "target_dimension": 784}
        samples = [{"sample_id": "a"}, {"sample_id": "b"}]
        errors = np.ones((2, 2, 1, 2), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_bundle(root, errors, meta, samples)
            np.testing.assert_array_equal(load_bundle(root)[0], errors)
            with (root / "errors.npy").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                load_bundle(root)
        with self.assertRaises(ValueError):
            validate(errors, meta, [{"sample_id": "same"}] * 2)
        with self.assertRaises(ValueError):
            validate(errors, dict(meta, subsets=[[], []]), samples)


if __name__ == "__main__":
    unittest.main()
