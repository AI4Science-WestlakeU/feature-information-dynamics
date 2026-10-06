"""Synthetic fixtures exercise protocol rejection; they are not paper results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from feature_information_dynamics.bundle import sha256, write_bundle, write_json
from feature_information_dynamics.suite import FEATURES, REPRESENTATIONS, fingerprint, reproduce, validate_suite


def fixture(root, mutation=None, protocol_mutation=None):
    tag = fingerprint({"purpose": "synthetic unit test only"})
    samples = [{"sample_id": str(i), "class_id": i % 2, "rgb_sha256": tag,
                "mask_sha256": tag, "masked_canny_sha256": tag} for i in range(6)]
    protocol = {"schema": "unified-protocol-v2", "status": "frozen", "shared": {
        "features": FEATURES,
        "data": {**{key: tag for key in ("train_samples_sha256", "selection_samples_sha256",
                 "rgb_preprocessing_sha256", "condition_artifacts_sha256", "class_mapping_sha256")},
                 "evaluation_samples_sha256": fingerprint(samples)},
        "conditions": {"canny": "raw_rgb_masked_then_gray_canny_100_200_then_nearest_center_crop_256",
                       "mask_generation_sha256": tag, "absent_condition_semantics_sha256": tag},
        "observation": {"channel": "tX+(1-t)epsilon;gamma=(t/(1-t))^2", "log10_snr": [-2., 0.],
                        "noise_algorithm": "sha256_sample_id_pcg64_float64_to_float32_v1", "noise_seeds": [31]},
        "estimator": {"method": "native_conditioned_models", "initialization": "same_official_pretrained_per_representation",
                      "guidance_scale": 1, "error_reduction": "sum"}},
        "representations": {rep: {"pretrained_sha256": tag, "encoding_spec_sha256": tag,
                                 "native_source_sha256": tag, "conditioning_patch_sha256": tag,
                                 "training_recipe_sha256": fingerprint({"recipe": rep}), "max_updates": 100*(i+1),
                                 "weight_state": "raw" if rep == "pixel" else "ema",
                                 "training_target": "clean_x" if rep == "pixel" else "velocity",
                                 "training_time_strategy": "fixed_snr_existing_experiment" if rep == "pixel" else "native",
                                 "target_dimension": 4} for i, rep in enumerate(REPRESENTATIONS)}}
    if protocol_mutation:
        protocol_mutation(protocol)
    identity = fingerprint(protocol)
    subsets = [[f for i, f in enumerate(FEATURES) if bits & (1 << i)] for bits in range(8)]
    bundles = {}
    for rep in REPRESENTATIONS:
        meta = {"features": FEATURES, "subsets": subsets, "representation": rep,
                "log10_snr": [-2., 0.], "error_reduction": "sum", "target_dimension": 4,
                "protocol_sha256": identity, "shared_protocol": copy.deepcopy(protocol["shared"]),
                "representation_spec": copy.deepcopy(protocol["representations"][rep]),
                "checkpoints": [{"subset": s, "log10_snr": snr, "checkpoint_sha256": tag,
                                 "execution_record_sha256": tag,
                                 **{key: protocol["representations"][rep].get(key) for key in (
                                     "native_source_sha256", "conditioning_patch_sha256", "encoding_spec_sha256",
                                     "pretrained_sha256", "training_recipe_sha256", "weight_state",
                                     "training_target", "training_time_strategy")}, "selected_update": 100}
                                for s in subsets for snr in [-2., 0.]]}
        if rep == "rae" and mutation:
            mutation(meta)
        errors = np.broadcast_to(np.arange(8, dtype=np.float32)[:, None, None, None]+1, (8, 2, 1, 6)).copy()
        write_bundle(root/rep, errors, meta, samples)
        bundles[rep] = {"path": rep, "bundle_sha256": sha256(root/rep/"bundle.json")}
    suite = {"schema": "unified-suite-v1", "protocol": protocol, "protocol_sha256": identity,
             "bundles": bundles, "purpose": "synthetic unit test only"}
    write_json(root/"suite.json", suite)
    return root/"suite.json"


class SuiteTests(unittest.TestCase):
    def test_same_tensor_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = fixture(root)
            reproduce(path, root/"output", bootstrap=20)
            for rep in REPRESENTATIONS:
                data = json.loads((root/"output"/rep/"summary.json").read_text())
                self.assertLess(data["chain"]["finite_integral_nats"][0], 0)
                np.testing.assert_allclose(np.sum(data["chain"]["gap"], axis=0),
                                           np.sum(data["shapley"]["gap"], axis=0))

    def test_rejects_mixed_protocol_before_output(self):
        mutations = {
            "ordinary canny": lambda m: m["shared_protocol"]["conditions"].update(canny="ordinary_canny"),
            "different noise": lambda m: m["shared_protocol"]["observation"].update(noise_seeds=[99]),
            "different grid": lambda m: m.update(log10_snr=[-3., 0.]),
            "different samples": lambda m: m["shared_protocol"]["data"].update(evaluation_samples_sha256="0"*64),
            "different encoder": lambda m: m["representation_spec"].update(encoding_spec_sha256="0"*64),
            "wrong native weight state": lambda m: m["checkpoints"][0].update(weight_state="raw"),
            "different parent": lambda m: m["checkpoints"][0].update(pretrained_sha256="0"*64),
            "missing snr checkpoint": lambda m: m["checkpoints"].pop(),
            "different recipe": lambda m: m["checkpoints"][0].update(training_recipe_sha256="0"*64),
            "different native source": lambda m: m["checkpoints"][0].update(native_source_sha256="0"*64),
            "missing checkpoint patch": lambda m: m["checkpoints"][0].pop("conditioning_patch_sha256"),
            "outside own budget": lambda m: m["checkpoints"][0].update(selected_update=401),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = fixture(root, mutate)
                with self.assertRaises(ValueError):
                    reproduce(path, root/"output", bootstrap=20)
                self.assertFalse((root/"output").exists())

    def test_accepts_native_recipes_budgets_and_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, paths = validate_suite(fixture(Path(tmp)))
            specs = suite["protocol"]["representations"]
            self.assertEqual(len({s["training_recipe_sha256"] for s in specs.values()}), 4)
            self.assertEqual(len({s["max_updates"] for s in specs.values()}), 4)
            self.assertEqual(specs["rae"]["weight_state"], "ema")
            self.assertEqual(set(paths), set(REPRESENTATIONS))

    def test_rejects_missing_training_and_source_identity(self):
        mutations = {
            "missing target": lambda p: p["representations"]["rae"].pop("training_target"),
            "missing time strategy": lambda p: p["representations"]["sdvae"].pop("training_time_strategy"),
            "missing native source": lambda p: p["representations"]["vavae"].pop("native_source_sha256"),
            "missing conditioning patch": lambda p: p["representations"]["rae"].pop("conditioning_patch_sha256"),
            "old protocol": lambda p: p.update(schema="unified-protocol-v1"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = fixture(root, protocol_mutation=mutate)
                with self.assertRaises(ValueError):
                    reproduce(path, root/"output", bootstrap=20)
                self.assertFalse((root/"output").exists())

    def test_accepts_native_pixel_time_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(Path(tmp), protocol_mutation=lambda p:
                           p["representations"]["pixel"].update(training_time_strategy="native"))
            validate_suite(path)

    def test_accepts_native_step_zero_and_shared_snr_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(Path(tmp), mutation=lambda m: [r.update(selected_update=0) for r in m["checkpoints"]],
                           protocol_mutation=lambda p: p["representations"]["rae"].update(max_updates=0))
            validate_suite(path)

    def test_accepts_declared_chained_initialization(self):
        def chained(p):
            p["shared"]["estimator"]["initialization"] = "native_or_chained_actual_parent"
            p["representations"]["rae"]["training_time_strategy"] = "logit_normal_shift_6.9282032"
        with tempfile.TemporaryDirectory() as tmp:
            validate_suite(fixture(Path(tmp), protocol_mutation=chained))

    def test_torch_rank_noise_requires_execution_identity(self):
        def torch_noise(p):
            p["shared"]["observation"].update(
                noise_algorithm="torch_rank_polynomial_seed_v1",
                noise_execution={"world_size": 2, "ranks": [0, 1],
                                 "ordered_rank_samples_sha256": "1"*64,
                                 "rng_draw_schedule_sha256": "2"*64,
                                 "torch_version": "2.2.0+cu121", "device": "cuda", "dtype": "float32"})
        with tempfile.TemporaryDirectory() as tmp:
            validate_suite(fixture(Path(tmp), protocol_mutation=torch_noise))
        def incomplete(p):
            torch_noise(p)
            del p["shared"]["observation"]["noise_execution"]["ordered_rank_samples_sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                validate_suite(fixture(Path(tmp), protocol_mutation=incomplete))

    def test_rejects_missing_rep_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = fixture(root)
            data = json.loads(path.read_text())
            del data["bundles"]["pixel"]
            write_json(path, data)
            with self.assertRaises(ValueError):
                validate_suite(path)
            write_json(path, {"schema": "legacy-aggregate-v1"})
            with self.assertRaises(ValueError):
                validate_suite(path)


if __name__ == "__main__":
    unittest.main()
