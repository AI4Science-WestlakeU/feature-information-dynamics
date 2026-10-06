"""Regression against AST fingerprints of the executed historical Pixel source.

Tests avoid importing GPU dependencies. Fingerprints pin scientific functions,
including native training/validation behavior, independently of source formatting.
"""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'train': {'load_pretrained_jit_backbone_film': '62a45cae49021c63618280f62ec41b0a8d31935e3780b2344ca68efd69d56109',
           'load_weights_with_shape_check': 'e78f1b21088b126ae4b1605568b249eeba1325680509bd9bdcc258b5825bece6',
           '_sample_t_lognormal': 'a67b8b2e39d83b87e9424774d61219cebc1ec0fdbe659be27029a0bb61613aad',
           '_sample_t_full_snr_mixture': '869119a86f7ae6d8c7e83bfa7f62486909d45f73b014a5901de903a30ba183d1',
           '_sample_t_fixed': 'de8ba0b12a244f70d1547811c2b738e596c0491758e65c2825f75960d5df65e6',
           '_fixed_quota_drop_mask': '6d1af99315da3923c72605abaa69fafd1e6f82370c031b129f189b94c07cfd24',
           '_compute_velocity_loss': 'f19581bb6a67858b83df40bf4aea3f779d636206ba8810db2066c88783732394',
           'evaluate_deterministic_pixel': '1aec7437864c0b2dcf05abdef993b4c1f4ab4ea69b204a915ead68a49607aba1',
           'update_ema': 'fa3a7f2120b38b8af4d5345b2d1cfb93c37086cd7d2ecc7e62772b698b36af2c',
           'requires_grad': '6ccc86481ae44fe46244d8de471d48158b1abf4f67e407e021f9453f7ceb3cf5',
           'load_config': '64f1f7c69ee6daa74b3c640664efacb19186c74e7e611e20a39bf2165221ec5c',
           'create_logger': '168a77ba18f6f5af83187b2b7bae4c0065b68cc2386e7ff053afae1778156d3e',
           'do_train': '58a286e76373fea22bfc76b6a5c96349a915e738aa112299f81e2c0e492f8fae'},
 'evaluate': {'_load_state_dict_from_ckpt': 'e80ca658b83177812dbfad38bf6de2f3d25a16176043e2a47f79ce106f426857',
              'build_jit_model': '6c49831394453ae72f8203d2a4769cc532f492673b7133c4f45a255011d7d851',
              'build_val_dataset': '3c76599d71af655f230ae9f0cefe095ff4001b76aecd1f48a741de9a92856e08',
              'sample_subset_indices': '868acddb425268cd782084a908bf7e195359e74ba45e74df1939e73a47525768',
              'shard_indices_for_rank': '927a8ffe0f131783226a6e3ba2bd3f0729a5b95425b7554f8e8cec274b5ac5a8',
              'make_mode_kwargs_fn': 'd3d8e021c7767878a658272138de7a4e57b8465384602b5292ab059a8af09f38',
              '_x0_pred_from_jit': '279643b2b00a7ac782614ce1bedf27018373fddc5fe2958b0ecce05e3465d510',
              'compute_mmse_curve': '0ea65fe1a6c3eaa950d0bb8b3b6e7e541d7383abc79739acfd09969cd0087b09',
              'save_mode_outputs': 'b8f0f46d1fb35a84fdfa91fe5a5a74c2aa4dd476cc3cf5511384d9600f796c99',
              'save_combined_outputs': 'd871f7ad1c32dd2c26ffa1bb3e8ff3ba00e95eef8d7b0576f0e676003249d51e'}}


class NativePixelTests(unittest.TestCase):
    def test_historical_function_fingerprints(self):
        for stem, expected in EXPECTED.items():
            tree = ast.parse((ROOT / "src/feature_information_dynamics/pixel" / f"{stem}.py").read_text(encoding="utf-8"))
            functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            for name, digest in expected.items():
                with self.subTest(module=stem, function=name):
                    node = functions[name]
                    if name in ("do_train", "build_jit_model"):
                        node.body = [n for n in node.body if not (isinstance(n, ast.ImportFrom) and n.module == "feature_information_dynamics.pixel.model")]
                    actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                    self.assertEqual(actual, digest)

    def test_no_research_workspace_dependencies(self):
        for path in (ROOT / "src/feature_information_dynamics/pixel").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/panjiashu/", text)
            self.assertNotIn("from train_m06_", text)

    def test_only_trailing_spatial_tokens_receive_conditions(self):
        text = (ROOT / "src/feature_information_dynamics/pixel/model.py").read_text(encoding="utf-8")
        self.assertIn("delta[:, -N_patch:, :]", text)
        self.assertNotIn("delta[:, :N_patch, :]", text)


if __name__ == "__main__":
    unittest.main()
