"""Contract checks for native source isolation and preserved scientific loops."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
from feature_information_dynamics import workflows as launch

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = json.loads((ROOT / 'docs/evidence/native_migration/layout.json').read_text(encoding='utf-8'))
REVERSE_MODULES = {new: old for old, new in LAYOUT['module_map'].items()}

def stable_ast(value):
    if isinstance(value, ast.AST):
        return [type(value).__name__, [(key, stable_ast(item)) for key, item in ast.iter_fields(value) if key != 'type_params']]
    if isinstance(value, list):
        return [stable_ast(item) for item in value]
    return value

class NativeLatentTests(unittest.TestCase):
    def test_original_scientific_functions_are_preserved(self):
        manifest = json.loads((ROOT / 'docs/evidence/native_migration/selected_sources.json').read_text(encoding='utf-8'))
        checked = 0
        for module, record in manifest.items():
            expected = record.get('preserved_function_ast_sha256', {})
            tree = ast.parse((ROOT / module).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    node.module = REVERSE_MODULES.get(node.module, node.module)
            defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            self.assertTrue(set(expected).issubset(defined), (module, set(expected) - defined))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in expected:
                    digest = hashlib.sha256(json.dumps(stable_ast(node), sort_keys=True, default=repr).encode()).hexdigest()
                    self.assertEqual(digest, expected[node.name], (module, node.name))
                    checked += 1
        self.assertGreaterEqual(checked, 12)

    def test_sdvae_needs_only_sit_source(self):
        with tempfile.TemporaryDirectory() as directory:
            sit = Path(directory)
            (sit / 'models.py').write_text('')
            paths = launch.source_paths('sdvae', 'train', str(sit))
            self.assertEqual(paths, [str(sit.resolve())])
            command = launch.build_command('sdvae', 'train', ['--config', 'example.yaml'], str(sit))
            self.assertEqual(command[-3:], ['--', '--config', 'example.yaml'])

    def test_only_vavae_supports_cache(self):
        for representation in ('rae', 'sdvae'):
            with self.assertRaisesRegex(ValueError, 'online'):
                launch.source_paths(representation, 'cache', '.')
        self.assertEqual({rep for rep, action in launch.MODULES if action == 'cache'}, {'vavae'})

    def test_vavae_checkpoint_argument_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'selected.pt'
            checkpoint.write_bytes(b'example')
            (root / 'config.yaml').write_text('placeholder architecture config')
            tokenizer = ModuleType('tokenizer')
            tokenizer.__file__ = str(root / '__init__.py')
            va_module = ModuleType('tokenizer.vavae')
            va_module.VA_VAE = lambda filename: Path(filename).read_text()
            omega = ModuleType('omegaconf')
            omega.OmegaConf = SimpleNamespace(
                load=lambda filename: SimpleNamespace(ckpt_path='wrong-checkpoint.pt'),
                save=lambda config, filename: Path(filename).write_text(config.ckpt_path))
            tree = ast.parse((ROOT / 'src/feature_information_dynamics/vavae/extract_latents.py').read_text(encoding='utf-8'))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_vavae')
            code = 'from __future__ import annotations\n' + ast.unparse(function)
            namespace = {'Path': Path}
            with patch.dict('sys.modules', {'tokenizer': tokenizer, 'tokenizer.vavae': va_module, 'omegaconf': omega}):
                exec(code, namespace)
                selected = namespace['build_vavae'](str(checkpoint), None)
            self.assertEqual(selected, str(checkpoint.resolve()))

    def test_sdvae_is_online_and_vavae_uses_sampled_cache(self):
        online = (ROOT / 'src/feature_information_dynamics/sdvae/train.py').read_text(encoding='utf-8')
        self.assertIn('latent_dist.sample().mul_(SDVAE_MULTIPLIER)', online)
        self.assertIn('SDVAE_MULTIPLIER: float = 0.18215', online)
        cache = (ROOT / 'src/feature_information_dynamics/vavae/extract_latents.py').read_text(encoding='utf-8')
        self.assertIn('vae.encode_images(images)', cache)
        self.assertIn('latents_flip', cache)
        self.assertIn('filenames', cache)

if __name__ == '__main__':
    unittest.main()
