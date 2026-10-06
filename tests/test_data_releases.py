"""Small release fixtures: compressed transport, atomic failure, paths and configs."""
import contextlib
import functools
import gzip
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from feature_information_dynamics.data.releases import download, verify, configure

class ReleasesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source = self.root / 'source'; self.source.mkdir()
        self.output = self.root / 'installed'
        self.manifest = self.source / 'manifest.json'
        self.rows = []
        for name, value, component, compressed in [('conditions/mask_shards/a.bin', b'00000' * 100, 'common', True), ('conditions/canny_shards/a.bin', b'11111', 'common', False), ('paired/train.json', b'{"filenames": []}', 'common', False), ('paired/val.json', b'{"filenames": []}', 'common', False), ('latents/a.bin', b'latent', 'vavae', True)]:
            source_path = name + ('.gz' if compressed else '')
            payload = gzip.compress(value) if compressed else value
            path = self.source / source_path; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
            row = {'path': name, 'source_path': source_path, 'size_bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest(), 'component': component}
            if compressed:
                row.update(encoding='gzip', download_size_bytes=len(payload), download_sha256=hashlib.sha256(payload).hexdigest())
            self.rows.append(row)
        self.data = {'schema_version': 1, 'release_id': 'fixture', 'files': self.rows, 'resources': {'mask_shard_dir': 'conditions/mask_shards', 'canny_shard_dir': 'conditions/canny_shards', 'paired_train_whitelist': 'paired/train.json', 'paired_val_whitelist': 'paired/val.json', 'vavae_latents': 'latents'}}
        self.save()
    def save(self):
        self.manifest.write_text(json.dumps(self.data))
    def tearDown(self):
        self.temp.cleanup()
    def test_local_gzip_optional_latents_resume_and_corruption(self):
        self.assertEqual(download(self.manifest, self.output)['installed_files'], 4)
        self.assertFalse((self.output / 'latents').exists())
        self.assertEqual(download(self.manifest, self.output)['reused_files'], 4)
        self.assertEqual(verify(self.manifest, self.output)['verified_files'], 4)
        with self.assertRaisesRegex(ValueError, 'Missing or corrupted'):
            verify(self.manifest, self.output, 'vavae')
        target = self.output / self.rows[0]['path']; target.write_bytes(b'corrupt')
        payload = self.source / self.rows[0]['source_path']; payload.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'mismatch|exceeds'):
            download(self.manifest, self.output)
        self.assertEqual(target.read_bytes(), b'corrupt')
        self.assertEqual(list(target.parent.glob('.release-*')), [])
    def test_decompressed_hash_failure_never_installs(self):
        self.rows[0]['sha256'] = '0' * 64; self.save()
        with self.assertRaisesRegex(ValueError, 'Decompressed'):
            download(self.manifest, self.output)
        self.assertFalse((self.output / self.rows[0]['path']).exists())
    def test_reject_paths_and_escaping_symlinks(self):
        for path in ['../escape', '/absolute', 'C:/drive', 'folder\\escape', 'x/../y', 'x//y', 'release.json', 'RELEASE.JSON']:
            with self.subTest(path=path):
                self.rows[0]['path'] = path; self.save()
                with self.assertRaisesRegex(ValueError, 'Unsafe|reserved'):
                    download(self.manifest, self.output)
    def test_semantic_alignment_declared_sentinels_and_labels(self):
        import numpy as np
        from safetensors.numpy import save_file
        names = ['n00000001/a.JPEG', 'n00000002/b.JPEG']
        self.rows.clear()
        for folder, key, flagkey, flags in [('mask_shards', 'masks', 'mask_valid', [1, 0]), ('canny_shards', 'cannys', 'canny_valid', [1, 1])]:
            path = self.source / folder / 'rank0.safetensors'; path.parent.mkdir()
            save_file({key: np.zeros((2, 8, 8), dtype=np.uint8), flagkey: np.asarray(flags, dtype=np.uint8), 'labels': np.asarray([0, 1], dtype=np.int64)}, str(path))
            path.with_suffix('.json').write_text(json.dumps({'filenames': names}))
            for source in [path, path.with_suffix('.json')]:
                self.rows.append({'path': source.relative_to(self.source).as_posix(), 'size_bytes': source.stat().st_size, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        for file, payload in [('paired/train.json', {'filenames': names[:1]}), ('paired/val.json', {'filenames': names[1:]})]:
            path = self.source / file; path.write_text(json.dumps(payload))
            self.rows.append({'path': file, 'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.data['resources'].update(mask_shard_dir='mask_shards', canny_shard_dir='canny_shards')
        self.data['validation'] = {'train_filenames': 1, 'val_filenames': 1, 'paired_filenames': 2, 'mask_invalid_paired': 1, 'canny_invalid_paired': 0}
        self.save(); download(self.manifest, self.output)
        self.assertEqual(verify(self.manifest, self.output)['validation']['mask_invalid_paired'], 1)
        self.data['validation']['mask_invalid_paired'] = 0; self.save()
        with self.assertRaisesRegex(ValueError, 'validation count differs'):
            verify(self.manifest, self.output)
        self.data['validation']['mask_invalid_paired'] = 1
        path = self.source / 'canny_shards/rank0.safetensors'
        save_file({'cannys': np.zeros((2, 8, 8), dtype=np.uint8), 'canny_valid': np.ones(2, dtype=np.uint8), 'labels': np.asarray([0, 2], dtype=np.int64)}, str(path))
        for row in self.rows:
            if row['path'] == 'canny_shards/rank0.safetensors':
                row.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.save(); download(self.manifest, self.output)
        with self.assertRaisesRegex(ValueError, 'global labels differ'):
            verify(self.manifest, self.output)

    def test_historical_shards_derive_mask_names_and_global_labels(self):
        import numpy as np
        from safetensors.numpy import save_file
        self.rows.clear()
        names = ['n00000001/a.JPEG', 'n00000002/b.JPEG']
        for folder, name, key, valid in [('mask_shards', 'masks_rank00_shard000', 'masks', 'mask_valid'), ('canny_shards', 'cannys_rank00_shard000', 'cannys', 'canny_valid')]:
            path = self.source / folder / (name + '.safetensors'); path.parent.mkdir()
            save_file({key: np.zeros((2, 8, 8), dtype=np.uint8), valid: np.ones(2, dtype=np.uint8)}, str(path))
            if key == 'cannys':
                path.with_suffix('.json').write_text(json.dumps({'filenames': names}))
        (self.source / 'paired/train.json').write_text(json.dumps({'filenames': names[:1]}))
        (self.source / 'paired/val.json').write_text(json.dumps({'filenames': names[1:]}))
        (self.source / 'class_ids.json').write_text(json.dumps({'n00000001': 0, 'n00000002': 1}))
        for path in list((self.source / 'mask_shards').iterdir()) + list((self.source / 'canny_shards').iterdir()) + list((self.source / 'paired').iterdir()) + [self.source / 'class_ids.json']:
            self.rows.append({'path': path.relative_to(self.source).as_posix(), 'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.data['resources'].update(mask_shard_dir='mask_shards', canny_shard_dir='canny_shards', class_ids='class_ids.json')
        self.data['validation'] = {'paired_filenames': 2, 'mask_invalid_paired': 0, 'canny_invalid_paired': 0}
        self.save(); download(self.manifest, self.output)
        self.assertEqual(verify(self.manifest, self.output)['validation']['paired_filenames'], 2)
        del self.data['resources']['class_ids']; self.save()
        with self.assertRaisesRegex(ValueError, 'require release resources.class_ids'):
            verify(self.manifest, self.output)

    def test_vavae_headers_labels_flip_and_condition_order(self):
        import numpy as np
        from safetensors.numpy import save_file
        self.rows.clear()
        names = ['n00000001/a.JPEG', 'n00000002/b.JPEG']
        labels = np.asarray([1, 2], dtype=np.int64)
        mapping = {f'n{index:08d}': index for index in range(1000)}
        (self.source / 'class_ids.json').write_text(json.dumps(mapping))
        for folder, prefix, key, flag in [('mask_shards', 'masks_', 'masks', 'mask_valid'), ('canny_shards', 'cannys_', 'cannys', 'canny_valid')]:
            path = self.source / folder / (prefix + 'rank00_shard000.safetensors'); path.parent.mkdir()
            save_file({key: np.zeros((2, 8, 8), dtype=np.uint8), flag: np.ones(2, dtype=np.uint8)}, str(path))
            if key == 'cannys':
                path.with_suffix('.json').write_text(json.dumps({'filenames': names}))
        path = self.source / 'vavae_latents/latents_rank00_shard000.safetensors'; path.parent.mkdir()
        arrays = {'latents': np.zeros((2, 32, 16, 16), dtype=np.float32), 'latents_flip': np.zeros((2, 32, 16, 16), dtype=np.float32), 'labels': labels}
        save_file(arrays, str(path)); path.with_suffix('.json').write_text(json.dumps({'filenames': names}))
        (self.source / 'paired/train.json').write_text(json.dumps({'filenames': names[:1]}))
        (self.source / 'paired/val.json').write_text(json.dumps({'filenames': names[1:]}))
        files = list((self.source / 'mask_shards').iterdir()) + list((self.source / 'canny_shards').iterdir()) + list((self.source / 'vavae_latents').iterdir()) + list((self.source / 'paired').iterdir()) + [self.source / 'class_ids.json']
        for file in files:
            self.rows.append({'path': file.relative_to(self.source).as_posix(), 'size_bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'component': 'vavae' if file.parent.name == 'vavae_latents' else 'common'})
        self.data['resources'].update(mask_shard_dir='mask_shards', canny_shard_dir='canny_shards', class_ids='class_ids.json', vavae_latents='vavae_latents')
        self.data['validation'] = {'paired_filenames': 2}
        self.save(); download(self.manifest, self.output, 'vavae')
        with self.assertRaisesRegex(ValueError, 'common component'):
            verify(self.manifest, self.output, 'vavae')
        download(self.manifest, self.output)
        result = verify(self.manifest, self.output, 'all')
        self.assertEqual(result['vavae_validation'], {'latent_shards': 1, 'latent_rows': 2, 'global_classes': 1000})
        def refresh(file):
            for row in self.rows:
                if row['path'] == file.relative_to(self.source).as_posix():
                    row.update(size_bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            self.save(); download(self.manifest, self.output, 'vavae')
        arrays['labels'] = np.asarray([1, 3], dtype=np.int64); save_file(arrays, str(path)); refresh(path)
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            verify(self.manifest, self.output, 'vavae')
        arrays['labels'] = labels; arrays['latents_flip'] = np.zeros((1, 32, 16, 16), dtype=np.float32)
        save_file(arrays, str(path)); refresh(path)
        with self.assertRaisesRegex(ValueError, 'original/flip shapes'):
            verify(self.manifest, self.output, 'vavae')
        arrays['latents_flip'] = arrays['latents']; arrays['labels'] = labels[::-1].copy()
        save_file(arrays, str(path)); refresh(path)
        path.with_suffix('.json').write_text(json.dumps({'filenames': names[::-1]})); refresh(path.with_suffix('.json'))
        with self.assertRaisesRegex(ValueError, 'filename order differs'):
            verify(self.manifest, self.output, 'vavae')

    def test_http_manifest_and_relative_source(self):
        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(self.source)))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/manifest.json'
            result = download(url, self.output)
            self.assertEqual(result['installed_files'], 4)
            self.assertEqual(Path(result['manifest']), self.output / 'release.json')
            self.assertEqual(verify(url, self.output)['verified_files'], 4)
        finally:
            server.shutdown(); server.server_close(); thread.join()
        installed_manifest = self.output / 'release.json'
        self.assertEqual(verify(installed_manifest, self.output)['verified_files'], 4)
        self.assertEqual(json.loads(installed_manifest.read_text())['source_manifest'], url)
        import yaml
        train = self.root / 'imagenet/train'; train.mkdir(parents=True)
        resources = self.root / 'resources.json'; resources.write_text(json.dumps({'imagenet_train': str(train)}))
        config = self.root / 'pixel/canny.yaml'; config.parent.mkdir(); config.write_text(yaml.safe_dump({'data': {'image_size': 256}, 'train': {'max_steps': 64200}}))
        configured = configure(installed_manifest, self.output, resources, [config], self.root / 'configured')
        self.assertEqual(yaml.safe_load(Path(configured['configs'][0]).read_text())['train']['max_steps'], 64200)
    def test_configure_keeps_science_and_common_needs_no_latents(self):
        import yaml
        download(self.manifest, self.output)
        train = self.root / 'imagenet/train'; train.mkdir(parents=True)
        resources = self.root / 'resources.json'; resources.write_text(json.dumps({'imagenet_train': str(train), 'pixel': {'resources.jit_source': '/opt/JiT', 'train.output_dir': '/runs/pixel'}}))
        config = self.root / 'templates/pixel/canny.yaml'; config.parent.mkdir(parents=True)
        recipe = {'data': {'image_size': 256, 'num_classes': 1000, 'phase': 'canny_mask'}, 'train': {'max_steps': 64200}, 'optimizer': {'lr': .0001}, 'transport': {'time_dist_shift': 6.9282032}}
        config.write_text(yaml.safe_dump(recipe))
        result = configure(self.manifest, self.output, resources, [config], self.root / 'configured')
        actual = yaml.safe_load(Path(result['configs'][0]).read_text())
        for key in ['optimizer', 'transport']:
            self.assertEqual(actual[key], recipe[key])
        self.assertEqual(actual['train']['max_steps'], 64200)
        self.assertEqual(actual['data']['data_path'], str(train.parent))
        resources.write_text(json.dumps({'imagenet_train': str(train), 'pixel': {'stages': {'canny': {'train.weight_init': '/runs/mask/checkpoints/best.pt'}}}}))
        configured = configure(self.manifest, self.output, resources, [config], self.root / 'stage_configured')
        self.assertEqual(yaml.safe_load(Path(configured['configs'][0]).read_text())['train']['weight_init'], '/runs/mask/checkpoints/best.pt')
        self.assertFalse((self.output / 'latents').exists())
        self.assertFalse(result['ready'])
        self.assertNotIn('data.fid_reference_file', [item['field'] for item in result['unresolved_resources']])
        for representation in ['rae', 'sdvae']:
            native = self.root / 'templates' / representation / 'warmup.yaml'; native.parent.mkdir()
            native.write_text(yaml.safe_dump(recipe))
            configured = configure(self.manifest, self.output, resources, [native], self.root / 'native_configured')
            actual_native = yaml.safe_load(Path(configured['configs'][0]).read_text())
            self.assertEqual(actual_native['data']['data_path'], str(train.parent))
            self.assertEqual(actual_native['optimizer'], recipe['optimizer'])
            if representation == 'sdvae':
                self.assertEqual(actual_native['data']['imagenet_root'], str(train.parent))
        resources.write_text(json.dumps({'imagenet_train': str(train), 'pixel': {'optimizer.lr': 1}}))
        with self.assertRaisesRegex(ValueError, 'Only resource'):
            configure(self.manifest, self.output, resources, [config], self.root / 'bad')
        vavae = self.root / 'templates/vavae/canny.yaml'; vavae.parent.mkdir(); vavae.write_text(yaml.safe_dump(recipe))
        resources.write_text(json.dumps({'imagenet_train': str(train)}))
        with self.assertRaisesRegex(ValueError, 'Missing or corrupted'):
            configure(self.manifest, self.output, resources, [vavae], self.root / 'missing')
        download(self.manifest, self.output, 'vavae')
        result = configure(self.manifest, self.output, resources, [vavae], self.root / 'configured')
        self.assertEqual(yaml.safe_load(Path(result['configs'][0]).read_text())['data']['data_path'], str(self.output / 'latents'))

if __name__ == '__main__': unittest.main()
