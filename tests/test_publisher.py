import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).parents[1] / 'tools/publish_huggingface_data.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublisherTests(unittest.TestCase):
    def test_batch_retries_keep_operations_and_manifest_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'original'
            source.write_bytes(b'prepared data')
            row = {'path': 'data', 'component': 'common', 'source_path': 'payload/data.gz',
                   **publisher.compress(source, root / 'payload/data.gz')}
            publisher.atomic_json(root / 'release.json', {'files': [row]})
            events, remote = [], {}
            class Operation:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
            class RateLimit(Exception):
                response = SimpleNamespace(status_code=429)
            class API:
                def repo_info(self, *args, **kwargs):
                    events.append('inventory')
                    return SimpleNamespace(siblings=list(remote.values()))
                def preupload_lfs_files(self, additions, **kwargs):
                    events.append('preupload')
                    if events.count('preupload') == 1:
                        raise ConnectionError('sensitive')
                    self.operations = additions
                def create_commit(self, operations, **kwargs):
                    self_outer.assertIs(operations, self.operations)
                    events.append('commit')
                    if events.count('commit') == 1:
                        raise RateLimit('sensitive')
                    remote[row['source_path']] = SimpleNamespace(rfilename=row['source_path'],
                        lfs=SimpleNamespace(sha256=row['download_sha256']), blob_id=None)
                def upload_file(self, path_in_repo, **kwargs):
                    events.append(path_in_repo)
            self_outer = self
            args = SimpleNamespace(staging_dir=str(root), repo_id='org/data', component='all', workers=4, batch_commit=True)
            module = SimpleNamespace(HfApi=API, CommitOperationAdd=Operation)
            with patch.dict(sys.modules, {'huggingface_hub': module}), patch.object(publisher.time, 'sleep'):
                self.assertTrue(publisher.upload(args))
            self.assertEqual(events, ['inventory', 'preupload', 'preupload', 'commit', 'commit', 'inventory', 'release.json'])

    def test_batch_preuploads_once_and_retries_commit_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'; source.write_bytes(b'data')
            row = {'path': 'data', 'component': 'common', 'source_path': 'payload/data.gz',
                   **publisher.compress(source, root / 'payload/data.gz')}
            publisher.atomic_json(root / 'release.json', {'files': [row]})
            events, remote = [], {}
            class Limited(Exception):
                response = SimpleNamespace(status_code=429)
            class API:
                attempts = 0
                def repo_info(self, *args, **kwargs):
                    return SimpleNamespace(siblings=list(remote.values()))
                def preupload_lfs_files(self, additions, **kwargs):
                    events.append('preupload')
                    self.additions = additions
                def create_commit(self, operations, **kwargs):
                    self.attempts += 1; events.append('commit')
                    if self.attempts == 1:
                        raise Limited()
                    for op in operations:
                        remote[op.path_in_repo] = SimpleNamespace(rfilename=op.path_in_repo, lfs=SimpleNamespace(sha256=publisher.digest(op.path_or_fileobj)))
                def upload_file(self, path_in_repo, **kwargs):
                    events.append(path_in_repo)
            args = SimpleNamespace(staging_dir=str(root), repo_id='org/data', component='all', workers=4, batch_commit=True)
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=API, CommitOperationAdd=lambda **kw: SimpleNamespace(**kw))}), patch.object(publisher.time, 'sleep') as sleep:
                self.assertTrue(publisher.upload(args))
                sleep.assert_called_once_with(60)
            self.assertEqual(events, ['preupload', 'commit', 'commit', 'release.json'])

    def test_exhausted_retry_never_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_bytes(b'data')
            row = {'path': 'data', 'component': 'common', 'source_path': 'payload/data.gz',
                   **publisher.compress(source, root / 'payload/data.gz')}
            publisher.atomic_json(root / 'release.json', {'files': [row]})
            events = []
            class API:
                def repo_info(self, *args, **kwargs):
                    events.append('inventory')
                    return SimpleNamespace(siblings=[])
                def upload_file(self, path_in_repo, **kwargs):
                    events.append(path_in_repo)
                    raise ConnectionError('sensitive URL')
            args = SimpleNamespace(staging_dir=str(root), repo_id='org/data', component='all', workers=4)
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=API)}), patch.object(publisher.time, 'sleep'):
                with self.assertRaises(ConnectionError):
                    publisher.upload(args)
            self.assertEqual(events, ['inventory'] + ['payload/data.gz'] * 3)

    def test_parallel_retry_finishes_before_manifest_and_uses_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, attempts, remote, events = [], {}, {}, []
            for number in range(4):
                original = root / f'original{number}'
                original.write_bytes(str(number).encode())
                path = f'payload/{number}.gz'
                rows.append({'path': str(number), 'component': 'common', 'source_path': path,
                             **publisher.compress(original, root / path)})
            original_manifest = {'files': rows}
            publisher.atomic_json(root / 'release.json', original_manifest)
            class API:
                def repo_info(self, *args, **kwargs):
                    events.append(('inventory', len(remote)))
                    return SimpleNamespace(siblings=list(remote.values()))
                def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
                    if path_in_repo == 'release.json':
                        self_manifest = json.loads(path_or_fileobj)
                        self_outer.assertEqual(self_manifest, original_manifest)
                        self_outer.assertEqual(len(remote), 4)
                        events.append(('manifest', 4))
                        return
                    attempts[path_in_repo] = attempts.get(path_in_repo, 0) + 1
                    if path_in_repo == 'payload/0.gz' and attempts[path_in_repo] == 1:
                        raise ConnectionError('secret signed URL must not be printed')
                    remote[path_in_repo] = SimpleNamespace(rfilename=path_in_repo,
                        lfs=SimpleNamespace(sha256=publisher.digest(path_or_fileobj)), blob_id=None)
                    if path_in_repo == 'payload/0.gz':
                        publisher.atomic_json(root / 'release.json', {'files': []})
            self_outer = self
            args = SimpleNamespace(staging_dir=str(root), repo_id='org/data', component='all', workers=4)
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=API)}), patch.object(publisher.time, 'sleep'):
                self.assertTrue(publisher.upload(args))
            self.assertEqual(attempts['payload/0.gz'], 2)
            self.assertEqual(events, [('inventory', 0), ('inventory', 4), ('manifest', 4)])

    def test_manifest_last_and_withheld_for_missing_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for component in ['common', 'vavae']:
                source = root / component
                source.write_bytes(component.encode())
                payload = f'payload/{component}.gz'
                rows.append({'path': component, 'component': component, 'source_path': payload,
                             **publisher.compress(source, root / payload)})
            publisher.atomic_json(root / 'release.json', {'files': rows})
            uploaded, remote = [], {}
            class API:
                def repo_info(self, *args, **kwargs):
                    return SimpleNamespace(siblings=list(remote.values()))
                def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
                    uploaded.append(path_in_repo)
                    remote[path_in_repo] = SimpleNamespace(rfilename=path_in_repo,
                        lfs=SimpleNamespace(sha256=publisher.hashlib.sha256(path_or_fileobj).hexdigest() if isinstance(path_or_fileobj, bytes) else publisher.digest(path_or_fileobj)), blob_id=None)
            args = SimpleNamespace(staging_dir=str(root), repo_id='org/data', component='common')
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=API)}):
                self.assertFalse(publisher.upload(args))
                self.assertNotIn('release.json', uploaded)
                args.component = 'all'
                self.assertTrue(publisher.upload(args))
                self.assertEqual(uploaded, ['payload/common.gz', 'payload/vavae.gz', 'release.json'])
                args.payload_only = True
                self.assertFalse(publisher.upload(args))
                self.assertEqual(uploaded.count('release.json'), 1)

    def test_gzip_is_reproducible_and_hashes_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'original'
            source.write_bytes(b'prepared data\0' * 10000)
            first = publisher.compress(source, root / 'one.gz')
            second = publisher.compress(source, root / 'two.gz')
            self.assertEqual(first, second)
            self.assertEqual(gzip.decompress((root / 'one.gz').read_bytes()), source.read_bytes())
            self.assertEqual(first['sha256'], publisher.digest(source))

    def test_stage_common_correction_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ['masks', 'edges', 'imagenet']:
                (root / folder).mkdir()
            for number in range(1000):
                (root / 'imagenet' / f'n{number:08d}').mkdir()
            for folder in ['masks', 'edges']:
                prefix = 'masks' if folder == 'masks' else 'cannys'
                (root / folder / f'{prefix}_rank0_shard0.safetensors').write_bytes(b'fixture payload')
            publisher.atomic_json(root / 'edges/cannys_rank0_shard0.json', {'filenames': ['n00000000/a.JPEG']})
            publisher.atomic_json(root / 'edges/manifest.json', {'private': '/do/not/publish'})
            overlap = 'n01440764/n01440764_10048.JPEG'
            publisher.atomic_json(root / 'train.json', {'filenames': [overlap]})
            publisher.atomic_json(root / 'val.json', {'filenames': [overlap, 'n00000000/a.JPEG']})
            publisher.atomic_json(root / 'sr95.json', {'synsets': ['n00000000'], 'source': '/private/path'})
            args = SimpleNamespace(staging_dir=str(root / 'stage'), component='common',
                                   mask_dir=str(root / 'masks'), canny_dir=str(root / 'edges'),
                                   imagenet_root=str(root / 'imagenet'), paired_train=str(root / 'train.json'),
                                   paired_val=str(root / 'val.json'), sr95=str(root / 'sr95.json'),
                                   mask_invalid_paired=0, workers=2)
            counts = {'train_filenames': 1, 'val_filenames': 1, 'paired_filenames': 2, 'mask_invalid_paired': 0, 'canny_invalid_paired': 0}
            with patch('feature_information_dynamics.data.releases._semantic', return_value=counts) as semantic:
                first = publisher.stage(args)
                second = publisher.stage(args)
                self.assertEqual(semantic.call_count, 2)
            self.assertEqual(first, second)
            self.assertEqual(first['validation']['val_filenames'], 1)
            self.assertNotIn('source', publisher.read(root / 'stage/metadata/sr95.json'))
            self.assertEqual(len(publisher.read(root / 'stage/metadata/class_ids.json')), 1000)
            self.assertEqual(publisher.read(root / 'stage/corrections.json')['dropped_validation_overlap'], [overlap])
            self.assertEqual(first['split_correction']['original_val_filenames'], 2)
            self.assertFalse(any(row['path'].endswith('manifest.json') for row in first['files']))


if __name__ == '__main__':
    unittest.main()
