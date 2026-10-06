"""Maintainer-only publication of existing prepared arrays; never extracts features."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, indent=2) + '\n'
    if path.is_file() and path.read_text(encoding='utf-8') == contents:
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(contents, encoding='utf-8')
    os.replace(temporary, path)


def digest(path, git=False):
    path = Path(path)
    result = hashlib.sha1() if git else hashlib.sha256()
    if git:
        result.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def transient(error):
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if status in (401, 403):
        return False
    return status in (408, 409, 429) or (status is not None and status >= 500) or isinstance(error, (TimeoutError, ConnectionError, OSError)) or type(error).__name__ in {'ConnectTimeout', 'ReadTimeout', 'ConnectionError', 'TimeoutException', 'ConnectError', 'ReadError', 'RemoteProtocolError'}


def network_retry(operation):
    """Retry transient transport failures without exposing exception URLs."""
    for attempt in range(3):
        try:
            return operation()
        except Exception as error:
            if getattr(getattr(error, 'response', None), 'status_code', None) == 429 or not transient(error) or attempt == 2:
                raise
            print(f'Retrying transient network failure ({attempt + 1}/2).', flush=True)
            time.sleep(2 ** attempt)


def compress(source, target):
    """Deterministic gzip and both decompressed and transport checksums."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    original = hashlib.sha256()
    count = 0
    try:
        with source.open('rb') as incoming, temporary.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024 * 1024), b''):
                    original.update(block)
                    count += len(block)
                    outgoing.write(block)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {'size_bytes': count, 'sha256': original.hexdigest(), 'encoding': 'gzip',
            'download_size_bytes': target.stat().st_size, 'download_sha256': digest(target)}


def stage(args):
    root = Path(args.staging_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generated = root / 'metadata'
    jobs = []
    resources = {'mask_shard_dir': 'mask_shards', 'canny_shard_dir': 'canny_shards',
                 'paired_train_whitelist': 'paired/train.json', 'paired_val_whitelist': 'paired/val.json',
                 'sr95_whitelist': 'paired/sr95.json', 'class_ids': 'paired/class_ids.json'}
    validation = None
    if args.component in ('common', 'all'):
        required = ('mask_dir', 'canny_dir', 'paired_train', 'paired_val', 'sr95', 'imagenet_root')
        if any(not getattr(args, key) for key in required):
            raise ValueError('Common staging requires mask/Canny directories, paired lists, SR95 and ImageNet root')
        train, val = read(args.paired_train), read(args.paired_val)
        overlap = set(train['filenames']) & set(val['filenames'])
        if overlap - {'n01440764/n01440764_10048.JPEG'}:
            raise ValueError('Unexpected train/validation overlap')
        split_correction = {'dropped_validation_overlap': sorted(overlap),
                            'original_train_filenames': len(train['filenames']),
                            'original_val_filenames': len(val['filenames'])}
        val = {'filenames': [name for name in val['filenames'] if name not in overlap]}
        train = {'filenames': train['filenames']}
        if len(set(train['filenames'])) != len(train['filenames']) or len(set(val['filenames'])) != len(val['filenames']):
            raise ValueError('Duplicate paired filenames')
        classes = sorted(path.name for path in Path(args.imagenet_root).iterdir() if path.is_dir() and path.name.startswith('n'))
        if len(classes) != 1000:
            raise ValueError('ImageNet root must contain all 1000 synset directories')
        sr95 = read(args.sr95)
        # Retain public selection fields, excluding original absolute filesystem provenance.
        def clean(value):
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items() if not (isinstance(item, str) and (item.startswith('/') or ':\\' in item))}
            if isinstance(value, list):
                return [clean(item) for item in value if not (isinstance(item, str) and item.startswith('/'))]
            return value
        for name, value in [('train', train), ('val', val), ('sr95', clean(sr95)), ('class_ids', dict(zip(classes, range(1000))))]:
            path = generated / f'{name}.json'
            atomic_json(path, value)
            jobs.append((path, f'paired/{name}.json', 'common'))
        for key, directory in [('mask_shards', args.mask_dir), ('canny_shards', args.canny_dir)]:
            prefix = 'masks' if key == 'mask_shards' else 'cannys'
            arrays = sorted(Path(directory).glob(f'{prefix}_rank*_shard*.safetensors'))
            if not arrays:
                raise ValueError(f'No arrays in {key}')
            jobs.extend((path, f'{key}/{path.name}', 'common') for path in arrays)
            jobs.extend((path, f'{key}/{path.name}', 'common') for path in sorted(Path(directory).glob(f'{prefix}_rank*_shard*.json')))
        atomic_json(root / 'corrections.json', split_correction)
    # Validate original prepared headers/flags/order before expensive compression.
    from feature_information_dynamics.data.releases import _semantic, _vavae_semantic
    context_path = root / 'source_context.json'
    if args.component in ('common', 'all'):
        context = {'mask_shard_dir': str(Path(args.mask_dir).resolve()),
                   'canny_shard_dir': str(Path(args.canny_dir).resolve()),
                   'paired_train_whitelist': str((generated / 'train.json').resolve()),
                   'paired_val_whitelist': str((generated / 'val.json').resolve()),
                   'class_ids': str((generated / 'class_ids.json').resolve())}
        atomic_json(context_path, context)
    elif context_path.exists():
        context = read(context_path)
    else:
        raise ValueError('Stage common first to provide source context for VAVAE validation')
    # Source context is local maintainer state, never published. Linux remote root.
    if os.name != 'nt':
        semantic_root = Path('/')
        source_resources = {key: value.lstrip('/') for key, value in context.items()}
    else:
        semantic_root = Path(args.staging_dir).resolve().anchor
        semantic_root = Path(semantic_root)
        source_resources = {key: str(Path(value).relative_to(semantic_root)).replace('\\', '/') for key, value in context.items()}
    source_manifest = {'resources': source_resources, 'validation': {}}
    validation = _semantic(source_manifest, semantic_root)
    print('Source validation: ' + json.dumps(validation, sort_keys=True), flush=True)
    if args.component in ('vavae', 'all'):
        if not args.vavae_dir:
            raise ValueError('VAVAE staging requires --vavae-dir')
        directory = Path(args.vavae_dir)
        arrays = sorted(directory.glob('latents_rank*.safetensors'))
        paths = [item for path in arrays for item in (path, path.with_suffix('.json'))] + [directory / 'latents_stats.pt']
        if not arrays or any(not path.is_file() for path in paths):
            raise ValueError('VAVAE requires arrays, filename sidecars and latents_stats.pt')
        jobs.extend((path, f'vavae_latents/{path.name}', 'vavae') for path in paths)
        resources['vavae_latents'] = 'vavae_latents'
        source_resources['vavae_latents'] = str(directory.resolve().relative_to(semantic_root)).replace('\\', '/')
        print('VAVAE source validation: ' + json.dumps(_vavae_semantic(source_manifest, semantic_root), sort_keys=True), flush=True)
    journal_path = root / 'stage_journal.json'
    journal = read(journal_path) if journal_path.exists() else {}
    lock = threading.Lock()
    def process(job):
        source, installed, component = job
        stat = source.stat()
        signature = {'source': str(source.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        transport = 'payload/' + installed + '.gz'
        cached = journal.get(installed)
        target = root / transport
        if cached and cached['signature'] == signature and target.is_file() and target.stat().st_size == cached['row']['download_size_bytes'] and digest(target) == cached['row']['download_sha256']:
            return cached['row']
        row = {'path': installed, 'component': component, 'source_path': transport, **compress(source, target)}
        with lock:
            journal[installed] = {'signature': signature, 'row': row}
            atomic_json(journal_path, journal)
            print('Staged ' + installed, flush=True)
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, jobs))
    # Allow separate component passes to build one complete manifest.
    manifest_path = root / 'release.json'
    old = read(manifest_path) if manifest_path.exists() else {}
    selected = {row['component'] for row in rows}
    rows += [row for row in old.get('files', []) if row['component'] not in selected]
    resources = {**old.get('resources', {}), **resources}
    manifest = {'schema_version': 1, 'release_id': 'feature-information-dynamics-imagenet-sr95-v1',
                'resources': resources, 'files': sorted(rows, key=lambda row: row['path'])}
    if validation is not None or 'validation' in old:
        manifest['validation'] = validation if validation is not None else old['validation']
    manifest['split_correction'] = split_correction if args.component in ('common', 'all') else old.get('split_correction', {})
    atomic_json(manifest_path, manifest)
    return manifest


def upload(args):
    from huggingface_hub import HfApi
    root = Path(args.staging_dir)
    manifest = read(root / 'release.json')
    api = HfApi()
    def inventory():
        info = api.repo_info(args.repo_id, repo_type='dataset', files_metadata=True)
        return {item.rfilename: item for item in info.siblings}
    remote = inventory()
    def matches(path, metadata):
        item = remote.get(path)
        if item is None:
            return False
        if item.lfs:
            return item.lfs.sha256 == metadata['download_sha256']
        return item.blob_id == digest(root / path, git=True)
    rows = [row for row in manifest['files'] if args.component == 'all' or row['component'] == args.component]
    if getattr(args, 'batch_commit', False):
        from huggingface_hub import CommitOperationAdd
        pending = []
        for row in rows:
            path = row['source_path']
            if digest(root / path) != row['download_sha256']:
                raise ValueError('Staged payload hash mismatch: ' + path)
            if not matches(path, row):
                pending.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(root / path)))
        if pending:
            print(f'Preuploading {len(pending)} remaining payloads as one batch.', flush=True)
            network_retry(lambda: api.preupload_lfs_files(repo_id=args.repo_id, repo_type='dataset', additions=pending, num_threads=args.workers))
            for attempt in range(71):
                try:
                    network_retry(lambda: api.create_commit(repo_id=args.repo_id, repo_type='dataset', operations=pending, commit_message='Add verified prepared data payloads'))
                    break
                except Exception as error:
                    status = getattr(getattr(error, 'response', None), 'status_code', None)
                    if status != 429 or attempt == 70:
                        raise
                    print('Repository commit limit: retaining uploaded payloads; retry in 60 seconds.', flush=True)
                    time.sleep(60)
            print('Batch payload commit published.', flush=True)
    def upload_payload(row):
        path = row['source_path']
        if digest(root / path) != row['download_sha256']:
            raise ValueError('Staged payload hash mismatch: ' + path)
        if not matches(path, row):
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=str(root / path), path_in_repo=path, repo_id=args.repo_id, repo_type='dataset', commit_message='Add prepared data payload')
                    break
                except Exception as error:
                    status = getattr(getattr(error, 'response', None), 'status_code', None)
                    if status in (401, 403) or attempt == 2:
                        raise
                    if not transient(error):
                        raise
                    print(f'Retrying payload upload ({attempt + 1}/2): {path}', flush=True)
                    time.sleep(2 ** attempt)
            print('Uploaded ' + path, flush=True)
    if not getattr(args, 'batch_commit', False):
        with ThreadPoolExecutor(max_workers=getattr(args, 'workers', 1)) as pool:
            list(pool.map(upload_payload, rows))
    if getattr(args, 'payload_only', False):
        print('Selected payload uploads finished; manifest publication disabled.', flush=True)
        return False
    remote = inventory()
    if any(not matches(row['source_path'], row) for row in manifest['files']):
        print('Selected component uploaded; release manifest withheld until all payloads are present.', flush=True)
        return False
    snapshot = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    for attempt in range(71):
        try:
            network_retry(lambda: api.upload_file(path_or_fileobj=snapshot, path_in_repo='release.json', repo_id=args.repo_id, repo_type='dataset', commit_message='Publish verified prepared data manifest'))
            break
        except Exception as error:
            if getattr(getattr(error, 'response', None), 'status_code', None) != 429 or attempt == 70:
                raise
            print('Manifest commit limit: retry in 60 seconds.', flush=True)
            time.sleep(60)
    print('Published complete release manifest.', flush=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    staging = commands.add_parser('stage')
    publishing = commands.add_parser('upload')
    for command in (staging, publishing):
        command.add_argument('--staging-dir', required=True)
        command.add_argument('--component', choices=['common', 'vavae', 'all'], default='all')
    staging.add_argument('--workers', type=int, default=4)
    for name in ('mask-dir', 'canny-dir', 'paired-train', 'paired-val', 'sr95', 'imagenet-root', 'vavae-dir'):
        staging.add_argument('--' + name)
    publishing.add_argument('--repo-id', required=True)
    publishing.add_argument('--workers', type=int, default=1)
    publishing.add_argument('--payload-only', action='store_true')
    publishing.add_argument('--batch-commit', action='store_true', help='Preupload remaining payloads and publish them in one repository commit')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return stage(args) if args.command == 'stage' else upload(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Network exceptions can contain signed URLs; do not disclose them in job logs.
        print(f'Publication stopped ({type(error).__name__}); resume after checking inputs/network.', file=sys.stderr)
        sys.exit(1)
