"""Download verified prepared data without importing neural/preparation dependencies."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.parse
import urllib.request


def _json(source):
    source = str(source)
    if urllib.parse.urlparse(source).scheme in {'http', 'https'}:
        with urllib.request.urlopen(source) as response:
            return json.load(response)
    return json.loads(Path(source).read_text(encoding='utf-8-sig'))


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError(f'Unsafe release path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {'', '.', '..'} for part in value.split('/')):
        raise ValueError(f'Unsafe release path: {value!r}')
    return path


def _target(root, relative):
    path = root.joinpath(*_relative(relative).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Release path escapes destination: {relative}')
    return path


def _manifest(source):
    data = _json(source)
    if data.get('schema_version') != 1 or not isinstance(data.get('files'), list):
        raise ValueError('Expected release manifest schema_version=1 and files list')
    names = set()
    for row in data['files']:
        _relative(row['path'])
        if row['path'].casefold() == 'release.json':
            raise ValueError('release.json is reserved for the installed release manifest')
        if row['path'] in names:
            raise ValueError('Duplicate release path')
        names.add(row['path'])
        if row.get('component', 'common') not in {'common', 'vavae'}:
            raise ValueError('Unknown release component')
        if row.get('encoding') not in {None, 'gzip'}:
            raise ValueError('Only uncompressed or gzip release files are supported')
        _check_metadata(row, 'size_bytes', 'sha256')
        if row.get('encoding') == 'gzip':
            _check_metadata(row, 'download_size_bytes', 'download_sha256')
        if 'source_path' in row:
            _relative(row['source_path'])
    for value in data.get('resources', {}).values():
        _relative(value)
    return data


def _check_metadata(row, size_key, hash_key):
    if type(row.get(size_key)) is not int or row[size_key] < 0:
        raise ValueError(f'{size_key} must be a nonnegative integer')
    if not re.fullmatch('[0-9a-f]{64}', row.get(hash_key, '')):
        raise ValueError(f'{hash_key} must be a lowercase SHA256')


def _check_file(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest() == digest


def _selected(data, component):
    return [row for row in data['files'] if component == 'all' or row.get('component', 'common') == component]


def verify(manifest, output, component='common'):
    data, root = _manifest(manifest), Path(output).resolve()
    rows = _selected(data, component)
    if not rows:
        raise ValueError(f'Manifest has no {component} files')
    failures = [row['path'] for row in rows if not _check_file(_target(root, row['path']), row['size_bytes'], row['sha256'])]
    if failures:
        raise ValueError('Missing or corrupted prepared data: ' + ', '.join(failures))
    result = {'release_id': data.get('release_id'), 'verified_files': len(rows), 'component': component}
    if component in {'common', 'all'} and data.get('validation') is not None:
        result['validation'] = _semantic(data, root)
    if component in {'vavae', 'all'} and data.get('validation') is not None:
        result['vavae_validation'] = _vavae_semantic(data, root)
    return result


def _semantic(data, root):
    """Validate filename alignment without materializing spatial condition arrays."""
    from safetensors import safe_open
    resources = data['resources']
    train = _json(_target(root, resources['paired_train_whitelist']))['filenames']
    val = _json(_target(root, resources['paired_val_whitelist']))['filenames']
    for names in [train, val]:
        if len(set(names)) != len(names):
            raise ValueError('Paired filename lists must be unique')
        for name in names:
            if len(_relative(name).parts) != 2:
                raise ValueError('Paired filenames must use synset/filename')
    if set(train).intersection(val):
        raise ValueError('Train and validation filenames overlap')
    mapping = _json(_target(root, resources['class_ids'])) if 'class_ids' in resources else None
    if mapping is not None and (not isinstance(mapping, dict) or any(type(value) is not int or value < 0 for value in mapping.values()) or len(set(mapping.values())) != len(mapping)):
        raise ValueError('class_ids must map synsets to unique nonnegative integer global IDs')
    indices = []
    for directory_key, tensor_key, valid_key in [('mask_shard_dir', 'masks', 'mask_valid'), ('canny_shard_dir', 'cannys', 'canny_valid')]:
        index = {}
        directory = _target(root, resources[directory_key])
        for path in sorted(directory.glob('*.safetensors')):
            sidecar_path = path.with_suffix('.json')
            if not sidecar_path.exists() and tensor_key == 'masks':
                sidecar_path = _target(root, resources['canny_shard_dir']) / path.name.replace('masks_', 'cannys_', 1)
                sidecar_path = sidecar_path.with_suffix('.json')
            sidecar = _json(sidecar_path)
            names = sidecar['filenames']
            with safe_open(str(path), framework='np') as tensor:
                labels = tensor.get_tensor('labels') if 'labels' in tensor.keys() else None
                flags = tensor.get_tensor(valid_key)
                shape = tensor.get_slice(tensor_key).get_shape()
            if labels is None:
                import numpy as np
                if 'class_ids' in sidecar:
                    labels = np.asarray(sidecar['class_ids'])
                else:
                    if mapping is None:
                        raise ValueError('Historical condition shards without labels require release resources.class_ids')
                    try:
                        labels = np.asarray([mapping[name.split('/')[0]] for name in names])
                    except KeyError as error:
                        raise ValueError('Filename synset absent from global class IDs') from error
            if len(shape) != 3 or len(names) != shape[0] or labels.shape != (len(names),) or flags.shape != (len(names),):
                raise ValueError(f'Condition arrays and filename sidecars misaligned: {path.name}')
            for name, label, flag in zip(names, labels, flags):
                if len(_relative(name).parts) != 2:
                    raise ValueError('Condition filenames must use synset/filename')
                if int(label) != label or label < 0:
                    raise ValueError('Condition labels must be nonnegative integer global IDs')
                if flag not in (0, 1):
                    raise ValueError('Condition validity flags must be binary')
                value = (int(label), int(flag))
                if name in index and index[name] != value:
                    raise ValueError(f'Conflicting condition duplicate: {name}')
                index[name] = value
        indices.append(index)
    masks, edges = indices
    required = set(train + val)
    missing = required - masks.keys() | required - edges.keys()
    if missing:
        raise ValueError(f'{len(missing)} paired filenames missing from condition sidecars')
    if any(masks[name][0] != edges[name][0] for name in required):
        raise ValueError('Mask and Canny global labels differ')
    if 'class_ids' in resources:
        if any(name.split('/')[0] not in mapping or masks[name][0] != mapping[name.split('/')[0]] for name in required):
            raise ValueError('Condition labels differ from full ImageFolder class IDs')
    counts = {'train_filenames': len(train), 'val_filenames': len(val), 'paired_filenames': len(required),
              'mask_invalid_paired': sum(not masks[name][1] for name in required),
              'canny_invalid_paired': sum(not edges[name][1] for name in required)}
    expected = data['validation']
    if any(key not in counts for key in expected):
        raise ValueError('Unknown manifest validation count')
    for key, value in expected.items():
        if type(value) is not int or counts[key] != value:
            raise ValueError(f'Manifest validation count differs: {key}: expected {value}, observed {counts[key]}')
    return counts


def _vavae_semantic(data, root):
    """Header and small label checks for the cached original/flipped views."""
    from safetensors import safe_open
    resources = data['resources']
    needed = ['mask_shard_dir', 'canny_shard_dir', 'class_ids', 'vavae_latents']
    if any(key not in resources for key in needed):
        raise ValueError('VAVAE semantic verification requires installed common conditions and full class_ids resources')
    mapping_path = _target(root, resources['class_ids'])
    if not mapping_path.is_file():
        raise ValueError('Install and verify the common component before VAVAE semantic verification')
    mapping = _json(mapping_path)
    if not isinstance(mapping, dict) or len(mapping) != 1000 or any(type(value) is not int for value in mapping.values()) or set(mapping.values()) != set(range(1000)):
        raise ValueError('VAVAE labels require the full 1000-class global ImageFolder mapping')
    directory = _target(root, resources['vavae_latents'])
    sources = sorted(directory.glob('latents_rank*.safetensors'))
    if not sources:
        raise ValueError('No VAVAE latent shards found')
    total = 0
    for path in sources:
        names = _json(path.with_suffix('.json'))['filenames']
        with safe_open(str(path), framework='np') as tensor:
            original = tensor.get_slice('latents').get_shape()
            flipped = tensor.get_slice('latents_flip').get_shape()
            labels = tensor.get_tensor('labels')
        if len(original) != 4 or original != flipped or original[0] != len(names) or labels.shape != (len(names),):
            raise ValueError(f'VAVAE original/flip shapes or labels misaligned: {path.name}')
        for name, label in zip(names, labels):
            if len(_relative(name).parts) != 2 or name.split('/')[0] not in mapping or label != mapping[name.split('/')[0]]:
                raise ValueError(f'VAVAE labels differ from global class IDs: {path.name}')
        for key, prefix, tensor_key in [('mask_shard_dir', 'masks_', 'masks'), ('canny_shard_dir', 'cannys_', 'cannys')]:
            condition = _target(root, resources[key]) / path.name.replace('latents_', prefix, 1)
            if not condition.is_file():
                raise ValueError(f'Install common conditions before VAVAE verification; missing {condition.name}')
            sidecar = condition.with_suffix('.json')
            if not sidecar.exists() and tensor_key == 'masks':
                sidecar = (_target(root, resources['canny_shard_dir']) / path.name.replace('latents_', 'cannys_', 1)).with_suffix('.json')
            if not sidecar.is_file():
                raise ValueError('Common condition filename sidecar is missing; install and verify common first')
            if _json(sidecar)['filenames'] != names:
                raise ValueError(f'VAVAE filename order differs from corresponding condition shard: {condition.name}')
            with safe_open(str(condition), framework='np') as tensor:
                shape = tensor.get_slice(tensor_key).get_shape()
                condition_labels = tensor.get_tensor('labels') if 'labels' in tensor.keys() else None
            if len(shape) != 3 or shape[0] != len(names):
                raise ValueError('VAVAE and corresponding condition shard row counts differ')
            if condition_labels is not None and (condition_labels.shape != labels.shape or not (condition_labels == labels).all()):
                raise ValueError('VAVAE and condition labels differ')
        total += len(names)
    return {'latent_shards': len(sources), 'latent_rows': total, 'global_classes': len(mapping)}


def _source(row, manifest, base):
    if row.get('url'):
        url = row['url']
        if urllib.parse.urlparse(url).scheme not in {'http', 'https'}:
            raise ValueError('File url must use HTTP or HTTPS')
        return url
    relative = row.get('source_path', row['path'])
    if base is None:
        if urllib.parse.urlparse(str(manifest)).scheme in {'http', 'https'}:
            base = urllib.parse.urljoin(str(manifest), './')
        else:
            base = str(Path(manifest).resolve().parent)
    if urllib.parse.urlparse(str(base)).scheme in {'http', 'https'}:
        return urllib.parse.urljoin(str(base).rstrip('/') + '/', urllib.parse.quote(relative))
    path = _target(Path(base).resolve(), relative)
    if not path.is_file():
        raise FileNotFoundError(f'No payload at {path}; supply --source-base pointing to the published release location or a local payload directory.')
    return path


def _copy_checked(source, destination, size, digest):
    remote = isinstance(source, str) and urllib.parse.urlparse(source).scheme in {'http', 'https'}
    hasher, count = hashlib.sha256(), 0
    with (urllib.request.urlopen(source) if remote else Path(source).open('rb')) as incoming, destination.open('wb') as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b''):
            count += len(block)
            if count > size:
                raise ValueError('Download exceeds declared size')
            hasher.update(block)
            outgoing.write(block)
    if count != size or hasher.hexdigest() != digest:
        raise ValueError('Download size or SHA256 mismatch')


def download(manifest, output, component='common', source_base=None):
    data, root = _manifest(manifest), Path(output).resolve()
    rows = _selected(data, component)
    if not rows:
        raise ValueError(f'Manifest has no {component} files')
    installed, reused = 0, 0
    for row in rows:
        destination = _target(root, row['path'])
        if _check_file(destination, row['size_bytes'], row['sha256']):
            reused += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Resolve once again after creating parents; never write through an escaping symlink.
        _target(root, row['path'])
        temporaries = []
        try:
            def temporary():
                fd, name = tempfile.mkstemp(prefix='.release-', dir=destination.parent)
                os.close(fd)
                path = Path(name); temporaries.append(path)
                return path
            payload = temporary()
            compressed = row.get('encoding') == 'gzip'
            _copy_checked(_source(row, manifest, source_base), payload,
                          row['download_size_bytes'] if compressed else row['size_bytes'],
                          row['download_sha256'] if compressed else row['sha256'])
            final = payload
            if compressed:
                final = temporary()
                count, hasher = 0, hashlib.sha256()
                with gzip.open(payload, 'rb') as incoming, final.open('wb') as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b''):
                        count += len(block)
                        if count > row['size_bytes']:
                            raise ValueError('Decompressed file exceeds declared size')
                        hasher.update(block); outgoing.write(block)
                if count != row['size_bytes'] or hasher.hexdigest() != row['sha256']:
                    raise ValueError('Decompressed file size or SHA256 mismatch')
            os.replace(final, destination)
            installed += 1
        finally:
            for path in temporaries:
                path.unlink(missing_ok=True)
    installed_manifest = _target(root, 'release.json')
    saved = {**data, 'source_manifest': str(manifest)}
    fd, temporary_name = tempfile.mkstemp(prefix='.release-manifest-', dir=root)
    temporary_manifest = Path(temporary_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(saved, indent=2, ensure_ascii=False) + '\n')
        os.replace(temporary_manifest, installed_manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return {'release_id': data.get('release_id'), 'installed_files': installed, 'reused_files': reused, 'component': component, 'manifest': str(installed_manifest)}


def configure(manifest, data_root, resources, configs, output):
    """Only edit enumerated resource fields; numerical experiment settings survive."""
    try:
        import yaml
    except ImportError as error:
        raise ImportError('configure requires PyYAML (included in the models extra)') from error
    data = _manifest(manifest)
    verify(manifest, data_root, 'common')
    explicit = _json(resources)
    train = Path(explicit['imagenet_train']).resolve()
    if train.name != 'train' or not train.is_dir():
        raise ValueError('imagenet_train must point to an existing ImageNet train directory')
    root = Path(data_root).resolve()
    package = {key: str(_target(root, value)) for key, value in data.get('resources', {}).items()}
    required = ['mask_shard_dir', 'canny_shard_dir', 'paired_train_whitelist', 'paired_val_whitelist']
    if any(key not in package for key in required):
        raise ValueError('Release resources must identify both conditions and both paired lists')
    output = Path(output).resolve()
    paths, unresolved, upstream_sources = [], [], {}
    vavae_verified = False
    for source in configs:
        source = Path(source).resolve()
        representation = source.parent.name
        if representation not in {'pixel', 'rae', 'sdvae', 'vavae'}:
            raise ValueError('Config must belong to a pixel/rae/sdvae/vavae directory')
        recipe = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
        representation_resources = dict(explicit.get(representation, {}))
        stages = representation_resources.pop('stages', {})
        upstream = representation_resources.pop('upstream_source', None)
        if representation != 'pixel':
            if upstream:
                upstream_sources[representation] = upstream
            else:
                unresolved.append({'config': representation + '/' + source.name, 'field': 'upstream_source (launcher --upstream-source)'})
        overrides = {**representation_resources, **stages.get(source.stem, {})}
        allowed = {'ckpt_path', 'train.weight_init', 'train.ckpt', 'train.output_dir',
                   'data.fid_reference_file', 'resources.jit_source', 'model.rae_config_yaml',
                   'data.vae_path'}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError('Only resource overrides are allowed: ' + ', '.join(sorted(unknown)))
        assignments = {'data.' + key: package[key] for key in required}
        assignments['data.data_path'] = str(train.parent)
        if representation == 'sdvae':
            assignments['data.imagenet_root'] = str(train.parent)
        if representation == 'vavae':
            if not vavae_verified:
                verify(manifest, data_root, 'vavae')
                vavae_verified = True
            if 'vavae_latents' not in package:
                raise ValueError('Release does not identify vavae_latents')
            assignments['data.data_path'] = package['vavae_latents']
        assignments.update(overrides)
        for dotted, value in assignments.items():
            keys = dotted.split('.'); node = recipe
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
        resource_fields = (set(assignments) - {'data.fid_reference_file', 'train.ckpt'}) | {'ckpt_path', 'train.weight_init', 'train.output_dir'}
        resource_fields |= {'resources.jit_source'} if representation == 'pixel' else set()
        resource_fields |= {'model.rae_config_yaml'} if representation == 'rae' else set()
        resource_fields |= {'data.vae_path'} if representation == 'sdvae' else set()
        for dotted in sorted(resource_fields):
            node = recipe
            for key in dotted.split('.'):
                node = node.get(key) if isinstance(node, dict) else None
            if node is None or node == '' or (isinstance(node, str) and ('YOUR_PATH' in node or 'path/to/' in node)):
                unresolved.append({'config': representation + '/' + source.name, 'field': dotted})
        target = output / representation / source.name
        if target == source:
            raise ValueError('Write configured recipes into a separate output directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding='utf-8')
        paths.append(str(target))
    return {'configs': paths, 'resource_file': str(Path(resources).resolve()), 'unresolved_resources': unresolved, 'ready': not unresolved, 'upstream_sources': upstream_sources}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ['download', 'verify']:
        stage = commands.add_parser(action)
        stage.add_argument('--manifest', required=True)
        stage.add_argument('--output', required=True)
        stage.add_argument('--component', choices=['common', 'vavae', 'all'], default='common')
        if action == 'download':
            stage.add_argument('--source-base')
    stage = commands.add_parser('configure')
    stage.add_argument('--manifest', required=True)
    stage.add_argument('--data-root', required=True)
    stage.add_argument('--resources', required=True, help='JSON containing imagenet_train and optional per-representation resource overrides')
    stage.add_argument('--configs', nargs='+', required=True)
    stage.add_argument('--output', required=True)
    args = vars(parser.parse_args(argv)); action = args.pop('action')
    result = {'download': download, 'verify': verify, 'configure': configure}[action](**args)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
