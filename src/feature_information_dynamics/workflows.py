"""Run each representation in its own process with explicit upstream sources."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import subprocess
import sys

PREFIX = 'feature_information_dynamics.'
MODULES = {
    ('rae', 'train'): 'rae.train', ('rae', 'eval'): 'rae.evaluate',
    ('sdvae', 'train'): 'sdvae.train', ('sdvae', 'eval'): 'sdvae.evaluate',
    ('vavae', 'train'): 'vavae.train', ('vavae', 'eval'): 'vavae.evaluate',
    ('vavae', 'cache'): 'vavae.extract_latents',
}


def source_paths(representation: str, action: str, upstream_source: str) -> list[str]:
    if (representation, action) not in MODULES:
        raise ValueError(f'{representation} uses online inputs; no cache action is supported.')
    root = Path(upstream_source).expanduser().resolve()
    if representation == 'rae':
        root = root / 'src' if (root / 'src/stage2').is_dir() else root
        expected = root / 'stage2/models/DDT.py'
        paths = [str(root)]
    elif representation == 'vavae':
        expected = root / 'models/lightningdit.py'
        paths = [str(root), str(root / 'vavae')]
    else:
        expected = root / 'models.py'
        paths = [str(root)]
    if not expected.is_file():
        raise ValueError(f'Official source layout not found: {expected}')
    return paths


def execute(representation: str, action: str, arguments: list[str], paths: list[str]) -> None:
    if action == 'cache':
        if '--encoder' in arguments:
            index = arguments.index('--encoder')
            if index + 1 >= len(arguments) or arguments[index + 1] != 'vavae':
                raise ValueError('VAVAE cache requires the VAVAE encoder.')
        else:
            arguments = ['--encoder', 'vavae', *arguments]
    sys.path[:0] = paths
    sys.argv = [MODULES[(representation, action)], *arguments]
    runpy.run_module(PREFIX + MODULES[(representation, action)], run_name='__main__')


def build_command(representation: str, action: str, arguments: list[str], upstream_source: str) -> list[str]:
    source_paths(representation, action, upstream_source)
    return [sys.executable, '-m', PREFIX + 'workflows', representation, action,
            '--upstream-source', upstream_source, '--in-process', '--', *arguments]


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    split = values.index('--') if '--' in values else len(values)
    own, original = values[:split], values[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('representation', choices=['rae', 'sdvae', 'vavae'])
    parser.add_argument('action', choices=['train', 'cache', 'eval'])
    parser.add_argument('--upstream-source', required=True)
    parser.add_argument('--in-process', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(own)
    paths = source_paths(args.representation, args.action, args.upstream_source)
    if args.in_process:
        execute(args.representation, args.action, original, paths)
        return 0
    return subprocess.call(build_command(args.representation, args.action, original, args.upstream_source))


if __name__ == '__main__':
    raise SystemExit(main())
