#!/usr/bin/env python3
"""Verify and harden only a trusted release's imported Nix store items."""

import argparse
import os
import re
import stat
import subprocess
from pathlib import Path

STORE_ITEM = re.compile(r'/nix/store/[0123456789abcdfghijklmnpqrsvwxyz]{32}-[A-Za-z0-9+._?=-]+')


def prepare(paths_file: Path, nix_store: str) -> None:
    if os.geteuid() != 0:
        raise SystemExit('Run with root privileges against a root-managed Nix store.')
    roots = paths_file.read_text().splitlines()
    if not roots or len(roots) > 4096 or len(set(roots)) != len(roots):
        raise SystemExit('Expected a bounded, unique list of imported store items.')
    # Validate the entire selection before running any privileged mutation.
    for root in roots:
        if len(root) > 4096 or STORE_ITEM.fullmatch(root) is None:
            raise SystemExit(f'Not a complete Nix store item: {root!r}')
        metadata = Path(root).lstat()
        if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
            raise SystemExit(f'Store item is not a regular file or directory: {root}')
        if Path(root).resolve(strict=True) != Path(root):
            raise SystemExit(f'Store item is not canonical: {root}')
    environment = dict(os.environ, NIX_REMOTE='local')
    # Import may reuse already registered paths. Check their contents too.
    for offset in range(0, len(roots), 32):
        subprocess.run([nix_store, '--verify-path', *roots[offset:offset + 32]], env=environment, check=True)
    for root in roots:
        metadata = Path(root).lstat()
        if metadata.st_uid != 0 or metadata.st_mode & 0o222:
            print(f'Hardening {root}: uid={metadata.st_uid} mode={stat.S_IMODE(metadata.st_mode):04o}', flush=True)
        # GNU tools do not traverse interior symlinks; never dereference them
        # for ownership. Executable bits and package contents are preserved.
        subprocess.run(['/usr/bin/chown', '-R', '--no-dereference', 'root:root', root], check=True)
        subprocess.run(['/usr/bin/chmod', '-R', 'a-w', root], check=True)
    for offset in range(0, len(roots), 32):
        subprocess.run([nix_store, '--verify-path', *roots[offset:offset + 32]], env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imported_paths', type=Path)
    parser.add_argument('--nix-store', required=True)
    args = parser.parse_args()
    prepare(args.imported_paths, args.nix_store)


if __name__ == '__main__':
    main()
