#!/usr/bin/env python3
"""Check the optional Nix toolchain without building or contacting an engine.

Run: scripts/nix-dev.sh --command python3 scripts/nix-smoke.py
The policy and native artifacts are created in a temporary directory and removed.
This is a toolchain check, not a qualification of the engine runtime sandbox or
of native release wheels built using Nix dependencies.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print('+', *args, flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=120)


def main() -> None:
    for tool in ('rustc', 'cargo', 'souffle', 'protoc', 'openssl', 'node', 'bun', 'g++', 'mcpp'):
        resolved = Path(shutil.which(tool) or '').resolve()
        if not resolved.is_relative_to('/nix/store'):
            raise RuntimeError(f'{tool} is not supplied by Nix: {resolved}')
        print(f'{tool}: {resolved}', flush=True)
    if not Path(sys.executable).resolve().is_relative_to('/nix/store'):
        raise RuntimeError('Run this check using the Nix shell Python')
    for command in (('rustc', '--version'), ('cargo', '--version'),
                    ('souffle', '--version'), ('protoc', '--version'),
                    ('openssl', 'version'), ('node', '--version'),
                    ('bun', '--version')):
        run(*command)
    cxx = os.environ['SASY_CXX']
    include = Path(os.environ['SOUFFLE_INCLUDE'])
    assert include.resolve().is_relative_to('/nix/store')
    assert Path(cxx).resolve().is_relative_to('/nix/store')
    assert os.environ['SASY_SOUFFLE_INCLUDE'] == str(include)
    assert (include / 'souffle/SouffleInterface.h').is_file()
    suffix = 'dylib' if os.uname().sysname == 'Darwin' else 'so'
    clang = Path(os.environ['LIBCLANG_PATH']) / f'libclang.{suffix}'
    ctypes.CDLL(str(clang))
    print('libclang load: PASS', flush=True)
    version = subprocess.check_output(['souffle', '--version'], text=True)
    bits = re.search(r'Word size: (32|64) bits', version).group(1)
    preprocess = runpy.run_path(str(ROOT / 'souffle/sugar.py'))['preprocess']
    common = (ROOT / 'policies/common_policy.dl').read_text()
    authored = '''
Principal("nix-smoke").
IsAuthorized(idx) :- Actions(idx, $CallTool("Read", args)), @json_get_str(args, "file_path") = "notes.txt".
Unauthorized(idx) :- Actions(idx, action), IsTool(action, "Write").
Actions(0, $CallTool("Read", "{\\"file_path\\":\\"notes.txt\\"}")).
Actions(1, $CallTool("Write", "{}")).
'''
    source = preprocess(common + '\n// === USER_POLICY_BEGIN ===\n' + authored)
    source = re.sub(r'^\.output (\w+).*$', r'.output \1', source, flags=re.M)
    with tempfile.TemporaryDirectory(prefix='sasy-nix-smoke-') as temporary:
        work = Path(temporary)
        policy = work / 'read-only.dl'
        policy.write_text(source)
        for relation in re.findall(r'^\.input (\w+)', source, flags=re.M):
            (work / f'{relation}.facts').touch()
        run(cxx, '-std=c++17', '-O2', '-shared', '-fPIC',
            f'-DRAM_DOMAIN_SIZE={bits}', f'-I{include}',
            str(ROOT / 'souffle/functors.cpp'),
            '-o', str(work / f'libfunctors.{suffix}'))
        # Both execution modes must compute the same actual SASY decision.
        for backend in ('interpreted', 'compiled'):
            output = work / backend
            output.mkdir()
            if backend == 'compiled':
                run('souffle', '-L', str(work), '-l', 'functors',
                    '-o', str(work / 'policy'), str(policy), cwd=work)
                # The smoke's library is outside the Nix store. Scope its loader
                # search path to this child; leave the development shell unchanged.
                loader = 'DYLD_LIBRARY_PATH' if suffix == 'dylib' else 'LD_LIBRARY_PATH'
                runtime_env = dict(os.environ, **{loader: str(work)})
                run(str(work / 'policy'), '-F', str(work), '-D', str(output),
                    cwd=work, env=runtime_env)
            else:
                run('souffle', '-L', str(work), '-l', 'functors',
                    '-F', str(work), '-D', str(output), str(policy), cwd=work)
            assert (output / 'Authorized.csv').read_text().strip() == '0', backend
            assert (output / 'Unauthorized.csv').read_text().strip() == '1', backend
            print(f'{backend}: Read authorized; Write denied: PASS', flush=True)
        rust = work / 'probe.rs'
        rust.write_text('fn main() { assert_eq!(2 + 2, 4); println!("Rust compile/run: PASS"); }')
        run('rustc', str(rust), '-o', str(work / 'rust-probe'))
        run(str(work / 'rust-probe'))
        proto = work / 'probe.proto'
        proto.write_text('syntax = "proto3"; message Probe { string value = 1; }')
        run('protoc', f'-I{work}', f'--descriptor_set_out={work / "probe.pb"}', str(proto))
        assert (work / 'probe.pb').stat().st_size > 0
    print('Nix toolchain smoke: PASS', flush=True)


if __name__ == '__main__':
    main()
