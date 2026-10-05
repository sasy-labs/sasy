"""The maintainer runtime builder uses matching ABI flags and propagates failures."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def setup_build(tmp_path, *, word_size='64', platform='Linux', fail=False, include=None):
    assets = tmp_path / 'policy assets'
    assets.mkdir()
    script = assets / 'build-test-runtime.sh'
    shutil.copy2(ROOT / 'souffle/build-test-runtime.sh', script)
    tools = tmp_path / 'tool prefix' / 'bin'
    tools.mkdir(parents=True)
    souffle = tools / 'souffle'
    souffle.write_text(f'#!/bin/sh\nprintf "Word size: {word_size} bits\\n"\n')
    cxx = tools / 'fake cxx'
    cxx.write_text(f'#!{sys.executable}\n' + '''import json, os, pathlib, sys
with open(os.environ['BUILD_COMMANDS'], 'a') as output:
    output.write(json.dumps({'cwd': str(pathlib.Path.cwd()), 'args': sys.argv[1:]}) + '\\n')
if os.environ.get('FAIL_BUILD'):
    sys.exit(9)
args = sys.argv[1:]
pathlib.Path(args[args.index('-o') + 1]).write_text('synthetic compiler output')
''')
    uname = tools / 'uname'
    uname.write_text(f'#!/bin/sh\nprintf "{platform}\\n"\n')
    for tool in (souffle, cxx, uname):
        tool.chmod(0o755)
    log = tmp_path / 'commands.jsonl'
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
               SASY_SOUFFLE=str(souffle), SASY_CXX=str(cxx), BUILD_COMMANDS=str(log))
    env.pop('SOUFFLE_INCLUDE', None)
    env.pop('SASY_SOUFFLE_INCLUDE', None)
    if fail:
        env['FAIL_BUILD'] = '1'
    if include:
        env['SOUFFLE_INCLUDE'] = str(include)
    result = subprocess.run(['bash', str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return assets, tools.parent / 'include', commands, result


@pytest.mark.parametrize('word_size,platform,library', [
    ('32', 'Linux', 'libfunctors.so'), ('64', 'Darwin', 'libfunctors.dylib'),
])
def test_runtime_uses_detected_word_size_and_matching_header_prefix(tmp_path, word_size, platform, library):
    assets, include, commands, result = setup_build(tmp_path, word_size=word_size, platform=platform)
    assert result.returncode == 0, result.stderr
    assert len(commands) == 2
    assert all(command['cwd'] == str(assets.resolve()) for command in commands)
    assert f'-DRAM_DOMAIN_SIZE={word_size}' in commands[1]['args']
    assert f'-I{include}' in commands[1]['args']
    assert (assets / 'souffle-interpreted').is_file()
    assert (assets / library).is_file()


def test_explicit_header_directory_is_preserved_as_one_argument(tmp_path):
    include = tmp_path / 'matching headers'
    _, _, commands, result = setup_build(tmp_path, include=include)
    assert result.returncode == 0, result.stderr
    assert f'-I{include}' in commands[1]['args']


def test_unknown_word_size_refuses_to_build(tmp_path):
    _, _, commands, result = setup_build(tmp_path, word_size='128')
    assert result.returncode != 0
    assert commands == []
    assert 'Cannot determine Souffle word size' in result.stderr


def test_failed_runtime_compile_stops_before_building_functors(tmp_path):
    _, _, commands, result = setup_build(tmp_path, fail=True)
    assert result.returncode == 9
    assert len(commands) == 1
