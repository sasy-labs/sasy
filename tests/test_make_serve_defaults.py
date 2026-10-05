"""Exercise quick-start startup defaults without starting a service."""
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    shutil.copy2(ROOT / 'Makefile', tmp_path / 'Makefile')
    (tmp_path / 'config/auth').mkdir(parents=True)
    for name in ('auth/apikey.json', 'auth_config.yaml', 'transforms.json'):
        (tmp_path / 'config' / name).write_text('{}')
    (tmp_path / 'certs').mkdir()
    for name in ('server.crt', 'server.key'):
        (tmp_path / 'certs' / name).write_text('test fixture')
    return tmp_path


def run_serve(checkout, *args):
    # Only the Makefile guards and echo run. Never load local project config.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('SASY_', 'MAKE', 'MFLAGS'))}
    return subprocess.run(['make', '--no-print-directory', 'serve', 'SASY_BIN=/bin/echo', *args],
                          cwd=checkout, env=env, capture_output=True, text=True, timeout=10)


def command(checkout, *args):
    result = run_serve(checkout, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.splitlines()[-1]


def test_serve_binds_loopback_by_default(checkout):
    assert '--addr 127.0.0.1:10089' in command(checkout)


def test_serve_requires_authentication_and_tls(checkout):
    cmd = command(checkout)
    for option in ('--auth-provider config/auth/apikey.json', '--auth-config config/auth_config.yaml',
                   '--tls-cert certs/server.crt', '--tls-key certs/server.key'):
        assert option in cmd


def test_serve_does_not_enable_caller_cpp_by_default(checkout):
    assert '--allow-user-functors' not in command(checkout)


@pytest.mark.parametrize('addr', ['0.0.0.0:51000', '127.0.0.1:51001'])
def test_address_override_preserves_secure_defaults(checkout, addr):
    cmd = command(checkout, f'SASY_ADDR={addr}')
    assert f'--addr {addr}' in cmd
    assert '--tls-cert' in cmd and '--auth-provider' in cmd
    assert '--allow-user-functors' not in cmd


def test_local_persistent_store_paths(checkout):
    cmd = command(checkout)
    assert '--data-dir data/graph' in cmd
    assert '--credentials-db' not in cmd  # the store lives under --data-dir


@pytest.mark.parametrize('missing', ['config/auth/apikey.json', 'config/auth_config.yaml', 'config/transforms.json'])
def test_missing_config_fails_before_engine_dispatch(checkout, missing):
    (checkout / missing).unlink()
    result = run_serve(checkout)
    assert result.returncode != 0
    assert 'run make init-config' in result.stderr
    assert not any(line.startswith('serve --addr') for line in result.stdout.splitlines())


@pytest.mark.parametrize('missing', ['certs/server.crt', 'certs/server.key'])
def test_missing_tls_fails_before_engine_dispatch(checkout, missing):
    (checkout / missing).unlink()
    result = run_serve(checkout)
    assert result.returncode != 0
    assert 'run make certs' in result.stderr
    assert not any(line.startswith('serve --addr') for line in result.stdout.splitlines())


@pytest.mark.parametrize('port', ['10089', '10090'])
def test_docker_service_prepares_and_reuses_local_state(tmp_path, port):
    if not shutil.which('openssl'):
        pytest.skip('OpenSSL is required for local development certificates')
    checkout = tmp_path / 'project with spaces'
    checkout.mkdir()
    shutil.copy2(ROOT / 'Makefile', checkout / 'Makefile')
    (checkout / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/setup_local_env.py', checkout / 'scripts/setup_local_env.py')
    for name in ('auth/apikey.example.json', 'auth/jwt.example.json',
                 'auth_config.example.yaml', 'transforms.example.json'):
        target = checkout / 'config' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'config' / name, target)
    commands = tmp_path / 'bin'
    commands.mkdir()
    docker = commands / 'docker'
    docker.write_text('#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n'
                      'Path(os.environ["DOCKER_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
    docker.chmod(0o755)
    capture = tmp_path / 'docker-args.json'
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('SASY_', 'MAKE', 'MFLAGS'))}
    env.update(PATH=str(commands) + os.pathsep + env['PATH'], DOCKER_CAPTURE=str(capture))
    invocation = ['make', '--no-print-directory', 'docker-serve',
                  f'SASY_DOCKER_PORT={port}', 'SASY_IMAGE=local-test-image']
    first = subprocess.run(invocation, cwd=checkout, env=env, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr
    args = json.loads(capture.read_text())
    assert args[:3] == ['run', '--rm', '--init']
    assert args[args.index('-p') + 1] == f'127.0.0.1:{port}:10089'
    assert args[args.index('--user') + 1] == f'{os.getuid()}:{os.getgid()}'
    assert args[-1] == 'local-test-image'
    for mount in ('config:/config:ro', 'certs:/certs:ro', 'data:/data'):
        assert f'{checkout}/{mount}' in args
    assert (checkout / 'data').is_dir()
    files = ['config/auth/apikey.json', 'certs/server.crt', 'certs/server.key', '.env']
    original = {name: (checkout / name).read_bytes() for name in files}
    keys = json.loads(original[files[0]])['static_keys']
    assert set(keys.values()) == {'client', 'admin'}
    assert all(len(key) >= 32 for key in keys)
    assert (checkout / files[0]).stat().st_mode & 0o077 == 0
    settings = dotenv_values(checkout / '.env')
    assert settings['SASY_URL'] == f'localhost:{port}'
    assert settings['TLS_CA_PATH'] == str(checkout / 'certs/server.crt')
    assert keys[settings['SASY_API_KEY']] == 'client'
    assert (checkout / '.env').stat().st_mode & 0o777 == 0o600
    assert all(key not in first.stdout + first.stderr for key in keys)
    second = subprocess.run(invocation, cwd=checkout, env=env, capture_output=True, text=True, timeout=20)
    assert second.returncode == 0, second.stderr
    assert original == {name: (checkout / name).read_bytes() for name in files}


def test_env_setup_preserves_existing_values_and_handles_quoted_paths(tmp_path):
    checkout = tmp_path / "project's \\ notes"
    checkout.mkdir()
    (checkout / 'config/auth').mkdir(parents=True)
    (checkout / 'config/auth/apikey.json').write_text(json.dumps({
        'static_keys': {'synthetic-client-key': 'client', 'synthetic-admin-key': 'admin'},
    }))
    path = checkout / '.env'
    original = '# Retain comments\nexport SASY_URL="localhost:10090"\nPROVIDER="line one\nSASY_API_KEY=not-a-setting\nline three"'
    path.write_text(original)
    invocation = ['python3', str(ROOT / 'scripts/setup_local_env.py'), '--url', 'localhost:10089']
    first = subprocess.run(invocation, cwd=checkout, capture_output=True, text=True, timeout=10)
    assert first.returncode == 0, first.stderr
    assert path.read_text().startswith(original + '\n')
    settings = dotenv_values(path)
    assert settings['SASY_URL'] == 'localhost:10090'
    assert settings['SASY_API_KEY'] == 'synthetic-client-key'
    assert settings['TLS_CA_PATH'] == str(checkout / 'certs/server.crt')
    assert 'synthetic-client-key' not in first.stdout + first.stderr
    saved = path.read_bytes()
    # Once supplied, even empty/custom values must be left to the user.
    path.write_bytes(saved + b"SASY_API_KEY=''\n")
    saved = path.read_bytes()
    (checkout / 'config/auth/apikey.json').write_text('{}')
    second = subprocess.run(invocation, cwd=checkout, capture_output=True, text=True, timeout=10)
    assert second.returncode == 0, second.stderr
    assert path.read_bytes() == saved
    assert dotenv_values(path)['SASY_API_KEY'] == ''


def image_serve_commands():
    command = next(line[4:] for line in (ROOT / 'Dockerfile').read_text().splitlines()
                   if line.startswith('CMD '))
    args = json.loads(command)
    assert args[:2] == ['/bin/sh', '-c']
    script = args[2]
    # Each `exec` starts one serve command; it ends at the next `;` or the end.
    serves = [shlex.split(part.split(';', 1)[0]) for part in script.split('exec ')[1:]]
    return script, serves


def test_image_default_serves_with_mounted_config_and_persistent_storage():
    script, (mounted, _) = image_serve_commands()
    assert script.startswith('if [ -e /config/auth/apikey.json ]; then exec sasy serve ')
    assert mounted[:2] == ['sasy', 'serve']
    for flag, path in [('--auth-provider', '/config/auth/apikey.json'),
                       ('--auth-config', '/config/auth_config.yaml'),
                       ('--tls-cert', '/certs/server.crt'), ('--tls-key', '/certs/server.key'),
                       ('--data-dir', '/data/graph')]:
        assert mounted[mounted.index(flag) + 1] == path
    assert '--allow-user-functors' not in mounted


def test_image_default_without_config_initializes_and_serves_local_state():
    script, (_, local) = image_serve_commands()
    assert 'sasy local-init /data/local > /dev/null && exec sasy serve ' in script
    for flag, path in [('--auth-provider', '/data/local/auth/apikey.json'),
                       ('--auth-config', '/data/local/auth_config.yaml'),
                       ('--transforms', '/data/local/transforms.json'),
                       ('--tls-cert', '/data/local/tls/server.crt'),
                       ('--tls-key', '/data/local/tls/server.key'),
                       ('--data-dir', '/data/graph')]:
        assert local[local.index(flag) + 1] == path
    assert '--allow-user-functors' not in local
