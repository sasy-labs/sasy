#!/usr/bin/env python3
"""Qualify a packaged full engine in an explicitly supplied, task-owned container.

Requires the Python SDK and grpcio on the host, and Python 3 in the container.
Example: python3 scripts/nix-engine-smoke.py --container nix-test \
  --engine /nix/store/...-sasy/bin/sasy --port 10089 --output ./nix-receipt
Docker must publish that container port to host loopback. No existing engine is
contacted: this helper launches an engine with fresh credentials and private data.
Add --require-sandbox --unrelated-store-path /nix/store/.../fixture to require
non-admin custom-functor admission and constructor/runtime confinement oracles.
The supplied store fixture must be outside the package's admitted closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time

# Executed inside the container. Keep the supervisor alive to own the listener
# and reap the engine. Its private pid receipt allows identity-checked cleanup.
SUPERVISOR = r'''
import ctypes, json, os, pathlib, signal, socket, subprocess, sys, threading
root, engine, port, sandbox, unrelated = sys.argv[1:]
p = pathlib.Path(root)
marker = p / 'private-sentinel'
marker.write_text('synthetic-nix-private-fixture')
marker.chmod(0o600)
if unrelated:
    assert pathlib.Path(unrelated).resolve().is_relative_to('/nix/store') and pathlib.Path(unrelated).is_file()
    with open(unrelated, 'rb') as f: f.read(1)
listener = socket.socket()
listener.bind(('127.0.0.1', 0)); listener.listen()
network_port = listener.getsockname()[1]
# Positive control: the same engine-host namespace can reach this listener.
control = socket.create_connection(('127.0.0.1', network_port), timeout=2)
peer, _ = listener.accept(); peer.close(); control.close()
libc = ctypes.CDLL(None, use_errno=True)
unshare_control = libc.unshare(0) == 0
connections = []
def accept_connections():
    while True:
        try: peer, _ = listener.accept()
        except OSError: return
        connections.append(1); peer.close()
threading.Thread(target=accept_connections, daemon=True).start()
env = dict(os.environ)
for key in ('DISABLE_BWRAP','SASY_EVALUATOR_BWRAP','SASY_SECCOMP_PRE_EXEC_DISABLE',
            'SASY_SKIP_IN_PROC_SANDBOX','SASY_SECCOMP_PRE_EXEC'):
    env.pop(key, None)
env.update(SASY_SOUFFLE_BUILD_CACHE_DIR=str(p / 'cache'),
           XDG_CACHE_HOME=str(p / 'xdg'), SASY_NIX_SMOKE_PRIVATE='synthetic-env-fixture',
           RUST_LOG='info', TMPDIR=str(p / 'tmp'))
(p / 'tmp').mkdir(); (p / 'empty-cwd').mkdir()
command = [engine, 'serve', '--addr', '0.0.0.0:' + port,
           '--auth-provider', str(p / 'provider.json'), '--auth-config', str(p / 'auth.yaml'),
           '--tls-cert', str(p / 'tls/server.pem'), '--tls-key', str(p / 'tls/server-key.pem'),
           '--data-dir', str(p / 'graph'), '--credentials-db', str(p / 'credentials.db')]
if sandbox == 'yes': command.append('--allow-user-functors=sandboxed')
with (p / 'engine.log').open('w') as log:
    child = subprocess.Popen(command, cwd=p / 'empty-cwd', env=env,
                             stdout=log, stderr=log, start_new_session=True)
    def start_time(pid):
        return pathlib.Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    state = dict(pid=child.pid, start_time=start_time(child.pid), network_port=network_port,
                 private_path=str(marker), control_private_readable=marker.read_text().startswith('synthetic'),
                 control_store_readable=bool(unrelated), control_network_connected=True,
                 control_unshare_noop_succeeded=unshare_control)
    (p / 'state.json').write_text(json.dumps(state))
    code = child.wait()
    (p / 'exit.json').write_text(json.dumps(dict(exit_code=code, evaluator_connections=len(connections))))
listener.close()
'''

CLEANUP = r'''
import json, os, pathlib, signal, sys, time
p = pathlib.Path(sys.argv[1])
assert p.parent == pathlib.Path('/tmp') and p.name.startswith('sasy-nix-engine-smoke-')
state_path = p / 'state.json'
if state_path.exists():
    state = json.loads(state_path.read_text()); pid = state['pid']
    def same_process():
        try:
            fields = pathlib.Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
            return fields[19] == state['start_time'] and fields[0] != 'Z'
        except FileNotFoundError: return False
    if same_process():
        os.killpg(pid, signal.SIGTERM)
        for _ in range(100):
            if not same_process(): break
            time.sleep(.1)
        if same_process(): os.killpg(pid, signal.SIGKILL)
for _ in range(30):
    if (p / 'exit.json').exists(): break
    time.sleep(.1)
print((p / 'exit.json').read_text() if (p / 'exit.json').exists() else '{}')
'''



PROVENANCE = r'''
import hashlib, json, os, pathlib, sys
pid, launcher, unrelated = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
exe = os.readlink('/proc/' + pid + '/exe')
env = dict(entry.split(b'=', 1) for entry in pathlib.Path('/proc', pid, 'environ').read_bytes().split(b'\0') if b'=' in entry)
for key, forbidden in ((b'DISABLE_BWRAP', b'1'), (b'SASY_EVALUATOR_BWRAP', b'0'),
                       (b'SASY_SECCOMP_PRE_EXEC_DISABLE', b'1'), (b'SASY_SKIP_IN_PROC_SANDBOX', b'1')):
    assert env.get(key) != forbidden, 'sandbox opt-out found in packaged process'
manifest = env.get(b'SASY_NIX_RUNTIME_MANIFEST', b'').decode()
result = dict(launcher=launcher, launcher_sha256=digest(launcher), executable=exe,
              executable_sha256=digest(exe), manifest=manifest)
if manifest:
    p = pathlib.Path(manifest)
    assert p.stat().st_size <= 1024 * 1024
    data = json.loads(p.read_text())
    roots = data['store_paths']
    assert isinstance(roots, list) and all(isinstance(root, str) for root in roots)
    result.update(manifest_sha256=digest(manifest), store_root_count=len(roots),
                  unrelated_outside_closure=bool(unrelated) and not any(
                      pathlib.Path(unrelated).resolve().is_relative_to(pathlib.Path(root).resolve()) for root in roots))
print(json.dumps(result))
'''

def checked(*args: str, timeout: int = 30) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed ({result.returncode}): {result.stderr[-3000:]}')
    return result.stdout



def check_startup_exit(docker, container_python: str, remote: str, output: Path) -> None:
    exit_text = docker(container_python, '-c',
        'import pathlib,sys; p=pathlib.Path(sys.argv[1]); '
        'print(p.read_text() if p.exists() else "null")', remote + '/exit.json')
    try:
        exited = json.loads(exit_text)
    except json.JSONDecodeError:
        # The supervisor may still be finishing its small exit receipt.
        return
    if exited is not None:
        raise RuntimeError(f'owned engine exited before readiness (exit code {exited.get("exit_code")}); '
                           f'see {output / "engine.log"} after cleanup')


def probe_source(private: str, unrelated: str, port: int) -> str:
    """Return bits, never file contents. Both ctor and callable use same probe."""
    return r'''
#include <souffle/SouffleInterface.h>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sched.h>
#include <arpa/inet.h>
static unsigned long probe() {
  unsigned long bits = 0;
  const char* paths[] = {PRIVATE_PATH, STORE_PATH};
  for (int i = 0; i != 2; ++i) {
    errno = 0; int fd = open(paths[i], O_RDONLY);
    if (fd >= 0) { bits |= (1u << i); close(fd); }
    else if (errno != ENOENT) bits |= (1u << (i + 2));
  }
  if (getenv("SASY_NIX_SMOKE_PRIVATE")) bits |= 16;
  errno = 0;
  if (syscall(SYS_unshare, 0) != -1 || errno != EPERM) bits |= 32;
  int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (fd >= 0) {
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    addr.sin_port = htons(NETWORK_PORT); addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int rc = connect(fd, (sockaddr*)&addr, sizeof(addr));
    // Loopback should fail immediately or remain unconnected. A later success
    // is checked with poll/SO_ERROR below, not inferred from EINPROGRESS.
    if (rc == 0) bits |= 64;
    else if (errno == EINPROGRESS) {
      pollfd pending{fd, POLLOUT, 0};
      if (poll(&pending, 1, 500) > 0) {
        int error = 0; socklen_t len = sizeof(error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &len) == 0 && error == 0) bits |= 64;
      }
    }
    close(fd);
  }
  return bits;
}
static unsigned long before_main;
static bool constructor_ran;
__attribute__((constructor)) static void initialize_probe() { before_main = probe(); constructor_ran = true; }
extern "C" souffle::RamDomain nix_smoke_probe(souffle::RamDomain which) {
  return which == 0 ? (constructor_ran ? before_main : 128) : (which == 1 ? probe() : 12345);
}
'''.replace('#include <arpa/inet.h>', '#include <arpa/inet.h>\n#include <poll.h>').replace(
        'PRIVATE_PATH', json.dumps(private)).replace('STORE_PATH', json.dumps(unrelated)).replace(
        'NETWORK_PORT', str(port))


def main() -> int:
    helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--engine', required=True, help='Absolute packaged launcher path in container')
    parser.add_argument('--container-python', default='python3')
    parser.add_argument('--port', type=int, default=10089, help='Published container port; host port auto-discovered')
    parser.add_argument('--output', type=Path, required=True, help='New host evidence directory')
    parser.add_argument('--require-sandbox', action='store_true')
    parser.add_argument('--unrelated-store-path', help='Readable file outside the admitted Nix closure')
    args = parser.parse_args()
    if not args.engine.startswith('/') or not 1 <= args.port <= 65535:
        parser.error('engine must be absolute and port must be 1..65535')
    if args.require_sandbox and not args.unrelated_store_path:
        parser.error('--require-sandbox requires --unrelated-store-path')
    if args.unrelated_store_path and not re.fullmatch(r'/nix/store/[^/]+/.+', args.unrelated_store_path):
        parser.error('unrelated store fixture must be a file beneath a store output')
    args.output.mkdir(parents=True, exist_ok=False)
    from sasy.proto import policy_engine_pb2 as pe
    from sasy.proto import policy_engine_pb2_grpc as pe_grpc
    from sasy.proto import reference_monitor_pb2 as rm
    from sasy.proto import reference_monitor_pb2_grpc as rm_grpc
    import grpc

    def docker(*command: str, timeout: int = 30) -> str:
        return checked('docker', 'exec', args.container, *command, timeout=timeout)

    mappings = checked('docker', 'port', args.container, f'{args.port}/tcp').splitlines()
    ipv4 = [line for line in mappings if re.fullmatch(r'(127\.0\.0\.1|0\.0\.0\.0):\d+', line)]
    if not ipv4:
        raise RuntimeError('Require a published IPv4 loopback mapping (Docker 0.0.0.0 also reachable via loopback)')
    endpoint = 'localhost:' + ipv4[0].rsplit(':', 1)[1]
    remote = docker(args.container_python, '-c',
                    'import tempfile; print(tempfile.mkdtemp(prefix="sasy-nix-engine-smoke-", dir="/tmp"))').strip()
    assert re.fullmatch('/tmp/sasy-nix-engine-smoke-[a-zA-Z0-9_-]+', remote)
    admin_key, client_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    checks = []
    channel = None
    failure = None
    report = dict(success=False, container=args.container, engine=args.engine, endpoint=endpoint,
                  sandbox_required=args.require_sandbox, checks=checks)
    try:
        # Refuse a port already held by another container process before any RPC.
        docker(args.container_python, '-c',
               'import socket,sys; s=socket.socket(); s.bind(("0.0.0.0", int(sys.argv[1]))); s.close()',
               str(args.port))
        with tempfile.TemporaryDirectory(prefix='sasy-nix-engine-client-') as temporary:
            local = Path(temporary)
            (local / 'provider.json').write_text(json.dumps({'type': 'api_key', 'metadata_key': 'x-api-key',
                'static_keys': {admin_key: 'admin', client_key: 'client'}}))
            (local / 'provider.json').chmod(0o600)
            (local / 'auth.yaml').write_text('trust_domain: sasy.local\ndefault_tenant: default\nentities:\n'
                '  admin:\n    roles: [admin, reference-monitor-user]\n'
                '  client:\n    roles: [reference-monitor-user]\n')
            for name in ('provider.json', 'auth.yaml'):
                checked('docker', 'cp', str(local / name), f'{args.container}:{remote}/{name}')
            docker(args.engine, 'guard-tls', '--output-dir', remote + '/tls', '--entity', 'nix-smoke')
            checked('docker', 'cp', f'{args.container}:{remote}/tls/ca.pem', str(local / 'ca.pem'))
            checked('docker', 'exec', '-d', args.container, args.container_python, '-c', SUPERVISOR,
                    remote, args.engine, str(args.port), 'yes' if args.require_sandbox else 'no',
                    args.unrelated_store_path or '')
            channel = grpc.secure_channel(endpoint, grpc.ssl_channel_credentials(
                root_certificates=(local / 'ca.pem').read_bytes()))
            policy = pe_grpc.PolicyEngineStub(channel)
            monitor = rm_grpc.RMProxyStub(channel)
            admin = [('x-api-key', admin_key)]; client = [('x-api-key', client_key)]
            deadline = time.monotonic() + 240
            while True:
                try:
                    health = policy.Health(pe.HealthRequest(), metadata=admin, timeout=2)
                    assert health.healthy
                    break
                except grpc.RpcError:
                    check_startup_exit(docker, args.container_python, remote, args.output)
                    if time.monotonic() >= deadline: raise RuntimeError('owned engine readiness timeout')
                    time.sleep(.5)
            checks.append('fresh_tls_authenticated_engine')
            for metadata in ([], [('x-api-key', 'synthetic-invalid-key')]):
                try: policy.Health(pe.HealthRequest(), metadata=metadata, timeout=5)
                except grpc.RpcError as error: assert error.code() == grpc.StatusCode.UNAUTHENTICATED
                else: raise AssertionError('missing/invalid key accepted')
            checks.append('missing_and_wrong_keys_rejected')
            state = json.loads(docker(args.container_python, '-c',
                'import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())', remote + '/state.json'))
            provenance = json.loads(docker(args.container_python, '-c', PROVENANCE,
                str(state['pid']), args.engine, args.unrelated_store_path or ''))
            report['provenance'] = provenance
            if args.require_sandbox:
                assert provenance['manifest'], 'packaged Nix manifest missing from engine environment'
                assert provenance['unrelated_outside_closure'], 'fixture is inside the admitted closure'
            source = '''
IsAuthorized(idx) :- Actions(idx, $CallTool("Read", args)), @json_get_str(args, "file_path") = "notes.txt".
IsAuthorized(idx) :- Actions(idx, action), IsTool(action, "Write").
Unauthorized(idx) :- Actions(idx, action), IsTool(action, "Write").
'''
            for backend in ('souffle', 'souffle-interpreted'):
                session = 'nix-smoke-' + backend + '-' + secrets.token_hex(8)
                scope = pe.PolicyScope(session=pe.SessionTarget(session_id=session))
                def bind(text: str, functor: str = '', accept: bool = True):
                    response = policy.SetPolicy(pe.SetPolicyRequest(policy_source=text, functor_source=functor,
                        backend=backend, scope=scope), metadata=client, timeout=240)
                    assert response.accepted == accept, response.error_output
                    return response
                def decision(tool='Read', file='notes.txt'):
                    return monitor.CheckToolCall(rm.ToolCallRequest(fn_name=tool,
                        args=json.dumps({'file_path': file}), session_id=session), metadata=client, timeout=30)
                fixture = source + f'\n// Unique cold-cache fixture {session}\n'
                bind(fixture)
                assert decision().authorized
                assert not decision(file='other.txt').authorized
                assert not decision('Write').authorized
                checks.append(backend + ':json_ffi_allow_wrong_path_deny_explicit_deny_overrides')
                ended = policy.EndSession(pe.EndSessionRequest(session_id=session), metadata=client, timeout=30)
                assert ended.was_active, 'expected a running evaluator before teardown'
                # EndSession removes the session pin; the tenant default denies.
                assert not decision().authorized, 'ended session retained its policy binding'
                bind(fixture)
                assert decision().authorized and not decision(file='other.txt').authorized
                assert not decision('Write').authorized
                checks.append(backend + ':session_teardown_and_cached_rebind_recreate_evaluator')
                if args.require_sandbox:
                    assert state['control_private_readable'] and state['control_store_readable']
                    assert state['control_network_connected'] and state['control_unshare_noop_succeeded']
                    custom = probe_source(state['private_path'], args.unrelated_store_path, state['network_port'])
                    probe_policy = '''
.functor nix_smoke_probe(number): number
IsAuthorized(idx) :- Actions(idx, action), IsTool(action, "Read"),
  @nix_smoke_probe(2) = 12345, @nix_smoke_probe(0) = 0, @nix_smoke_probe(1) = 0.
'''
                    bind(probe_policy, custom)
                    assert decision().authorized, 'constructor/runtime confinement probe failed'
                    assert not decision('Write').authorized
                    checks.append(backend + ':non_admin_sandboxed_functor_constructor_and_runtime_isolation')
                    # The same private paths must also be unreadable to the C++ compiler.
                    for sentinel in (state['private_path'], args.unrelated_store_path):
                        response = bind(source, '#include ' + json.dumps(sentinel) + '\n', accept=False)
                        assert 'No such file' in response.error_output or 'file not found' in response.error_output
                    checks.append(backend + ':compiler_private_and_unrelated_store_includes_rejected')
                bind('// No authorization rules.\n')
                assert not decision().authorized
                checks.append(backend + ':replacement_deny_all')
            report['success'] = True
    except BaseException as error:
        failure = (str(error) or type(error).__name__).replace(admin_key, '[redacted]').replace(client_key, '[redacted]')
        report['error'] = failure
    finally:
        if channel is not None: channel.close()
        try:
            exit_state = json.loads(docker(args.container_python, '-c', CLEANUP, remote))
            report['owned_engine_exit'] = exit_state
            if args.require_sandbox and exit_state.get('evaluator_connections') != 0:
                report['success'] = False
                report['error'] = 'listener observed evaluator connection or supervisor receipt missing'
            log = docker(args.container_python, '-c',
                'import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.exists() else "")',
                remote + '/engine.log')
            log = log.replace(admin_key, '[redacted]').replace(client_key, '[redacted]')
            (args.output / 'engine.log').write_text(log)
            report['engine_log_sha256'] = hashlib.sha256(log.encode()).hexdigest()
            docker(args.container_python, '-c',
                'import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); '
                'assert p.parent == pathlib.Path("/tmp") and p.name.startswith("sasy-nix-engine-smoke-"); '
                'shutil.rmtree(p)', remote)
            report['owned_runtime_removed'] = True
        except Exception as error:
            report['success'] = False
            report['cleanup_error'] = str(error).replace(admin_key, '[redacted]').replace(client_key, '[redacted]')
            report['remaining_owned_directory'] = remote
        report['helper_sha256'] = helper_sha256
        (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
