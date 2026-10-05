"""The release store helper validates and verifies before privileged changes."""
import importlib.util
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('prepare_store', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_rehearsal_store.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ROOT = '/nix/store/' + 'a' * 32 + '-runtime'


def call(tmp_path, selection, metadata=None, run=None):
    paths = tmp_path / 'imported-paths'
    paths.write_text(selection)
    metadata = metadata or SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=1000)
    with patch.object(module.os, 'geteuid', return_value=0), \
         patch.object(Path, 'lstat', return_value=metadata), \
         patch.object(Path, 'resolve', lambda self, **_: self), \
         patch.object(module.subprocess, 'run', side_effect=run) as commands:
        module.prepare(paths, '/nix/nix-store')
        return commands.call_args_list


@pytest.mark.parametrize('selection', ['', ROOT + '\n' + ROOT, '/nix/store/../etc', ROOT + '/bin'])
def test_invalid_selection_never_mutates(tmp_path, selection):
    with patch.object(module.subprocess, 'run') as commands:
        with pytest.raises(SystemExit):
            call(tmp_path, selection)
        commands.assert_not_called()


def test_symlink_root_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match='regular file or directory'):
        call(tmp_path, ROOT, SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_uid=0))


def test_contents_are_verified_before_and_after_metadata_changes(tmp_path):
    commands = call(tmp_path, ROOT)
    assert [c.args[0] for c in commands] == [
        ['/nix/nix-store', '--verify-path', ROOT],
        ['/usr/bin/chown', '-R', '--no-dereference', 'root:root', ROOT],
        ['/usr/bin/chmod', '-R', 'a-w', ROOT],
        ['/nix/nix-store', '--verify-path', ROOT],
    ]
    assert commands[0].kwargs['env']['NIX_REMOTE'] == 'local'
    assert all(c.kwargs['check'] is True for c in commands)


def test_bad_contents_stop_before_metadata_changes(tmp_path):
    commands = []
    def reject(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)
    with pytest.raises(subprocess.CalledProcessError):
        call(tmp_path, ROOT, run=reject)
    assert commands == [['/nix/nix-store', '--verify-path', ROOT]]


def test_later_verification_batch_failure_prevents_all_mutations(tmp_path):
    roots = [ROOT + str(i) for i in range(33)]
    commands = []
    def fail_second_batch(command, **kwargs):
        commands.append(command)
        if len(commands) == 2:
            raise subprocess.CalledProcessError(1, command)
    with pytest.raises(subprocess.CalledProcessError):
        call(tmp_path, '\n'.join(roots), run=fail_second_batch)
    assert commands == [
        ['/nix/nix-store', '--verify-path', *roots[:32]],
        ['/nix/nix-store', '--verify-path', roots[32]],
    ]
