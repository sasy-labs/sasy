"""Cleanup must never signal a recycled PID or an unrelated directory."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import signal
import tempfile
import unittest
from unittest.mock import patch

HELPER = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/nix-engine-smoke.py'))


class CleanupOwnershipTests(unittest.TestCase):
    def run_cleanup(self, *, same_identity):
        with tempfile.TemporaryDirectory(prefix='sasy-nix-engine-smoke-', dir='/tmp') as directory:
            root = Path(directory)
            (root / 'state.json').write_text(json.dumps({'pid': 123456789, 'start_time': '111'}))
            (root / 'exit.json').write_text('{"evaluator_connections": 0}')
            original = Path.read_text
            terminated = False

            def read_text(path, *args, **kwargs):
                if str(path) == '/proc/123456789/stat':
                    fields = ['Z' if terminated else 'S'] + ['0'] * 18 + ['111' if same_identity else '222']
                    return '123456789 (name with ) parentheses) ' + ' '.join(fields)
                return original(path, *args, **kwargs)

            def terminate(pid, sig):
                nonlocal terminated
                self.assertEqual(pid, 123456789)
                self.assertEqual(sig, signal.SIGTERM)
                terminated = True

            with patch.object(Path, 'read_text', read_text), patch('sys.argv', ['cleanup', directory]), \
                 patch('os.killpg', side_effect=terminate) as kill, patch('time.sleep'), redirect_stdout(io.StringIO()):
                exec(HELPER['CLEANUP'], {})
            return kill.call_count

    def test_recycled_pid_is_never_signalled(self):
        self.assertEqual(self.run_cleanup(same_identity=False), 0)

    def test_owned_process_group_is_stopped_once(self):
        self.assertEqual(self.run_cleanup(same_identity=True), 1)

    def test_unrelated_directory_is_rejected_before_signalling(self):
        with patch('sys.argv', ['cleanup', '/tmp/unrelated']), patch('os.killpg') as kill:
            with self.assertRaises(AssertionError):
                exec(HELPER['CLEANUP'], {})
            kill.assert_not_called()


class StartupExitTests(unittest.TestCase):
    def test_running_engine_keeps_waiting(self):
        HELPER['check_startup_exit'](lambda *args: 'null', 'python3', '/tmp/owned', Path('evidence'))

    def test_engine_exit_fails_before_timeout_with_log_location(self):
        for status in (0, 1):
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, f'exit code {status}.*evidence/engine.log'):
                    HELPER['check_startup_exit'](lambda *args: json.dumps({'exit_code': status}),
                                               'python3', '/tmp/owned', Path('evidence'))

    def test_incomplete_exit_receipt_is_retried(self):
        HELPER['check_startup_exit'](lambda *args: '{', 'python3', '/tmp/owned', Path('evidence'))


if __name__ == '__main__':
    unittest.main()
