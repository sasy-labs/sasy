"""Progress display must not interfere with sessions or redirected output."""
import importlib.util
import io
from contextlib import contextmanager
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "example_narration_progress", Path(__file__).parents[1] / "examples/narration.py")
narration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(narration)


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_spinner_clears_on_success_error_and_interrupt(error):
    output = Terminal()
    try:
        with narration.compiling_policy(output=output):
            assert "Compiling policy" in output.getvalue()
            if error:
                raise error()
    except (RuntimeError, KeyboardInterrupt) as caught:
        assert type(caught) is error
    assert output.getvalue().endswith("\r" + " " * 21 + "\r")


@pytest.mark.parametrize("terminal,enabled", [(False, True), (True, False)])
def test_spinner_keeps_noninteractive_and_quiet_output_clean(terminal, enabled):
    output = Terminal() if terminal else io.StringIO()
    with narration.compiling_policy(enabled=enabled, output=output):
        pass
    assert output.getvalue() == ""


def test_spinner_stops_before_run_and_session_cleanup_survives_failure(monkeypatch):
    output = Terminal()
    monkeypatch.setattr(narration.sys, "__stderr__", output)
    cleaned = []

    @contextmanager
    def session(**settings):
        assert "Compiling policy" in output.getvalue()
        try:
            yield settings
        finally:
            cleaned.append(True)

    with pytest.raises(RuntimeError, match="agent failure"):
        with narration.policy_session(session, policy="test") as active:
            assert active == {"policy": "test"}
            size = len(output.getvalue())
            assert output.getvalue().endswith("\r" + " " * 21 + "\r")
            raise RuntimeError("agent failure")
    assert cleaned == [True]
    assert len(output.getvalue()) == size


def test_pacing_only_applies_to_terminal_event_lines(monkeypatch):
    delays = []
    monkeypatch.setattr(narration.time, "sleep", delays.append)
    monkeypatch.setattr(narration.sys, "stdout", Terminal())
    narrator = narration.Narrator(pause=False)
    narrator.line("user", "request")
    narrator.line("SASY", "allowed")
    assert delays == [0.05, 0.05]
    monkeypatch.setattr(narration.sys, "stdout", io.StringIO())
    narration.Narrator(pause=False).line("user", "request")
    assert delays == [0.05, 0.05]
