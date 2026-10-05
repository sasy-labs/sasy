"""The SDK sends the API key it was given, whole and unchanged.

A key reaches the engine from ``SASY_API_KEY`` (environment or ``.env``),
from ``SasyConfig(SASY_API_KEY=...)`` or from an explicit auth hook. The
SDK never assembles one from parts: ``SASY_API_KEY_SUFFIX`` is an
environment variable it reads nothing from, so setting it neither
supplies a missing key nor changes one that is set.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import textwrap
from pathlib import Path

import pytest
import sasy
import sasy.config as shared_config
from sasy.auth.hooks import APIKeyAuthHook, NoAuthHook, StaticTokenAuthHook
from sasy.config import SasyConfig


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("SASY_ENGINE_PROFILE", raising=False)
    monkeypatch.delenv("SASY_API_KEY", raising=False)
    monkeypatch.delenv("SASY_API_KEY_SUFFIX", raising=False)


def test_environment_full_key_is_sent_unchanged(monkeypatch) -> None:
    key = "client-label-independent_secret-with-hyphens"
    monkeypatch.setenv("SASY_API_KEY", key)
    assert SasyConfig().auth_hook.get_metadata() == [("x-api-key", key)]


def test_full_key_from_dotenv_and_environment_precedence(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("SASY_API_KEY=file-full-key\n", encoding="utf-8")
    assert SasyConfig().auth_hook.get_metadata() == [("x-api-key", "file-full-key")]
    monkeypatch.setenv("SASY_API_KEY", "environment-full-key")
    assert SasyConfig().auth_hook.get_metadata() == [("x-api-key", "environment-full-key")]
    assert SasyConfig(SASY_API_KEY="explicit-full-key").auth_hook.get_metadata() == [("x-api-key", "explicit-full-key")]


@pytest.mark.parametrize("suffix", ["some-suffix", "   ", ""])
def test_suffix_in_the_environment_is_ignored(monkeypatch, suffix) -> None:
    monkeypatch.setenv("SASY_API_KEY_SUFFIX", suffix)
    assert isinstance(SasyConfig().auth_hook, NoAuthHook)
    assert SasyConfig().auth_hook.get_metadata() == []


def test_suffix_in_dotenv_is_ignored(tmp_path) -> None:
    (tmp_path / ".env").write_text("SASY_API_KEY_SUFFIX=some-suffix\n", encoding="utf-8")
    assert SasyConfig().auth_hook.get_metadata() == []


def test_suffix_does_not_modify_explicit_or_environment_full_key(monkeypatch) -> None:
    monkeypatch.setenv("SASY_API_KEY_SUFFIX", "some-suffix")
    monkeypatch.setenv("SASY_API_KEY", "environment-full-key")
    assert SasyConfig().auth_hook.get_metadata() == [("x-api-key", "environment-full-key")]
    assert SasyConfig(SASY_API_KEY="explicit-full-key").auth_hook.get_metadata() == [("x-api-key", "explicit-full-key")]
    assert APIKeyAuthHook("supplied-full-key").get_metadata() == [("x-api-key", "supplied-full-key")]


def test_explicit_auth_hook_wins_over_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("SASY_API_KEY", "environment-full-key")
    hook = StaticTokenAuthHook("explicit-token")
    settings = SasyConfig(auth_hook=hook)
    assert settings.auth_hook is hook
    assert settings.auth_hook.get_metadata() == [("authorization", "Bearer explicit-token")]


def test_missing_or_empty_key_sends_no_credential(monkeypatch) -> None:
    assert SasyConfig().auth_hook.get_metadata() == []
    monkeypatch.setenv("SASY_API_KEY", "environment-full-key")
    monkeypatch.setenv("SASY_API_KEY_SUFFIX", "some-suffix")
    assert SasyConfig(SASY_API_KEY="").auth_hook.get_metadata() == []


def test_no_setting_or_export_assembles_a_key_from_parts() -> None:
    # The SDK sends whole keys only: nothing in its public surface takes a
    # key suffix, or an entity name, and builds a key out of it.
    for module in (sasy, shared_config):
        assert not hasattr(module, "entity_api_key")
        assert not hasattr(module, "SasyApiKeySuffixNotConfigured")
    assert "entity_api_key" not in sasy.__all__
    assert "SasyApiKeySuffixNotConfigured" not in sasy.__all__
    assert "api_key_suffix" not in SasyConfig.model_fields
    assert not hasattr(SasyConfig(), "api_key_suffix")


def test_module_example_uses_the_actual_exported_configuration_api(monkeypatch) -> None:
    # Exercise the real public wrapper, not the similarly named lower-level
    # configure(url=...). Do not activate instrumentation or open a connection.
    monkeypatch.setenv("SASY_URL", "127.0.0.1:1")
    monkeypatch.setenv("SASY_API_KEY", "example-full-key")
    source = textwrap.dedent((sasy.__doc__ or "").split("Usage::", 1)[1].split("The SDK", 1)[0])
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sasy"
        and node.func.attr == "configure"
    ]
    assert len(calls) == 1
    call = calls[0]
    inspect.signature(sasy.configure).bind(**{k.arg: None for k in call.keywords})
    instrumentation_config = importlib.import_module("sasy.instrumentation.config")
    session_config = importlib.import_module("sasy.instrumentation.session")
    monkeypatch.setattr(shared_config, "_config", None)
    monkeypatch.setattr(instrumentation_config, "_config", None)
    monkeypatch.setattr(session_config, "_default_session_id", None)
    eval(
        compile(ast.Expression(call), "<SDK example>", "eval"),
        {"sasy": sasy, "os": os, "APIKeyAuthHook": APIKeyAuthHook},
    )
    assert shared_config.get_config().url == "127.0.0.1:1"
    assert shared_config.get_config().auth_hook.get_metadata() == [("x-api-key", "example-full-key")]
