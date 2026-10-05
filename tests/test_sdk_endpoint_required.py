"""The Python SDK must not ship a default endpoint.

An unconfigured SDK sends nothing anywhere: with no ``SASY_URL`` (and no
``sasy.configure(url=...)``) every call that would open a connection
raises instead of contacting a host the caller never chose.
"""

from pathlib import Path

import pytest
from sasy.config import SasyConfig, SasyEndpointNotConfigured

_ENDPOINT_VARS = ("SASY_URL",)


@pytest.fixture
def unset_endpoints(monkeypatch, tmp_path):
    """A config built with no endpoint, dotenv file, or provisioned profile."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("SASY_ENGINE_PROFILE", raising=False)
    for var in _ENDPOINT_VARS:
        monkeypatch.delenv(var, raising=False)
    return SasyConfig(_env_file=None)


def test_no_default_endpoints(unset_endpoints):
    cfg = unset_endpoints
    assert cfg.url == ""


@pytest.mark.parametrize(
    "accessor,var",
    [
        ("require_url", "SASY_URL"),
    ],
)
def test_missing_endpoint_raises_naming_the_variable(unset_endpoints, accessor, var):
    with pytest.raises(SasyEndpointNotConfigured) as excinfo:
        getattr(unset_endpoints, accessor)()
    message = str(excinfo.value)
    assert var in message
    assert "localhost:10089" in message


def test_channel_creation_refuses_without_an_endpoint(unset_endpoints):
    with pytest.raises(SasyEndpointNotConfigured):
        unset_endpoints.create_channel()


def test_configured_endpoint_is_used(unset_endpoints):
    unset_endpoints.url = "localhost:10089"
    assert unset_endpoints.require_url() == "localhost:10089"
