"""Local connection bundles, settings isolation, and hostile profile paths."""

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
from sasy import cli, config
from sasy.auth.hooks import NoAuthHook, StaticTokenAuthHook
from sasy.local_profiles import ProfileError, load_profile, save_profile, selected_profile


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in ("SASY_URL", "SASY_API_KEY", "TLS_CA_PATH", "TLS_CERT_PATH", "TLS_KEY_PATH", "SASY_ENGINE_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_config", None)
    return home


def provision(profile="local", **changes):
    data = {
        "url": "localhost:10089",
        "api_key": "local-secret",
        "name": "sasy-engine",
        "image": "sasy:v1",
        "volume": "sasy-engine-data",
        "port": "10089",
        **changes,
    }
    return save_profile(profile, data, "synthetic CA")


def test_connection_works_across_project_directories(tmp_path, isolated_home, monkeypatch):
    data = provision()
    for name in ("first", "second"):
        project = tmp_path / name
        project.mkdir()
        monkeypatch.chdir(project)
        cfg = config.SasyConfig()
        assert (cfg.url, cfg.ca_path, cfg.get_metadata()) == (
            data["url"],
            data["ca_path"],
            [("x-api-key", "local-secret")],
        )
        assert list(project.iterdir()) == []
    assert selected_profile() == "local"
    for path in (isolated_home / ".sasy").rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


def test_directory_alone_is_not_a_host_choice(isolated_home):
    (isolated_home / ".sasy").mkdir(mode=0o700)
    cfg = config.SasyConfig()
    assert cfg.url == ""
    with pytest.raises(config.SasyEndpointNotConfigured):
        cfg.require_url()


def test_named_profile_selection_in_environment_and_dotenv(tmp_path, monkeypatch):
    provision("first", api_key="first-key")
    provision("second", api_key="second-key")
    (tmp_path / ".env").write_text("SASY_ENGINE_PROFILE=first\n")
    assert config.SasyConfig().api_key == "first-key"
    monkeypatch.setenv("SASY_ENGINE_PROFILE", "second")
    assert config.SasyConfig().api_key == "second-key"
    assert config.SasyConfig(SASY_ENGINE_PROFILE="first").api_key == "first-key"


@pytest.mark.parametrize("source", ["constructor", "env", "dotenv"])
def test_deployment_url_never_acquires_local_credentials(source, tmp_path, monkeypatch):
    provision()
    kwargs = {}
    if source == "constructor":
        kwargs["SASY_URL"] = "remote.example:443"
    elif source == "env":
        monkeypatch.setenv("SASY_URL", "remote.example:443")
    else:
        (tmp_path / ".env").write_text("SASY_URL=remote.example:443\n")
    cfg = config.SasyConfig(**kwargs)
    assert cfg.url == "remote.example:443"
    assert cfg.api_key == "" and cfg.ca_path is None
    assert isinstance(cfg.auth_hook, NoAuthHook)


def test_environment_and_dotenv_override_profile_settings(tmp_path, monkeypatch):
    provision()
    (tmp_path / ".env").write_text("SASY_API_KEY=project-key\nTLS_CA_PATH=project-ca\n")
    monkeypatch.setenv("SASY_API_KEY", "env-key")
    assert config.SasyConfig().get_metadata() == [("x-api-key", "env-key")]
    assert config.SasyConfig().ca_path == "project-ca"
    assert config.SasyConfig(SASY_API_KEY="explicit-key").api_key == "explicit-key"


def test_configure_remote_clears_profile_bundle():
    provision()
    assert config.get_config().api_key == "local-secret"
    cfg = config.configure(url="remote.example:443")
    assert cfg.api_key == "" and cfg.ca_path is None and cfg.get_metadata() == []


def test_configure_remote_preserves_explicit_hook_and_ca():
    provision()
    hook = StaticTokenAuthHook("custom-token")
    config._config = config.SasyConfig(auth_hook=hook, TLS_CA_PATH="custom-ca")
    cfg = config.configure(url="remote.example:443")
    assert cfg.auth_hook is hook and cfg.ca_path == "custom-ca"


def test_configure_first_call_remote_ignores_unrelated_local_selection(isolated_home):
    (isolated_home / ".sasy").symlink_to(isolated_home)
    assert config.configure(url="remote.example:443").url == "remote.example:443"


def test_explicit_missing_profile_is_clear(monkeypatch):
    monkeypatch.setenv("SASY_ENGINE_PROFILE", "absent")
    with pytest.raises(ProfileError, match="absent.*missing"):
        config.SasyConfig()
    assert cli.main(["engine", "stop"]) == 1


@pytest.mark.parametrize("name", ["../outside", "a/b", "", ".", "-option", "a" * 65])
def test_unsafe_names_rejected(name):
    with pytest.raises(ProfileError):
        provision(name)


@pytest.mark.parametrize("component", ["root", "profiles", "profile", "json", "ca", "selection"])
def test_symlink_paths_never_read_or_overwrite_outside(component, tmp_path, isolated_home):
    saved = provision()
    root = isolated_home / ".sasy"
    paths = {
        "root": root,
        "profiles": root / "profiles",
        "profile": root / "profiles/local",
        "json": root / "profiles/local/profile.json",
        "ca": saved["ca_path"],
        "selection": root / "selected-profile",
    }
    path = Path(paths[component])
    moved = path.with_name(path.name + "-saved")
    path.rename(moved)
    outside = tmp_path / "outside"
    outside.write_text("do not change")
    outside.chmod(0o600)
    path.symlink_to(outside)
    with pytest.raises(ProfileError):
        load_profile()
    with pytest.raises(ProfileError):
        provision()
    assert outside.read_text() == "do not change"


def test_fifo_and_hardlink_profiles_rejected(isolated_home):
    provision()
    path = isolated_home / ".sasy/selected-profile"
    path.unlink()
    os.mkfifo(path, 0o600)
    with pytest.raises(ProfileError):
        selected_profile()
    path.unlink()
    source = isolated_home / "other"
    source.write_text("local\n")
    source.chmod(0o600)
    os.link(source, path)
    with pytest.raises(ProfileError):
        selected_profile()


def test_world_readable_profile_is_rejected(isolated_home):
    provision()
    path = isolated_home / ".sasy/profiles/local/profile.json"
    path.chmod(0o644)
    with pytest.raises(ProfileError):
        load_profile()


def test_publish_failure_keeps_previous_complete_connection(monkeypatch):
    first = provision()
    original = os.replace

    def failing_replace(source, dest, **kwargs):
        if dest == "profile.json":
            raise OSError("injected publication failure")
        original(source, dest, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(ProfileError):
        save_profile("local", {**first, "api_key": "new-key"}, "new CA")
    assert load_profile()["api_key"] == "local-secret"
    assert not list(Path(first["ca_path"]).parent.glob(".tmp-*"))


def test_named_cli_uses_saved_metadata_for_stop_and_status(monkeypatch):
    provision("work", name="work-engine", port="12345", volume="work-data")
    calls = []
    monkeypatch.setattr(cli, "stop", lambda name: calls.append(("stop", name)) or 0)
    monkeypatch.setattr(cli, "status", lambda name: calls.append(("status", name)) or 0)
    assert cli.main(["engine", "stop"]) == 0
    assert cli.main(["engine", "status", "--profile", "work"]) == 0
    assert calls == [("stop", "work-engine"), ("status", "work-engine")]


def test_cli_forwards_only_explicit_identity_settings_and_reuses_profile(monkeypatch):
    provision("work", name="work-engine", port="12345", volume="work-data")
    calls = []
    monkeypatch.setattr(cli, "start", lambda **kw: calls.append(kw) or 0)
    assert cli.main(["engine", "start", "--tenant", "team", "--port", "23456"]) == 0
    assert calls[0]["init_args"] == ("--tenant=team",)
    assert calls[0]["explicit"] == {"port"}
    assert calls[0]["port"] == 23456 and calls[0]["name"] == "work-engine"
    assert calls[0]["env_file"] is None and calls[0]["volume"] == "work-data"


def test_running_container_rejects_explicit_mismatch(monkeypatch):
    info = [
        {
            "NetworkSettings": {"Ports": {"10089/tcp": [{"HostIp": "127.0.0.1", "HostPort": "10089"}]}},
            "Config": {"Image": "actual"},
            "Mounts": [{"Destination": "/data", "Type": "volume", "Name": "data"}],
        }
    ]
    monkeypatch.setattr(cli, "_docker", lambda *a: subprocess.CompletedProcess(a, 0, json.dumps(info), ""))
    for field in ("image", "port", "volume"):
        with pytest.raises(cli.EngineError, match=f"different {field}"):
            cli._running_settings("engine", "other", 12345, "other", {field})
    assert cli._running_settings("engine", "other", 12345, "other", set()) == (10089, "actual", "data")


def test_start_publishes_profile_without_project_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_require_daemon", lambda: None)
    monkeypatch.setattr(cli, "_state", lambda name: None)
    monkeypatch.setattr(cli, "_client_settings", lambda *a, **kw: {"api_key": "generated", "ca_cert_pem": "CA"})
    monkeypatch.setattr(cli, "_docker", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_wait_ready", lambda *a: None)
    assert cli.start("image", "engine", 10089, "data", timeout=1) == 0
    assert load_profile()["api_key"] == "generated"
    assert not (tmp_path / ".env").exists() and not (tmp_path / ".sasy").exists()


@pytest.mark.parametrize("component", ["directory", "certificate"])
def test_start_does_not_follow_old_project_ca_symlinks(component, tmp_path, monkeypatch):
    """Regression for the original launcher CA overwrite outside the project."""
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "keep.crt"
    victim.write_text("do not replace")
    managed = project / ".sasy"
    if component == "directory":
        managed.symlink_to(outside, target_is_directory=True)
    else:
        managed.mkdir()
        (managed / "engine-ca.crt").symlink_to(victim)
    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "_require_daemon", lambda: None)
    monkeypatch.setattr(cli, "_state", lambda name: None)
    monkeypatch.setattr(cli, "_client_settings", lambda *a, **kw: {"api_key": "generated", "ca_cert_pem": "CA"})
    monkeypatch.setattr(cli, "_docker", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_wait_ready", lambda *a: None)

    assert cli.start("image", "engine", 10089, "data", project=project) == 0
    assert load_profile()["api_key"] == "generated"
    assert victim.read_text() == "do not replace"
    assert list(outside.iterdir()) == [victim]
    assert not (project / ".env").exists()


@pytest.mark.parametrize("component", ["root", "profiles", "profile", "ca"])
def test_start_rejects_shared_ca_symlinks_without_external_writes(component, tmp_path, isolated_home, monkeypatch):
    saved = provision()
    root = isolated_home / ".sasy"
    path = {
        "root": root,
        "profiles": root / "profiles",
        "profile": root / "profiles/local",
        "ca": Path(saved["ca_path"]),
    }[component]
    path.rename(path.with_name(path.name + "-saved"))
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    victim = outside / "keep.crt"
    victim.write_text("do not replace")
    victim.chmod(0o600)
    path.symlink_to(victim if component == "ca" else outside, target_is_directory=component != "ca")
    monkeypatch.setattr(cli, "_require_daemon", lambda: None)
    monkeypatch.setattr(cli, "_state", lambda name: None)
    monkeypatch.setattr(
        cli, "_client_settings", lambda *a, **kw: {"api_key": "generated", "ca_cert_pem": "synthetic CA"}
    )
    monkeypatch.setattr(cli, "_docker", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_wait_ready", lambda *a: None)

    with pytest.raises(ProfileError):
        cli.start("image", "engine", 10089, "data")
    assert victim.read_text() == "do not replace"
    assert list(outside.iterdir()) == [victim]
    assert path.is_symlink()


def test_start_rejects_host_bind_mounts_before_docker(monkeypatch):
    monkeypatch.setattr(cli, "_require_daemon", lambda: pytest.fail("must not touch Docker"))
    with pytest.raises(cli.EngineError, match="Docker volume"):
        cli.start("image", "engine", 10089, "/host")


def test_env_writer_rejects_hardlinks_and_fifo(tmp_path):
    target = tmp_path / "target"
    target.write_text("unchanged")
    target.chmod(0o644)
    hardlink = tmp_path / "linked.env"
    os.link(target, hardlink)
    with pytest.raises(cli.EngineError, match="unsafe"):
        cli.write_env(hardlink, {"SASY_API_KEY": "secret"})
    assert target.read_text() == "unchanged" and stat.S_IMODE(target.stat().st_mode) == 0o644
    fifo = tmp_path / "fifo.env"
    os.mkfifo(fifo)
    with pytest.raises(cli.EngineError, match="unsafe"):
        cli.write_env(fifo, {"SASY_API_KEY": "secret"})


def test_cli_explicit_empty_profile_is_not_treated_as_default():
    assert cli.main(["engine", "start", "--profile", ""]) == 1


def test_new_named_profile_gets_separate_default_container_and_volume(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "start", lambda **kw: calls.append(kw) or 0)
    assert cli.main(["engine", "start", "--profile", "work"]) == 0
    assert calls[0]["name"] == "sasy-engine-work"
    assert calls[0]["volume"] == "sasy-engine-work-data"


def test_start_writes_only_explicit_env_file_and_reports_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_require_daemon", lambda: None)
    monkeypatch.setattr(cli, "_state", lambda name: None)
    monkeypatch.setattr(cli, "_client_settings", lambda *a, **kw: {"api_key": "generated", "ca_cert_pem": "CA"})
    monkeypatch.setattr(cli, "_docker", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_wait_ready", lambda *a: None)
    target = tmp_path / "custom.env"
    assert cli.start("image", "engine", 10089, "data", env_file=target) == 0
    assert "SASY_API_KEY='generated'" in target.read_text()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    target.write_text("SASY_API_KEY=stale\n")
    with pytest.raises(cli.EngineError, match="conflicting SASY_API_KEY"):
        cli.start("image", "engine", 10089, "data", env_file=target)
    assert target.read_text().startswith("SASY_API_KEY=stale\n")
    assert not (tmp_path / ".env").exists()


def test_local_init_flags_are_after_command_and_output_contains_no_admin_key(monkeypatch):
    calls = []
    settings = {"api_key": "client-key", "ca_cert_pem": "CA", "tenant": "team", "admin_api_key": "must-not-expose"}

    def docker(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps(settings), "")

    monkeypatch.setattr(cli, "_docker", docker)
    client = cli._client_settings("exec", "engine", init_args=("--tenant=team",))
    assert calls == [("exec", "engine", "sasy", "local-init", "/data/local", "--tenant=team")]
    assert client == {"api_key": "client-key", "ca_cert_pem": "CA", "tenant": "team"}


def test_cli_identity_values_starting_with_hyphen_are_forwarded_as_values(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "start", lambda **kw: calls.append(kw) or 0)
    assert cli.main(["engine", "start", "--client-entity=-worker", "--tenant=-team"]) == 0
    assert calls[0]["init_args"] == ("--client-entity=-worker", "--tenant=-team")


@pytest.mark.parametrize("source", ["constructor", "environment", "project"])
def test_explicit_endpoint_ignores_missing_selected_profile(source, tmp_path, monkeypatch):
    monkeypatch.setenv("SASY_ENGINE_PROFILE", "removed-profile")
    if source == "constructor":
        cfg = config.SasyConfig(SASY_URL="explicit.example:443")
    else:
        if source == "environment":
            monkeypatch.setenv("SASY_URL", "explicit.example:443")
        else:
            (tmp_path / ".env").write_text("SASY_URL=explicit.example:443\n")
        cfg = config.SasyConfig()
    assert cfg.url == "explicit.example:443" and cfg.api_key == "" and cfg.ca_path is None
