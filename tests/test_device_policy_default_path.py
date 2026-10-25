"""[contract] The default fleet registry is neutral: no private LINC path in the package.

The public package must never default to a path inside a private LINC checkout. With
``LINC_DEVICE_FLEET_FILE`` unset the registry resolves under ``$LINC_HOME`` (default
``~/.linc``); a stranger without that file runs in "absent" mode (REVIEW B4a addendum).
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from linc_crucible.core import device_policy, ios_signing
from linc_crucible.core.device_policy import _fleet_registry_path, load_device_policy

_PRIVATE_FRAGMENTS = ("projects/LINC", "memory/hot")
_PRIVATE_SEGMENTS = frozenset({"LINC", "memory", "hot"})


@pytest.fixture(autouse=True)
def _capture_structlog() -> Iterator[None]:
    """Record structlog events instead of printing them to an import-time-bound stream.

    mutmut re-runs ``pytest.main()`` in one process; structlog's stream bound at
    ``linc_crucible.cli`` import is the first session's capture stream, closed by the
    second, so a policy warning raised "I/O operation on closed file".
    """
    with structlog.testing.capture_logs():
        yield


@pytest.fixture
def _no_fleet_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LINC_DEVICE_FLEET_FILE", raising=False)
    monkeypatch.delenv("LINC_HOME", raising=False)
    return home


def _string_constants(module: object) -> set[str]:
    tree = ast.parse(inspect.getsource(module))  # type: ignore[arg-type]
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


@pytest.mark.parametrize("module", [device_policy, ios_signing], ids=lambda m: m.__name__)
def test_contract_policy_module_source_names_no_private_linc_path(module: object) -> None:
    """[contract] Neither the joined nor the segment-by-segment private path is in source."""
    source = inspect.getsource(module)  # type: ignore[arg-type]
    for fragment in _PRIVATE_FRAGMENTS:
        assert fragment not in source, f"{fragment!r} in {module.__name__}"  # type: ignore[attr-defined]
    assert _PRIVATE_SEGMENTS.isdisjoint(_string_constants(module))


def test_contract_default_resolves_under_linc_home(
    _no_fleet_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] ``$LINC_HOME/device-fleet.yaml`` when set, else ``~/.linc/device-fleet.yaml``."""
    assert _fleet_registry_path() == _no_fleet_env / ".linc" / "device-fleet.yaml"

    linc_home = tmp_path / "linc_home"
    monkeypatch.setenv("LINC_HOME", str(linc_home))
    resolved = _fleet_registry_path()
    assert resolved == linc_home / "device-fleet.yaml"
    for fragment in _PRIVATE_FRAGMENTS:
        assert fragment not in resolved.as_posix()


def test_contract_private_checkout_file_is_never_read_by_default(_no_fleet_env: Path) -> None:
    """[contract] A fleet file at the old private location does not leak into the default."""
    private = _no_fleet_env.joinpath("projects", "LINC", "memory", "hot", "device-fleet.yaml")
    private.parent.mkdir(parents=True)
    private.write_text('devices:\n  - serial: "PRIVATEWATCH01"\n    personal: true\n')

    policy = load_device_policy()

    assert policy.state == "absent"
    assert policy.source == str(_no_fleet_env / ".linc" / "device-fleet.yaml")
    assert policy.is_personal("PRIVATEWATCH01") is False
    assert policy.refusal("PRIVATEWATCH01") is None


def test_contract_override_env_still_wins(
    _no_fleet_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] ``LINC_DEVICE_FLEET_FILE`` overrides the ``$LINC_HOME`` default."""
    monkeypatch.setenv("LINC_HOME", str(tmp_path / "linc_home"))
    fleet = tmp_path / "elsewhere" / "fleet.yaml"
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(fleet))
    assert _fleet_registry_path() == fleet


def test_empty_override_env_falls_back_to_default(
    _no_fleet_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hostile input: an empty ``LINC_DEVICE_FLEET_FILE`` is ignored, not read as ``.``."""
    monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", "")
    assert _fleet_registry_path() == _no_fleet_env / ".linc" / "device-fleet.yaml"
