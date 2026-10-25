"""Tests for iOS code-signing config resolution."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
import structlog

from linc_crucible.core import ios_signing


@pytest.fixture(autouse=True)
def _capture_structlog() -> Iterator[None]:
    """Record structlog events instead of printing them to an import-time-bound stream.

    Importing ``linc_crucible.cli`` points structlog at the then-current ``sys.stderr``,
    a pytest capture stream. mutmut re-runs ``pytest.main()`` in one process; the first
    session closes that stream, so any later log call raised "I/O operation on closed
    file". Mirrors ``captured_logs`` in ``tests/test_crucible_bvt_mcp.py``.
    """
    with structlog.testing.capture_logs():
        yield


def _write_fleet(tmp_path, content: str):
    fleet = tmp_path / "device-fleet.yaml"
    fleet.write_text(content)
    return fleet


class TestResolveTeamId:
    def test_reads_team_from_fleet_file(self, tmp_path, monkeypatch) -> None:
        fleet = _write_fleet(tmp_path, "ios_signing:\n  team_id: TEAMID0000\n")
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(fleet))
        monkeypatch.delenv("CRUCIBLE_IOS_TEAM_ID", raising=False)
        assert ios_signing.resolve_team_id() == "TEAMID0000"

    def test_env_var_wins_and_logs_info(self, tmp_path, monkeypatch) -> None:
        fleet = _write_fleet(tmp_path, "ios_signing:\n  team_id: FILETEAM01\n")
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(fleet))
        monkeypatch.setenv("CRUCIBLE_IOS_TEAM_ID", "ENVTEAM999")
        with patch("linc_crucible.core.ios_signing.logger") as mock_logger:
            assert ios_signing.resolve_team_id() == "ENVTEAM999"
            mock_logger.info.assert_called_once()

    def test_env_var_without_fleet_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(tmp_path / "absent.yaml"))
        monkeypatch.setenv("CRUCIBLE_IOS_TEAM_ID", "ENVTEAM999")
        assert ios_signing.resolve_team_id() == "ENVTEAM999"

    def test_missing_everywhere_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(tmp_path / "absent.yaml"))
        monkeypatch.delenv("CRUCIBLE_IOS_TEAM_ID", raising=False)
        assert ios_signing.resolve_team_id() is None

    def test_read_fleet_team_missing_file_returns_none_without_warning(
        self, tmp_path, monkeypatch
    ) -> None:
        """[contract] Pin: _read_fleet_team returns None silently when the fleet file is absent."""
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(tmp_path / "absent.yaml"))
        with patch("linc_crucible.core.ios_signing.logger") as mock_logger:
            assert ios_signing._read_fleet_team() is None
            mock_logger.warning.assert_not_called()

    def test_fleet_file_without_signing_block(self, tmp_path, monkeypatch) -> None:
        fleet = _write_fleet(tmp_path, "version: 1\ndevices: []\n")
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(fleet))
        monkeypatch.delenv("CRUCIBLE_IOS_TEAM_ID", raising=False)
        assert ios_signing.resolve_team_id() is None

    def test_malformed_yaml_returns_none(self, tmp_path, monkeypatch) -> None:
        fleet = _write_fleet(tmp_path, "ios_signing: [unclosed\n")
        monkeypatch.setenv("LINC_DEVICE_FLEET_FILE", str(fleet))
        monkeypatch.delenv("CRUCIBLE_IOS_TEAM_ID", raising=False)
        assert ios_signing.resolve_team_id() is None

    def test_identity_constant(self) -> None:
        assert ios_signing.CODE_SIGN_IDENTITY == "Apple Development"
