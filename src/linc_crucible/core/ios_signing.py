"""iOS code-signing config for physical device builds.

The signing team lives in the personal-device registry (``LINC_DEVICE_FLEET_FILE``,
default ``$LINC_HOME/device-fleet.yaml``), not in crucible's own
``~/.linc/crucible/fleet.yaml`` (a runtime device cache). The identity is
hardcoded: "Apple Development" is the only correct value for Debug
device builds with automatic signing.
"""

from __future__ import annotations

import os

import structlog
import yaml

from linc_crucible.core.device_policy import _fleet_registry_path, _probe_fleet_path

logger = structlog.get_logger().bind(domain="build_pipeline")

CODE_SIGN_IDENTITY = "Apple Development"


def _read_fleet_team() -> str | None:
    """Return ``ios_signing.team_id`` from the fleet file, or ``None`` if absent/unreadable."""
    path = _fleet_registry_path()
    if _probe_fleet_path(path) is not None:
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("device_fleet_read_failed", path=str(path), error=str(exc))
        return None
    if not isinstance(data, dict):
        return None
    signing = data.get("ios_signing")
    if not isinstance(signing, dict):
        return None
    team = signing.get("team_id")
    return str(team) if team else None


def resolve_team_id() -> str | None:
    """Resolve the Apple development team for device builds.

    Order: CRUCIBLE_IOS_TEAM_ID env var, then ios_signing.team_id in the
    device fleet file (see ``_fleet_registry_path``). Logs INFO when the env var overrides a
    fleet-file value so config divergence is visible.
    """
    env_team = os.environ.get("CRUCIBLE_IOS_TEAM_ID")
    if env_team:
        file_team = _read_fleet_team()
        if file_team and file_team != env_team:
            logger.info(
                "ios_team_env_override_active",
                env_team=env_team,
                fleet_file_team=file_team,
            )
        return env_team
    return _read_fleet_team()
