"""Device data models for the Crucible fleet."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

# Canonical enums from device-kit — re-exported for backward compat
from linc_device_kit.models import DeviceType, Platform

logger = structlog.get_logger().bind(domain="build_pipeline")


_SIMULATOR_UDID_RE = re.compile(
    r"\A[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)


def is_simulator_udid(udid: str) -> bool:
    """Heuristic: simulator UDIDs are UUID-format (with dashes).

    Physical iOS device UDIDs are 40-char hex without dashes.
    Used as a fallback when the discovery source is unknown.
    """
    return bool(_SIMULATOR_UDID_RE.fullmatch(udid))


def _parse_enum[T](enum_cls: Any, val: Any, default: T, warn_type: str) -> T:
    try:
        return enum_cls(val)
    except ValueError:
        logger.warning(
            warn_type,
            value=val,
            fallback=str(default),
        )
        return default


@dataclass
class FleetDevice:
    """A device in the Crucible fleet.

    Attributes:
        serial: Unique device identifier (ADB serial or UDID).
        platform: Android or iOS.
        device_type: Form factor.
        model: Human-readable model name.
        os_version: OS version string.
        nickname: Optional friendly name.
        state: Connection state (online/offline).
        last_seen: ISO 8601 timestamp of last contact.
        is_simulator: Whether this is a simulator/emulator (vs physical).
            Defaults to ``True`` for backward compatibility with existing
            fleet.yaml files that lack this field.
    """

    serial: str
    platform: Platform
    device_type: DeviceType = DeviceType.PHONE
    model: str = ""
    os_version: str = ""
    nickname: str = ""
    state: str = "online"
    last_seen: str = ""
    is_simulator: bool = True

    def matches(
        self,
        serial: str | None = None,
        platform: Platform | None = None,
        device_type: DeviceType | None = None,
        nickname: str | None = None,
    ) -> bool:
        """Check if this device matches the given filters.

        All provided filters must match. Omitted filters are ignored.
        """
        if serial is not None and self.serial != serial:
            return False
        if platform is not None and self.platform != platform:
            return False
        if device_type is not None and self.device_type != device_type:
            return False
        return not (nickname is not None and self.nickname != nickname)

    def to_dict(self) -> dict:
        """Serialize to a dictionary for YAML persistence."""
        d = {
            "serial": self.serial,
            "platform": self.platform.value,
            "device_type": self.device_type.value,
        }
        if self.model:
            d["model"] = self.model
        if self.os_version:
            d["os_version"] = self.os_version
        if self.nickname:
            d["nickname"] = self.nickname
        d["state"] = self.state
        if self.last_seen:
            d["last_seen"] = self.last_seen
        d["is_simulator"] = self.is_simulator
        return d

    @classmethod
    def from_dict(cls, data: dict) -> FleetDevice:
        """Deserialize from a YAML dictionary."""
        device_type = _parse_enum(
            DeviceType,
            data.get("device_type", "phone"),
            DeviceType.PHONE,
            "invalid_device_type",
        )
        platform = _parse_enum(
            Platform,
            data.get("platform", "android"),
            Platform.ANDROID,
            "invalid_platform",
        )

        return cls(
            serial=data.get("serial", ""),
            platform=platform,
            device_type=device_type,
            model=data.get("model", ""),
            os_version=str(data.get("os_version", "")),
            nickname=data.get("nickname", ""),
            state=data.get("state", "online"),
            last_seen=data.get("last_seen", ""),
            is_simulator=data.get("is_simulator", True),
        )
