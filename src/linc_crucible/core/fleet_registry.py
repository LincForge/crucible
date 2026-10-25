"""Fleet registry — manages the Crucible device fleet.

Stores device information in ``~/.linc/crucible/fleet.yaml`` (legacy
``~/.crucible/fleet.yaml`` read if only it exists — see
:mod:`linc_crucible.core.state_paths`) and supports auto-discovery via the
nerve HTTP API.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
import yaml

from .._bounded_read import _MAX_REGISTRY_BYTES, read_capped
from .deployer import _device_is_personal
from .device import DeviceType, FleetDevice, Platform
from .device_policy import load_device_policy
from .nerve_client import NerveClient
from .state_paths import ensure_private_dir, state_path

logger = structlog.get_logger().bind(domain="build_pipeline")


def _default_fleet_path() -> Path:
    """Return the default fleet registry path, honouring ``LINC_HOME``.

    Returns:
        ``<state root>/fleet.yaml`` via :func:`state_paths.state_path` —
        canonical ``$LINC_HOME/crucible/fleet.yaml`` (default
        ``~/.linc/crucible/fleet.yaml``), legacy ``~/.crucible/fleet.yaml``
        when only it exists. Re-evaluated on every call so that
        ``os.environ`` monkeypatching in tests takes effect.
    """
    return state_path("fleet.yaml")


def _safe_parse_platform(raw_platform: str) -> Platform:
    try:
        return Platform(raw_platform)
    except ValueError:
        logger.warning(
            "invalid_device_platform",
            value=raw_platform,
            fallback="android",
        )
        return Platform.ANDROID


def _safe_parse_device_type(raw_device_type: str) -> DeviceType:
    try:
        return DeviceType(raw_device_type)
    except ValueError:
        logger.warning(
            "invalid_device_type",
            value=raw_device_type,
            fallback="phone",
        )
        return DeviceType.PHONE


class FleetRegistry:
    """CRUD registry for the Crucible device fleet.

    Devices are persisted in a YAML file and can be auto-discovered
    via the nerve HTTP API.
    """

    def __init__(
        self,
        fleet_path: Path | None = None,
        nerve_client: NerveClient | None = None,
    ) -> None:
        env_path = os.environ.get("CRUCIBLE_FLEET_FILE")
        if fleet_path:
            self._path = fleet_path
        elif env_path:
            self._path = Path(env_path)
        else:
            self._path = _default_fleet_path()

        self._nerve = nerve_client or NerveClient()
        self._devices: dict[str, FleetDevice] = {}
        self.last_discover_skipped_personal: int = 0
        self._load()

    def _load(self) -> None:
        """Load fleet from YAML file."""
        self._devices = {}

        if not self._path.is_file():
            return

        try:
            data = yaml.safe_load(read_capped(self._path, _MAX_REGISTRY_BYTES))
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("fleet_file_read_failed", path=str(self._path), error=str(exc))
            return

        if not isinstance(data, dict):
            return

        fleet = data.get("fleet", {})
        if not isinstance(fleet, dict):
            return

        device_list = fleet.get("devices", [])
        if not isinstance(device_list, list):
            return

        for entry in device_list:
            if not isinstance(entry, dict):
                continue
            try:
                device = FleetDevice.from_dict(entry)
                if device.serial:
                    self._devices[device.serial] = device
            except (ValueError, KeyError) as exc:
                logger.warning("invalid_device_entry", error=str(exc))

    def _save(self) -> None:
        """Write fleet to YAML file."""
        ensure_private_dir(self._path.parent)

        device_list = [d.to_dict() for d in self._devices.values()]
        data = {"fleet": {"devices": device_list}}

        with open(self._path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def list_devices(
        self,
        platform: Platform | None = None,
        device_type: DeviceType | None = None,
    ) -> list[FleetDevice]:
        """List registered devices, optionally filtered."""
        return [
            d
            for d in self._devices.values()
            if d.matches(platform=platform, device_type=device_type)
        ]

    def get(self, serial: str) -> FleetDevice | None:
        """Get a device by serial."""
        return self._devices.get(serial)

    def add(self, device: FleetDevice) -> FleetDevice:
        """Add or update a device in the fleet."""
        self._devices[device.serial] = device
        self._save()
        return device

    def remove(self, serial: str) -> bool:
        """Remove a device by serial. Returns True if found."""
        if serial in self._devices:
            del self._devices[serial]
            self._save()
            return True
        return False

    def discover(self) -> list[FleetDevice]:
        """Auto-discover devices via the nerve HTTP API.

        Merges discovered devices with existing fleet, preserving nicknames
        and other user-set fields. Returns list of newly discovered devices.
        """
        new_devices = self._discover_via_nerve()

        if new_devices:
            self._save()

        return new_devices

    def _discover_non_personal(self) -> list[dict]:
        """Return non-personal device dicts from nerve, or [] when policy is unreadable."""
        self.last_discover_skipped_personal = 0
        policy = load_device_policy()
        if policy.state not in ("loaded", "absent"):
            logger.warning("fleet_discover_policy_unreadable", source=policy.source)
            return []
        filtered: list[dict] = []
        for d in self._nerve.discover():
            if _device_is_personal(d, policy):
                self.last_discover_skipped_personal += 1
                continue
            filtered.append(d)
        return filtered

    def _discover_via_nerve(self) -> list[FleetDevice]:
        """Discover devices via nerve HTTP API."""
        raw_devices = self._discover_non_personal()
        new_devices: list[FleetDevice] = []
        for d in raw_devices:
            serial = d.get("serial", "")
            if not serial:
                continue
            if serial in self._devices:
                self._devices[serial].state = "online"
                continue
            platform = _safe_parse_platform(d.get("platform", "android"))
            device_type = _safe_parse_device_type(d.get("device_type", "phone"))

            device = FleetDevice(
                serial=serial,
                platform=platform,
                device_type=device_type,
                model=d.get("model", ""),
                os_version=str(d.get("os_version", "")),
                state="online",
                is_simulator=device_type in (DeviceType.EMULATOR, DeviceType.SIMULATOR),
            )
            self._devices[serial] = device
            new_devices.append(device)
        return new_devices
