"""Tests for device data models."""

import pytest

from linc_crucible.core.device import DeviceType, FleetDevice, Platform, is_simulator_udid


class TestPlatformEnum:
    def test_android(self) -> None:
        assert Platform.ANDROID == "android"
        assert Platform("android") == Platform.ANDROID

    def test_ios(self) -> None:
        assert Platform.IOS == "ios"
        assert Platform("ios") == Platform.IOS

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            Platform("windows")


class TestDeviceTypeEnum:
    def test_all_types(self) -> None:
        assert DeviceType.PHONE == "phone"
        assert DeviceType.TABLET == "tablet"
        assert DeviceType.WATCH == "watch"
        assert DeviceType.TV == "tv"
        assert DeviceType.EMULATOR == "emulator"
        assert DeviceType.SIMULATOR == "simulator"

    def test_from_string(self) -> None:
        assert DeviceType("watch") == DeviceType.WATCH


class TestFleetDevice:
    def test_basic_creation(self) -> None:
        device = FleetDevice(serial="ABC123", platform=Platform.ANDROID)
        assert device.serial == "ABC123"
        assert device.platform == Platform.ANDROID
        assert device.device_type == DeviceType.PHONE
        assert device.model == ""
        assert device.state == "online"

    def test_full_creation(self) -> None:
        device = FleetDevice(
            serial="PERSONALWATCH01",
            platform=Platform.ANDROID,
            device_type=DeviceType.WATCH,
            model="Pixel Watch 2",
            os_version="13",
            nickname="watch-pw2",
            state="online",
            last_seen="2026-02-20T14:30:00Z",
        )
        assert device.model == "Pixel Watch 2"
        assert device.nickname == "watch-pw2"
        assert device.device_type == DeviceType.WATCH

    def test_matches_no_filters(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.ANDROID)
        assert device.matches() is True

    def test_matches_serial(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.ANDROID)
        assert device.matches(serial="ABC") is True
        assert device.matches(serial="XYZ") is False

    def test_matches_platform(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.ANDROID)
        assert device.matches(platform=Platform.ANDROID) is True
        assert device.matches(platform=Platform.IOS) is False

    def test_matches_device_type(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.ANDROID, device_type=DeviceType.WATCH)
        assert device.matches(device_type=DeviceType.WATCH) is True
        assert device.matches(device_type=DeviceType.PHONE) is False

    def test_matches_nickname(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.ANDROID, nickname="my-watch")
        assert device.matches(nickname="my-watch") is True
        assert device.matches(nickname="other") is False

    def test_matches_combined_filters(self) -> None:
        device = FleetDevice(
            serial="ABC",
            platform=Platform.ANDROID,
            device_type=DeviceType.WATCH,
            nickname="pw2",
        )
        assert device.matches(platform=Platform.ANDROID, device_type=DeviceType.WATCH)
        assert not device.matches(platform=Platform.IOS, device_type=DeviceType.WATCH)

    def test_to_dict_minimal(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.ANDROID)
        d = device.to_dict()
        assert d["serial"] == "ABC"
        assert d["platform"] == "android"
        assert d["device_type"] == "phone"
        assert d["state"] == "online"
        assert "nickname" not in d  # Empty nickname omitted

    def test_to_dict_full(self) -> None:
        device = FleetDevice(
            serial="ABC",
            platform=Platform.IOS,
            device_type=DeviceType.SIMULATOR,
            model="iPhone 15",
            os_version="17.2",
            nickname="sim-iphone15",
            state="offline",
            last_seen="2026-02-20T00:00:00Z",
        )
        d = device.to_dict()
        assert d["platform"] == "ios"
        assert d["model"] == "iPhone 15"
        assert d["nickname"] == "sim-iphone15"
        assert d["last_seen"] == "2026-02-20T00:00:00Z"

    def test_from_dict_minimal(self) -> None:
        data = {"serial": "ABC", "platform": "android"}
        device = FleetDevice.from_dict(data)
        assert device.serial == "ABC"
        assert device.platform == Platform.ANDROID
        assert device.device_type == DeviceType.PHONE

    def test_from_dict_invalid_platform(self) -> None:
        data = {"serial": "X", "platform": "windows"}
        device = FleetDevice.from_dict(data)
        assert device.platform == Platform.ANDROID

    def test_from_dict_full(self) -> None:
        data = {
            "serial": "PERSONALWATCH01",
            "platform": "android",
            "device_type": "watch",
            "model": "Pixel Watch 2",
            "os_version": "13",
            "nickname": "watch-pw2",
            "state": "online",
            "last_seen": "2026-02-20T14:30:00Z",
        }
        device = FleetDevice.from_dict(data)
        assert device.device_type == DeviceType.WATCH
        assert device.model == "Pixel Watch 2"
        assert device.nickname == "watch-pw2"

    def test_roundtrip(self) -> None:
        original = FleetDevice(
            serial="X",
            platform=Platform.IOS,
            device_type=DeviceType.TABLET,
            model="iPad Pro",
            os_version="17.0",
            nickname="ipad-dev",
            state="online",
            last_seen="2026-01-01T00:00:00Z",
        )
        restored = FleetDevice.from_dict(original.to_dict())
        assert restored.serial == original.serial
        assert restored.platform == original.platform
        assert restored.device_type == original.device_type
        assert restored.model == original.model
        assert restored.nickname == original.nickname

    def test_from_dict_os_version_int(self) -> None:
        """os_version should be coerced to str even if YAML gives an int."""
        data = {"serial": "ABC", "platform": "android", "os_version": 13}
        device = FleetDevice.from_dict(data)
        assert device.os_version == "13"

    def test_is_simulator_default_true(self) -> None:
        """is_simulator defaults to True for backward compatibility."""
        device = FleetDevice(serial="ABC", platform=Platform.IOS)
        assert device.is_simulator is True

    def test_is_simulator_false(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.IOS, is_simulator=False)
        assert device.is_simulator is False

    def test_to_dict_includes_is_simulator(self) -> None:
        device = FleetDevice(serial="ABC", platform=Platform.IOS, is_simulator=False)
        d = device.to_dict()
        assert d["is_simulator"] is False

    def test_from_dict_is_simulator_true(self) -> None:
        data = {"serial": "ABC", "platform": "ios", "is_simulator": True}
        device = FleetDevice.from_dict(data)
        assert device.is_simulator is True

    def test_from_dict_is_simulator_false(self) -> None:
        data = {"serial": "ABC", "platform": "ios", "is_simulator": False}
        device = FleetDevice.from_dict(data)
        assert device.is_simulator is False

    def test_from_dict_missing_is_simulator_defaults_true(self) -> None:
        """Backward compat: missing is_simulator key defaults to True."""
        data = {"serial": "ABC", "platform": "ios"}
        device = FleetDevice.from_dict(data)
        assert device.is_simulator is True

    def test_roundtrip_is_simulator(self) -> None:
        original = FleetDevice(
            serial="X",
            platform=Platform.IOS,
            is_simulator=False,
        )
        restored = FleetDevice.from_dict(original.to_dict())
        assert restored.is_simulator is False


class TestDeviceKitReExport:
    """Verify crucible re-exports device-kit canonical enums."""

    def test_platform_is_device_kit_enum(self) -> None:
        from linc_device_kit.models import Platform as DKPlatform

        from linc_crucible.core.device import Platform

        assert Platform is DKPlatform

    def test_device_type_is_device_kit_enum(self) -> None:
        from linc_device_kit.models import DeviceType as DKDeviceType

        from linc_crucible.core.device import DeviceType

        assert DeviceType is DKDeviceType


class TestIsSimulatorUdid:
    """Tests for is_simulator_udid() heuristic."""

    def test_simulator_udid(self) -> None:
        assert is_simulator_udid("AABBCCDD-1122-3344-5566-778899AABBCC") is True

    def test_physical_udid_40_hex(self) -> None:
        assert is_simulator_udid("00008120AABBCCDDEE112233445566778899AABB") is False

    def test_physical_udid_no_dashes(self) -> None:
        assert is_simulator_udid("00008120AABBCCDD") is False

    def test_empty_string(self) -> None:
        assert is_simulator_udid("") is False

    def test_short_string(self) -> None:
        assert is_simulator_udid("abc") is False

    def test_trailing_newline_rejected(self) -> None:
        from linc_crucible.core.device import _SIMULATOR_UDID_RE

        assert _SIMULATOR_UDID_RE.pattern.startswith(r"\A")
        assert _SIMULATOR_UDID_RE.pattern.endswith(r"\Z")
        assert is_simulator_udid("AABBCCDD-1122-3344-5566-778899AABBCC\n") is False
        assert is_simulator_udid("AABBCCDD-1122-3344-5566-778899AABBCC") is True

