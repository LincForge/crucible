from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml

from linc_crucible._bounded_read import _MAX_REGISTRY_BYTES, RegistryTooLargeError, read_capped
from linc_crucible.core.state_paths import linc_home

logger = structlog.get_logger(domain="device_policy")

DevicePolicyState = Literal["loaded", "absent", "unreadable"]
_STATES: frozenset[str] = frozenset({"loaded", "absent", "unreadable"})

_MAX_YAML_ALIASES = 100
_MAX_YAML_NODES = 100_000

_absent_warned: bool = False
_ABSENT_HINT = "set LINC_DEVICE_FLEET_FILE to register personal devices"


class RegistryComplexityError(yaml.YAMLError):
    """A YAML registry exceeded the alias, node-count, or recursion safety budget."""


class RegistryNodeBudgetError(ValueError):
    """A parsed registry exceeded the iterative node-walk budget."""


class _BoundedSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that caps alias events and expanded nodes and rejects recursive aliases."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._alias_count = 0
        self._node_count = 0
        self._composing_anchors: set[str] = set()
        self._node_weights: dict[int, int] = {}

    def _add_nodes(self, count: int) -> None:
        self._node_count += count
        if self._node_count > _MAX_YAML_NODES:
            raise RegistryComplexityError(f"YAML node count exceeded limit of {_MAX_YAML_NODES}")

    def compose_node(self, *args: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            self._alias_count += 1
            if self._alias_count > _MAX_YAML_ALIASES:
                raise RegistryComplexityError(
                    f"YAML alias count exceeded limit of {_MAX_YAML_ALIASES}"
                )
            alias_event = self.peek_event()
            if alias_event.anchor in self._composing_anchors:
                raise RegistryComplexityError(
                    f"recursive YAML alias {alias_event.anchor!r} is not allowed"
                )
            node = super().compose_node(*args)
            self._add_nodes(self._node_weights[id(node)])
            return node

        self._add_nodes(1)
        before = self._node_count - 1
        anchor = self.peek_event().anchor
        if anchor is not None:
            self._composing_anchors.add(anchor)
        try:
            node = super().compose_node(*args)
        finally:
            if anchor is not None:
                self._composing_anchors.discard(anchor)
        self._node_weights[id(node)] = self._node_count - before
        return node


def _load_bounded_yaml(raw: bytes) -> Any:
    loader = _BoundedSafeLoader(raw)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _fleet_registry_path() -> Path:
    """Return the personal-device registry path.

    ``LINC_DEVICE_FLEET_FILE`` when set and non-empty, otherwise the neutral
    ``$LINC_HOME/device-fleet.yaml`` (``~/.linc/device-fleet.yaml``). A deployment that
    keeps its registry elsewhere points the env var at it; the package never defaults
    into a private checkout.
    """
    env = os.environ.get("LINC_DEVICE_FLEET_FILE")
    if env:
        return Path(env)
    return linc_home() / "device-fleet.yaml"


@dataclass(frozen=True, init=False)
class DevicePolicy:
    """Personal-device policy loaded from the LINC device fleet registry."""

    personal_serials: frozenset[str]
    state: DevicePolicyState
    source: str
    error: str = ""

    def __init__(
        self,
        personal_serials: frozenset[str],
        state: DevicePolicyState = "loaded",
        source: str = "",
        error: str = "",
        *,
        loaded: bool | None = None,
    ) -> None:
        resolved: DevicePolicyState = (
            ("loaded" if loaded else "unreadable") if loaded is not None else state
        )
        if resolved not in _STATES:
            raise ValueError(
                f"unknown DevicePolicy state {resolved!r}; expected one of {sorted(_STATES)}"
            )
        object.__setattr__(self, "personal_serials", personal_serials)
        object.__setattr__(self, "state", resolved)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "error", error)

    @property
    def loaded(self) -> bool:
        """True when the fleet registry was loaded and parsed as a mapping."""
        return self.state == "loaded"

    def __call__(self) -> DevicePolicy:
        return self

    def is_personal(self, serial: str) -> bool:
        """Return whether ``serial`` must be treated as a personal device."""
        if self.state == "absent":
            return False
        if self.state != "loaded":
            return True
        if not serial:
            return False
        cleaned = str(serial).strip().casefold()
        if not cleaned:
            return False
        return any(str(s).strip().casefold() == cleaned for s in self.personal_serials)

    def refusal(self, serial: str) -> str | None:
        """Return a refusal message for ``serial``, or ``None`` if allowed."""
        if self.state not in ("loaded", "absent"):
            return (
                f"device registry {self.source} could not be read ({self.error}); "
                "refusing device operations without an explicit allow_personal=True"
            )
        if self.is_personal(serial):
            return (
                f"device {serial} is registered personal: true in {self.source}; "
                "pass allow_personal=True to override"
            )
        return None


def _extract_personal_serials(node: Any, *, node_budget: int = _MAX_YAML_NODES) -> set[str]:
    serials: set[str] = set()
    visited: set[int] = set()
    stack: list[Any] = [node]
    steps = 0
    while stack:
        current = stack.pop()
        steps += 1
        if steps > node_budget:
            raise RegistryNodeBudgetError(
                f"device registry walk exceeded {node_budget}-node budget"
            )
        if not isinstance(current, (dict, list)):
            continue
        node_id = id(current)
        if node_id in visited:
            continue
        visited.add(node_id)
        if isinstance(current, dict):
            if current.get("personal"):
                for key in ("serial", "hardware_id"):
                    val = current.get(key)
                    cleaned = str(val).strip() if val else ""
                    if cleaned:
                        serials.add(cleaned)
            stack.extend(current.values())
        else:
            stack.extend(current)
    return serials


def _probe_fleet_path(path: Path) -> tuple[DevicePolicyState, str] | None:
    """Classify ``path`` before reading it; ``None`` means a regular file to read.

    Only a clean ``FileNotFoundError`` from ``lstat`` (no directory entry at all) is
    ``"absent"``. An entry that exists but cannot be stat-ed (dangling symlink, ELOOP,
    EACCES) or is not a regular file (directory, FIFO, device) is ``"unreadable"``.
    """
    try:
        path.lstat()
    except FileNotFoundError:
        return "absent", f"file not found: {path}"
    except (OSError, ValueError) as exc:
        return "unreadable", str(exc)
    try:
        mode = path.stat().st_mode
    except (OSError, ValueError) as exc:
        return "unreadable", str(exc)
    if not stat.S_ISREG(mode):
        return "unreadable", f"not a regular file: {path}"
    return None


def _unreadable(source: str, error: str) -> DevicePolicy:
    logger.warning("device_fleet_policy_read_failed", path=source, error=error)
    return DevicePolicy(
        personal_serials=frozenset(), state="unreadable", source=source, error=error
    )


def load_device_policy(path: Path | None = None) -> DevicePolicy:
    """Load the personal-device policy from ``path`` or the default fleet path."""
    global _absent_warned
    target_path = _fleet_registry_path() if path is None else Path(path)
    source_str = str(target_path)

    probe = _probe_fleet_path(target_path)
    if probe is not None:
        state, error = probe
        if state == "unreadable":
            return _unreadable(source_str, error)
        if not _absent_warned:
            _absent_warned = True
            logger.warning("device_fleet_absent", path=source_str, hint=_ABSENT_HINT)
        return DevicePolicy(
            personal_serials=frozenset(), state="absent", source=source_str, error=error
        )

    try:
        raw = read_capped(target_path, _MAX_REGISTRY_BYTES)
        # Bytes in: PyYAML detects UTF-8/UTF-16 from the BOM and reports undecodable
        # input as a ReaderError (a YAMLError), so decode failures still fail closed.
        data = _load_bounded_yaml(raw)
        if not isinstance(data, dict):
            return _unreadable(source_str, "device registry is not a YAML mapping")
        serials = _extract_personal_serials(data)
    except RegistryTooLargeError as exc:
        return _unreadable(source_str, f"is too large: {exc}")
    except (yaml.YAMLError, OSError, RegistryNodeBudgetError, RecursionError) as exc:
        return _unreadable(source_str, str(exc))

    return DevicePolicy(
        personal_serials=frozenset(serials),
        source=source_str,
        error="",
    )
