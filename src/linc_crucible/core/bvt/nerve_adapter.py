"""NerveAdapter — BVT-facing wrapper over crucible's NerveClient.

Was: linc_forge.bvt.nerve_adapter using its own HTTPNerveClient.
Now: thin layer over linc_crucible.core.nerve_client.NerveClient,
adding bounds-parsing and tap-by-id / tap-by-text element queries.

All device operations go through the nerve daemon HTTP API (localhost:8111).
The daemon must be running for non-mocked use.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from linc_crucible.core.bvt.assertions import assert_element
from linc_crucible.core.nerve_client import NerveClient, UiTreeUnavailableError

log = structlog.get_logger()

# Pattern: [x1,y1][x2,y2] — Android nerve_elements bounds format
_BOUNDS_RE = re.compile(r"\A\[(\d+),(\d+)\]\[(\d+),(\d+)\]\Z")

# scroll_to_element: swipe from ~70% to ~30% of screen height at mid-width —
# a large-but-not-edge-to-edge vertical gesture that scrolls a full screen's
# worth of content without tripping edge-of-screen system gestures (nav bar /
# notification shade).
_SCROLL_TOP_FRACTION = 0.7
_SCROLL_BOTTOM_FRACTION = 0.3
DEFAULT_MAX_SWIPES = 5

# Default tap down→up duration. Compose `combinedClickable` uses a pointerInput
# gesture detector that discards a near-zero-duration synthetic tap; a manual
# `adb shell input tap` works because it synthesizes ~50ms between ACTION_DOWN
# and ACTION_UP. We match that by routing taps through nerve_long_press with a
# 50ms default. duration_ms=0 opts
# back into the legacy zero-duration nerve_tap (raw-coordinate escape hatch).
DEFAULT_TAP_DURATION_MS = 50

# Default long-press hold duration. Samsung devices can have a secure
# `long_press_timeout` of 500ms; a hold that does not EXCEED that threshold
# resolves as a click, so Compose `combinedClickable` fires onClick instead of
# onLongClick. Using a 1200ms default clears that threshold reliably across
# emulators and physical devices.
DEFAULT_LONG_PRESS_MS = 1200


def _parse_bounds_center(bounds: str) -> tuple[int, int] | None:
    """Parse nerve bounds string and return center point."""
    m = _BOUNDS_RE.fullmatch(bounds)
    if not m:
        return None
    x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _normalize_nerve_result(result: dict[str, Any] | None, verb: str) -> dict[str, Any]:
    """Normalize a raw result dictionary from a NerveClient call.

    - None or empty dict: failure dict with "nerve {verb} returned empty".
    - "success" already in result: returned unchanged.
    - "error" in result or "elements_error" in result: stamped with success=False.
    - otherwise: stamped with success=True.
    """
    if not result:
        return {"success": False, "error": f"nerve {verb} returned empty"}
    if "success" in result:
        return result
    if "error" in result or "elements_error" in result:
        return {**result, "success": False}
    return {**result, "success": True}


def _read_tree_strict(nerve: NerveClient, device: str) -> list[dict[str, Any]] | dict[str, Any]:
    """STRICT tree read for the element-resolving verbs (tap/long_press/scroll).

    Returns the element list, or — when the dump cannot be read
    (``UiTreeUnavailableError``) — a harness failure dict
    ``{"success": False, "error": "HARNESS: UI tree unavailable — <err>",
    "harness_error": True}`` (the wording the step handlers' ``_assert_tree``
    uses), so an unread tree is never reported as "No element ... found".
    """
    try:
        return nerve.elements_strict(device=device)
    except UiTreeUnavailableError as exc:
        return {
            "success": False,
            "error": f"HARNESS: UI tree unavailable — {exc}",
            "harness_error": True,
        }


def _id_center(
    elems: list[dict[str, Any]], resource_id: str, verb: str
) -> tuple[int, int] | dict[str, Any]:
    """Center of the element whose id is ``resource_id`` (exact or ``/suffix``).

    Returns an error dict naming ``verb`` when no element matches or its
    bounds cannot be parsed.
    """
    for el in elems:
        el_id = el.get("id", "") or ""
        if el_id == resource_id or el_id.endswith(f"/{resource_id}"):
            bounds = el.get("bounds", "")
            center = _parse_bounds_center(bounds)
            if center is None:
                return {
                    "success": False,
                    "error": f"Cannot parse bounds '{bounds}' for '{resource_id}'",
                }
            return center
    return {"success": False, "error": f"Element '{resource_id}' not found for {verb}"}


def _text_center(elems: list[dict[str, Any]], text: str) -> tuple[int, int] | dict[str, Any]:
    """Center of the first element whose text contains ``text`` (case-insensitive).

    Returns an error dict when no element matches or its bounds cannot be parsed.
    """
    text_lower = text.lower()
    for el in elems:
        el_text = el.get("text", "") or ""
        if text_lower in el_text.lower():
            center = _parse_bounds_center(el.get("bounds", ""))
            if center is None:
                return {"success": False, "error": f"Cannot parse bounds for text '{text}'"}
            return center
    return {"success": False, "error": f"No element with text '{text}' found"}


class NerveAdapter:
    """Device interaction via crucible's NerveClient.

    Forge HTTPNerveClient raised on any `{"error": ...}` body; NerveClient
    swallows most errors and returns {} or None. The methods below preserve
    the BVT runner's expected contract:

      - launch/kill: catch exceptions, return {"success": False, "error": ...}
      - tap: pass through (raises ConnectionError on daemon down)
      - screenshot: pass through (raises RuntimeError on missing path)
      - elements: pass through (returns [] via NerveClient)
      - tap_by_id / tap_by_text / long_press_by_id / scroll_to_element: read
        the tree STRICTLY; an unreadable tree returns
        {"success": False, "error": "HARNESS: ...", "harness_error": True}
      - navigate: catch exceptions, return {"success": False, "error": ...}
      - metrics: pass through (returns None on error via NerveClient)
      - install: pass through

    For testing, inject a mock ``nerve`` (see tests/conftest.py
    make_mock_nerve_client).
    """

    def __init__(self, nerve: NerveClient | None = None) -> None:
        self._nerve = nerve or NerveClient()

    @property
    def available(self) -> bool:
        """Whether the nerve daemon is reachable."""
        return self._nerve.available

    def discover(self) -> list[dict[str, Any]]:
        """Discover connected devices across both platforms."""
        return self._nerve.discover()

    def ensure_ready(self, device: str) -> dict[str, Any]:
        """Readiness verdict from nerve. A transport failure means not ready —
        skip the run rather than execute against an unverifiable device."""
        try:
            return self._nerve._call_tool("nerve_ensure_ready", {"device": device})
        except Exception as e:
            log.warning("bvt_ensure_ready_failed", device=device, error=str(e))
            return {"ready": False, "reason": f"nerve_ensure_ready transport error: {e}"}

    def launch(self, device: str, package: str) -> dict[str, Any]:
        """Launch an app on a device. Returns {"success": False, "error": ...} on error."""
        try:
            result = self._nerve.launch(device=device, package=package)
            return _normalize_nerve_result(result, "launch")
        except Exception as e:
            log.warning("bvt_launch_failed", device=device, package=package, error=str(e))
            return {"success": False, "error": str(e)}

    def kill(self, device: str, package: str) -> dict[str, Any]:
        """Kill an app on a device. Returns {"success": False, "error": ...} on error."""
        try:
            result = self._nerve.kill(device=device, package=package)
            return _normalize_nerve_result(result, "kill")
        except Exception as e:
            log.warning("bvt_kill_failed", device=device, package=package, error=str(e))
            return {"success": False, "error": str(e)}

    def clear_app_data(self, device: str, package: str) -> dict[str, Any]:
        """Wipe app data (Android ``pm clear``) for a cold start.

        Returns {"success": False, "error": ...} on error. CAVEAT: pm clear
        revokes runtime permissions + Keystore aliases — re-grant via
        ``grant_permission`` if a later step needs one.
        """
        try:
            result = self._nerve.clear_app_data(device=device, package=package)
            return _normalize_nerve_result(result, "clear_app_data")
        except Exception as e:
            log.warning("bvt_clear_app_data_failed", device=device, package=package, error=str(e))
            return {"success": False, "error": str(e)}

    def grant_permission(self, device: str, package: str, permission: str) -> dict[str, Any]:
        """Grant a runtime permission (Android ``pm grant``).

        Returns {"success": False, "error": ...} on error.
        """
        try:
            result = self._nerve.grant_permission(
                device=device, package=package, permission=permission
            )
            return _normalize_nerve_result(result, "grant_permission")
        except Exception as e:
            log.warning(
                "bvt_grant_permission_failed",
                device=device,
                package=package,
                permission=permission,
                error=str(e),
            )
            return {"success": False, "error": str(e)}

    def screenshot(self, device: str) -> bytes:
        """Take a screenshot, returning raw PNG bytes. Raises on missing path."""
        return self._nerve.screenshot(device=device)

    def elements(self, device: str) -> list[dict[str, Any]]:
        """Get flattened UI element tree as list of dicts."""
        return self._nerve.elements(device=device)

    def elements_strict(self, device: str) -> list[dict[str, Any]]:
        """Get the UI element tree, raising UiTreeUnavailableError if unreadable.

        STRICT counterpart to ``elements()`` for BVT assert steps — passes
        ``NerveClient.elements_strict``'s raise straight through (no catch).
        """
        return self._nerve.elements_strict(device=device)

    def tap(self, device: str, x: int, y: int) -> dict[str, Any]:
        """Tap at coordinates (raw, zero-duration ``nerve_tap``).

        Literal-coordinate path used by ``tap_by_text`` / ``tap_xy``. Element
        taps that must register on Compose ``combinedClickable`` go through
        ``_tap_with_duration`` instead (see ``tap_by_id``).
        """
        return self._nerve.tap(device=device, x=x, y=y)

    def _tap_with_duration(self, device: str, x: int, y: int, duration_ms: int) -> dict[str, Any]:
        """Tap with a synthesized down→up duration via ``nerve_long_press``.

        Compose ``combinedClickable`` discards a near-zero-duration synthetic
        tap, so a non-zero duration (default ~50ms, matching ``adb input tap``)
        is needed for the gesture to register. ``duration_ms <= 0`` falls back
        to the raw zero-duration ``tap`` — re-tapping a 0ms tap would just
        re-discard it.
        """
        if duration_ms <= 0:
            return self.tap(device, x, y)
        return self._nerve.long_press(device, x, y, duration_ms)

    def _foreground_key(self, device: str) -> Any:
        """Return a hashable snapshot of the current foreground, or None.

        Used by the verify-and-retry backstop to detect whether a tap changed
        the screen. ``screen_state`` swallows transport errors to ``{}`` and may
        intermittently return an empty state mid-poll, so an empty/unknown read
        maps to ``None`` (treated as "unchanged" — conservative: it triggers a
        retry rather than a false short-circuit).
        """
        try:
            state = self._nerve.screen_state(device=device)
        except Exception as e:  # transient transport error — treat as unknown
            log.warning("bvt_tap_foreground_unavailable", device=device, error=str(e))
            return None
        fg = (state or {}).get("foreground") or {}
        if not fg:
            return None
        return (fg.get("package"), fg.get("activity"))

    def _tap_with_verify_retry(
        self, device: str, x: int, y: int, duration_ms: int, retries: int
    ) -> dict[str, Any]:
        """Tap (x, y), then re-tap while the foreground stays unchanged.

        SECONDARY backstop for stale-frame races: after each tap, re-read the
        foreground; an unchanged foreground means the tap likely didn't land, so
        re-tap up to ``retries`` times. A changed foreground (nav landed)
        short-circuits. Callers gate ``retries`` on a non-zero ``duration_ms`` —
        re-tapping a 0ms tap would just re-discard it.
        """
        before = self._foreground_key(device)
        result = self._tap_with_duration(device, x, y, duration_ms)
        for _ in range(retries):
            after = self._foreground_key(device)
            if after is not None and after != before:
                break  # foreground changed → tap landed
            before = after
            result = self._tap_with_duration(device, x, y, duration_ms)
        return result

    def tap_by_id(
        self,
        device: str,
        resource_id: str,
        duration_ms: int = DEFAULT_TAP_DURATION_MS,
        max_retries: int = 0,
    ) -> dict[str, Any]:
        """Tap on an element by its resource ID.

        PRIMARY fix: the tap uses a non-zero ``duration_ms`` (default ~50ms) so
        Compose ``combinedClickable`` registers the gesture (see ``tap``).

        SECONDARY backstop (``max_retries`` > 0): re-tap while the foreground
        stays unchanged, up to ``max_retries`` — see ``_tap_with_verify_retry``.
        The retry only fires for a non-zero ``duration_ms``.
        """
        center = self._resolve_center(device, resource_id)
        if not isinstance(center, tuple):
            return center  # error dict
        retries = max_retries if duration_ms > 0 else 0
        if retries == 0:
            return self._tap_with_duration(device, center[0], center[1], duration_ms)
        return self._tap_with_verify_retry(device, center[0], center[1], duration_ms, retries)

    def _resolve_center(
        self, device: str, resource_id: str, verb: str = "tap"
    ) -> tuple[int, int] | dict[str, Any]:
        """Resolve a resource id to its element center over a STRICT tree read.

        Returns an error dict when the element is absent/unparseable, or the
        harness dict (``harness_error: True``) when the tree is unreadable.
        """
        elems = _read_tree_strict(self._nerve, device)
        if isinstance(elems, dict):
            return elems
        return _id_center(elems, resource_id, verb)

    def tap_by_text(self, device: str, text: str) -> dict[str, Any]:
        """Tap on the first element whose text matches (case-insensitive substring).

        Reads the tree STRICTLY: an unreadable tree returns the harness dict.
        """
        elems = _read_tree_strict(self._nerve, device)
        if isinstance(elems, dict):
            return elems
        center = _text_center(elems, text)
        if isinstance(center, dict):
            return center
        return self.tap(device, center[0], center[1])

    def type(self, device: str, text: str) -> dict[str, Any]:
        """Type text into the focused field. Returns {"success": False, ...} on error."""
        try:
            result = self._nerve.type_text(device=device, text=text)
            return _normalize_nerve_result(result, "type")
        except Exception as e:
            log.warning("bvt_type_failed", device=device, error=str(e))
            return {"success": False, "error": str(e)}

    def long_press_by_id(
        self, device: str, resource_id: str, duration_ms: int = DEFAULT_LONG_PRESS_MS
    ) -> dict[str, Any]:
        """Long-press an element by resource ID (resolves bounds → center).

        Reads the tree STRICTLY: an unreadable tree returns the harness dict.
        """
        center = self._resolve_center(device, resource_id, verb="long_press")
        if isinstance(center, dict):
            return center
        return self._nerve.long_press(device, center[0], center[1], duration_ms)

    def key(self, device: str, key: str = "ENTER") -> dict[str, Any]:
        """Press a hardware/IME key by name (ENTER, BACK, TAB, ...).

        Submits IME-"Done" forms that ``type`` (adb input text) can't trigger —
        ``adb input text`` does not fire Compose ``keyboardActions.onDone``, but
        a KEYCODE_ENTER does. Defaults to ENTER for the common IME-submit case.

        Routes straight to the nerve ``nerve_key`` tool (the NerveClient has no
        typed ``key`` wrapper). Returns {"success": False, "error": ...} on a
        transport error or an empty daemon response, mirroring launch/type.
        """
        try:
            result = self._nerve._call_tool("nerve_key", {"device": device, "key": key})
            return _normalize_nerve_result(result, "key")
        except Exception as e:
            log.warning("bvt_key_failed", device=device, key=key, error=str(e))
            return {"success": False, "error": str(e)}

    def density(self, device: str) -> dict[str, Any]:
        """Return the device dp→px density dict, or {} if unavailable (never raises).

        The runner hard-fails dp assertions when {} comes back, so transport
        errors surface as a clean step failure rather than an exception.
        """
        try:
            return self._nerve.density(device=device) or {}
        except Exception as e:
            log.warning("bvt_density_unavailable", device=device, error=str(e))
            return {}

    def scroll_to_element(
        self, device: str, resource_id: str, max_swipes: int = DEFAULT_MAX_SWIPES
    ) -> dict[str, Any]:
        """Swipe until an element with ``resource_id`` becomes visible, or give up.

        Scroll-until-visible only — does NOT tap; pair with a ``tap``/
        ``assert_element`` step. Queries the element tree first (so an
        already-visible element costs zero swipes); on a miss, swipes
        vertically from ~70% to ~30% of screen height at mid-width and
        re-queries, up to ``max_swipes`` times.

        Screen dimensions come from ``screen_state()``'s ``screen_size`` (the
        same Android ``wm size`` plumbing ``nerve_state`` already reports) —
        no new device-info call is introduced. If the daemon can't report a
        screen size, gives up immediately rather than swiping blind.

        Returns ``{"found": bool, "swipes": n}`` (plus ``"error"`` on the
        screen-size-unavailable path). The tree is read STRICTLY: an
        unreadable tree stops at once with the harness dict merged in
        (``success: False``, ``"error": "HARNESS: ..."``, ``harness_error: True``).
        """
        swipes = 0
        while True:
            elems = _read_tree_strict(self._nerve, device)
            if isinstance(elems, dict):
                return {**elems, "found": False, "swipes": swipes}
            if assert_element(elems, resource_id).passed:
                return {"found": True, "swipes": swipes}
            if swipes >= max_swipes:
                return {"found": False, "swipes": swipes}

            size = (self._nerve.screen_state(device=device) or {}).get("screen_size") or {}
            width, height = size.get("width") or 0, size.get("height") or 0
            if not width or not height:
                return {
                    "found": False,
                    "swipes": swipes,
                    "error": "screen size unavailable",
                }

            x = width // 2
            y_top = int(height * _SCROLL_TOP_FRACTION)
            y_bottom = int(height * _SCROLL_BOTTOM_FRACTION)
            self._nerve.swipe(device, x, y_top, x, y_bottom)
            swipes += 1

    def navigate(self, device: str, uri: str) -> dict[str, Any]:
        """Navigate to a deep link URI on a device."""
        try:
            result = self._nerve.navigate(device=device, uri=uri)
            return _normalize_nerve_result(result, "navigate")
        except Exception as e:
            log.warning("bvt_navigate_failed", device=device, uri=uri, error=str(e))
            return {"success": False, "error": str(e)}

    def inspect(self, device: str, package: str) -> dict[str, Any]:
        """Non-invasive process state via nerve_inspect, or {} if unavailable.

        Never raises — used as a last-resort launch-verification fallback on
        platforms whose state carries no foreground (iOS simulators), where a
        transport error must degrade to "could not verify", not abort the run.
        """
        try:
            result = self._nerve.inspect(device=device, package=package)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            log.warning("bvt_inspect_failed", device=device, package=package, error=str(e))
            return {}

    def foreground_probe(self, device: str) -> dict[str, Any]:
        """Return ``{"foreground": {...} or {}, "error": str | None}``. Never raises.

        Same ``nerve_state`` read as ``foreground()``, but preserves WHY no
        foreground was read instead of collapsing every failure to ``{}`` —
        the returned dict's own ``error``/``elements_error`` when nerve
        reported one, or ``f"{type(exc).__name__}: {exc}"`` when the call
        itself raises. Lets the runner's failure message cite the real cause
        instead of guessing "daemon unreachable".
        """
        try:
            state = self._nerve.screen_state(device=device)
        except Exception as e:
            log.warning("bvt_foreground_probe_failed", device=device, error=str(e))
            return {"foreground": {}, "error": f"{type(e).__name__}: {e}"}
        state = state or {}
        detail = state.get("error") or state.get("elements_error")
        return {
            "foreground": state.get("foreground") or {},
            "error": str(detail) if detail else None,
        }

    def metrics(self, device: str, package: str) -> dict[str, Any] | None:
        """Capture performance metrics. NerveClient already returns None on error."""
        return self._nerve.metrics(device=device, package=package)

    def install(self, device: str, apk_path: str) -> dict[str, Any]:
        """Install an app package on a device."""
        return self._nerve.install(device=device, path=apk_path)
