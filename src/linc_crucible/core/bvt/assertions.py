"""BVT assertion helpers — evaluate conditions against device state.

Works with nerve's element dict format::

    {"id": "level_screen", "text": "Level", "bounds": "[0,0][1080,2400]", ...}
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BOUNDS_RE = re.compile(r"\A\[(\d+),(\d+)\]\[(\d+),(\d+)\]\Z")

# Platform minimum touch-target sizes, in density-INDEPENDENT units.
# Android (Material accessibility): 48dp. iOS (HIG): 44pt.
# These are the only constants here — the pixel threshold is always derived
# from the live device density (see ``min_touch_target_px``), never hardcoded.
_MIN_TOUCH_TARGET_DP = {"android": 48.0, "ios": 44.0}
_DEFAULT_TOUCH_TARGET_PLATFORM = "android"


@dataclass(frozen=True)
class AssertResult:
    """Result of a BVT assertion."""

    passed: bool
    message: str
    element_found: dict | None = None


def _id_matches(el: dict, resource_id: str) -> bool:
    """True if element id matches exactly or as a '/suffix' (same rule as assert_element)."""
    el_id = el.get("id", "") or ""
    return el_id == resource_id or el_id.endswith(f"/{resource_id}")


def assert_element(elements: list[dict], resource_id: str) -> AssertResult:
    """Assert that an element with the given resource ID is visible.

    Searches element ``id`` fields. Nerve elements use the format
    ``{"id": "resource_id", "text": "...", "bounds": "[x1,y1][x2,y2]"}``.
    """
    for el in elements:
        el_id = el.get("id", "") or ""
        # Match exact id or id suffix (e.g., "com.foo:id/level_screen" matches "level_screen")
        if el_id == resource_id or el_id.endswith(f"/{resource_id}"):
            return AssertResult(
                passed=True,
                message=f"Element '{resource_id}' found",
                element_found=el,
            )

    return AssertResult(
        passed=False,
        message=f"Element '{resource_id}' not found in {len(elements)} elements",
    )


def assert_text(elements: list[dict], text: str) -> AssertResult:
    """Assert that an element containing the given text is visible.

    Case-insensitive substring match on element ``text`` fields.
    """
    text_lower = text.lower()
    for el in elements:
        el_text = el.get("text", "") or ""
        if text_lower in el_text.lower():
            return AssertResult(
                passed=True,
                message=f"Text '{text}' found in element",
                element_found=el,
            )

    return AssertResult(
        passed=False,
        message=f"Text '{text}' not found in any of {len(elements)} elements",
    )


def _parse_bounds(bounds_str: str) -> tuple[int, int, int, int] | None:
    """Parse nerve bounds string '[x1,y1][x2,y2]' into coordinates."""
    m = _BOUNDS_RE.fullmatch(bounds_str)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def assert_content_description(elements: list[dict], resource_id: str) -> AssertResult:
    """Assert that an element has a contentDescription / accessibilityLabel."""
    for el in elements:
        el_id = el.get("id", "") or ""
        if el_id == resource_id or el_id.endswith(f"/{resource_id}"):
            desc = el.get("desc", "") or ""
            if desc:
                return AssertResult(
                    passed=True,
                    message=f"Element '{resource_id}' has contentDescription: '{desc}'",
                    element_found=el,
                )
            return AssertResult(
                passed=False,
                message=f"Element '{resource_id}' has no contentDescription",
                element_found=el,
            )
    # Fallback: match by content-description TEXT (case-insensitive substring).
    # Some accessibility suites assert description strings ("Main canvas grid")
    # rather than resource ids. The id path above is unchanged; this only runs
    # when no element id matched.
    target_lc = resource_id.lower()
    for el in elements:
        desc = el.get("desc", "") or ""
        if desc and target_lc in desc.lower():
            return AssertResult(
                passed=True,
                message=f"contentDescription matching '{resource_id}': '{desc}'",
                element_found=el,
            )
    return AssertResult(
        passed=False,
        message=f"Element '{resource_id}' not found in {len(elements)} elements",
    )


def min_touch_target_px(density_factor: float, platform: str = "android") -> int:
    """Platform minimum touch target in pixels for a device of this density.

    48dp (Android) / 44pt (iOS) x the live ``density_factor`` reported by
    ``nerve_density`` — the same density source ``assert_min_size`` converts
    dp with. Replaces the old hardcoded 132px, which was only correct at
    exactly 2.75x and silently passed sub-minimum targets on higher-DPI
    screens (e.g. 48dp is 144px at 3.0x).
    """
    dp = _MIN_TOUCH_TARGET_DP.get(
        str(platform).strip().lower(), _MIN_TOUCH_TARGET_DP[_DEFAULT_TOUCH_TARGET_PLATFORM]
    )
    return round(dp * float(density_factor))


def assert_touch_target(elements: list[dict], resource_id: str, min_size_px: int) -> AssertResult:
    """Assert that an element meets a minimum touch target size, in pixels.

    Pure pixels in, pure result out — dp->px conversion is the caller's job
    (it owns device density), matching ``assert_min_size``. Callers derive
    ``min_size_px`` from ``min_touch_target_px(density_factor, platform)``.
    """
    for el in elements:
        el_id = el.get("id", "") or ""
        if el_id == resource_id or el_id.endswith(f"/{resource_id}"):
            bounds_str = el.get("bounds", "") or ""
            coords = _parse_bounds(bounds_str)
            if coords is None:
                return AssertResult(
                    passed=False,
                    message=f"Element '{resource_id}' has invalid bounds: '{bounds_str}'",
                    element_found=el,
                )
            x1, y1, x2, y2 = coords
            width, height = x2 - x1, y2 - y1
            if width >= min_size_px and height >= min_size_px:
                return AssertResult(
                    passed=True,
                    message=(
                        f"Element '{resource_id}' touch target "
                        f"{width}x{height}px >= {min_size_px}px"
                    ),
                    element_found=el,
                )
            return AssertResult(
                passed=False,
                message=(
                    f"Element '{resource_id}' touch target {width}x{height}px "
                    f"below minimum {min_size_px}px"
                ),
                element_found=el,
            )
    return AssertResult(
        passed=False,
        message=f"Element '{resource_id}' not found in {len(elements)} elements",
    )


def assert_screen_reader(elements: list[dict], resource_id: str) -> AssertResult:
    """Assert that an element is visible to screen readers with a readable label.

    Checks that the element exists in the accessibility tree (nerve element list)
    and has either ``text`` or ``desc`` for TalkBack/VoiceOver to announce.
    """
    for el in elements:
        el_id = el.get("id", "") or ""
        if el_id == resource_id or el_id.endswith(f"/{resource_id}"):
            text = el.get("text", "") or ""
            desc = el.get("desc", "") or ""
            if text or desc:
                label = desc or text
                return AssertResult(
                    passed=True,
                    message=f"Element '{resource_id}' has screen reader label: '{label}'",
                    element_found=el,
                )
            return AssertResult(
                passed=False,
                message=(
                    f"Element '{resource_id}' has no text or contentDescription for screen readers"
                ),
                element_found=el,
            )
    return AssertResult(
        passed=False,
        message=(f"Element '{resource_id}' not in accessibility tree ({len(elements)} elements)"),
    )


def assert_clickable(elements: list[dict], resource_id: str) -> AssertResult:
    """Assert that an element (by id) is clickable / TalkBack-activatable.

    nerve emits ``clickable: True`` only when the node is clickable; the WDA /
    raw-uiautomator paths may serialize the string ``"true"`` — accept both.
    """
    for el in elements:
        el_id = el.get("id", "") or ""
        if el_id == resource_id or el_id.endswith(f"/{resource_id}"):
            if el.get("clickable") in (True, "true"):
                return AssertResult(
                    passed=True,
                    message=f"Element '{resource_id}' is clickable",
                    element_found=el,
                )
            return AssertResult(
                passed=False,
                message=f"Element '{resource_id}' is not clickable",
                element_found=el,
            )
    return AssertResult(
        passed=False,
        message=f"Element '{resource_id}' not found in {len(elements)} elements",
    )


def assert_min_size(
    elements: list[dict], resource_id: str, min_w_px: int, min_h_px: int
) -> AssertResult:
    """Assert an element's pixel bounds meet a minimum width and height.

    Pure pixels in, pure result out — dp→px conversion is the runner's job
    (it owns device density). Keeps this assertion device-agnostic and
    trivially unit-testable.
    """
    for el in elements:
        el_id = el.get("id", "") or ""
        if el_id == resource_id or el_id.endswith(f"/{resource_id}"):
            coords = _parse_bounds(el.get("bounds", "") or "")
            if coords is None:
                return AssertResult(
                    passed=False,
                    message=f"Element '{resource_id}' has invalid bounds",
                    element_found=el,
                )
            x1, y1, x2, y2 = coords
            w, h = x2 - x1, y2 - y1
            if w >= min_w_px and h >= min_h_px:
                return AssertResult(
                    passed=True,
                    message=f"Element '{resource_id}' {w}x{h}px >= {min_w_px}x{min_h_px}px",
                    element_found=el,
                )
            return AssertResult(
                passed=False,
                message=f"Element '{resource_id}' {w}x{h}px below {min_w_px}x{min_h_px}px",
                element_found=el,
            )
    return AssertResult(
        passed=False,
        message=f"Element '{resource_id}' not found in {len(elements)} elements",
    )


def assert_element_not_present(elements: list[dict], resource_id: str) -> AssertResult:
    """Assert NO element matches ``resource_id`` (exact or '/suffix').

    Single-shot against the supplied (fresh) dump — the inverse of
    ``assert_element``. No polling: "retry until absent" would false-pass on a
    transient render. Negative assertions are single-shot by design (see spec).
    """
    for el in elements:
        if _id_matches(el, resource_id):
            return AssertResult(
                passed=False,
                message=f"Element '{resource_id}' unexpectedly present",
                element_found=el,
            )
    return AssertResult(passed=True, message=f"Element '{resource_id}' absent (as expected)")


def assert_text_not_present(elements: list[dict], text: str) -> AssertResult:
    """Assert NO element text contains ``text`` (case-insensitive substring)."""
    text_lower = text.lower()
    for el in elements:
        el_text = el.get("text", "") or ""
        if text_lower in el_text.lower():
            return AssertResult(
                passed=False,
                message=f"Text '{text}' unexpectedly present",
                element_found=el,
            )
    return AssertResult(passed=True, message=f"Text '{text}' absent (as expected)")


def _resolve_count_bounds(
    expected: int | None, min_count: int | None, max_count: int | None
) -> tuple[int, int]:
    """Normalize count bounds to an inclusive (low, high) range.

    ``expected`` is exact-match sugar for ``min==max``. Raises ValueError on
    contradictory or missing bounds (AC4 — clear errors).
    """
    if expected is not None:
        if min_count is not None or max_count is not None:
            msg = "assert_element_count: 'expected' cannot be combined with min/max"
            raise ValueError(msg)
        return expected, expected
    if min_count is None and max_count is None:
        msg = "assert_element_count: at least one of expected/min/max is required"
        raise ValueError(msg)
    low = min_count if min_count is not None else 0
    high = max_count if max_count is not None else 1_000_000
    if low > high:
        msg = f"assert_element_count: min ({low}) > max ({high})"
        raise ValueError(msg)
    return low, high


def assert_element_count(
    elements: list[dict],
    resource_id: str,
    *,
    expected: int | None = None,
    min_count: int | None = None,
    max_count: int | None = None,
) -> AssertResult:
    """Assert the number of elements matching ``resource_id`` is within bounds.

    ``expected`` = exact count (sugar for min==max). Otherwise supply
    ``min_count`` and/or ``max_count``. Counts via the same '/suffix' matcher as
    ``assert_element``. Raises ValueError on missing/contradictory bounds.
    """
    low, high = _resolve_count_bounds(expected, min_count, max_count)
    count = sum(1 for el in elements if _id_matches(el, resource_id))
    if low <= count <= high:
        return AssertResult(
            passed=True,
            message=f"Element '{resource_id}' count {count} within [{low},{high}]",
        )
    return AssertResult(
        passed=False,
        message=f"Element '{resource_id}' count {count} outside [{low},{high}]",
    )
