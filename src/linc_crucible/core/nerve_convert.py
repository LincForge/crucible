"""Convert nerve element format to device-kit ElementNode format."""

from __future__ import annotations

import re

from linc_device_kit.models import ElementNode, Rect, SemanticRole

# ------------------------------------------------------------------ #
# Android class → SemanticRole mapping                                 #
# ------------------------------------------------------------------ #

_ANDROID_EXACT: dict[str, SemanticRole] = {
    "android.widget.Button": SemanticRole.BUTTON,
    "android.widget.ImageButton": SemanticRole.BUTTON,
    "android.widget.TextView": SemanticRole.TEXT,
    "android.widget.ImageView": SemanticRole.IMAGE,
    "android.widget.CheckBox": SemanticRole.CHECKBOX,
    "android.widget.Switch": SemanticRole.SWITCH,
    "android.widget.ToggleButton": SemanticRole.SWITCH,
    "android.widget.SeekBar": SemanticRole.SLIDER,
    "android.widget.EditText": SemanticRole.TEXT_FIELD,
    "android.widget.ScrollView": SemanticRole.CONTAINER,
    "android.widget.LinearLayout": SemanticRole.CONTAINER,
    "android.widget.FrameLayout": SemanticRole.CONTAINER,
    "android.widget.RelativeLayout": SemanticRole.CONTAINER,
    "android.widget.ListView": SemanticRole.LIST,
    "androidx.recyclerview.widget.RecyclerView": SemanticRole.LIST,
    "android.widget.TabWidget": SemanticRole.TAB,
}

# ------------------------------------------------------------------ #
# iOS XCUIElementType → SemanticRole mapping                           #
# ------------------------------------------------------------------ #

_IOS_EXACT: dict[str, SemanticRole] = {
    "XCUIElementTypeButton": SemanticRole.BUTTON,
    "XCUIElementTypeStaticText": SemanticRole.TEXT,
    "XCUIElementTypeImage": SemanticRole.IMAGE,
    "XCUIElementTypeCheckBox": SemanticRole.CHECKBOX,
    "XCUIElementTypeSwitch": SemanticRole.SWITCH,
    "XCUIElementTypeSlider": SemanticRole.SLIDER,
    "XCUIElementTypeTextField": SemanticRole.TEXT_FIELD,
    "XCUIElementTypeSecureTextField": SemanticRole.TEXT_FIELD,
    "XCUIElementTypeTextView": SemanticRole.TEXT_FIELD,
    "XCUIElementTypeScrollView": SemanticRole.CONTAINER,
    "XCUIElementTypeOther": SemanticRole.CONTAINER,
    "XCUIElementTypeTable": SemanticRole.LIST,
    "XCUIElementTypeCell": SemanticRole.LIST_ITEM,
    "XCUIElementTypeTab": SemanticRole.TAB,
    "XCUIElementTypeLink": SemanticRole.LINK,
}

# ------------------------------------------------------------------ #
# Compose partial keyword → SemanticRole mapping                       #
# (checked in order — first match wins)                                #
# ------------------------------------------------------------------ #

_COMPOSE_PARTIAL: list[tuple[str, SemanticRole]] = [
    ("Button", SemanticRole.BUTTON),
    ("TextField", SemanticRole.TEXT_FIELD),
    ("Checkbox", SemanticRole.CHECKBOX),
    ("Switch", SemanticRole.SWITCH),
    ("Slider", SemanticRole.SLIDER),
    ("LazyColumn", SemanticRole.LIST),
    ("LazyRow", SemanticRole.LIST),
    ("Column", SemanticRole.CONTAINER),
    ("Row", SemanticRole.CONTAINER),
    ("Box", SemanticRole.CONTAINER),
    ("Tab", SemanticRole.TAB),
    ("Image", SemanticRole.IMAGE),
    ("Icon", SemanticRole.IMAGE),
    ("Text", SemanticRole.TEXT),
]

# Regex to parse "[x1,y1][x2,y2]" bounds strings
_BOUNDS_RE = re.compile(r"\A\[(\d+),(\d+)\]\[(\d+),(\d+)\]\Z")


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #


def parse_nerve_bounds(bounds_str: str) -> Rect | None:
    """Parse nerve bounds string "[x1,y1][x2,y2]" into a Rect.

    Returns None if the string is empty or malformed.
    """
    if not bounds_str:
        return None
    m = _BOUNDS_RE.fullmatch(bounds_str.strip())
    if not m:
        return None
    x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def map_class_to_role(class_name: str) -> SemanticRole:
    """Map a widget class name to a SemanticRole.

    Priority:
    1. Android exact match
    2. iOS exact match
    3. Compose partial keyword match (class contains keyword)
    4. UNKNOWN
    """
    if not class_name:
        return SemanticRole.UNKNOWN

    # 1. Android exact
    role = _ANDROID_EXACT.get(class_name)
    if role is not None:
        return role

    # 2. iOS exact
    role = _IOS_EXACT.get(class_name)
    if role is not None:
        return role

    # 3. Compose / partial — check if any keyword is contained in the class name
    #    Use the simple class name (last component) for matching to reduce false positives.
    simple = class_name.split(".")[-1]
    for keyword, mapped_role in _COMPOSE_PARTIAL:
        if keyword in simple:
            return mapped_role

    return SemanticRole.UNKNOWN


def _convert_single(nerve_dict: dict) -> ElementNode:
    """Convert a single nerve element dict to an ElementNode."""
    bounds = parse_nerve_bounds(nerve_dict.get("bounds", ""))
    role = map_class_to_role(nerve_dict.get("class", ""))
    children = convert_nerve_elements(nerve_dict.get("children", []))

    return ElementNode(
        id=nerve_dict.get("id", ""),
        text=nerve_dict.get("text", ""),
        description=nerve_dict.get("desc", ""),
        role=role.value,
        bounds=bounds,
        clickable=nerve_dict.get("clickable", False),
        # nerve omits "enabled" when True; it's only present when False
        enabled=nerve_dict.get("enabled", True),
        focused=nerve_dict.get("focused", False),
        scrollable=nerve_dict.get("scrollable", False),
        checked=nerve_dict.get("checked"),
        children=children,
    )


def convert_nerve_elements(nerve_elements: list[dict]) -> list[ElementNode]:
    """Convert a list of nerve element dicts to a list of ElementNode objects."""
    return [_convert_single(el) for el in nerve_elements]
