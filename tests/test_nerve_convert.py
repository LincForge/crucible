"""Tests for nerve element format converter."""

from __future__ import annotations

from linc_device_kit.models import ElementNode, Rect, SemanticRole

# ------------------------------------------------------------------ #
# parse_nerve_bounds                                                   #
# ------------------------------------------------------------------ #


def test_parse_nerve_bounds_valid() -> None:
    from linc_crucible.core.nerve_convert import parse_nerve_bounds

    result = parse_nerve_bounds("[100,200][300,400]")
    assert result == Rect(x=100, y=200, width=200, height=200)


def test_parse_nerve_bounds_empty_string() -> None:
    from linc_crucible.core.nerve_convert import parse_nerve_bounds

    assert parse_nerve_bounds("") is None


def test_parse_nerve_bounds_bad_string() -> None:
    from linc_crucible.core.nerve_convert import parse_nerve_bounds

    assert parse_nerve_bounds("bad") is None


def test_parse_nerve_bounds_trailing_junk_rejected() -> None:
    from linc_crucible.core.nerve_convert import _BOUNDS_RE, parse_nerve_bounds

    assert _BOUNDS_RE.pattern == r"\A\[(\d+),(\d+)\]\[(\d+),(\d+)\]\Z"
    assert parse_nerve_bounds("[0,0][1,1]") == Rect(x=0, y=0, width=1, height=1)
    assert parse_nerve_bounds("[0,0][1,1]extra") is None
    assert parse_nerve_bounds("[0,0][1,1]\nextra") is None
    assert _BOUNDS_RE.fullmatch("[0,0][1,1]\n") is None


# ------------------------------------------------------------------ #
# map_class_to_role — Android                                          #
# ------------------------------------------------------------------ #


def test_map_class_android_button() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("android.widget.Button") == SemanticRole.BUTTON


def test_map_class_android_textview() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("android.widget.TextView") == SemanticRole.TEXT


def test_map_class_android_imageview() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("android.widget.ImageView") == SemanticRole.IMAGE


def test_map_class_android_checkbox() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("android.widget.CheckBox") == SemanticRole.CHECKBOX


def test_map_class_android_switch() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("android.widget.Switch") == SemanticRole.SWITCH


def test_map_class_android_seekbar() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("android.widget.SeekBar") == SemanticRole.SLIDER


def test_map_class_android_edittext() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("android.widget.EditText") == SemanticRole.TEXT_FIELD


# ------------------------------------------------------------------ #
# map_class_to_role — iOS                                              #
# ------------------------------------------------------------------ #


def test_map_class_ios_button() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("XCUIElementTypeButton") == SemanticRole.BUTTON


def test_map_class_ios_statictext() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("XCUIElementTypeStaticText") == SemanticRole.TEXT


def test_map_class_ios_image() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("XCUIElementTypeImage") == SemanticRole.IMAGE


def test_map_class_ios_switch() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("XCUIElementTypeSwitch") == SemanticRole.SWITCH


def test_map_class_ios_slider() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("XCUIElementTypeSlider") == SemanticRole.SLIDER


def test_map_class_ios_textfield() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("XCUIElementTypeTextField") == SemanticRole.TEXT_FIELD


# ------------------------------------------------------------------ #
# map_class_to_role — unknown / empty                                  #
# ------------------------------------------------------------------ #


def test_map_class_custom_unknown() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("com.custom.Widget") == SemanticRole.UNKNOWN


def test_map_class_empty_unknown() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("") == SemanticRole.UNKNOWN


# ------------------------------------------------------------------ #
# map_class_to_role — Compose partial matches                          #
# ------------------------------------------------------------------ #


def test_map_class_compose_button() -> None:
    from linc_crucible.core.nerve_convert import map_class_to_role

    assert map_class_to_role("androidx.compose.material3.Button") == SemanticRole.BUTTON


# ------------------------------------------------------------------ #
# convert_nerve_elements                                               #
# ------------------------------------------------------------------ #


def test_convert_nerve_elements_single() -> None:
    from linc_crucible.core.nerve_convert import convert_nerve_elements

    nerve_elements = [
        {
            "id": "btn",
            "text": "Submit",
            "desc": "Submit form",
            "class": "android.widget.Button",
            "bounds": "[100,200][300,280]",
            "clickable": True,
        }
    ]
    result = convert_nerve_elements(nerve_elements)

    assert len(result) == 1
    el = result[0]
    assert isinstance(el, ElementNode)
    assert el.id == "btn"
    assert el.text == "Submit"
    assert el.description == "Submit form"
    assert el.role == SemanticRole.BUTTON.value
    assert el.bounds == Rect(x=100, y=200, width=200, height=80)
    assert el.clickable is True
    assert el.enabled is True  # default
    assert el.children == []


def test_convert_nerve_elements_nested_children() -> None:
    from linc_crucible.core.nerve_convert import convert_nerve_elements

    nerve_elements = [
        {
            "id": "root",
            "class": "android.widget.LinearLayout",
            "bounds": "[0,0][1080,1920]",
            "children": [
                {
                    "id": "child_btn",
                    "text": "OK",
                    "class": "android.widget.Button",
                    "bounds": "[100,100][200,150]",
                    "clickable": True,
                }
            ],
        }
    ]
    result = convert_nerve_elements(nerve_elements)

    assert len(result) == 1
    root = result[0]
    assert root.role == SemanticRole.CONTAINER.value
    assert len(root.children) == 1
    child = root.children[0]
    assert child.role == SemanticRole.BUTTON.value
    assert child.text == "OK"
    assert child.clickable is True


def test_convert_nerve_elements_empty_list() -> None:
    from linc_crucible.core.nerve_convert import convert_nerve_elements

    assert convert_nerve_elements([]) == []


def test_convert_nerve_elements_minimal_element() -> None:
    """An empty dict should produce an ElementNode with sensible defaults."""
    from linc_crucible.core.nerve_convert import convert_nerve_elements

    result = convert_nerve_elements([{}])

    assert len(result) == 1
    el = result[0]
    assert isinstance(el, ElementNode)
    assert el.id == ""
    assert el.text == ""
    assert el.description == ""
    assert el.role == SemanticRole.UNKNOWN.value
    assert el.bounds is None
    assert el.clickable is False
    assert el.enabled is True
    assert el.checked is None
    assert el.children == []
