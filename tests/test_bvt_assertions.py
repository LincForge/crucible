"""Tests for BVT assertion helpers."""

from __future__ import annotations

import pytest

from linc_crucible.core.bvt.assertions import (
    assert_clickable,
    assert_content_description,
    assert_element,
    assert_element_count,
    assert_element_not_present,
    assert_min_size,
    assert_screen_reader,
    assert_text,
    assert_text_not_present,
    assert_touch_target,
    min_touch_target_px,
)

# Sample elements matching nerve's format
SAMPLE_ELEMENTS: list[dict] = [
    {
        "id": "level_screen",
        "text": "",
        "desc": "",
        "bounds": "[0,0][1080,2400]",
        "class": "FrameLayout",
    },
    {
        "id": "pitch_readout",
        "text": "0.0\u00b0",
        "desc": "Pitch angle",
        "bounds": "[100,200][300,400]",
        "class": "TextView",
    },
    {
        "id": "settings_button",
        "text": "Settings",
        "desc": "Open settings",
        "bounds": "[900,50][1050,200]",
        "class": "Button",
    },
    {
        "id": "",
        "text": "MEASUREMENT UNIT",
        "desc": "",
        "bounds": "[50,500][500,550]",
        "class": "TextView",
    },
    {
        "id": "com.lincmobile.level:id/calibrate_btn",
        "text": "Calibrate",
        "desc": "Calibrate sensor",
        "bounds": "[50,600][300,750]",
        "class": "Button",
    },
]


class TestAssertElement:
    """Test assert_element with various ID formats."""

    def test_exact_match(self) -> None:
        result = assert_element(SAMPLE_ELEMENTS, "level_screen")
        assert result.passed
        assert result.element_found is not None
        assert result.element_found["id"] == "level_screen"

    def test_qualified_id_match(self) -> None:
        """Match 'calibrate_btn' against 'com.lincmobile.level:id/calibrate_btn'."""
        result = assert_element(SAMPLE_ELEMENTS, "calibrate_btn")
        assert result.passed
        assert "calibrate_btn" in result.element_found["id"]

    def test_not_found(self) -> None:
        result = assert_element(SAMPLE_ELEMENTS, "nonexistent_element")
        assert not result.passed
        assert "not found" in result.message
        assert result.element_found is None

    def test_empty_elements(self) -> None:
        result = assert_element([], "any_id")
        assert not result.passed
        assert "0 elements" in result.message

    def test_element_with_none_id(self) -> None:
        elements = [{"id": None, "text": "foo", "bounds": "[0,0][100,100]"}]
        result = assert_element(elements, "foo")
        assert not result.passed

    def test_message_on_success(self) -> None:
        result = assert_element(SAMPLE_ELEMENTS, "pitch_readout")
        assert "found" in result.message.lower()


class TestAssertText:
    """Test assert_text with various text patterns."""

    def test_exact_match(self) -> None:
        result = assert_text(SAMPLE_ELEMENTS, "Settings")
        assert result.passed
        assert result.element_found["text"] == "Settings"

    def test_case_insensitive(self) -> None:
        result = assert_text(SAMPLE_ELEMENTS, "settings")
        assert result.passed

    def test_substring_match(self) -> None:
        result = assert_text(SAMPLE_ELEMENTS, "MEASUREMENT")
        assert result.passed

    def test_not_found(self) -> None:
        result = assert_text(SAMPLE_ELEMENTS, "Nonexistent Text")
        assert not result.passed
        assert "not found" in result.message

    def test_empty_elements(self) -> None:
        result = assert_text([], "anything")
        assert not result.passed

    def test_element_with_none_text(self) -> None:
        elements = [{"id": "foo", "text": None, "bounds": "[0,0][100,100]"}]
        result = assert_text(elements, "bar")
        assert not result.passed

    def test_degree_symbol(self) -> None:
        """Nerve returns text with unicode degree symbol."""
        result = assert_text(SAMPLE_ELEMENTS, "0.0\u00b0")
        assert result.passed


class TestAssertContentDescription:
    """Test assert_content_description."""

    def test_has_description(self) -> None:
        result = assert_content_description(SAMPLE_ELEMENTS, "settings_button")
        assert result.passed
        assert "Open settings" in result.message

    def test_missing_description(self) -> None:
        result = assert_content_description(SAMPLE_ELEMENTS, "level_screen")
        assert not result.passed
        assert "no contentDescription" in result.message
        assert result.element_found is not None

    def test_not_found(self) -> None:
        result = assert_content_description(SAMPLE_ELEMENTS, "nonexistent")
        assert not result.passed
        assert "not found" in result.message

    def test_qualified_id(self) -> None:
        result = assert_content_description(SAMPLE_ELEMENTS, "calibrate_btn")
        assert result.passed

    def test_none_desc(self) -> None:
        elements = [{"id": "btn", "text": "OK", "desc": None, "bounds": "[0,0][100,100]"}]
        result = assert_content_description(elements, "btn")
        assert not result.passed

    def test_desc_text_fallback_when_no_id_match(self) -> None:
        # BlueprintRoom asserts by content-description TEXT, not resource id.
        result = assert_content_description(SAMPLE_ELEMENTS, "Open settings")
        assert result.passed
        assert "Open settings" in result.message

    def test_desc_text_fallback_case_insensitive_substring(self) -> None:
        result = assert_content_description(SAMPLE_ELEMENTS, "open sett")
        assert result.passed

    def test_desc_text_fallback_no_match(self) -> None:
        result = assert_content_description(SAMPLE_ELEMENTS, "totally absent label")
        assert not result.passed
        assert "not found" in result.message


class TestAssertTouchTarget:
    """Test assert_touch_target."""

    # 132px is what the old hardcoded default stood in for: 48dp at 2.75x.
    LEGACY_132 = min_touch_target_px(2.75)

    def test_meets_minimum(self) -> None:
        result = assert_touch_target(SAMPLE_ELEMENTS, "settings_button", self.LEGACY_132)
        assert result.passed
        assert "150x150" in result.message

    def test_too_small(self) -> None:
        elements = [{"id": "tiny", "text": "", "bounds": "[0,0][50,50]", "class": "Button"}]
        result = assert_touch_target(elements, "tiny", self.LEGACY_132)
        assert not result.passed
        assert "below minimum" in result.message

    def test_too_narrow(self) -> None:
        elements = [{"id": "narrow", "text": "", "bounds": "[0,0][50,200]", "class": "Button"}]
        result = assert_touch_target(elements, "narrow", self.LEGACY_132)
        assert not result.passed

    def test_too_short(self) -> None:
        elements = [{"id": "short", "text": "", "bounds": "[0,0][200,50]", "class": "Button"}]
        result = assert_touch_target(elements, "short", self.LEGACY_132)
        assert not result.passed

    def test_custom_min_size(self) -> None:
        elements = [{"id": "btn", "text": "", "bounds": "[0,0][100,100]", "class": "Button"}]
        result = assert_touch_target(elements, "btn", min_size_px=96)
        assert result.passed

    def test_invalid_bounds(self) -> None:
        elements = [{"id": "bad", "text": "", "bounds": "garbage", "class": "Button"}]
        result = assert_touch_target(elements, "bad", self.LEGACY_132)
        assert not result.passed
        assert "invalid bounds" in result.message

    def test_not_found(self) -> None:
        result = assert_touch_target(SAMPLE_ELEMENTS, "nonexistent", self.LEGACY_132)
        assert not result.passed
        assert "not found" in result.message

    def test_empty_bounds(self) -> None:
        elements = [{"id": "btn", "text": "", "bounds": "", "class": "Button"}]
        result = assert_touch_target(elements, "btn", self.LEGACY_132)
        assert not result.passed

    def test_bounds_trailing_junk_and_newline_rejected(self) -> None:
        from linc_crucible.core.bvt.assertions import _BOUNDS_RE, _parse_bounds

        assert _BOUNDS_RE.pattern == r"\A\[(\d+),(\d+)\]\[(\d+),(\d+)\]\Z"
        assert _parse_bounds("[0,0][1,1]") == (0, 0, 1, 1)
        assert _parse_bounds("[0,0][1,1]\n") is None
        assert _parse_bounds("[0,0][1,1]extra") is None
        assert not assert_touch_target(
            [{"id": "btn", "text": "", "bounds": "[0,0][200,200]\n", "class": "Button"}],
            "btn",
            self.LEGACY_132,
        ).passed
        assert not assert_touch_target(
            [{"id": "btn", "text": "", "bounds": "[0,0][200,200]extra", "class": "Button"}],
            "btn",
            self.LEGACY_132,
        ).passed


class TestMinTouchTargetPx:
    """Density-derived platform minimum touch target (replaces the 132px hardcode)."""

    def test_android_high_dpi_is_stricter_than_the_old_132_hardcode(self) -> None:
        # Real bench device: SM-S921U (Galaxy S24), 480dpi -> density_factor 3.0.
        # 48dp * 3.0 = 144px, i.e. MORE than the retired 132px default.
        assert min_touch_target_px(3.0) == 144
        assert min_touch_target_px(3.0) > 132

    def test_android_low_dpi_is_looser_than_the_old_132_hardcode(self) -> None:
        # Real bench device: Pixel 8 Pro, 360dpi -> density_factor 2.25.
        assert min_touch_target_px(2.25) == 108
        assert min_touch_target_px(2.25) < 132

    def test_android_mdpi_baseline(self) -> None:
        assert min_touch_target_px(1.0) == 48

    def test_legacy_default_reproduced_at_2_75x(self) -> None:
        # The retired hardcode was "48dp at 2.75x" -> the helper agrees exactly.
        assert min_touch_target_px(2.75) == 132

    def test_ios_uses_44pt(self) -> None:
        assert min_touch_target_px(3.0, "ios") == 132
        assert min_touch_target_px(2.0, "ios") == 88

    def test_platform_match_is_case_insensitive(self) -> None:
        assert min_touch_target_px(3.0, "iOS") == 132

    def test_unknown_platform_falls_back_to_android(self) -> None:
        assert min_touch_target_px(3.0, "harmony") == 144


# 140x140px button: passed under the old 132px default, but is only 46.7dp on a
# 3.0x screen — below the 48dp Android minimum.
_SUB_MINIMUM_ON_HIGH_DPI: list[dict] = [
    {"id": "btn", "text": "", "bounds": "[0,0][140,140]", "class": "Button"}
]


class TestAssertTouchTargetDensityDerived:
    """The regression the 132px hardcode masked, at real bench densities."""

    SUB_MINIMUM_ON_HIGH_DPI = _SUB_MINIMUM_ON_HIGH_DPI

    def test_high_dpi_now_fails_what_132_wrongly_passed(self) -> None:
        result = assert_touch_target(self.SUB_MINIMUM_ON_HIGH_DPI, "btn", min_touch_target_px(3.0))
        assert not result.passed
        assert "below minimum 144px" in result.message

    def test_same_element_wrongly_passed_under_the_old_132_hardcode(self) -> None:
        # Pins the bug: with the retired threshold this element is reported OK.
        assert assert_touch_target(self.SUB_MINIMUM_ON_HIGH_DPI, "btn", 132).passed

    def test_low_dpi_passes_at_its_own_derived_threshold(self) -> None:
        elements = [{"id": "btn", "text": "", "bounds": "[0,0][110,110]", "class": "Button"}]
        result = assert_touch_target(elements, "btn", min_touch_target_px(2.25))
        assert result.passed
        assert "110x110px >= 108px" in result.message

    def test_low_dpi_still_catches_a_genuinely_small_target(self) -> None:
        elements = [{"id": "btn", "text": "", "bounds": "[0,0][100,100]", "class": "Button"}]
        assert not assert_touch_target(elements, "btn", min_touch_target_px(2.25)).passed

    def test_no_hardcoded_default_remains(self) -> None:
        # The threshold is now a required, caller-supplied (density-derived) value.
        with pytest.raises(TypeError):
            assert_touch_target(SAMPLE_ELEMENTS, "settings_button")  # type: ignore[call-arg]


class TestAssertScreenReader:
    """Test assert_screen_reader."""

    def test_has_desc(self) -> None:
        result = assert_screen_reader(SAMPLE_ELEMENTS, "pitch_readout")
        assert result.passed
        assert "screen reader label" in result.message

    def test_has_text_only(self) -> None:
        """Element with text but no desc is still screen-reader visible."""
        elements = [{"id": "btn", "text": "OK", "desc": "", "bounds": "[0,0][100,100]"}]
        result = assert_screen_reader(elements, "btn")
        assert result.passed

    def test_no_label(self) -> None:
        result = assert_screen_reader(SAMPLE_ELEMENTS, "level_screen")
        assert not result.passed
        assert "no text or contentDescription" in result.message

    def test_not_found(self) -> None:
        result = assert_screen_reader(SAMPLE_ELEMENTS, "nonexistent")
        assert not result.passed
        assert "not in accessibility tree" in result.message

    def test_none_fields(self) -> None:
        elements = [{"id": "btn", "text": None, "desc": None, "bounds": "[0,0][100,100]"}]
        result = assert_screen_reader(elements, "btn")
        assert not result.passed


class TestAssertClickable:
    """Test assert_clickable — element by id must be clickable."""

    def test_clickable_true_bool(self) -> None:
        elements = [{"id": "info_button", "bounds": "[0,0][100,100]", "clickable": True}]
        result = assert_clickable(elements, "info_button")
        assert result.passed
        assert "is clickable" in result.message

    def test_clickable_true_string_wire_form(self) -> None:
        # Defensive: WDA / raw uiautomator may serialize the string "true".
        elements = [{"id": "info_button", "bounds": "[0,0][100,100]", "clickable": "true"}]
        result = assert_clickable(elements, "info_button")
        assert result.passed

    def test_not_clickable_when_key_absent(self) -> None:
        # nerve omits the clickable key when false (device.py to_dict).
        elements = [{"id": "info_button", "bounds": "[0,0][100,100]"}]
        result = assert_clickable(elements, "info_button")
        assert not result.passed
        assert "not clickable" in result.message

    def test_qualified_id(self) -> None:
        elements = [{"id": "com.x:id/info_button", "bounds": "[0,0][1,1]", "clickable": True}]
        result = assert_clickable(elements, "info_button")
        assert result.passed

    def test_not_found(self) -> None:
        result = assert_clickable([], "info_button")
        assert not result.passed
        assert "not found" in result.message


class TestAssertMinSize:
    """Test assert_min_size — pure pixel comparison (dp conversion is the runner's job)."""

    def test_meets_minimum(self) -> None:
        elements = [{"id": "btn", "bounds": "[0,0][147,147]"}]
        result = assert_min_size(elements, "btn", 147, 147)
        assert result.passed
        assert "147x147" in result.message

    def test_too_narrow(self) -> None:
        elements = [{"id": "btn", "bounds": "[0,0][100,147]"}]
        result = assert_min_size(elements, "btn", 147, 147)
        assert not result.passed
        assert "below" in result.message

    def test_too_short(self) -> None:
        elements = [{"id": "btn", "bounds": "[0,0][147,100]"}]
        result = assert_min_size(elements, "btn", 147, 147)
        assert not result.passed

    def test_invalid_bounds(self) -> None:
        elements = [{"id": "btn", "bounds": "garbage"}]
        result = assert_min_size(elements, "btn", 147, 147)
        assert not result.passed
        assert "invalid bounds" in result.message

    def test_not_found(self) -> None:
        result = assert_min_size([], "btn", 147, 147)
        assert not result.passed
        assert "not found" in result.message


# --- T5: negative + count assertions ---

COUNT_ELEMENTS: list[dict] = [
    {"id": "row", "text": "A", "bounds": "[0,0][10,10]"},
    {"id": "row", "text": "B", "bounds": "[0,10][10,20]"},
    {"id": "com.x:id/row", "text": "C", "bounds": "[0,20][10,30]"},
    {"id": "header", "text": "H", "bounds": "[0,30][10,40]"},
]


class TestAssertElementNotPresent:
    def test_absent_passes(self) -> None:
        assert assert_element_not_present(SAMPLE_ELEMENTS, "nonexistent").passed

    def test_present_exact_fails(self) -> None:
        assert not assert_element_not_present(SAMPLE_ELEMENTS, "level_screen").passed

    def test_present_suffix_fails(self) -> None:
        # 'calibrate_btn' matches 'com.lincmobile.level:id/calibrate_btn' (suffix parity)
        assert not assert_element_not_present(SAMPLE_ELEMENTS, "calibrate_btn").passed

    def test_message_names_target(self) -> None:
        assert "ghost" in assert_element_not_present(SAMPLE_ELEMENTS, "ghost").message


class TestAssertTextNotPresent:
    def test_absent_passes(self) -> None:
        assert assert_text_not_present(SAMPLE_ELEMENTS, "no such text").passed

    def test_present_substring_fails(self) -> None:
        # 'Settings' element text; case-insensitive substring parity with assert_text
        assert not assert_text_not_present(SAMPLE_ELEMENTS, "setting").passed

    def test_present_case_insensitive_fails(self) -> None:
        assert not assert_text_not_present(SAMPLE_ELEMENTS, "measurement unit").passed


class TestAssertElementCount:
    def test_expected_match(self) -> None:
        # 3 'row' elements (two exact + one suffix)
        assert assert_element_count(COUNT_ELEMENTS, "row", expected=3).passed

    def test_expected_mismatch(self) -> None:
        r = assert_element_count(COUNT_ELEMENTS, "row", expected=2)
        assert not r.passed
        assert "3" in r.message  # observed count surfaced

    def test_min_only_pass(self) -> None:
        assert assert_element_count(COUNT_ELEMENTS, "row", min_count=1).passed

    def test_min_only_fail(self) -> None:
        assert not assert_element_count(COUNT_ELEMENTS, "row", min_count=5).passed

    def test_max_only_pass(self) -> None:
        assert assert_element_count(COUNT_ELEMENTS, "row", max_count=3).passed

    def test_range_pass(self) -> None:
        assert assert_element_count(COUNT_ELEMENTS, "row", min_count=2, max_count=4).passed

    def test_no_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            assert_element_count(COUNT_ELEMENTS, "row")

    def test_expected_with_range_raises(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            assert_element_count(COUNT_ELEMENTS, "row", expected=3, min_count=1)

    def test_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min"):
            assert_element_count(COUNT_ELEMENTS, "row", min_count=5, max_count=2)
