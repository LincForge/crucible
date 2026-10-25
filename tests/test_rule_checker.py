"""Tests for WCAGRuleChecker — platform-aware accessibility rule validation."""

from __future__ import annotations

from linc_device_kit import ElementNode, Platform, Rect, SemanticRole

from linc_crucible.core.rule_checker import RuleViolation, WCAGRuleChecker

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _el(
    id: str = "",
    text: str = "",
    desc: str = "",
    role: str = "button",
    clickable: bool = False,
    bounds: Rect | None = None,
    value: str | None = None,
    scrollable: bool = False,
    children: list | None = None,
) -> ElementNode:
    return ElementNode(
        id=id,
        text=text,
        description=desc,
        role=role,
        clickable=clickable,
        bounds=bounds,
        value=value,
        scrollable=scrollable,
        children=children or [],
    )


# ---------------------------------------------------------------------------
# Rule 1: missing_description
# ---------------------------------------------------------------------------


class TestMissingDescription:
    def test_clickable_without_desc_or_text_is_major(self):
        el = _el(id="btn1", clickable=True, desc="", text="")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_description" in rules
        v = next(v for v in violations if v.rule == "missing_description")
        assert v.severity == "major"
        assert v.element_id == "btn1"

    def test_clickable_with_desc_is_ok(self):
        el = _el(id="btn2", clickable=True, desc="Submit form", text="")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_description" not in rules

    def test_clickable_with_text_is_ok(self):
        el = _el(id="btn3", clickable=True, desc="", text="Submit")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_description" not in rules

    def test_non_clickable_without_desc_is_ok(self):
        el = _el(id="lbl1", clickable=False, desc="", text="")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_description" not in rules

    def test_image_role_without_desc_is_major(self):
        el = _el(id="img1", role=SemanticRole.IMAGE.value, desc="", text="", clickable=False)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_description" in rules
        v = next(v for v in violations if v.rule == "missing_description")
        assert v.severity == "major"

    def test_image_role_with_desc_is_ok(self):
        el = _el(id="img2", role=SemanticRole.IMAGE.value, desc="Company logo", text="")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_description" not in rules


# ---------------------------------------------------------------------------
# Rule 2: touch_target_size
# ---------------------------------------------------------------------------


class TestTouchTargetSize:
    def test_android_small_target_is_major(self):
        el = _el(id="btn_small", clickable=True, bounds=Rect(0, 0, 40, 40))
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "touch_target_size" in rules
        v = next(v for v in violations if v.rule == "touch_target_size")
        assert v.severity == "major"

    def test_android_minimum_target_is_ok(self):
        el = _el(id="btn_ok", clickable=True, bounds=Rect(0, 0, 48, 48))
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "touch_target_size" not in rules

    def test_ios_small_target_is_major(self):
        el = _el(id="btn_ios_small", clickable=True, bounds=Rect(0, 0, 40, 40))
        violations = WCAGRuleChecker.check([el], Platform.IOS)
        rules = [v.rule for v in violations]
        assert "touch_target_size" in rules
        v = next(v for v in violations if v.rule == "touch_target_size")
        assert v.severity == "major"

    def test_ios_minimum_target_is_ok(self):
        el = _el(id="btn_ios_ok", clickable=True, bounds=Rect(0, 0, 44, 44))
        violations = WCAGRuleChecker.check([el], Platform.IOS)
        rules = [v.rule for v in violations]
        assert "touch_target_size" not in rules

    def test_no_bounds_is_skipped(self):
        el = _el(id="btn_no_bounds", clickable=True, bounds=None)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "touch_target_size" not in rules

    def test_non_clickable_small_target_is_ok(self):
        el = _el(id="lbl_small", clickable=False, bounds=Rect(0, 0, 10, 10))
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "touch_target_size" not in rules


# ---------------------------------------------------------------------------
# Rule 3: empty_text
# ---------------------------------------------------------------------------


class TestEmptyText:
    def test_text_role_empty_text_is_minor(self):
        el = _el(id="txt1", role=SemanticRole.TEXT.value, text="")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "empty_text" in rules
        v = next(v for v in violations if v.rule == "empty_text")
        assert v.severity == "minor"

    def test_text_role_with_text_is_ok(self):
        el = _el(id="txt2", role=SemanticRole.TEXT.value, text="Hello world")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "empty_text" not in rules

    def test_non_text_role_empty_is_ok(self):
        el = _el(id="btn_empty", role=SemanticRole.BUTTON.value, text="")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "empty_text" not in rules


# ---------------------------------------------------------------------------
# Rule 4: duplicate_descriptions
# ---------------------------------------------------------------------------


class TestDuplicateDescriptions:
    def test_two_elements_same_desc_is_minor(self):
        els = [
            _el(id="a", desc="Submit"),
            _el(id="b", desc="Submit"),
        ]
        violations = WCAGRuleChecker.check(els, Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "duplicate_descriptions" in rules
        v = next(v for v in violations if v.rule == "duplicate_descriptions")
        assert v.severity == "minor"

    def test_unique_descriptions_are_ok(self):
        els = [
            _el(id="a", desc="Submit"),
            _el(id="b", desc="Cancel"),
        ]
        violations = WCAGRuleChecker.check(els, Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "duplicate_descriptions" not in rules

    def test_empty_descriptions_not_flagged(self):
        els = [
            _el(id="a", desc=""),
            _el(id="b", desc=""),
        ]
        violations = WCAGRuleChecker.check(els, Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "duplicate_descriptions" not in rules


# ---------------------------------------------------------------------------
# Rule 5: missing_headings
# ---------------------------------------------------------------------------


class TestMissingHeadings:
    def test_scrollable_with_10_children_no_heading_is_minor(self):
        children = [
            _el(id=f"child_{i}", role=SemanticRole.TEXT.value, text=f"Item {i}") for i in range(10)
        ]
        el = _el(id="scroll1", scrollable=True, children=children)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_headings" in rules
        v = next(v for v in violations if v.rule == "missing_headings")
        assert v.severity == "minor"

    def test_scrollable_with_heading_child_is_ok(self):
        children = [
            _el(id="heading1", role=SemanticRole.HEADING.value, text="Section"),
        ] + [
            _el(id=f"child_{i}", role=SemanticRole.TEXT.value, text=f"Item {i}") for i in range(9)
        ]
        el = _el(id="scroll2", scrollable=True, children=children)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_headings" not in rules

    def test_scrollable_with_fewer_than_5_children_is_ok(self):
        children = [
            _el(id=f"child_{i}", role=SemanticRole.TEXT.value, text=f"Item {i}") for i in range(4)
        ]
        el = _el(id="scroll3", scrollable=True, children=children)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_headings" not in rules

    def test_non_scrollable_with_many_children_is_ok(self):
        children = [
            _el(id=f"child_{i}", role=SemanticRole.TEXT.value, text=f"Item {i}") for i in range(10)
        ]
        el = _el(id="container1", scrollable=False, children=children)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "missing_headings" not in rules


# ---------------------------------------------------------------------------
# Rule 6: value_not_announced
# ---------------------------------------------------------------------------


class TestValueNotAnnounced:
    def test_slider_without_value_is_major(self):
        el = _el(id="slider1", role=SemanticRole.SLIDER.value, value=None)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "value_not_announced" in rules
        v = next(v for v in violations if v.rule == "value_not_announced")
        assert v.severity == "major"

    def test_slider_with_value_is_ok(self):
        el = _el(id="slider2", role=SemanticRole.SLIDER.value, value="50%")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "value_not_announced" not in rules

    def test_switch_without_value_is_major(self):
        el = _el(id="switch1", role=SemanticRole.SWITCH.value, value=None)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "value_not_announced" in rules
        v = next(v for v in violations if v.rule == "value_not_announced")
        assert v.severity == "major"

    def test_switch_with_value_is_ok(self):
        el = _el(id="switch2", role=SemanticRole.SWITCH.value, value="on")
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "value_not_announced" not in rules

    def test_button_without_value_is_ok(self):
        el = _el(id="btn_noval", role=SemanticRole.BUTTON.value, value=None)
        violations = WCAGRuleChecker.check([el], Platform.ANDROID)
        rules = [v.rule for v in violations]
        assert "value_not_announced" not in rules


# ---------------------------------------------------------------------------
# violations_to_score
# ---------------------------------------------------------------------------


class TestViolationsToScore:
    def test_no_violations_gives_100(self):
        assert WCAGRuleChecker.violations_to_score([]) == 100

    def test_critical_penalty_is_10(self):
        v = RuleViolation(rule="x", element_id="e", severity="critical", message="m")
        assert WCAGRuleChecker.violations_to_score([v]) == 90

    def test_major_penalty_is_5(self):
        v = RuleViolation(rule="x", element_id="e", severity="major", message="m")
        assert WCAGRuleChecker.violations_to_score([v]) == 95

    def test_minor_penalty_is_2(self):
        v = RuleViolation(rule="x", element_id="e", severity="minor", message="m")
        assert WCAGRuleChecker.violations_to_score([v]) == 98

    def test_multiple_violations_accumulate(self):
        violations = [
            RuleViolation(rule="a", element_id="e1", severity="critical", message="m"),  # -10
            RuleViolation(rule="b", element_id="e2", severity="major", message="m"),  # -5
            RuleViolation(rule="c", element_id="e3", severity="minor", message="m"),  # -2
        ]
        assert WCAGRuleChecker.violations_to_score(violations) == 83

    def test_score_floors_at_zero(self):
        violations = [
            RuleViolation(rule="x", element_id=f"e{i}", severity="critical", message="m")
            for i in range(15)  # 15 * 10 = 150 penalty
        ]
        assert WCAGRuleChecker.violations_to_score(violations) == 0


# ---------------------------------------------------------------------------
# RuleViolation.to_dict
# ---------------------------------------------------------------------------


class TestRuleViolationToDict:
    def test_to_dict_returns_expected_keys(self):
        v = RuleViolation(
            rule="missing_description",
            element_id="btn1",
            severity="major",
            message="No description",
        )
        d = v.to_dict()
        assert d["rule"] == "missing_description"
        assert d["element_id"] == "btn1"
        assert d["severity"] == "major"
        assert d["message"] == "No description"
