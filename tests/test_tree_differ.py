from __future__ import annotations

from linc_device_kit.models import ElementNode

from linc_crucible.core.tree_differ import TreeDiffer


def _el(
    id: str,
    text: str = "",
    desc: str = "",
    role: str = "button",
    clickable: bool = False,
    enabled: bool = True,
) -> ElementNode:
    return ElementNode(
        id=id,
        text=text,
        description=desc,
        role=role,
        clickable=clickable,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# 1. Identical lists → score 100, empty diff
# ---------------------------------------------------------------------------
def test_identical_lists_score_100() -> None:
    baseline = [_el("a"), _el("b")]
    current = [_el("a"), _el("b")]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 100
    assert report.missing == []
    assert report.added == []
    assert report.changed == []


# ---------------------------------------------------------------------------
# 2. Empty lists → score 100
# ---------------------------------------------------------------------------
def test_empty_lists_score_100() -> None:
    report = TreeDiffer.diff([], [])
    assert report.score == 100
    assert report.missing == []
    assert report.added == []
    assert report.changed == []


# ---------------------------------------------------------------------------
# 3. 1 missing element → score 90, 1 missing entry
# ---------------------------------------------------------------------------
def test_one_missing_element() -> None:
    baseline = [_el("a")]
    current: list[ElementNode] = []
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 90
    assert len(report.missing) == 1
    assert report.missing[0].element_id == "a"
    assert report.missing[0].change_type == "missing"


# ---------------------------------------------------------------------------
# 4. 3 missing elements → score 70
# ---------------------------------------------------------------------------
def test_three_missing_elements() -> None:
    baseline = [_el("a"), _el("b"), _el("c")]
    current: list[ElementNode] = []
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 70


# ---------------------------------------------------------------------------
# 5. 1 added element → score 99
# ---------------------------------------------------------------------------
def test_one_added_element() -> None:
    baseline: list[ElementNode] = []
    current = [_el("x")]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 99
    assert len(report.added) == 1
    assert report.added[0].element_id == "x"
    assert report.added[0].change_type == "added"


# ---------------------------------------------------------------------------
# 6. 10 added elements → score 90
# ---------------------------------------------------------------------------
def test_ten_added_elements() -> None:
    baseline: list[ElementNode] = []
    current = [_el(f"x{i}") for i in range(10)]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 90


# ---------------------------------------------------------------------------
# 7. Changed text → score 95, change_type="changed_label"
# ---------------------------------------------------------------------------
def test_changed_text_label() -> None:
    baseline = [_el("a", text="Hello")]
    current = [_el("a", text="World")]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 95
    assert len(report.changed) == 1
    assert report.changed[0].element_id == "a"
    assert report.changed[0].change_type == "changed_label"


# ---------------------------------------------------------------------------
# 8. Changed description → score 95, change_type="changed_label"
# ---------------------------------------------------------------------------
def test_changed_description_label() -> None:
    baseline = [_el("a", desc="Old desc")]
    current = [_el("a", desc="New desc")]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 95
    assert len(report.changed) == 1
    assert report.changed[0].change_type == "changed_label"


# ---------------------------------------------------------------------------
# 9. Changed role → score 92, change_type="changed_role"
# ---------------------------------------------------------------------------
def test_changed_role() -> None:
    baseline = [_el("a", role="button")]
    current = [_el("a", role="checkbox")]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 92
    assert len(report.changed) == 1
    assert report.changed[0].change_type == "changed_role"
    assert report.changed[0].baseline_value == "button"
    assert report.changed[0].current_value == "checkbox"


# ---------------------------------------------------------------------------
# 10. Changed clickable → score 97, change_type="changed_interactivity"
# ---------------------------------------------------------------------------
def test_changed_clickable() -> None:
    baseline = [_el("a", clickable=False)]
    current = [_el("a", clickable=True)]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 97
    assert len(report.changed) == 1
    assert report.changed[0].change_type == "changed_interactivity"


# ---------------------------------------------------------------------------
# 11. Changed enabled → score 97
# ---------------------------------------------------------------------------
def test_changed_enabled() -> None:
    baseline = [_el("a", enabled=True)]
    current = [_el("a", enabled=False)]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 97
    assert len(report.changed) == 1
    assert report.changed[0].change_type == "changed_interactivity"


# ---------------------------------------------------------------------------
# 12. Same element with role + label change → score 87 (both penalties)
# ---------------------------------------------------------------------------
def test_role_and_label_change_both_penalized() -> None:
    baseline = [_el("a", role="button", text="Submit")]
    current = [_el("a", role="checkbox", text="Agree")]
    report = TreeDiffer.diff(baseline, current)
    # -8 role, -5 label = -13 → score 87
    assert report.score == 87
    assert len(report.changed) == 2


# ---------------------------------------------------------------------------
# 13. Compound: 1 missing + 1 changed label + 1 added → score 84
# ---------------------------------------------------------------------------
def test_compound_missing_changed_added() -> None:
    baseline = [_el("a"), _el("b", text="Old")]
    current = [_el("b", text="New"), _el("c")]
    report = TreeDiffer.diff(baseline, current)
    # -10 missing "a", -5 changed_label "b", -1 added "c" = -16 → score 84
    assert report.score == 84
    assert len(report.missing) == 1
    assert len(report.changed) == 1
    assert len(report.added) == 1


# ---------------------------------------------------------------------------
# 14. Score floors at 0 (20 missing elements)
# ---------------------------------------------------------------------------
def test_score_floors_at_zero() -> None:
    baseline = [_el(f"el{i}") for i in range(20)]
    current: list[ElementNode] = []
    report = TreeDiffer.diff(baseline, current)
    # -200 penalty capped at 0
    assert report.score == 0


# ---------------------------------------------------------------------------
# 15. DiffReport.to_dict() serialization
# ---------------------------------------------------------------------------
def test_diff_report_to_dict() -> None:
    baseline = [_el("a"), _el("b", text="Hello")]
    current = [_el("b", text="World"), _el("c")]
    report = TreeDiffer.diff(baseline, current)
    d = report.to_dict()
    assert "score" in d
    assert "missing" in d
    assert "added" in d
    assert "changed" in d
    assert "baseline_count" in d
    assert "current_count" in d
    assert isinstance(d["missing"], list)
    assert isinstance(d["added"], list)
    assert isinstance(d["changed"], list)
    assert d["score"] == report.score
    # DiffEntry serialization
    if d["missing"]:
        entry = d["missing"][0]
        assert "element_id" in entry
        assert "change_type" in entry


# ---------------------------------------------------------------------------
# 16. Nested children are flattened for comparison
# ---------------------------------------------------------------------------
def test_nested_children_flattened() -> None:
    child = _el("child", text="Child text")
    parent = ElementNode(
        id="parent",
        role="layout",
        children=[child],
    )
    baseline = [parent]
    # current has both as flat list — child still present, parent still present
    current = [
        ElementNode(id="parent", role="layout"),
        _el("child", text="Child text"),
    ]
    report = TreeDiffer.diff(baseline, current)
    assert report.score == 100
    assert report.missing == []
    assert report.changed == []


def test_nested_children_missing_child_detected() -> None:
    child = _el("child-id", text="Button")
    parent = ElementNode(id="par", role="layout", children=[child])
    # current: parent present but child removed
    current = [ElementNode(id="par", role="layout")]
    report = TreeDiffer.diff([parent], current)
    # child is missing → -10
    assert report.score == 90
    assert any(e.element_id == "child-id" for e in report.missing)
