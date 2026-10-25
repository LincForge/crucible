from __future__ import annotations

from dataclasses import dataclass, field

from linc_device_kit.models import ElementNode

# ---------------------------------------------------------------------------
# Scoring penalties
# ---------------------------------------------------------------------------
PENALTY_MISSING = 10
PENALTY_CHANGED_ROLE = 8
PENALTY_CHANGED_LABEL = 5
PENALTY_CHANGED_INTERACTIVITY = 3
PENALTY_ADDED = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DiffEntry:
    """A single structural difference between baseline and current."""

    element_id: str
    change_type: (
        str  # "missing" | "added" | "changed_role" | "changed_label" | "changed_interactivity"
    )
    baseline_value: str = ""
    current_value: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "element_id": self.element_id,
            "change_type": self.change_type,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
        }


@dataclass
class DiffReport:
    """Full diff result between two element trees."""

    score: int = 100
    missing: list[DiffEntry] = field(default_factory=list)
    added: list[DiffEntry] = field(default_factory=list)
    changed: list[DiffEntry] = field(default_factory=list)
    baseline_count: int = 0
    current_count: int = 0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "missing": [e.to_dict() for e in self.missing],
            "added": [e.to_dict() for e in self.added],
            "changed": [e.to_dict() for e in self.changed],
            "baseline_count": self.baseline_count,
            "current_count": self.current_count,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(elements: list[ElementNode]) -> list[ElementNode]:
    """Recursively flatten nested children into a single list."""
    result: list[ElementNode] = []
    for el in elements:
        result.append(el)
        if el.children:
            result.extend(_flatten(el.children))
    return result


# ---------------------------------------------------------------------------
# TreeDiffer
# ---------------------------------------------------------------------------


class TreeDiffer:
    """Structural diff of two ElementNode lists, scored 0-100."""

    @staticmethod
    def _check_changed(b: ElementNode, c: ElementNode, eid: str) -> tuple[list[DiffEntry], int]:
        changed: list[DiffEntry] = []
        deduction = 0
        if b.role != c.role:
            changed.append(
                DiffEntry(
                    element_id=eid,
                    change_type="changed_role",
                    baseline_value=b.role,
                    current_value=c.role,
                )
            )
            deduction += PENALTY_CHANGED_ROLE

        if b.text != c.text or b.description != c.description:
            changed.append(
                DiffEntry(
                    element_id=eid,
                    change_type="changed_label",
                    baseline_value=b.text or b.description,
                    current_value=c.text or c.description,
                )
            )
            deduction += PENALTY_CHANGED_LABEL

        if b.clickable != c.clickable or b.enabled != c.enabled:
            changed.append(
                DiffEntry(
                    element_id=eid,
                    change_type="changed_interactivity",
                    baseline_value=f"clickable={b.clickable},enabled={b.enabled}",
                    current_value=f"clickable={c.clickable},enabled={c.enabled}",
                )
            )
            deduction += PENALTY_CHANGED_INTERACTIVITY

        return changed, deduction

    @staticmethod
    def diff(
        baseline: list[ElementNode],
        current: list[ElementNode],
    ) -> DiffReport:
        flat_baseline = _flatten(baseline)
        flat_current = _flatten(current)

        # Index by id, skip empty ids
        baseline_map: dict[str, ElementNode] = {el.id: el for el in flat_baseline if el.id}
        current_map: dict[str, ElementNode] = {el.id: el for el in flat_current if el.id}

        missing: list[DiffEntry] = []
        added: list[DiffEntry] = []
        changed: list[DiffEntry] = []
        deduction = 0

        # Missing: in baseline but not in current
        for eid in baseline_map:
            if eid not in current_map:
                missing.append(DiffEntry(element_id=eid, change_type="missing"))
                deduction += PENALTY_MISSING

        # Added: in current but not in baseline
        for eid in current_map:
            if eid not in baseline_map:
                added.append(DiffEntry(element_id=eid, change_type="added"))
                deduction += PENALTY_ADDED

        # Changed: present in both — check role, label, interactivity
        for eid in baseline_map:
            if eid not in current_map:
                continue
            b = baseline_map[eid]
            c = current_map[eid]
            c_entries, c_deduct = TreeDiffer._check_changed(b, c, eid)
            changed.extend(c_entries)
            deduction += c_deduct

        score = max(0, 100 - deduction)

        return DiffReport(
            score=score,
            missing=missing,
            added=added,
            changed=changed,
            baseline_count=len(flat_baseline),
            current_count=len(flat_current),
        )
