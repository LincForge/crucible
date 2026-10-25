"""WCAG RuleChecker — platform-aware accessibility rule validation."""

from __future__ import annotations

from dataclasses import dataclass

from linc_device_kit import ElementNode, Platform, SemanticRole

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOUCH_MIN_ANDROID: int = 48  # dp
TOUCH_MIN_IOS: int = 44  # pt

VALUE_ROLES: frozenset[str] = frozenset({SemanticRole.SLIDER.value, SemanticRole.SWITCH.value})
IMAGE_ROLES: frozenset[str] = frozenset({SemanticRole.IMAGE.value})

SCROLLABLE_CHILD_THRESHOLD: int = 5

PENALTY_CRITICAL: int = 10
PENALTY_MAJOR: int = 5
PENALTY_MINOR: int = 2


# ---------------------------------------------------------------------------
# RuleViolation
# ---------------------------------------------------------------------------


@dataclass
class RuleViolation:
    rule: str
    element_id: str
    severity: str  # "critical" | "major" | "minor"
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "element_id": self.element_id,
            "severity": self.severity,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(elements: list[ElementNode]) -> list[ElementNode]:
    """Flatten a nested element tree into a single list."""
    result: list[ElementNode] = []
    stack = list(elements)
    while stack:
        el = stack.pop()
        result.append(el)
        stack.extend(el.children)
    return result


def _has_heading_descendant(children: list[ElementNode]) -> bool:
    """Recursively check if any descendant has role=HEADING."""
    for child in children:
        if child.role == SemanticRole.HEADING.value:
            return True
        if _has_heading_descendant(child.children):
            return True
    return False


# ---------------------------------------------------------------------------
# WCAGRuleChecker
# ---------------------------------------------------------------------------


class WCAGRuleChecker:
    """Run WCAG-aligned accessibility rules against an ElementNode list."""

    @staticmethod
    def check(elements: list[ElementNode], platform: Platform) -> list[RuleViolation]:
        """Run all 6 rules and return a flat list of violations."""
        violations: list[RuleViolation] = []
        flat = _flatten(elements)

        WCAGRuleChecker._check_missing_description(flat, violations)
        WCAGRuleChecker._check_touch_target_size(flat, platform, violations)
        WCAGRuleChecker._check_empty_text(flat, violations)
        WCAGRuleChecker._check_duplicate_descriptions(flat, violations)
        WCAGRuleChecker._check_value_not_announced(flat, violations)

        # Rule 5: missing_headings (recursive check via _check_scrollable_headings)
        for el in elements:
            WCAGRuleChecker._check_scrollable_headings(el, violations)

        return violations

    @staticmethod
    def _check_missing_description(
        flat: list[ElementNode], violations: list[RuleViolation]
    ) -> None:
        """Rule 1: missing_description"""
        for el in flat:
            is_image = el.role in IMAGE_ROLES
            if (el.clickable or is_image) and not el.description and not el.text:
                kind = "interactive" if el.clickable else "an image"
                violations.append(
                    RuleViolation(
                        rule="missing_description",
                        element_id=el.id,
                        severity="major",
                        message=(f"Element '{el.id}' is {kind} but has no description or text."),
                    )
                )

    @staticmethod
    def _check_touch_target_size(
        flat: list[ElementNode], platform: Platform, violations: list[RuleViolation]
    ) -> None:
        """Rule 2: touch_target_size"""
        min_size = TOUCH_MIN_ANDROID if platform == Platform.ANDROID else TOUCH_MIN_IOS
        for el in flat:
            if (
                el.clickable
                and el.bounds is not None
                and (el.bounds.width < min_size or el.bounds.height < min_size)
            ):
                w, h = el.bounds.width, el.bounds.height
                violations.append(
                    RuleViolation(
                        rule="touch_target_size",
                        element_id=el.id,
                        severity="major",
                        message=(
                            f"Element '{el.id}' touch target {w}x{h} "
                            f"is below minimum {min_size}x{min_size} for {platform.value}."
                        ),
                    )
                )

    @staticmethod
    def _check_empty_text(flat: list[ElementNode], violations: list[RuleViolation]) -> None:
        """Rule 3: empty_text"""
        for el in flat:
            if el.role == SemanticRole.TEXT.value and not el.text:
                violations.append(
                    RuleViolation(
                        rule="empty_text",
                        element_id=el.id,
                        severity="minor",
                        message=f"Text element '{el.id}' has role=TEXT but empty text.",
                    )
                )

    @staticmethod
    def _check_duplicate_descriptions(
        flat: list[ElementNode], violations: list[RuleViolation]
    ) -> None:
        """Rule 4: duplicate_descriptions"""
        desc_counts: dict[str, list[str]] = {}
        for el in flat:
            if el.description:
                desc_counts.setdefault(el.description, []).append(el.id)
        for desc, ids in desc_counts.items():
            if len(ids) > 1:
                others = len(ids) - 1
                for eid in ids:
                    violations.append(
                        RuleViolation(
                            rule="duplicate_descriptions",
                            element_id=eid,
                            severity="minor",
                            message=(
                                f"Element '{eid}' shares description '{desc}' "
                                f"with {others} other element(s)."
                            ),
                        )
                    )

    @staticmethod
    def _check_value_not_announced(
        flat: list[ElementNode], violations: list[RuleViolation]
    ) -> None:
        """Rule 6: value_not_announced"""
        for el in flat:
            if el.role in VALUE_ROLES and el.value is None:
                violations.append(
                    RuleViolation(
                        rule="value_not_announced",
                        element_id=el.id,
                        severity="major",
                        message=f"Element '{el.id}' has role={el.role} but no value announced.",
                    )
                )

    @staticmethod
    def _check_scrollable_headings(el: ElementNode, violations: list[RuleViolation]) -> None:
        """Recursively check scrollable containers for missing headings."""
        if (
            el.scrollable
            and len(el.children) >= SCROLLABLE_CHILD_THRESHOLD
            and not _has_heading_descendant(el.children)
        ):
            violations.append(
                RuleViolation(
                    rule="missing_headings",
                    element_id=el.id,
                    severity="minor",
                    message=(
                        f"Scrollable element '{el.id}' has {len(el.children)} children "
                        "but no heading descendant."
                    ),
                )
            )
        for child in el.children:
            WCAGRuleChecker._check_scrollable_headings(child, violations)

    @staticmethod
    def violations_to_score(violations: list[RuleViolation]) -> int:
        """Convert a violation list to a 0-100 quality score."""
        penalty = 0
        for v in violations:
            if v.severity == "critical":
                penalty += PENALTY_CRITICAL
            elif v.severity == "major":
                penalty += PENALTY_MAJOR
            elif v.severity == "minor":
                penalty += PENALTY_MINOR
        return max(0, 100 - penalty)
