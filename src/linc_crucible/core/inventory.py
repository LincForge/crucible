"""Honest-inventory fields for a zero-entry store result.

A bare ``count: 0`` reads identically for "no entries match this filter" and
"the store has never existed on this host" (2026-08-06 device-verification
audit: the baseline store was absent on the mini while `crucible_baseline
list` succeeded vacuously — see ``tests/test_baseline_inventory_guard.py``).
This module lifts that fix out of ``baseline_tools`` so any zero-entry store
lookup (baselines, scores, ...) can say explicitly whether the store root
exists and where it was looked for, instead of each caller re-deriving its
own vacuous-empty message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Verb used in the "store does not exist" message, keyed by `kind`. Falls
# back to "captured" (the original baseline-store wording) for any kind not
# listed here.
_NEVER_HAPPENED_VERBS: dict[str, str] = {
    "scores": "assayed",
}


def empty_inventory(store_root: Path, scope: str, kind: str) -> dict[str, Any]:
    """Build the standard fields for a zero-entry inventory result.

    Args:
        store_root: Directory the caller looked in for entries.
        scope: Human-readable description of what was searched for, e.g.
            ``"project 'myapp'"`` or ``"project 'X', screen 'home'"``.
        kind: Human-readable noun for what's stored, e.g. ``"scores"`` or
            ``"screenshot baselines"``. Also selects the "nothing has ever
            been ___" verb via `_NEVER_HAPPENED_VERBS`.

    Returns:
        A dict with ``empty: True``, ``store_root`` (str), ``store_exists``
        (bool), and a ``message`` explaining which of the two it is.
    """
    exists = store_root.is_dir()
    if exists:
        message = f"No {kind} stored for {scope} (store {store_root} exists)."
    else:
        verb = _NEVER_HAPPENED_VERBS.get(kind, "captured")
        message = (
            f"No {kind}: store {store_root} does not exist on this host — "
            f"nothing has ever been {verb}."
        )
    return {
        "empty": True,
        "store_root": str(store_root),
        "store_exists": exists,
        "message": message,
    }
