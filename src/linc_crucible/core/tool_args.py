"""Tool argument validation primitives and per-tool precheck helpers.

Enforces strict type, length, character, and numeric range bounds across all
Crucible MCP/CLI tool entrypoints (rejecting NaN and lax type coercion).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from linc_crucible.core.store_names import (
    InvalidStoreName,
    _preview,
    required_names_error,
    required_project_error,
    validate_name,
    validation_error,
)

BUILD_TYPE_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9]{0,31}\Z")

MAX_SHORT_STR_LEN = 256
MAX_PATH_LEN = 1024
MAX_TAGS_COUNT = 64
MAX_TAG_ITEM_LEN = 64

MAX_ASSETS_MIN = 1
MAX_ASSETS_MAX = 50
TOP_N_MIN = 1
TOP_N_MAX = 1000
LIMIT_MIN = 1
LIMIT_MAX = 1000
KEEP_MIN = 0
KEEP_MAX = 1000
SINCE_HOURS_MIN = 0.0
SINCE_HOURS_MAX = 87600.0
BUDGET_S_MIN = 0.0
BUDGET_S_MAX = 86400.0


class InvalidToolArg(ValueError):
    """Raised when a tool argument violates its type, format, or range contract."""

    def __init__(self, field: str, reason: str, value: object) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid {field}: {reason} (got {_preview(value)})")


def check_str(
    value: object,
    field: str,
    max_len: int = MAX_SHORT_STR_LEN,
    allow_empty: bool = True,
) -> str:
    """Validate that ``value`` is a ``str`` without NUL bytes and within ``max_len``."""
    if not isinstance(value, str):
        raise InvalidToolArg(field, "must be a string", value)
    if "\x00" in value:
        raise InvalidToolArg(field, "must not contain NUL bytes", value)
    if not allow_empty and value == "":
        raise InvalidToolArg(field, "must not be empty", value)
    if len(value) > max_len:
        raise InvalidToolArg(field, f"must be at most {max_len} characters", value)
    return value


def check_optional_str(
    value: object,
    field: str,
    max_len: int = MAX_PATH_LEN,
    *,
    allow_empty: bool = False,
) -> str | None:
    """Validate an optional string argument (``None`` or valid ``str``)."""
    if value is None:
        return None
    res = check_str(value, field, max_len=max_len, allow_empty=allow_empty)
    return res


def check_int(value: object, field: str, lo: int, hi: int) -> int:
    """Validate that ``value`` is an ``int`` (not ``bool``) in ``[lo, hi]``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidToolArg(field, "must be an integer", value)
    if value < lo or value > hi:
        raise InvalidToolArg(field, f"must be between {lo} and {hi}", value)
    return value


def check_finite(value: object, field: str, lo: float, hi: float) -> float:
    """Validate that ``value`` is a finite ``int`` or ``float`` (not ``bool``) in ``[lo, hi]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidToolArg(field, "must be a number", value)
    fval = float(value)
    if not math.isfinite(fval):
        raise InvalidToolArg(field, "must be finite", value)
    if fval < lo or fval > hi:
        raise InvalidToolArg(field, f"must be between {lo} and {hi}", value)
    return fval


def check_optional_finite(value: object, field: str, lo: float, hi: float) -> float | None:
    """Validate an optional finite float argument (``None`` or finite number in ``[lo, hi]``)."""
    if value is None:
        return None
    res = check_finite(value, field, lo, hi)
    return res


def check_bool(value: object, field: str) -> bool:
    """Validate that ``value`` is strictly a ``bool``."""
    if not isinstance(value, bool):
        raise InvalidToolArg(field, "must be a boolean", value)
    return value


def check_build_type(value: object, field: str = "build_type") -> str:
    """Validate that ``value`` matches ``\\A[A-Za-z][A-Za-z0-9]{0,31}\\Z``."""
    s = check_str(value, field, max_len=32, allow_empty=False)
    if not BUILD_TYPE_RE.match(s):
        raise InvalidToolArg(field, "must match [A-Za-z][A-Za-z0-9]{0,31}", value)
    return s


def check_relative_dir(
    value: object,
    field: str,
    max_len: int = MAX_SHORT_STR_LEN,
) -> str:
    """Validate that ``value`` is a non-empty relative path without ``..`` traversal."""
    s = check_str(value, field, max_len=max_len, allow_empty=False)
    path = Path(s)
    if path.is_absolute():
        raise InvalidToolArg(field, "must be a relative path", value)
    if ".." in path.parts:
        raise InvalidToolArg(field, "must not contain '..' segments", value)
    return s


def check_str_list(
    value: object,
    field: str,
    max_items: int = MAX_TAGS_COUNT,
    max_len: int = MAX_TAG_ITEM_LEN,
) -> list[str] | None:
    """Validate that ``value`` is ``None`` or a bounded ``list`` of non-empty strings."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise InvalidToolArg(field, "must be a list of strings", value)
    if len(value) > max_items:
        raise InvalidToolArg(field, f"must contain at most {max_items} items", value)
    for idx, item in enumerate(value):
        check_str(item, f"{field}[{idx}]", max_len=max_len, allow_empty=False)
    return value


_Spec = tuple[str, Callable[..., Any], tuple[Any, ...]]

_POUR_SPECS: tuple[_Spec, ...] = (
    ("device", check_str, ()),
    ("platform", check_str, ()),
    ("module", check_str, ()),
    ("device_type", check_str, ()),
    ("launch_intent", check_str, ()),
    ("build_type", check_build_type, ()),
    ("launch", check_bool, ()),
    ("skip_build", check_bool, ()),
    ("build_only", check_bool, ()),
    ("allow_personal", check_bool, ()),
)

_ASSAY_SPECS: tuple[_Spec, ...] = (
    ("mode", check_str, ()),
    ("device", check_str, ()),
    ("perf", check_bool, ()),
    ("a11y", check_bool, ()),
    ("a11y_tree", check_bool, ()),
    ("allow_personal", check_bool, ()),
)

_SCORE_SPECS: tuple[_Spec, ...] = (
    ("history", check_bool, ()),
    ("limit", check_int, (LIMIT_MIN, LIMIT_MAX)),
)

_BVT_SPECS: tuple[_Spec, ...] = (
    ("project", validate_name, ()),
    ("device", check_str, (MAX_SHORT_STR_LEN, False)),
    ("suite_dir", check_relative_dir, ()),
    ("tags", check_str_list, ()),
    ("exclude_tags", check_str_list, ()),
    ("output_dir", check_optional_str, ()),
    ("app_path", check_optional_str, ()),
    ("budget_s", check_optional_finite, (BUDGET_S_MIN, BUDGET_S_MAX)),
    ("compare_against_baseline", check_bool, ()),
    ("allow_personal", check_bool, ()),
)

_EXTRACT_ASSETS_SPECS: tuple[_Spec, ...] = (
    ("video_path", check_str, (MAX_PATH_LEN, False)),
    ("max_assets", check_int, (MAX_ASSETS_MIN, MAX_ASSETS_MAX)),
)

_USAGE_STATS_SPECS: tuple[_Spec, ...] = (
    ("server_filter", check_str, ()),
    ("since_hours", check_finite, (SINCE_HOURS_MIN, SINCE_HOURS_MAX)),
    ("include_rotated", check_bool, ()),
    ("top_n", check_int, (TOP_N_MIN, TOP_N_MAX)),
)

_BASELINE_SPECS: tuple[_Spec, ...] = (
    ("action", check_str, (MAX_SHORT_STR_LEN, False)),
    ("image_path", check_str, (MAX_PATH_LEN,)),
    ("device_info", check_str, ()),
    ("app_version", check_str, ()),
    ("keep", check_int, (KEEP_MIN, KEEP_MAX)),
    ("mode", check_str, (MAX_SHORT_STR_LEN, False)),
    ("device", check_str, ()),
    ("source_score", check_optional_finite, (0.0, 1.0)),
    ("allow_empty", check_bool, ()),
    ("allow_personal", check_bool, ()),
)


def _run_specs(specs: tuple[_Spec, ...], values: dict[str, object]) -> dict[str, Any] | None:
    """Run declarative ``(field, fn, extra_args)`` validators against ``values``."""
    try:
        for field, fn, extra in specs:
            fn(values[field], field, *extra)
    except (InvalidToolArg, InvalidStoreName) as exc:
        err = validation_error(exc)
        return err
    return None


def _run_specs_json(specs: tuple[_Spec, ...], values: dict[str, object]) -> str | None:
    """Run declarative ``(field, fn, extra_args)`` validators and return JSON or ``None``."""
    err = _run_specs(specs, values)
    out = json.dumps(err) if err is not None else None
    return out


def pour_args_error(
    project: object,
    device: object,
    platform: object,
    launch: object,
    skip_build: object,
    module: object,
    device_type: object,
    launch_intent: object,
    build_only: object,
    build_type: object,
    allow_personal: object,
) -> dict[str, Any] | None:
    """Precheck arguments for ``crucible_pour``."""
    err = required_project_error(project) or _run_specs(_POUR_SPECS, locals())
    return err


def assay_args_error(
    project: object,
    mode: object,
    perf: object,
    a11y: object,
    a11y_tree: object,
    device: object,
    allow_personal: object,
) -> dict[str, Any] | None:
    """Precheck arguments for ``crucible_assay``."""
    err = required_project_error(project) or _run_specs(_ASSAY_SPECS, locals())
    return err


def score_args_error(
    project: object,
    history: object,
    limit: object,
) -> dict[str, Any] | None:
    """Precheck arguments for ``crucible_score``."""
    err = required_project_error(project) or _run_specs(_SCORE_SPECS, locals())
    return err


def bvt_args_error(
    project: object,
    device: object,
    suite_dir: object,
    tags: object,
    exclude_tags: object,
    output_dir: object,
    compare_against_baseline: object,
    app_path: object,
    budget_s: object,
    allow_personal: object,
) -> str | None:
    """Precheck arguments for ``crucible_bvt`` (returns JSON string or ``None``)."""
    out = _run_specs_json(_BVT_SPECS, locals())
    return out


def extract_assets_args_error(video_path: object, max_assets: object) -> str | None:
    """Precheck arguments for ``crucible_extract_assets`` (returns JSON string or ``None``)."""
    out = _run_specs_json(_EXTRACT_ASSETS_SPECS, locals())
    return out


def usage_stats_args_error(
    server_filter: object,
    since_hours: object,
    include_rotated: object,
    top_n: object,
) -> dict[str, Any] | None:
    """Precheck arguments for ``crucible_usage_stats``."""
    err = _run_specs(_USAGE_STATS_SPECS, locals())
    return err


def baseline_args_error(
    action: object,
    project: object,
    screen_name: object,
    image_path: object,
    device_info: object,
    app_version: object,
    keep: object,
    mode: object,
    device: object,
    source_score: object,
    allow_empty: object,
    allow_personal: object,
) -> dict[str, Any] | None:
    """Precheck arguments for ``crucible_baseline``."""
    err = required_names_error(project, screen_name) or _run_specs(_BASELINE_SPECS, locals())
    return err
