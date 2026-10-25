"""Tests for tool input validation, build_type allowlist, and .forge.yaml command guards."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import structlog.testing
from click.testing import CliRunner

from linc_crucible import server_context as ctx
from linc_crucible.assay_tools import crucible_assay, crucible_preflight, crucible_score
from linc_crucible.asset_tools import crucible_extract_assets
from linc_crucible.baseline_tools import crucible_baseline
from linc_crucible.bvt_tools import _attach_visual_compare, crucible_bvt
from linc_crucible.cli import cli
from linc_crucible.core.assayer import Assayer, _missing_lint_result, _missing_test_result
from linc_crucible.core.deployer import Deployer
from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader, ModuleConfig
from linc_crucible.core.scorer import QualityScorer
from linc_crucible.core.subprocess_utils import CommandParseError, safe_subprocess_run
from linc_crucible.core.tool_args import (
    BUDGET_S_MAX,
    BUDGET_S_MIN,
    BUILD_TYPE_RE,
    KEEP_MAX,
    KEEP_MIN,
    LIMIT_MAX,
    LIMIT_MIN,
    MAX_ASSETS_MAX,
    MAX_ASSETS_MIN,
    MAX_PATH_LEN,
    MAX_SHORT_STR_LEN,
    MAX_TAG_ITEM_LEN,
    MAX_TAGS_COUNT,
    SINCE_HOURS_MAX,
    SINCE_HOURS_MIN,
    TOP_N_MAX,
    TOP_N_MIN,
    InvalidToolArg,
    assay_args_error,
    baseline_args_error,
    bvt_args_error,
    check_bool,
    check_build_type,
    check_finite,
    check_int,
    check_optional_finite,
    check_optional_str,
    check_relative_dir,
    check_str,
    check_str_list,
    extract_assets_args_error,
    pour_args_error,
    score_args_error,
    usage_stats_args_error,
)
from linc_crucible.deploy_tools import crucible_pour
from linc_crucible.stats_tools import crucible_usage_stats


def _ascii_id(val: object) -> str:
    """Return a mutmut-safe ASCII pytest node ID."""
    raw = repr(val)
    if len(raw) > 32:
        raw = f"{raw[:24]}..len{len(raw)}"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw)
    return safe.strip("_") or "case"


def _assert_validation_dict(res: dict[str, Any]) -> None:
    assert isinstance(res, dict), f"Expected dict, got {type(res).__name__}: {res!r}"
    assert res.get("validation_error") is True, f"Expected validation_error=True, got {res!r}"
    assert res.get("error_category") == "validation", (
        f"Expected category='validation', got {res!r}"
    )
    assert isinstance(res.get("error"), str) and res["error"], f"Missing error text in {res!r}"


def _reject_json_constant(const: str) -> Any:
    raise ValueError(f"Non-standard JSON constant: {const}")


def _assert_validation_json(raw: str) -> None:
    assert isinstance(raw, str), f"Expected JSON string, got {type(raw).__name__}: {raw!r}"
    parsed = json.loads(raw, parse_constant=_reject_json_constant)
    _assert_validation_dict(parsed)


def _make_reader_for_yaml(tmp_path: Path, yaml_content: str) -> ForgeProjectReader:
    proj_dir = tmp_path / "demo"
    proj_dir.mkdir(exist_ok=True)
    (proj_dir / ".forge.yaml").write_text(yaml_content, encoding="utf-8")
    reg_path = tmp_path / "projects.yaml"
    reg_path.write_text(
        f"projects:\n  - name: demo\n    path: {proj_dir}\n",
        encoding="utf-8",
    )
    return ForgeProjectReader(registry_path=reg_path)


# ---------------------------------------------------------------------------
# 1. [contract] build_type allowlist
# ---------------------------------------------------------------------------

HOSTILE_BUILD_TYPES: list[Any] = [
    "debug;id",
    "debug x",
    "debug\n",
    "--offline",
    "debug --init-script x",
    "a" * 10000,
    "",
    "1debug",
    "debug-variant",
    "debug_variant",
    "debug\x00",
    123,
    None,
    ["debug"],
]

VALID_BUILD_TYPES: list[tuple[str, str]] = [
    ("release", "./gradlew :app:assembleRelease"),
    ("debug", "./gradlew :app:assembleDebug"),
    ("benchmark", "./gradlew :app:assembleBenchmark"),
    ("staging2", "./gradlew :app:assembleStaging2"),
]


class TestBuildTypeAllowlist:
    """R4 Medium 5: build_type must match \\A[A-Za-z][A-Za-z0-9]{0,31}\\Z."""

    @pytest.mark.parametrize("bad_build_type", HOSTILE_BUILD_TYPES, ids=_ascii_id)
    def test_crucible_pour_rejects_hostile_build_type_with_zero_subprocess_calls(
        self, bad_build_type: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader = _make_reader_for_yaml(
            tmp_path,
            "modules:\n  - name: app\n    platform: android\n    device_type: phone\n",
        )
        deployer = Deployer(forge_reader=reader, nerve_client=MagicMock())
        monkeypatch.setattr(ctx, "_ensure_initialized", lambda: None)
        monkeypatch.setattr(ctx, "_deployer", deployer)

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("subprocess.run") as mock_run,
        ):
            mock_popen.return_value.__enter__.return_value = mock_popen.return_value
            mock_popen.return_value.communicate.return_value = ("", "")
            mock_popen.return_value.returncode = 0
            res = asyncio.run(
                crucible_pour(
                    project="demo",
                    module="app",
                    platform="android",
                    build_only=True,
                    build_type=bad_build_type,
                )
            )
            mock_popen.assert_not_called()
            mock_run.assert_not_called()
        _assert_validation_dict(res)

    @pytest.mark.parametrize("bad_build_type", HOSTILE_BUILD_TYPES, ids=_ascii_id)
    def test_deployer_module_scoped_build_command_rejects_hostile_build_type(
        self, bad_build_type: Any
    ) -> None:
        mod = ModuleConfig(name="app")
        with pytest.raises(ValueError, match="build_type"):
            Deployer._module_scoped_android_build_command(mod, "android", bad_build_type)

    @pytest.mark.parametrize("bad_build_type", HOSTILE_BUILD_TYPES, ids=_ascii_id)
    def test_deployer_module_scoped_build_command_rejects_even_on_non_android(
        self, bad_build_type: Any
    ) -> None:
        mod = ModuleConfig(name="app")
        with pytest.raises(ValueError, match="build_type"):
            Deployer._module_scoped_android_build_command(mod, "ios", bad_build_type)

    @pytest.mark.parametrize("build_type,expected_cmd", VALID_BUILD_TYPES, ids=_ascii_id)
    def test_valid_build_types_accepted(self, build_type: str, expected_cmd: str) -> None:
        mod = ModuleConfig(name="app")
        assert BUILD_TYPE_RE.match(build_type) is not None
        assert check_build_type(build_type) == build_type
        assert (
            Deployer._module_scoped_android_build_command(mod, "android", build_type)
            == expected_cmd
        )
        assert Deployer._module_scoped_android_build_command(mod, "ios", build_type) == ""


# ---------------------------------------------------------------------------
# 2. [contract] crash.py hostile input matrix across all 8 tool functions
# ---------------------------------------------------------------------------

CRASH_POUR_CASES: list[dict[str, Any]] = [
    {"project": "a" * 10000},
    {"project": "a\x00b"},
    {"project": ["x"]},
    {"project": 123},
    {"project": None},
    {"project": "demo", "device": "a" * 10000},
    {"project": "demo", "device": "a\x00b"},
    {"project": "demo", "device": ["emulator-5554"]},
    {"project": "demo", "device": 123},
    {"project": "demo", "platform": "a" * 10000},
    {"project": "demo", "platform": "a\x00b"},
    {"project": "demo", "platform": ["android"]},
    {"project": "demo", "platform": 123},
    {"project": "demo", "module": "a" * 10000},
    {"project": "demo", "module": "a\x00b"},
    {"project": "demo", "module": ["app"]},
    {"project": "demo", "module": 123},
    {"project": "demo", "device_type": "a" * 10000},
    {"project": "demo", "device_type": "a\x00b"},
    {"project": "demo", "device_type": ["phone"]},
    {"project": "demo", "device_type": 123},
    {"project": "demo", "launch_intent": "a" * 10000},
    {"project": "demo", "launch_intent": "a\x00b"},
    {"project": "demo", "launch_intent": ["x"]},
    {"project": "demo", "launch_intent": 123},
    {"project": "demo", "build_only": "yes"},
    {"project": "demo", "skip_build": "no"},
    {"project": "demo", "launch": 1},
    {"project": "demo", "allow_personal": "true"},
]

CRASH_ASSAY_CASES: list[dict[str, Any]] = [
    {"project": "a" * 10000},
    {"project": "a\x00b"},
    {"project": ["x"]},
    {"project": 123},
    {"project": None},
    {"project": "demo", "mode": "a" * 10000},
    {"project": "demo", "mode": "a\x00b"},
    {"project": "demo", "mode": 123},
    {"project": "demo", "device": "a" * 10000},
    {"project": "demo", "device": "a\x00b"},
    {"project": "demo", "device": 123},
    {"project": "demo", "perf": "yes"},
    {"project": "demo", "a11y": 1},
    {"project": "demo", "a11y_tree": "true"},
    {"project": "demo", "allow_personal": "false"},
]

CRASH_SCORE_CASES: list[dict[str, Any]] = [
    {"project": "a" * 10000},
    {"project": "a\x00b"},
    {"project": ["x"]},
    {"project": 123},
    {"project": None},
    {"project": "demo", "limit": -5},
    {"project": "demo", "limit": 0},
    {"project": "demo", "limit": 1001},
    {"project": "demo", "limit": 10**12},
    {"project": "demo", "limit": 1e308},
    {"project": "demo", "limit": float("nan")},
    {"project": "demo", "limit": float("inf")},
    {"project": "demo", "limit": float("-inf")},
    {"project": "demo", "limit": "5"},
    {"project": "demo", "limit": True},
    {"project": "demo", "history": "yes"},
    {"project": "demo", "history": 1},
]

CRASH_PREFLIGHT_CASES: list[dict[str, Any]] = [
    {"project": "a" * 10000},
    {"project": "a\x00b"},
    {"project": ["x"]},
    {"project": 123},
    {"project": None},
    {"project": "../escape"},
]

CRASH_BVT_CASES: list[dict[str, Any]] = [
    {"project": "a" * 10000, "device": "emulator-5554"},
    {"project": "a\x00b", "device": "emulator-5554"},
    {"project": ["x"], "device": "emulator-5554"},
    {"project": 123, "device": "emulator-5554"},
    {"project": None, "device": "emulator-5554"},
    {"project": "", "device": "emulator-5554"},
    {"project": "demo", "device": ""},
    {"project": "demo", "device": "a" * 10000},
    {"project": "demo", "device": "a\x00b"},
    {"project": "demo", "device": ["emulator-5554"]},
    {"project": "demo", "device": 123},
    {"project": "demo", "device": "emulator-5554", "suite_dir": "a" * 10000},
    {"project": "demo", "device": "emulator-5554", "suite_dir": "a\x00b"},
    {"project": "demo", "device": "emulator-5554", "suite_dir": ["bvt"]},
    {"project": "demo", "device": "emulator-5554", "suite_dir": 123},
    {"project": "demo", "device": "emulator-5554", "suite_dir": ""},
    {"project": "demo", "device": "emulator-5554", "suite_dir": "/abs/suite"},
    {"project": "demo", "device": "emulator-5554", "suite_dir": "../escape"},
    {"project": "demo", "device": "emulator-5554", "suite_dir": "sub/../../escape"},
    {"project": "demo", "device": "emulator-5554", "tags": "smoke"},
    {"project": "demo", "device": "emulator-5554", "tags": 123},
    {"project": "demo", "device": "emulator-5554", "tags": ["a" * 10000]},
    {"project": "demo", "device": "emulator-5554", "tags": ["a\x00b"]},
    {"project": "demo", "device": "emulator-5554", "tags": [123]},
    {"project": "demo", "device": "emulator-5554", "tags": [""]},
    {"project": "demo", "device": "emulator-5554", "tags": ["t"] * 65},
    {"project": "demo", "device": "emulator-5554", "exclude_tags": "ios"},
    {"project": "demo", "device": "emulator-5554", "exclude_tags": [123]},
    {"project": "demo", "device": "emulator-5554", "output_dir": "a" * 10000},
    {"project": "demo", "device": "emulator-5554", "output_dir": "a\x00b"},
    {"project": "demo", "device": "emulator-5554", "output_dir": ["out"]},
    {"project": "demo", "device": "emulator-5554", "output_dir": 123},
    {"project": "demo", "device": "emulator-5554", "output_dir": ""},
    {"project": "demo", "device": "emulator-5554", "app_path": "a\x00b"},
    {"project": "demo", "device": "emulator-5554", "app_path": 123},
    {"project": "demo", "device": "emulator-5554", "budget_s": float("nan")},
    {"project": "demo", "device": "emulator-5554", "budget_s": float("inf")},
    {"project": "demo", "device": "emulator-5554", "budget_s": -5},
    {"project": "demo", "device": "emulator-5554", "budget_s": 1e308},
    {"project": "demo", "device": "emulator-5554", "compare_against_baseline": "yes"},
    {"project": "demo", "device": "emulator-5554", "allow_personal": "yes"},
]

CRASH_EXTRACT_ASSETS_CASES: list[dict[str, Any]] = [
    {"video_path": ""},
    {"video_path": "a" * 10000},
    {"video_path": "a\x00b"},
    {"video_path": ["video.mp4"]},
    {"video_path": 123},
    {"video_path": None},
    {"video_path": "video.mp4", "max_assets": -5},
    {"video_path": "video.mp4", "max_assets": 0},
    {"video_path": "video.mp4", "max_assets": 51},
    {"video_path": "video.mp4", "max_assets": 10**12},
    {"video_path": "video.mp4", "max_assets": 1e308},
    {"video_path": "video.mp4", "max_assets": float("nan")},
    {"video_path": "video.mp4", "max_assets": float("inf")},
    {"video_path": "video.mp4", "max_assets": float("-inf")},
    {"video_path": "video.mp4", "max_assets": 1.5},
    {"video_path": "video.mp4", "max_assets": "5"},
    {"video_path": "video.mp4", "max_assets": True},
]

CRASH_USAGE_STATS_CASES: list[dict[str, Any]] = [
    {"server_filter": "a" * 10000},
    {"server_filter": "a\x00b"},
    {"server_filter": 123},
    {"server_filter": ["linc-crucible"]},
    {"include_rotated": "yes"},
    {"include_rotated": 1},
    {"since_hours": float("nan")},
    {"since_hours": float("inf")},
    {"since_hours": float("-inf")},
    {"since_hours": 1e308},
    {"since_hours": -5},
    {"since_hours": -0.001},
    {"since_hours": 87600.1},
    {"since_hours": 10**12},
    {"since_hours": "24"},
    {"since_hours": True},
    {"since_hours": None},
    {"since_hours": [24]},
    {"top_n": -5},
    {"top_n": -1},
    {"top_n": 0},
    {"top_n": 1001},
    {"top_n": 10**12},
    {"top_n": 1e308},
    {"top_n": float("nan")},
    {"top_n": float("inf")},
    {"top_n": float("-inf")},
    {"top_n": 1.5},
    {"top_n": "10"},
    {"top_n": True},
]

CRASH_BASELINE_CASES: list[dict[str, Any]] = [
    {"action": "", "project": "demo"},
    {"action": 123, "project": "demo"},
    {"action": ["list"], "project": "demo"},
    {"action": "a" * 10000, "project": "demo"},
    {"action": "a\x00b", "project": "demo"},
    {"action": "list", "project": "a" * 10000},
    {"action": "list", "project": "a\x00b"},
    {"action": "list", "project": ["demo"]},
    {"action": "list", "project": 123},
    {"action": "list", "project": None},
    {"action": "list", "project": "demo", "screen_name": "a" * 10000},
    {"action": "list", "project": "demo", "screen_name": "a\x00b"},
    {"action": "list", "project": "demo", "screen_name": ["home"]},
    {"action": "list", "project": "demo", "screen_name": 123},
    {"action": "capture", "project": "demo", "screen_name": "home", "image_path": "a" * 10000},
    {"action": "capture", "project": "demo", "screen_name": "home", "image_path": "a\x00b"},
    {"action": "capture", "project": "demo", "screen_name": "home", "image_path": ["img.png"]},
    {"action": "capture", "project": "demo", "screen_name": "home", "image_path": 123},
    {"action": "capture", "project": "demo", "screen_name": "home", "device_info": "a" * 10000},
    {"action": "capture", "project": "demo", "screen_name": "home", "device_info": "a\x00b"},
    {"action": "capture", "project": "demo", "screen_name": "home", "device_info": 123},
    {"action": "capture", "project": "demo", "screen_name": "home", "app_version": "a" * 10000},
    {"action": "capture", "project": "demo", "screen_name": "home", "app_version": "a\x00b"},
    {"action": "capture", "project": "demo", "screen_name": "home", "app_version": ["1.0"]},
    {"action": "list", "project": "demo", "mode": ""},
    {"action": "list", "project": "demo", "mode": "a" * 10000},
    {"action": "list", "project": "demo", "mode": "a\x00b"},
    {"action": "list", "project": "demo", "mode": 123},
    {"action": "list", "project": "demo", "device": "a" * 10000},
    {"action": "list", "project": "demo", "device": "a\x00b"},
    {"action": "list", "project": "demo", "device": 123},
    {"action": "promote", "project": "demo", "screen_name": "home", "source_score": float("nan")},
    {"action": "promote", "project": "demo", "screen_name": "home", "source_score": 1.5},
    {"action": "promote", "project": "demo", "screen_name": "home", "source_score": -0.1},
    {"action": "list", "project": "demo", "allow_empty": "yes"},
    {"action": "list", "project": "demo", "allow_personal": 1},
    {"action": "prune", "project": "demo", "keep": -5},
    {"action": "prune", "project": "demo", "keep": -1},
    {"action": "prune", "project": "demo", "keep": 1001},
    {"action": "prune", "project": "demo", "keep": 10**12},
    {"action": "prune", "project": "demo", "keep": 1e308},
    {"action": "prune", "project": "demo", "keep": float("nan")},
    {"action": "prune", "project": "demo", "keep": float("inf")},
    {"action": "prune", "project": "demo", "keep": float("-inf")},
    {"action": "prune", "project": "demo", "keep": 1.7},
    {"action": "prune", "project": "demo", "keep": "10"},
    {"action": "prune", "project": "demo", "keep": True},
]


class TestCrashHostileInputs:
    """Every crash.py hostile input returns a validation error instead of raising."""

    @pytest.mark.parametrize("kwargs", CRASH_POUR_CASES, ids=_ascii_id)
    def test_crucible_pour_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        res = asyncio.run(crucible_pour(**kwargs))
        _assert_validation_dict(res)

    @pytest.mark.parametrize("kwargs", CRASH_ASSAY_CASES, ids=_ascii_id)
    def test_crucible_assay_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        res = asyncio.run(crucible_assay(**kwargs))
        _assert_validation_dict(res)

    @pytest.mark.parametrize("kwargs", CRASH_SCORE_CASES, ids=_ascii_id)
    def test_crucible_score_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        res = crucible_score(**kwargs)
        _assert_validation_dict(res)

    @pytest.mark.parametrize("kwargs", CRASH_PREFLIGHT_CASES, ids=_ascii_id)
    def test_crucible_preflight_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        res = crucible_preflight(**kwargs)
        _assert_validation_dict(res)

    @pytest.mark.parametrize("kwargs", CRASH_BVT_CASES, ids=_ascii_id)
    def test_crucible_bvt_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        raw = asyncio.run(crucible_bvt(**kwargs))
        _assert_validation_json(raw)

    @pytest.mark.parametrize("kwargs", CRASH_EXTRACT_ASSETS_CASES, ids=_ascii_id)
    def test_crucible_extract_assets_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        raw = asyncio.run(crucible_extract_assets(**kwargs))
        _assert_validation_json(raw)

    @pytest.mark.parametrize("kwargs", CRASH_USAGE_STATS_CASES, ids=_ascii_id)
    def test_crucible_usage_stats_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        res = crucible_usage_stats(**kwargs)
        _assert_validation_dict(res)

    @pytest.mark.parametrize("kwargs", CRASH_BASELINE_CASES, ids=_ascii_id)
    def test_crucible_baseline_hostile_inputs(self, kwargs: dict[str, Any]) -> None:
        res = asyncio.run(crucible_baseline(**kwargs))
        _assert_validation_dict(res)


# ---------------------------------------------------------------------------
# 3. [contract] since_hours NaN/±Inf/1e308 rejected and strict JSON output
# ---------------------------------------------------------------------------


class TestSinceHoursFiniteAndStrictJson:
    """R4 Info: since_hours NaN/±Inf/1e308 rejected; JSON parses without bare NaN."""

    @pytest.mark.parametrize(
        "bad_since",
        [float("nan"), float("inf"), float("-inf"), 1e308, -1.0, 87600.5],
        ids=_ascii_id,
    )
    def test_since_hours_non_finite_or_out_of_range_rejected_and_strict_json(
        self, bad_since: float
    ) -> None:
        res = crucible_usage_stats(since_hours=bad_since)
        _assert_validation_dict(res)
        encoded = json.dumps(res, allow_nan=False)
        parsed = json.loads(
            encoded,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Bare constant: {c}")),
        )
        assert parsed["validation_error"] is True

    @pytest.mark.parametrize("valid_since", [0, 0.0, 24, 24.5, 87600.0], ids=_ascii_id)
    def test_since_hours_valid_values_produce_strict_json(
        self, valid_since: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LINC_MCP_CALL_LOG", str(tmp_path / "calls.jsonl"))
        res = crucible_usage_stats(since_hours=valid_since, top_n=10)
        assert "error" not in res
        encoded = json.dumps(res, allow_nan=False)
        parsed = json.loads(
            encoded,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Bare constant: {c}")),
        )
        expected_win = None if float(valid_since) == 0.0 else float(valid_since)
        assert parsed["window_hours"] == expected_win


# ---------------------------------------------------------------------------
# 4. [test] .forge.yaml int/list/unbalanced-quote commands (R4 Low 11)
# ---------------------------------------------------------------------------


class TestForgeYamlCommandGuards:
    """R4 Low 11: non-string or unbalanced-quote .forge.yaml commands never raise."""

    @pytest.mark.parametrize(
        "field_name,yaml_val,expected_type",
        [
            ("test_command", "123", "int"),
            ("test_command", "[pytest, -q]", "list"),
            ("lint_command", "456", "int"),
            ("lint_command", "[ruff, check]", "list"),
            ("build_command", "789", "int"),
            ("build_command", "[./gradlew, assembleDebug]", "list"),
            ("ios_build_command", "999", "int"),
            ("ios_build_command", "[xcodebuild]", "list"),
            ("ios_watch_scheme", "111", "int"),
        ],
        ids=_ascii_id,
    )
    def test_merge_forge_yaml_drops_non_string_and_logs_once(
        self, field_name: str, yaml_val: str, expected_type: str, tmp_path: Path
    ) -> None:
        with structlog.testing.capture_logs() as cap_logs:
            reader = _make_reader_for_yaml(tmp_path, f"{field_name}: {yaml_val}\n")

        project = reader.get("demo")
        assert project is not None
        assert getattr(project, field_name) == ""
        matching_logs = [
            entry
            for entry in cap_logs
            if entry.get("event") == "forge_yaml_command_not_string"
            and entry.get("field") == field_name
        ]
        assert len(matching_logs) == 1
        assert matching_logs[0].get("type") == expected_type
        assert any(field_name in err and expected_type in err for err in project.config_errors)

    def test_merge_forge_yaml_null_command_is_clean_empty_string(self, tmp_path: Path) -> None:
        with structlog.testing.capture_logs() as cap_logs:
            reader = _make_reader_for_yaml(tmp_path, "test_command: null\n")
        project = reader.get("demo")
        assert project is not None
        assert project.test_command == ""
        assert project.config_errors == []
        assert not [e for e in cap_logs if e.get("event") == "forge_yaml_command_not_string"]

    def test_safe_subprocess_run_unbalanced_quote_raises_command_parse_error(self) -> None:
        with pytest.raises(CommandParseError, match="unbalanced quote") as exc_info:
            safe_subprocess_run('echo "oops', cwd=".", timeout=5.0)
        assert isinstance(exc_info.value, ValueError)

    @pytest.mark.parametrize(
        "yaml_content,expected_dim",
        [
            ("test_command: 123\n", "tests"),
            ("test_command:\n  - pytest\n  - -q\n", "tests"),
            ("test_command: 'echo \"oops'\n", "tests"),
            ("lint_command: 456\n", "lint"),
            ("lint_command:\n  - ruff\n", "lint"),
            ("lint_command: 'ruff \"oops'\n", "lint"),
        ],
        ids=_ascii_id,
    )
    def test_assayer_handles_malformed_forge_yaml_commands_as_unmeasured(
        self, yaml_content: str, expected_dim: str, tmp_path: Path
    ) -> None:
        reader = _make_reader_for_yaml(tmp_path, yaml_content)
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        assayer = Assayer(forge_reader=reader, scorer=scorer)
        report = assayer.assay("demo", mode="quick")

        assert report.not_found is False
        assert report.quality_score is not None
        assert expected_dim in report.quality_score.attempted_unmeasured
        assert report.quality_score.ship_gate is False

        if expected_dim == "tests":
            assert report.tests.attempted is True
            assert report.tests.attempted_unmeasured is True
            assert report.tests.score_input is None
            assert report.tests.output
            assert report.tests.output != "No test command configured"
        else:
            assert report.lint.attempted is True
            assert report.lint.attempted_unmeasured is True
            assert report.lint.score_input is None
            assert report.lint.output
            assert report.lint.output != "No lint command configured"

    @pytest.mark.parametrize(
        "yaml_content",
        [
            "build_command: 123\n",
            "build_command:\n  - ./gradlew\n  - assembleDebug\n",
            "build_command: 'echo \"oops'\n",
        ],
        ids=_ascii_id,
    )
    def test_deployer_handles_malformed_forge_yaml_build_command_cleanly(
        self, yaml_content: str, tmp_path: Path
    ) -> None:
        reader = _make_reader_for_yaml(tmp_path, yaml_content)
        deployer = Deployer(forge_reader=reader, nerve_client=MagicMock())
        result = deployer.pour("demo", build_only=True)
        payload = result.to_dict()

        assert result.success is False
        assert payload.get("build_error") is True
        assert result.error
        assert "No build command configured" not in result.error


# ---------------------------------------------------------------------------
# 5. [test] a6.py lax-coercion cases pinned
# ---------------------------------------------------------------------------


class TestLaxCoercionPinned:
    """a6.py lax-coercion cases: each returns a validation error, never a traceback."""

    def test_score_limit_string_rejected(self) -> None:
        _assert_validation_dict(crucible_score(project="demo", limit="5"))  # type: ignore[arg-type]

    def test_score_history_string_rejected(self) -> None:
        _assert_validation_dict(crucible_score(project="demo", history="yes"))  # type: ignore[arg-type]

    def test_baseline_keep_float_rejected(self) -> None:
        _assert_validation_dict(
            asyncio.run(crucible_baseline(action="prune", project="demo", keep=1.7))  # type: ignore[arg-type]
        )

    def test_usage_stats_top_n_negative_rejected(self) -> None:
        _assert_validation_dict(crucible_usage_stats(top_n=-1))


# ---------------------------------------------------------------------------
# 6. [test] CliRunner hostile build_type → exit 2, validation envelope on stderr
# ---------------------------------------------------------------------------


class TestCliHostileBuildType:
    """CLI hostile build_type exits 2 with a validation envelope on stderr."""

    def test_cli_pour_hostile_build_type_exits_2(self) -> None:
        runner = CliRunner()
        with (
            patch("subprocess.Popen") as mock_popen,
            patch("subprocess.run") as mock_run,
        ):
            result = runner.invoke(
                cli,
                [
                    "pour",
                    "--project",
                    "x",
                    "--build-only",
                    "--build-type",
                    "debug --init-script x",
                ],
            )
            mock_popen.assert_not_called()
            mock_run.assert_not_called()

        assert result.exit_code == 2, (
            f"Expected exit_code=2, got {result.exit_code}: {result.output}"
        )
        stderr_payload = json.loads(result.stderr)
        assert stderr_payload["error"]["category"] == "validation"
        assert stderr_payload["error"]["code"] == "VALIDATION_ERROR"
        assert "build_type" in stderr_payload["error"]["message"]


# ---------------------------------------------------------------------------
# 7. Unit tests for tool_args helpers
# ---------------------------------------------------------------------------


class TestToolArgsHelpers:
    """Direct tests for core.tool_args validators."""

    def test_check_str_accepts_valid_and_rejects_bad(self) -> None:
        assert MAX_SHORT_STR_LEN == 256
        assert check_str("", "f") == ""
        assert check_str("a" * 256, "f") == "a" * 256
        with pytest.raises(InvalidToolArg) as exc_info:
            check_str("a" * 257, "f")
        assert exc_info.value.field == "f"
        assert exc_info.value.reason == "must be at most 256 characters"
        assert str(exc_info.value).startswith("invalid f: must be at most 256 characters (got ")

        assert check_str("hello", "f", max_len=10) == "hello"
        assert check_str("", "f", max_len=10, allow_empty=True) == ""
        with pytest.raises(InvalidToolArg) as exc_empty:
            check_str("", "f", max_len=10, allow_empty=False)
        assert exc_empty.value.field == "f"
        assert exc_empty.value.reason == "must not be empty"
        assert str(exc_empty.value) == "invalid f: must not be empty (got '')"

        with pytest.raises(InvalidToolArg) as exc_type:
            check_str(123, "f")
        assert exc_type.value.field == "f"
        assert exc_type.value.reason == "must be a string"
        assert str(exc_type.value) == "invalid f: must be a string (got int)"

        with pytest.raises(InvalidToolArg) as exc_nul:
            check_str("a\x00b", "f")
        assert exc_nul.value.field == "f"
        assert exc_nul.value.reason == "must not contain NUL bytes"
        assert str(exc_nul.value) == "invalid f: must not contain NUL bytes (got 'a\\x00b')"

        with pytest.raises(InvalidToolArg) as exc_len:
            check_str("123456", "f", max_len=5)
        assert exc_len.value.field == "f"
        assert exc_len.value.reason == "must be at most 5 characters"
        assert str(exc_len.value) == "invalid f: must be at most 5 characters (got '123456')"

    def test_check_optional_str_accepts_none_and_str(self) -> None:
        assert MAX_PATH_LEN == 1024
        assert check_optional_str(None, "f") is None
        assert check_optional_str("a" * 1024, "f") == "a" * 1024
        with pytest.raises(InvalidToolArg) as exc_len:
            check_optional_str("a" * 1025, "f")
        assert exc_len.value.field == "f"
        assert exc_len.value.reason == "must be at most 1024 characters"
        with pytest.raises(InvalidToolArg) as exc_empty:
            check_optional_str("", "f")
        assert exc_empty.value.field == "f"
        assert exc_empty.value.reason == "must not be empty"
        assert check_optional_str("", "f", allow_empty=True) == ""
        assert check_optional_str("abc", "f", max_len=3) == "abc"
        with pytest.raises(InvalidToolArg) as exc_short:
            check_optional_str("abcd", "f", max_len=3)
        assert exc_short.value.reason == "must be at most 3 characters"

    def test_check_int_accepts_valid_and_rejects_bad(self) -> None:
        assert check_int(1, "n", 1, 10) == 1
        assert check_int(10, "n", 1, 10) == 10
        with pytest.raises(InvalidToolArg) as exc_bool:
            check_int(True, "n", 1, 10)
        assert exc_bool.value.field == "n"
        assert exc_bool.value.reason == "must be an integer"
        assert str(exc_bool.value) == "invalid n: must be an integer (got bool)"
        with pytest.raises(InvalidToolArg) as exc_float:
            check_int(1.5, "n", 1, 10)
        assert str(exc_float.value) == "invalid n: must be an integer (got float)"
        with pytest.raises(InvalidToolArg) as exc_lo:
            check_int(0, "n", 1, 10)
        assert exc_lo.value.field == "n"
        assert exc_lo.value.reason == "must be between 1 and 10"
        assert str(exc_lo.value) == "invalid n: must be between 1 and 10 (got int)"
        with pytest.raises(InvalidToolArg) as exc_hi:
            check_int(11, "n", 1, 10)
        assert str(exc_hi.value) == "invalid n: must be between 1 and 10 (got int)"

    def test_check_finite_accepts_valid_and_rejects_bad(self) -> None:
        assert check_finite(0, "h", 0.0, 100.0) == 0.0
        assert check_finite(100.0, "h", 0.0, 100.0) == 100.0
        assert check_finite(24.5, "h", 0.0, 100.0) == 24.5
        with pytest.raises(InvalidToolArg) as exc_bool:
            check_finite(True, "h", 0.0, 100.0)
        assert exc_bool.value.field == "h"
        assert exc_bool.value.reason == "must be a number"
        assert str(exc_bool.value) == "invalid h: must be a number (got bool)"
        with pytest.raises(InvalidToolArg) as exc_str:
            check_finite("24", "h", 0.0, 100.0)
        assert str(exc_str.value) == "invalid h: must be a number (got '24')"
        with pytest.raises(InvalidToolArg) as exc_nan:
            check_finite(float("nan"), "h", 0.0, 100.0)
        assert exc_nan.value.field == "h"
        assert exc_nan.value.reason == "must be finite"
        assert str(exc_nan.value) == "invalid h: must be finite (got float)"
        with pytest.raises(InvalidToolArg) as exc_inf:
            check_finite(float("inf"), "h", 0.0, 100.0)
        assert str(exc_inf.value) == "invalid h: must be finite (got float)"
        with pytest.raises(InvalidToolArg) as exc_lo:
            check_finite(-0.1, "h", 0.0, 100.0)
        assert exc_lo.value.field == "h"
        assert exc_lo.value.reason == "must be between 0.0 and 100.0"
        assert str(exc_lo.value) == "invalid h: must be between 0.0 and 100.0 (got float)"
        with pytest.raises(InvalidToolArg) as exc_hi:
            check_finite(100.1, "h", 0.0, 100.0)
        assert str(exc_hi.value) == "invalid h: must be between 0.0 and 100.0 (got float)"

    def test_check_optional_finite_accepts_none_and_finite(self) -> None:
        assert check_optional_finite(None, "s", 0.0, 1.0) is None
        assert check_optional_finite(0.0, "s", 0.0, 1.0) == 0.0
        assert check_optional_finite(1.0, "s", 0.0, 1.0) == 1.0
        with pytest.raises(InvalidToolArg) as exc_nan:
            check_optional_finite(float("nan"), "s", 0.0, 1.0)
        assert exc_nan.value.field == "s"
        assert exc_nan.value.reason == "must be finite"

    def test_check_bool_accepts_only_bool(self) -> None:
        assert check_bool(True, "b") is True
        assert check_bool(False, "b") is False
        with pytest.raises(InvalidToolArg) as exc_int:
            check_bool(1, "b")
        assert exc_int.value.field == "b"
        assert exc_int.value.reason == "must be a boolean"
        assert str(exc_int.value) == "invalid b: must be a boolean (got int)"
        with pytest.raises(InvalidToolArg) as exc_str:
            check_bool("true", "b")
        assert str(exc_str.value) == "invalid b: must be a boolean (got 'true')"

    def test_check_build_type_exact_boundaries(self) -> None:
        assert check_build_type("a" * 32) == "a" * 32
        with pytest.raises(InvalidToolArg) as exc_len:
            check_build_type("a" * 33)
        assert exc_len.value.field == "build_type"
        assert exc_len.value.reason == "must be at most 32 characters"
        with pytest.raises(InvalidToolArg) as exc_pat:
            check_build_type("9release")
        assert exc_pat.value.field == "build_type"
        assert exc_pat.value.reason == "must match [A-Za-z][A-Za-z0-9]{0,31}"
        assert (
            str(exc_pat.value)
            == "invalid build_type: must match [A-Za-z][A-Za-z0-9]{0,31} (got '9release')"
        )
        with pytest.raises(InvalidToolArg) as exc_empty:
            check_build_type("", "custom_bt")
        assert exc_empty.value.field == "custom_bt"
        assert exc_empty.value.reason == "must not be empty"

    def test_check_relative_dir_accepts_relative_and_rejects_traversal(self) -> None:
        assert check_relative_dir(".crucible/bvt", "suite_dir") == ".crucible/bvt"
        assert check_relative_dir("tests/bvt", "suite_dir") == "tests/bvt"
        assert check_relative_dir("a" * 256, "suite_dir") == "a" * 256
        with pytest.raises(InvalidToolArg) as exc_len:
            check_relative_dir("a" * 257, "suite_dir")
        assert exc_len.value.field == "suite_dir"
        assert exc_len.value.reason == "must be at most 256 characters"
        with pytest.raises(InvalidToolArg) as exc_short:
            check_relative_dir("abcd", "suite_dir", max_len=3)
        assert exc_short.value.reason == "must be at most 3 characters"
        with pytest.raises(InvalidToolArg) as exc_abs:
            check_relative_dir("/etc/passwd", "suite_dir")
        assert exc_abs.value.field == "suite_dir"
        assert exc_abs.value.reason == "must be a relative path"
        assert (
            str(exc_abs.value) == "invalid suite_dir: must be a relative path (got '/etc/passwd')"
        )
        with pytest.raises(InvalidToolArg) as exc_trav:
            check_relative_dir("../secret", "suite_dir")
        assert exc_trav.value.field == "suite_dir"
        assert exc_trav.value.reason == "must not contain '..' segments"
        assert (
            str(exc_trav.value)
            == "invalid suite_dir: must not contain '..' segments (got '../secret')"
        )

    def test_check_str_list_accepts_valid_and_rejects_bad(self) -> None:
        assert MAX_TAGS_COUNT == 64
        assert MAX_TAG_ITEM_LEN == 64
        assert check_str_list(None, "tags") is None
        assert check_str_list(["a" * 64] * 64, "tags") == ["a" * 64] * 64
        with pytest.raises(InvalidToolArg) as exc_cnt:
            check_str_list(["t"] * 65, "tags")
        assert exc_cnt.value.field == "tags"
        assert exc_cnt.value.reason == "must contain at most 64 items"
        assert str(exc_cnt.value).startswith("invalid tags: must contain at most 64 items")
        with pytest.raises(InvalidToolArg) as exc_item_len:
            check_str_list(["a" * 65], "tags")
        assert exc_item_len.value.field == "tags[0]"
        assert exc_item_len.value.reason == "must be at most 64 characters"
        with pytest.raises(InvalidToolArg) as exc_type:
            check_str_list("smoke", "tags")
        assert exc_type.value.field == "tags"
        assert exc_type.value.reason == "must be a list of strings"
        assert str(exc_type.value) == "invalid tags: must be a list of strings (got 'smoke')"
        with pytest.raises(InvalidToolArg) as exc_max2:
            check_str_list(["a", "b", "c"], "tags", max_items=2)
        assert exc_max2.value.reason == "must contain at most 2 items"
        with pytest.raises(InvalidToolArg) as exc_item_max3:
            check_str_list(["abcd"], "tags", max_len=3)
        assert exc_item_max3.value.field == "tags[0]"
        assert exc_item_max3.value.reason == "must be at most 3 characters"
        with pytest.raises(InvalidToolArg) as exc_empty_item:
            check_str_list(["ok", ""], "tags")
        assert exc_empty_item.value.field == "tags[1]"
        assert exc_empty_item.value.reason == "must not be empty"

    def test_missing_lint_and_test_result_helpers(self) -> None:
        clean_proj = ForgeProject(name="demo")
        lint_unconf = _missing_lint_result(clean_proj)
        assert lint_unconf.output == "No lint command configured"
        assert lint_unconf.success is False
        assert lint_unconf.configured is False
        assert lint_unconf.attempted is False

        test_unconf = _missing_test_result(clean_proj)
        assert test_unconf.output == "No test command configured"
        assert test_unconf.success is False
        assert test_unconf.configured is False
        assert test_unconf.attempted is False

        bad_proj = ForgeProject(
            name="demo",
            config_errors=[
                "lint_command: .forge.yaml value must be a string (got int)",
                "test_command: .forge.yaml value must be a string (got list)",
            ],
        )
        lint_bad = _missing_lint_result(bad_proj)
        assert lint_bad.output == "lint_command: .forge.yaml value must be a string (got int)"
        assert lint_bad.success is False
        assert lint_bad.configured is False
        assert lint_bad.attempted is True

        test_bad = _missing_test_result(bad_proj)
        assert test_bad.output == "test_command: .forge.yaml value must be a string (got list)"
        assert test_bad.success is False
        assert test_bad.configured is False
        assert test_bad.attempted is True

    def test_attach_visual_compare_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t1 = MagicMock(screenshots=["s1.png", "s2.png"])
        t2 = MagicMock(screenshots=["s3.png"])

        res_off: dict[str, Any] = {}
        _attach_visual_compare(res_off, [t1, t2], False, "demo")
        assert res_off == {"visual_advisory": None}

        fake_mgr = MagicMock()
        monkeypatch.setattr(ctx, "_baseline_manager", fake_mgr)

        with patch(
            "linc_crucible.core.bvt.visual_batch.compare_run",
            return_value={"aggregate_verdict": "PASS", "verdict": "OTHER"},
        ) as mock_cmp:
            res_agg: dict[str, Any] = {}
            _attach_visual_compare(res_agg, [t1, t2], True, "demo")
            mock_cmp.assert_called_once_with(fake_mgr, "demo", ["s1.png", "s2.png", "s3.png"])
            assert res_agg["visual_compare"] == {"aggregate_verdict": "PASS", "verdict": "OTHER"}
            assert res_agg["visual_advisory"] == "PASS"

        with patch(
            "linc_crucible.core.bvt.visual_batch.compare_run",
            return_value={"verdict": "WARN"},
        ):
            res_verdict: dict[str, Any] = {}
            _attach_visual_compare(res_verdict, [t1], True, "demo")
            assert res_verdict["visual_compare"] == {"verdict": "WARN"}
            assert res_verdict["visual_advisory"] == "WARN"

        with (
            structlog.testing.capture_logs() as cap_logs,
            patch(
                "linc_crucible.core.bvt.visual_batch.compare_run",
                side_effect=RuntimeError("boom"),
            ),
        ):
            res_err: dict[str, Any] = {}
            _attach_visual_compare(res_err, [t1], True, "demo")
            assert res_err["visual_compare"] == {"verdict": "ERROR", "error": "boom"}
            assert res_err["visual_advisory"] == "ERROR"
            assert any(
                e.get("event") == "crucible_bvt_visual_compare_error" and e.get("error") == "boom"
                for e in cap_logs
            )

    def test_precheck_helpers_accept_boundary_valid_inputs(self) -> None:
        assert (MAX_ASSETS_MIN, MAX_ASSETS_MAX) == (1, 50)
        assert (TOP_N_MIN, TOP_N_MAX) == (1, 1000)
        assert (LIMIT_MIN, LIMIT_MAX) == (1, 1000)
        assert (KEEP_MIN, KEEP_MAX) == (0, 1000)
        assert (SINCE_HOURS_MIN, SINCE_HOURS_MAX) == (0.0, 87600.0)
        assert (BUDGET_S_MIN, BUDGET_S_MAX) == (0.0, 86400.0)

        assert (
            pour_args_error("demo", "", "", False, False, "", "", "", False, "debug", False)
            is None
        )
        assert assay_args_error("demo", "quick", False, False, False, "", False) is None
        assert score_args_error("demo", False, 1) is None
        assert score_args_error("demo", True, 1000) is None
        assert score_args_error("demo", False, 0) is not None
        assert score_args_error("demo", False, 1001) is not None

        assert (
            bvt_args_error(
                "demo", "emulator-5554", ".bvt", None, None, None, False, None, 0.0, False
            )
            is None
        )
        assert (
            bvt_args_error(
                "demo", "emulator-5554", ".bvt", None, None, None, False, None, 86400.0, False
            )
            is None
        )
        assert (
            bvt_args_error(
                "demo", "emulator-5554", ".bvt", None, None, None, False, None, -0.01, False
            )
            is not None
        )
        assert (
            bvt_args_error(
                "demo", "emulator-5554", ".bvt", None, None, None, False, None, 86400.01, False
            )
            is not None
        )

        assert extract_assets_args_error("v.mp4", 1) is None
        assert extract_assets_args_error("v.mp4", 50) is None
        assert extract_assets_args_error("a" * 1024, 5) is None
        assert extract_assets_args_error("a" * 1025, 5) is not None
        assert extract_assets_args_error("v.mp4", 0) is not None
        assert extract_assets_args_error("v.mp4", 51) is not None

        assert usage_stats_args_error("", 0.0, False, 1) is None
        assert usage_stats_args_error("", 87600.0, True, 1000) is None

        assert (
            baseline_args_error(
                "prune", "demo", "", "a" * 1024, "", "", 0, "screenshot", "", 0.0, False, False
            )
            is None
        )
        assert (
            baseline_args_error(
                "prune", "demo", "home", "", "", "", 1000, "a11y", "", 1.0, True, True
            )
            is None
        )
        assert (
            baseline_args_error(
                "prune", "demo", "", "a" * 1025, "", "", 5, "screenshot", "", None, False, False
            )
            is not None
        )
