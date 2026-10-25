"""Tests for BVT schema parsing."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from linc_crucible.core.bvt.schema import (
    BVTStep,
    _parse_step,
    load_bvt_dir,
    load_bvt_file,
)


class TestParseStep:
    """Test step shorthand and dict parsing."""

    def test_shorthand_launch(self) -> None:
        step = _parse_step({"launch": "com.foo.bar"}, package="com.foo.bar")
        assert step == BVTStep(action="launch", target="com.foo.bar")

    def test_shorthand_wait(self) -> None:
        step = _parse_step({"wait": "3s"}, package="")
        assert step == BVTStep(action="wait", target="3s")

    def test_shorthand_assert_element(self) -> None:
        step = _parse_step({"assert_element": "level_screen"}, package="")
        assert step == BVTStep(action="assert_element", target="level_screen")

    def test_shorthand_screenshot(self) -> None:
        step = _parse_step({"screenshot": "launch_verified"}, package="")
        assert step == BVTStep(action="screenshot", target="launch_verified")

    def test_package_placeholder(self) -> None:
        step = _parse_step({"launch": "{package}"}, package="com.lincmobile.level")
        assert step.target == "com.lincmobile.level"

    def test_dict_form(self) -> None:
        raw = {"action": "tap", "target": "settings_button", "params": {"long": True}}
        step = _parse_step(raw, package="")
        assert step.action == "tap"
        assert step.target == "settings_button"
        assert step.params == {"long": True}

    def test_params_none_parses_to_empty_dict(self) -> None:
        raw = yaml.safe_load(
            dedent(
                """
                action: wait
                target: 1s
                params:
                """
            )
        )
        step = _parse_step(raw, package="")
        assert step.params == {}
        assert step.params.get("anything") is None

    def test_dict_valued_shorthand_routes_to_params(self) -> None:
        # e.g. `assert_min_size: {element: info_button, width: 56dp, height: 56dp}`
        raw = {"assert_min_size": {"element": "info_button", "width": "56dp", "height": "56dp"}}
        step = _parse_step(raw, package="com.x")
        assert step.action == "assert_min_size"
        assert step.target == ""
        assert step.params == {"element": "info_button", "width": "56dp", "height": "56dp"}

    def test_bare_string(self) -> None:
        step = _parse_step("screenshot", package="")
        assert step == BVTStep(action="screenshot")

    def test_none_target(self) -> None:
        step = _parse_step({"screenshot": None}, package="")
        assert step.target == ""

    def test_ambiguous_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="Ambiguous"):
            _parse_step({"launch": "foo", "wait": "1s"}, package="")

    def test_dismiss_dialog_shorthand(self) -> None:
        step = _parse_step({"dismiss_dialog": "Accept"}, package="")
        assert step == BVTStep(action="dismiss_dialog", target="Accept")

    def test_dismiss_dialog_dict_form(self) -> None:
        raw = {
            "action": "dismiss_dialog",
            "target": "Allow",
            "params": {"timeout": 5, "optional": True},
        }
        step = _parse_step(raw, package="")
        assert step.action == "dismiss_dialog"
        assert step.target == "Allow"
        assert step.params == {"timeout": 5, "optional": True}

    def test_tap_text_shorthand(self) -> None:
        step = _parse_step({"tap_text": "Settings"}, package="")
        assert step == BVTStep(action="tap_text", target="Settings")

    def test_tap_xy_dict_form(self) -> None:
        raw = {"action": "tap_xy", "target": "", "params": {"x": 120, "y": 340}}
        step = _parse_step(raw, package="")
        assert step.action == "tap_xy"
        assert step.params == {"x": 120, "y": 340}

    def test_tap_xy_dict_valued_shorthand(self) -> None:
        step = _parse_step({"tap_xy": {"x": 50, "y": 60}}, package="")
        assert step.action == "tap_xy"
        assert step.params == {"x": 50, "y": 60}

    def test_press_key_shorthand(self) -> None:
        step = _parse_step({"press_key": "ENTER"}, package="")
        assert step == BVTStep(action="press_key", target="ENTER")

    def test_press_key_bare_string(self) -> None:
        step = _parse_step("press_key", package="")
        assert step == BVTStep(action="press_key")

    def test_enter_bare_string(self) -> None:
        step = _parse_step("enter", package="")
        assert step == BVTStep(action="enter")

    def test_ime_action_dict_form(self) -> None:
        raw = {"action": "ime_action", "target": "", "params": {"key": "TAB"}}
        step = _parse_step(raw, package="")
        assert step.action == "ime_action"
        assert step.params == {"key": "TAB"}

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            _parse_step(42, package="")  # type: ignore[arg-type]


class TestLoadBVTFile:
    """Test loading a single YAML file."""

    def test_basic_file(self, tmp_path: Path) -> None:
        bvt_file = tmp_path / "test.yaml"
        bvt_file.write_text(
            dedent("""\
            package: com.lincmobile.level
            tests:
              - name: app_launches
                steps:
                  - launch: "{package}"
                  - wait: 3s
                  - assert_element: level_screen
                  - screenshot: launch_verified
        """)
        )

        package, tests, device = load_bvt_file(bvt_file)
        assert package == "com.lincmobile.level"
        assert device is None
        assert len(tests) == 1
        assert tests[0].name == "app_launches"
        assert len(tests[0].steps) == 4
        assert tests[0].steps[0].target == "com.lincmobile.level"

    def test_with_device(self, tmp_path: Path) -> None:
        bvt_file = tmp_path / "test.yaml"
        bvt_file.write_text(
            dedent("""\
            package: com.foo
            device: ABC123
            tests:
              - name: t1
                steps:
                  - launch: com.foo
        """)
        )

        _package, _tests, device = load_bvt_file(bvt_file)
        assert device == "ABC123"

    def test_file_package_wins_over_default(self, tmp_path: Path) -> None:
        # The file's own package wins; package_default is a fallback only.
        bvt_file = tmp_path / "test.yaml"
        bvt_file.write_text(
            dedent("""\
            package: com.original
            tests:
              - name: t1
                steps:
                  - launch: "{package}"
        """)
        )

        package, tests, _ = load_bvt_file(bvt_file, package_default="com.fallback")
        assert package == "com.original"
        assert tests[0].steps[0].target == "com.original"
        assert tests[0].package == "com.original"

    def test_default_used_when_file_omits_package(self, tmp_path: Path) -> None:
        bvt_file = tmp_path / "test.yaml"
        bvt_file.write_text(
            dedent("""\
            tests:
              - name: t1
                steps:
                  - launch: "{package}"
        """)
        )

        package, tests, _ = load_bvt_file(bvt_file, package_default="com.fallback")
        assert package == "com.fallback"
        assert tests[0].steps[0].target == "com.fallback"

    def test_invalid_yaml_type(self, tmp_path: Path) -> None:
        bvt_file = tmp_path / "test.yaml"
        bvt_file.write_text("just a string")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_bvt_file(bvt_file)

    def test_with_tags(self, tmp_path: Path) -> None:
        bvt_file = tmp_path / "test.yaml"
        bvt_file.write_text(
            dedent("""\
            package: com.foo
            tests:
              - name: t1
                tags: [smoke, phone]
                steps:
                  - launch: com.foo
        """)
        )

        _, tests, _ = load_bvt_file(bvt_file)
        assert tests[0].tags == ["smoke", "phone"]


class TestLoadBVTDir:
    """Test loading a .bvt/ directory."""

    def test_single_file(self, tmp_path: Path) -> None:
        bvt_dir = tmp_path / ".bvt"
        bvt_dir.mkdir()
        (bvt_dir / "launch.yaml").write_text(
            dedent("""\
            package: com.lincmobile.level
            tests:
              - name: launch_test
                steps:
                  - launch: "{package}"
                  - screenshot: launched
        """)
        )

        suite = load_bvt_dir(bvt_dir)
        assert suite.package == "com.lincmobile.level"
        assert len(suite.tests) == 1
        assert suite.device is None

    def test_multiple_files_merged(self, tmp_path: Path) -> None:
        bvt_dir = tmp_path / ".bvt"
        bvt_dir.mkdir()

        (bvt_dir / "01-launch.yaml").write_text(
            dedent("""\
            package: com.foo
            device: DEV1
            tests:
              - name: launch
                steps:
                  - launch: "{package}"
        """)
        )
        (bvt_dir / "02-settings.yaml").write_text(
            dedent("""\
            tests:
              - name: settings
                steps:
                  - tap: settings_button
        """)
        )

        suite = load_bvt_dir(bvt_dir)
        assert suite.package == "com.foo"
        assert suite.device == "DEV1"
        assert len(suite.tests) == 2
        assert suite.tests[0].name == "launch"
        assert suite.tests[1].name == "settings"

    def test_yml_extension(self, tmp_path: Path) -> None:
        bvt_dir = tmp_path / ".bvt"
        bvt_dir.mkdir()
        (bvt_dir / "test.yml").write_text(
            dedent("""\
            package: com.foo
            tests:
              - name: t1
                steps:
                  - launch: com.foo
        """)
        )

        suite = load_bvt_dir(bvt_dir)
        assert len(suite.tests) == 1

    def test_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_bvt_dir(tmp_path / "nonexistent")

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        bvt_dir = tmp_path / ".bvt"
        bvt_dir.mkdir()

        with pytest.raises(ValueError, match="No YAML files"):
            load_bvt_dir(bvt_dir)

    def test_no_package_raises(self, tmp_path: Path) -> None:
        bvt_dir = tmp_path / ".bvt"
        bvt_dir.mkdir()
        (bvt_dir / "test.yaml").write_text(
            dedent("""\
            tests:
              - name: t1
                steps:
                  - screenshot: test
        """)
        )

        with pytest.raises(ValueError, match="no package"):
            load_bvt_dir(bvt_dir)

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        """Files are sorted alphabetically for deterministic test order."""
        bvt_dir = tmp_path / ".bvt"
        bvt_dir.mkdir()

        (bvt_dir / "z-last.yaml").write_text(
            dedent("""\
            package: com.foo
            tests:
              - name: z_test
                steps:
                  - launch: com.foo
        """)
        )
        (bvt_dir / "a-first.yaml").write_text(
            dedent("""\
            package: com.foo
            tests:
              - name: a_test
                steps:
                  - launch: com.foo
        """)
        )

        suite = load_bvt_dir(bvt_dir)
        assert suite.tests[0].name == "a_test"
        assert suite.tests[1].name == "z_test"


class TestParseConditional:
    """T5: if/then/else conditional steps (reactor b46ad539)."""

    def test_if_element_then_else(self) -> None:
        raw = {
            "if": {"element": "consent_dialog"},
            "then": [{"tap": "accept_btn"}],
            "else": [],
        }
        step = _parse_step(raw, package="com.x")
        assert step.action == "if"
        assert step.params["condition"] == {"element": "consent_dialog"}
        assert [s.action for s in step.params["then"]] == ["tap"]
        assert step.params["else"] == []

    def test_if_text_condition(self) -> None:
        step = _parse_step({"if": {"text": "Allow"}, "then": []}, package="")
        assert step.params["condition"] == {"text": "Allow"}

    def test_if_missing_else_defaults_empty(self) -> None:
        step = _parse_step({"if": {"element": "x"}, "then": [{"tap": "y"}]}, package="")
        assert step.params["else"] == []

    def test_if_package_substitution_in_element(self) -> None:
        step = _parse_step({"if": {"element": "{package}:id/d"}, "then": []}, package="com.x")
        assert step.params["condition"]["element"] == "com.x:id/d"

    def test_nested_if_rejected(self) -> None:
        raw = {"if": {"element": "x"}, "then": [{"if": {"element": "y"}, "then": []}]}
        with pytest.raises(ValueError, match=r"[Nn]ested"):
            _parse_step(raw, package="")

    def test_condition_not_mapping_rejected(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            _parse_step({"if": "consent_dialog", "then": []}, package="")

    def test_condition_both_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _parse_step({"if": {"element": "x", "text": "y"}, "then": []}, package="")

    def test_condition_neither_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _parse_step({"if": {"foo": "bar"}, "then": []}, package="")

    def test_then_not_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="list"):
            _parse_step({"if": {"element": "x"}, "then": {"tap": "y"}}, package="")

    def test_unknown_sibling_key_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"[Uu]nexpected"):
            _parse_step({"if": {"element": "x"}, "then": [], "finally": []}, package="")
