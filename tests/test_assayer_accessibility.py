"""Tests for _run_accessibility() detection logic in assayer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from linc_crucible.core.assayer import Assayer
from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader
from linc_crucible.core.scorer import QualityScorer, a11y_issues_to_score


@pytest.fixture
def assayer(tmp_path: Path) -> Assayer:
    forge = MagicMock(spec=ForgeProjectReader)
    scorer = QualityScorer(scores_dir=tmp_path / "scores")
    return Assayer(forge_reader=forge, scorer=scorer)


def _make_project(tmp_path: Path, name: str = "TestProject") -> ForgeProject:
    """Create a ForgeProject pointing at a tmp directory."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)
    return ForgeProject(
        name=name,
        path=str(project_dir),
        test_command="true",
        lint_command="true",
        framework="gradle_kmp",
    )


def _layout_dir(tmp_path: Path) -> Path:
    """Create and return the standard layout directory."""
    d = tmp_path / "project" / "src" / "main" / "res" / "layout"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kotlin_dir(tmp_path: Path) -> Path:
    """Create and return a Kotlin source directory."""
    d = tmp_path / "project" / "src" / "main" / "kotlin" / "com" / "app"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# XML contentDescription tests
# ---------------------------------------------------------------------------


class TestXmlContentDescription:
    def test_imageview_missing(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        (layout / "activity_main.xml").write_text(
            '<LinearLayout>\n  <ImageView android:src="@drawable/ic" />\n</LinearLayout>'
        )

        result = assayer._run_accessibility(project)

        assert result.success is True
        assert result.major == 1
        assert any("ImageView" in d for d in result.details)

    def test_imagebutton_missing(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        (layout / "activity_main.xml").write_text(
            '<LinearLayout>\n  <ImageButton android:src="@drawable/ic" />\n</LinearLayout>'
        )

        result = assayer._run_accessibility(project)

        assert result.major == 1
        assert any("ImageButton" in d for d in result.details)

    def test_imageview_has_content_description(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        (layout / "activity_main.xml").write_text(
            "<LinearLayout>\n"
            '  <ImageView android:src="@drawable/ic"\n'
            '    android:contentDescription="Logo" />\n'
            "</LinearLayout>"
        )

        result = assayer._run_accessibility(project)

        assert result.major == 0
        assert result.score == 100.0

    def test_multiple_images_same_file(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        (layout / "activity_main.xml").write_text(
            "<LinearLayout>\n"
            '  <ImageView android:src="@drawable/a" />\n'
            '  <ImageView android:src="@drawable/b" />\n'
            "</LinearLayout>"
        )

        result = assayer._run_accessibility(project)

        assert result.major == 2

    def test_xml_not_in_layout_dir_skipped(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        values_dir = tmp_path / "project" / "src" / "main" / "res" / "values"
        values_dir.mkdir(parents=True)
        (values_dir / "strings.xml").write_text(
            '<resources>\n  <ImageView android:src="@drawable/ic" />\n</resources>'
        )

        result = assayer._run_accessibility(project)

        assert result.major == 0


# ---------------------------------------------------------------------------
# Compose contentDescription tests
# ---------------------------------------------------------------------------


class TestComposeContentDescription:
    def test_image_missing(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        kt_dir = _kotlin_dir(tmp_path)
        (kt_dir / "Screen.kt").write_text("Image(painter = painterResource(R.drawable.logo))")

        result = assayer._run_accessibility(project)

        assert result.major == 1
        assert any("Image()" in d for d in result.details)

    def test_icon_missing(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        kt_dir = _kotlin_dir(tmp_path)
        (kt_dir / "Screen.kt").write_text("Icon(imageVector = Icons.Default.Add)")

        result = assayer._run_accessibility(project)

        assert result.major == 1
        assert any("Icon()" in d for d in result.details)

    def test_image_with_content_description(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        kt_dir = _kotlin_dir(tmp_path)
        (kt_dir / "Screen.kt").write_text(
            "Image(painter = painterResource(R.drawable.logo),\n"
            '    contentDescription = "App logo")'
        )

        result = assayer._run_accessibility(project)

        assert result.major == 0

    def test_icon_with_content_description(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        kt_dir = _kotlin_dir(tmp_path)
        (kt_dir / "Screen.kt").write_text(
            'Icon(imageVector = Icons.Default.Add,\n    contentDescription = "Add item")'
        )

        result = assayer._run_accessibility(project)

        assert result.major == 0


# ---------------------------------------------------------------------------
# Touch target size tests
# ---------------------------------------------------------------------------


class TestTouchTargetSize:
    def test_small_target_flagged(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        (layout / "activity_main.xml").write_text(
            "<LinearLayout>\n"
            '  <Button android:clickable="true"\n'
            '    android:layout_width="32dp"\n'
            '    android:layout_height="32dp" />\n'
            "</LinearLayout>"
        )

        result = assayer._run_accessibility(project)

        assert result.major >= 1
        assert any("32dp" in d for d in result.details)

    def test_48dp_not_flagged(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        (layout / "activity_main.xml").write_text(
            "<LinearLayout>\n"
            '  <Button android:clickable="true"\n'
            '    android:layout_width="48dp"\n'
            '    android:layout_height="48dp" />\n'
            "</LinearLayout>"
        )

        result = assayer._run_accessibility(project)

        assert result.major == 0

    def test_non_clickable_not_flagged(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        (layout / "activity_main.xml").write_text(
            "<LinearLayout>\n"
            '  <View android:layout_width="32dp"\n'
            '    android:layout_height="32dp" />\n'
            "</LinearLayout>"
        )

        result = assayer._run_accessibility(project)

        assert result.major == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAccessibilityEdgeCases:
    def test_project_path_not_found(self, assayer: Assayer, tmp_path: Path) -> None:
        project = ForgeProject(
            name="Ghost",
            path=str(tmp_path / "nonexistent"),
            test_command="true",
            lint_command="true",
        )

        result = assayer._run_accessibility(project)

        assert result.success is False
        assert "not found" in result.output

    def test_empty_files(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        kt_dir = _kotlin_dir(tmp_path)
        (layout / "empty.xml").write_text("")
        (kt_dir / "Empty.kt").write_text("")

        result = assayer._run_accessibility(project)

        assert result.success is True
        assert result.major == 0

    def test_unreadable_file_graceful(
        self, assayer: Assayer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        good_file = layout / "good.xml"
        good_file.write_text('<ImageView android:src="@drawable/ic" />')
        bad_file = layout / "bad.xml"
        bad_file.write_text('<ImageView android:src="@drawable/x" />')

        original_read_text = Path.read_text

        def _patched_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "bad.xml":
                raise OSError("Permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _patched_read_text)

        result = assayer._run_accessibility(project)

        # The good file should still be processed
        assert result.success is True
        assert result.major >= 1

    def test_score_computed(self, assayer: Assayer, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        layout = _layout_dir(tmp_path)
        kt_dir = _kotlin_dir(tmp_path)
        # 1 XML issue + 2 Compose issues = 3 major
        (layout / "main.xml").write_text(
            '<LinearLayout>\n  <ImageView android:src="@drawable/ic" />\n</LinearLayout>'
        )
        (kt_dir / "Screen.kt").write_text(
            "Image(painter = painterResource(R.drawable.a))\n"
            "Icon(imageVector = Icons.Default.Star)\n"
        )

        result = assayer._run_accessibility(project)

        expected_score = a11y_issues_to_score(critical=0, major=3, minor=0)
        assert result.score == expected_score
        assert result.major == 3


# ---------------------------------------------------------------------------
# Live a11y_tree personal-device policy tests (R4 H2, M3, L9)
# ---------------------------------------------------------------------------

_PERSONAL_SERIAL = "R58N000TEST"
_SAFE_SERIAL = "BENCH_SAFE_01"


def _personal_policy(tmp_path: Path):
    from linc_crucible.core.device_policy import load_device_policy

    fleet = tmp_path / "device-fleet.yaml"
    fleet.write_text(
        f"version: 1\n"
        f"devices:\n"
        f"  - id: personal-phone\n"
        f"    serial: '{_PERSONAL_SERIAL}'\n"
        f"    personal: true\n"
        f"  - id: bench-phone\n"
        f"    serial: '{_SAFE_SERIAL}'\n"
        f"    personal: false\n"
    )
    return load_device_policy(fleet)


class TestAssayA11yTreePersonalDevicePolicy:
    """[contract] crucible_assay(a11y_tree=True) goes through the DevicePolicy gate."""

    @pytest.mark.asyncio
    async def test_crucible_assay_explicit_personal_serial_refused_without_elements_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] crucible_assay(project, a11y_tree=True, device=<personal serial>)
        makes no nerve elements call and returns a11y_tree.error_category == 'personal_device'."""
        import linc_crucible.server_context as ctx
        from linc_crucible.assay_tools import crucible_assay

        policy = _personal_policy(tmp_path)
        project = _make_project(tmp_path)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        nerve = MagicMock()
        nerve.discover.return_value = [
            {"id": _PERSONAL_SERIAL, "serial": _PERSONAL_SERIAL, "platform": "android"}
        ]
        assayer_inst = Assayer(
            forge_reader=forge,
            scorer=scorer,
            nerve_client=nerve,
            device_policy=policy,
        )
        monkeypatch.setattr(ctx, "_initialized", True)
        monkeypatch.setattr(ctx, "_assayer", assayer_inst)
        monkeypatch.setattr(ctx, "_cached_device_policy", policy)

        res = await crucible_assay(
            project="TestProject",
            a11y_tree=True,
            device=_PERSONAL_SERIAL,
        )

        nerve.elements_strict.assert_not_called()
        nerve.elements.assert_not_called()
        assert res["a11y_tree"]["success"] is False
        assert res["a11y_tree"]["configured"] is True
        assert res["a11y_tree"]["attempted_unmeasured"] is False
        assert res["a11y_tree"]["error_category"] == "personal_device"
        assert _PERSONAL_SERIAL in res["a11y_tree"]["error"]
        assert "allow_personal=True" in res["a11y_tree"]["error"]
        assert res["a11y_tree"]["output"] == res["a11y_tree"]["error"]

    def test_auto_pick_with_personal_and_safe_reads_safe(self, tmp_path: Path) -> None:
        """[contract] auto-pick with [personal, safe] discover reads safe."""
        policy = _personal_policy(tmp_path)
        project = _make_project(tmp_path)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        nerve = MagicMock()
        nerve.discover.return_value = [
            {"id": _PERSONAL_SERIAL, "serial": _PERSONAL_SERIAL, "platform": "android"},
            {"id": _SAFE_SERIAL, "serial": _SAFE_SERIAL, "platform": "android"},
        ]
        nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "OK", "bounds": "[0,0][100,100]"}
        ]
        tree_capture = MagicMock()
        tree_capture.get_latest.return_value = None
        assayer_inst = Assayer(
            forge_reader=forge,
            scorer=scorer,
            nerve_client=nerve,
            tree_capture=tree_capture,
            device_policy=lambda: policy,
        )

        report = assayer_inst.assay("TestProject", a11y_tree=True)

        nerve.elements_strict.assert_called_once_with(_SAFE_SERIAL)
        tree_capture.get_latest.assert_called_once_with("TestProject", "default")
        assert report.a11y_tree is not None
        assert report.a11y_tree.success is True

    def test_auto_pick_with_only_personal_devices_refuses(self, tmp_path: Path) -> None:
        """[contract] auto-pick with only personal devices -> refusal."""
        policy = _personal_policy(tmp_path)
        project = _make_project(tmp_path)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        nerve = MagicMock()
        nerve.discover.return_value = [
            {"id": _PERSONAL_SERIAL, "serial": _PERSONAL_SERIAL, "platform": "android"}
        ]
        assayer_inst = Assayer(
            forge_reader=forge,
            scorer=scorer,
            nerve_client=nerve,
            device_policy=policy,
        )

        report = assayer_inst.assay("TestProject", a11y_tree=True)

        nerve.elements_strict.assert_not_called()
        assert report.a11y_tree is not None
        assert report.a11y_tree.success is False
        assert report.a11y_tree.configured is True
        assert report.a11y_tree.attempted is False
        d = report.to_dict()
        assert d["a11y_tree"]["error_category"] == "personal_device"
        assert d["a11y_tree"]["attempted_unmeasured"] is False

    def test_unreadable_registry_refuses_a11y_tree(self, tmp_path: Path) -> None:
        """[contract] unreadable registry -> refusal (fail closed) unless allow_personal."""
        from linc_crucible.core.device_policy import load_device_policy

        bad_fleet = tmp_path / "bad-fleet.yaml"
        bad_fleet.write_text("::: unparseable :::\n")
        policy = load_device_policy(bad_fleet)
        project = _make_project(tmp_path)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        nerve = MagicMock()
        nerve.discover.return_value = [
            {"id": _SAFE_SERIAL, "serial": _SAFE_SERIAL, "platform": "android"}
        ]
        assayer_inst = Assayer(
            forge_reader=forge,
            scorer=scorer,
            nerve_client=nerve,
            device_policy=policy,
        )

        report = assayer_inst.assay("TestProject", a11y_tree=True)

        nerve.elements_strict.assert_not_called()
        d = report.to_dict()
        assert d["a11y_tree"]["success"] is False
        assert d["a11y_tree"]["error_category"] == "personal_device"
        assert "bad-fleet.yaml" in d["a11y_tree"]["error"]

    @pytest.mark.asyncio
    async def test_allow_personal_reads_personal_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[contract] allow_personal=True -> reads the personal device."""
        import linc_crucible.server_context as ctx
        from linc_crucible.assay_tools import crucible_assay

        policy = _personal_policy(tmp_path)
        project = _make_project(tmp_path)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        nerve = MagicMock()
        nerve.discover.return_value = [
            {"id": _PERSONAL_SERIAL, "serial": _PERSONAL_SERIAL, "platform": "android"}
        ]
        nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "OK", "bounds": "[0,0][100,100]"}
        ]
        assayer_inst = Assayer(
            forge_reader=forge,
            scorer=scorer,
            nerve_client=nerve,
            device_policy=policy,
        )
        monkeypatch.setattr(ctx, "_initialized", True)
        monkeypatch.setattr(ctx, "_assayer", assayer_inst)
        monkeypatch.setattr(ctx, "_cached_device_policy", policy)

        res = await crucible_assay(
            project="TestProject",
            a11y_tree=True,
            device=_PERSONAL_SERIAL,
            allow_personal=True,
        )

        nerve.elements_strict.assert_called_once_with(_PERSONAL_SERIAL)
        assert res["a11y_tree"]["success"] is True
        assert "error_category" not in res["a11y_tree"]

    def test_alias_with_personal_hardware_id_refused(self, tmp_path: Path) -> None:
        """[contract] Explicit alias (192.0.2.10:5555) with personal hardware_id is refused."""
        policy = _personal_policy(tmp_path)
        project = _make_project(tmp_path)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        nerve = MagicMock()
        nerve.discover.return_value = [
            {
                "id": "192.0.2.10:5555",
                "serial": "192.0.2.10:5555",
                "hardware_id": _PERSONAL_SERIAL,
                "platform": "android",
            }
        ]
        assayer_inst = Assayer(
            forge_reader=forge,
            scorer=scorer,
            nerve_client=nerve,
            device_policy=policy,
        )

        report = assayer_inst.assay("TestProject", a11y_tree=True, device="192.0.2.10:5555")

        nerve.elements_strict.assert_not_called()
        d = report.to_dict()
        assert d["a11y_tree"]["success"] is False
        assert d["a11y_tree"]["error_category"] == "personal_device"
        assert "192.0.2.10:5555" in d["a11y_tree"]["error"]

    def test_crucible_assay_docstring_documents_allow_personal(self) -> None:
        from linc_crucible.assay_tools import crucible_assay

        doc = crucible_assay.__doc__ or ""
        assert "allow_personal:" in doc

    def test_nerve_unreachable_during_a11y_tree_still_refuses_personal_serial(
        self, tmp_path: Path
    ) -> None:
        """[test] When nerve.discover() raises ConnectionError, the string-only check
        still refuses an exact personal serial before any elements call."""
        policy = _personal_policy(tmp_path)
        project = _make_project(tmp_path)
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = project
        scorer = QualityScorer(scores_dir=tmp_path / "scores")
        nerve = MagicMock()
        nerve.discover.side_effect = ConnectionError("nerve unreachable")
        assayer_inst = Assayer(
            forge_reader=forge,
            scorer=scorer,
            nerve_client=nerve,
            device_policy=policy,
        )

        report = assayer_inst.assay("TestProject", a11y_tree=True, device=_PERSONAL_SERIAL.lower())

        nerve.elements_strict.assert_not_called()
        d = report.to_dict()
        assert d["a11y_tree"]["success"] is False
        assert d["a11y_tree"]["error_category"] == "personal_device"
        assert d["a11y_tree"]["attempted_unmeasured"] is False

    def test_resolve_a11y_policy_fallback_and_empty_discover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import linc_crucible.server_context as ctx

        policy = _personal_policy(tmp_path)
        monkeypatch.setattr(ctx, "_cached_device_policy", policy)
        nerve = MagicMock()
        nerve.discover.return_value = [{}]
        assayer_inst = Assayer(nerve_client=nerve, device_policy=None)
        assert assayer_inst._resolve_device_policy() is policy
        discovered, refusal = assayer_inst._resolve_a11y_policy("", False)
        assert discovered == [{}]
        assert refusal is None
        nerve.discover.return_value = []
        assert assayer_inst._resolve_a11y_policy("", False) == ([], None)
        nerve.discover.return_value = [{"id": _PERSONAL_SERIAL, "serial": _PERSONAL_SERIAL}]
        assert assayer_inst._run_a11y_tree("TestProject", _PERSONAL_SERIAL).success is False
