"""Tests for core.app_id — package / bundle-ID inference (moved from test_deployer.py, C4)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from linc_crucible.core.app_id import (
    _infer_bundle_id,
    _infer_package,
    _parse_application_id,
    _parse_bundle_id,
)
from linc_crucible.core.forge_integration import ForgeProject


class TestParseApplicationId:
    """Tests for _parse_application_id — parsing from build.gradle.kts."""

    def test_parses_from_app_module(self, tmp_path: Path) -> None:
        gradle = tmp_path / "app" / "build.gradle.kts"
        gradle.parent.mkdir()
        gradle.write_text(
            textwrap.dedent("""\
            android {
                namespace = "com.lincmobile.level"
                defaultConfig {
                    applicationId = "com.lincmobile.level"
                    minSdk = 26
                }
            }
        """)
        )
        assert _parse_application_id(str(tmp_path)) == "com.lincmobile.level"

    def test_parses_from_compose_app_module(self, tmp_path: Path) -> None:
        gradle = tmp_path / "composeApp" / "build.gradle.kts"
        gradle.parent.mkdir()
        gradle.write_text('applicationId = "com.sniperpulse.app"')
        assert _parse_application_id(str(tmp_path)) == "com.sniperpulse.app"

    def test_parses_from_android_app_module(self, tmp_path: Path) -> None:
        gradle = tmp_path / "androidApp" / "build.gradle.kts"
        gradle.parent.mkdir()
        gradle.write_text('applicationId = "app.imagination"')
        assert _parse_application_id(str(tmp_path)) == "app.imagination"

    def test_prefers_app_over_compose_app(self, tmp_path: Path) -> None:
        for name, pkg in [("app", "com.first"), ("composeApp", "com.second")]:
            gradle = tmp_path / name / "build.gradle.kts"
            gradle.parent.mkdir(exist_ok=True)
            gradle.write_text(f'applicationId = "{pkg}"')
        assert _parse_application_id(str(tmp_path)) == "com.first"

    def test_returns_none_when_no_gradle(self, tmp_path: Path) -> None:
        assert _parse_application_id(str(tmp_path)) is None

    def test_returns_none_when_no_application_id(self, tmp_path: Path) -> None:
        gradle = tmp_path / "app" / "build.gradle.kts"
        gradle.parent.mkdir()
        gradle.write_text('namespace = "com.example"')
        assert _parse_application_id(str(tmp_path)) is None

    def test_handles_spaces_around_equals(self, tmp_path: Path) -> None:
        gradle = tmp_path / "app" / "build.gradle.kts"
        gradle.parent.mkdir()
        gradle.write_text('applicationId  =  "com.lincmobile.level"')
        assert _parse_application_id(str(tmp_path)) == "com.lincmobile.level"


class TestParseBundleId:
    """Tests for _parse_bundle_id — parsing from iOS project files."""

    def test_parses_bundle_id_prefix_from_project_yml(self, tmp_path: Path) -> None:
        yml = tmp_path / "iosApp" / "project.yml"
        yml.parent.mkdir()
        yml.write_text(
            textwrap.dedent("""\
            name: PrecisionLevel
            bundleIdPrefix: com.lincmobile.level
            targets:
              iosApp:
                type: application
        """)
        )
        assert _parse_bundle_id(str(tmp_path)) == "com.lincmobile.level"

    def test_parses_product_bundle_id_from_project_yml(self, tmp_path: Path) -> None:
        yml = tmp_path / "iosApp" / "project.yml"
        yml.parent.mkdir()
        yml.write_text(
            textwrap.dedent("""\
            targets:
              iosApp:
                settings:
                  PRODUCT_BUNDLE_IDENTIFIER: com.sniperpulse.app
        """)
        )
        assert _parse_bundle_id(str(tmp_path)) == "com.sniperpulse.app"

    def test_parses_from_pbxproj(self, tmp_path: Path) -> None:
        pbxproj = tmp_path / "iosApp" / "MyApp.xcodeproj" / "project.pbxproj"
        pbxproj.parent.mkdir(parents=True)
        pbxproj.write_text(
            textwrap.dedent("""\
            buildSettings = {
                PRODUCT_BUNDLE_IDENTIFIER = com.example.myapp;
            };
        """)
        )
        assert _parse_bundle_id(str(tmp_path)) == "com.example.myapp"

    def test_skips_parameterized_bundle_id(self, tmp_path: Path) -> None:
        pbxproj = tmp_path / "iosApp" / "MyApp.xcodeproj" / "project.pbxproj"
        pbxproj.parent.mkdir(parents=True)
        pbxproj.write_text(
            textwrap.dedent("""\
            PRODUCT_BUNDLE_IDENTIFIER = "${BUNDLE_ID}${TEAM_ID}";
            PRODUCT_BUNDLE_IDENTIFIER = com.real.app;
        """)
        )
        assert _parse_bundle_id(str(tmp_path)) == "com.real.app"

    def test_prefers_project_yml_over_pbxproj(self, tmp_path: Path) -> None:
        yml = tmp_path / "iosApp" / "project.yml"
        yml.parent.mkdir()
        yml.write_text("bundleIdPrefix: com.from.yml")

        pbxproj = tmp_path / "iosApp" / "X.xcodeproj" / "project.pbxproj"
        pbxproj.parent.mkdir()
        pbxproj.write_text("PRODUCT_BUNDLE_IDENTIFIER = com.from.pbx;")

        assert _parse_bundle_id(str(tmp_path)) == "com.from.yml"

    def test_returns_none_when_no_ios_dir(self, tmp_path: Path) -> None:
        assert _parse_bundle_id(str(tmp_path)) is None

    def test_parse_bundle_id_root_level_pbxproj(self, tmp_path: Path) -> None:
        """_parse_bundle_id finds bundle ID from root-level xcodeproj."""
        xcodeproj = tmp_path / "MyApp.xcodeproj"
        xcodeproj.mkdir()
        (xcodeproj / "project.pbxproj").write_text(
            'PRODUCT_BUNDLE_IDENTIFIER = "com.example.myapp";\n'
        )

        result = _parse_bundle_id(str(tmp_path))
        assert result == "com.example.myapp"

    def test_returns_none_when_empty_project_yml(self, tmp_path: Path) -> None:
        yml = tmp_path / "iosApp" / "project.yml"
        yml.parent.mkdir()
        yml.write_text("name: Test")
        assert _parse_bundle_id(str(tmp_path)) is None


class TestInferPackage:
    """Tests for _infer_package — tier order: badged APK -> manifest package -> Gradle -> ''."""

    def test_parses_from_gradle(self, tmp_path: Path) -> None:
        gradle = tmp_path / "app" / "build.gradle.kts"
        gradle.parent.mkdir()
        gradle.write_text('applicationId = "com.lincmobile.level"')
        p = ForgeProject(name="PrecisionLevel", path=str(tmp_path))
        assert _infer_package(p) == "com.lincmobile.level"

    def test_returns_empty_when_no_path_and_no_manifest_package(self) -> None:
        p = ForgeProject(name="PrecisionLevel")
        assert _infer_package(p) == ""

    def test_returns_empty_when_no_gradle_and_no_manifest_package(self, tmp_path: Path) -> None:
        p = ForgeProject(name="ZenBalance", path=str(tmp_path))
        assert _infer_package(p) == ""

    def test_returns_empty_for_hyphenated_name_without_identity(self) -> None:
        p = ForgeProject(name="Zen-Balance")
        assert _infer_package(p) == ""

    def test_uses_debug_suffix_via_source(self, tmp_path: Path) -> None:
        mod = tmp_path / "composeApp"
        mod.mkdir()
        (mod / "build.gradle.kts").write_text(
            'android { defaultConfig { applicationId = "com.sniperpulse.app" }\n'
            '  buildTypes { getByName("debug") { applicationIdSuffix = ".dev" } } }'
        )
        p = ForgeProject(name="SniperPulseApp", path=str(tmp_path))
        assert _infer_package(p) == "com.sniperpulse.app.dev"

    def test_prefers_apk_badging(self, tmp_path: Path, monkeypatch) -> None:
        mod = tmp_path / "composeApp"
        mod.mkdir()
        (mod / "build.gradle.kts").write_text(
            'android { defaultConfig { applicationId = "com.base" } }'
        )
        p = ForgeProject(name="SniperPulseApp", path=str(tmp_path))
        monkeypatch.setattr(
            "linc_crucible.core.package_resolver.badge_package",
            lambda a: "com.badged.dev",
        )
        assert _infer_package(p, apk_path="/tmp/x.apk") == "com.badged.dev"

    def test_manifest_package_beats_source_parse_and_loses_to_badged_apk(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        mod = tmp_path / "app"
        mod.mkdir()
        (mod / "build.gradle.kts").write_text('applicationId = "com.from.gradle"')
        p = ForgeProject(
            name="MyApp",
            path=str(tmp_path),
            package="com.from.manifest",
        )
        # Without APK badging (or when badging misses), manifest identity.package beats Gradle
        monkeypatch.setattr(
            "linc_crucible.core.package_resolver.badge_package",
            lambda a: None,
        )
        assert _infer_package(p) == "com.from.manifest"
        assert _infer_package(p, apk_path="/tmp/x.apk") == "com.from.manifest"

        # With a badged APK, the badged APK beats manifest identity.package
        monkeypatch.setattr(
            "linc_crucible.core.package_resolver.badge_package",
            lambda a: "com.from.badged",
        )
        assert _infer_package(p, apk_path="/tmp/x.apk") == "com.from.badged"

    def test_manifest_package_used_when_no_path(self) -> None:
        p = ForgeProject(name="MyApp", package="com.from.manifest")
        assert _infer_package(p) == "com.from.manifest"


class TestInferBundleId:
    """Tests for _infer_bundle_id — tier order: manifest bundle_id -> project files -> ''."""

    def test_parses_from_project_yml(self, tmp_path: Path) -> None:
        yml = tmp_path / "iosApp" / "project.yml"
        yml.parent.mkdir()
        yml.write_text("bundleIdPrefix: com.lincmobile.level")
        p = ForgeProject(name="PrecisionLevel", path=str(tmp_path))
        assert _infer_bundle_id(p) == "com.lincmobile.level"

    def test_manifest_bundle_id_beats_project_yml(self, tmp_path: Path) -> None:
        yml = tmp_path / "iosApp" / "project.yml"
        yml.parent.mkdir()
        yml.write_text("bundleIdPrefix: com.from.yml")
        p = ForgeProject(
            name="MyApp",
            path=str(tmp_path),
            bundle_id="com.from.manifest",
        )
        assert _infer_bundle_id(p) == "com.from.manifest"

    def test_manifest_bundle_id_used_when_no_path(self) -> None:
        p = ForgeProject(name="MyApp", bundle_id="com.from.manifest")
        assert _infer_bundle_id(p) == "com.from.manifest"

    def test_returns_empty_when_no_path(self) -> None:
        p = ForgeProject(name="PrecisionLevel")
        assert _infer_bundle_id(p) == ""

    def test_returns_empty_when_no_ios_files(self, tmp_path: Path) -> None:
        p = ForgeProject(name="ZenBalance", path=str(tmp_path))
        assert _infer_bundle_id(p) == ""


def test_no_lincmobile_in_src() -> None:
    """[contract] git grep -n lincmobile -- src prints nothing."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    hits: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "lincmobile" in line:
                hits.append(f"{path.relative_to(src_root.parent)}:{lineno}:{line}")
    assert hits == []

