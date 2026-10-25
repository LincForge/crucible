"""Tests for forge integration (read-only registry access)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from linc_crucible.core.forge_integration import (
    ForgeProject,
    ForgeProjectReader,
    ModuleConfig,
    _apply_manifest_modules,
    _manifest_deployment_modules,
)


@pytest.fixture
def forge_registry(tmp_path: Path) -> Path:
    """Create a temporary forge registry with test projects."""
    registry_path = tmp_path / ".forge" / "projects.yaml"
    registry_path.parent.mkdir(parents=True)

    # Create project directories with .forge.yaml
    proj1_dir = tmp_path / "projects" / "PrecisionLevel"
    proj1_dir.mkdir(parents=True)
    (proj1_dir / ".forge.yaml").write_text(
        yaml.safe_dump(
            {
                "framework": "gradle_kmp",
                "test_command": "./gradlew test",
                "build_command": "./gradlew assembleDebug",
                "lint_command": "./gradlew lint",
                "target_branch_default": "main",
                "linked_projects": ["SensorKit"],
            }
        )
    )

    proj2_dir = tmp_path / "projects" / "linc-sinew"
    proj2_dir.mkdir(parents=True)
    (proj2_dir / ".forge.yaml").write_text(
        yaml.safe_dump(
            {
                "framework": "python_setup",
                "test_command": "pytest tests/ -v",
                "build_command": "pip install -e .",
                "lint_command": "ruff check src/ tests/",
            }
        )
    )

    # Registry file
    data = {
        "projects": [
            {
                "name": "PrecisionLevel",
                "path": str(proj1_dir),
                "github_project": "LincForge/PrecisionLevel",
            },
            {
                "name": "linc-sinew",
                "path": str(proj2_dir),
                "github_project": "LincForge/linc-sinew",
            },
        ]
    }
    registry_path.write_text(yaml.safe_dump(data))

    return registry_path


class TestForgeProjectReaderLoad:
    def test_load_registry(self, forge_registry: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)
        projects = reader.list_projects()
        assert len(projects) == 2

    def test_get_project(self, forge_registry: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)
        proj = reader.get("PrecisionLevel")
        assert proj is not None
        assert proj.name == "PrecisionLevel"
        assert proj.github_project == "LincForge/PrecisionLevel"
        assert proj.framework == "gradle_kmp"
        assert proj.build_command == "./gradlew assembleDebug"
        assert proj.test_command == "./gradlew test"
        assert "SensorKit" in proj.linked_projects

    def test_get_nonexistent(self, forge_registry: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)
        assert reader.get("NonExistent") is None

    def test_nonexistent_registry(self, tmp_path: Path) -> None:
        reader = ForgeProjectReader(registry_path=tmp_path / "nope.yaml")
        assert reader.list_projects() == []

    def test_empty_registry(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "empty.yaml"
        registry_path.write_text("")
        reader = ForgeProjectReader(registry_path=registry_path)
        assert reader.list_projects() == []


class TestForgeProjectReaderArtifacts:
    def test_find_apk(self, forge_registry: Path, tmp_path: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)

        # Create a fake APK
        proj_dir = tmp_path / "projects" / "PrecisionLevel"
        apk_dir = proj_dir / "composeApp" / "build" / "outputs" / "apk" / "debug"
        apk_dir.mkdir(parents=True)
        apk_file = apk_dir / "composeApp-debug.apk"
        apk_file.write_text("fake apk")

        result = reader.find_apk("PrecisionLevel")
        assert result is not None
        assert result.endswith(".apk")

    def test_find_apk_nonexistent_project(self, forge_registry: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)
        assert reader.find_apk("NonExistent") is None

    def test_find_apk_no_artifacts(self, forge_registry: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)
        assert reader.find_apk("PrecisionLevel") is None

    def test_find_ios_app(self, forge_registry: Path, tmp_path: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)

        # Create a fake .app
        proj_dir = tmp_path / "projects" / "PrecisionLevel"
        app_dir = proj_dir / "build" / "Build" / "Products" / "Debug-iphonesimulator"
        app_dir.mkdir(parents=True)
        app_path = app_dir / "PrecisionLevel.app"
        app_path.mkdir()

        result = reader.find_ios_app("PrecisionLevel")
        assert result is not None
        assert result.endswith(".app")

    def test_find_ios_app_no_artifacts(
        self, forge_registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        reader = ForgeProjectReader(registry_path=forge_registry)
        assert reader.find_ios_app("PrecisionLevel") is None


class TestForgeProjectReaderEnvVar:
    def test_env_var_override(self, forge_registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_PROJECTS_FILE", str(forge_registry))
        monkeypatch.setenv("LINC_PROJECTS_DIR", str(forge_registry.parent.parent / "projects"))
        reader = ForgeProjectReader()
        assert len(reader.list_projects()) == 2

    def test_forge_projects_file_with_manifest_on_disk_discovers_and_wins_on_clash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[test] FORGE_PROJECTS_FILE set + a manifest on disk -> manifest project is
        discovered and wins on a name clash.
        """
        legacy_Clash_dir = tmp_path / "legacy" / "ClashApp"
        legacy_Clash_dir.mkdir(parents=True)
        legacy_only_dir = tmp_path / "legacy" / "LegacyOnly"
        legacy_only_dir.mkdir(parents=True)

        registry_path = tmp_path / "custom-projects.yaml"
        registry_path.write_text(
            yaml.safe_dump(
                {
                    "projects": [
                        {"name": "ClashApp", "path": str(legacy_Clash_dir)},
                        {"name": "LegacyOnly", "path": str(legacy_only_dir)},
                    ]
                }
            ),
            encoding="utf-8",
        )

        projects_dir = tmp_path / "projects"
        manifest_clash_dir = projects_dir / "ClashApp"
        manifest_clash_dir.mkdir(parents=True)
        (manifest_clash_dir / "linc-manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "identity": {
                        "name": "ClashApp",
                        "package": "com.example.clash",
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_only_dir = projects_dir / "ManifestOnly"
        manifest_only_dir.mkdir(parents=True)
        (manifest_only_dir / "linc-manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "identity": {
                        "name": "ManifestOnly",
                        "package": "com.example.manifestonly",
                    },
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("FORGE_PROJECTS_FILE", str(registry_path))
        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))

        reader = ForgeProjectReader()
        manifest_only = reader.get("ManifestOnly")
        assert manifest_only is not None
        assert manifest_only.path == str(manifest_only_dir)

        clash = reader.get("ClashApp")
        assert clash is not None
        assert clash.path == str(manifest_clash_dir)
        assert clash.package == "com.example.clash"

        legacy_only = reader.get("LegacyOnly")
        assert legacy_only is not None
        assert legacy_only.path == str(legacy_only_dir)


# --- Multi-target deployment tests ---


class TestModuleConfig:
    def test_defaults(self) -> None:
        m = ModuleConfig(name="app")
        assert m.name == "app"
        assert m.device_types == []
        assert m.build_command == ""
        assert m.apk_pattern == ""

    def test_all_fields(self) -> None:
        m = ModuleConfig(
            name="watchApp",
            device_types=["watch"],
            build_command="./gradlew :watchApp:assembleDebug",
            apk_pattern="watchApp/build/outputs/apk/debug/*.apk",
        )
        assert m.name == "watchApp"
        assert m.device_types == ["watch"]
        assert m.build_command == "./gradlew :watchApp:assembleDebug"
        assert m.apk_pattern == "watchApp/build/outputs/apk/debug/*.apk"


class TestMergeForgeYamlNested:
    """Tests for the forge: key unwrapping fix."""

    def _make_registry(self, tmp_path: Path, forge_yaml_content: str) -> Path:
        """Helper to create a registry with a single project."""
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "TestProject"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(forge_yaml_content)

        data = {"projects": [{"name": "TestProject", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))
        return registry_path

    def test_nested_forge_key(self, tmp_path: Path) -> None:
        """Real .forge.yaml files nest under forge: — verify we unwrap."""
        content = yaml.safe_dump(
            {
                "forge": {
                    "test_command": "./gradlew test",
                    "build_command": "./gradlew build",
                    "framework": "gradle_kmp",
                }
            }
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("TestProject")
        assert proj is not None
        assert proj.test_command == "./gradlew test"
        assert proj.build_command == "./gradlew build"
        assert proj.framework == "gradle_kmp"

    def test_flat_format_backwards_compat(self, tmp_path: Path) -> None:
        """Flat format (no forge: wrapper) still works for backwards compat."""
        content = yaml.safe_dump(
            {
                "test_command": "./gradlew test",
                "build_command": "./gradlew build",
            }
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("TestProject")
        assert proj is not None
        assert proj.test_command == "./gradlew test"
        assert proj.build_command == "./gradlew build"

    def test_nested_with_extra_top_level_keys(self, tmp_path: Path) -> None:
        """Extra top-level keys alongside forge: are ignored."""
        content = yaml.safe_dump(
            {
                "forge": {
                    "build_command": "./gradlew assembleDebug",
                    "lint_command": "./gradlew lint",
                },
                "other_tool": {"key": "value"},
            }
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("TestProject")
        assert proj.build_command == "./gradlew assembleDebug"
        assert proj.lint_command == "./gradlew lint"


class TestForgeProjectModules:
    """Tests for parsing modules from .forge.yaml."""

    def _make_registry(self, tmp_path: Path, forge_yaml_content: str) -> Path:
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "MultiApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(forge_yaml_content)

        data = {"projects": [{"name": "MultiApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))
        return registry_path

    def test_parse_modules_from_yaml(self, tmp_path: Path) -> None:
        content = yaml.safe_dump(
            {
                "forge": {
                    "build_command": "./gradlew build",
                    "modules": [
                        {
                            "name": "composeApp",
                            "device_types": ["phone", "tablet", "emulator"],
                            "build_command": "./gradlew :composeApp:assembleDebug",
                            "apk_pattern": "composeApp/build/outputs/apk/debug/*.apk",
                        },
                        {
                            "name": "watchApp",
                            "device_types": ["watch"],
                            "build_command": "./gradlew :watchApp:assembleDebug",
                            "apk_pattern": "watchApp/build/outputs/apk/debug/*.apk",
                        },
                    ],
                }
            }
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("MultiApp")

        assert proj is not None
        assert proj.is_multi_target is True
        assert len(proj.modules) == 2
        assert proj.modules[0].name == "composeApp"
        assert proj.modules[0].device_types == ["phone", "tablet", "emulator"]
        assert proj.modules[1].name == "watchApp"
        assert proj.modules[1].device_types == ["watch"]

    def test_empty_modules_list(self, tmp_path: Path) -> None:
        content = yaml.safe_dump({"forge": {"build_command": "./gradlew build", "modules": []}})
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("MultiApp")
        assert proj.is_multi_target is False
        assert proj.modules == []

    def test_missing_modules_key(self, tmp_path: Path) -> None:
        content = yaml.safe_dump({"forge": {"build_command": "./gradlew build"}})
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("MultiApp")
        assert proj.is_multi_target is False

    def test_invalid_module_entries_skipped(self, tmp_path: Path) -> None:
        content = yaml.safe_dump(
            {
                "forge": {
                    "modules": [
                        {"name": "validModule", "device_types": ["phone"]},
                        {"no_name": True},  # Missing name — skip
                        "not a dict",  # Not a dict — skip
                    ],
                }
            }
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("MultiApp")
        assert len(proj.modules) == 1
        assert proj.modules[0].name == "validModule"

    def test_is_multi_target_property(self) -> None:
        proj_no_modules = ForgeProject(name="Simple")
        assert proj_no_modules.is_multi_target is False

        proj_with_modules = ForgeProject(
            name="Multi",
            modules=[ModuleConfig(name="app", device_types=["phone"])],
        )
        assert proj_with_modules.is_multi_target is True


class TestFindModuleApks:
    """Tests for find_module_apks()."""

    def _make_registry_with_modules(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a registry with a multi-module project. Returns (registry_path, proj_dir)."""
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "MultiApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(
            yaml.safe_dump(
                {
                    "forge": {
                        "modules": [
                            {
                                "name": "composeApp",
                                "device_types": ["phone"],
                                "apk_pattern": "composeApp/build/outputs/apk/debug/*.apk",
                            },
                            {
                                "name": "watchApp",
                                "device_types": ["watch"],
                                "apk_pattern": "watchApp/build/outputs/apk/debug/*.apk",
                            },
                        ],
                    }
                }
            )
        )

        data = {"projects": [{"name": "MultiApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))
        return registry_path, proj_dir

    def test_both_modules_found(self, tmp_path: Path) -> None:
        registry, proj_dir = self._make_registry_with_modules(tmp_path)

        # Create fake APKs
        for module in ["composeApp", "watchApp"]:
            apk_dir = proj_dir / module / "build" / "outputs" / "apk" / "debug"
            apk_dir.mkdir(parents=True)
            (apk_dir / f"{module}-debug.apk").write_text("fake")

        reader = ForgeProjectReader(registry_path=registry)
        apks = reader.find_module_apks("MultiApp")
        assert len(apks) == 2
        assert "composeApp" in apks
        assert "watchApp" in apks
        assert apks["composeApp"].endswith(".apk")

    def test_one_module_missing(self, tmp_path: Path) -> None:
        registry, proj_dir = self._make_registry_with_modules(tmp_path)

        # Only create APK for composeApp
        apk_dir = proj_dir / "composeApp" / "build" / "outputs" / "apk" / "debug"
        apk_dir.mkdir(parents=True)
        (apk_dir / "composeApp-debug.apk").write_text("fake")

        reader = ForgeProjectReader(registry_path=registry)
        apks = reader.find_module_apks("MultiApp")
        assert len(apks) == 1
        assert "composeApp" in apks
        assert "watchApp" not in apks

    def test_default_pattern_when_no_apk_pattern(self, tmp_path: Path) -> None:
        """When apk_pattern is empty, use default <module>/build/outputs/apk/debug/*.apk."""
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "DefaultApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(
            yaml.safe_dump(
                {
                    "forge": {
                        "modules": [
                            {"name": "app", "device_types": ["phone"]},
                        ],
                    }
                }
            )
        )

        data = {"projects": [{"name": "DefaultApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))

        # Create APK at default location
        apk_dir = proj_dir / "app" / "build" / "outputs" / "apk" / "debug"
        apk_dir.mkdir(parents=True)
        (apk_dir / "app-debug.apk").write_text("fake")

        reader = ForgeProjectReader(registry_path=registry_path)
        apks = reader.find_module_apks("DefaultApp")
        assert len(apks) == 1
        assert "app" in apks

    def test_no_modules_returns_empty(self, forge_registry: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)
        apks = reader.find_module_apks("PrecisionLevel")
        assert apks == {}

    def test_nonexistent_project_returns_empty(self, forge_registry: Path) -> None:
        reader = ForgeProjectReader(registry_path=forge_registry)
        apks = reader.find_module_apks("NonExistent")
        assert apks == {}


class TestIosBuildCommandParsed:
    """Tests for ios_build_command and app_pattern parsing."""

    def _make_registry(self, tmp_path: Path, forge_yaml_content: str) -> Path:
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "IOSApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(forge_yaml_content)

        data = {"projects": [{"name": "IOSApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))
        return registry_path

    def test_ios_build_command_parsed(self, tmp_path: Path) -> None:
        content = yaml.safe_dump(
            {
                "forge": {
                    "build_command": "./gradlew assembleDebug",
                    "ios_build_command": (
                        "xcodebuild -workspace iosApp/iosApp.xcworkspace "
                        "-scheme iosApp -sdk iphonesimulator build"
                    ),
                }
            }
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("IOSApp")
        assert proj is not None
        assert proj.ios_build_command == (
            "xcodebuild -workspace iosApp/iosApp.xcworkspace"
            " -scheme iosApp -sdk iphonesimulator build"
        )

    def test_ios_build_command_defaults_empty(self, tmp_path: Path) -> None:
        content = yaml.safe_dump({"forge": {"build_command": "./gradlew build"}})
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("IOSApp")
        assert proj.ios_build_command == ""

    def test_ios_watch_scheme_parsed(self, tmp_path: Path) -> None:
        content = yaml.safe_dump(
            {"forge": {"build_command": "./gradlew build", "ios_watch_scheme": "MyAppWatch"}}
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("IOSApp")
        assert proj is not None
        assert proj.ios_watch_scheme == "MyAppWatch"

    def test_ios_watch_scheme_defaults_empty(self, tmp_path: Path) -> None:
        content = yaml.safe_dump({"forge": {"build_command": "./gradlew build"}})
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("IOSApp")
        assert proj.ios_watch_scheme == ""

    def test_module_app_pattern_parsed(self, tmp_path: Path) -> None:
        content = yaml.safe_dump(
            {
                "forge": {
                    "modules": [
                        {
                            "name": "iosApp",
                            "device_types": ["phone"],
                            "app_pattern": (
                                "iosApp/build/Build/Products/Debug-iphonesimulator/*.app"
                            ),
                        },
                    ],
                }
            }
        )
        registry = self._make_registry(tmp_path, content)
        reader = ForgeProjectReader(registry_path=registry)
        proj = reader.get("IOSApp")
        assert len(proj.modules) == 1
        assert proj.modules[0].app_pattern == (
            "iosApp/build/Build/Products/Debug-iphonesimulator/*.app"
        )


class TestFindModuleIosApps:
    """Tests for find_module_ios_apps()."""

    def _make_registry(self, tmp_path: Path) -> tuple[Path, Path]:
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "KMPApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(
            yaml.safe_dump(
                {
                    "forge": {
                        "modules": [
                            {
                                "name": "iosApp",
                                "device_types": ["phone"],
                                "app_pattern": (
                                    "iosApp/build/Build/Products/Debug-iphonesimulator/*.app"
                                ),
                            },
                            {
                                "name": "composeApp",
                                "device_types": ["phone"],
                                "apk_pattern": "composeApp/build/outputs/apk/debug/*.apk",
                            },
                        ],
                    }
                }
            )
        )

        data = {"projects": [{"name": "KMPApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))
        return registry_path, proj_dir

    def test_find_module_ios_apps(self, tmp_path: Path) -> None:
        registry, proj_dir = self._make_registry(tmp_path)

        # Create a fake .app bundle
        app_dir = proj_dir / "iosApp" / "build" / "Build" / "Products" / "Debug-iphonesimulator"
        app_dir.mkdir(parents=True)
        (app_dir / "KMPApp.app").mkdir()

        reader = ForgeProjectReader(registry_path=registry)
        apps = reader.find_module_ios_apps("KMPApp")
        assert "iosApp" in apps
        assert apps["iosApp"].endswith(".app")

    def test_find_module_ios_apps_missing(self, tmp_path: Path) -> None:
        registry, _proj_dir = self._make_registry(tmp_path)
        reader = ForgeProjectReader(registry_path=registry)
        apps = reader.find_module_ios_apps("KMPApp")
        assert apps == {}

    def test_find_module_ios_apps_nonexistent_project(self, tmp_path: Path) -> None:
        registry, _proj_dir = self._make_registry(tmp_path)
        reader = ForgeProjectReader(registry_path=registry)
        apps = reader.find_module_ios_apps("NonExistent")
        assert apps == {}


class TestWatchModuleIosApps:
    """Tests for watchOS module .app discovery."""

    def _make_registry(self, tmp_path: Path, device_types: list[str]) -> tuple[Path, Path]:
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "WatchApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(
            yaml.safe_dump(
                {
                    "forge": {
                        "modules": [
                            {
                                "name": "watchModule",
                                "device_types": device_types,
                            },
                        ],
                    }
                }
            )
        )

        data = {"projects": [{"name": "WatchApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))
        return registry_path, proj_dir

    def test_find_module_ios_apps_watch_default_path(self, tmp_path: Path) -> None:
        """Module with device_types=['watch'] searches Debug-watchsimulator."""
        registry, proj_dir = self._make_registry(tmp_path, ["watch"])

        products = proj_dir / "watchModule" / "build" / "Build" / "Products"
        app_dir = products / "Debug-watchsimulator"
        app_dir.mkdir(parents=True)
        (app_dir / "WatchApp.app").mkdir()

        reader = ForgeProjectReader(registry_path=registry)
        apps = reader.find_module_ios_apps("WatchApp")
        assert "watchModule" in apps
        assert "Debug-watchsimulator" in apps["watchModule"]

    def test_find_module_ios_apps_phone_unchanged(self, tmp_path: Path) -> None:
        """Module with device_types=['phone'] still searches Debug-iphonesimulator."""
        registry, proj_dir = self._make_registry(tmp_path, ["phone"])

        products = proj_dir / "watchModule" / "build" / "Build" / "Products"
        app_dir = products / "Debug-iphonesimulator"
        app_dir.mkdir(parents=True)
        (app_dir / "PhoneApp.app").mkdir()

        reader = ForgeProjectReader(registry_path=registry)
        apps = reader.find_module_ios_apps("WatchApp")
        assert "watchModule" in apps
        assert "Debug-iphonesimulator" in apps["watchModule"]

    def test_find_ios_app_watch_device_type(self, tmp_path: Path) -> None:
        """find_ios_app with device_type='watch' searches Debug-watchsimulator."""
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "MyApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(yaml.safe_dump({"build_command": "make"}))

        app_dir = proj_dir / "build" / "Build" / "Products" / "Debug-watchsimulator"
        app_dir.mkdir(parents=True)
        (app_dir / "MyWatch.app").mkdir()

        data = {"projects": [{"name": "MyApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))

        reader = ForgeProjectReader(registry_path=registry_path)
        result = reader.find_ios_app("MyApp", device_type="watch")
        assert result is not None
        assert "Debug-watchsimulator" in result

    def test_find_ios_app_phone_device_type(self, tmp_path: Path) -> None:
        """find_ios_app with default device_type searches Debug-iphonesimulator."""
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "MyApp"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(yaml.safe_dump({"build_command": "make"}))

        app_dir = proj_dir / "build" / "Build" / "Products" / "Debug-iphonesimulator"
        app_dir.mkdir(parents=True)
        (app_dir / "MyPhone.app").mkdir()

        data = {"projects": [{"name": "MyApp", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))

        reader = ForgeProjectReader(registry_path=registry_path)
        result = reader.find_ios_app("MyApp")
        assert result is not None
        assert "Debug-iphonesimulator" in result


class TestModuleConfigIosBuildCommand:
    """Tests for ios_build_command on ModuleConfig."""

    def test_module_config_ios_build_command_parsed(self, tmp_path: Path) -> None:
        """ios_build_command in .forge.yaml is parsed into ModuleConfig."""
        registry_path = tmp_path / ".forge" / "projects.yaml"
        registry_path.parent.mkdir(parents=True)

        proj_dir = tmp_path / "projects" / "App"
        proj_dir.mkdir(parents=True)
        (proj_dir / ".forge.yaml").write_text(
            yaml.safe_dump(
                {
                    "forge": {
                        "modules": [
                            {
                                "name": "watchApp",
                                "device_types": ["watch"],
                                "ios_build_command": (
                                    "xcodebuild -scheme WatchScheme -sdk watchsimulator"
                                ),
                            },
                        ],
                    }
                }
            )
        )

        data = {"projects": [{"name": "App", "path": str(proj_dir)}]}
        registry_path.write_text(yaml.safe_dump(data))

        reader = ForgeProjectReader(registry_path=registry_path)
        project = reader.get("App")
        assert project is not None
        assert len(project.modules) == 1
        expected = "xcodebuild -scheme WatchScheme -sdk watchsimulator"
        assert project.modules[0].ios_build_command == expected

    def test_module_config_ios_build_command_defaults_empty(self) -> None:
        """ModuleConfig.ios_build_command defaults to empty string."""
        mc = ModuleConfig(name="app")
        assert mc.ios_build_command == ""


# --- Manifest discovery tests ---


class TestManifestDiscovery:
    """ForgeProjectReader discovers projects from linc-manifest.yaml files."""

    @staticmethod
    def _write_manifest(project_dir: Path, manifest: dict) -> None:
        """Write a linc-manifest.yaml inside a project directory."""
        project_dir.mkdir(parents=True, exist_ok=True)
        with open(project_dir / "linc-manifest.yaml", "w") as f:
            yaml.safe_dump(manifest, f)

    @staticmethod
    def _dawncup_manifest() -> dict:
        return {
            "schema_version": 1,
            "identity": {
                "name": "DawnCup",
                "package": "com.lincmobile.dawncup",
                "display_name": "Dawn Cup",
                "app_type": "watch_only",
            },
        }

    @staticmethod
    def _precisionlevel_manifest() -> dict:
        return {
            "schema_version": 1,
            "identity": {
                "name": "PrecisionLevel",
                "package": "com.lincmobile.level",
                "display_name": "Precision Level",
                "app_type": "phone_and_watch",
            },
        }

    def test_discover_from_manifests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-discover projects from linc-manifest.yaml when no registry_path given."""
        projects_dir = tmp_path / "projects"
        self._write_manifest(projects_dir / "DawnCup", self._dawncup_manifest())
        self._write_manifest(projects_dir / "PrecisionLevel", self._precisionlevel_manifest())

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        # Redirect default registry to nonexistent path for isolation
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        assert reader.get("DawnCup") is not None
        assert reader.get("PrecisionLevel") is not None
        assert len(reader.list_projects()) == 2

    def test_manifest_sets_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Discovered project path is the directory containing the manifest."""
        projects_dir = tmp_path / "projects"
        dc_dir = projects_dir / "DawnCup"
        self._write_manifest(dc_dir, self._dawncup_manifest())

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        proj = reader.get("DawnCup")
        assert proj is not None
        assert proj.path == str(dc_dir)

    def test_manifest_merges_forge_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After manifest discovery, .forge.yaml fields are still merged."""
        projects_dir = tmp_path / "projects"
        dc_dir = projects_dir / "DawnCup"
        self._write_manifest(dc_dir, self._dawncup_manifest())
        (dc_dir / ".forge.yaml").write_text(
            yaml.safe_dump(
                {
                    "forge": {"test_command": "pytest", "framework": "gradle_kmp"},
                }
            )
        )

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        proj = reader.get("DawnCup")
        assert proj is not None
        assert proj.test_command == "pytest"
        assert proj.framework == "gradle_kmp"

    def test_no_discovery_when_registry_path_given(self, tmp_path: Path) -> None:
        """When registry_path is explicitly provided, no manifest discovery happens."""
        projects_dir = tmp_path / "projects"
        self._write_manifest(projects_dir / "DawnCup", self._dawncup_manifest())

        # Empty registry — should NOT find DawnCup via manifest
        registry_path = tmp_path / "empty.yaml"
        registry_path.write_text("")
        reader = ForgeProjectReader(registry_path=registry_path)
        assert reader.get("DawnCup") is None
        assert reader.list_projects() == []

    def test_manifest_skips_invalid_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid manifest files are silently skipped."""
        projects_dir = tmp_path / "projects"
        bad_dir = projects_dir / "BadApp"
        bad_dir.mkdir(parents=True)
        (bad_dir / "linc-manifest.yaml").write_text("invalid: yaml: {{")

        self._write_manifest(projects_dir / "DawnCup", self._dawncup_manifest())

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        assert reader.get("DawnCup") is not None
        assert reader.get("BadApp") is None

    def test_manifest_skips_missing_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Manifest without identity.name is skipped."""
        projects_dir = tmp_path / "projects"
        no_name_dir = projects_dir / "NoName"
        no_name_dir.mkdir(parents=True)
        (no_name_dir / "linc-manifest.yaml").write_text(
            yaml.safe_dump({"schema_version": 1, "identity": {"package": "com.test"}})
        )

        self._write_manifest(projects_dir / "DawnCup", self._dawncup_manifest())

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        assert reader.get("DawnCup") is not None
        assert reader.get("NoName") is None

    def test_manifest_wins_over_projects_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Manifest updates path even if project was already in projects.yaml."""
        projects_dir = tmp_path / "projects"
        new_dir = projects_dir / "DawnCup"
        self._write_manifest(new_dir, self._dawncup_manifest())

        # Create legacy registry with old path
        old_dir = tmp_path / "old" / "DawnCup"
        old_dir.mkdir(parents=True)
        legacy_registry = tmp_path / "legacy.yaml"
        legacy_registry.write_text(
            yaml.safe_dump({"projects": [{"name": "DawnCup", "path": str(old_dir)}]})
        )

        import linc_crucible.core.forge_integration as mod

        monkeypatch.setattr(mod, "_default_forge_registry", lambda: legacy_registry)
        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))

        reader = ForgeProjectReader()
        proj = reader.get("DawnCup")
        assert proj is not None
        assert proj.path == str(new_dir)

    def test_works_without_projects_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Works with only manifests, no projects.yaml at all."""
        projects_dir = tmp_path / "projects"
        self._write_manifest(projects_dir / "DawnCup", self._dawncup_manifest())
        self._write_manifest(projects_dir / "PrecisionLevel", self._precisionlevel_manifest())

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        assert len(reader.list_projects()) == 2
        assert reader.get("DawnCup") is not None
        assert reader.get("PrecisionLevel") is not None

    @staticmethod
    def _precisionlevel_manifest_with_modules() -> dict:
        """PrecisionLevel manifest shape with a flavored `app` module + `watchApp`."""
        return {
            "schema_version": 1,
            "identity": {
                "name": "PrecisionLevel",
                "package": "com.lincmobile.level",
                "display_name": "Precision Level",
                "app_type": "phone_and_watch",
            },
            "deployment": {
                "modules": [
                    {
                        "name": "app",
                        "device_types": ["phone", "tablet", "emulator"],
                        "build_command": "./gradlew :app:assemblePlaystoreDebug",
                        "apk_pattern": "app/build/outputs/apk/playstore/debug/*.apk",
                    },
                    {
                        "name": "watchApp",
                        "device_types": ["watch"],
                        "build_command": "./gradlew :watchApp:assembleDebug",
                        "apk_pattern": "watchApp/build/outputs/apk/debug/*.apk",
                    },
                ],
            },
        }

    def test_discover_modules_from_manifest_deployment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`deployment.modules` in linc-manifest.yaml populates ForgeProject.modules.

        Regression for the pour routing bug: modules were only ever parsed
        from .forge.yaml, so a manifest-only project (no .forge.yaml modules
        block) never became multi-target and --module was silently ignored.
        """
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "PrecisionLevel"
        self._write_manifest(proj_dir, self._precisionlevel_manifest_with_modules())

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        app_apk_dir = proj_dir / "app" / "build" / "outputs" / "apk" / "playstore" / "debug"
        app_apk_dir.mkdir(parents=True)
        (app_apk_dir / "app-playstore-debug.apk").write_text("fake")

        reader = ForgeProjectReader()
        proj = reader.get("PrecisionLevel")
        assert proj is not None
        assert proj.is_multi_target is True

        apks = reader.find_module_apks("PrecisionLevel")
        assert apks["app"].endswith(
            "app/build/outputs/apk/playstore/debug/app-playstore-debug.apk"
        )

    def test_manifest_modules_applied_when_project_already_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A project already loaded from projects.yaml still picks up
        deployment.modules from its manifest — the "manifest wins" update
        path (``name in self._projects``), not just the fresh-discovery path.
        """
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "PrecisionLevel"
        self._write_manifest(proj_dir, self._precisionlevel_manifest_with_modules())

        old_dir = tmp_path / "old" / "PrecisionLevel"
        old_dir.mkdir(parents=True)
        legacy_registry = tmp_path / "legacy.yaml"
        legacy_registry.write_text(
            yaml.safe_dump({"projects": [{"name": "PrecisionLevel", "path": str(old_dir)}]})
        )

        import linc_crucible.core.forge_integration as mod

        monkeypatch.setattr(mod, "_default_forge_registry", lambda: legacy_registry)
        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))

        reader = ForgeProjectReader()
        proj = reader.get("PrecisionLevel")
        assert proj is not None
        assert proj.path == str(proj_dir)
        assert proj.is_multi_target is True
        assert {m.name for m in proj.modules} == {"app", "watchApp"}

    def test_manifest_deployment_not_dict_yields_no_modules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed (non-dict) ``deployment`` block is tolerated — the
        project still discovers, just without manifest-declared modules.
        """
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "DawnCup"
        manifest = self._dawncup_manifest()
        manifest["deployment"] = "not-a-dict"
        self._write_manifest(proj_dir, manifest)

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        proj = reader.get("DawnCup")
        assert proj is not None
        assert proj.modules == []
        assert proj.is_multi_target is False

    def test_manifest_reads_identity_package_and_bundle_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """identity.package and identity.bundle_id in linc-manifest.yaml populate ForgeProject."""
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "MyApp"
        self._write_manifest(
            proj_dir,
            {
                "schema_version": 1,
                "identity": {
                    "name": "MyApp",
                    "package": "com.example.myapp",
                    "bundle_id": "com.example.myapp.ios",
                },
            },
        )

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        reader = ForgeProjectReader()
        proj = reader.get("MyApp")
        assert proj is not None
        assert proj.package == "com.example.myapp"
        assert proj.bundle_id == "com.example.myapp.ios"

    def test_manifest_updates_identity_package_and_bundle_id_when_already_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A project already in projects.yaml picks up identity.package and identity.bundle_id."""
        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "MyApp"
        self._write_manifest(
            proj_dir,
            {
                "schema_version": 1,
                "identity": {
                    "name": "MyApp",
                    "package": "com.example.myapp",
                    "bundle_id": "com.example.myapp.ios",
                },
            },
        )
        old_dir = tmp_path / "old" / "MyApp"
        old_dir.mkdir(parents=True)
        legacy_registry = tmp_path / "legacy.yaml"
        legacy_registry.write_text(
            yaml.safe_dump({"projects": [{"name": "MyApp", "path": str(old_dir)}]})
        )

        import linc_crucible.core.forge_integration as mod

        monkeypatch.setattr(mod, "_default_forge_registry", lambda: legacy_registry)
        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))

        reader = ForgeProjectReader()
        proj = reader.get("MyApp")
        assert proj is not None
        assert proj.package == "com.example.myapp"
        assert proj.bundle_id == "com.example.myapp.ios"

    @pytest.mark.parametrize(
        "bad_val,expected_type",
        [
            (123, "int"),
            (["com.example.app"], "list"),
            ({"id": "com.example.app"}, "dict"),
            (True, "bool"),
        ],
    )
    def test_manifest_ignores_non_string_identity_package_and_bundle_id_with_warning(
        self,
        bad_val: object,
        expected_type: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-string identity.package / identity.bundle_id are ignored with a warning."""
        import structlog.testing

        projects_dir = tmp_path / "projects"
        proj_dir = projects_dir / "MyApp"
        self._write_manifest(
            proj_dir,
            {
                "schema_version": 1,
                "identity": {
                    "name": "MyApp",
                    "package": bad_val,
                    "bundle_id": bad_val,
                },
            },
        )

        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)

        with structlog.testing.capture_logs() as cap_logs:
            reader = ForgeProjectReader()

        proj = reader.get("MyApp")
        assert proj is not None
        assert proj.package == ""
        assert proj.bundle_id == ""
        pkg_logs = [
            e
            for e in cap_logs
            if e.get("event") == "forge_manifest_identity_not_string"
            and e.get("field") == "package"
        ]
        bundle_logs = [
            e
            for e in cap_logs
            if e.get("event") == "forge_manifest_identity_not_string"
            and e.get("field") == "bundle_id"
        ]
        assert len(pkg_logs) == 1
        assert pkg_logs[0].get("type") == expected_type
        assert len(bundle_logs) == 1
        assert bundle_logs[0].get("type") == expected_type


class TestManifestDeploymentModulesHelpers:
    """Direct unit coverage for the extracted module-parsing helpers."""

    def test_manifest_deployment_modules_missing_deployment(self) -> None:
        """No ``deployment`` key at all -> empty list, not an error."""
        assert _manifest_deployment_modules({"identity": {"name": "X"}}) == []

    def test_manifest_deployment_modules_non_dict_deployment(self) -> None:
        """``deployment`` present but not a dict -> empty list."""
        assert _manifest_deployment_modules({"deployment": ["oops"]}) == []

    def test_manifest_deployment_modules_parses_entries(self) -> None:
        raw = {"deployment": {"modules": [{"name": "app"}, {"name": "watchApp"}]}}
        modules = _manifest_deployment_modules(raw)
        assert [m.name for m in modules] == ["app", "watchApp"]

    def test_apply_manifest_modules_overwrites_when_present(self) -> None:
        project = ForgeProject(name="X")
        project.modules = [ModuleConfig(name="stale")]
        fresh = [ModuleConfig(name="app")]
        _apply_manifest_modules(project, fresh)
        assert project.modules == fresh

    def test_apply_manifest_modules_leaves_existing_when_empty(self) -> None:
        """An empty manifest_modules list (manifest declared none) doesn't
        clobber modules .forge.yaml already set.
        """
        project = ForgeProject(name="X")
        existing = [ModuleConfig(name="from-forge-yaml")]
        project.modules = existing
        _apply_manifest_modules(project, [])
        assert project.modules == existing


def test_find_ios_app_watchos(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    reader = ForgeProjectReader()
    project_path = tmp_path / "TestProject"

    # Setup nested Watch App.app structure in Derived Data path
    debug_path = project_path / "build" / "derived" / "Build" / "Products" / "Debug-watchsimulator"
    nested_app = debug_path / "Main.app" / "Watch App.app"
    nested_app.mkdir(parents=True)

    # Mock getting the project
    reader._projects["test"] = MagicMock(path=str(project_path), name="TestProject")
    # Setting name attribute on mock specifically
    reader._projects["test"].name = "TestProject"

    app_path = reader.find_ios_app("test", device_type="watch", is_simulator=True)
    assert app_path == str(nested_app)


class TestFindIosAppCoverage:
    """Characterization tests for find_ios_app — covers CRAP-critical branches."""

    def test_find_ios_app_no_project(
        self, forge_registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown project name returns None immediately."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: forge_registry.parent))
        reader = ForgeProjectReader(registry_path=forge_registry)
        assert reader.find_ios_app("TotallyUnknown") is None

    def test_find_ios_app_deterministic_path(
        self, forge_registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finds .app in build/derived/Build/Products/Debug-iphonesimulator/."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))
        reader = ForgeProjectReader(registry_path=forge_registry)

        proj_dir = tmp_path / "projects" / "PrecisionLevel"
        app_dir = proj_dir / "build" / "derived" / "Build" / "Products" / "Debug-iphonesimulator"
        app_dir.mkdir(parents=True)
        app_bundle = app_dir / "PrecisionLevel.app"
        app_bundle.mkdir()

        result = reader.find_ios_app("PrecisionLevel")
        assert result is not None
        assert result.endswith(".app")
        assert "Debug-iphonesimulator" in result
        assert "build/derived" in result

    def test_find_ios_app_watch_nested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """device_type='watch' returns nested 'Watch App.app' when present."""
        from unittest.mock import MagicMock

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))
        reader = ForgeProjectReader()
        project_path = tmp_path / "WatchProject"
        debug_path = (
            project_path / "build" / "derived" / "Build" / "Products" / "Debug-watchsimulator"
        )
        outer_app = debug_path / "Main.app"
        nested_app = outer_app / "Watch App.app"
        nested_app.mkdir(parents=True)

        reader._projects["WatchProj"] = MagicMock(path=str(project_path), name="WatchProj")
        reader._projects["WatchProj"].name = "WatchProj"

        result = reader.find_ios_app("WatchProj", device_type="watch", is_simulator=True)
        assert result is not None
        assert result.endswith("Watch App.app")

    def test_find_ios_app_physical_device(
        self, forge_registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_simulator=False uses Debug-iphoneos SDK directory."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))
        reader = ForgeProjectReader(registry_path=forge_registry)

        proj_dir = tmp_path / "projects" / "PrecisionLevel"
        app_dir = proj_dir / "build" / "derived" / "Build" / "Products" / "Debug-iphoneos"
        app_dir.mkdir(parents=True)
        (app_dir / "PrecisionLevel.app").mkdir()

        result = reader.find_ios_app("PrecisionLevel", is_simulator=False)
        assert result is not None
        assert "Debug-iphoneos" in result

    def test_find_ios_app_fallback_derived_data(
        self, forge_registry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to ~/Library/Developer/Xcode/DerivedData when no deterministic match."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))
        reader = ForgeProjectReader(registry_path=forge_registry)

        # Create DerivedData with matching project prefix
        derived_data = tmp_path / "fakehome" / "Library" / "Developer" / "Xcode" / "DerivedData"
        dd_project = derived_data / "precisionlevel-abcdef"
        app_dir = dd_project / "Build" / "Products" / "Debug-iphonesimulator"
        app_dir.mkdir(parents=True)
        (app_dir / "PrecisionLevel.app").mkdir()

        result = reader.find_ios_app("PrecisionLevel")
        assert result is not None
        assert "DerivedData" in result
        assert result.endswith(".app")


# --- Manifest duplicate/worktree resolution + lookup re-scan (issue #113) ---


class TestManifestDuplicateResolutionAndRescan:
    """Worktree-poisoning fix: duplicate manifest names resolve deterministically,
    and `get()` self-heals on a lookup miss or a stale (deleted) cached path.
    """

    @staticmethod
    def _write_manifest(project_dir: Path, name: str) -> None:
        project_dir.mkdir(parents=True, exist_ok=True)
        with open(project_dir / "linc-manifest.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "identity": {"name": name, "package": "com.test", "app_type": "phone"},
                },
                f,
            )

    def _reader_with_projects_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, projects_dir: Path
    ) -> ForgeProjectReader:
        monkeypatch.setenv("LINC_PROJECTS_DIR", str(projects_dir))
        import linc_crucible.core.forge_integration as mod

        nonexistent = tmp_path / "nonexistent.yaml"
        monkeypatch.setattr(mod, "_default_forge_registry", lambda: nonexistent)
        return ForgeProjectReader()

    def test_worktree_duplicate_prefers_shorter_canonical_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two manifest dirs declare the same identity.name; the worktree copy
        (longer, suffixed dir name) must not silently overwrite the canonical one.
        """
        projects_dir = tmp_path / "projects"
        canonical_dir = projects_dir / "SniperPulseApp"
        worktree_dir = projects_dir / "SniperPulseApp-v033"
        self._write_manifest(canonical_dir, "SniperPulse")
        self._write_manifest(worktree_dir, "SniperPulse")

        reader = self._reader_with_projects_dir(tmp_path, monkeypatch, projects_dir)

        proj = reader.get("SniperPulse")
        assert proj is not None
        assert proj.path == str(canonical_dir)

    def test_stale_path_recovery_on_deleted_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the resolved dir is deleted entirely, get() re-scans and
        returns the surviving duplicate's (still-valid) path.
        """
        projects_dir = tmp_path / "projects"
        canonical_dir = projects_dir / "SniperPulseApp"
        worktree_dir = projects_dir / "SniperPulseApp-v033"
        self._write_manifest(canonical_dir, "SniperPulse")
        self._write_manifest(worktree_dir, "SniperPulse")

        reader = self._reader_with_projects_dir(tmp_path, monkeypatch, projects_dir)

        proj = reader.get("SniperPulse")
        assert proj is not None
        assert proj.path == str(canonical_dir)

        # Delete the resolved (canonical) directory out from under the reader.
        shutil.rmtree(canonical_dir)

        proj2 = reader.get("SniperPulse")
        assert proj2 is not None
        assert Path(proj2.path).is_dir()
        assert proj2.path == str(worktree_dir)

    def test_lookup_miss_triggers_rescan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest created AFTER the reader was constructed is still found —
        get() re-scans on a miss instead of returning None for the process lifetime.
        """
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir(parents=True)

        reader = self._reader_with_projects_dir(tmp_path, monkeypatch, projects_dir)

        assert reader.get("LateApp") is None

        self._write_manifest(projects_dir / "LateApp", "LateApp")

        proj = reader.get("LateApp")
        assert proj is not None
        assert proj.name == "LateApp"

    def test_read_identity_str_missing_and_non_string(self) -> None:
        from linc_crucible.core.forge_integration import _read_identity_str

        assert _read_identity_str({}, "package") == ""
        assert _read_identity_str({"package": None}, "package") == ""
        assert _read_identity_str({"package": 123}, "package") == ""
        assert _read_identity_str({"package": "com.example.app"}, "package") == "com.example.app"
