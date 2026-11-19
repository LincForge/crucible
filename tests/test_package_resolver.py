import textwrap

from linc_crucible.core.package_resolver import (
    parse_base_application_id,
    parse_build_type_suffix,
    source_parse_package,
)

GETBYNAME = textwrap.dedent("""
    android {
        defaultConfig { applicationId = "com.sniperpulse.app" }
        buildTypes {
            getByName("debug") { applicationIdSuffix = ".dev" }
            getByName("release") { isMinifyEnabled = true }
        }
    }
""")
BRACE = textwrap.dedent("""
    android {
        defaultConfig { applicationId = "com.x.app" }
        buildTypes { debug { applicationIdSuffix = ".dbg" } }
    }
""")
NAMED = textwrap.dedent("""
    android {
        defaultConfig { applicationId = "com.y.app" }
        buildTypes { named("debug") { applicationIdSuffix = ".d" } }
    }
""")
NOSUFFIX = 'android { defaultConfig { applicationId = "com.z.app" } }'


def test_parse_base():
    assert parse_base_application_id(GETBYNAME) == "com.sniperpulse.app"


def test_parse_base_absent():
    assert parse_base_application_id('rootProject.name = "x"') is None


def test_suffix_getbyname():
    assert parse_build_type_suffix(GETBYNAME, "debug") == ".dev"


def test_suffix_brace():
    assert parse_build_type_suffix(BRACE, "debug") == ".dbg"


def test_suffix_named():
    assert parse_build_type_suffix(NAMED, "debug") == ".d"


def test_suffix_absent():
    assert parse_build_type_suffix(NOSUFFIX, "debug") is None


def test_suffix_release_block_ignored():
    assert parse_build_type_suffix(GETBYNAME, "release") is None


def test_source_parse_combines(tmp_path):
    mod = tmp_path / "composeApp"
    mod.mkdir()
    (mod / "build.gradle.kts").write_text(GETBYNAME)
    assert source_parse_package(str(tmp_path), "debug") == "com.sniperpulse.app.dev"


def test_source_parse_base_when_no_suffix(tmp_path):
    mod = tmp_path / "app"
    mod.mkdir()
    (mod / "build.gradle.kts").write_text(NOSUFFIX)
    assert source_parse_package(str(tmp_path), "debug") == "com.z.app"


def test_source_parse_none_when_no_appid(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "build.gradle.kts").write_text("// nothing")
    assert source_parse_package(str(tmp_path), "debug") is None


# --- Tier 1: aapt badging + discovery, and layered resolve ---

import subprocess  # noqa: E402

from linc_crucible.core.package_resolver import (  # noqa: E402
    badge_package,
    find_aapt,
    resolve_effective_package,
)

BADGING = "package: name='com.sniperpulse.app.dev' versionCode='13' versionName='0.3.4'\n"


def test_find_aapt_via_which(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda t: "/usr/bin/aapt2" if t == "aapt2" else None)
    assert find_aapt() == "/usr/bin/aapt2"


def test_find_aapt_via_android_home(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda t: None)
    bt = tmp_path / "build-tools" / "34.0.0"
    bt.mkdir(parents=True)
    (bt / "aapt2").write_text("")
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))
    assert find_aapt() == str(bt / "aapt2")


def test_find_aapt_none(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda t: None)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    assert find_aapt() is None


def test_contract_find_aapt_picks_highest_version_tuple_over_string_order(monkeypatch, tmp_path):
    """[contract] build-tools 9.0.0 and 35.0.0 present -> 35.0.0 chosen."""
    monkeypatch.setattr("shutil.which", lambda _t: None)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))

    for ver in ("9.0.0", "35.0.0"):
        bt = tmp_path / "build-tools" / ver
        bt.mkdir(parents=True)
        (bt / "aapt2").write_text("", encoding="utf-8")

    assert find_aapt() == str(tmp_path / "build-tools" / "35.0.0" / "aapt2")


def test_find_aapt_version_sorting_and_hostile_dirs(monkeypatch, tmp_path):
    """Handles pre-releases, non-numeric dirs, directory binaries, and aapt2 > aapt."""
    monkeypatch.setattr("shutil.which", lambda _t: None)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))

    # Non-numeric dir alone still resolves without crashing
    (tmp_path / "build-tools" / "custom").mkdir(parents=True)
    (tmp_path / "build-tools" / "custom" / "aapt2").write_text("", encoding="utf-8")
    assert find_aapt() == str(tmp_path / "build-tools" / "custom" / "aapt2")

    # Directory named aapt2 in 99.0.0 is ignored; 35.0.0 beats 35.0.0-rc1 and 9.0.0
    (tmp_path / "build-tools" / "99.0.0" / "aapt2").mkdir(parents=True)
    for ver in ("9.0.0", "35.0.0-rc1"):
        bt = tmp_path / "build-tools" / ver
        bt.mkdir(parents=True)
        (bt / "aapt2").write_text("", encoding="utf-8")
    (tmp_path / "build-tools" / "35.0.0").mkdir(parents=True)
    (tmp_path / "build-tools" / "35.0.0" / "aapt").write_text("", encoding="utf-8")
    assert find_aapt() == str(tmp_path / "build-tools" / "35.0.0" / "aapt")

    # Within the same version, aapt2 beats aapt
    (tmp_path / "build-tools" / "35.0.0" / "aapt2").write_text("", encoding="utf-8")
    assert find_aapt() == str(tmp_path / "build-tools" / "35.0.0" / "aapt2")


def test_badge_package_parses_name(monkeypatch):
    monkeypatch.setattr("linc_crucible.core.package_resolver.find_aapt", lambda: "/x/aapt")
    cp = subprocess.CompletedProcess(args=[], returncode=0, stdout=BADGING, stderr="")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: cp)
    assert badge_package("/tmp/app.apk") == "com.sniperpulse.app.dev"


def test_badge_package_no_aapt(monkeypatch):
    monkeypatch.setattr("linc_crucible.core.package_resolver.find_aapt", lambda: None)
    assert badge_package("/tmp/app.apk") is None


def test_badge_package_nonzero_exit(monkeypatch):
    monkeypatch.setattr("linc_crucible.core.package_resolver.find_aapt", lambda: "/x/aapt")
    cp = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: cp)
    assert badge_package("/tmp/app.apk") is None


def test_resolve_prefers_badging(monkeypatch, tmp_path):
    mod = tmp_path / "composeApp"
    mod.mkdir()
    (mod / "build.gradle.kts").write_text(GETBYNAME)
    monkeypatch.setattr(
        "linc_crucible.core.package_resolver.badge_package",
        lambda apk: "com.from.badging",
    )
    assert (
        resolve_effective_package(str(tmp_path), "debug", apk_path="/tmp/a.apk")
        == "com.from.badging"
    )


def test_resolve_falls_back_to_source(monkeypatch, tmp_path):
    mod = tmp_path / "composeApp"
    mod.mkdir()
    (mod / "build.gradle.kts").write_text(GETBYNAME)
    monkeypatch.setattr("linc_crucible.core.package_resolver.badge_package", lambda apk: None)
    assert (
        resolve_effective_package(str(tmp_path), "debug", apk_path="/tmp/a.apk")
        == "com.sniperpulse.app.dev"
    )


def test_resolve_source_only_when_no_apk(tmp_path):
    mod = tmp_path / "app"
    mod.mkdir()
    (mod / "build.gradle.kts").write_text(NOSUFFIX)
    assert resolve_effective_package(str(tmp_path), "debug") == "com.z.app"


def test_resolve_manifest_package_beats_source_and_loses_to_badging(monkeypatch, tmp_path):
    mod = tmp_path / "composeApp"
    mod.mkdir()
    (mod / "build.gradle.kts").write_text(GETBYNAME)

    monkeypatch.setattr("linc_crucible.core.package_resolver.badge_package", lambda apk: None)
    assert (
        resolve_effective_package(
            str(tmp_path),
            "debug",
            apk_path="/tmp/a.apk",
            manifest_package="com.from.manifest",
        )
        == "com.from.manifest"
    )
    assert (
        resolve_effective_package(
            str(tmp_path),
            "debug",
            manifest_package="com.from.manifest",
        )
        == "com.from.manifest"
    )

    monkeypatch.setattr(
        "linc_crucible.core.package_resolver.badge_package",
        lambda apk: "com.from.badging",
    )
    assert (
        resolve_effective_package(
            str(tmp_path),
            "debug",
            apk_path="/tmp/a.apk",
            manifest_package="com.from.manifest",
        )
        == "com.from.badging"
    )


def test_resolve_returns_none_when_no_tier_matches(monkeypatch, tmp_path):
    monkeypatch.setattr("linc_crucible.core.package_resolver.badge_package", lambda apk: None)
    assert resolve_effective_package("", "debug") is None
    assert resolve_effective_package(str(tmp_path), "debug") is None


def test_resolve_passes_build_type_and_defaults_manifest_package(monkeypatch, tmp_path):
    seen = []

    def fake_source_parse(path: str, build_type: str = "debug") -> str | None:
        seen.append((path, build_type))
        return "com.example.release" if build_type == "release" else "com.example.debug"

    monkeypatch.setattr(
        "linc_crucible.core.package_resolver.source_parse_package", fake_source_parse
    )
    assert resolve_effective_package(str(tmp_path), "release") == "com.example.release"
    assert seen == [(str(tmp_path), "release")]


def test_contract_build_tools_version_key_exact_tuples() -> None:
    """[contract] exact sort keys: unparseable names sort below every version, a release
    above its own release candidate, and versions compare numerically."""
    from linc_crucible.core.package_resolver import _build_tools_version_key as key

    assert key("abc") == (0, (), 0, "abc")
    assert key("35.0.0") == (1, (35, 0, 0), 1, "")
    assert key("35.0.0-rc1") == (1, (35, 0, 0), 0, "-rc1")
    assert key("9.0.0") < key("35.0.0-rc1") < key("35.0.0")
    assert key("abc") < key("1")
