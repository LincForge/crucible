"""[contract] Every env var src/ reads is documented in ONE pinned table (REVIEW B4b).

``linc_crucible.config_contract.ENV_VARS`` is the single source of truth; this module pins it
against the code and against the README "Configuration" env table rows.

The code side is an AST walk, not a grep: every ``os.environ.get`` / ``os.getenv`` /
``environ.get`` / ``getenv`` call, ``map(<getter>, names)`` and ``environ[...]`` subscript is
resolved through literal and module-level string (tuple) constants. Any other reference to
``os.environ`` / ``os.getenv`` / ``environ`` / ``getenv`` (an alias, an aliased or star import,
a non-call use, a computed name) raises, so a read the walk cannot analyse turns the pin red
instead of shipping undocumented.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import structlog.testing

from linc_crucible.config_contract import (
    ENV_VARS,
    EnvVar,
    env_var_names,
    render_env_table,
)
from tests.env_read_scan import UnanalysableEnvReadError
from tests.env_read_scan import names_read_under as _names_read_under

REPO = Path(__file__).resolve().parents[1]

# These two scan the source tree beside tests/. A wheel-only test root (linc-publish's
# selfcheck installs the built wheel and copies only tests/) has no src/ to scan, so they
# skip there; every checkout (private CI, mutmut's mutants/, the public repo) runs them.
needs_src_tree = pytest.mark.skipif(
    not (REPO / "src").is_dir(), reason="no src/ tree beside tests/ (wheel-only test root)"
)


def _names_read_under_src() -> set[str]:
    return _names_read_under(REPO)


@needs_src_tree
def test_env_table_equals_env_vars_read_under_src() -> None:
    """[contract] the table and the code agree in both directions."""
    assert env_var_names() == frozenset(_names_read_under_src())


@needs_src_tree
def test_src_scan_works_outside_a_git_checkout(tmp_path: Path) -> None:
    """[contract] the scan reads the tree, not git: mutmut's mutants/ and tarballs work."""
    shutil.copytree(REPO / "src", tmp_path / "src")
    scanned = _names_read_under(tmp_path)
    assert scanned == _names_read_under_src()
    assert "LINC_HOME" in scanned


def test_src_scan_ignores_mutmut_mutant_copies(tmp_path: Path) -> None:
    """[contract] mutmut's injected code (trampoline, ``x_f__mutmut_N`` copies) is skipped."""
    pkg = tmp_path / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "m.py").write_text(
        "import os\n\n\n"
        'def _mutmut_trampoline(orig):\n    return os.environ["MUTANT_UNDER_TEST"]\n\n\n'
        'def x_f__mutmut_orig():\n    return os.environ.get("REAL_VAR")\n\n\n'
        'def x_f__mutmut_1():\n    return os.environ.get("XXREAL_VARXX")\n\n\n'
        "def x_f__mutmut_2():\n    return os.environ.get(None)\n\n\n"
        'class C:\n    def xǁCǁg__mutmut_3(self):\n        return os.getenv("XXOTHERXX")\n'
    )
    assert _names_read_under(tmp_path) == {"REAL_VAR"}


def _scan_one(tmp_path: Path, source: str, rel: str = "pkg/m.py") -> set[str]:
    path = tmp_path / "src" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return _names_read_under(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        'from os import environ\n\nenviron.get("X")\n',
        'import os\n\nlist(map(os.getenv, ("X",)))\n',
        'from os import getenv\n\ngetenv("X")\n',
        'from os import environ\n\nenviron["X"]\n',
        'import os\n\nos.environ["X"]\n',
        'import os\n\nNAME = "X"\nos.environ.get(NAME)\n',
        'import os\n\nNAME: str = "X"\nos.getenv(NAME, "d")\n',
        'import os\n\nVARS = ("X",)\nmap(os.getenv, VARS)\n',
        'import os\n\nmap(os.environ.get, ["X"])\n',
    ],
)
def test_src_scan_sees_every_analysable_read_form(tmp_path: Path, source: str) -> None:
    """[contract] from-imports, map(getter, names) and constant names are all resolved."""
    assert _scan_one(tmp_path, source) == {"X"}


@pytest.mark.parametrize(
    "source",
    [
        'import os\n\nenv = os.environ\nenv.get("X")\n',
        'import os\n\ng = os.getenv\ng("X")\n',
        'from os import environ as env\n\nenv.get("X")\n',
        'from os import getenv as g\n\ng("X")\n',
        'from os import *\n\nenviron.get("X")\n',
        'import os as o\n\no.environ.get("X")\n',
        "import os\n\ndef f(name):\n    return os.getenv(name)\n",
        'import os\n\nos.environ.get(key="X")\n',
        'import os\n\n"X" in os.environ\n',
        "import os\n\nos.environ.copy()\n",
        'import os\n\nos.environ.setdefault("X", "1")\n',
        "import os\n\ndict(os.environ)\n",
        "import os\n\nmap(os.getenv, names())\n",
        'import os\n\nos.environb.get(b"X")\n',
        'import os\n\nos.getenv("X" + "Y")\n',
    ],
)
def test_src_scan_fails_on_a_read_it_cannot_analyse(tmp_path: Path, source: str) -> None:
    """[contract] an alias, star/aliased import or non-call use is red, never invisible."""
    with pytest.raises(UnanalysableEnvReadError, match=r"pkg/m\.py:\d+: "):
        _scan_one(tmp_path, source)


@pytest.mark.parametrize(
    ("rel", "source"),
    [
        ("linc_crucible/core/deployer.py", "import os\n\nbuild_env = dict(os.environ)\n"),
        ("linc_crucible/core/subprocess_utils.py", "import os\n\nkw = {**os.environ, **env}\n"),
    ],
)
def test_src_scan_allows_only_the_pinned_whole_environment_copies(
    tmp_path: Path, rel: str, source: str
) -> None:
    """The two subprocess pass-through copies are allowed where they live, nowhere else."""
    assert _scan_one(tmp_path, source, rel) == set()
    with pytest.raises(UnanalysableEnvReadError):
        _scan_one(tmp_path / "elsewhere", source)


def test_env_table_has_no_duplicates_and_every_row_is_filled() -> None:
    """[contract] one row per name; every row has a default and a purpose."""
    names = [v.name for v in ENV_VARS]
    assert len(names) == len(set(names))
    for var in ENV_VARS:
        assert var.default and var.purpose, var.name
        assert "|" not in var.default + var.purpose, var.name


def test_known_env_vars_are_pinned() -> None:
    assert {
        "LINC_HOME",
        "LINC_WORKSPACE",
        "LINC_PROJECTS_DIR",
        "LINC_DEVICE_FLEET_FILE",
        "LINC_LOG_LEVEL",
        "LINC_BUILD_GATE",
        "LINC_CRUCIBLE_BUILD_TIMEOUT",
        "FORGE_PROJECTS_FILE",
        "FORGE_RUN_ID",
        "CRUCIBLE_BASELINES_DIR",
        "CRUCIBLE_SCORES_DIR",
        "CRUCIBLE_FLEET_FILE",
        "CRUCIBLE_IOS_TEAM_ID",
        "CRUCIBLE_MOBSF_API_KEY",
        "ANDROID_HOME",
        "ANDROID_SDK_ROOT",
    } <= env_var_names()


def test_render_env_table_exact_shape() -> None:
    table = render_env_table([EnvVar("A_VAR", "`1`", "Does a thing")])
    assert table == (
        "| Environment variable | Default | Purpose |\n"
        "|----------------------|---------|---------|\n"
        "| `A_VAR` | `1` | Does a thing |"
    )


def test_render_env_table_defaults_to_env_vars() -> None:
    rows = render_env_table().splitlines()[2:]
    assert rows == [f"| `{v.name}` | {v.default} | {v.purpose} |" for v in ENV_VARS]


def test_no_row_is_documented_ahead_of_its_release() -> None:
    """[contract] every row describes behaviour main ships today: no "next release" marker."""
    for var in ENV_VARS:
        assert "next release" not in (var.default + var.purpose).lower(), var.name
    assert "next release" not in render_env_table().lower()


def test_usage_log_rows_describe_the_released_behaviour() -> None:
    """[contract] the #188 usage-log rows match observability.py on main."""
    rows = {v.name: v for v in ENV_VARS}
    assert rows["CRUCIBLE_USAGE_LOG"].default == "`redacted`"
    assert rows["CRUCIBLE_USAGE_LOG"].purpose == (
        "Local usage log mode: `off`, `redacted` or `full`; an unknown value means `redacted` "
        "(see [Trust](https://github.com/LincForge/crucible#trust))"
    )
    assert rows["CRUCIBLE_RUN_ID"].default == "`FORGE_RUN_ID`, else `manual`"
    assert rows["CRUCIBLE_RUN_ID"].purpose == (
        "Session id stamped on usage log entries; a value outside `[A-Za-z0-9._-]{1,128}` "
        "is logged as `manual`"
    )
    assert rows["FORGE_RUN_ID"].default == "`manual`"
    assert rows["FORGE_RUN_ID"].purpose == (
        "Deprecated session id, read when `CRUCIBLE_RUN_ID` is unset or empty"
    )


def test_usage_log_rows_agree_with_observability() -> None:
    """[contract] the documented default mode and run-id grammar are the ones the code uses."""
    from linc_crucible.core import observability

    rows = {v.name: v for v in ENV_VARS}
    assert f"`{observability._DEFAULT_MODE}`" == rows["CRUCIBLE_USAGE_LOG"].default
    assert f"`{observability._RUN_ID_RE.pattern}`" in rows["CRUCIBLE_RUN_ID"].purpose


FLEET_FILE_PURPOSE = (
    "Personal-device registry; devices marked `personal: true` are refused unless "
    "`allow_personal=True`. A missing file means no personal devices are registered; a file "
    "that exists but cannot be read or parsed refuses EVERY device operation unless "
    "`allow_personal=True`"
)


def test_fleet_file_row_describes_the_fail_open_device_policy() -> None:
    """[contract] the row states main's DevicePolicy: absent fails open, unreadable closed."""
    row = {v.name: v for v in ENV_VARS}["LINC_DEVICE_FLEET_FILE"]
    assert row.default == "`$LINC_HOME/device-fleet.yaml`"
    assert row.purpose == FLEET_FILE_PURPOSE


def test_fleet_file_default_agrees_with_device_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] the documented default is the path ``device_policy`` resolves."""
    from linc_crucible.core import device_policy

    monkeypatch.setenv("LINC_HOME", str(tmp_path))
    monkeypatch.delenv("LINC_DEVICE_FLEET_FILE", raising=False)
    resolved = device_policy._fleet_registry_path().relative_to(tmp_path).as_posix()
    row = {v.name: v for v in ENV_VARS}["LINC_DEVICE_FLEET_FILE"]
    assert row.default == f"`$LINC_HOME/{resolved}`"


def test_fleet_file_behaviour_agrees_with_device_policy(tmp_path: Path) -> None:
    """[contract] missing -> nothing refused; present but unreadable -> everything refused."""
    from linc_crucible.core.device_policy import load_device_policy

    unreadable = tmp_path / "dir.yaml"
    unreadable.mkdir()
    with structlog.testing.capture_logs():
        absent = load_device_policy(tmp_path / "missing.yaml")
        broken = load_device_policy(unreadable)
    assert (absent.state, absent.refusal("S1")) == ("absent", None)
    assert broken.state == "unreadable"
    assert broken.refusal("S1") is not None


def test_readme_env_rows_equal_env_vars() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- env-table:start -->\n(.*?)\n<!-- env-table:end -->", readme, re.S)
    assert match is not None, "README.md is missing the env-table markers"
    assert match.group(1) == render_env_table()


def test_render_constants_are_pinned() -> None:
    from linc_crucible import config_contract as cc

    assert cc._UNSET == "*(unset)*"
    assert render_env_table([]) == cc._HEADER


def test_linc_workspace_default_states_both_linc_home_and_unset_fallback() -> None:
    """[test] LINC_WORKSPACE row states both $LINC_HOME/projects/LINC and ~/projects/LINC."""
    row = {v.name: v for v in ENV_VARS}["LINC_WORKSPACE"]
    assert "$LINC_HOME/projects/LINC" in row.default
    assert "~/projects/LINC" in row.default
    assert "LINC_HOME" in row.default
