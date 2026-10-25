"""Fix A: per-file package resolution in the BVT loader.

The loader must let each file's own ``package:`` win, fall back to the
suite-level default only when a file omits it, and fail loudly at load
time when no package can be resolved. Root cause 8f09da13.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from linc_crucible.core.bvt.schema import load_bvt_dir


def _write(d: Path, name: str, body: str) -> None:
    (d / name).write_text(body)


def _suite_yaml(name: str, package: str | None) -> str:
    """Build a one-test suite yaml. package=None omits the `package:` key."""
    head = f"package: {package}\n" if package is not None else ""
    return f'{head}tests:\n  - name: {name}\n    steps:\n      - launch: "{{package}}"\n'


def test_per_file_package_wins_over_suite_default(tmp_path: Path) -> None:
    _write(tmp_path, "01-phone.yaml", _suite_yaml("t_phone", "com.app"))
    _write(tmp_path, "watch-01.yaml", _suite_yaml("t_watch", "com.app.watch"))

    suite = load_bvt_dir(tmp_path)
    by_name = {t.name: t for t in suite.tests}

    assert by_name["t_phone"].package == "com.app"
    assert by_name["t_watch"].package == "com.app.watch"
    # {package} substitution uses the per-file package
    assert by_name["t_watch"].steps[0].target == "com.app.watch"
    # Suite-level display package stays the first file's package
    assert suite.package == "com.app"


def test_package_less_file_inherits_suite_default(tmp_path: Path) -> None:
    _write(tmp_path, "01-main.yaml", _suite_yaml("t1", "com.app"))
    _write(tmp_path, "02-more.yaml", _suite_yaml("t2", None))

    suite = load_bvt_dir(tmp_path)
    by_name = {t.name: t for t in suite.tests}
    assert by_name["t1"].package == "com.app"
    assert by_name["t2"].package == "com.app"  # inherited default


def test_package_less_file_with_no_default_raises(tmp_path: Path) -> None:
    _write(tmp_path, "01-nopkg.yaml", _suite_yaml("t1", None))
    with pytest.raises(ValueError, match="no package"):
        load_bvt_dir(tmp_path)


def test_empty_string_package_treated_as_missing(tmp_path: Path) -> None:
    _write(tmp_path, "01-empty.yaml", _suite_yaml("t1", '""'))
    with pytest.raises(ValueError, match="no package"):
        load_bvt_dir(tmp_path)


def test_single_package_suite_regression(tmp_path: Path) -> None:
    _write(tmp_path, "01-a.yaml", _suite_yaml("a", "com.app"))
    _write(tmp_path, "02-b.yaml", _suite_yaml("b", "com.app"))
    suite = load_bvt_dir(tmp_path)
    assert suite.package == "com.app"
    assert {t.package for t in suite.tests} == {"com.app"}
