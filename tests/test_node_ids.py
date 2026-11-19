"""Guard test: every collected pytest node id is ASCII-only and backslash-free.

mutmut re-selects tests by node id, and pytest cannot reliably select a node id
that holds a backslash escape sequence or non-ASCII text (`'run\\nid'`, `'\\xe9'`,
`'a\\\\b'`), causing pr-diff-mutation to fail with an instrument error (exit 4).
``tests/conftest.py::pytest_make_parametrize_id`` sanitizes string parameters at
collection time; this test guards the full collected suite against regressions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _collect_node_ids() -> list[str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    node_ids = [line for line in proc.stdout.splitlines() if "::" in line]
    assert node_ids, proc.stdout
    return node_ids


def test_collected_node_ids_are_ascii_and_backslash_free() -> None:
    """[test] Every collected node id is printable ASCII with no backslash."""
    offending = [
        node_id
        for node_id in _collect_node_ids()
        if "\\" in node_id or not node_id.isascii() or not node_id.isprintable()
    ]
    assert not offending, (
        f"{len(offending)} node id(s) contain a backslash or non-ASCII/non-printable char:\n"
        + "\n".join(f"  {nid}" for nid in offending)
    )


def test_pytest_make_parametrize_id_contract() -> None:
    """Direct unit contract for tests/conftest.py::pytest_make_parametrize_id."""
    from tests.conftest import pytest_make_parametrize_id

    assert pytest_make_parametrize_id(None, 42, "x") is None
    assert pytest_make_parametrize_id(None, None, "x") is None
    assert pytest_make_parametrize_id(None, ("a", "b"), "x") is None

    assert pytest_make_parametrize_id(None, "simple_ascii", "x") == "simple_ascii"
    assert pytest_make_parametrize_id(None, "", "x") == ""

    escaped = [
        pytest_make_parametrize_id(None, val, "x")
        for val in ("a\nb", "a\tb", "a\x00b", "a\\b", "café", "é", "a" * 129, "a" * 10000)
    ]
    for token in escaped:
        assert isinstance(token, str) and token
        assert token.isascii() and token.isprintable()
        assert "\\" not in token
    assert len(set(escaped)) == len(escaped)
