"""Regression: conftest.py must isolate LINC_HOME to a tmp dir (not real ~/.linc).

The autouse ``_isolate_linc_env`` fixture redirects LINC_HOME to a per-test tmp
dir so nothing (including ``log_mcp_call`` via ``_observed_tool``) writes to the
production ``~/.linc``. Earlier this fixture merely *deleted* LINC_HOME, which
let default-path writes leak into the real log (imp 01606251).
"""

import os
from pathlib import Path


def test_linc_home_isolated_to_tmp() -> None:
    val = os.environ.get("LINC_HOME")
    assert val is not None, "LINC_HOME unset — check tests/conftest.py for _isolate_linc_env"
    # Must point at a pytest tmp dir, never the user's real ~/.linc.
    assert "pytest" in val or "tmp" in val.lower()
    assert Path(val) != Path.home() / ".linc"
