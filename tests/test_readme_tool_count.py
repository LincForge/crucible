"""README.md tool count drift detector.

Pins the README tool count to len(list_tools()) like
tests/test_mcp_docstrings.py:28-46 does for LINC.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import anyio

from linc_crucible.mcp_server import server

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_readme_tool_count_matches_registry() -> None:
    """README.md's "## MCP Tools (N)" header must equal the live tool registry count."""
    tools = anyio.run(server.list_tools)
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## MCP Tools \((\d+)\)$", readme, re.MULTILINE)
    assert m, "README.md is missing the '## MCP Tools (N)' header"
    assert int(m.group(1)) == len(tools), (
        f"README.md claims {m.group(1)} MCP tools but the registry has {len(tools)}: "
        f"{sorted(t.name for t in tools)}"
    )

    m_status = re.search(r"(\d+) MCP tools", readme)
    assert m_status, "README.md is missing 'N MCP tools' in status"
    assert int(m_status.group(1)) == len(tools), (
        f"README.md status claims {m_status.group(1)} MCP tools "
        f"but the registry has {len(tools)}: {sorted(t.name for t in tools)}"
    )


def _live_tool_count() -> int:
    return len(anyio.run(server.list_tools))


def test_architecture_tool_counts_match_registry() -> None:
    """Every "N tools" / "N MCP tool..." claim in ARCHITECTURE.md equals the live registry."""
    text = (PROJECT_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    claims = [int(n) for n in re.findall(r"\b(\d+) (?:MCP )?tool(?:s| registrations)\b", text)]
    assert claims, "ARCHITECTURE.md states no tool count"
    assert claims == [_live_tool_count()] * len(claims), (
        f"ARCHITECTURE.md tool counts {claims} != registry {_live_tool_count()}"
    )


def test_architecture_component_map_lists_bvt_and_usage_stats() -> None:
    text = (PROJECT_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    component_map = text.split("## Component Map", 1)[1].split("\n## ", 1)[0]
    assert "bvt" in component_map
    assert "usage_stats" in component_map


def test_changelog_latest_release_tool_line_matches_registry() -> None:
    """The newest CHANGELOG release section's "MCP Tools (N)" line equals the live registry."""
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 2)[1]
    counts = [int(n) for n in re.findall(r"MCP Tools \((\d+)\)", latest)]
    assert counts == [_live_tool_count()], (
        f"CHANGELOG latest release tool lines {counts} != registry {_live_tool_count()}"
    )
