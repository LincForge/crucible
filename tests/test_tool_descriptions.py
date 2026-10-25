"""Tool-description discoverability tests (issue #97).

45 head-to-head mobile-agent-bench runs driving the full LINC MCP stack
(nerve+crucible+spectra) showed agents invoking crucible 0 times, including
on tasks crucible_preflight exists for. Transcript inspection ruled out a
registration failure — crucible's mcp_servers entry looks identical to
nerve's, and the deferred-tool ("tools" list empty at session init,
loaded on demand via ToolSearch) pattern is shared by all three servers —
but 0 of 108 ToolSearch queries across those runs ever contained the word
"crucible". This is a discoverability defect: descriptions must lead with
the concrete agent-task trigger phrase an agent would type into ToolSearch
or reach for when planning, not internal architecture vocabulary.

These tests assert literal trigger-phrase substrings per tool. They are
written to FAIL against the pre-#97 description text (verified in the
RED step of TDD) and pass once mcp_server.py's docstrings are updated to
lead with the trigger phrase.
"""

import linc_crucible.mcp_server as m


def _description(tool_name: str) -> str:
    """Return the exact text FastMCP exposes to clients for one tool.

    Reads from the live tool registry (not the raw ``__doc__``) so the
    assertion matches what an agent's ToolSearch / tool-selection actually
    sees, not just what's in the source.
    """
    tools = {t.name: t for t in m.server._tool_manager.list_tools()}
    assert tool_name in tools, f"{tool_name} is not registered on the MCP server"
    description = tools[tool_name].description
    assert description, f"{tool_name} has no description"
    return description


def test_crucible_pour_names_build_and_deploy_trigger():
    """crucible_pour must name the build+deploy trigger the bench task used."""
    desc = _description("crucible_pour")
    assert "build and deploy an app to a device" in desc


def test_crucible_pour_positions_against_raw_gradle_xcodebuild():
    """crucible_pour must explicitly say it replaces raw ./gradlew / xcodebuild calls."""
    desc = _description("crucible_pour")
    assert "instead of ./gradlew or xcodebuild" in desc


def test_crucible_assay_names_quality_check_trigger():
    """crucible_assay must name the concrete "run quality checks" trigger."""
    desc = _description("crucible_assay")
    assert "USE WHEN asked to run quality checks" in desc


def test_crucible_preflight_names_ship_store_readiness_trigger():
    """crucible_preflight must name the literal ship/store readiness trigger."""
    desc = _description("crucible_preflight")
    assert "check ship/store readiness" in desc


def test_crucible_preflight_positions_against_raw_gradle_and_nerve():
    """crucible_preflight must claim the readiness-check task shape an agent
    would otherwise solve by manually chaining gradle + nerve — the exact
    competing path that left it unused in the bench (agents solved
    everything with nerve primitives)."""
    desc = _description("crucible_preflight")
    assert "instead of manually chaining a ./gradlew build with ad-hoc nerve checks" in desc


def test_crucible_score_names_cached_score_trigger():
    """crucible_score must name the "read the latest score" trigger."""
    desc = _description("crucible_score")
    assert "USE WHEN asked for the latest quality score" in desc


def test_crucible_baseline_names_golden_baseline_trigger():
    """crucible_baseline must name the golden-screenshot/a11y-baseline trigger."""
    desc = _description("crucible_baseline")
    assert "USE WHEN asked to capture or compare golden screenshots" in desc


def test_crucible_bvt_names_smoke_flow_trigger():
    """crucible_bvt must name the scripted smoke-flow verification trigger."""
    desc = _description("crucible_bvt")
    assert "USE WHEN asked to run a scripted smoke-flow verification suite" in desc


def test_all_six_core_tools_are_registered():
    """Sanity check: the 6 tools this issue targets are actually on the server.

    Guards against a typo silently turning every trigger-phrase test into a
    vacuous pass-by-KeyError-never-happening — each test above already
    asserts registration individually, but this pins the full set in one
    place for a quick read of what's covered.
    """
    tools = {t.name for t in m.server._tool_manager.list_tools()}
    core = {
        "crucible_pour",
        "crucible_assay",
        "crucible_preflight",
        "crucible_score",
        "crucible_baseline",
        "crucible_bvt",
    }
    assert core <= tools
