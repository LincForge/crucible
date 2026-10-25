"""Conformance: every crucible MCP tool is a CLI command under the kit's output contract."""

from __future__ import annotations

import anyio
import pytest
from linc_cli_kit.registry import stubbed_tool_fn
from linc_cli_kit.testing import assert_cli_parity
from mcp.server.fastmcp.exceptions import ToolError as FastMCPToolError

from linc_crucible.cli import _registry_server, cli

# `crucible_score` is a hand-written override (collision rule (b) — it pins `.score` as an
# alias of the scorer's `.composite`, which the generated command couldn't do on its own).
# The sample gives the conformance suite a valid invocation to drive checks 3/4/8 against.
_OVERRIDE_SAMPLES = {
    "crucible_score": {"project": "X"},
}


def test_cli_parity() -> None:
    report = assert_cli_parity(
        cli,
        _registry_server(),
        prefix="crucible_",
        skip={
            "crucible_baseline": (
                "covered by the hand-written `baseline` group (capture/list/prune); "
                "a11y mode + latest/promote stay MCP-only pending Phase 2"
            )
        },
        samples=_OVERRIDE_SAMPLES,
    )
    assert report.tools == 8
    assert report.skipped <= 1
    assert report.overrides == ["score"]


def test_contract_mcp_tools_reject_unknown_arguments_and_forbid_extra_schema() -> None:
    """[contract] Every linc-crucible MCP tool sets additionalProperties: false
    and rejects unknown arguments."""
    server = _registry_server()
    tools = anyio.run(server.list_tools)
    assert len(tools) == 8
    for tool in tools:
        assert tool.inputSchema.get("additionalProperties") is False, tool.name

    with (
        stubbed_tool_fn(server, "crucible_pour"),
        pytest.raises(FastMCPToolError, match="__unknown_arg__"),
    ):
        anyio.run(
            server.call_tool,
            "crucible_pour",
            {"project": "X", "__unknown_arg__": "unexpected"},
        )
