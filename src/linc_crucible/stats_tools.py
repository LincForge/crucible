from typing import Any

from linc_crucible.core.tool_args import usage_stats_args_error
from linc_crucible.server_context import _observed_tool, server


@server.tool()
@_observed_tool
def crucible_usage_stats(
    server_filter: str = "",
    since_hours: float = 0.0,
    include_rotated: bool = False,
    top_n: int = 50,
) -> dict[str, Any]:
    """Usage analytics over the shared MCP call log (~/.linc/mcp-calls.jsonl).

    The analysis layer the JSONL writer always lacked: per-tool call counts,
    success rate, and p50/p95 latency, plus the dormant (registered-but-unused)
    crucible tools and any test-fixture junk that leaked into the prod log.

    USE FOR: auditing which MCP tools are hot/cold/broken, spotting 0%-success
        tools (signature bugs), finding shelf-ware, measuring latency.
    NOT FOR: project quality scores (use crucible_score) or ship gates.
    TIP: the log is shared across nerve/crucible/spectra — omit server_filter for
        the whole-host picture, or pass "linc-crucible" to scope per-tool stats.
        Dormancy is always computed against crucible's own tool registry.

    Args:
        server_filter: Optional server filter (e.g. "linc-crucible"); empty = all.
        since_hours: Only count calls newer than this many hours; 0 = all history.
        include_rotated: Also read rotated mcp-calls.<epoch>.jsonl siblings.
        top_n: Cap the per_tool list to the N busiest tools (default 50).

    Returns:
        {log_path, server, window_hours, window, total_calls, junk_calls,
        overall_success_pct, distinct_tools, per_tool: [{tool, server, calls,
        success_pct, p50_ms, p95_ms, avg_ms, last_seen}], junk, dormant}. An
        absent call log yields zero counts, not an error.
    """
    from linc_crucible.core.usage_stats import compute_usage_stats

    args_error = usage_stats_args_error(server_filter, since_hours, include_rotated, top_n)
    if args_error:
        return args_error

    registered = [t.name for t in server._tool_manager.list_tools()]
    return compute_usage_stats(
        server=server_filter or None,
        registered_tools=registered,
        since_hours=since_hours or None,
        include_rotated=include_rotated,
        top_n=top_n,
    )
