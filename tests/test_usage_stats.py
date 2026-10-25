"""Tests for the mcp-calls usage-stats reader (the analysis layer)."""

from __future__ import annotations

import json
from pathlib import Path

from linc_crucible.core.usage_stats import compute_usage_stats


def _write_log(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(
    server="linc-crucible", tool="crucible_pour", ms=10.0, ok=True, ts="2026-06-27T00:00:00Z"
):
    return {
        "timestamp": ts,
        "server": server,
        "tool": tool,
        "params": {},
        "duration_ms": ms,
        "success": ok,
        "error": None,
    }


class TestComputeUsageStats:
    def test_per_tool_counts_and_success(self, tmp_path: Path) -> None:
        log = tmp_path / "mcp-calls.jsonl"
        _write_log(
            log,
            [
                _rec(tool="crucible_pour", ok=True),
                _rec(tool="crucible_pour", ok=False),
                _rec(tool="crucible_assay", ok=True),
            ],
        )
        out = compute_usage_stats(log_path=log)
        assert out["total_calls"] == 3
        assert out["distinct_tools"] == 2
        by_tool = {t["tool"]: t for t in out["per_tool"]}
        assert by_tool["crucible_pour"]["calls"] == 2
        assert by_tool["crucible_pour"]["success_pct"] == 50.0
        assert out["per_tool"][0]["tool"] == "crucible_pour"  # busiest first

    def test_latency_percentiles(self, tmp_path: Path) -> None:
        log = tmp_path / "mcp-calls.jsonl"
        _write_log(log, [_rec(ms=float(d)) for d in (10, 20, 30, 40, 100)])
        out = compute_usage_stats(log_path=log)
        t = out["per_tool"][0]
        assert t["p50_ms"] == 30.0
        assert t["p95_ms"] == 100.0
        assert t["avg_ms"] == 40.0

    def test_junk_quarantined(self, tmp_path: Path) -> None:
        log = tmp_path / "mcp-calls.jsonl"
        _write_log(
            log,
            [
                _rec(tool="crucible_pour"),
                _rec(tool="broken_dummy_tool"),
                _rec(tool="crucible_fake_block"),
                _rec(tool="crucible_bad"),
                {"timestamp": "2026-06-27T00:00:00Z", "server": None, "tool": "x", "success": 1},
            ],
        )
        out = compute_usage_stats(log_path=log)
        assert out["total_calls"] == 1
        assert out["junk_calls"] == 4
        assert "linc-crucible/broken_dummy_tool" in out["junk"]

    def test_server_filter(self, tmp_path: Path) -> None:
        log = tmp_path / "mcp-calls.jsonl"
        _write_log(
            log,
            [
                _rec(server="linc-crucible", tool="crucible_pour"),
                _rec(server="linc-nerve", tool="nerve_state"),
            ],
        )
        out = compute_usage_stats(log_path=log, server="linc-crucible")
        assert out["total_calls"] == 1
        assert out["per_tool"][0]["tool"] == "crucible_pour"

    def test_dormant_list(self, tmp_path: Path) -> None:
        log = tmp_path / "mcp-calls.jsonl"
        _write_log(log, [_rec(tool="crucible_pour")])
        out = compute_usage_stats(
            log_path=log, registered_tools=["crucible_pour", "crucible_assay", "crucible_score"]
        )
        assert out["registered_count"] == 3
        assert out["dormant"] == ["crucible_assay", "crucible_score"]
        assert out["dormant_count"] == 2

    def test_since_hours_window(self, tmp_path: Path) -> None:
        log = tmp_path / "mcp-calls.jsonl"
        _write_log(
            log,
            [
                _rec(tool="crucible_old", ts="2000-01-01T00:00:00Z"),
                _rec(tool="crucible_new", ts="2999-01-01T00:00:00Z"),
            ],
        )
        out = compute_usage_stats(log_path=log, since_hours=1.0)
        assert {t["tool"] for t in out["per_tool"]} == {"crucible_new"}

    def test_missing_log_is_empty(self, tmp_path: Path) -> None:
        out = compute_usage_stats(log_path=tmp_path / "nope.jsonl")
        assert out["total_calls"] == 0
        assert out["per_tool"] == []

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "mcp-calls.jsonl"
        with log.open("w") as f:
            f.write(json.dumps(_rec(tool="crucible_pour")) + "\n")
            f.write("not json\n")
            f.write("\n")
        out = compute_usage_stats(log_path=log)
        assert out["total_calls"] == 1
