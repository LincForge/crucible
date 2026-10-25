from typing import Any

import anyio

import linc_crucible.server_context as ctx
from linc_crucible.core.inventory import empty_inventory
from linc_crucible.core.store_names import name_guarded, required_project_error
from linc_crucible.core.tool_args import assay_args_error, score_args_error
from linc_crucible.server_context import _observed_tool, server


def _not_found_error(project_name: str) -> dict[str, Any]:
    return {
        "error": f"Project '{project_name}' not found in forge registry",
        "error_category": "not_found",
    }


@server.tool()
@_observed_tool
async def crucible_assay(
    project: str,
    mode: str = "quick",
    perf: bool = False,
    a11y: bool = False,
    a11y_tree: bool = False,
    device: str = "",
    allow_personal: bool = False,
) -> dict[str, Any]:
    """Run a fresh quality assay (lint, tests, security, a11y) on a project.

    USE WHEN asked to run quality checks — lint, tests, security, a11y —
    on a project, or asked "is the code quality OK" before opening a PR.

    Heavyweight: takes minutes. Recomputes all quality dimensions and writes a
    new score record. Use this when an agent needs current quality data — for
    example pre-deploy gates, pre-PR verification, or post-refactor
    re-evaluation.

    USE FOR: running the full quality pipeline now, recomputing scores when the
        cached value is stale, fresh quality runs before opening a PR, pre-deploy
        gates, re-evaluating after a refactor.
    NOT FOR: reading a previously computed score without executing checks (use
        crucible_score — reads the persisted record cheaply, no recompute);
        ship-readiness across multiple dimensions (use crucible_preflight);
        runtime debugging of a deployed app (use nerve_logcat for live errors or
        nerve_inspect for crash signatures); deploying built artifacts to
        devices (use crucible_pour).
    TIP: full assay on a KMP project takes 90-180s; cache via crucible_score for
        dashboard reads. Use mode="quick" for fast feedback (lint + tests). Use
        mode="full" to add MobSF security scan (requires CRUCIBLE_MOBSF_API_KEY).
        Add perf=True to measure APK/IPA sizes. Add a11y=True to run static
        accessibility checks. Add a11y_tree=True to run a live a11y tree BVT via
        nerve (requires a connected device). Returns ~100-500 tokens (structured
        response).

    Args:
        project: Forge project name.
        mode: "quick" (lint + tests) or "full" (MobSF scan, requires CRUCIBLE_MOBSF_API_KEY).
        perf: If True, measure artifact sizes (APK/IPA) and include in score.
        a11y: If True, run static accessibility checks (missing labels, touch targets).
        a11y_tree: If True, fetch live a11y element tree via nerve and run BVT checks.
        device: Device serial for a11y_tree check (auto-discovers if empty).
        allow_personal: a11y_tree only. Permit reading a device registered
            ``personal: true`` (and proceed when the registry is unreadable).
            Default False.

    Returns:
        Quality results with overall score and ship gate status.
    """
    ctx._ensure_initialized()

    args_error = assay_args_error(project, mode, perf, a11y, a11y_tree, device, allow_personal)
    if args_error:
        return args_error
    if mode not in ("quick", "full"):
        return {"error": "mode must be 'quick' or 'full'."}

    def _do_assay() -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "mode": mode,
            "perf": perf,
            "a11y": a11y,
            "a11y_tree": a11y_tree,
            "device": device,
        }
        if allow_personal:
            kwargs["allow_personal"] = True
        report = ctx._assayer.assay(project, **kwargs)
        if getattr(report, "not_found", False):
            return _not_found_error(project)
        return report.to_dict()

    return await anyio.to_thread.run_sync(_do_assay, limiter=ctx._get_thread_limiter())


@server.tool()
@_observed_tool
def crucible_score(
    project: str,
    history: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    """Read the persisted quality score record for a project — no checks executed.

    USE WHEN asked for the latest quality score without re-running checks —
    a cheap, read-only lookup of the previously computed quality score.
    Returns immediately; never executes assay checks. Use this when an agent
    wants the stored score without triggering a fresh recompute.

    USE FOR: reading the cached or stored score record for a project; score
        history queries; previously computed quality value lookups; dashboard
        reads where freshness within hours is acceptable; pulling a persisted
        score record without running anything.
    NOT FOR: running a fresh quality evaluation that recomputes scores (use
        crucible_assay — runs lint, tests, security, a11y, writes a new score
        record); reading per-file CRAP scores (use crucible_assay output's
        category breakdown).

    TIP: returns ~100-500 tokens (structured response). Reads from the score
        store; never executes assay checks.

    Args:
        project: Forge project name.
        history: If True, return score history instead of just the latest entry.
        limit: Number of history entries to return (default 10).

    Returns:
        Score with dimensions and optional history.
    """
    ctx._ensure_initialized()

    # project is a score-store path component: refuse anything off the
    # [A-Za-z0-9._-] allowlist with validation_error=True (CLI exit 2).
    args_error = score_args_error(project, history, limit)
    if args_error:
        return args_error
    return name_guarded(lambda: _read_score(project, history, limit))


def _read_score(project: str, history: bool, limit: int) -> dict[str, Any]:
    """Body of ``crucible_score`` after argument validation (latest or history)."""
    if history:
        scores = ctx._scorer.load_history(project, limit=limit)
        result: dict[str, Any] = {
            "project": project,
            "count": len(scores),
            "scores": [s.to_dict() for s in scores],
        }
        if not scores:
            result.update(
                empty_inventory(ctx._scorer.scores_dir, f"project '{project}'", "scores")
            )
        return result
    else:
        latest = ctx._scorer.load_latest(project)
        if not latest:
            return {
                "project": project,
                "error": f"No scores found. Run 'crucible assay {project}' first.",
                "error_category": "not_found",
                **empty_inventory(ctx._scorer.scores_dir, f"project '{project}'", "scores"),
            }
        return latest.to_dict()


@server.tool()
@_observed_tool
def crucible_preflight(
    project: str,
) -> dict[str, Any]:
    """Ship-readiness gate — pre-submission check across the 5 release dimensions.

    USE WHEN asked to check ship/store readiness —
    instead of manually chaining a ./gradlew build with ad-hoc nerve checks.
    The pre-submission readiness audit before tagging a build for the App
    Store or Play Store. Composite score across 5 release dimensions
    (code quality, store listing, visual assets, brand brief, build
    artifacts) against the 80% ship threshold.

    USE FOR: app-store readiness checks, pre-submission readiness audit
        across the 5 release dimensions, ship-readiness gate before
        tagging a build, "are we ready to submit to App Store / Play
        Store right now?" questions, the 80% release-threshold gate
        before publishing.
    NOT FOR: running fresh quality checks like lint/tests/security on
        the codebase (use crucible_assay), deploying a build onto a
        device (use crucible_pour), reading a previously-stored quality
        score with no recompute (use crucible_score).
    TIP: run crucible_assay first so the latest quality score is on
        disk; preflight reads it as the code-quality dimension. The
        readiness score is weighted: code quality 30%, store listing
        20%, visual assets 20%, brand brief 15%, build artifacts 15%;
        ship threshold 80%. Returns ~100-500 tokens.

    Args:
        project: Forge project name (e.g. "myapp").

    Returns:
        Readiness report with per-category checks, composite score, and ready/not-ready status.
    """
    ctx._ensure_initialized()

    args_error = required_project_error(project)
    if args_error:
        return args_error

    report = ctx._preflight_checker.preflight(project)
    if getattr(report, "not_found", False):
        return _not_found_error(project)
    return report.to_dict()
