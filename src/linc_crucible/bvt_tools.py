import json
import time

import structlog

import linc_crucible.server_context as ctx
from linc_crucible.core.deployer import (
    NERVE_UNREACHABLE_MSG,
    discover_for_policy,
    personal_device_error,
    resolve_policy_target,
)
from linc_crucible.core.tool_args import bvt_args_error
from linc_crucible.server_context import _observed_tool, server

logger = structlog.get_logger().bind(domain="build_pipeline")

_BVT_PRECONDITION_REASONS = (
    "unknown-project",
    "suite-dir-missing",
    "nerve-daemon-unreachable",
    "no-tests-match-tags",
    "device-not-ready",
    "package-mismatch",
    "device-reserved",
    "platform-mismatch",
)

# Tags that name a device platform rather than a feature/area. A test whose
# tags include one of these OTHER than the resolved device's own platform is
# a platform mismatch — see _device_platform / crucible_bvt below.
_PLATFORM_TAGS = {"android", "ios"}


def _device_platform(devices: list[dict], serial: str) -> str | None:
    """Resolve a device's platform ("android"/"ios") from discover() entries.

    Exact serial match only. Returns None when the serial isn't listed or
    the matching entry has no `platform` key — callers treat that as
    "unknown" and leave the suite untouched rather than guessing.
    """
    for d in devices:
        if d.get("serial") == serial:
            return d.get("platform")
    return None


def _partition_platform_mismatches(
    tests: list, platform: str
) -> tuple[list, list[tuple[object, str]]]:
    """Split suite tests into (matching, mismatched) against a device platform.

    A test is a mismatch when it carries a platform tag (from _PLATFORM_TAGS)
    other than `platform` — e.g. a test tagged "ios" against an Android
    device. Untagged tests, and tests tagged with the device's own platform,
    always match.
    """
    matching: list = []
    mismatched: list[tuple[object, str]] = []
    for t in tests:
        mismatch_tag = next(
            (tag for tag in (t.tags or []) if tag in _PLATFORM_TAGS and tag != platform),
            None,
        )
        if mismatch_tag:
            mismatched.append((t, mismatch_tag))
        else:
            matching.append(t)
    return matching, mismatched


def _personal_device_refusal(device: str, allow_personal: bool) -> str | None:
    """Serialized refusal when ``device`` is personal (or the registry is unreadable).

    Returns None when the run may proceed. Shape mirrors crucible_pour's refusal
    plus ``ran: false`` so callers can branch on ``error_category`` uniformly.
    """
    devices = discover_for_policy(ctx._nerve_client) if ctx._nerve_client else []
    _, refusal = resolve_policy_target(devices, device, ctx._device_policy, allow_personal)
    if not refusal:
        return None
    logger.warning("crucible_bvt_personal_device_refused", device=device)
    return json.dumps({"ran": False, **personal_device_error(refusal)})


def _reservation_refusal(device: str) -> str | None:
    """Reservation guard — serialized precondition or None.

    Refuses to drive a device another nerve session is actively holding so a
    concurrent BVT run never steals the foreground mid-comparison from a device
    another session has reserved. nerve_pool degrades gracefully: an
    unreachable daemon here means the ledger itself couldn't be read, so the
    check is skipped rather than treating "unreachable" as "unreserved".
    """
    from .core.nerve_client import find_reservation

    try:
        pool_result = ctx._nerve_client.pool()
    except ConnectionError:
        return None
    reservation = find_reservation((pool_result or {}).get("reservations", []), device)
    if not reservation:
        return None
    return _bvt_precondition(
        "device-reserved",
        f"{device} reserved by session {reservation.get('session_id')} "
        f"(purpose: {reservation.get('purpose')}, "
        f"age {reservation.get('age_seconds')}s)",
    )


def _device_guard(device: str, allow_personal: bool) -> str | None:
    """Personal-device policy first (a personal device is never touched — not
    even by the reservation lookup), then the reservation guard."""
    return _personal_device_refusal(device, allow_personal) or _reservation_refusal(device)


def _bvt_precondition(reason: str, detail: str) -> str:
    """Serialize a crucible_bvt precondition failure as {ran: false, reason, detail}.

    Only KNOWN preconditions route here — unexpected exceptions must keep
    raising (they indicate real crucible bugs, not environment problems).
    """
    if reason not in _BVT_PRECONDITION_REASONS:
        raise ValueError(f"unknown precondition reason: {reason}")
    logger.warning("crucible_bvt_precondition_failed", reason=reason, detail=detail)
    return json.dumps({"ran": False, "reason": reason, "detail": detail})


@server.tool()
@_observed_tool
async def crucible_bvt(
    project: str,
    device: str,
    suite_dir: str = ".bvt",
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    output_dir: str | None = None,
    compare_against_baseline: bool = False,
    app_path: str | None = None,
    # Mirrors core.bvt.runner._SUITE_BUDGET_DEFAULT_S (pinned equal by a test); a
    # literal keeps the heavy runner import lazy, inside the tool body.
    budget_s: float | None = 1800.0,
    allow_personal: bool = False,
) -> str:
    """Run BVT (Build Verification Test) suite for a project on a device.

    USE WHEN asked to run a scripted smoke-flow verification suite on a
    real device or emulator.

    USE FOR: scripted smoke-flow verification on a real device/emulator —
        launch, tap, assert-element steps defined in the project's .bvt yaml.
    NOT FOR: building or installing the app (use crucible_pour first),
        static quality checks (use crucible_assay), or freeform device
        interaction — drive nerve tools directly for that.

    Personal devices: a device registered ``personal: true`` in the LINC
    device registry (device-fleet.yaml) is refused before the reservation
    guard, readiness gate or runner — nothing touches it. An unreadable
    registry refuses every device (fail closed). Pass allow_personal=True
    only when the owner has asked for that device.

    Args:
        project: Project name as registered in LINC (e.g. "myapp").
                 Resolved against the project registry.
        device: Device serial — passed through to nerve_discover for resolution.
        suite_dir: Path to BVT yaml directory, relative to project root.
                   Default ".bvt" matches all 6 apps' current layout.
        tags: Optional tag filter (e.g. ["phone"], ["watch"]). Tests
              without a matching tag are skipped. None = run everything.
        exclude_tags: Optional tag filter applied AFTER `tags`. A test is
              dropped if ANY of its tags intersects `exclude_tags` (for
              example, passing `exclude_tags=["ios"]` on an Android lane).
              Untagged tests are never dropped by this filter. None = no
              exclusion (identical to today's behavior).
        output_dir: Where to write screenshots + perf artifacts.
                    Default: tempfile.mkdtemp(prefix="crucible_bvt_").
        compare_against_baseline: When True, diff each captured screenshot
                    against its latest golden (BaselineManager) and attach a
                    `visual_compare` block + top-level `visual_advisory` mirror.
                    The compare is OBSERVATIONAL — it never flips `passed`.
                    Default False keeps the output identical to a normal run.
        app_path: Built .app (or .apk) used to reinstall the app after a
                    cold_start wipe. Required only for iOS-simulator
                    cold_start tests (nerve's clear there is uninstall-based);
                    Android runs ignore it. Default None matches today's
                    behavior — pass a path only when running cold_start
                    against an iOS simulator.
        budget_s: Wall-clock budget for the whole suite, in seconds (default
                    1800; None = unlimited). Checked before each test — a
                    running step is never preempted. Once spent, every
                    remaining test is recorded skipped with a skip_reason
                    starting "HARNESS: suite budget", `budget_exhausted` is
                    true, and `passed` is false.
        allow_personal: Permit running on a device registered
                    ``personal: true`` (and proceed when the registry is
                    unreadable). Default False.

    Returns:
        On a run that executed: JSON-serialized BVTRunResult with shape:
            {ran: true, device, package, tests:[{name, passed, steps_run,
             steps_passed, duration_ms, failure_message, harness_error,
             screenshots:[...]}],
             total_run, total_passed, total_failed, total_skipped, passed,
             total_harness_errors, harness_errors:[<test name>, ...],
             budget_exhausted, duration_ms, output_dir}
        Skipped tests count in NEITHER total_passed nor total_failed, and
        `passed` requires at least one executed (non-skipped) test green.
        A test whose fail-fast step is a HARNESS fault (the UI tree itself
        could not be read — see `failure_message` starting "HARNESS:") still
        counts as failed for `passed`/`total_failed`; `total_harness_errors`
        and `harness_errors` let a caller distinguish "the harness broke"
        from "the product broke" without changing pass/fail semantics.
        On a KNOWN precondition failure (environment not ready — the suite
        did not execute): {ran: false, reason: <code>, detail: <message>}
        with one distinct machine-readable reason code per precondition:
            unknown-project          project not in the registry
            suite-dir-missing        suite_dir doesn't exist under the project
            nerve-daemon-unreachable :8111 health probe failed (connect error /
                                     read timeout / daemon down)
            no-tests-match-tags      tag filter excluded every test
            device-not-ready         readiness gate said not ready / absent
            package-mismatch         suite yaml `package:` contradicts the
                                     project's resolved applicationId
            device-reserved          target serial has an active reservation
                                     held by another nerve_pool session
        On a personal-device refusal: {ran: false, success: false,
        error: <message>, error_category: "personal_device"}.

    Behavior:
        Runs fail-fast WITHIN a single test (first failing step aborts that
        test). Continues BETWEEN tests (one failed test does not abort the
        suite). Reading "2/3 passed" means the runner kept going past one
        failure to attempt the remaining tests.

    Raises:
        Unexpected errors (anything NOT a known precondition above) still
        raise — a raised error means crucible itself broke, not that the
        environment wasn't ready.
    """
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    from .core.bvt.runner import BVTRunner, BVTTestResult
    from .core.bvt.schema import load_bvt_dir

    ctx._ensure_initialized()

    args_error = bvt_args_error(
        project,
        device,
        suite_dir,
        tags,
        exclude_tags,
        output_dir,
        compare_against_baseline,
        app_path,
        budget_s,
        allow_personal,
    )
    if args_error:
        return args_error

    start = time.monotonic()
    # Known precondition failures return {ran: false, reason} instead of
    # raising — each check below is explicit; nothing is wrapped
    # in a broad except, so unexpected exceptions still raise.
    # Resolve project path via the same registry crucible_pour uses
    forge_project = ctx._forge_reader.get(project)
    if not forge_project or not forge_project.path:
        return _bvt_precondition("unknown-project", f"Unknown project: {project}")

    bvt_path = Path(forge_project.path) / suite_dir
    if not bvt_path.is_dir():
        return _bvt_precondition("suite-dir-missing", f"BVT directory not found: {bvt_path}")

    # Daemon liveness check
    if not ctx._bvt_nerve_adapter.available:
        return _bvt_precondition(
            "nerve-daemon-unreachable",
            NERVE_UNREACHABLE_MSG,
        )

    # Device guards: personal-device policy, then the reservation guard —
    # both before the readiness gate, so a refused device is never touched.
    refused = _device_guard(device, allow_personal)
    if refused:
        return refused

    # Load + optionally filter by tag
    suite = load_bvt_dir(bvt_path)

    # Package-mismatch guard: a stale yaml `package:` that
    # contradicts the project's real applicationId would launch/assert against
    # the wrong app and report nonsense. Only a POSITIVE mismatch blocks —
    # an unresolvable package (no gradle source, no APK) trusts the yaml.
    from .core.package_resolver import resolve_effective_package

    effective = resolve_effective_package(forge_project.path, "debug") if suite.package else None
    if effective and suite.package and effective != suite.package:
        return _bvt_precondition(
            "package-mismatch",
            f"suite yaml package: {suite.package} != resolved applicationId "
            f"{effective} for {project} — fix the .bvt yaml or the gradle config",
        )

    if tags:
        tag_set = set(tags)
        suite.tests = [t for t in suite.tests if tag_set & set(t.tags or [])]

    if exclude_tags:
        exclude_set = set(exclude_tags)
        suite.tests = [t for t in suite.tests if not (exclude_set & set(t.tags or []))]

    if (tags or exclude_tags) and not suite.tests:
        return _bvt_precondition(
            "no-tests-match-tags",
            f"No tests in {bvt_path} match tags={tags} exclude_tags={exclude_tags}",
        )

    # Platform-mismatch guard: a test tagged for the
    # platform the resolved device is NOT (e.g. "ios" on an Android device)
    # never reaches the runner — it would launch the wrong bundle and report
    # a bogus failure. `discover()` not listing the device, or raising,
    # leaves device_platform None and the suite untouched: the existing
    # device-readiness gate already handles an absent/unreachable device.
    try:
        devices = ctx._bvt_nerve_adapter.discover()
    except Exception as e:
        logger.warning("crucible_bvt_discover_failed", device=device, error=str(e))
        devices = []

    mismatched_tests: list[tuple[object, str]] = []
    device_platform = _device_platform(devices, device)
    if device_platform:
        matching_tests, mismatched_tests = _partition_platform_mismatches(
            suite.tests, device_platform
        )
        if mismatched_tests and not matching_tests:
            _, mismatch_tag = mismatched_tests[0]
            return _bvt_precondition(
                "platform-mismatch",
                f"{len(mismatched_tests)} test(s) tagged {mismatch_tag} but "
                f"{device} is {device_platform}",
            )
        suite.tests = matching_tests

    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="crucible_bvt_"))
    out.mkdir(parents=True, exist_ok=True)

    runner = BVTRunner(nerve=ctx._bvt_nerve_adapter, output_dir=out)
    result = runner.run_suite(suite, device=device, app_path=app_path, budget_s=budget_s)

    # Device-readiness gate skipped the whole suite → precondition, not a run.
    if result.suite_skip_reason:
        return _bvt_precondition("device-not-ready", result.suite_skip_reason)

    # Mismatched tests never ran — report each as skipped rather than silently
    # dropping it, so total_skipped/total_run reflect the whole selected set.
    for test, mismatch_tag in mismatched_tests:
        result.tests.append(
            BVTTestResult(
                name=test.name,
                passed=False,
                package=result.package,
                skipped=True,
                skip_reason=(
                    f"platform mismatch: test tagged {mismatch_tag}, "
                    f"device {device} is {device_platform}"
                ),
            )
        )

    # Serialize BVTRunResult dataclass to dict, then add computed fields
    result_dict = asdict(result)
    result_dict["ran"] = True
    result_dict["total_run"] = result.total_run
    result_dict["total_passed"] = result.total_passed
    result_dict["total_failed"] = result.total_failed
    result_dict["total_skipped"] = result.total_skipped
    result_dict["passed"] = result.passed
    result_dict["output_dir"] = str(out)
    result_dict["total_harness_errors"] = result.total_harness_errors
    result_dict["harness_errors"] = [t.name for t in result.tests if t.harness_error]

    # Observational visual compare (never flips result.passed). Merge into the
    # dict before the single json.dumps so the block is real JSON, not a string.
    _attach_visual_compare(result_dict, result.tests, compare_against_baseline, project)

    logger.info(
        "crucible_bvt_complete",
        project=project,
        device=device,
        passed=result.passed,
        elapsed_s=round(time.monotonic() - start, 2),
    )
    return json.dumps(result_dict, default=str)


def _attach_visual_compare(
    result_dict: dict, tests: list, compare_against_baseline: bool, project: str
) -> None:
    """Attach observational visual-baseline comparison fields to ``result_dict``."""
    if not compare_against_baseline:
        result_dict["visual_advisory"] = None
        return
    from .core.bvt.visual_batch import compare_run

    screenshots = [s for t in tests for s in t.screenshots]
    try:
        visual = compare_run(ctx._baseline_manager, project, screenshots)
    except Exception as exc:
        logger.warning("crucible_bvt_visual_compare_error", error=str(exc))
        visual = {"verdict": "ERROR", "error": str(exc)}
    result_dict["visual_compare"] = visual
    result_dict["visual_advisory"] = visual.get("aggregate_verdict") or visual.get("verdict")
