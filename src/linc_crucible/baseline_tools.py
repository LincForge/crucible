from pathlib import Path
from typing import Any

import anyio

import linc_crucible.server_context as ctx
from linc_crucible.core.deployer import (
    _apply_device_policy,
    discover_for_policy,
    personal_device_error,
)
from linc_crucible.core.inventory import empty_inventory
from linc_crucible.core.store_names import name_guarded, store_error
from linc_crucible.core.tool_args import baseline_args_error
from linc_crucible.server_context import _observed_tool, server


def _empty_inventory(
    store_root: Path, project: str, screen_name: str, mode: str
) -> dict[str, Any]:
    """Honest-inventory fields for a zero-entry list result.

    Delegate onto :func:`linc_crucible.core.inventory.empty_inventory` — kept
    here so `crucible_baseline list` output stays byte-identical (pinned by
    tests/test_baseline_inventory_guard.py) after the shared logic moved.
    """
    scope = f"project '{project}'" + (f", screen '{screen_name}'" if screen_name else "")
    return empty_inventory(store_root, scope, f"{mode} baselines")


def _a11y_empty_guard(project: str, screen_name: str, allow_empty: bool) -> dict[str, Any]:
    """Result for a zero-element a11y capture.

    Loud error by default (capture almost certainly failed); a documented skip
    that stores nothing under the ``allow_empty`` platform opt-out (for devices
    that cannot read an a11y tree). Extracted as a small, fully-covered helper
    so the empty-handling branch does not inflate ``_do_a11y_baseline`` CRAP.
    """
    if not allow_empty:
        return {
            "error": (
                f"Refusing to store empty a11y baseline for "
                f"{project}/{screen_name} (0 elements) — capture likely "
                f"failed. Pass allow_empty=true to opt out on platforms "
                f"without a11y-tree support."
            )
        }
    return {
        "action": "capture",
        "mode": "a11y",
        "skipped": True,
        "stored": False,
        "reason": "a11y capture unsupported on this platform (allow_empty opt-out)",
    }


def _a11y_target(
    nerve: Any, device: str, allow_personal: bool
) -> tuple[str, dict[str, Any] | None]:
    """Resolve the a11y capture device through the personal-device policy.

    Returns ``(serial, None)`` or ``("", error_dict)``. An explicit personal
    serial (or an unreadable registry) is refused; auto-pick takes the first
    NON-personal device nerve reports.
    """
    devices = discover_for_policy(nerve) if device else nerve.discover()
    if not device and not devices:
        return "", {"error": "No devices available."}
    allowed, excluded, refusal = _apply_device_policy(
        devices, device, ctx._device_policy, allow_personal
    )
    if refusal:
        return "", personal_device_error(refusal)
    if device:
        return device, None
    if not allowed:
        error = personal_device_error(
            f"No non-personal device available (excluded personal: {', '.join(excluded)}); "
            "pass allow_personal=True to override"
        )
        return "", {**error, "excluded_personal": excluded}
    return allowed[0].get("serial", ""), None


def _a11y_capture(
    tree_capture: Any,
    project: str,
    screen_name: str,
    device: str,
    device_info: str,
    app_version: str,
    allow_empty: bool,
    allow_personal: bool = False,
) -> dict[str, Any]:
    """Capture an a11y tree baseline via nerve + tree_capture.

    Extracted from ``_do_a11y_baseline`` so the dispatcher stays a thin router
    (CRAP ratchet — never inline branches into a high-complexity function). An
    empty converted tree is rejected via ``_a11y_empty_guard``.
    """
    from .core.nerve_client import NerveClient
    from .core.nerve_convert import convert_nerve_elements

    if not screen_name:
        return {"error": "screen_name is required for a11y capture."}
    nerve = ctx._nerve_client or NerveClient()
    target_device, error = _a11y_target(nerve, device, allow_personal)
    if error:
        return error
    raw = nerve.elements(target_device)
    if not raw:
        return {"error": f"No elements from device {target_device}"}
    elements = convert_nerve_elements(raw)
    if len(elements) == 0:
        return _a11y_empty_guard(project, screen_name, allow_empty)
    entry = tree_capture.store(
        project=project,
        screen_name=screen_name,
        elements=elements,
        device_info=device_info,
        app_version=app_version,
    )
    return {"action": "capture", "mode": "a11y", "entry": entry.to_dict()}


def _do_a11y_baseline(
    action: str,
    project: str,
    screen_name: str,
    device: str,
    device_info: str,
    app_version: str,
    keep: int,
    allow_empty: bool = False,
    allow_personal: bool = False,
) -> dict[str, Any]:
    """Handle a11y tree baseline operations.

    Captures, retrieves, lists, and prunes a11y tree baselines via nerve + tree_capture.
    Called from crucible_baseline when mode="a11y".
    """
    from .core.tree_capture import TreeCapture

    tree_capture = ctx._tree_capture or TreeCapture()

    if action == "capture":
        return _a11y_capture(
            tree_capture,
            project,
            screen_name,
            device,
            device_info,
            app_version,
            allow_empty,
            allow_personal,
        )

    elif action == "latest":
        if not screen_name:
            return {"error": "screen_name is required for latest."}
        entry = tree_capture.get_latest(project, screen_name)
        if not entry:
            return {
                "action": "latest",
                "mode": "a11y",
                "entry": None,
                "message": "No a11y baselines found.",
            }
        return {"action": "latest", "mode": "a11y", "entry": entry.to_dict()}

    elif action == "list":
        entries = tree_capture.list_entries(project, screen_name=screen_name)
        result = {
            "action": "list",
            "mode": "a11y",
            "count": len(entries),
            "entries": [e.to_dict() for e in entries],
        }
        if not entries:
            result.update(
                _empty_inventory(tree_capture.baselines_dir, project, screen_name, mode="a11y")
            )
        return result

    elif action == "prune":
        pruned = tree_capture.prune(project, screen_name=screen_name, keep=keep)
        return {"action": "prune", "mode": "a11y", "pruned": pruned, "keep": keep}

    else:
        return {"error": f"Unknown action: {action}"}


@server.tool()
@_observed_tool
async def crucible_baseline(
    action: str,
    project: str,
    screen_name: str = "",
    image_path: str = "",
    device_info: str = "",
    app_version: str = "",
    keep: int = 5,
    mode: str = "screenshot",
    device: str = "",
    source_score: float | None = None,
    allow_empty: bool = False,
    allow_personal: bool = False,
) -> dict[str, Any]:
    """Manage baselines for visual screenshot and a11y tree regression detection.

    USE WHEN asked to capture or compare golden screenshots or accessibility-tree
    baselines for regression detection.

    USE FOR: Capturing golden screenshots or a11y tree baselines, retrieving the
        latest baseline, listing baseline history, and pruning old baselines.
    NOT FOR: Comparing screenshots (use crucible_bvt compare_against_baseline), taking live
        screenshots (use nerve_screenshot), quality scoring (use crucible_score).
    TIP: Use mode="screenshot" (default) for visual baselines. Use mode="a11y"
        to manage live accessibility tree baselines captured via nerve.
        action="capture" for screenshots requires image_path and screen_name.
        action="capture" for a11y fetches elements from a connected device via nerve.
        action="latest" returns the newest baseline for a project+screen.
        action="list" shows all baselines; filter with screen_name.
        action="prune" deletes old baselines, keeping the newest `keep` (default 5).
        action="promote" installs a candidate image_path as the new golden with
            provenance (source="promoted", source_score, promoted_at). Explicit
            only — a passing compare never auto-promotes.

    Personal devices: a11y capture never reads a device registered
    ``personal: true`` in the LINC device registry (device-fleet.yaml).
    Auto-discovery picks the first non-personal device; an explicit personal
    serial is refused with ``error_category: "personal_device"``, as is every
    capture when the registry is unreadable (fail closed). Pass
    allow_personal=True only when the owner has asked for that device.

    Args:
        action: "capture", "latest", "list", "prune", or "promote".
        project: Project name.
        screen_name: Screen identifier (required for capture/latest/promote).
        image_path: Path to PNG file (required for screenshot capture/promote).
        device_info: Device description (for capture).
        app_version: App version string (for capture).
        keep: Baselines to keep per screen when pruning (default 5).
        mode: "screenshot" (default) or "a11y". Selects which baseline type to manage.
        device: Device serial for a11y capture (auto-discovers if empty).
        source_score: SSIM score to record as provenance when action="promote".
        allow_empty: a11y opt-out. When True, a zero-element capture is a
            documented skip (stores nothing) instead of a loud error — for
            platforms that cannot read an a11y tree.
        allow_personal: a11y capture only. Permit reading a device registered
            ``personal: true`` (and proceed when the registry is unreadable).
            Default False.

    Returns:
        Action-shaped dict: capture/promote → the stored baseline entry
        (project, screen, path, provenance); latest → the newest entry;
        list → {count, entries: [...]} — a zero-entry list additionally
        carries honest-inventory fields (empty=true, store_root,
        store_exists, message) so an absent/empty store is explicit rather
        than a vacuous success; prune → {pruned: N, kept: N}.
        Failures return {error: <message>} instead of raising. A project or
        screen_name outside ``[A-Za-z0-9._-]`` (absolute path, ``..``, leading
        ``.``/``-``, over 128 chars) returns ``validation_error: True`` with
        ``error_category: "validation"`` and touches no file.
    """
    ctx._ensure_initialized()

    # project/screen_name become store path components: refuse anything off the
    # [A-Za-z0-9._-] allowlist with validation_error=True (CLI exit 2).
    args_error = baseline_args_error(
        action,
        project,
        screen_name,
        image_path,
        device_info,
        app_version,
        keep,
        mode,
        device,
        source_score,
        allow_empty,
        allow_personal,
    )
    if args_error:
        return args_error

    if mode == "a11y":

        def _do_a11y() -> dict[str, Any]:
            return _do_a11y_baseline(
                action=action,
                project=project,
                screen_name=screen_name,
                device=device,
                device_info=device_info,
                app_version=app_version,
                keep=keep,
                allow_empty=allow_empty,
                allow_personal=allow_personal,
            )

        return await anyio.to_thread.run_sync(
            name_guarded, _do_a11y, limiter=ctx._get_thread_limiter()
        )

    def _do_baseline() -> dict[str, Any]:
        if action == "capture":
            if not screen_name:
                return {"error": "screen_name is required for capture."}
            if not image_path:
                return {"error": "image_path is required for capture."}
            try:
                entry = ctx._baseline_manager.capture(
                    project=project,
                    screen_name=screen_name,
                    image_path=Path(image_path),
                    device_info=device_info,
                    app_version=app_version,
                )
                return {"action": "capture", "entry": entry.to_dict()}
            except (FileNotFoundError, ValueError) as exc:
                return store_error(exc)

        elif action == "latest":
            if not screen_name:
                return {"error": "screen_name is required for latest."}
            entry = ctx._baseline_manager.get_latest(project, screen_name)
            if not entry:
                return {
                    "action": "latest",
                    "project": project,
                    "screen_name": screen_name,
                    "entry": None,
                    "message": "No baselines found.",
                }
            return {"action": "latest", "entry": entry.to_dict()}

        elif action == "list":
            entries = ctx._baseline_manager.list_baselines(project, screen_name=screen_name)
            result = {
                "action": "list",
                "project": project,
                "count": len(entries),
                "entries": [e.to_dict() for e in entries],
            }
            if not entries:
                result.update(
                    _empty_inventory(
                        ctx._baseline_manager.baselines_dir,
                        project,
                        screen_name,
                        mode="screenshot",
                    )
                )
            return result

        elif action == "prune":
            pruned = ctx._baseline_manager.prune(project, screen_name=screen_name, keep=keep)
            return {
                "action": "prune",
                "project": project,
                "pruned": pruned,
                "keep": keep,
            }

        elif action == "promote":
            if not screen_name:
                return {"error": "screen_name is required for promote."}
            if not image_path:
                return {"error": "image_path is required for promote."}
            try:
                entry = ctx._baseline_manager.promote(
                    project=project,
                    screen_name=screen_name,
                    image_path=Path(image_path),
                    source_score=source_score,
                    device_info=device_info,
                    app_version=app_version,
                )
                return {"action": "promote", "entry": entry.to_dict()}
            except (FileNotFoundError, ValueError) as exc:
                return store_error(exc)

        else:
            return {
                "error": (
                    f"Unknown action: {action}. Use capture, latest, list, prune, or promote."
                )
            }

    return await anyio.to_thread.run_sync(
        name_guarded, _do_baseline, limiter=ctx._get_thread_limiter()
    )
