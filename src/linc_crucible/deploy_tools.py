from typing import Any

import anyio

import linc_crucible.server_context as ctx
from linc_crucible.core.tool_args import pour_args_error
from linc_crucible.server_context import _observed_tool, server


def _pour_payload(payload: dict[str, Any], build_only: bool) -> dict[str, Any]:
    """Pin the documented build_only response shape: ``build_only: True``, no ``devices``.

    The deployer already emits this for build-only pours; normalising here keeps
    the tool contract independent of which deployer path produced the result.
    """
    if build_only:
        payload.pop("devices", None)
        payload["build_only"] = True
    return payload


@server.tool()
@_observed_tool
async def crucible_pour(
    project: str,
    device: str = "",
    platform: str = "",
    launch: bool = False,
    skip_build: bool = False,
    module: str = "",
    device_type: str = "",
    launch_intent: str = "",
    build_only: bool = False,
    build_type: str = "debug",
    allow_personal: bool = False,
) -> dict[str, Any]:
    """Build, install, and optionally launch a project on fleet devices.

    USE WHEN asked to build and deploy an app to a device —
    instead of ./gradlew or xcodebuild directly.

    USE FOR: Build, install, and launch on one or more devices. Single-device and multi-device.
    Supports both Android (Gradle) and iOS (xcodebuild) builds. For KMP projects with
    an iosApp/ directory, the iOS build command is auto-detected when targeting iOS devices.
    Configure explicit iOS builds via ``ios_build_command`` in ``.forge.yaml``.
    NOT FOR: running quality checks (use crucible_assay), discovering devices (use nerve_discover),
        or post-install interaction (tapping, swiping, typing) — use linc-nerve for that.
    TIP: Use skip_build=True to reinstall an existing artifact.
    Set launch=True to open the app after install.
    Use launch_intent to open a deep-link URI on launch (e.g., "myapp://settings").
    For multi-target projects (phone + watch), crucible routes the
    correct artifact (APK or .app) to each device type automatically. Use module to deploy
    only a specific module, or device_type to filter devices.
    Set build_only=True to build without installing or launching. The result will omit
    the "devices" key and include "build_only": True.

    Personal devices: devices registered ``personal: true`` in the LINC device
    registry (device-fleet.yaml) are never targeted by default. Fan-out skips
    them and lists them in ``excluded_personal``; an explicit personal serial is
    refused before any build with ``error_category: "personal_device"``. If the
    registry cannot be read every deploying pour is refused (fail closed). A
    build_only pour touches no device, so it is never refused. Pass
    allow_personal=True only when the owner has asked for that device.

    Args:
        project: Forge project name (e.g. "myapp").
        device: Specific device serial (empty = all online).
        platform: Filter by platform ("android" or "ios").
        launch: Launch the app after installation.
        skip_build: Skip build step, use existing artifact.
        module: Deploy only this module (for multi-target projects, e.g. "watchApp").
        device_type: Filter devices by type ("phone", "watch", "tablet", "emulator").
        launch_intent: Deep-link URI to open on launch (e.g., "myapp://settings").
            Implies launch=True.
        build_only: If True, build only — skip install and launch. Result omits "devices".
        build_type: Gradle build type ("debug" or "release") for module-scoped
            Android builds. Only consulted when the build command is
            auto-derived (module= + platform="android" narrowing); an explicit
            build_command in the manifest always wins. Default "debug".
        allow_personal: Permit targeting devices registered ``personal: true``
            (and proceed when the registry is unreadable). Default False.

    Returns:
        PourResult dict: {project, success, devices: [{device_serial,
        device_model, platform, build_success, artifact_path, install_success,
        launch_success, error, module?}], artifacts?}. On failure adds error
        and/or build_error: true (build failed before any device deploy).
        build_only=True omits "devices" and adds build_only: true.
        excluded_personal: [serials] when fan-out skipped personal devices;
        a personal-device refusal returns success: false, error and
        error_category: "personal_device" with nothing built or installed.
    """
    ctx._ensure_initialized()

    args_error = pour_args_error(
        project,
        device,
        platform,
        launch,
        skip_build,
        module,
        device_type,
        launch_intent,
        build_only,
        build_type,
        allow_personal,
    )
    if args_error:
        return args_error

    # launch_intent implies launch=True (only when not build_only)
    if launch_intent and not build_only:
        if "://" not in launch_intent:
            return {
                "error": (
                    "launch_intent must be a deep-link URI such as myapp://settings; "
                    "activity launch is not supported by the nerve daemon"
                ),
                "error_category": "validation",
                "validation_error": True,
            }
        launch = True

    def _do_pour() -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "project_name": project,
            "device": device,
            "platform": platform,
            "launch": launch if not build_only else False,
            "skip_build": skip_build,
            "module": module,
            "device_type": device_type,
            "launch_intent": launch_intent if not build_only else "",
            "build_type": build_type,
            "policy": ctx._device_policy,
            "allow_personal": allow_personal,
        }
        if build_only:
            kwargs["build_only"] = True
        return _pour_payload(ctx._deployer.pour(**kwargs).to_dict(), build_only)

    return await anyio.to_thread.run_sync(_do_pour, limiter=ctx._get_thread_limiter())
