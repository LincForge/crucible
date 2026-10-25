"""Config contract — every environment variable crucible reads, in ONE pinned table.

``ENV_VARS`` is the single source of truth. ``tests/test_config_contract.py`` pins it in both
directions: against the ``os.environ.get`` / ``os.getenv`` / ``environ[`` reads under ``src/``
(a new env var cannot ship undocumented, a removed one cannot linger) and against the README
"Configuration" table, which is generated from :func:`render_env_table`
(``crucible fuelcheck --env-table``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvVar:
    """One environment variable crucible reads.

    Attributes:
        name: The variable name.
        default: Markdown-rendered default used when the variable is unset.
        purpose: One-line, markdown-rendered description of what it controls.
        optional: ``True`` when crucible runs without it (every variable today).
    """

    name: str
    default: str
    purpose: str
    optional: bool = True


_UNSET = "*(unset)*"

ENV_VARS: tuple[EnvVar, ...] = (
    EnvVar(
        "LINC_HOME",
        "`~/.linc`",
        "Root of crucible state (`$LINC_HOME/crucible/*`) and the MCP call log",
    ),
    EnvVar(
        "LINC_PROJECTS_DIR",
        "`~/projects`",
        "Directory scanned for `*/linc-manifest.yaml` project discovery",
    ),
    EnvVar(
        "LINC_WORKSPACE",
        "`$LINC_HOME/projects/LINC` (or `~/projects/LINC` when `LINC_HOME` is unset)",
        "Optional LINC workspace fallback read by `preflight` when `<project>/store/` is absent",
    ),
    EnvVar(
        "LINC_DEVICE_FLEET_FILE",
        "`$LINC_HOME/device-fleet.yaml`",
        "Personal-device registry; devices marked `personal: true` are refused unless "
        "`allow_personal=True`. A missing file means no personal devices are registered; a file "
        "that exists but cannot be read or parsed refuses EVERY device operation unless "
        "`allow_personal=True`",
    ),
    EnvVar("LINC_LOG_LEVEL", "`WARNING`", "Level of the structured logs written to stderr"),
    EnvVar(
        "LINC_BUILD_GATE",
        _UNSET,
        "`1` wraps builds in `linc-build-gate` (host build-slot mutex)",
    ),
    EnvVar(
        "LINC_CRUCIBLE_BUILD_TIMEOUT",
        "`600`",
        "Build timeout in seconds for `pour`; invalid values fall back to the default",
    ),
    EnvVar(
        "FORGE_PROJECTS_FILE",
        "`$LINC_HOME/.forge/projects.yaml`",
        "Legacy project registry, read-only; manifests discovered on disk take precedence",
    ),
    EnvVar(
        "FORGE_RUN_ID",
        "`manual`",
        "Deprecated session id, read when `CRUCIBLE_RUN_ID` is unset or empty",
    ),
    EnvVar(
        "CRUCIBLE_USAGE_LOG",
        "`redacted`",
        "Local usage log mode: `off`, `redacted` or `full`; an unknown value means `redacted` "
        "(see [Trust](https://github.com/LincForge/crucible#trust))",
    ),
    EnvVar(
        "CRUCIBLE_RUN_ID",
        "`FORGE_RUN_ID`, else `manual`",
        "Session id stamped on usage log entries; a value outside `[A-Za-z0-9._-]{1,128}` "
        "is logged as `manual`",
    ),
    EnvVar(
        "CRUCIBLE_SCORES_DIR",
        "`$LINC_HOME/crucible/scores`",
        "Quality score history",
    ),
    EnvVar(
        "CRUCIBLE_BASELINES_DIR",
        "`$LINC_HOME/crucible/baselines`",
        "Visual and a11y-tree baselines",
    ),
    EnvVar(
        "CRUCIBLE_FLEET_FILE",
        "`$LINC_HOME/crucible/fleet.yaml`",
        "Device fleet registry",
    ),
    EnvVar(
        "CRUCIBLE_IOS_TEAM_ID",
        _UNSET,
        "Apple Developer Team ID for physical-iOS device builds",
    ),
    EnvVar(
        "CRUCIBLE_MOBSF_API_KEY",
        _UNSET,
        "API key for the optional MobSF security scan (`assay` mode `full`)",
    ),
    EnvVar(
        "ANDROID_HOME",
        _UNSET,
        "Android SDK root; its `build-tools/*/aapt2` reads the package name from a built APK",
    ),
    EnvVar(
        "ANDROID_SDK_ROOT",
        _UNSET,
        "Legacy Android SDK root, used when `ANDROID_HOME` is unset",
    ),
)

_HEADER = (
    "| Environment variable | Default | Purpose |\n|----------------------|---------|---------|"
)


def env_var_names() -> frozenset[str]:
    """Return the set of documented environment variable names."""
    return frozenset(var.name for var in ENV_VARS)


def _row(var: EnvVar) -> str:
    return f"| `{var.name}` | {var.default} | {var.purpose} |"


def render_env_table(env_vars: Iterable[EnvVar] = ENV_VARS) -> str:
    """Render ``env_vars`` as the README markdown table (no trailing newline).

    Args:
        env_vars: Rows to render; defaults to :data:`ENV_VARS`.

    Returns:
        A header row, a separator row and one ``| `NAME` | default | purpose |`` row each.
    """
    rows = [_row(var) for var in env_vars]
    return "\n".join([_HEADER, *rows])
