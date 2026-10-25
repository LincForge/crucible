"""BVT test schema — YAML parsing and dataclasses.

Reads ``.bvt/`` directories containing YAML test definitions and produces
typed dataclasses for the runner to execute.

YAML format::

    package: com.example.app
    device: emulator-5554    # optional
    tests:
      - name: app_launches
        steps:
          - launch: "{package}"
          - wait_for_element: level_screen    # poll until present (default 5s)
          - assert_element: level_screen
          - screenshot: launch_verified

``wait_for_element`` polls the element tree until an element with the given
resource id appears, or the timeout elapses. Prefer it over ``wait: 3s`` +
``assert_element`` — it's deterministic on slow devices. With a timeout::

      - wait_for_element: {target: level_screen, timeout: 8s}   # "5s"/"500ms"/8 ok

``launch``/``navigate`` (and the ``deeplink`` alias) poll for the launched
package to reach the foreground before the step passes, with a per-step
``timeout`` override (same convention as ``wait_for_element`` — ``12``,
``"12s"``, ``"12000ms"``; clamped to 60s)::

      - launch: {target: "{package}", params: {timeout: 12s}}

With no ``timeout`` param the window is a per-test default: 5s normally, 15s
for a test tagged ``cold_start`` (a first-install boot is slower than a warm
relaunch). A step whose app never foregrounds fails with a message citing
nerve's own last-reported error (or its absence) and the poll count — never a
guessed "daemon unreachable".

Per-step ``retries`` adds flake tolerance: on a FAILED step result the runner
re-runs the whole step up to N extra times (``retries: 1`` => up to 2 total
attempts), pausing ``retry_backoff`` seconds (default 0.5) between. Params must
be NESTED under ``params:`` — a top-level ``retries:`` sibling is silently
dropped::

      - action: tap
        target: submit_btn
        params: {retries: 2, retry_backoff: 500ms}

Caveats:
  * A raised device error (daemon down) is NOT retried — only ``passed=False``.
  * Retrying re-runs side effects. Do NOT use ``retries`` with ``type`` (re-types
    the full string -> ``foofoo``) or ``long_press`` (may double-fire a menu);
    the runner warns if you do.
  * ``retries: N`` on ``wait_for_element`` multiplies the wait by ``(N+1)`` full
    timeout windows.
  * ``retries`` is clamped to 10.

Conditional branching (v1: 1 nesting level)::

      - if: {element: consent_dialog}   # or {text: "Allow"}
        then:
          - tap: accept_btn
        else: []                         # optional, defaults to []

The condition is evaluated against the current element tree at branch entry
(one fresh dump), then/else are ordinary step lists (no nested ``if``). For a
dialog that is *slow to appear*, prefer ``dismiss_dialog`` with ``optional:
true`` (it polls), or precede the ``if`` with ``wait_for_element`` — the ``if``
branches on *current* state, it does not wait.

Negative / count assertions::

      - assert_element_not_present: error_banner
      - assert_text_not_present: "Network error"
      - assert_element_count: {element: list_row, expected: 3}
      - assert_element_count: {element: list_row, min: 1, max: 5}

Negative assertions are single-shot against a fresh dump (no polling — "retry
until absent" would false-pass on a transient render). ``assert_element_count``
takes ``expected`` (exact) OR ``min``/``max``; combining ``expected`` with
``min``/``max``, or ``min > max``, is a clear error.

Cold-start gate (consent/onboarding-gated apps)::

    tests:
      - name: consent_screen_first_run     # default: kill+relaunch, keeps state
        steps: [...]
      - name: main_screen_after_consent
        tags: [cold_start]                 # `pm clear` before this test only
        steps:
          - grant_permission: android.permission.CAMERA   # re-grant — see below
          - launch: "{package}"
          - ...

A test tagged ``cold_start`` is reset with ``pm clear`` (full first-install
state) instead of the default kill+relaunch — so app-level consent/onboarding
stored in SharedPreferences/DataStore reappears. Only the tagged test is
cleared; untagged tests keep kill+relaunch.

ORDER-INDEPENDENCE: because ``pm clear`` wipes state, every test must
establish its own preconditions and never rely on leftover state from a prior
test. A ``cold_start`` test that lands you on a fresh consent screen and an
untagged test that assumes consent-already-granted can coexist in one suite
*only* if each owns its setup — otherwise the suite is order-dependent and
silently false-greens on device.

CAVEATS of ``pm clear`` (Android-only):
  * It REVOKES granted runtime permissions. A cold_start test that needs a
    permission pre-granted (camera/mic/etc.) must re-grant it as part of its
    flow via a ``grant_permission`` step::

          - grant_permission: android.permission.CAMERA

  * It REMOVES AndroidKeystore aliases. Do NOT tag a test ``cold_start`` if the
    app stores keys/credentials in the AndroidKeystore unless the flow
    re-provisions them — the wipe is silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BVTStep:
    """A single step in a BVT test."""

    action: str
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BVTTest:
    """A named BVT test with ordered steps."""

    name: str
    steps: list[BVTStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    package: str = ""


@dataclass
class BVTSuite:
    """A collection of BVT tests for a single app package."""

    package: str
    tests: list[BVTTest] = field(default_factory=list)
    device: str | None = None


def _parse_conditional(raw: dict[str, Any], package: str) -> BVTStep:
    """Parse an ``if``/``then``/``else`` conditional step (v1: 1 nesting level).

    The condition is a mapping with exactly one of ``element`` / ``text``.
    ``then``/``else`` are ordinary step lists, parsed with conditionals disabled
    (no nested ``if``). Raises ValueError on any malformed shape (AC4).
    """
    allowed = {"if", "then", "else"}
    extra = set(raw) - allowed
    if extra:
        msg = f"Unexpected keys in conditional step: {sorted(extra)}; allowed: if/then/else"
        raise ValueError(msg)

    condition = raw["if"]
    if not isinstance(condition, dict) or not condition:
        msg = "'if' condition must be a non-empty mapping, e.g. {element: dialog} or {text: Allow}"
        raise ValueError(msg)
    keys = set(condition) & {"element", "text"}
    if len(keys) != 1:
        msg = f"'if' condition must have exactly one of 'element'/'text', got: {sorted(condition)}"
        raise ValueError(msg)

    then_raw = raw.get("then", [])
    else_raw = raw.get("else", [])
    if not isinstance(then_raw, list) or not isinstance(else_raw, list):
        msg = "'then'/'else' must be lists of steps"
        raise ValueError(msg)

    if "element" in condition:
        norm = {"element": str(condition["element"]).replace("{package}", package)}
    else:
        norm = {"text": str(condition["text"])}

    then_steps = [_parse_step(s, package, allow_conditional=False) for s in then_raw]
    else_steps = [_parse_step(s, package, allow_conditional=False) for s in else_raw]
    return BVTStep(
        action="if",
        target="",
        params={"condition": norm, "then": then_steps, "else": else_steps},
    )


def _parse_step(
    raw: dict[str, Any] | str, package: str, *, allow_conditional: bool = True
) -> BVTStep:
    """Parse a single step from YAML shorthand or dict form.

    Shorthand: ``{action: target}`` e.g. ``launch: com.foo``
    Dict form: ``{action: launch, target: com.foo, params: {...}}``
    Conditional: ``{if: {element|text}, then: [...], else: [...]}`` (top-level only;
    ``allow_conditional=False`` inside a branch rejects nested ``if``).
    """
    if isinstance(raw, str):
        # Bare string like "screenshot" — action with no target
        return BVTStep(action=raw)

    if not isinstance(raw, dict):
        msg = f"Step must be a dict or string, got {type(raw).__name__}"
        raise ValueError(msg)

    # Conditional step: reserved keys if/then/else.
    if "if" in raw:
        if not allow_conditional:
            msg = "Nested 'if' steps are not allowed (v1 supports 1 branching level)"
            raise ValueError(msg)
        return _parse_conditional(raw, package)

    # Dict form with explicit 'action' key
    if "action" in raw:
        target = str(raw.get("target", "")).replace("{package}", package)
        return BVTStep(
            action=raw["action"],
            target=target,
            params=raw.get("params") or {},
        )

    return _parse_shorthand_step(raw, package)


def _parse_shorthand_step(raw: dict[str, Any], package: str) -> BVTStep:
    """Parse single-key shorthand like ``{launch: "com.foo"}``.

    A dict-valued shorthand (e.g. ``assert_min_size: {element:.., width:..}``)
    routes the mapping into params rather than stringifying it into target.
    Raises ValueError on an ambiguous multi-key dict.
    """
    if len(raw) == 1:
        action, target = next(iter(raw.items()))
        if isinstance(target, dict):
            return BVTStep(action=action, target="", params=target)
        target_str = str(target) if target is not None else ""
        target_str = target_str.replace("{package}", package)
        return BVTStep(action=action, target=target_str)

    msg = (
        f"Ambiguous step dict with keys: {list(raw.keys())}. "
        "Use single key shorthand or explicit 'action' key."
    )
    raise ValueError(msg)


def _parse_test(raw: dict[str, Any], package: str) -> BVTTest:
    """Parse a single test definition from YAML."""
    name = raw.get("name", "unnamed")
    tags = raw.get("tags", [])
    steps = [_parse_step(s, package) for s in raw.get("steps", [])]
    return BVTTest(name=name, steps=steps, tags=tags, package=package)


def load_bvt_file(path: Path, package_default: str = "") -> tuple[str, list[BVTTest], str | None]:
    """Load a single .bvt YAML file.

    The file's own ``package:`` wins; ``package_default`` (the suite-level
    fallback) is used only when the file omits a package. An empty-string
    ``package: ""`` is treated as absent. Raises ValueError at load time when
    no package can be resolved (so a no-op launch can never happen on device).

    Returns:
        Tuple of (package, tests, device).
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = f"BVT file {path} must be a YAML mapping, got {type(data).__name__}"
        raise ValueError(msg)

    package = (data.get("package") or "").strip() or package_default
    device = data.get("device")
    raw_tests = data.get("tests", [])
    if not package:
        names = [t.get("name", "unnamed") for t in raw_tests]
        msg = f"BVT file {path}: no package (set 'package:' in the file). tests={names}"
        raise ValueError(msg)
    tests = [_parse_test(t, package) for t in raw_tests]
    return package, tests, device


def load_bvt_dir(bvt_dir: Path) -> BVTSuite:
    """Load all ``.yaml`` files from a ``.bvt/`` directory into a suite.

    Files are sorted by name for deterministic ordering. The first file's
    ``package`` field is used as the suite package; subsequent files inherit
    it unless they override.

    Args:
        bvt_dir: Path to the ``.bvt/`` directory.

    Returns:
        Merged BVTSuite.

    Raises:
        FileNotFoundError: If bvt_dir doesn't exist.
        ValueError: If no YAML files found or no package defined.
    """
    if not bvt_dir.is_dir():
        msg = f"BVT directory not found: {bvt_dir}"
        raise FileNotFoundError(msg)

    yaml_files = sorted(bvt_dir.glob("*.yaml")) + sorted(bvt_dir.glob("*.yml"))
    if not yaml_files:
        msg = f"No YAML files found in {bvt_dir}"
        raise ValueError(msg)

    suite_package = ""
    suite_device: str | None = None
    all_tests: list[BVTTest] = []

    for yf in yaml_files:
        package, tests, device = load_bvt_file(yf, package_default=suite_package)
        if not suite_package and package:
            suite_package = package
        if device and not suite_device:
            suite_device = device
        all_tests.extend(tests)

    if not suite_package:
        msg = f"No 'package' field found in any BVT file in {bvt_dir}"
        raise ValueError(msg)

    return BVTSuite(package=suite_package, tests=all_tests, device=suite_device)
