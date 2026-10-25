"""Safe subprocess execution for user-controlled commands."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Mapping

import structlog

logger = structlog.get_logger()

_SHELL_METACHAR_RE = re.compile(r"[|;&]|\$\(|`")

# JDK major version probed by resolve_java_home(). Bump this alongside a fleet
# JDK upgrade — it drives both the `java_home -v` flag and the hardcoded
# Homebrew `openjdk@<N>` fallback paths below.
_JAVA_VERSION = 17

# Process sessions / groups (``start_new_session`` + ``os.killpg``) are POSIX-only.
_IS_POSIX = os.name == "posix"


class ShellMetacharError(ValueError):
    """Raised when a command string contains shell metacharacters.

    Build/test/lint commands are sourced from each project's ``.forge.yaml``
    (config-controlled, not fully trusted). Running such a string through a
    shell is a command-injection surface, so it is refused. Authors who
    genuinely need pipes or compound commands must wrap them in a committed
    script file and invoke that single executable instead.
    """


class CommandParseError(ValueError):
    """Raised when a command string cannot be parsed by shlex (e.g. unbalanced quote)."""


def _split_command(command: str) -> list[str]:
    """Split ``command`` via shlex, raising ``CommandParseError`` on ``ValueError``."""
    try:
        return shlex.split(command)
    except ValueError as exc:
        raise CommandParseError(f"Command could not be parsed (unbalanced quote): {exc}") from exc


def _run_kwargs(*, cwd: str, env: Mapping[str, str] | None) -> dict[str, object]:
    """Build the ``subprocess.Popen`` kwargs, merging ``env`` over ``os.environ``
    only when it is supplied (``None`` inherits the parent env untouched).

    On POSIX the child gets ``start_new_session=True`` so it leads its own
    session and process group: a timeout can then kill the whole tree it spawned
    (see ``_kill_process_group``) without ever touching the caller's group.

    Extracted so ``safe_subprocess_run`` carries no extra branch for the optional
    env override."""
    kwargs: dict[str, object] = {
        "shell": False,
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "start_new_session": _IS_POSIX,
    }
    if env is not None:
        kwargs["env"] = {**os.environ, **env}
    return kwargs


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Best-effort SIGKILL of the process group a timed-out child leads.

    The child was started with ``start_new_session=True``, so its pgid IS its
    pid — ``os.killpg(proc.pid, ...)`` equals ``os.killpg(os.getpgid(pid), ...)``
    but stays correct when the leader has already exited (a zombie leader's
    ``getpgid`` fails on macOS while its grandchildren still hold the group), and
    can never resolve to the caller's own group. A group that is already gone
    (``ProcessLookupError``) is swallowed. Non-POSIX hosts have no process
    groups, so only the direct child is killed there.
    """
    if not _IS_POSIX:
        proc.kill()
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


def _communicate_or_kill_group(proc: subprocess.Popen[str], timeout: float) -> tuple[str, str]:
    """Collect a child's (stdout, stderr), bounded by ``timeout`` seconds.

    On ``subprocess.TimeoutExpired`` the child's whole process group is killed
    and the child reaped before the exception is re-raised unchanged, so callers
    (e.g. ``Deployer._run_build``) keep their ``timed_out`` handling.
    """
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("subprocess_timeout_killing_group", pid=proc.pid, timeout_s=timeout)
        _kill_process_group(proc)
        proc.wait()
        raise


def safe_subprocess_run(
    command: str,
    *,
    cwd: str,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command string without ever invoking a shell.

    Args:
        command: Command string (from .forge.yaml or auto-detected).
        cwd: Working directory.
        timeout: Timeout in seconds.
        env: Optional overrides merged over ``os.environ``. When ``None``
            (default), the subprocess inherits the parent env exactly as
            before — no ``env=`` kwarg is passed to ``subprocess.Popen``.

    Returns:
        CompletedProcess result (``args``/``returncode``/``stdout``/``stderr``),
        the same shape ``subprocess.run`` returned.

    Raises:
        ShellMetacharError: If the command contains shell metacharacters
            (pipes, semicolons, ``&&``, command substitution). Such commands
            are never run through a shell — wrap them in a committed script
            file instead.
        subprocess.TimeoutExpired: If the command outlives ``timeout``. Before
            it propagates, the command's whole process group is SIGKILLed (on
            POSIX the child leads its own session), so a timed-out build takes
            its children with it — the ``gradlew`` JVM and ``xcodebuild`` die.
            The Gradle daemon is unaffected: it detaches into its own session
            and is meant to outlive the build.

    The command is always run via ``shlex.split()`` with ``shell=False``.
    ``subprocess.Popen`` + ``communicate(timeout=...)`` is used instead of
    ``subprocess.run`` because ``run`` never exposes the child once it times
    out, so its process group could not be killed.
    """
    if _SHELL_METACHAR_RE.search(command):
        logger.warning(
            "shell_metacharacters_rejected",
            command=command,
            note="shell execution is disabled — wrap pipes/compound commands in a script",
        )
        raise ShellMetacharError(
            f"Command contains shell metacharacters and was refused (shell execution "
            f"is disabled for security): {command!r}. Wrap pipes, semicolons, '&&', or "
            f"command substitution in a committed script file and invoke that script "
            f"as a single command instead."
        )

    argv = _split_command(command)
    with subprocess.Popen(argv, **_run_kwargs(cwd=cwd, env=env)) as proc:  # type: ignore[call-overload]
        stdout, stderr = _communicate_or_kill_group(proc, timeout)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _is_valid_java_home(path: str) -> bool:
    """A JAVA_HOME candidate is valid only if it contains ``bin/java``.

    A directory that exists but lacks ``bin/java`` (e.g. a stale/partial
    Homebrew keg) is treated as not-found so resolution falls through to the
    next probe rather than injecting a broken JAVA_HOME.
    """
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "bin", "java"))


def resolve_java_home() -> str | None:
    """Resolve a JAVA_HOME path for build subprocesses that run under launchd.

    launchd strips the interactive-shell PATH/JAVA_HOME, so a keg-only
    Homebrew JDK that works fine in a terminal is invisible to a
    launchd-spawned build. Probed in order, first match wins:

    1. macOS only: ``/usr/libexec/java_home -v <_JAVA_VERSION>``. A non-zero
       exit (no matching JDK registered, or the tool itself is broken under a
       stripped env) never raises — it silently falls through to the next
       probe.
    2. Hardcoded Homebrew ``openjdk@<_JAVA_VERSION>`` paths: Apple Silicon
       prefix first, then Intel. Deliberately NOT ``brew --prefix`` — brew is
       itself PATH-dependent and unavailable under launchd.
    3. ``None`` if nothing resolves.

    Never raises. Returns ``None`` immediately on any non-macOS platform.
    """
    if sys.platform != "darwin":
        return None

    candidates: list[str] = []

    try:
        proc = subprocess.run(
            ["/usr/libexec/java_home", "-v", str(_JAVA_VERSION)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            path = proc.stdout.strip()
            if path:
                candidates.append(path)
    except (OSError, subprocess.SubprocessError):
        pass

    candidates.append(
        f"/opt/homebrew/opt/openjdk@{_JAVA_VERSION}/libexec/openjdk.jdk/Contents/Home"
    )
    candidates.append(
        f"/usr/local/opt/openjdk@{_JAVA_VERSION}/libexec/openjdk.jdk/Contents/Home"
    )

    for path in candidates:
        if _is_valid_java_home(path):
            return path

    return None


def trim_output(output: str, *, head: int = 500, tail: int = 1500) -> str:
    """Trim long output keeping head and tail.

    Build tools emit errors at the end of output. Naive head-only truncation
    (``output[:N]``) loses the error signal. This keeps the first *head* chars
    (command echo / context) and the last *tail* chars (errors, stack traces).

    Args:
        output: Raw subprocess output.
        head: Characters to keep from the start.
        tail: Characters to keep from the end.

    Returns:
        Original string if within budget, or head + separator + tail.
    """
    limit = head + tail
    if len(output) <= limit:
        return output
    omitted = len(output) - limit
    return f"{output[:head]}\n\n... [{omitted} chars truncated] ...\n\n{output[-tail:]}"
