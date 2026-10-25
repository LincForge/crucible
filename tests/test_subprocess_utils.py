"""Tests for safe_subprocess_run."""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from linc_crucible.core import subprocess_utils
from linc_crucible.core.subprocess_utils import (
    ShellMetacharError,
    _kill_process_group,
    resolve_java_home,
    safe_subprocess_run,
    trim_output,
)


class _RunBackedPopen:
    """``subprocess.Popen`` stand-in that runs the command through ``subprocess.run``.

    safe_subprocess_run spawns via Popen + communicate (so a timeout can kill the
    child's process group), but older test modules fake the build executor with
    ``@patch("subprocess.run")``. This bridge hands those fakes exactly the
    ``subprocess.run`` call the pre-Popen implementation made (same argv and
    shell/cwd/capture_output/text/timeout/env kwargs), so they keep working
    unchanged. ``pid`` is None so it can never address a real process (group).
    """

    def __init__(self, args: list[str], **kwargs: object) -> None:
        self.args = args
        self.pid = None
        self.returncode: int | None = None
        dropped = {"stdout", "stderr", "start_new_session"}
        self._kwargs = {k: v for k, v in kwargs.items() if k not in dropped}

    def __enter__(self) -> _RunBackedPopen:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        result = subprocess.run(  # the (usually patched) executor the caller fakes
            self.args, capture_output=True, timeout=timeout, **self._kwargs
        )
        self.returncode = result.returncode
        return result.stdout, result.stderr

    def wait(self) -> int | None:
        return self.returncode


class _SubprocessViaRun:
    """Proxy for ``subprocess_utils.subprocess``: Popen is the bridge, all else real."""

    Popen = _RunBackedPopen

    def __getattr__(self, name: str) -> object:
        return getattr(subprocess, name)


def bridge_safe_run_to_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route safe_subprocess_run's spawn through ``subprocess.run`` for one test.

    Only ``subprocess_utils``'s own module reference is proxied, so the real
    ``subprocess.run`` (and its real Popen) is untouched everywhere else. The
    process-group kill is disarmed: a bridged "process" has no group to kill.
    """
    monkeypatch.setattr(subprocess_utils, "subprocess", _SubprocessViaRun())
    monkeypatch.setattr(subprocess_utils, "_kill_process_group", lambda proc: None)


class TestRunBackedBridge:
    """The bridge reproduces the pre-Popen subprocess.run call and result."""

    def test_bridge_hands_run_the_legacy_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bridge_safe_run_to_subprocess_run(monkeypatch)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=2, stdout="o", stderr="e")
            result = safe_subprocess_run("make build", cwd="/project", timeout=60)
        args, kwargs = mock_run.call_args
        assert args[0] == ["make", "build"]
        assert kwargs == {
            "shell": False,
            "cwd": "/project",
            "capture_output": True,
            "text": True,
            "timeout": 60,
        }
        assert (result.returncode, result.stdout, result.stderr) == (2, "o", "e")

    def test_bridge_timeout_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bridge_safe_run_to_subprocess_run(monkeypatch)
        timeout = subprocess.TimeoutExpired(cmd="make", timeout=1)
        with (
            patch("subprocess.run", side_effect=timeout),
            pytest.raises(subprocess.TimeoutExpired),
        ):
            safe_subprocess_run("make build", cwd="/project", timeout=1)


def _fake_popen(stdout: str = "ok", stderr: str = "", returncode: int = 0) -> MagicMock:
    """A ``subprocess.Popen`` stand-in: context manager whose communicate() returns output."""
    proc = MagicMock()
    proc.__enter__.return_value = proc
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.pid = 4242
    return MagicMock(return_value=proc)


class TestSafeSubprocessRun:
    """Verify shell execution is never used and metachar commands are rejected."""

    def test_simple_command_uses_shell_false(self) -> None:
        """Simple commands should be split into a list with shell disabled."""
        with patch("subprocess.Popen", _fake_popen()) as mock_popen:
            safe_subprocess_run("./gradlew assembleDebug", cwd="/tmp", timeout=60)
        args, kwargs = mock_popen.call_args
        assert args[0] == ["./gradlew", "assembleDebug"]
        assert kwargs["shell"] is False

    @pytest.mark.parametrize(
        ("command", "label"),
        [
            ("cat foo | grep bar", "pipe"),
            ("cd /tmp; make", "semicolon"),
            ("echo $(date)", "command substitution"),
            ("make && make install", "ampersand chain"),
            ("echo `date`", "backtick substitution"),
        ],
    )
    def test_metachar_command_rejected(self, command: str, label: str) -> None:
        """Shell metacharacters are rejected outright — never run through a shell."""
        with (
            patch("subprocess.Popen", _fake_popen()) as mock_popen,
            pytest.raises(ShellMetacharError),
        ):
            safe_subprocess_run(command, cwd="/tmp", timeout=60)
        # The injection surface is closed: no process is ever spawned.
        mock_popen.assert_not_called()

    def test_rejection_message_is_actionable(self) -> None:
        """The error names the offending command and tells authors how to fix it."""
        with pytest.raises(ShellMetacharError) as exc_info:
            safe_subprocess_run("cat foo | grep bar", cwd="/tmp", timeout=60)
        message = str(exc_info.value)
        assert "cat foo | grep bar" in message
        assert "script" in message.lower()

    def test_shell_metachar_error_is_value_error(self) -> None:
        """ShellMetacharError is a ValueError so callers can catch it broadly."""
        assert issubclass(ShellMetacharError, ValueError)

    def test_passes_cwd_and_timeout(self) -> None:
        """cwd goes to Popen (captured text pipes); timeout bounds communicate()."""
        with patch("subprocess.Popen", _fake_popen()) as mock_popen:
            safe_subprocess_run("make build", cwd="/project", timeout=120)
        _, kwargs = mock_popen.call_args
        assert kwargs["cwd"] == "/project"
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["text"] is True
        mock_popen.return_value.communicate.assert_called_once_with(timeout=120)

    @pytest.mark.skipif(os.name != "posix", reason="process sessions are POSIX-only")
    def test_child_gets_its_own_session_on_posix(self) -> None:
        """POSIX: the child leads a new session/process group so a timeout can kill it all."""
        with patch("subprocess.Popen", _fake_popen()) as mock_popen:
            safe_subprocess_run("make build", cwd="/project", timeout=60)
        _, kwargs = mock_popen.call_args
        assert kwargs["start_new_session"] is True

    def test_env_none_omits_env_kwarg(self) -> None:
        """Default env=None inherits the parent env exactly as before — no env kwarg."""
        with patch("subprocess.Popen", _fake_popen()) as mock_popen:
            safe_subprocess_run("make build", cwd="/project", timeout=60)
        _, kwargs = mock_popen.call_args
        assert "env" not in kwargs

    def test_env_merges_over_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A provided env is merged over os.environ, not a replacement."""
        monkeypatch.setenv("PRE_EXISTING_VAR", "from-parent")
        with patch("subprocess.Popen", _fake_popen()) as mock_popen:
            safe_subprocess_run(
                "make build", cwd="/project", timeout=60, env={"JAVA_HOME": "/opt/jdk"}
            )
        _, kwargs = mock_popen.call_args
        assert kwargs["env"]["JAVA_HOME"] == "/opt/jdk"
        assert kwargs["env"]["PRE_EXISTING_VAR"] == "from-parent"

    def test_env_provided_metachar_still_rejected(self) -> None:
        """The shell-metachar guard fires before env handling, regardless of env."""
        with (
            patch("subprocess.Popen", _fake_popen()) as mock_popen,
            pytest.raises(ShellMetacharError),
        ):
            safe_subprocess_run(
                "cat foo | grep bar", cwd="/tmp", timeout=60, env={"JAVA_HOME": "/opt/jdk"}
            )
        mock_popen.assert_not_called()

    def test_returns_completed_process_fields(self, tmp_path) -> None:
        """Pin: a command finishing inside the timeout returns the same
        CompletedProcess fields (args/returncode/stdout/stderr) as subprocess.run did."""
        script = tmp_path / "emit.py"
        script.write_text(
            "import sys\nsys.stdout.write('out-text')\nsys.stderr.write('err-text')\nsys.exit(3)\n"
        )
        command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"

        result = safe_subprocess_run(command, cwd=str(tmp_path), timeout=30)

        assert isinstance(result, subprocess.CompletedProcess)
        assert result.args == [sys.executable, str(script)]
        assert result.returncode == 3
        assert result.stdout == "out-text"
        assert result.stderr == "err-text"


_GRANDCHILD_PARENT = """\
import pathlib
import subprocess
import sys
import time

marker, ready = sys.argv[1], sys.argv[2]
subprocess.Popen([sys.executable, "-c", "import time\\ntime.sleep(30)", marker])
pathlib.Path(ready).write_text("spawned")
time.sleep(30)
"""


def _pgrep(marker: str) -> list[int]:
    """PIDs whose full command line carries ``marker`` (only processes this test spawned)."""
    proc = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True, timeout=10)
    return [int(pid) for pid in proc.stdout.split()]


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
@pytest.mark.skipif(shutil.which("pgrep") is None, reason="pgrep not available")
class TestTimeoutKillsProcessGroup:
    """A timed-out command takes its whole process group down, grandchildren included."""

    def test_timeout_kills_grandchild(self, tmp_path) -> None:
        marker = f"crucible-pgkill-{uuid.uuid4().hex}"
        ready = tmp_path / "ready"
        script = tmp_path / "spawn_grandchild.py"
        script.write_text(_GRANDCHILD_PARENT)
        command = " ".join(
            shlex.quote(part) for part in (sys.executable, str(script), marker, str(ready))
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                safe_subprocess_run(command, cwd=str(tmp_path), timeout=0.5)
            # Guard against a vacuous pass: the grandchild must have been spawned.
            assert ready.exists(), "grandchild never spawned before the timeout"

            deadline = time.monotonic() + 2.0
            survivors = _pgrep(marker)
            while survivors and time.monotonic() < deadline:
                time.sleep(0.05)
                survivors = _pgrep(marker)
            assert survivors == [], f"processes outlived the timeout: {survivors}"
        finally:
            # Only ever reap what THIS test started (the unique marker scopes it).
            for pid in _pgrep(marker):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


class TestKillProcessGroup:
    """_kill_process_group is best-effort and never raises for a vanished group."""

    @pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
    def test_posix_kills_the_group_led_by_the_child(self) -> None:
        proc = MagicMock(pid=4242)
        with patch("linc_crucible.core.subprocess_utils.os.killpg") as mock_killpg:
            _kill_process_group(proc)
        mock_killpg.assert_called_once_with(4242, signal.SIGKILL)
        proc.kill.assert_not_called()

    @pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
    def test_posix_vanished_group_is_swallowed(self) -> None:
        proc = MagicMock(pid=4242)
        with patch(
            "linc_crucible.core.subprocess_utils.os.killpg", side_effect=ProcessLookupError
        ):
            _kill_process_group(proc)  # must not raise

    def test_non_posix_falls_back_to_killing_the_child(self) -> None:
        proc = MagicMock(pid=4242)
        with (
            patch("linc_crucible.core.subprocess_utils._IS_POSIX", False),
            patch("linc_crucible.core.subprocess_utils.os.killpg") as mock_killpg,
        ):
            _kill_process_group(proc)
        proc.kill.assert_called_once_with()
        mock_killpg.assert_not_called()

    def test_timeout_kills_group_reaps_and_reraises(self) -> None:
        """TimeoutExpired: kill the group, reap the child, re-raise for the caller."""
        fake = _fake_popen()
        proc = fake.return_value
        proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="make", timeout=1)
        with (
            patch("subprocess.Popen", fake),
            patch("linc_crucible.core.subprocess_utils._kill_process_group") as mock_kill,
            pytest.raises(subprocess.TimeoutExpired),
        ):
            safe_subprocess_run("make build", cwd="/project", timeout=1)
        mock_kill.assert_called_once_with(proc)
        proc.wait.assert_called_once_with()


class TestResolveJavaHome:
    """resolve_java_home() probes java_home, then hardcoded Homebrew paths."""

    def test_returns_none_on_non_darwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "linux")
        assert resolve_java_home() is None

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_uses_java_home_binary_when_it_succeeds(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.return_value = MagicMock(returncode=0, stdout="/some/jdk-17\n", stderr="")
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        result = resolve_java_home()

        assert result == "/some/jdk-17"
        args, _ = mock_run.call_args
        assert args[0] == ["/usr/libexec/java_home", "-v", "17"]

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_falls_back_to_apple_silicon_homebrew_path(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Unable to find")
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        result = resolve_java_home()

        assert result == "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_falls_back_to_intel_homebrew_path_when_apple_silicon_missing(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        def isdir_side_effect(path: str) -> bool:
            return "/usr/local/opt/openjdk@17" in path

        mock_isdir.side_effect = isdir_side_effect
        mock_isfile.side_effect = lambda path: "/usr/local/opt/openjdk@17" in path

        result = resolve_java_home()

        assert result == "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_java_home_binary_nonzero_exit_never_raises(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Unable to find")
        mock_isdir.return_value = False
        mock_isfile.return_value = False

        assert resolve_java_home() is None

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_java_home_binary_missing_never_raises(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.side_effect = FileNotFoundError("no such file")
        mock_isdir.return_value = False
        mock_isfile.return_value = False

        assert resolve_java_home() is None

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_directory_exists_but_missing_bin_java_falls_through(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A candidate dir that exists but lacks bin/java is not-found, not a crash."""
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.return_value = MagicMock(returncode=0, stdout="/broken/jdk\n", stderr="")
        mock_isdir.return_value = True
        mock_isfile.return_value = False

        assert resolve_java_home() is None

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_returns_none_when_nothing_found(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        mock_isdir.return_value = False
        mock_isfile.return_value = False

        assert resolve_java_home() is None

    @patch("linc_crucible.core.subprocess_utils.os.path.isfile")
    @patch("linc_crucible.core.subprocess_utils.os.path.isdir")
    @patch("linc_crucible.core.subprocess_utils.subprocess.run")
    def test_never_shells_out_to_brew(
        self,
        mock_run: MagicMock,
        mock_isdir: MagicMock,
        mock_isfile: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """brew --prefix is PATH-dependent and unavailable under launchd — never probed."""
        monkeypatch.setattr("linc_crucible.core.subprocess_utils.sys.platform", "darwin")
        mock_run.return_value = MagicMock(returncode=0, stdout="/some/jdk-17\n", stderr="")
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        resolve_java_home()

        for call in mock_run.call_args_list:
            args = call.args[0] if call.args else call.kwargs.get("args", [])
            assert "brew" not in " ".join(args)


class TestTrimOutput:
    """Verify trim_output keeps head + tail of long output."""

    def test_short_output_unchanged(self) -> None:
        """Output within budget is returned as-is."""
        short = "BUILD SUCCESSFUL in 3s"
        assert trim_output(short) == short

    def test_exactly_at_limit_unchanged(self) -> None:
        """Output exactly at head+tail limit is returned as-is."""
        exact = "x" * 2000
        assert trim_output(exact) == exact

    def test_long_output_preserves_tail(self) -> None:
        """Long output keeps the tail where BUILD FAILED lives."""
        head_content = "A" * 600
        middle = "M" * 5000
        tail_content = "T" * 400 + "\nBUILD FAILED\n"
        output = head_content + middle + tail_content
        result = trim_output(output)
        assert result.startswith("A" * 500)
        assert result.endswith("BUILD FAILED\n")
        assert "truncated" in result

    def test_separator_shows_omitted_count(self) -> None:
        """Separator reports how many chars were dropped."""
        output = "H" * 1000 + "T" * 3000
        result = trim_output(output)
        # Total 4000, budget 2000, omitted 2000
        assert "[2000 chars truncated]" in result

    def test_custom_head_tail(self) -> None:
        """Custom head/tail params are respected."""
        output = "H" * 500 + "T" * 500
        result = trim_output(output, head=100, tail=100)
        assert result.startswith("H" * 100)
        assert result.endswith("T" * 100)
        assert "truncated" in result
