"""Tests for assayer orchestrator."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog

from linc_crucible.core.assayer import (
    A11yTreeResult,
    AccessibilityResult,
    Assayer,
    AssayReport,
    LintResult,
    PerformanceResult,
    RunResult,
    SecurityResult,
    _count_lint_issues,
    _exit_contradicts_summary,
    _parse_test_results,
    _pytest_counts,
)
from linc_crucible.core.forge_integration import ForgeProject, ForgeProjectReader
from linc_crucible.core.nerve_client import UiTreeUnavailableError
from linc_crucible.core.scorer import QualityScorer
from tests.test_subprocess_utils import bridge_safe_run_to_subprocess_run


@pytest.fixture(autouse=True)
def _capture_structlog() -> Iterator[None]:
    """Record structlog events instead of printing them to an import-time-bound stream.

    Importing ``linc_crucible.cli`` points structlog at the then-current ``sys.stderr``,
    a pytest capture stream. mutmut re-runs ``pytest.main()`` in one process; the first
    session closes that stream, so any later log call raised "I/O operation on closed
    file". Mirrors ``captured_logs`` in ``tests/test_crucible_bvt_mcp.py``.
    """
    with structlog.testing.capture_logs():
        yield


@pytest.fixture(autouse=True)
def _safe_run_via_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module fakes builds with @patch("subprocess.run"); safe_subprocess_run now
    spawns via Popen (linc-crucible-64q.5), so bridge its spawn back to that seam."""
    bridge_safe_run_to_subprocess_run(monkeypatch)


@pytest.fixture
def mock_forge() -> MagicMock:
    forge = MagicMock(spec=ForgeProjectReader)
    forge.get.return_value = ForgeProject(
        name="TestProject",
        path="/tmp/test-project",
        test_command="pytest tests/ -v",
        lint_command="ruff check src/",
        framework="python_setup",
    )
    forge.find_apk.return_value = None
    return forge


@pytest.fixture
def scorer(tmp_path: Path) -> QualityScorer:
    return QualityScorer(scores_dir=tmp_path / "scores")


class TestCountLintIssues:
    def test_ruff_output(self) -> None:
        output = (
            "src/main.py:10:5: E302 expected 2 blank lines, got 1\n"
            "src/main.py:15:1: F401 'os' imported but unused\n"
            "src/utils.py:3:1: E501 line too long\n"
        )
        assert _count_lint_issues(output, "python_setup") == 3

    def test_empty_output(self) -> None:
        assert _count_lint_issues("", "python_setup") == 0

    def test_clean_output(self) -> None:
        assert _count_lint_issues("All checks passed!", "python_setup") == 0

    def test_gradle_warnings(self) -> None:
        output = "w: /src/Main.kt: warning\nw: /src/Other.kt: warning\ne: /src/Bad.kt: error"
        assert _count_lint_issues(output, "gradle_kmp") == 3

    def test_generic_warnings(self) -> None:
        output = "file.py:10: Warning: something\nfile.py:20: Error: bad"
        assert _count_lint_issues(output, "unknown") == 2


class TestParseRunResults:
    def test_pytest_all_passed(self) -> None:
        output = "====== 54 passed in 12.5s ======"
        total, passed, failed, skipped = _parse_test_results(output, "python_setup")
        assert total == 54
        assert passed == 54
        assert failed == 0
        assert skipped == 0

    def test_pytest_mixed(self) -> None:
        output = "====== 10 passed, 3 failed, 2 skipped in 5.0s ======"
        total, passed, failed, skipped = _parse_test_results(output, "python_setup")
        assert total == 15
        assert passed == 10
        assert failed == 3
        assert skipped == 2

    def test_pytest_failed_only(self) -> None:
        output = "====== 5 passed, 2 failed in 3.0s ======"
        total, passed, failed, _skipped = _parse_test_results(output, "python_setup")
        assert total == 7
        assert passed == 5
        assert failed == 2

    def test_pytest_failed_first(self) -> None:
        output = "====== 3 failed, 10 passed, 1 skipped in 2.1s ======"
        total, passed, failed, skipped = _parse_test_results(output, "python_setup")
        assert (total, passed, failed, skipped) == (14, 10, 3, 1)

    def test_pytest_failed_passed_short(self) -> None:
        output = "1 failed, 1 passed in 0.01s"
        total, passed, failed, skipped = _parse_test_results(output, "python_setup")
        assert (total, passed, failed, skipped) == (2, 1, 1, 0)

    def test_pytest_single_error(self) -> None:
        output = "2 passed, 1 error in 0.5s"
        total, passed, failed, skipped = _parse_test_results(output, "python_setup")
        assert failed == 1
        assert total == 3
        assert passed == 2
        assert skipped == 0

    def test_pytest_multiple_errors(self) -> None:
        output = "2 errors in 0.5s"
        total, passed, failed, skipped = _parse_test_results(output, "python_setup")
        assert failed == 2
        assert total == 2
        assert passed == 0
        assert skipped == 0

    def test_gradle_results(self) -> None:
        output = "Tests run: 20, Failures: 1, Errors: 0, Skipped: 2"
        total, passed, failed, skipped = _parse_test_results(output, "gradle_kmp")
        assert total == 20
        assert passed == 17
        assert failed == 1
        assert skipped == 2

    def test_kotlin_test(self) -> None:
        output = "50 tests completed, 3 failed"
        total, passed, failed, _skipped = _parse_test_results(output, "gradle_kmp")
        assert total == 50
        assert passed == 47
        assert failed == 3

    def test_empty_output(self) -> None:
        total, _passed, _failed, _skipped = _parse_test_results("", "python_setup")
        assert total == 0


class TestPytestCounts:
    """Direct per-branch unit tests for _pytest_counts helper (mutmut scope)."""

    def test_empty_and_no_match(self) -> None:
        assert _pytest_counts("") == (0, 0, 0, 0)
        assert _pytest_counts("Random text without test summary") == (0, 0, 0, 0)

    def test_passed_token(self) -> None:
        assert _pytest_counts("10 passed") == (10, 10, 0, 0)

    def test_failed_token(self) -> None:
        assert _pytest_counts("3 failed") == (3, 0, 3, 0)

    def test_error_token_singular(self) -> None:
        assert _pytest_counts("1 error") == (1, 0, 1, 0)

    def test_errors_token_plural(self) -> None:
        assert _pytest_counts("2 errors") == (2, 0, 2, 0)

    def test_skipped_token(self) -> None:
        assert _pytest_counts("4 skipped") == (4, 0, 0, 4)

    def test_xfailed_token_total_only(self) -> None:
        assert _pytest_counts("2 xfailed") == (2, 0, 0, 0)

    def test_xpassed_token_total_only(self) -> None:
        assert _pytest_counts("3 xpassed") == (3, 0, 0, 0)

    def test_all_tokens_combined(self) -> None:
        # passed=5, failed=2+1+3=6, skipped=4, xfailed=1 (total only), xpassed=2 (total only)
        # total = 5 + 6 + 4 + 1 + 2 = 18
        output = "5 passed, 2 failed, 1 error, 3 errors, 4 skipped, 1 xfailed, 2 xpassed"
        assert _pytest_counts(output) == (18, 5, 6, 4)

    def test_collection_error_not_parsed_as_test(self) -> None:
        assert _pytest_counts("!!!! Interrupted: 1 error during collection !!!!") == (0, 0, 0, 0)


class TestExitContradictsSummary:
    """Direct per-branch unit tests for _exit_contradicts_summary predicate (mutmut scope)."""

    def test_nonzero_exit_with_zero_failed_is_contradiction(self) -> None:
        assert _exit_contradicts_summary(1, 10, 0) is True
        assert _exit_contradicts_summary(2, 0, 0) is True

    def test_zero_exit_with_zero_failed_is_not_contradiction(self) -> None:
        assert _exit_contradicts_summary(0, 10, 0) is False
        assert _exit_contradicts_summary(0, 0, 0) is False

    def test_nonzero_exit_with_failures_is_not_contradiction(self) -> None:
        assert _exit_contradicts_summary(1, 10, 1) is False
        assert _exit_contradicts_summary(1, 10, 3) is False

    def test_zero_exit_with_failures_is_not_contradiction(self) -> None:
        assert _exit_contradicts_summary(0, 10, 1) is False


class TestAssayer:
    def test_assay_project_not_found(self, scorer: QualityScorer) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = None
        assayer = Assayer(forge_reader=forge, scorer=scorer)

        report = assayer.assay("NonExistent")
        assert report.tests.success is False

    @patch("subprocess.run")
    def test_assay_quick_mode(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        # Lint returns clean
        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""

        # Tests return all passing
        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 20 passed in 3.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", mode="quick")

        assert report.mode == "quick"
        assert report.lint.success is True
        assert report.lint.issues == 0
        assert report.tests.total == 20
        assert report.tests.passed == 20
        assert report.tests.pass_rate == 1.0
        assert report.security is None
        assert report.quality_score is not None
        assert report.quality_score.composite == 100.0
        assert report.quality_score.ship_gate is True

    @patch("subprocess.run")
    def test_assay_with_lint_issues(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock()
        lint_result.returncode = 1
        lint_result.stdout = (
            "src/main.py:10:5: E302 expected 2 blank lines\nsrc/main.py:15:1: F401 unused import\n"
        )
        lint_result.stderr = ""

        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 10 passed in 1.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject")

        assert report.lint.issues == 2
        assert report.lint.success is False
        # tests: 100 * 0.55 = 55, lint: (100-4) * 0.45 = 43.2
        assert report.quality_score.composite == 98.2

    @patch("subprocess.run")
    def test_assay_quick_red_test_run_fails_ship_gate(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(
            returncode=1, stdout="====== 3 failed, 10 passed, 1 skipped in 2.1s ======", stderr=""
        )
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", mode="quick")

        assert report.tests.score_input is not None
        assert report.tests.score_input < 0.8
        assert report.quality_score is not None
        assert report.quality_score.ship_gate is False
        assert report.quality_score.composite != 95.0

    @patch("subprocess.run")
    def test_assay_quick_exit_contradicts_summary_unmeasured(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=1, stdout="10 passed in 1s", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", mode="quick")

        assert report.tests.attempted_unmeasured is True
        assert report.tests.score_input is None
        assert report.quality_score is not None
        assert report.quality_score.ship_gate is False
        assert "unmeasured" in report.quality_score.mode

    @patch("subprocess.run")
    def test_assay_quick_returncode_zero_clean_pass_pin(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="10 passed", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", mode="quick")

        assert report.tests.total == 10
        assert report.tests.passed == 10
        assert report.tests.failed == 0
        assert report.tests.pass_rate == 1.0
        assert report.tests.score_input == 1.0
        assert report.tests.attempted_unmeasured is False
        assert report.quality_score is not None
        assert report.quality_score.ship_gate is True

    @patch("subprocess.run")
    def test_assay_no_test_command(
        self, mock_subprocess: MagicMock, scorer: QualityScorer
    ) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="NoTests",
            path="/tmp/no-tests",
            lint_command="",
            test_command="",
        )

        assayer = Assayer(forge_reader=forge, scorer=scorer)
        report = assayer.assay("NoTests")

        # No subprocess calls
        mock_subprocess.assert_not_called()
        assert report.quality_score is not None
        # Neither lint nor tests ran — the assay measured nothing, so it must
        # NOT read as a false-green 100. "nothing checked" != "everything passed".
        assert report.lint.configured is False
        assert report.tests.configured is False
        assert report.quality_score.composite == 0.0
        assert report.quality_score.ship_gate is False
        assert report.quality_score.mode == "unmeasured"
        assert report.quality_score.dimensions == []

    @patch("subprocess.run")
    def test_assay_broken_toolchain_does_not_ship_gate(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """linc-crucible#136 end-to-end: lint AND test commands are configured
        (unlike test_assay_no_test_command above) but the binaries don't
        exist. Before the fix this reads composite=100.0/ship_gate=True with
        zero checks executed — a store release approved by a gate that
        measured nothing. After the fix it must read as unmeasured/not
        shippable, exactly like the no-command-configured case.
        """
        mock_subprocess.side_effect = FileNotFoundError("no such file or directory")

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", mode="quick")

        assert report.lint.configured is False
        assert report.tests.configured is False
        assert report.quality_score is not None
        assert report.quality_score.composite == 0.0
        assert report.quality_score.ship_gate is False
        assert report.quality_score.mode == "unmeasured"

    @patch("subprocess.run")
    def test_assay_mixed_broken_toolchain_with_green_sibling_does_not_ship(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """linc-crucible#136 finding 2 (MINOS round 0, BLOCKER): a broken
        toolchain must never ship just because a SIBLING dimension happened
        to be green. Lint runs clean; the test binary is missing
        (`FileNotFoundError`). Before this fix the crashed `tests` dimension
        was simply EXCLUDED — indistinguishable from "no test command
        configured" — so `lint` alone renormalised to weight 1.0 and
        `compute()` reported `composite=100.0, ship_gate=True, mode='quick'`,
        with nothing downstream able to tell the test suite never ran.
        """
        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All checks passed!"
        lint_result.stderr = ""

        mock_subprocess.side_effect = [
            lint_result,
            FileNotFoundError("no such file or directory"),
        ]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", mode="quick")

        assert report.lint.success is True
        assert report.lint.score_input == 0
        assert report.tests.configured is False
        assert report.tests.attempted_unmeasured is True
        assert report.quality_score is not None
        # The composite over the surviving `lint` dimension alone would still
        # be 100.0 — the point of this test is that ship_gate is forced False
        # regardless.
        assert report.quality_score.composite == 100.0
        assert report.quality_score.ship_gate is False
        assert "partial-unmeasured" in report.quality_score.mode
        assert report.quality_score.attempted_unmeasured == ["tests"]


class TestRunLintTestsShellMetacharRejection:
    """A metachar-containing command degrades to a failed result, not a crash."""

    def test_run_lint_rejects_metachar_command(self, scorer: QualityScorer) -> None:
        """A piped lint_command returns a failed LintResult with the actionable error."""
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="Pipey",
            path="/tmp/pipey",
            lint_command="ruff check src/ | tee lint.log",
            framework="python_setup",
        )

        result = assayer._run_lint(project)

        assert result.success is False
        assert "metacharacters" in result.output
        # A rejected/crashed lint run is UNMEASURED, not a clean pass — it must
        # not masquerade as "0 issues found". configured=False excludes it from
        # scoring instead of letting the scorer read a perfect lint score.
        assert result.configured is False
        assert result.score_input is None
        # A command WAS configured and invoked (it was refused, not skipped) —
        # `attempted` must stay True so attempted_unmeasured can distinguish
        # this from "no lint command configured" (linc-crucible#136 finding 2).
        assert result.attempted is True
        assert result.attempted_unmeasured is True

    def test_run_tests_rejects_metachar_command(self, scorer: QualityScorer) -> None:
        """A compound test_command returns a failed RunResult with the actionable error."""
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="Chainy",
            path="/tmp/chainy",
            test_command="pytest && echo done",
            framework="python_setup",
        )

        result = assayer._run_tests(project)

        assert result.success is False
        assert "metacharacters" in result.output
        # Same guard on the test path: a refused/crashed run is UNMEASURED, not
        # a perfect pass_rate. configured=False is what keeps score_input from
        # returning the vacuous 1.0 that `total == 0` produces.
        assert result.configured is False
        assert result.score_input is None
        assert result.attempted is True
        assert result.attempted_unmeasured is True


class TestRunLintTestsCrashExcludedFromScoring:
    """`_run_lint`/`_run_tests` exception branches must mark the result
    unmeasured (``configured=False``), not leave the ``True`` default —
    otherwise a crashed toolchain scores as a measured perfect pass.

    linc-crucible#136: `crucible_assay` returned `composite=100.0,
    ship_gate=True` on a broken/absent toolchain because the except branch in
    both `_run_lint` and `_run_tests` set `success=False` but never touched
    `configured`, so `score_input` fell through to `pass_rate` /
    `issues` — a "measured and clean" reading of a run that never happened.
    """

    def test_run_tests_filenotfound_is_excluded_from_scoring(self, scorer: QualityScorer) -> None:
        """A test command whose binary is missing must not score as pass_rate=1.0."""
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="MissingTestBinary",
            path="/tmp/missing-test-binary",
            test_command="nonexistent-test-runner",
            framework="python_setup",
        )

        with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
            result = assayer._run_tests(project)

        assert result.success is False
        # THE BUG: on unfixed code, `configured` stays at its True default here,
        # so `score_input` returns `pass_rate` (1.0, vacuous at total==0) instead
        # of None. This assertion is the one that fails on current main.
        assert result.configured is False
        assert result.score_input is None
        assert result.attempted is True
        assert result.attempted_unmeasured is True

    def test_run_lint_filenotfound_is_excluded_from_scoring(self, scorer: QualityScorer) -> None:
        """A lint command whose binary is missing must not score as 0 issues."""
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="MissingLintBinary",
            path="/tmp/missing-lint-binary",
            lint_command="nonexistent-linter",
            framework="python_setup",
        )

        with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
            result = assayer._run_lint(project)

        assert result.success is False
        assert result.configured is False
        assert result.score_input is None
        assert result.attempted is True
        assert result.attempted_unmeasured is True

    def test_run_tests_timeout_is_excluded_from_scoring(self, scorer: QualityScorer) -> None:
        """A hung test run (timeout) is unmeasured, not a pass."""
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="HangingTests",
            path="/tmp/hanging-tests",
            test_command="pytest",
            framework="python_setup",
        )

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=300),
        ):
            result = assayer._run_tests(project)

        assert result.success is False
        assert result.configured is False
        assert result.score_input is None
        assert result.attempted is True
        assert result.attempted_unmeasured is True


class TestRunLintTestsWidenedExceptionSurface:
    """linc-crucible#136 finding 5: `safe_subprocess_run` -> `subprocess.run`
    can raise `PermissionError` (binary present but not executable) and
    `NotADirectoryError` (a `.forge.yaml` path pointing at a file) — both
    `OSError` subclasses — and `shlex.split()` can raise `ValueError` on an
    unbalanced quote in a configured command. None of these were in the
    original `(TimeoutExpired, FileNotFoundError, ShellMetacharError)` tuple,
    so they propagated out of `assay()` and aborted the whole tool call
    instead of degrading to an unmeasured result like every other crash
    shape. Widening to `(TimeoutExpired, OSError, ShellMetacharError,
    ValueError)` catches them (`FileNotFoundError` is already an `OSError`
    subclass, so this strictly widens, not narrows)."""

    def test_run_tests_permission_error_is_unmeasured(self, scorer: QualityScorer) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="NonExecutableBinary",
            path="/tmp/non-executable",
            test_command="pytest",
            framework="python_setup",
        )

        with patch("subprocess.run", side_effect=PermissionError("permission denied")):
            result = assayer._run_tests(project)

        assert result.success is False
        assert result.configured is False
        assert result.attempted is True
        assert result.score_input is None
        assert "permission denied" in result.output

    def test_run_lint_not_a_directory_error_is_unmeasured(self, scorer: QualityScorer) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="CwdIsAFile",
            path="/tmp/cwd-is-a-file",
            lint_command="ruff check src/",
            framework="python_setup",
        )

        with patch(
            "subprocess.run",
            side_effect=NotADirectoryError("not a directory"),
        ):
            result = assayer._run_lint(project)

        assert result.success is False
        assert result.configured is False
        assert result.attempted is True
        assert result.score_input is None

    def test_run_tests_value_error_from_bad_quoting_is_unmeasured(
        self, scorer: QualityScorer
    ) -> None:
        """`shlex.split()` inside `safe_subprocess_run` raises `ValueError` on
        an unbalanced quote in a `.forge.yaml`-configured command."""
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="BadQuoting",
            path="/tmp/bad-quoting",
            test_command='pytest "unterminated',
            framework="python_setup",
        )

        result = assayer._run_tests(project)

        assert result.success is False
        assert result.configured is False
        assert result.attempted is True
        assert result.score_input is None


class TestScoreInputExcludesUnparseableNonExceptionCrashes:
    """linc-crucible#136 finding 1 (MINOS round 0, the dominant crash path):
    the tool RUNS, exits non-zero, and emits nothing `_parse_test_results` /
    `_count_lint_issues` can parse. This path never raises, so `configured`
    stays at its `True` default — before this fix, `score_input` still fell
    through to the vacuous `total == 0` `pass_rate` of 1.0. One test per
    shape MINOS measured end-to-end against the unfixed PR head."""

    def test_build_failed_exception_rc1_is_unmeasured(self, scorer: QualityScorer) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="BuildBroken",
            path="/tmp/build-broken",
            test_command="./gradlew test",
            framework="gradle_kmp",
        )
        proc = MagicMock()
        proc.returncode = 1
        proc.stdout = "FAILURE: Build failed with an exception."
        proc.stderr = ""

        with patch("subprocess.run", return_value=proc):
            result = assayer._run_tests(project)

        assert result.total == 0
        assert result.success is False
        assert result.configured is True  # a command WAS configured
        assert result.attempted is True
        assert result.score_input is None
        assert result.attempted_unmeasured is True

    def test_collection_error_rc2_is_unmeasured(self, scorer: QualityScorer) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="CollectionError",
            path="/tmp/collection-error",
            test_command="pytest",
            framework="python_setup",
        )
        proc = MagicMock()
        proc.returncode = 2
        proc.stdout = "!!!! Interrupted: 1 error during collection !!!!"
        proc.stderr = ""

        with patch("subprocess.run", return_value=proc):
            result = assayer._run_tests(project)

        assert result.total == 0
        assert result.success is False
        assert result.score_input is None
        assert result.attempted_unmeasured is True

    def test_no_tests_ran_rc5_is_unmeasured(self, scorer: QualityScorer) -> None:
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="NoTestsCollected",
            path="/tmp/no-tests-collected",
            test_command="pytest -k nonexistent_marker",
            framework="python_setup",
        )
        proc = MagicMock()
        proc.returncode = 5
        proc.stdout = "no tests ran in 0.01s"
        proc.stderr = ""

        with patch("subprocess.run", return_value=proc):
            result = assayer._run_tests(project)

        assert result.total == 0
        assert result.success is False
        assert result.score_input is None
        assert result.attempted_unmeasured is True

    def test_lint_rc127_no_issue_lines_is_unmeasured(self, scorer: QualityScorer) -> None:
        """A linter exiting 127 (command not found via shell, or a broken
        wrapper script) whose output has no `error:`/`warning:` lines yields
        `issues=0` from `_count_lint_issues` — that must NOT read as a
        measured, perfectly clean lint pass."""
        forge = MagicMock(spec=ForgeProjectReader)
        assayer = Assayer(forge_reader=forge, scorer=scorer)
        project = ForgeProject(
            name="BrokenLintWrapper",
            path="/tmp/broken-lint-wrapper",
            lint_command="./lint.sh",
            framework="python_setup",
        )
        proc = MagicMock()
        proc.returncode = 127
        proc.stdout = ""
        proc.stderr = "./lint.sh: line 3: ruff: command not found"

        with patch("subprocess.run", return_value=proc):
            result = assayer._run_lint(project)

        assert result.issues == 0
        assert result.success is False
        assert result.configured is True
        assert result.attempted is True
        assert result.score_input is None
        assert result.attempted_unmeasured is True


class TestAssayReport:
    def test_to_dict(self) -> None:
        report = AssayReport(
            project="Test",
            mode="quick",
            lint=LintResult(issues=5, success=True),
            tests=RunResult(total=10, passed=8, failed=2, success=True),
        )
        d = report.to_dict()
        assert d["project"] == "Test"
        assert d["lint"]["issues"] == 5
        assert d["tests"]["pass_rate"] == 0.8

    def test_to_dict_surfaces_configured_and_output_for_crashed_dimensions(self) -> None:
        """linc-crucible#136 finding 3: the exact payload `crucible_assay`
        returns to an agent must be able to tell a measured "0 issues" apart
        from a crashed run that never measured anything — before this fix,
        `configured`/`attempted_unmeasured` never left `core/assayer.py` at
        all, so the only hint in the returned payload was the pre-existing
        `success: false`.
        """
        report = AssayReport(
            project="Test",
            mode="quick",
            lint=LintResult(
                output="ruff: command not found",
                success=False,
                configured=False,
                attempted=True,
            ),
            tests=RunResult(
                output="pytest: command not found",
                success=False,
                configured=False,
                attempted=True,
            ),
        )
        d = report.to_dict()
        assert d["lint"]["configured"] is False
        assert d["lint"]["attempted_unmeasured"] is True
        assert d["lint"]["output"] == "ruff: command not found"
        assert d["tests"]["configured"] is False
        assert d["tests"]["attempted_unmeasured"] is True
        assert d["tests"]["output"] == "pytest: command not found"
        assert d["tests"]["pass_rate"] is None
        assert d["measured"] is False
        assert d["status"] == "unmeasured"

    def test_to_dict_measured_dimensions_are_not_flagged_unmeasured(self) -> None:
        """Regression guard: a clean, measured run must NOT read as
        attempted-but-unmeasured just because `configured`/`attempted` fields
        now exist on the payload."""
        report = AssayReport(
            project="Test",
            mode="quick",
            lint=LintResult(issues=0, output="All checks passed!", success=True, attempted=True),
            tests=RunResult(total=10, passed=10, output="10 passed", success=True, attempted=True),
        )
        d = report.to_dict()
        assert d["lint"]["configured"] is True
        assert d["lint"]["attempted_unmeasured"] is False
        assert d["tests"]["configured"] is True
        assert d["tests"]["attempted_unmeasured"] is False
        assert d["tests"]["pass_rate"] == 1.0
        assert d["measured"] is True
        assert d["status"] == "measured"

    def test_to_dict_unmeasured_report_emits_null_pass_rate_and_unmeasured_status(self) -> None:
        """[contract] assay with nothing measured: MCP payload has measured: false,
        status: 'unmeasured', tests.pass_rate: null, and no top-level 'error' key."""
        report = AssayReport(
            project="UnmeasuredProj",
            mode="quick",
            lint=LintResult(
                issues=0,
                output="No lint command configured",
                success=False,
                configured=False,
                attempted=False,
            ),
            tests=RunResult(
                total=0,
                passed=0,
                output="No test command configured",
                success=False,
                configured=False,
                attempted=False,
            ),
        )
        d = report.to_dict()
        assert d["measured"] is False
        assert d["status"] == "unmeasured"
        assert d["tests"]["pass_rate"] is None
        assert "error" not in d

    def test_to_dict_partial_status_when_measured_and_attempted_unmeasured(self) -> None:
        """When at least one dimension is measured and another was attempted but
        produced no usable score, status is 'partial' and unmeasured tests emit
        pass_rate: None."""
        report = AssayReport(
            project="PartialProj",
            mode="quick",
            lint=LintResult(issues=0, output="ok", success=True, configured=True, attempted=True),
            tests=RunResult(
                total=0,
                passed=0,
                output="collection error",
                success=False,
                configured=True,
                attempted=True,
            ),
        )
        d = report.to_dict()
        assert d["measured"] is True
        assert d["status"] == "partial"
        assert d["tests"]["pass_rate"] is None

    def test_to_dict_lint_only_configured_is_measured_with_null_test_pass_rate(self) -> None:
        """When lint is measured and tests are simply not configured (not attempted),
        status is 'measured' and tests.pass_rate is None."""
        report = AssayReport(
            project="LintOnlyProj",
            mode="quick",
            lint=LintResult(issues=0, output="ok", success=True, configured=True, attempted=True),
            tests=RunResult(
                total=0,
                passed=0,
                output="No test command configured",
                success=False,
                configured=False,
                attempted=False,
            ),
        )
        d = report.to_dict()
        assert d["measured"] is True
        assert d["status"] == "measured"
        assert d["tests"]["pass_rate"] is None

    def test_to_dict_rounds_pass_rate_to_three_decimals(self) -> None:
        report = AssayReport(
            project="RoundingProj",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=3, passed=1, failed=2, success=False),
        )
        assert report.to_dict()["tests"]["pass_rate"] == 0.333

    def test_to_dict_optional_dimensions_drive_measured_and_partial_status(self) -> None:
        from linc_crucible.core.scorer import QualityScore, ScoreDimension

        unmeasured_lint = LintResult(issues=0, success=False, configured=False)
        unmeasured_tests = RunResult(total=0, passed=0, success=False, configured=False)

        for kwargs in (
            {"security": SecurityResult(score=90.0, success=True)},
            {"performance": PerformanceResult(apk_size_mb=10.0, success=True)},
            {"performance": PerformanceResult(ipa_size_mb=20.0, success=True)},
            {"accessibility": AccessibilityResult(score=95.0, success=True)},
            {"a11y_tree": A11yTreeResult(structural_score=90.0, success=True)},
            {
                "quality_score": QualityScore(
                    project="P",
                    timestamp="T",
                    mode="quick",
                    composite=90.0,
                    ship_gate=True,
                    dimensions=[
                        ScoreDimension(
                            name="lint",
                            raw_value=0.0,
                            score=100.0,
                            weight=1.0,
                            weighted_score=100.0,
                        )
                    ],
                )
            },
        ):
            r = AssayReport(
                project="P",
                mode="quick",
                lint=unmeasured_lint,
                tests=unmeasured_tests,
                **kwargs,
            )
            d = r.to_dict()
            assert d["measured"] is True
            assert d["status"] == "measured"

        # Failed optional dimensions on an otherwise-unmeasured report stay unmeasured
        for bad_kwargs in (
            {"security": SecurityResult(score=0.0, success=False)},
            {"performance": PerformanceResult(apk_size_mb=10.0, success=False)},
            {"performance": PerformanceResult(apk_size_mb=None, ipa_size_mb=None, success=True)},
            {"accessibility": AccessibilityResult(score=0.0, success=False)},
            {"a11y_tree": A11yTreeResult(structural_score=None, success=False, attempted=True)},
        ):
            r = AssayReport(
                project="P",
                mode="quick",
                lint=unmeasured_lint,
                tests=unmeasured_tests,
                **bad_kwargs,
            )
            d = r.to_dict()
            assert d["measured"] is False
            assert d["status"] == "unmeasured"

        # Failed optional dimension alongside a measured dimension yields status='partial'
        measured_lint = LintResult(issues=0, success=True, configured=True, attempted=True)
        for partial_kwargs in (
            {
                "lint": LintResult(issues=0, success=False, configured=True, attempted=True),
                "tests": RunResult(
                    total=5, passed=5, success=True, configured=True, attempted=True
                ),
            },
            {
                "lint": measured_lint,
                "tests": unmeasured_tests,
                "security": SecurityResult(success=False),
            },
            {
                "lint": measured_lint,
                "tests": unmeasured_tests,
                "performance": PerformanceResult(success=False),
            },
            {
                "lint": measured_lint,
                "tests": unmeasured_tests,
                "accessibility": AccessibilityResult(success=False),
            },
            {
                "lint": measured_lint,
                "tests": unmeasured_tests,
                "a11y_tree": A11yTreeResult(success=False, attempted=True),
            },
            {
                "lint": measured_lint,
                "tests": unmeasured_tests,
                "quality_score": QualityScore(
                    project="P",
                    timestamp="T",
                    mode="quick",
                    composite=100.0,
                    ship_gate=False,
                    attempted_unmeasured=["perf"],
                ),
            },
        ):
            r = AssayReport(project="P", mode="quick", **partial_kwargs)
            d = r.to_dict()
            assert d["measured"] is True
            assert d["status"] == "partial"


class TestRunResult:
    def test_pass_rate_all_passed(self) -> None:
        r = RunResult(total=10, passed=10)
        assert r.pass_rate == 1.0

    def test_pass_rate_partial(self) -> None:
        r = RunResult(total=10, passed=7, failed=3)
        assert r.pass_rate == 0.7

    def test_pass_rate_zero_total_is_vacuous_not_a_measured_pass(self) -> None:
        """Replaces the old assertion (linc-crucible#136):

            OLD: assert r.pass_rate == 1.0  # No tests = assume pass

        The "assume pass" comment was the bug's own rationale written down as
        a test — it read a division-by-zero guard as a real "everything
        passed" signal. `pass_rate` still legitimately returns 1.0 at
        `total == 0` (it has to return *something*), but that value is only
        ever a measured pass when `configured` says the run actually
        happened. This test keeps the raw-property assertion (unchanged
        behavior) and adds the assertion the old one was missing: an
        unconfigured zero-total result must NOT surface as a scoreable pass.

            NEW: assert r.pass_rate == 1.0            # vacuous guard, unchanged
                 assert r.score_input is None          # not a scoring signal when unconfigured
        """
        r = RunResult(total=0, configured=False)
        assert r.pass_rate == 1.0  # division-by-zero guard, not "assume pass"
        assert r.score_input is None

    def test_pass_rate_zero_total_attempted_and_failed_is_also_excluded(self) -> None:
        """linc-crucible#136 finding 4 (MINOS round 0): the sibling shape the
        original replacement test above did NOT cover — `configured` left at
        its default (`True`, exactly what every non-exception crash produces:
        the command WAS configured, `_run_tests`/`_run_lint` never touch
        `configured` on that path), but the run attempted and produced
        nothing parseable (`total == 0`, `success == False`). Before finding
        1's fix this was RED: `score_input` fell through to `pass_rate`
        (1.0) because only `configured` gated it.
        """
        r = RunResult(total=0, success=False)
        assert r.configured is True  # default — a command WAS configured
        assert r.pass_rate == 1.0  # the vacuous guard is unchanged...
        assert r.score_input is None  # ...but it is NOT a scoring signal here

    def test_pass_rate_zero_total_success_true_is_a_measured_pass(self) -> None:
        """The genuinely-empty-but-green case is a SEPARATE, deliberate
        decision from the two crash shapes above — `success is True` means
        the run completed cleanly, so `score_input` still returns the
        vacuous `pass_rate` of 1.0 rather than being lumped in with a crash.
        """
        r = RunResult(total=0, success=True)
        assert r.score_input == 1.0


class TestLintResultScoreInput:
    """`LintResult.score_input` mirrors `RunResult.score_input`'s guard
    shape (linc-crucible#136 finding 1): excluded when not configured, OR
    when the run exited non-zero with nothing parseable."""

    def test_clean_pass_is_measured(self) -> None:
        r = LintResult(issues=0, success=True)
        assert r.score_input == 0

    def test_issues_found_is_measured_even_on_nonzero_exit(self) -> None:
        """Many linters (ruff, eslint) exit non-zero when issues are found —
        that must still score the parsed issue count, not be excluded."""
        r = LintResult(issues=3, success=False)
        assert r.score_input == 3

    def test_nonzero_exit_with_no_parsed_issues_is_excluded(self) -> None:
        r = LintResult(issues=0, success=False)
        assert r.score_input is None

    def test_not_configured_is_excluded(self) -> None:
        r = LintResult(configured=False)
        assert r.score_input is None


class TestPerformanceResult:
    def test_defaults(self) -> None:
        r = PerformanceResult()
        assert r.apk_size_mb is None
        assert r.ipa_size_mb is None
        assert r.success is True

    def test_with_apk_size(self) -> None:
        r = PerformanceResult(apk_size_mb=12.5, success=True)
        assert r.apk_size_mb == 12.5

    def test_with_ipa_size(self) -> None:
        r = PerformanceResult(ipa_size_mb=45.0, success=True)
        assert r.ipa_size_mb == 45.0


class TestAssayWithPerf:
    @patch("subprocess.run")
    def test_assay_perf_flag(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        # Lint returns clean
        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""

        # Tests return all passing
        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 20 passed in 3.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        # No APK found — perf should report failure
        mock_forge.find_apk.return_value = None
        mock_forge.find_ios_app.return_value = None

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", mode="quick", perf=True)

        assert report.performance is not None
        assert report.performance.success is False
        # Score should still work (no perf dimension if no artifact)
        assert report.quality_score is not None


class TestA11yTreeResult:
    def test_defaults(self) -> None:
        r = A11yTreeResult()
        assert r.structural_score is None
        assert r.rules_score == 100.0
        assert r.diff_report is None
        assert r.violations is None
        assert r.success is True
        assert r.error == ""
        assert r.configured is True
        assert r.attempted is False
        assert r.attempted_unmeasured is False

    def test_with_error(self) -> None:
        r = A11yTreeResult(success=False, error="nerve unreachable")
        assert r.success is False
        assert r.error == "nerve unreachable"

    def test_serialized_in_assay_report_to_dict(self) -> None:
        a11y_tree = A11yTreeResult(structural_score=90.0, rules_score=80.0)
        report = AssayReport(
            project="Test",
            mode="quick",
            lint=LintResult(issues=0, success=True),
            tests=RunResult(total=5, passed=5, success=True),
            a11y_tree=a11y_tree,
        )
        d = report.to_dict()
        assert "a11y_tree" in d
        assert d["a11y_tree"]["structural_score"] == 90.0
        assert d["a11y_tree"]["rules_score"] == 80.0
        assert d["a11y_tree"]["success"] is True

    def test_absent_from_to_dict_when_none(self) -> None:
        report = AssayReport(
            project="Test",
            mode="quick",
            lint=LintResult(),
            tests=RunResult(),
        )
        d = report.to_dict()
        assert "a11y_tree" not in d


class TestAssayWithA11yTree:
    @patch("subprocess.run")
    def test_assay_a11y_tree_no_nerve_client(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """When no nerve_client configured, a11y_tree result has success=False."""
        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""

        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 5 passed in 1.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        # No nerve_client provided
        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", a11y_tree=True, device="emulator-5554")

        assert report.a11y_tree is not None
        assert report.a11y_tree.success is False
        assert "nerve" in report.a11y_tree.error.lower()

    @patch("subprocess.run")
    def test_assay_a11y_tree_with_nerve_client(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """When nerve_client and tree_capture provided, a11y_tree is populated."""
        from unittest.mock import MagicMock

        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""

        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 5 passed in 1.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        # Mock nerve_client
        mock_nerve = MagicMock()
        mock_nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "OK", "bounds": "[0,0][100,100]"}
        ]
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]

        # Mock tree_capture: no baseline
        mock_tree_capture = MagicMock()
        mock_tree_capture.get_latest.return_value = None

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
            tree_capture=mock_tree_capture,
        )
        report = assayer.assay("TestProject", a11y_tree=True, device="emulator-5554")

        assert report.a11y_tree is not None
        assert report.a11y_tree.success is True

    def test_run_a11y_tree_no_baseline(self, mock_forge: MagicMock, scorer: QualityScorer) -> None:
        """AC1: _run_a11y_tree with no baseline yields success True, structural_score None."""
        mock_nerve = MagicMock()
        mock_nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "OK", "bounds": "[0,0][100,100]"}
        ]
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]

        mock_tree_capture = MagicMock()
        mock_tree_capture.get_latest.return_value = None

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
            tree_capture=mock_tree_capture,
        )
        res = assayer._run_a11y_tree("TestProject", device="emulator-5554")
        assert res.success is True
        assert res.structural_score is None
        assert res.rules_score is not None
        assert res.rules_score == 100.0

    @patch("subprocess.run")
    def test_assay_a11y_tree_no_baseline_excludes_structural_dimension(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """AC2: assay(a11y_tree=True) with no baseline passes a11y_structural_score=None."""
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        mock_nerve = MagicMock()
        mock_nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "OK", "bounds": "[0,0][100,100]"}
        ]
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]

        mock_tree_capture = MagicMock()
        mock_tree_capture.get_latest.return_value = None

        with patch.object(scorer, "compute", wraps=scorer.compute) as spy_compute:
            assayer = Assayer(
                forge_reader=mock_forge,
                scorer=scorer,
                nerve_client=mock_nerve,
                tree_capture=mock_tree_capture,
            )
            report = assayer.assay("TestProject", a11y_tree=True, device="emulator-5554")

            assert report.quality_score is not None
            spy_compute.assert_called_once()
            _, kwargs = spy_compute.call_args
            assert kwargs.get("a11y_structural_score") is None
            dim_names = [d.name for d in report.quality_score.dimensions]
            assert "a11y_structural" not in dim_names
            assert "a11y_rules" in dim_names

    @patch("subprocess.run")
    def test_assay_a11y_tree_crashed_run_forces_ship_gate_false(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """AC3: crashed a11y_tree run adds 'a11y_tree' to attempted_unmeasured
        and forces ship_gate False."""
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        mock_nerve = MagicMock()
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]
        mock_nerve.elements_strict.side_effect = RuntimeError(
            "nerve crashed during element extraction"
        )

        mock_tree_capture = MagicMock()

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
            tree_capture=mock_tree_capture,
        )
        report = assayer.assay("TestProject", a11y_tree=True, device="emulator-5554")

        assert report.quality_score is not None
        assert "a11y_tree" in report.quality_score.attempted_unmeasured
        assert report.quality_score.ship_gate is False

    @patch("subprocess.run")
    def test_a11y_tree_healthy_baseline_vs_unconfigured_vs_raising(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """AC4: one test covering healthy baseline (score 87), unconfigured,
        and raising-elements."""
        # Part A: Stored baseline with TreeDiffer score 87
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout="All good", stderr=""),
            MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr=""),
        ]
        mock_nerve = MagicMock()
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]
        mock_nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "OK", "bounds": "[0,0][100,100]"}
        ]

        mock_tree_capture = MagicMock()
        mock_tree_capture.get_latest.return_value = "baseline-1"
        mock_tree_capture.load_elements.return_value = []

        mock_diff_report = MagicMock()
        mock_diff_report.score = 87

        with patch(
            "linc_crucible.core.tree_differ.TreeDiffer.diff", return_value=mock_diff_report
        ):
            assayer_healthy = Assayer(
                forge_reader=mock_forge,
                scorer=scorer,
                nerve_client=mock_nerve,
                tree_capture=mock_tree_capture,
            )
            report_healthy = assayer_healthy.assay(
                "TestProject", a11y_tree=True, device="emulator-5554"
            )

            assert report_healthy.a11y_tree is not None
            assert report_healthy.a11y_tree.structural_score == 87.0
            dim_names_healthy = [d.name for d in report_healthy.quality_score.dimensions]
            assert "a11y_structural" in dim_names_healthy
            assert "a11y_tree" not in report_healthy.quality_score.attempted_unmeasured

        # Part B: nerve client None (not configured)
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout="All good", stderr=""),
            MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr=""),
        ]
        assayer_unconfigured = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=None,
        )
        report_unconfigured = assayer_unconfigured.assay(
            "TestProject", a11y_tree=True, device="emulator-5554"
        )
        dim_names_unconf = [d.name for d in report_unconfigured.quality_score.dimensions]
        assert "a11y_structural" not in dim_names_unconf
        assert "a11y_tree" not in report_unconfigured.quality_score.attempted_unmeasured

        # Part C: raising-elements case
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout="All good", stderr=""),
            MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr=""),
        ]
        mock_nerve_raising = MagicMock()
        mock_nerve_raising.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]
        mock_nerve_raising.elements_strict.side_effect = RuntimeError("nerve failure")

        assayer_raising = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve_raising,
            tree_capture=mock_tree_capture,
        )
        report_raising = assayer_raising.assay(
            "TestProject", a11y_tree=True, device="emulator-5554"
        )
        assert "a11y_tree" in report_raising.quality_score.attempted_unmeasured

    @patch("subprocess.run")
    def test_assay_a11y_tree_auto_discovers_device(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """When device="" is passed, discovers first available device."""
        from unittest.mock import MagicMock

        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""

        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 5 passed in 1.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        mock_nerve = MagicMock()
        mock_nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "OK", "bounds": "[0,0][100,100]"}
        ]
        mock_nerve.discover.return_value = [{"id": "auto-device-1", "platform": "android"}]

        mock_tree_capture = MagicMock()
        mock_tree_capture.get_latest.return_value = None

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
            tree_capture=mock_tree_capture,
        )
        # No device specified — should auto-discover
        report = assayer.assay("TestProject", a11y_tree=True, device="")

        assert report.a11y_tree is not None
        assert report.a11y_tree.success is True
        # elements_strict() should have been called with the discovered device
        mock_nerve.elements_strict.assert_called_once_with("auto-device-1")

    @patch("subprocess.run")
    def test_assay_a11y_tree_no_device_available(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """When no device is available and none specified, returns error."""
        from unittest.mock import MagicMock

        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""

        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 5 passed in 1.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        mock_nerve = MagicMock()
        mock_nerve.discover.return_value = []  # no devices

        mock_tree_capture = MagicMock()

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
            tree_capture=mock_tree_capture,
        )
        report = assayer.assay("TestProject", a11y_tree=True, device="")

        assert report.a11y_tree is not None
        assert report.a11y_tree.success is False

    @patch("subprocess.run")
    def test_assay_without_a11y_tree_default(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """By default, a11y_tree is not run and report.a11y_tree is None."""
        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""

        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 5 passed in 1.0s ======"
        test_result.stderr = ""

        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject")

        assert report.a11y_tree is None


class TestMeasureArtifactSize:
    """Characterization tests for _measure_artifact_size — covers CRAP-critical branches."""

    def test_measure_artifact_size_apk_only(self, tmp_path: Path, scorer: QualityScorer) -> None:
        """APK found, no iOS artifact."""
        apk_file = tmp_path / "app.apk"
        apk_file.write_bytes(b"x" * (5 * 1024 * 1024))  # ~5 MB

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(tmp_path),
            framework="gradle_kmp",
        )
        forge.find_apk.return_value = str(apk_file)
        forge.find_ios_app.return_value = None

        assayer = Assayer(forge_reader=forge, scorer=scorer)
        result = assayer._measure_artifact_size(forge.get("TestProject"))

        assert result.success is True
        assert result.apk_size_mb is not None
        assert result.apk_size_mb == pytest.approx(5.0, abs=0.01)
        assert result.ipa_size_mb is None
        assert "APK" in result.output

    def test_measure_artifact_size_ios_dir(self, tmp_path: Path, scorer: QualityScorer) -> None:
        """iOS .app directory found — sums all file sizes."""
        app_dir = tmp_path / "MyApp.app"
        app_dir.mkdir()
        (app_dir / "binary").write_bytes(b"x" * (2 * 1024 * 1024))
        (app_dir / "assets").mkdir()
        (app_dir / "assets" / "image.png").write_bytes(b"y" * (1 * 1024 * 1024))

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(tmp_path),
            framework="gradle_kmp",
        )
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = str(app_dir)

        assayer = Assayer(forge_reader=forge, scorer=scorer)
        result = assayer._measure_artifact_size(forge.get("TestProject"))

        assert result.success is True
        assert result.ipa_size_mb is not None
        assert result.ipa_size_mb == pytest.approx(3.0, abs=0.01)
        assert result.apk_size_mb is None
        assert "iOS" in result.output

    def test_measure_artifact_size_ios_file(self, tmp_path: Path, scorer: QualityScorer) -> None:
        """iOS artifact is a flat file (IPA) — uses file size directly."""
        ipa_file = tmp_path / "MyApp.ipa"
        ipa_file.write_bytes(b"z" * (4 * 1024 * 1024))

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(tmp_path),
            framework="gradle_kmp",
        )
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = str(ipa_file)

        assayer = Assayer(forge_reader=forge, scorer=scorer)
        result = assayer._measure_artifact_size(forge.get("TestProject"))

        assert result.success is True
        assert result.ipa_size_mb is not None
        assert result.ipa_size_mb == pytest.approx(4.0, abs=0.01)

    def test_measure_artifact_size_both(self, tmp_path: Path, scorer: QualityScorer) -> None:
        """Both APK and iOS artifacts found."""
        apk_file = tmp_path / "app.apk"
        apk_file.write_bytes(b"x" * (3 * 1024 * 1024))

        app_dir = tmp_path / "MyApp.app"
        app_dir.mkdir()
        (app_dir / "binary").write_bytes(b"y" * (6 * 1024 * 1024))

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(tmp_path),
            framework="gradle_kmp",
        )
        forge.find_apk.return_value = str(apk_file)
        forge.find_ios_app.return_value = str(app_dir)

        assayer = Assayer(forge_reader=forge, scorer=scorer)
        result = assayer._measure_artifact_size(forge.get("TestProject"))

        assert result.success is True
        assert result.apk_size_mb is not None
        assert result.ipa_size_mb is not None
        assert "APK" in result.output
        assert "iOS" in result.output

    def test_measure_artifact_size_none(self, tmp_path: Path, scorer: QualityScorer) -> None:
        """No artifacts found at all."""
        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="TestProject",
            path=str(tmp_path),
            framework="gradle_kmp",
        )
        forge.find_apk.return_value = None
        forge.find_ios_app.return_value = None

        assayer = Assayer(forge_reader=forge, scorer=scorer)
        result = assayer._measure_artifact_size(forge.get("TestProject"))

        assert result.success is False
        assert result.apk_size_mb is None
        assert result.ipa_size_mb is None
        assert "No APK or iOS app found" in result.output


class TestAssayWithA11y:
    @patch("subprocess.run")
    def test_assay_with_a11y_flag(
        self, mock_subprocess: MagicMock, tmp_path: Path, scorer: QualityScorer
    ) -> None:
        # Create a real project dir with an a11y issue
        project_dir = tmp_path / "a11y_project"
        layout_dir = project_dir / "src" / "main" / "res" / "layout"
        layout_dir.mkdir(parents=True)
        (layout_dir / "activity_main.xml").write_text(
            '<LinearLayout>\n  <ImageView android:src="@drawable/ic" />\n</LinearLayout>'
        )

        forge = MagicMock(spec=ForgeProjectReader)
        forge.get.return_value = ForgeProject(
            name="A11yProject",
            path=str(project_dir),
            test_command="pytest",
            lint_command="ruff check",
            framework="gradle_kmp",
        )

        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""
        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 5 passed in 1.0s ======"
        test_result.stderr = ""
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=forge, scorer=scorer)
        report = assayer.assay("A11yProject", a11y=True)

        assert report.accessibility is not None
        assert report.accessibility.success is True
        assert report.accessibility.major >= 1

    @patch("subprocess.run")
    def test_assay_without_a11y_default(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock()
        lint_result.returncode = 0
        lint_result.stdout = "All good"
        lint_result.stderr = ""
        test_result = MagicMock()
        test_result.returncode = 0
        test_result.stdout = "====== 5 passed in 1.0s ======"
        test_result.stderr = ""
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject")

        assert report.accessibility is None


class TestAssayFullModeSecurityBranch:
    """CRAP-ratchet coverage (linc-crucible#136 repair round): `assay()`'s
    `mode == "full"` block (security_score wiring) had never been exercised
    end-to-end through `assay()` — every prior test used the "quick" default.
    Patch `_run_security` directly (the network/MobSF plumbing is already
    covered separately in TestAssayer) so these tests isolate exactly the
    branch under test: does a successful vs. failed security scan actually
    flow into `quality_score`."""

    @patch("subprocess.run")
    def test_full_mode_security_success_feeds_quality_score(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        with patch.object(
            assayer,
            "_run_security",
            return_value=SecurityResult(score=85.0, success=True),
        ):
            report = assayer.assay("TestProject", mode="full")

        assert report.security is not None
        assert report.security.success is True
        assert report.quality_score is not None
        security_dim = next(d for d in report.quality_score.dimensions if d.name == "security")
        assert security_dim.score == 85.0
        assert "full" in report.quality_score.mode

    @patch("subprocess.run")
    def test_full_mode_security_failure_excludes_dimension(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        with patch.object(
            assayer,
            "_run_security",
            return_value=SecurityResult(success=False, output="MobSF unreachable"),
        ):
            report = assayer.assay("TestProject", mode="full")

        assert report.security is not None
        assert report.security.success is False
        assert report.quality_score is not None
        assert all(d.name != "security" for d in report.quality_score.dimensions)


class TestAssayPerfSuccessBranches:
    """CRAP-ratchet coverage: `TestAssayWithPerf` only exercised the
    "no artifact found" failure path — the two success branches (APK size
    vs. IPA size feeding `perf_score`) were never taken through `assay()`."""

    @patch("subprocess.run")
    def test_perf_apk_size_feeds_quality_score(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        with patch.object(
            assayer,
            "_measure_artifact_size",
            return_value=PerformanceResult(apk_size_mb=12.5, success=True),
        ):
            report = assayer.assay("TestProject", perf=True)

        assert report.performance is not None
        assert report.performance.apk_size_mb == 12.5
        assert report.quality_score is not None
        assert any(d.name == "perf" for d in report.quality_score.dimensions)

    @patch("subprocess.run")
    def test_perf_ipa_size_feeds_quality_score(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        with patch.object(
            assayer,
            "_measure_artifact_size",
            return_value=PerformanceResult(ipa_size_mb=45.0, success=True),
        ):
            report = assayer.assay("TestProject", perf=True)

        assert report.performance is not None
        assert report.performance.ipa_size_mb == 45.0
        assert report.quality_score is not None
        assert any(d.name == "perf" for d in report.quality_score.dimensions)


class TestAssayA11yFailureBranch:
    """CRAP-ratchet coverage: `TestAssayWithA11y` only exercised the
    accessibility-success path — the `report.accessibility.success is False`
    branch (a11y_score_val stays None, dimension excluded) was never taken."""

    @patch("subprocess.run")
    def test_a11y_failure_excludes_dimension(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        with patch.object(
            assayer,
            "_run_accessibility",
            return_value=AccessibilityResult(success=False, output="scan failed"),
        ):
            report = assayer.assay("TestProject", a11y=True)

        assert report.accessibility is not None
        assert report.accessibility.success is False
        assert report.quality_score is not None
        assert all(d.name != "a11y" for d in report.quality_score.dimensions)


class TestA11yTreeElementsStrict:
    """Acceptance tests for bead linc-crucible-64q.3 (audit F4)."""

    @patch("subprocess.run")
    def test_elements_strict_raising_ui_tree_unavailable_yields_unmeasured(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        mock_nerve = MagicMock()
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]
        mock_nerve.elements_strict.side_effect = UiTreeUnavailableError("device connection lost")

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
        )
        res = assayer._run_a11y_tree("TestProject", device="emulator-5554")
        assert res.success is False
        assert res.attempted is True
        assert res.rules_score is None

        report = assayer.assay("TestProject", a11y_tree=True, device="emulator-5554")
        assert report.quality_score is not None
        assert report.quality_score.ship_gate is False
        assert "a11y_tree" in report.quality_score.attempted_unmeasured
        assert "partial-unmeasured" in report.quality_score.mode

        loaded = scorer.load_latest("TestProject")
        assert loaded is not None
        assert loaded.ship_gate is False
        assert "a11y_tree" in loaded.attempted_unmeasured
        assert "partial-unmeasured" in loaded.mode

    @patch("subprocess.run")
    def test_elements_strict_returning_empty_list_yields_unmeasured(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        mock_nerve = MagicMock()
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]
        mock_nerve.elements_strict.return_value = []

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
        )
        res = assayer._run_a11y_tree("TestProject", device="emulator-5554")
        assert res.success is False
        assert res.attempted is True
        assert res.rules_score is None

        report = assayer.assay("TestProject", a11y_tree=True, device="emulator-5554")
        assert report.quality_score is not None
        assert report.quality_score.ship_gate is False
        assert "a11y_tree" in report.quality_score.attempted_unmeasured
        assert "partial-unmeasured" in report.quality_score.mode

        loaded = scorer.load_latest("TestProject")
        assert loaded is not None
        assert loaded.ship_gate is False
        assert "a11y_tree" in loaded.attempted_unmeasured
        assert "partial-unmeasured" in loaded.mode

    def test_elements_strict_returning_12_elements_with_2_violations(
        self, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        mock_nerve = MagicMock()
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]

        elements_12 = [
            {
                "id": f"elem_{i}",
                "text": f"Label {i}",
                "clickable": False,
                "bounds": "[0,0][100,100]",
            }
            for i in range(10)
        ] + [
            {
                "id": "bad_1",
                "clickable": True,
                "bounds": "[0,0][100,100]",
                "text": "",
                "desc": "",
            },
            {
                "id": "bad_2",
                "clickable": True,
                "bounds": "[0,0][10,10]",
                "text": "Tiny",
            },
        ]
        mock_nerve.elements_strict.return_value = elements_12

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
        )
        res = assayer._run_a11y_tree("TestProject", device="emulator-5554")
        assert res.success is True
        assert res.attempted is True
        assert res.violations is not None
        assert len(res.violations) == 2
        assert res.rules_score == 90.0

    def test_lenient_elements_never_called_by_assayer(
        self, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        mock_nerve = MagicMock()
        mock_nerve.discover.return_value = [{"id": "emulator-5554", "platform": "android"}]
        mock_nerve.elements_strict.return_value = [
            {"id": "elem1", "text": "Label", "clickable": False, "bounds": "[0,0][100,100]"}
        ]

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            nerve_client=mock_nerve,
        )
        res = assayer._run_a11y_tree("TestProject", device="emulator-5554")
        assert res.success is True
        assert mock_nerve.elements.call_count == 0
        assert mock_nerve.elements_strict.call_count == 1

    def test_capture_a11y_elements_branch_unavailable(
        self, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        mock_nerve = MagicMock()
        mock_nerve.elements_strict.side_effect = UiTreeUnavailableError("socket closed")
        assayer = Assayer(forge_reader=mock_forge, scorer=scorer, nerve_client=mock_nerve)
        reason = assayer._capture_a11y_elements("dev1")
        assert reason == "ui tree unavailable: socket closed"

    def test_capture_a11y_elements_branch_empty(
        self, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        mock_nerve = MagicMock()
        mock_nerve.elements_strict.return_value = []
        assayer = Assayer(forge_reader=mock_forge, scorer=scorer, nerve_client=mock_nerve)
        reason = assayer._capture_a11y_elements("dev1")
        expected_msg = "ui tree empty: 0 elements — measurement failure, not perfect accessibility"
        assert reason == expected_msg

    def test_capture_a11y_elements_branch_success(
        self, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        mock_nerve = MagicMock()
        mock_nerve.elements_strict.return_value = [{"id": "btn1", "text": "Click"}]
        assayer = Assayer(forge_reader=mock_forge, scorer=scorer, nerve_client=mock_nerve)
        result = assayer._capture_a11y_elements("dev1")
        assert result == [{"id": "btn1", "text": "Click"}]


class TestAuditF6OptionalDimensionsAndMobSF:
    """Acceptance tests for audit F6 (linc-crucible-kiy.2):
    - assay(mode="full") with no MobSF key -> ship_gate False, mode names security as unmeasured
    - assay(perf=True) with no perf artifact -> perf unmeasured, ship_gate False
    - assay(a11y=True, a11y_path=<nonexistent>) -> a11y unmeasured
    - mode="full" with fake MobSF returning a score -> security scored, mode is full
    - server_context builds Assayer with CRUCIBLE_MOBSF_API_KEY
    - _run_optional_dims per branch
    """

    @patch("subprocess.run")
    def test_assay_full_mode_no_mobsf_key_unmeasured(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """assay(mode="full") with no MobSF key -> persisted score has ship_gate False
        and mode string contains 'security' as unmeasured (fails pre-fix: ship_gate True,
        mode 'quick')."""
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer, mobsf_api_key="")
        report = assayer.assay("TestProject", mode="full")

        assert report.quality_score is not None
        assert report.quality_score.ship_gate is False
        assert "security" in report.quality_score.mode
        assert "unmeasured" in report.quality_score.mode
        assert "security" in report.quality_score.attempted_unmeasured

    @patch("subprocess.run")
    def test_assay_perf_no_artifact_unmeasured(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """assay(perf=True) with no perf artifact -> 'perf' unmeasured, ship_gate False."""
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]
        mock_forge.find_apk.return_value = None
        mock_forge.find_ios_app.return_value = None

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", perf=True)

        assert report.quality_score is not None
        assert report.quality_score.ship_gate is False
        assert "perf" in report.quality_score.attempted_unmeasured

    @patch("subprocess.run")
    def test_assay_a11y_nonexistent_path_unmeasured(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """assay(a11y=True, a11y_path=<nonexistent>) -> 'a11y' unmeasured, ship_gate False."""
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]
        mock_forge.get.return_value = ForgeProject(
            name="TestProject",
            path="/tmp/nonexistent-project-path-xyz-12345",
            test_command="pytest tests/ -v",
            lint_command="ruff check src/",
            framework="python_setup",
        )

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer)
        report = assayer.assay("TestProject", a11y=True)

        assert report.quality_score is not None
        assert report.quality_score.ship_gate is False
        assert "a11y" in report.quality_score.attempted_unmeasured

    @patch("subprocess.run")
    def test_assay_full_mode_with_fake_mobsf_scored(
        self, mock_subprocess: MagicMock, mock_forge: MagicMock, scorer: QualityScorer
    ) -> None:
        """mode="full" with a fake MobSF returning a score -> security scored and mode
        string is "full" (pin)."""
        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        assayer = Assayer(forge_reader=mock_forge, scorer=scorer, mobsf_api_key="valid-key")
        with patch.object(
            assayer,
            "_run_security",
            return_value=SecurityResult(score=90.0, success=True),
        ):
            report = assayer.assay("TestProject", mode="full")

        assert report.quality_score is not None
        assert report.quality_score.mode == "full"
        assert any(
            d.name == "security" and d.score == 90.0 for d in report.quality_score.dimensions
        )
        assert report.quality_score.ship_gate is True

    def test_server_context_builds_assayer_with_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """server_context builds the Assayer with CRUCIBLE_MOBSF_API_KEY from environment."""
        monkeypatch.setenv("CRUCIBLE_MOBSF_API_KEY", "fake-mobsf-key")
        import linc_crucible.server_context as sc

        sc._initialized = False
        sc._ensure_initialized()
        assert sc._assayer._mobsf_api_key == "fake-mobsf-key"

    def test_run_optional_dims_security_branches(self, mock_forge: MagicMock) -> None:
        """_run_optional_dims exercises security success and failure branches."""
        assayer = Assayer(forge_reader=mock_forge)
        project = mock_forge.get.return_value

        # Branch 1: full mode with security success
        report = AssayReport(project="TestProject", mode="full")
        with patch.object(
            assayer, "_run_security", return_value=SecurityResult(score=95.0, success=True)
        ):
            scores, unmeasured = assayer._run_optional_dims(
                project, mode="full", perf=False, a11y=False, report=report
            )
        assert scores.get("security") == 95.0
        assert "security" not in unmeasured

        # Branch 2: full mode with security failure
        report = AssayReport(project="TestProject", mode="full")
        with patch.object(
            assayer, "_run_security", return_value=SecurityResult(success=False, output="err")
        ):
            scores, unmeasured = assayer._run_optional_dims(
                project, mode="full", perf=False, a11y=False, report=report
            )
        assert "security" not in scores
        assert "security" in unmeasured

        # Branch 3: quick mode (security not run)
        report = AssayReport(project="TestProject", mode="quick")
        scores, unmeasured = assayer._run_optional_dims(
            project, mode="quick", perf=False, a11y=False, report=report
        )
        assert "security" not in scores
        assert "security" not in unmeasured

    def test_run_optional_dims_perf_branches(self, mock_forge: MagicMock) -> None:
        """_run_optional_dims exercises perf branches (apk, ipa, failure, skipped)."""
        assayer = Assayer(forge_reader=mock_forge)
        project = mock_forge.get.return_value

        # Branch 1: perf with apk_size_mb
        report = AssayReport(project="TestProject", mode="quick")
        with patch.object(
            assayer,
            "_measure_artifact_size",
            return_value=PerformanceResult(apk_size_mb=30.0, ipa_size_mb=None, success=True),
        ):
            scores, unmeasured = assayer._run_optional_dims(
                project, mode="quick", perf=True, a11y=False, report=report
            )
        assert scores.get("perf") == 30.0
        assert "perf" not in unmeasured

        # Branch 2: perf with ipa_size_mb (apk is None)
        report = AssayReport(project="TestProject", mode="quick")
        with patch.object(
            assayer,
            "_measure_artifact_size",
            return_value=PerformanceResult(apk_size_mb=None, ipa_size_mb=40.0, success=True),
        ):
            scores, unmeasured = assayer._run_optional_dims(
                project, mode="quick", perf=True, a11y=False, report=report
            )
        assert scores.get("perf") == 40.0
        assert "perf" not in unmeasured

        # Branch 3: perf failure
        report = AssayReport(project="TestProject", mode="quick")
        with patch.object(
            assayer,
            "_measure_artifact_size",
            return_value=PerformanceResult(success=False, output="no artifact"),
        ):
            scores, unmeasured = assayer._run_optional_dims(
                project, mode="quick", perf=True, a11y=False, report=report
            )
        assert "perf" not in scores
        assert "perf" in unmeasured

        # Branch 4: perf=False
        report = AssayReport(project="TestProject", mode="quick")
        scores, unmeasured = assayer._run_optional_dims(
            project, mode="quick", perf=False, a11y=False, report=report
        )
        assert "perf" not in scores
        assert "perf" not in unmeasured

    def test_run_optional_dims_a11y_branches(self, mock_forge: MagicMock) -> None:
        """_run_optional_dims exercises a11y branches (success, failure, skipped)."""
        assayer = Assayer(forge_reader=mock_forge)
        project = mock_forge.get.return_value

        # Branch 1: a11y success
        report = AssayReport(project="TestProject", mode="quick")
        with patch.object(
            assayer,
            "_run_accessibility",
            return_value=AccessibilityResult(score=88.0, success=True),
        ):
            scores, unmeasured = assayer._run_optional_dims(
                project, mode="quick", perf=False, a11y=True, report=report
            )
        assert scores.get("a11y") == 88.0
        assert "a11y" not in unmeasured

        # Branch 2: a11y failure
        report = AssayReport(project="TestProject", mode="quick")
        with patch.object(
            assayer,
            "_run_accessibility",
            return_value=AccessibilityResult(success=False, output="missing dir"),
        ):
            scores, unmeasured = assayer._run_optional_dims(
                project, mode="quick", perf=False, a11y=True, report=report
            )
        assert "a11y" not in scores
        assert "a11y" in unmeasured

        # Branch 3: a11y=False
        report = AssayReport(project="TestProject", mode="quick")
        scores, unmeasured = assayer._run_optional_dims(
            project, mode="quick", perf=False, a11y=False, report=report
        )
        assert "a11y" not in scores
        assert "a11y" not in unmeasured


class TestMobSFLoopbackProxyIsolation:
    """[contract] Assayer MobSF scan uses httpx.Client(trust_env=False) and ignores proxy env vars
    (linc-crucible-dq9, R4 Medium 4)."""

    @patch("subprocess.run")
    def test_full_mode_mobsf_scan_never_contacts_proxy_listener(
        self,
        mock_subprocess: MagicMock,
        mock_forge: MagicMock,
        scorer: QualityScorer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import contextlib
        import socket
        import threading
        from collections.abc import Callable

        lint_result = MagicMock(returncode=0, stdout="All good", stderr="")
        test_result = MagicMock(returncode=0, stdout="====== 5 passed in 1.0s ======", stderr="")
        mock_subprocess.side_effect = [lint_result, test_result]

        apk_path = tmp_path / "app.apk"
        apk_path.write_bytes(b"PK\x03\x04fake-apk-bytes")
        mock_forge.find_apk.return_value = str(apk_path)

        def _start_listener(response_body: bytes) -> tuple[int, list[bytes], Callable[[], None]]:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(5)
            srv.settimeout(0.05)
            port = srv.getsockname()[1]
            captured: list[bytes] = []
            stop = threading.Event()

            def _serve() -> None:
                while not stop.is_set():
                    try:
                        conn, _ = srv.accept()
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                    with conn:
                        conn.settimeout(0.5)
                        with contextlib.suppress(OSError):
                            chunks: list[bytes] = []
                            while True:
                                chunk = conn.recv(4096)
                                if not chunk:
                                    break
                                chunks.append(chunk)
                                joined = b"".join(chunks)
                                if b"\r\n\r\n" in joined:
                                    header_part, _, rest = joined.partition(b"\r\n\r\n")
                                    cl_val = 0
                                    for line in header_part.split(b"\r\n"):
                                        if line.lower().startswith(b"content-length:"):
                                            cl_val = int(line.split(b":", 1)[1].strip())
                                    if len(rest) >= cl_val:
                                        break
                            if chunks:
                                captured.append(b"".join(chunks))
                            reply = (
                                b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: application/json\r\n"
                                + f"Content-Length: {len(response_body)}\r\n".encode("ascii")
                                + b"Connection: close\r\n\r\n"
                                + response_body
                            )
                            conn.sendall(reply)

            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()

            def _close() -> None:
                stop.set()
                with contextlib.suppress(OSError):
                    srv.close()
                thread.join(timeout=1.0)

            return port, captured, _close

        body = b'{"hash":"deadbeef","security_score":91.0}'
        proxy_port, proxy_captured, close_proxy = _start_listener(body)
        target_port, target_captured, close_target = _start_listener(body)
        try:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            for var in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ):
                monkeypatch.setenv(var, proxy_url)
            for no_proxy_var in ("NO_PROXY", "no_proxy"):
                monkeypatch.delenv(no_proxy_var, raising=False)

            assayer = Assayer(
                forge_reader=mock_forge,
                scorer=scorer,
                mobsf_url=f"http://127.0.0.1:{target_port}",
                mobsf_api_key="test-mobsf-secret-key",
            )
            report = assayer.assay("TestProject", mode="full")
        finally:
            close_proxy()
            close_target()

        proxy_lines = [
            line
            for raw in proxy_captured
            for line in raw.decode("latin-1", errors="replace").split("\r\n")
            if line.startswith("POST ") or line.lower().startswith("authorization:")
        ]
        assert proxy_captured == [], f"Proxy listener received MobSF traffic: {proxy_lines}"
        assert len(target_captured) == 3
        assert report.security is not None
        assert report.security.success is True
        assert report.security.score == 91.0

    def test_run_security_client_construction_and_per_request_timeouts(
        self, mock_forge: MagicMock, scorer: QualityScorer, tmp_path: Path
    ) -> None:
        """Pin httpx.Client(trust_env=False, timeout=30) and per-request timeouts (60, 120, 30)."""
        import httpx

        apk_path = tmp_path / "app.apk"
        apk_path.write_bytes(b"fake-apk")
        mock_forge.find_apk.return_value = str(apk_path)
        project = mock_forge.get.return_value

        upload_resp = MagicMock()
        upload_resp.json.return_value = {"hash": "hash123"}
        scan_resp = MagicMock()
        report_resp = MagicMock()
        report_resp.json.return_value = {"security_score": 88.0}

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [upload_resp, scan_resp, report_resp]

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            mobsf_url="http://localhost:8000",
            mobsf_api_key="k-123",
        )
        with patch.object(httpx, "Client", return_value=mock_client) as mock_client_cls:
            result = assayer._run_security(project)

        assert result.success is True
        assert result.score == 88.0
        mock_client_cls.assert_called_once_with(trust_env=False, timeout=30)
        assert mock_client.post.call_count == 3

        upload_call, scan_call, report_call = mock_client.post.call_args_list
        assert upload_call.args == ("http://localhost:8000/api/v1/upload",)
        assert upload_call.kwargs["headers"] == {"Authorization": "k-123"}
        assert upload_call.kwargs["timeout"] == 60
        assert list(upload_call.kwargs["files"].keys()) == ["file"]
        assert upload_call.kwargs["files"]["file"].mode == "rb"

        assert scan_call.args == ("http://localhost:8000/api/v1/scan",)
        assert scan_call.kwargs == {
            "data": {"hash": "hash123"},
            "headers": {"Authorization": "k-123"},
            "timeout": 120,
        }
        upload_resp.raise_for_status.assert_called_once_with()
        scan_resp.raise_for_status.assert_called_once_with()

        assert report_call.args == ("http://localhost:8000/api/v1/report_json",)
        assert report_call.kwargs == {
            "data": {"hash": "hash123"},
            "headers": {"Authorization": "k-123"},
            "timeout": 30,
        }
        report_resp.raise_for_status.assert_called_once_with()

    def test_run_security_defaults_missing_hash_to_empty_string(
        self, mock_forge: MagicMock, scorer: QualityScorer, tmp_path: Path
    ) -> None:
        """When upload response JSON omits 'hash', default '' is forwarded to scan/report."""
        import httpx

        apk_path = tmp_path / "app.apk"
        apk_path.write_bytes(b"fake-apk")
        mock_forge.find_apk.return_value = str(apk_path)
        project = mock_forge.get.return_value

        upload_resp = MagicMock()
        upload_resp.json.return_value = {}
        scan_resp = MagicMock()
        report_resp = MagicMock()
        report_resp.json.return_value = {"security_score": 70.0}

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [upload_resp, scan_resp, report_resp]

        assayer = Assayer(
            forge_reader=mock_forge,
            scorer=scorer,
            mobsf_url="http://localhost:8000",
            mobsf_api_key="k-123",
        )
        with patch.object(httpx, "Client", return_value=mock_client):
            result = assayer._run_security(project)

        assert result.success is True
        assert result.score == 70.0
        _, scan_call, report_call = mock_client.post.call_args_list
        assert scan_call.kwargs["data"] == {"hash": ""}
        assert report_call.kwargs["data"] == {"hash": ""}
