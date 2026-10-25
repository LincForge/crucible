"""Store-name allowlist + containment contracts (R4 High 1 + Medium 6).

Every crucible store (screenshot baselines, a11y baselines, `.compare`, scores)
joins caller text into a filesystem path. ``core/store_names.py`` is the single
gate: a ``[A-Za-z0-9._-]`` allowlist on every name plus a resolved-path proof
that the joined directory stays inside its store root.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import structlog

from linc_crucible.core.store_names import (
    _ECHO_LIMIT,
    _FIELDS,
    _RULES,
    MAX_NAME_LEN,
    NAME_RE,
    InvalidStoreName,
    contained_dir,
    name_validation_error,
    prepare_dir,
    required_names_error,
    required_project_error,
    store_error,
    validate_name,
)

HOSTILE_NAMES: list[object] = [
    "/tmp/victim",
    "../x",
    "a/../b",
    "..",
    ".hidden",
    "-flag",
    "a b",
    "a\nb",
    "a\x00b",
    "a" * 129,
    "a" * 10000,
    "",
    5,
    None,
    "a\\b",
    "a..b",
    "a\tb",
    "café",
    "a:b",
]

VALID_NAMES = ["myapp", "My.App_2-x", "home", "default", "a", "A" * MAX_NAME_LEN]


def ascii_id(value: object) -> str:
    """An ASCII-only, escape-free test id for a hostile parameter.

    mutmut re-selects tests by node id, and pytest cannot find an id that holds an escape
    sequence or non-ASCII text (`'a\\x00b'`, `'caf\\xe9'`), so pr-diff-mutation died with
    an instrument error (exit 4, "not found"). pytest dedupes collisions with a suffix.
    """
    text = repr(value).encode("ascii", "backslashreplace").decode("ascii")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:40] or "_"


def _listing(root: Path) -> list[tuple[str, list[str], list[str]]]:
    return [(d, sorted(ds), sorted(fs)) for d, ds, fs in os.walk(root)]


class TestValidateName:
    @pytest.mark.parametrize("value", HOSTILE_NAMES, ids=ascii_id)
    def test_contract_hostile_name_rejected(self, value: object) -> None:
        """[contract] every hostile name in the R4 table raises InvalidStoreName."""
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name(value, "project")
        assert excinfo.value.field == "project"
        assert "project" in str(excinfo.value)

    @pytest.mark.parametrize("value", VALID_NAMES)
    def test_contract_valid_name_accepted(self, value: str) -> None:
        assert validate_name(value, "screen_name") == value

    def test_invalid_store_name_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            validate_name("..", "project")

    def test_regex_is_anchored_allowlist(self) -> None:
        assert NAME_RE.pattern == r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z"
        assert NAME_RE.fullmatch("ok\n") is None

    @pytest.mark.parametrize(
        ("value", "rule"),
        [
            ("", "must not be empty"),
            ("a" * 129, "at most 128 characters"),
            ("a\x00b", "NUL"),
            ("a/b", "path separator"),
            ("a\\b", "path separator"),
            ("a..b", "'..'"),
            ("a b", "whitespace"),
            (".x", "must not start with '.' or '-'"),
            ("-x", "must not start with '.' or '-'"),
            ("a:b", "letters, digits"),
            (5, "must be a string"),
            (None, "must be a string"),
        ],
        ids=ascii_id,
    )
    def test_error_names_the_rule(self, value: object, rule: str) -> None:
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name(value, "screen_name")
        assert rule in str(excinfo.value)
        assert excinfo.value.rule in str(excinfo.value)

    def test_error_never_echoes_more_than_40_chars(self) -> None:
        value = "Z" * 10000
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name(value, "project")
        message = str(excinfo.value)
        assert "Z" * 40 in message
        assert "Z" * 41 not in message
        assert len(message) < 200

    def test_error_does_not_echo_raw_nul(self) -> None:
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name("a\x00b", "project")
        assert "\x00" not in str(excinfo.value)


class TestContainedDir:
    def test_returns_unresolved_join_for_valid_names(self, tmp_path: Path) -> None:
        assert contained_dir(tmp_path, "proj", "home") == tmp_path / "proj" / "home"

    def test_no_names_returns_root(self, tmp_path: Path) -> None:
        assert contained_dir(tmp_path) == tmp_path

    def test_does_not_create_anything(self, tmp_path: Path) -> None:
        contained_dir(tmp_path / "store", "proj", "home")
        assert not (tmp_path / "store").exists()

    @pytest.mark.parametrize("bad", ["../x", "/abs", ".."])
    def test_hostile_project_rejected_with_field(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(InvalidStoreName) as excinfo:
            contained_dir(tmp_path, bad, "home")
        assert excinfo.value.field == "project"

    def test_hostile_screen_rejected_with_field(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidStoreName) as excinfo:
            contained_dir(tmp_path, "proj", "../../x")
        assert excinfo.value.field == "screen_name"

    def test_third_name_field_is_generic(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidStoreName) as excinfo:
            contained_dir(tmp_path, "proj", "home", "..")
        assert excinfo.value.field == "name"

    def test_contract_symlinked_project_refused(self, tmp_path: Path) -> None:
        """[contract] `<store>/proj` symlinked outside the store is refused."""
        store = tmp_path / "store"
        outside = tmp_path / "outside"
        store.mkdir()
        outside.mkdir()
        (store / "proj").symlink_to(outside, target_is_directory=True)
        with pytest.raises(InvalidStoreName) as excinfo:
            contained_dir(store, "proj", "home")
        assert excinfo.value.field == "project"
        assert "outside the store" in str(excinfo.value)

    def test_symlinked_store_root_is_fine(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        assert contained_dir(link, "proj") == link / "proj"


class TestPrepareDir:
    def test_creates_and_returns_dir(self, tmp_path: Path) -> None:
        path = prepare_dir(tmp_path / "store", "proj", "home")
        assert path == tmp_path / "store" / "proj" / "home"
        assert path.is_dir()

    def test_hostile_name_creates_nothing(self, tmp_path: Path) -> None:
        store = tmp_path / "a" / "b" / "store"
        with pytest.raises(InvalidStoreName):
            prepare_dir(store, "../../../escaped")
        assert not (tmp_path / "escaped").exists()
        assert not store.exists()

    def test_contract_symlink_raced_in_after_mkdir_refused(self, tmp_path: Path) -> None:
        """[contract] a symlink swapped in between check and mkdir is caught by the re-check."""
        store = tmp_path / "store"
        outside = tmp_path / "outside"
        outside.mkdir()
        store.mkdir()
        real_mkdir = Path.mkdir

        def racing_mkdir(self: Path, *args: object, **kwargs: object) -> None:
            if self == store / "proj":
                self.symlink_to(outside, target_is_directory=True)
                return
            real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

        with (
            patch.object(Path, "mkdir", racing_mkdir),
            pytest.raises(InvalidStoreName, match="outside the store"),
        ):
            prepare_dir(store, "proj")


class TestNameValidationError:
    def test_returns_none_for_valid_names(self) -> None:
        assert name_validation_error(project="myapp", screen_name="home") is None

    def test_empty_optional_screen_is_skipped(self) -> None:
        assert name_validation_error(project="myapp", screen_name="") is None

    def test_returns_validation_envelope(self) -> None:
        result = name_validation_error(project="../x", screen_name="")
        assert result == {
            "error": result["error"],  # type: ignore[index]
            "error_category": "validation",
            "validation_error": True,
        }
        assert result is not None
        assert result["error"].startswith("invalid project:")

    def test_screen_name_checked(self) -> None:
        result = name_validation_error(project="ok", screen_name="a b")
        assert result is not None
        assert result["error"].startswith("invalid screen_name:")


# ── MCP layer: crucible_baseline / crucible_score ──────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["screenshot", "a11y"])
@pytest.mark.parametrize("action", ["capture", "latest", "list", "prune", "promote"])
@pytest.mark.parametrize(
    ("project", "screen_name"),
    [("ABS", "home"), ("../../../escaped", "home"), ("ok", "../x"), ("ok", "-flag")],
)
@patch("linc_crucible.server_context._ensure_initialized")
async def test_crucible_baseline_hostile_names_validation_error(
    mock_init: MagicMock,
    tmp_path: Path,
    mode: str,
    action: str,
    project: str,
    screen_name: str,
) -> None:
    """[test] hostile project/screen_name → validation envelope, both modes, every action."""
    from linc_crucible.baseline_tools import crucible_baseline
    from linc_crucible.core.baseline import BaselineManager
    from linc_crucible.core.tree_capture import TreeCapture

    if project == "ABS":  # an absolute path, kept under tmp_path so a red run stays local
        project = str(tmp_path / "victim")
    store = tmp_path / "store"
    with (
        patch("linc_crucible.server_context._baseline_manager", BaselineManager(store)),
        patch("linc_crucible.server_context._tree_capture", TreeCapture(store)),
        patch("linc_crucible.server_context._nerve_client", MagicMock()),
    ):
        result = await crucible_baseline(
            action=action,
            project=project,
            screen_name=screen_name,
            image_path=str(tmp_path / "missing.png"),
            mode=mode,
            keep=0,
        )
    assert result["validation_error"] is True
    assert result["error_category"] == "validation"
    assert result["error"].startswith(("invalid project:", "invalid screen_name:"))
    assert not store.exists()


@pytest.mark.asyncio
@patch("linc_crucible.server_context._ensure_initialized")
async def test_crucible_baseline_valid_name_still_works(
    mock_init: MagicMock, tmp_path: Path
) -> None:
    from linc_crucible.baseline_tools import crucible_baseline
    from linc_crucible.core.baseline import BaselineManager

    with patch("linc_crucible.server_context._baseline_manager", BaselineManager(tmp_path)):
        result = await crucible_baseline(action="prune", project="myapp", keep=0)
    assert result == {"action": "prune", "project": "myapp", "pruned": 0, "keep": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "action"),
    [
        ("screenshot", "capture"),
        ("screenshot", "promote"),
        ("screenshot", "latest"),
        ("screenshot", "list"),
        ("screenshot", "prune"),
        ("a11y", "latest"),
        ("a11y", "list"),
        ("a11y", "prune"),
    ],
)
@patch("linc_crucible.server_context._ensure_initialized")
async def test_crucible_baseline_symlinked_project_is_validation_error(
    mock_init: MagicMock, tmp_path: Path, mode: str, action: str
) -> None:
    """A store-layer refusal (valid name, symlinked dir) maps to the validation envelope."""
    from linc_crucible.baseline_tools import crucible_baseline
    from linc_crucible.core.baseline import BaselineManager
    from linc_crucible.core.tree_capture import TreeCapture

    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "outside"
    (outside / "home" / "20260101T000000Z").mkdir(parents=True)
    (store / "proj").symlink_to(outside, target_is_directory=True)
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + b"\x00" * 16)
    before = sorted(str(p) for p in outside.rglob("*"))
    with (
        patch("linc_crucible.server_context._baseline_manager", BaselineManager(store)),
        patch("linc_crucible.server_context._tree_capture", TreeCapture(store)),
    ):
        result = await crucible_baseline(
            action=action,
            project="proj",
            screen_name="home",
            image_path=str(image),
            mode=mode,
            keep=0,
        )
    assert result["validation_error"] is True
    assert result["error_category"] == "validation"
    assert result["error"].startswith("invalid project: resolves outside the store")
    assert sorted(str(p) for p in outside.rglob("*")) == before


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("project", ["/nonexistent/victim", "../x", "a b", "a" * 129])
@patch("linc_crucible.server_context._ensure_initialized")
@patch("linc_crucible.server_context._scorer")
def test_crucible_score_hostile_project_validation_error(
    mock_scorer: MagicMock, mock_init: MagicMock, project: str, history: bool
) -> None:
    """[test] crucible_score with a hostile project → validation envelope, store untouched."""
    from linc_crucible.assay_tools import crucible_score

    result = crucible_score(project=project, history=history)
    assert result["validation_error"] is True
    assert result["error_category"] == "validation"
    assert result["error"].startswith("invalid project:")
    mock_scorer.load_latest.assert_not_called()
    mock_scorer.load_history.assert_not_called()


@patch("linc_crucible.server_context._ensure_initialized")
def test_crucible_score_store_raise_is_validation_error(
    mock_init: MagicMock, tmp_path: Path
) -> None:
    """A store-layer refusal (symlinked project dir) also maps to the validation envelope."""
    from linc_crucible.assay_tools import crucible_score
    from linc_crucible.core.scorer import QualityScorer

    scores = tmp_path / "scores"
    scores.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (scores / "proj").symlink_to(outside, target_is_directory=True)
    with patch("linc_crucible.server_context._scorer", QualityScorer(scores_dir=scores)):
        result = crucible_score(project="proj")
    assert result["validation_error"] is True
    assert result["error_category"] == "validation"


# ── visual_batch.compare_run ───────────────────────────────────────────────────


def test_compare_run_invalid_stem_records_error(tmp_path: Path) -> None:
    """[test] a stem the allowlist refuses is an ERROR comparison, not a raise."""
    from linc_crucible.core.baseline import BaselineManager
    from linc_crucible.core.bvt.visual_batch import compare_run

    manager = BaselineManager(baselines_dir=tmp_path / "store")
    result = compare_run(manager, "myapp", [str(tmp_path / "-x.png")])
    assert result["comparisons"] == [
        {"screen": "-x", "verdict": "ERROR", "reason": "invalid baseline name"}
    ]
    assert result["aggregate_verdict"] == "ERROR"
    assert result["available"] is True
    assert not (tmp_path / "store").exists()


def test_compare_run_invalid_project_records_error_per_screen(tmp_path: Path) -> None:
    from linc_crucible.core.baseline import BaselineManager
    from linc_crucible.core.bvt.visual_batch import compare_run

    manager = BaselineManager(baselines_dir=tmp_path / "store")
    result = compare_run(manager, "../x", [str(tmp_path / "a.png"), str(tmp_path / "b.png")])
    assert [c["verdict"] for c in result["comparisons"]] == ["ERROR", "ERROR"]
    assert [c["screen"] for c in result["comparisons"]] == ["a", "b"]
    assert result["aggregate_verdict"] == "ERROR"


# ── Exact-value pins (pr-diff-mutation survivors) ──────────────────────────────


class TestExactMessages:
    def test_module_constants_are_pinned(self) -> None:
        assert _ECHO_LIMIT == 40
        assert _FIELDS == ("project", "screen_name")
        assert [rule for _, rule in _RULES] == [
            "must not be empty",
            "must be at most 128 characters",
            "must not contain a NUL byte",
            "must not contain a path separator",
            "must not contain '..'",
            "must not contain whitespace",
            "must not start with '.' or '-'",
            "may only contain letters, digits, '.', '_' and '-'",
        ]

    def test_rule_order_first_match_wins(self) -> None:
        """A value breaking several rules is blamed on the earliest one in ``_RULES``."""
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name("-a b/..", "project")
        assert excinfo.value.rule == "must not contain a path separator"

    def test_field_labels_follow_store_order(self, tmp_path: Path) -> None:
        for index, field in enumerate(_FIELDS):
            names = ["ok"] * index + ["-bad"]
            with pytest.raises(InvalidStoreName) as excinfo:
                contained_dir(tmp_path, *names)
            assert excinfo.value.field == field

    def test_non_string_preview_is_its_type_name(self) -> None:
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name(5, "project")
        assert str(excinfo.value) == "invalid project: must be a string (got int)"

    def test_preview_of_exactly_40_chars_has_no_ellipsis(self) -> None:
        value = "-" + "a" * (_ECHO_LIMIT - 1)
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name(value, "project")
        assert str(excinfo.value) == (
            f"invalid project: must not start with '.' or '-' (got '{value}')"
        )

    def test_preview_of_41_chars_is_cut_to_40_plus_ellipsis(self) -> None:
        value = "-" + "a" * _ECHO_LIMIT
        with pytest.raises(InvalidStoreName) as excinfo:
            validate_name(value, "project")
        assert str(excinfo.value) == (
            f"invalid project: must not start with '.' or '-' (got '{value[:_ECHO_LIMIT]}'...)"
        )

    def test_required_project_error_empty_project(self) -> None:
        assert required_project_error("") == {"error": "project is required."}

    def test_required_project_error_valid_and_invalid(self) -> None:
        assert required_project_error("myapp") is None
        assert required_project_error("a b") == {
            "error": "invalid project: must not contain whitespace (got 'a b')",
            "error_category": "validation",
            "validation_error": True,
        }

    def test_required_names_error_empty_project(self) -> None:
        assert required_names_error("", "home") == {"error": "project is required."}

    def test_required_names_error_checks_screen_after_valid_project(self) -> None:
        assert required_names_error("myapp", "home") is None
        assert required_names_error("myapp", "") is None
        assert required_names_error("myapp", "a b") == {
            "error": "invalid screen_name: must not contain whitespace (got 'a b')",
            "error_category": "validation",
            "validation_error": True,
        }

    def test_store_error_plain_os_error(self) -> None:
        assert store_error(OSError("disk full")) == {"error": "disk full"}

    def test_store_error_refused_name_is_validation_envelope(self) -> None:
        exc = InvalidStoreName("project", "must not be empty", "")
        assert store_error(exc) == {
            "error": "invalid project: must not be empty (got '')",
            "error_category": "validation",
            "validation_error": True,
        }


class TestCrucibleScoreExact:
    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._scorer")
    def test_history_limit_is_passed_through(
        self, mock_scorer: MagicMock, mock_init: MagicMock, tmp_path: Path
    ) -> None:
        from linc_crucible.assay_tools import crucible_score

        mock_scorer.scores_dir = tmp_path
        mock_scorer.load_history.return_value = []
        crucible_score(project="proj", history=True, limit=3)
        mock_scorer.load_history.assert_called_once_with("proj", limit=3)

    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._scorer")
    def test_empty_history_inventory_is_exact(
        self, mock_scorer: MagicMock, mock_init: MagicMock, tmp_path: Path
    ) -> None:
        from linc_crucible.assay_tools import crucible_score

        mock_scorer.scores_dir = tmp_path
        mock_scorer.load_history.return_value = []
        result = crucible_score(project="proj", history=True, limit=3)
        assert result == {
            "project": "proj",
            "count": 0,
            "scores": [],
            "empty": True,
            "store_root": str(tmp_path),
            "store_exists": True,
            "message": f"No scores stored for project 'proj' (store {tmp_path} exists).",
        }

    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._scorer")
    def test_no_latest_inventory_is_exact(
        self, mock_scorer: MagicMock, mock_init: MagicMock, tmp_path: Path
    ) -> None:
        from linc_crucible.assay_tools import crucible_score

        mock_scorer.scores_dir = tmp_path
        mock_scorer.load_latest.return_value = None
        result = crucible_score(project="proj")
        assert result == {
            "project": "proj",
            "error": "No scores found. Run 'crucible assay proj' first.",
            "error_category": "not_found",
            "empty": True,
            "store_root": str(tmp_path),
            "store_exists": True,
            "message": f"No scores stored for project 'proj' (store {tmp_path} exists).",
        }
        mock_scorer.load_latest.assert_called_once_with("proj")

    @patch("linc_crucible.server_context._ensure_initialized")
    @patch("linc_crucible.server_context._scorer")
    def test_empty_project_is_required(self, mock_scorer: MagicMock, mock_init: MagicMock) -> None:
        from linc_crucible.assay_tools import crucible_score

        assert crucible_score(project="") == {"error": "project is required."}
        mock_scorer.load_latest.assert_not_called()


class TestBaselineManagerExact:
    def test_capture_rejects_bad_screen_before_reading_image(self, tmp_path: Path) -> None:
        """screen_name is validated before the image is looked at (missing image, bad screen)."""
        from linc_crucible.core.baseline import BaselineManager

        manager = BaselineManager(baselines_dir=tmp_path / "store")
        with pytest.raises(InvalidStoreName) as excinfo:
            manager.capture("proj", "-x", tmp_path / "missing.png")
        assert excinfo.value.field == "screen_name"
        assert not (tmp_path / "store").exists()

    def test_compare_dir_is_per_screen(self, tmp_path: Path) -> None:
        from linc_crucible.core.baseline import BaselineManager

        store = tmp_path / "store"
        compare_dir = BaselineManager(baselines_dir=store)._compare_dir("proj", "home")
        assert compare_dir == store / ".compare" / "proj" / "home"
        assert compare_dir.is_dir()

    def test_prune_counts_and_logs_every_removed_dir(self, tmp_path: Path) -> None:
        from linc_crucible.core.baseline import BaselineManager

        store = tmp_path / "store"
        screen = store / "proj" / "home"
        for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
            (screen / stamp).mkdir(parents=True)
        with structlog.testing.capture_logs() as logs:
            pruned = BaselineManager(baselines_dir=store).prune("proj", "home", keep=0)
        assert pruned == 3
        assert [e["event"] for e in logs] == ["baseline_pruned"] * 3
        assert list(screen.iterdir()) == []


class TestTreeCapturePruneExact:
    def test_prune_counts_and_logs_every_removed_dir(self, tmp_path: Path) -> None:
        from linc_crucible.core.tree_capture import TreeCapture

        store = tmp_path / "store"
        screen = store / "proj" / "home"
        for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
            (screen / stamp).mkdir(parents=True)
            (screen / stamp / "a11y_tree.json").write_text("[]")
        with structlog.testing.capture_logs() as logs:
            pruned = TreeCapture(baselines_dir=store).prune("proj", "home", keep=0)
        assert pruned == 3
        assert [e["event"] for e in logs] == ["a11y_baseline_pruned"] * 3
        assert list(screen.iterdir()) == []


class TestCompareRunArgs:
    def test_manager_compare_gets_project_screen_and_path(self) -> None:
        from linc_crucible.core.bvt.visual_batch import compare_run

        manager = MagicMock()
        manager.compare.return_value = {"verdict": "PASS"}
        result = compare_run(manager, "proj", ["shots/home.png", "shots/settings.png"])
        assert manager.compare.call_args_list == [
            call("proj", "home", Path("shots/home.png")),
            call("proj", "settings", Path("shots/settings.png")),
        ]
        assert result["comparisons"] == [
            {"screen": "home", "verdict": "PASS"},
            {"screen": "settings", "verdict": "PASS"},
        ]
