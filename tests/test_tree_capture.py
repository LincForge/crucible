"""Tests for a11y tree baseline capture and retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from linc_device_kit.models import ElementNode, Rect, SemanticRole

from linc_crucible.core.tree_capture import A11yBaselineEntry, TreeCapture


@pytest.fixture
def baselines_dir(tmp_path):
    return tmp_path / "baselines"


@pytest.fixture
def capture(baselines_dir):
    return TreeCapture(baselines_dir=baselines_dir)


def _sample_elements() -> list[ElementNode]:
    return [
        ElementNode(
            id="btn_submit",
            text="Submit",
            description="Submit form",
            role=SemanticRole.BUTTON.value,
            bounds=Rect(x=100, y=200, width=200, height=80),
            clickable=True,
        ),
        ElementNode(
            id="txt_title",
            text="Hello World",
            role=SemanticRole.TEXT.value,
            bounds=Rect(x=0, y=0, width=1080, height=60),
        ),
    ]


# ── Test 1: store() creates a11y_tree.json + metadata.json in correct directory ──


def test_store_creates_files(capture, baselines_dir):
    elements = _sample_elements()
    entry = capture.store(
        "myapp", "home", elements, device_info="Pixel 8 Pro", app_version="1.0.0"
    )

    tree_path = Path(entry.tree_path)
    assert tree_path.exists(), "a11y_tree.json not created"
    metadata_path = tree_path.parent / "metadata.json"
    assert metadata_path.exists(), "metadata.json not created"

    # Verify path structure: baselines_dir/project/screen_name/<timestamp>/a11y_tree.json
    parts = tree_path.parts
    assert parts[-1] == "a11y_tree.json"
    assert parts[-3] == "home"
    assert parts[-4] == "myapp"


# ── Test 2: store() with empty elements list works ──


def test_store_empty_elements(capture):
    entry = capture.store("myapp", "splash", [])
    assert entry.element_count == 0
    tree_data = json.loads(Path(entry.tree_path).read_text())
    assert tree_data == []


# ── Test 3: get_latest() returns most recent entry for a screen ──


def test_get_latest_returns_entry(capture):
    elements = _sample_elements()
    capture.store("myapp", "home", elements)
    latest = capture.get_latest("myapp", "home")
    assert latest is not None
    assert latest.screen_name == "home"
    assert latest.project == "myapp"
    assert latest.element_count == 2


# ── Test 4: get_latest() returns None when no baselines exist ──


def test_get_latest_none_when_empty(capture):
    result = capture.get_latest("nonexistent", "home")
    assert result is None


# ── Test 5: load_elements() deserializes back to ElementNode list with correct fields ──


def test_load_elements_roundtrip(capture):
    elements = _sample_elements()
    entry = capture.store("myapp", "home", elements)
    loaded = capture.load_elements(entry)

    assert len(loaded) == 2

    btn = next(e for e in loaded if e.id == "btn_submit")
    assert btn.text == "Submit"
    assert btn.description == "Submit form"
    assert btn.role == SemanticRole.BUTTON.value
    assert btn.bounds.x == 100
    assert btn.bounds.y == 200
    assert btn.bounds.width == 200
    assert btn.bounds.height == 80
    assert btn.clickable is True

    txt = next(e for e in loaded if e.id == "txt_title")
    assert txt.text == "Hello World"
    assert txt.role == SemanticRole.TEXT.value
    assert txt.bounds.width == 1080


# ── Test 6: Multiple baselines → get_latest() returns newest ──


def test_get_latest_returns_newest(capture):
    elements = _sample_elements()
    capture.store("myapp", "home", elements[:1])
    time.sleep(1)  # ensure different timestamp
    entry2 = capture.store("myapp", "home", elements)

    latest = capture.get_latest("myapp", "home")
    assert latest is not None
    assert latest.capture_time == entry2.capture_time
    assert latest.element_count == 2


# ── Test 7: list_entries() returns all entries for a project ──


def test_list_entries_all(capture):
    capture.store("myapp", "home", _sample_elements())
    time.sleep(1)
    capture.store("myapp", "settings", _sample_elements()[:1])

    entries = capture.list_entries("myapp")
    assert len(entries) == 2
    # newest-first
    assert entries[0].capture_time >= entries[1].capture_time


# ── Test 8: list_entries() with screen_name filter ──


def test_list_entries_filtered_by_screen(capture):
    capture.store("myapp", "home", _sample_elements())
    capture.store("myapp", "settings", _sample_elements()[:1])

    home_entries = capture.list_entries("myapp", screen_name="home")
    assert len(home_entries) == 1
    assert home_entries[0].screen_name == "home"


# ── Test 9: prune() keeps N newest, removes rest ──


def test_prune_keeps_n_newest(capture):
    for _ in range(3):
        capture.store("myapp", "home", _sample_elements())
        time.sleep(1)

    pruned = capture.prune("myapp", screen_name="home", keep=2)
    assert pruned == 1

    remaining = capture.list_entries("myapp", screen_name="home")
    assert len(remaining) == 2


# ── Test 10: A11yBaselineEntry.to_dict() / from_dict() roundtrip ──


def test_a11y_baseline_entry_roundtrip():
    entry = A11yBaselineEntry(
        project="myapp",
        screen_name="home",
        capture_time="20260413T120000Z",
        tree_path="/some/path/a11y_tree.json",
        device_info="Pixel 8 Pro",
        app_version="1.2.3",
        element_count=5,
    )
    d = entry.to_dict()
    assert d["mode"] == "a11y"
    assert d["project"] == "myapp"
    assert d["element_count"] == 5

    restored = A11yBaselineEntry.from_dict(d)
    assert restored.project == entry.project
    assert restored.screen_name == entry.screen_name
    assert restored.capture_time == entry.capture_time
    assert restored.tree_path == entry.tree_path
    assert restored.device_info == entry.device_info
    assert restored.app_version == entry.app_version
    assert restored.element_count == entry.element_count


# ── Additional: only a11y dirs (with a11y_tree.json) listed, not screenshot dirs ──


def test_list_entries_ignores_screenshot_dirs(capture, baselines_dir):
    # Create a fake screenshot baseline dir (no a11y_tree.json)
    fake_dir = baselines_dir / "myapp" / "home" / "20260101T000000Z"
    fake_dir.mkdir(parents=True)
    (fake_dir / "screenshot.png").write_bytes(b"fakepng")
    meta = {"project": "myapp", "screen_name": "home", "capture_time": "20260101T000000Z"}
    (fake_dir / "metadata.json").write_text(json.dumps(meta))

    # Also store a real a11y entry
    capture.store("myapp", "home", _sample_elements())

    entries = capture.list_entries("myapp", screen_name="home")
    assert len(entries) == 1
    assert entries[0].element_count == 2


# ── Additional: metadata.json includes device_info and app_version ──


def test_store_persists_metadata_fields(capture):
    entry = capture.store(
        "myapp", "login", _sample_elements(), device_info="Galaxy S10e", app_version="2.5.1"
    )
    meta_path = Path(entry.tree_path).parent / "metadata.json"
    meta = json.loads(meta_path.read_text())

    assert meta["device_info"] == "Galaxy S10e"
    assert meta["app_version"] == "2.5.1"
    assert meta["mode"] == "a11y"
    assert meta["element_count"] == 2


# ── Store containment (R4 High 1 + Medium 6) ─────────────────────────────────


def _listing(root: Path) -> list[tuple[str, list[str], list[str]]]:
    import os

    return [(d, sorted(ds), sorted(fs)) for d, ds, fs in os.walk(root)]


def _a11y_victim(root: Path) -> Path:
    for ts in ("20260101T000000Z", "20260102T000000Z"):
        d = root / "home" / ts
        d.mkdir(parents=True)
        (d / "a11y_tree.json").write_text("[]")
        (d / "metadata.json").write_text("{}")
    return root


def _attempt(fn, *args, **kwargs) -> BaseException | None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # the contract asserts the exact type below
        return exc
    return None


def _assert_refused(exc: BaseException | None, field: str) -> None:
    from linc_crucible.core.store_names import InvalidStoreName

    assert isinstance(exc, InvalidStoreName), f"expected InvalidStoreName, got {exc!r}"
    assert exc.field == field


def test_contract_a11y_prune_absolute_project_deletes_nothing(capture, tmp_path):
    """[contract] a11y prune keep=0 with an absolute project path deletes nothing."""
    victim = _a11y_victim(tmp_path / "victim")
    before = _listing(victim)

    exc = _attempt(capture.prune, str(victim), keep=0)

    assert _listing(victim) == before
    _assert_refused(exc, "project")


def test_contract_a11y_store_dotdot_writes_nothing_outside(tmp_path):
    """[contract] a11y store with ``../../escaped`` creates nothing outside the store."""
    store = tmp_path / "a" / "b" / "store"
    before = _listing(tmp_path)

    exc = _attempt(TreeCapture(store).store, "../../escaped", "home", _sample_elements())

    assert not (tmp_path / "escaped").exists()
    assert _listing(tmp_path) == before
    _assert_refused(exc, "project")


@pytest.mark.parametrize(
    ("project", "screen_name", "field"),
    [("../x", "home", "project"), ("ok", "a b", "screen_name"), ("ok", "", "screen_name")],
)
def test_a11y_read_paths_with_invalid_name_raise(capture, project, screen_name, field):
    _assert_refused(_attempt(capture.get_latest, project, screen_name), field)


@pytest.mark.parametrize("op", ["list_entries", "prune"])
def test_a11y_list_and_prune_invalid_screen_raise(capture, op):
    _assert_refused(_attempt(getattr(capture, op), "ok", screen_name="/abs"), "screen_name")


def test_contract_a11y_symlinked_project_every_op_refused(tmp_path):
    """[contract] ``<store>/proj`` → outside: store/prune/list/latest refused, target intact."""
    store = tmp_path / "store"
    store.mkdir()
    outside = _a11y_victim(tmp_path / "outside")
    (store / "proj").symlink_to(outside, target_is_directory=True)
    before = _listing(outside)
    tc = TreeCapture(baselines_dir=store)

    for exc in (
        _attempt(tc.store, "proj", "home", _sample_elements()),
        _attempt(tc.prune, "proj", keep=0),
        _attempt(tc.prune, "proj", screen_name="home", keep=0),
        _attempt(tc.list_entries, "proj"),
        _attempt(tc.get_latest, "proj", "home"),
    ):
        _assert_refused(exc, "project")
    assert _listing(outside) == before


def test_contract_a11y_symlinked_timestamp_dir_skipped_by_prune(capture, baselines_dir, tmp_path):
    """[contract] a symlinked a11y timestamp dir is skipped by prune, not followed."""
    capture.store("proj", "home", _sample_elements())
    outside = tmp_path / "outside-ts"
    outside.mkdir()
    (outside / "a11y_tree.json").write_text("[]")
    (outside / "metadata.json").write_text("{}")
    link = baselines_dir / "proj" / "home" / "99999999T999999Z"
    link.symlink_to(outside, target_is_directory=True)

    pruned = capture.prune("proj", keep=0)

    assert pruned == 1
    assert link.is_symlink()
    assert sorted(p.name for p in outside.iterdir()) == ["a11y_tree.json", "metadata.json"]


def test_a11y_valid_layout_unchanged(capture, baselines_dir):
    entry = capture.store("My.App_2-x", "default", _sample_elements())
    ts_dir = baselines_dir / "My.App_2-x" / "default" / entry.capture_time
    assert entry.tree_path == str(ts_dir / "a11y_tree.json")
    assert sorted(p.name for p in ts_dir.iterdir()) == ["a11y_tree.json", "metadata.json"]
