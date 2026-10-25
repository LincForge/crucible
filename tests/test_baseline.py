"""Tests for baseline manager — capture, retrieval, listing, and pruning."""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from linc_crucible.core.asset_config import _default_assets_dir
from linc_crucible.core.baseline import (
    BaselineEntry,
    BaselineManager,
    _default_baselines_dir,
    _read_png_dimensions,
)


def _make_png(path: Path, width: int = 100, height: int = 50) -> Path:
    """Write a minimal valid PNG file with the given dimensions."""
    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: length(13) + "IHDR" + width(4) + height(4) + bit_depth(1) + color_type(1)
    # + compression(1) + filter(1) + interlace(1) + CRC(4)
    ihdr_data = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    ihdr_crc = b"\x00\x00\x00\x00"  # Doesn't matter for our parser
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc
    path.write_bytes(signature + ihdr_chunk)
    return path


@pytest.fixture()
def manager(tmp_path: Path) -> BaselineManager:
    """BaselineManager backed by a temp directory."""
    return BaselineManager(baselines_dir=tmp_path)


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """A minimal valid PNG at tmp_path/sample.png."""
    return _make_png(tmp_path / "sample.png")


class TestReadPngDimensions:
    def test_reads_dimensions(self, tmp_path: Path) -> None:
        img = _make_png(tmp_path / "test.png", width=320, height=640)
        w, h = _read_png_dimensions(img)
        assert w == 320
        assert h == 640

    def test_invalid_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a png")
        with pytest.raises(ValueError, match="Not a valid PNG"):
            _read_png_dimensions(bad)


class TestBaselineEntry:
    def test_round_trip(self) -> None:
        entry = BaselineEntry(
            project="TestApp",
            screen_name="home",
            capture_time="20260318T180552Z",
            screenshot_path="/tmp/screenshot.png",
            device_info="Pixel 7",
            app_version="1.0.0",
            width=1080,
            height=2400,
        )
        data = entry.to_dict()
        restored = BaselineEntry.from_dict(data)
        assert restored == entry

    def test_from_dict_defaults(self) -> None:
        entry = BaselineEntry.from_dict({})
        assert entry.project == ""
        assert entry.width == 0


class TestBaselineCapture:
    def test_capture_creates_files(self, manager: BaselineManager, sample_image: Path) -> None:
        entry = manager.capture("MyApp", "home", sample_image)
        screenshot = Path(entry.screenshot_path)
        assert screenshot.exists()
        assert (screenshot.parent / "metadata.json").exists()

    def test_capture_writes_correct_metadata(
        self, manager: BaselineManager, sample_image: Path
    ) -> None:
        entry = manager.capture(
            "MyApp", "settings", sample_image, device_info="Pixel 7", app_version="2.0"
        )
        metadata_path = Path(entry.screenshot_path).parent / "metadata.json"
        with open(metadata_path) as f:
            data = json.load(f)
        assert data["project"] == "MyApp"
        assert data["screen_name"] == "settings"
        assert data["device_info"] == "Pixel 7"
        assert data["app_version"] == "2.0"

    def test_capture_nonexistent_image_raises(self, manager: BaselineManager) -> None:
        with pytest.raises(FileNotFoundError):
            manager.capture("MyApp", "home", Path("/nonexistent/image.png"))

    def test_capture_reads_png_dimensions(self, manager: BaselineManager, tmp_path: Path) -> None:
        img = _make_png(tmp_path / "sized.png", width=1080, height=2400)
        entry = manager.capture("MyApp", "home", img)
        assert entry.width == 1080
        assert entry.height == 2400


class TestBaselineGetLatest:
    def test_returns_none_when_empty(self, manager: BaselineManager) -> None:
        assert manager.get_latest("NoSuchApp", "home") is None

    def test_returns_newest_entry(self, manager: BaselineManager, sample_image: Path) -> None:
        manager.capture("MyApp", "home", sample_image, app_version="1.0")
        time.sleep(1.1)  # Ensure different timestamp (second granularity)
        entry2 = manager.capture("MyApp", "home", sample_image, app_version="2.0")
        latest = manager.get_latest("MyApp", "home")
        assert latest is not None
        assert latest.app_version == entry2.app_version


class TestBaselineList:
    def test_lists_all_for_project(self, manager: BaselineManager, sample_image: Path) -> None:
        manager.capture("MyApp", "home", sample_image)
        time.sleep(1.1)
        manager.capture("MyApp", "settings", sample_image)
        entries = manager.list_baselines("MyApp")
        assert len(entries) == 2
        screens = {e.screen_name for e in entries}
        assert screens == {"home", "settings"}

    def test_filters_by_screen(self, manager: BaselineManager, sample_image: Path) -> None:
        manager.capture("MyApp", "home", sample_image)
        time.sleep(1.1)
        manager.capture("MyApp", "settings", sample_image)
        entries = manager.list_baselines("MyApp", screen_name="home")
        assert len(entries) == 1
        assert entries[0].screen_name == "home"

    def test_empty_project_returns_empty(self, manager: BaselineManager) -> None:
        assert manager.list_baselines("NoSuchApp") == []


class TestBaselinePrune:
    def test_prune_keeps_n_newest(self, manager: BaselineManager, sample_image: Path) -> None:
        for i in range(4):
            manager.capture("MyApp", "home", sample_image, app_version=str(i))
            time.sleep(1.1)
        pruned = manager.prune("MyApp", screen_name="home", keep=2)
        assert pruned == 2
        remaining = manager.list_baselines("MyApp", screen_name="home")
        assert len(remaining) == 2

    def test_prune_returns_count(self, manager: BaselineManager, sample_image: Path) -> None:
        for _ in range(3):
            manager.capture("MyApp", "home", sample_image)
            time.sleep(1.1)
        count = manager.prune("MyApp", screen_name="home", keep=1)
        assert count == 2

    def test_prune_noop_when_under_limit(
        self, manager: BaselineManager, sample_image: Path
    ) -> None:
        manager.capture("MyApp", "home", sample_image)
        assert manager.prune("MyApp", screen_name="home", keep=5) == 0

    def test_prune_specific_screen(self, manager: BaselineManager, sample_image: Path) -> None:
        for _ in range(3):
            manager.capture("MyApp", "home", sample_image)
            time.sleep(1.1)
        for _ in range(3):
            manager.capture("MyApp", "settings", sample_image)
            time.sleep(1.1)
        pruned = manager.prune("MyApp", screen_name="home", keep=1)
        assert pruned == 2
        # Settings should be untouched
        assert len(manager.list_baselines("MyApp", screen_name="settings")) == 3

    def test_prune_empty_project_returns_zero(self, manager: BaselineManager) -> None:
        assert manager.prune("NoSuchApp") == 0


def _real_png(path: Path, *, invert: bool = False, size: int = 64) -> Path:
    """A real gradient PNG that PIL/skimage can open (unlike _make_png)."""
    img = Image.new("L", (size, size))
    px = img.load()
    for x in range(size):
        v = int(255 * x / (size - 1))
        if invert:
            v = 255 - v
        for y in range(size):
            px[x, y] = v
    img.save(path, "PNG")
    return path


class TestCompare:
    def test_round_trip_identical_is_pass(self, manager: BaselineManager, tmp_path: Path) -> None:
        # AC1: golden captured by manager IS the file compare reads.
        manager.capture("demo", "home", _real_png(tmp_path / "golden.png"))
        candidate = _real_png(tmp_path / "candidate.png")
        res = manager.compare("demo", "home", candidate)
        assert res["ok"] is True
        assert res["verdict"] == "PASS"
        assert res["score"] == pytest.approx(1.0, abs=1e-6)

    def test_round_trip_divergent_is_fail(self, manager: BaselineManager, tmp_path: Path) -> None:
        manager.capture("demo", "home", _real_png(tmp_path / "golden.png"))
        candidate = _real_png(tmp_path / "candidate.png", invert=True)
        res = manager.compare("demo", "home", candidate)
        assert res["verdict"] == "FAIL"

    def test_no_baseline_is_benign_surfaced_skip(
        self, manager: BaselineManager, tmp_path: Path
    ) -> None:
        res = manager.compare("demo", "never-captured", _real_png(tmp_path / "c.png"))
        assert res["ok"] is True
        assert res["skipped"] is True
        assert res["no_baseline"] is True
        assert res.get("deps_missing") is not True  # distinct from deps-missing

    def test_no_baseline_carries_unbaselined_verdict(
        self, manager: BaselineManager, tmp_path: Path
    ) -> None:
        # A no-baseline compare must be distinguishable from "nothing ran" at the
        # aggregation layer — it now carries an explicit verdict of its own.
        res = manager.compare("demo", "never_captured", _real_png(tmp_path / "c.png"))
        assert res["verdict"] == "UNBASELINED"
        assert res["ok"] is True
        assert res["skipped"] is True
        assert res["no_baseline"] is True
        assert res["available"] is True

    def test_compare_respects_linc_home(self, tmp_path: Path, monkeypatch) -> None:
        # AC4: with LINC_HOME set and no ctor arg, capture+compare both resolve under it.
        monkeypatch.delenv("CRUCIBLE_BASELINES_DIR", raising=False)
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        mgr = BaselineManager()
        assert str(tmp_path) in str(mgr._baselines_dir)
        mgr.capture("demo", "home", _real_png(tmp_path / "g.png"))
        res = mgr.compare("demo", "home", _real_png(tmp_path / "c.png"))
        assert res["verdict"] == "PASS"

    def test_compare_artifacts_do_not_pollute_get_latest(
        self, manager: BaselineManager, tmp_path: Path
    ) -> None:
        manager.capture("demo", "home", _real_png(tmp_path / "g.png"))
        before = manager.get_latest("demo", "home")
        manager.compare("demo", "home", _real_png(tmp_path / "c.png", invert=True))
        after = manager.get_latest("demo", "home")
        assert after is not None
        assert before is not None
        assert after.capture_time == before.capture_time  # .compare/ did not become "latest"

    def test_compare_deps_missing_propagates_loud(
        self, manager: BaselineManager, tmp_path: Path, monkeypatch
    ) -> None:
        manager.capture("demo", "home", _real_png(tmp_path / "g.png"))
        monkeypatch.setitem(sys.modules, "skimage", None)
        monkeypatch.setitem(sys.modules, "skimage.metrics", None)
        res = manager.compare("demo", "home", _real_png(tmp_path / "c.png"))
        assert res["ok"] is False
        assert res["available"] is False
        assert res["deps_missing"] is True


class TestBaselineProvenance:
    def test_normal_capture_has_empty_provenance(
        self, manager: BaselineManager, sample_image: Path
    ) -> None:
        entry = manager.capture("MyApp", "home", sample_image)
        assert entry.source == ""
        assert entry.source_score is None
        assert entry.promoted_at == ""

    def test_promote_stamps_provenance(self, manager: BaselineManager, sample_image: Path) -> None:
        entry = manager.promote("MyApp", "home", sample_image, source_score=0.97)
        assert entry.source == "promoted"
        assert entry.source_score == 0.97
        assert entry.promoted_at != ""
        # provenance is persisted to metadata.json (single serializer, no drift)
        metadata = json.loads((Path(entry.screenshot_path).parent / "metadata.json").read_text())
        assert metadata["source"] == "promoted"
        assert metadata["source_score"] == 0.97

    def test_promoted_entry_is_resolvable_as_latest(
        self, manager: BaselineManager, sample_image: Path
    ) -> None:
        manager.promote("MyApp", "home", sample_image, source_score=0.91)
        latest = manager.get_latest("MyApp", "home")
        assert latest is not None
        assert latest.source == "promoted"
        assert latest.source_score == 0.91

    def test_provenance_round_trips(self) -> None:
        entry = BaselineEntry(
            project="A",
            screen_name="s",
            capture_time="20260612T000000Z",
            screenshot_path="/tmp/x.png",
            device_info="",
            app_version="",
            width=10,
            height=10,
            source="promoted",
            source_score=0.88,
            promoted_at="20260612T010000Z",
        )
        assert BaselineEntry.from_dict(entry.to_dict()) == entry


class TestDefaultDirConvention:
    """Issue #117: baselines and assets must share one root+segment convention."""

    def test_unified_root_without_linc_home(self, monkeypatch) -> None:
        # AC1: with LINC_HOME unset, both defaults resolve under the SAME base
        # directory (same parent), not baselines->~/.crucible vs assets->~/.linc/crucible.
        monkeypatch.delenv("LINC_HOME", raising=False)
        baselines_dir = _default_baselines_dir()
        assets_dir = _default_assets_dir()
        assert baselines_dir.parent == assets_dir.parent
        assert baselines_dir.parent.name == "crucible"

    def test_unified_root_with_linc_home(self, tmp_path: Path, monkeypatch) -> None:
        # AC2: with LINC_HOME set, both defaults resolve under $LINC_HOME with
        # the SAME segment convention (no .crucible vs crucible split).
        monkeypatch.setenv("LINC_HOME", str(tmp_path))
        baselines_dir = _default_baselines_dir()
        assets_dir = _default_assets_dir()
        assert baselines_dir.parent == assets_dir.parent
        assert baselines_dir.parent == tmp_path / "crucible"
        assert str(tmp_path) in str(baselines_dir)
        assert str(tmp_path) in str(assets_dir)


# ── Store containment (R4 High 1 + Medium 6) ─────────────────────────────────


def _listing(root: Path) -> list[tuple[str, list[str], list[str]]]:
    """Deterministic ``os.walk`` listing used as a before/after fingerprint."""
    import os

    return [(d, sorted(ds), sorted(fs)) for d, ds, fs in os.walk(root)]


def _victim_tree(root: Path) -> Path:
    """A directory OUTSIDE the store laid out like ``<screen>/<timestamp>/metadata.json``."""
    for screen in ("home", "settings"):
        for ts in ("20260101T000000Z", "20260102T000000Z"):
            d = root / screen / ts
            d.mkdir(parents=True)
            (d / "metadata.json").write_text("{}")
            (d / "precious.txt").write_text("do not delete")
    return root


def _attempt(fn, *args, **kwargs) -> BaseException | None:
    """Run ``fn`` and return the exception it raised (or None)."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # the contract asserts the exact type below
        return exc
    return None


def _assert_refused(exc: BaseException | None, field: str) -> None:
    from linc_crucible.core.store_names import InvalidStoreName

    assert isinstance(exc, InvalidStoreName), f"expected InvalidStoreName, got {exc!r}"
    assert exc.field == field


class TestStoreContainment:
    def test_contract_trav_prune_absolute_project_deletes_nothing(self, tmp_path: Path) -> None:
        """[contract] trav.py: prune keep=0 with an absolute project path deletes nothing."""
        victim = _victim_tree(tmp_path / "victim")
        before = _listing(victim)
        manager = BaselineManager(baselines_dir=tmp_path / "store")

        exc = _attempt(manager.prune, str(victim), keep=0)

        assert _listing(victim) == before
        _assert_refused(exc, "project")

    def test_contract_trav_prune_absolute_screen_deletes_nothing(self, tmp_path: Path) -> None:
        victim = _victim_tree(tmp_path / "victim")
        before = _listing(victim)
        manager = BaselineManager(baselines_dir=tmp_path / "store")

        exc = _attempt(manager.prune, "proj", screen_name=str(victim / "home"), keep=0)

        assert _listing(victim) == before
        _assert_refused(exc, "screen_name")

    def test_contract_capture_dotdot_project_writes_nothing_outside(self, tmp_path: Path) -> None:
        """[contract] capture ``../../../escaped`` creates no directory outside the store."""
        store = tmp_path / "a" / "b" / "c" / "store"
        image = _make_png(tmp_path / "shot.png")
        before = _listing(tmp_path)
        manager = BaselineManager(baselines_dir=store)

        exc = _attempt(manager.capture, "../../../escaped", "home", image)

        assert not (tmp_path / "escaped").exists()
        assert _listing(tmp_path) == before
        _assert_refused(exc, "project")

    def test_contract_capture_dotdot_screen_writes_nothing_outside(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        image = _make_png(tmp_path / "shot.png")
        before = _listing(tmp_path)

        exc = _attempt(BaselineManager(store).capture, "proj", "../../escaped", image)

        assert _listing(tmp_path) == before
        _assert_refused(exc, "screen_name")

    @pytest.mark.parametrize("op", ["get_latest", "list_baselines", "prune", "compare"])
    def test_read_paths_with_invalid_name_raise(self, tmp_path: Path, op: str) -> None:
        """Read paths raise on a hostile name — no silent empty result."""
        manager = BaselineManager(baselines_dir=tmp_path / "store")
        calls = {
            "get_latest": lambda: manager.get_latest("../x", "home"),
            "list_baselines": lambda: manager.list_baselines("/abs"),
            "prune": lambda: manager.prune("a b"),
            "compare": lambda: manager.compare("ok", "-x", tmp_path / "c.png"),
        }
        exc = _attempt(calls[op])
        _assert_refused(exc, "screen_name" if op == "compare" else "project")

    def test_promote_hostile_project_refused(self, tmp_path: Path) -> None:
        image = _make_png(tmp_path / "shot.png")
        before = _listing(tmp_path)
        exc = _attempt(BaselineManager(tmp_path / "store").promote, "..", "home", image)
        assert _listing(tmp_path) == before
        _assert_refused(exc, "project")

    def test_contract_symlinked_project_every_op_refused(self, tmp_path: Path) -> None:
        """[contract] ``<store>/proj`` → outside: every op refused, target intact."""
        store = tmp_path / "store"
        store.mkdir()
        outside = _victim_tree(tmp_path / "outside")
        (store / "proj").symlink_to(outside, target_is_directory=True)
        image = _make_png(tmp_path / "shot.png")
        before = _listing(outside)
        manager = BaselineManager(baselines_dir=store)

        for exc in (
            _attempt(manager.capture, "proj", "home", image),
            _attempt(manager.promote, "proj", "home", image),
            _attempt(manager.prune, "proj", keep=0),
            _attempt(manager.prune, "proj", screen_name="home", keep=0),
            _attempt(manager.list_baselines, "proj"),
            _attempt(manager.get_latest, "proj", "home"),
        ):
            _assert_refused(exc, "project")
        assert _listing(outside) == before

    def test_contract_symlinked_timestamp_dir_skipped_by_prune(
        self, tmp_path: Path, sample_image: Path
    ) -> None:
        """[contract] a symlinked timestamp dir in a valid screen is skipped, not followed."""
        store = tmp_path / "store"
        manager = BaselineManager(baselines_dir=store)
        manager.capture("proj", "home", sample_image)
        outside = tmp_path / "outside-ts"
        outside.mkdir()
        (outside / "metadata.json").write_text("{}")
        (outside / "precious.txt").write_text("do not delete")
        link = store / "proj" / "home" / "99999999T999999Z"
        link.symlink_to(outside, target_is_directory=True)

        pruned = manager.prune("proj", keep=0)

        assert pruned == 1
        assert link.is_symlink()
        assert (outside / "precious.txt").read_text() == "do not delete"
        assert sorted(p.name for p in (store / "proj" / "home").iterdir()) == [link.name]

    def test_contract_symlinked_screen_dir_not_followed_by_prune(
        self, tmp_path: Path, sample_image: Path
    ) -> None:
        """A symlinked screen dir under a valid project is never pruned through."""
        store = tmp_path / "store"
        manager = BaselineManager(baselines_dir=store)
        manager.capture("proj", "home", sample_image)
        outside = _victim_tree(tmp_path / "outside")
        before = _listing(outside)
        (store / "proj" / "evil").symlink_to(outside / "home", target_is_directory=True)

        pruned = manager.prune("proj", keep=0)

        assert pruned == 1
        assert _listing(outside) == before

    def test_compare_dir_symlink_refused(self, tmp_path: Path) -> None:
        """``.compare/<project>`` symlinked outside is refused before compute_ssim writes."""
        store = tmp_path / "store"
        manager = BaselineManager(baselines_dir=store)
        manager.capture("proj", "home", _real_png(tmp_path / "golden.png"))
        outside = tmp_path / "outside"
        outside.mkdir()
        (store / ".compare").mkdir()
        (store / ".compare" / "proj").symlink_to(outside, target_is_directory=True)

        exc = _attempt(manager.compare, "proj", "home", _real_png(tmp_path / "c.png"))

        _assert_refused(exc, "project")
        assert list(outside.iterdir()) == []

    def test_valid_layout_unchanged(self, tmp_path: Path, sample_image: Path) -> None:
        manager = BaselineManager(baselines_dir=tmp_path / "store")
        entry = manager.capture("My.App_2-x", "home", sample_image)
        ts_dir = tmp_path / "store" / "My.App_2-x" / "home" / entry.capture_time
        assert entry.screenshot_path == str(ts_dir / "screenshot.png")
        assert sorted(p.name for p in ts_dir.iterdir()) == ["metadata.json", "screenshot.png"]
