"""[contract] structlog processor home-redacts every path-bearing log event."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import logging
import math
import shutil
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import structlog
from click.testing import CliRunner
from structlog.typing import EventDict, WrappedLogger

import linc_crucible
from linc_crucible import _logging
from linc_crucible import fuelcheck as fc
from linc_crucible.cli import cli
from linc_crucible.core import device_policy, forge_integration

PKG_SRC = Path(linc_crucible.__file__).resolve().parent
_REAL_PROBE_ENV = fc.default_probe_env


@pytest.fixture(autouse=True)
def _restore_structlog_and_root_logger() -> Iterator[None]:
    saved_config = structlog.get_config()
    saved_processors = list(saved_config["processors"])
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        saved_config["processors"][:] = saved_processors
        structlog.configure(**saved_config)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _offline_real_env() -> fc.ProbeEnv:
    """Real probe env with only PATH, docker, and nerve network probes stubbed."""
    return replace(
        _REAL_PROBE_ENV(),
        which=lambda _name: None,
        nerve_up=lambda: False,
        docker_ps=lambda: "",
    )


def _install_event_spy() -> list[EventDict]:
    """Attach a spy processor right before the final renderer to record processed EventDicts."""
    captured: list[EventDict] = []

    def _spy(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
        captured.append(dict(event_dict))
        return event_dict

    chain = structlog.get_config()["processors"]
    chain.insert(max(len(chain) - 1, 0), _spy)
    return captured


def _setup_tmp_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "stranger-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LINC_HOME", str(home / ".linc"))
    monkeypatch.setenv("LINC_PROJECTS_DIR", str(home / "projects"))
    for var in ("LINC_DEVICE_FLEET_FILE", "FORGE_PROJECTS_FILE", "CRUCIBLE_FLEET_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(device_policy, "_absent_warned", False)
    return home


def test_contract_fuelcheck_and_load_device_policy_redact_tmp_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] With HOME set to a tmp dir and no device-fleet registry, run
    ``crucible fuelcheck`` (CliRunner) and ``load_device_policy()``: captured stderr and
    structlog events contain no occurrence of the tmp HOME path."""
    home = _setup_tmp_home(monkeypatch, tmp_path)
    _logging.configure_cli_structlog("WARNING")
    events = _install_event_spy()

    with patch.object(fc, "default_probe_env", _offline_real_env):
        result = CliRunner().invoke(cli, ["fuelcheck"])
    assert result.exit_code == 0, result.output
    assert str(home) not in result.stdout
    assert str(home) not in result.stderr

    monkeypatch.setattr(device_policy, "_absent_warned", False)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        policy = device_policy.load_device_policy()
    assert policy.state == "absent"
    stderr_text = err.getvalue()
    assert "device_fleet_absent" in stderr_text
    assert "~/.linc/device-fleet.yaml" in stderr_text
    assert str(home) not in stderr_text

    absent_events = [e for e in events if e.get("event") == "device_fleet_absent"]
    assert absent_events, f"expected device_fleet_absent event, got {events!r}"
    for event in events:
        for value in event.values():
            assert str(home) not in str(value), f"HOME leaked in structlog event: {event!r}"
    assert absent_events[-1]["path"] == "~/.linc/device-fleet.yaml"


def test_contract_forge_integration_duplicate_manifest_warnings_redact_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] Leak sites in core/forge_integration.py:382 and :389 redact HOME."""
    home = _setup_tmp_home(monkeypatch, tmp_path)
    _logging.configure_cli_structlog("WARNING")
    monkeypatch.setattr(
        forge_integration,
        "logger",
        structlog.get_logger().bind(domain="build_pipeline"),
    )
    events = _install_event_spy()

    reader = forge_integration.ForgeProjectReader.__new__(forge_integration.ForgeProjectReader)
    kept_dir = str(home / "projects" / "app")
    worktree_dir = str(home / "projects" / "app-worktree")
    reader._projects = {"app": forge_integration.ForgeProject(name="app", path=kept_dir)}
    seen = {"app"}

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        # 1) Existing canonical path beats worktree candidate -> line 382
        reader._update_existing_project("app", worktree_dir, [], seen)
        # 2) Canonical candidate beats existing worktree path -> line 389
        reader._projects["app"].path = worktree_dir
        reader._update_existing_project("app", kept_dir, [], seen)

    stderr_text = err.getvalue()
    assert "forge_manifest_duplicate_name_kept_existing" in stderr_text
    assert "forge_manifest_duplicate_name_resolved" in stderr_text
    assert str(home) not in stderr_text
    assert "~/projects/app" in stderr_text
    assert "~/projects/app-worktree" in stderr_text

    by_name = {e["event"]: e for e in events}
    assert by_name["forge_manifest_duplicate_name_kept_existing"]["kept_path"] == "~/projects/app"
    assert (
        by_name["forge_manifest_duplicate_name_kept_existing"]["shadowed_path"]
        == "~/projects/app-worktree"
    )
    assert by_name["forge_manifest_duplicate_name_resolved"]["kept_path"] == "~/projects/app"
    assert (
        by_name["forge_manifest_duplicate_name_resolved"]["shadowed_path"]
        == "~/projects/app-worktree"
    )


def _discover_path_logger_calls(
    pkg_root: Path = PKG_SRC,
) -> list[tuple[str, str, list[str]]]:
    """Return (rel_file, event_name, keyword_names) for every path-bearing logger call."""
    calls: list[tuple[str, str, list[str]]] = []
    path_Tokens = ("path", "dir", "file", "cwd", "source")
    for py_file in sorted(pkg_root.rglob("*.py")):
        rel = py_file.relative_to(pkg_root.parent).as_posix()
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"logger", "log"}
            ):
                kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
                if any(any(tok in k for tok in path_Tokens) for k in kw_names):
                    event_name = (
                        node.args[0].value
                        if node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        else f"event_{py_file.stem}_{node.lineno}"
                    )
                    calls.append((rel, event_name, kw_names))
    return calls


def test_contract_sweep_path_logger_calls_and_container_types_redact_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[contract] Emit one event per path-bearing logger call in linc_crucible plus a
    representative event with str, Path, list-of-Path, and dict-of-Path values; none contain
    HOME."""
    home = _setup_tmp_home(monkeypatch, tmp_path)
    _logging.configure_cli_structlog("DEBUG")
    events = _install_event_spy()

    calls = _discover_path_logger_calls()
    assert calls, "expected path-bearing logger calls in linc_crucible"

    err = io.StringIO()
    log = structlog.get_logger()
    with contextlib.redirect_stderr(err):
        for _rel, event_name, kw_names in calls:
            kwargs = {kw: str(home / "sub" / kw) for kw in kw_names}
            log.warning(event_name, **kwargs)

        log.warning(
            "representative_paths",
            as_str=f"{home}/a/file.txt",
            as_path=home / "b" / "file.txt",
            as_list=[home / "c" / "one.txt", f"{home}/c/two.txt"],
            as_dict={"first": home / "d" / "one.txt", "second": f"{home}/d/two.txt"},
        )

    stderr_text = err.getvalue()
    assert str(home) not in stderr_text
    assert len(events) == len(calls) + 1
    for event in events:
        for value in event.values():
            assert str(home) not in str(value), f"HOME leaked in {event!r}"

    rep = events[-1]
    assert rep["as_str"] == "~/a/file.txt"
    assert rep["as_path"] == "~/b/file.txt"
    assert rep["as_list"] == ["~/c/one.txt", "~/c/two.txt"]
    assert rep["as_dict"] == {"first": "~/d/one.txt", "second": "~/d/two.txt"}


def test_contract_sweep_discovers_logger_calls_without_src_tree(tmp_path: Path) -> None:
    """[contract] Path-logger discovery returns non-empty calls when no src/ tree exists."""
    isolated_test = tmp_path / "tests" / Path(__file__).name
    isolated_test.parent.mkdir(parents=True)
    shutil.copy2(__file__, isolated_test)
    assert not (tmp_path / "src").exists()

    spec = importlib.util.spec_from_file_location("_isolated_home_redaction", isolated_test)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calls = mod._discover_path_logger_calls()
    assert calls, "expected path-bearing logger calls when src/ is absent"
    assert calls == _discover_path_logger_calls()

    dev_src = Path(__file__).resolve().parent.parent / "src" / "linc_crucible"
    if dev_src.is_dir():
        assert calls == _discover_path_logger_calls(dev_src)


@pytest.mark.parametrize(
    ("label", "hostile_value"),
    [
        ("bytes", b"/tmp/raw-bytes"),
        ("int", 42),
        ("none", None),
        ("nan", float("nan")),
    ],
    ids=["bytes", "int", "none", "nan"],
)
def test_hostile_non_pathlike_values_pass_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, label: str, hostile_value: Any
) -> None:
    """[test] Hostile: a value that is not a str/PathLike (bytes, int, None, NaN float)
    passes through unchanged without error."""
    _setup_tmp_home(monkeypatch, tmp_path)
    _logging.configure_cli_structlog("WARNING")
    events = _install_event_spy()

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        structlog.get_logger().warning(
            "hostile_event",
            kind=label,
            payload=hostile_value,
            in_list=[hostile_value],
            in_dict={"k": hostile_value},
        )

    assert len(events) == 1
    recorded = events[0]
    if isinstance(hostile_value, float) and math.isnan(hostile_value):
        assert isinstance(recorded["payload"], float) and math.isnan(recorded["payload"])
        assert isinstance(recorded["in_list"][0], float) and math.isnan(recorded["in_list"][0])
        assert isinstance(recorded["in_dict"]["k"], float) and math.isnan(recorded["in_dict"]["k"])
    else:
        assert recorded["payload"] == hostile_value
        assert recorded["in_list"] == [hostile_value]
        assert recorded["in_dict"] == {"k": hostile_value}


def test_unresolvable_home_never_drops_or_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """[test] Hostile: when Path.home() raises RuntimeError, events still pass through."""
    _logging.configure_cli_structlog("WARNING")
    events = _install_event_spy()
    monkeypatch.setattr(
        Path, "home", classmethod(lambda _cls: (_ for _ in ()).throw(RuntimeError("no HOME")))
    )

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        structlog.get_logger().warning("unresolvable_home", path="/some/path")

    assert len(events) == 1
    assert events[0]["path"] == "/some/path"
    assert "unresolvable_home" in err.getvalue()


@pytest.mark.parametrize(
    ("verbose", "expected_level"),
    [(False, logging.INFO), (True, logging.DEBUG)],
    ids=["info", "debug"],
)
def test_configure_stdlib_chain_redacts_home_and_sets_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verbose: bool, expected_level: int
) -> None:
    """[contract] _logging.configure() wires home redaction into the stdlib processor chain."""
    home = _setup_tmp_home(monkeypatch, tmp_path)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)

    root = logging.getLogger()
    stale_handler = logging.NullHandler()
    root.addHandler(stale_handler)

    if verbose:
        _logging.configure(verbose=True)
    else:
        _logging.configure()
    assert root.level == expected_level
    assert stale_handler not in root.handlers
    assert len(root.handlers) == 1

    cfg = structlog.get_config()
    assert cfg["cache_logger_on_first_use"] is True
    assert cfg["wrapper_class"] is structlog.stdlib.BoundLogger
    assert isinstance(cfg["logger_factory"], structlog.stdlib.LoggerFactory)
    assert cfg["processors"][-2] is _logging.redact_home_processor
    assert cfg["processors"][-1] is structlog.stdlib.ProcessorFormatter.wrap_for_formatter

    events = _install_event_spy()
    structlog.get_logger().warning("stdlib_warning", path=home / "secret" / "token.txt")
    output = err.getvalue()
    assert str(home) not in output
    assert "~/secret/token.txt" in output
    assert "T" in output
    assert events[-1]["path"] == "~/secret/token.txt"


def test_configure_cli_structlog_idempotent_and_level_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[test] configure_cli_structlog inserts redact_home_processor once before renderer and
    resolves level from arg, LINC_LOG_LEVEL, or WARNING fallback."""
    chain = structlog.get_config()["processors"]
    chain[:] = [p for p in chain if p is not _logging.redact_home_processor]

    monkeypatch.delenv("LINC_LOG_LEVEL", raising=False)
    _logging.configure_cli_structlog()
    _logging.configure_cli_structlog("info")
    assert chain.count(_logging.redact_home_processor) == 1
    assert chain[-2] is _logging.redact_home_processor

    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    structlog.get_logger().info("info_via_arg")
    assert "info_via_arg" in err.getvalue()

    monkeypatch.setenv("LINC_LOG_LEVEL", "info")
    _logging.configure_cli_structlog()
    err_env = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err_env)
    structlog.get_logger().info("info_via_env")
    assert "info_via_env" in err_env.getvalue()

    monkeypatch.setenv("LINC_LOG_LEVEL", "NOT_A_VALID_LEVEL")
    _logging.configure_cli_structlog()
    err_fallback = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err_fallback)
    structlog.get_logger().info("suppressed_info")
    structlog.get_logger().warning("allowed_warning")
    assert "suppressed_info" not in err_fallback.getvalue()
    assert "allowed_warning" in err_fallback.getvalue()



# --- blockers from the 2026-10-29 desk review: tuple/set values, exc_info, exception objects ---


@pytest.mark.parametrize("container", [tuple, set, frozenset], ids=["tuple", "set", "frozenset"])
def test_contract_tuple_and_set_path_values_redact_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, container: type
) -> None:
    """[contract] A tuple/set/frozenset of paths under HOME is redacted in the CLI chain."""
    home = _setup_tmp_home(monkeypatch, tmp_path)
    _logging.configure_cli_structlog("WARNING")
    events = _install_event_spy()

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        structlog.get_logger().warning(
            "container_paths", paths=container([home / "a.txt", f"{home}/b.txt"])
        )

    assert str(home) not in err.getvalue()
    assert sorted(events[-1]["paths"]) == ["~/a.txt", "~/b.txt"]


def _raise_missing_file_under(home: Path) -> None:
    (home / "missing" / "secret.yaml").read_text(encoding="utf-8")


def _configure_chain(chain: str, monkeypatch: pytest.MonkeyPatch, err: io.StringIO) -> None:
    monkeypatch.setattr(sys, "stderr", err)
    if chain == "cli":
        _logging.configure_cli_structlog("WARNING")
    else:
        _logging.configure()


@pytest.mark.parametrize("chain", ["cli", "stdlib"])
def test_contract_log_exception_traceback_redacts_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chain: str
) -> None:
    """[contract] ``log.exception()`` on a FileNotFoundError naming a path under HOME renders
    the traceback with HOME as ``~`` in both the CLI and the stdlib (ProcessorFormatter) chain."""
    home = _setup_tmp_home(monkeypatch, tmp_path)
    err = io.StringIO()
    _configure_chain(chain, monkeypatch, err)

    try:
        _raise_missing_file_under(home)
    except FileNotFoundError:
        structlog.get_logger().exception("load_failed")

    output = err.getvalue()
    assert "load_failed" in output
    assert "FileNotFoundError" in output
    assert "~/missing/secret.yaml" in output
    assert str(home) not in output


@pytest.mark.parametrize("chain", ["cli", "stdlib"])
def test_contract_exception_object_value_redacts_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chain: str
) -> None:
    """[contract] An exception OBJECT logged as a value (``error=exc``) is rendered with HOME
    as ``~`` in both chains, nested in a list too."""
    home = _setup_tmp_home(monkeypatch, tmp_path)
    err = io.StringIO()
    _configure_chain(chain, monkeypatch, err)

    exc = ValueError(f"bad config at {home}/.linc/cfg.yaml")
    structlog.get_logger().warning("bad_config", error=exc, errors=[exc])

    output = err.getvalue()
    assert "bad_config" in output
    assert "~/.linc/cfg.yaml" in output
    assert "ValueError" in output
    assert str(home) not in output


def test_exc_info_without_active_exception_and_unresolvable_home_never_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[test] Hostile: ``exc_info=True`` outside an ``except`` block, and an exception value
    while ``Path.home()`` raises, both still log instead of raising."""
    err = io.StringIO()
    _configure_chain("cli", monkeypatch, err)
    structlog.get_logger().warning("no_active_exception", exc_info=True)
    monkeypatch.setattr(
        Path, "home", classmethod(lambda _cls: (_ for _ in ()).throw(RuntimeError("no HOME")))
    )
    structlog.get_logger().warning("no_home", error=ValueError("/some/path"))
    assert "no_active_exception" in err.getvalue()
    assert "no_home" in err.getvalue()
