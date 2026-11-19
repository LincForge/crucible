"""android_activity: launch one launcher of a multi-launcher package.

SniperPulse's .dev build has two launcher activities (MainActivity and the
debug Shot Lab, SniperPulseApp#458). nerve refuses a package-only launch of such
a package ("has 2 launcher activities ... pass activity= to choose one"), and
older daemons picked one at random, so the cold_start BVT either failed at
launch or opened Shot Lab and missed the age gate. The spec already declares
``android_activity:`` (linc-ops' daily runner honours it); crucible ignored it.

Contract:
- file-level ``android_activity`` reaches every test; a per-test key overrides;
- a hostile value is rejected at load time (input boundary), never sent to nerve;
- the test's activity goes with every launch of the test's OWN package (the
  reset relaunch and the ``launch`` step), never with another package;
- NerveClient sends ``activity`` to nerve_launch only when set.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import BVTRunner
from linc_crucible.core.bvt.schema import BVTStep, BVTSuite, BVTTest, load_bvt_file
from linc_crucible.core.nerve_client import _launch_params
from tests.conftest import make_mock_nerve_client as _base_mock


def make_mock_nerve_client(**kwargs):
    """The shared mock, also recording the ``activity`` passed to each launch."""
    mock = _base_mock(**kwargs)
    mock.launch_activities = []
    record_launch = mock.launch.side_effect

    def _launch(device: str, package: str, uri: str = "", activity: str = "") -> dict:
        mock.launch_activities.append(activity)
        return record_launch(device, package)

    mock.launch.side_effect = _launch
    return mock


@pytest.fixture(autouse=True)
def _capture_structlog() -> Iterator[None]:
    with structlog.testing.capture_logs():
        yield


FG_OK = {"foreground": {"package": "com.app.dev", "activity": "com.app.MainActivity"}}


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "90-cold-start.yaml"
    p.write_text(body)
    return p


def test_contract_file_level_activity_reaches_every_test(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "package: com.app.dev\n"
        "android_activity: com.app.MainActivity\n"
        "tests:\n"
        "  - name: a\n    steps: [{launch: '{package}'}]\n"
        "  - name: b\n    android_activity: .Other\n    steps: [{launch: '{package}'}]\n",
    )
    _pkg, tests, _dev = load_bvt_file(path)
    assert [t.android_activity for t in tests] == ["com.app.MainActivity", ".Other"]


def test_contract_absent_activity_is_empty(tmp_path: Path) -> None:
    path = _write(tmp_path, "package: com.app\ntests:\n  - name: a\n    steps: []\n")
    _pkg, tests, _dev = load_bvt_file(path)
    assert tests[0].android_activity == ""


@pytest.mark.parametrize(
    "hostile",
    [
        "Main;reboot",
        "Main\\nreboot",
        "Main Activity",
        "$(id)",
        "-n x/.Y",
        "com..Bad",
        "1abc",
        "[a]",
    ],
)
def test_contract_hostile_activity_rejected_at_load(tmp_path: Path, hostile: str) -> None:
    path = _write(
        tmp_path,
        f'package: com.app\nandroid_activity: "{hostile}"\ntests:\n  - name: a\n    steps: []\n',
    )
    with pytest.raises(ValueError, match="android_activity"):
        load_bvt_file(path)


def test_contract_non_string_activity_rejected_at_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "package: com.app\ntests:\n  - name: a\n    android_activity: 42\n    steps: []\n",
    )
    with pytest.raises(ValueError, match="android_activity"):
        load_bvt_file(path)


def test_contract_activity_sent_with_reset_and_launch_of_own_package(tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[FG_OK])
    suite = BVTSuite(
        package="com.app.dev",
        tests=[
            BVTTest(
                name="cold_start_to_home",
                package="com.app.dev",
                android_activity="com.app.MainActivity",
                steps=[BVTStep(action="launch", target="com.app.dev")],
            )
        ],
    )
    BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path).run_suite(suite, device="DEV")

    assert mock.launch_calls == [("DEV", "com.app.dev"), ("DEV", "com.app.dev")]
    assert mock.launch_activities == ["com.app.MainActivity", "com.app.MainActivity"]


def test_contract_activity_not_sent_for_another_package(tmp_path: Path) -> None:
    mock = make_mock_nerve_client(
        screen_state_responses=[{"foreground": {"package": "com.other", "activity": ".A"}}]
    )
    suite = BVTSuite(
        package="com.app.dev",
        tests=[
            BVTTest(
                name="handoff",
                package="com.app.dev",
                android_activity="com.app.MainActivity",
                steps=[BVTStep(action="launch", target="com.other")],
            )
        ],
    )
    BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path).run_suite(suite, device="DEV")

    assert ("DEV", "com.other") in mock.launch_calls
    other = mock.launch_calls.index(("DEV", "com.other"))
    assert mock.launch_activities[other] == ""


def test_activity_does_not_leak_into_the_next_test(tmp_path: Path) -> None:
    mock = make_mock_nerve_client(screen_state_responses=[FG_OK, FG_OK])
    suite = BVTSuite(
        package="com.app.dev",
        tests=[
            BVTTest(
                name="first",
                package="com.app.dev",
                android_activity="com.app.MainActivity",
                steps=[BVTStep(action="launch", target="com.app.dev")],
            ),
            BVTTest(
                name="second",
                package="com.app.dev",
                steps=[BVTStep(action="launch", target="com.app.dev")],
            ),
        ],
    )
    BVTRunner(nerve=NerveAdapter(nerve=mock), output_dir=tmp_path).run_suite(suite, device="DEV")

    assert mock.launch_activities[-2:] == ["", ""]


def test_contract_launch_params_include_activity_only_when_set() -> None:
    assert _launch_params("D", "com.app", "") == {"device": "D", "package": "com.app"}
    assert _launch_params("D", "com.app", "", "com.app.Main") == {
        "device": "D",
        "package": "com.app",
        "activity": "com.app.Main",
    }
