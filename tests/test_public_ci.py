"""The public repo's CI runs the public suite on ubuntu + macOS and uses trusted publishing.

Ships publicly: in the private repo the files live under public/.github, and linc-publish's
path_map moves them to .github in the export, so this test reads whichever exists.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
GH = REPO / "public" / ".github" if (REPO / "public" / ".github").is_dir() else REPO / ".github"
WF = GH / "workflows"


def _wf(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text())


def test_public_ci_files_exist() -> None:
    assert (WF / "test.yml").is_file()
    assert (WF / "release.yml").is_file()
    assert (GH / "dependabot.yml").is_file()


def test_test_matrix_is_ubuntu_and_macos_on_python_313_only() -> None:
    job = _wf("test.yml")["jobs"]["test"]
    matrix = job["strategy"]["matrix"]
    assert sorted(matrix["os"]) == ["macos-latest", "ubuntu-latest"]
    assert matrix["python"] == ["3.13"]
    runs = [step.get("run", "") for step in job["steps"]]
    assert "uv run pytest -q -x" in runs


def test_release_uses_trusted_publishing_no_token() -> None:
    rel = _wf("release.yml")
    job = next(iter(rel["jobs"].values()))
    assert job["permissions"]["id-token"] == "write"
    assert job["environment"] == "pypi"
    text = (WF / "release.yml").read_text()
    assert "pypa/gh-action-pypi-publish" in text
    assert "password" not in text and "PYPI_TOKEN" not in text
    publish = [s for s in job["steps"] if "pypa/gh-action-pypi-publish" in s.get("uses", "")]
    assert publish and all("with" not in s for s in publish)


def test_workflows_reference_no_private_repos_or_secrets() -> None:
    files = sorted(WF.glob("*.yml"))
    assert files
    for f in files:
        text = f.read_text()
        assert "secrets." not in text, f
        assert "LincForge/linc-" not in text, f


def test_no_smoke_workflow_yet() -> None:
    assert not (WF / "smoke.yml").exists()


def test_dependabot_version_updates_disabled() -> None:
    cfg = yaml.safe_load((GH / "dependabot.yml").read_text())
    assert cfg["updates"]
    assert all(u.get("open-pull-requests-limit") == 0 for u in cfg["updates"])
