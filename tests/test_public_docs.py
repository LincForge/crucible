"""Public-facing files a stranger relies on. Content checks only — prose is reviewed by the CEO."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import anyio

from linc_crucible.mcp_server import server

REPO = Path(__file__).resolve().parents[1]
PUBLIC_REPO = "https://github.com/LincForge/crucible"
APACHE_2_0_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
PUBLIC_DOC_FILES = ("LICENSE", "NOTICE", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md")
_LINK_TARGET = re.compile(r"\]\(\s*<?([^)\s>]+)|(?:href|src)=[\"']([^\"']+)[\"']")


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_licence_is_apache_everywhere() -> None:
    project = _pyproject()["project"]
    assert project["license"] == "Apache-2.0"
    assert not any("MIT" in c for c in project.get("classifiers", []))

    license_path = REPO / "LICENSE"
    assert license_path.is_file()
    license_bytes = license_path.read_bytes()
    license_text = license_bytes.decode("utf-8")
    first_nonempty_line = license_text.lstrip().splitlines()[0].strip()
    assert first_nonempty_line == "Apache License"
    assert hashlib.sha256(license_bytes).hexdigest() == APACHE_2_0_SHA256


def test_pep639_licence_metadata_ships_licence_and_notice() -> None:
    project = _pyproject()["project"]
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert project["authors"] == [{"name": "LINC Innovations"}]
    assert project["description"] == (
        "Quality gates for mobile apps: build, deploy, score and ship-check through linc-nerve"
    )


def test_notice_names_linc_innovations_and_no_vendored_binary() -> None:
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "LINC Crucible" in notice
    assert "Copyright 2026 LINC Innovations LLC" in notice
    assert "scrcpy" not in notice.lower()
    assert "vendor" not in notice.lower()


def test_contributing_requires_dco() -> None:
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Signed-off-by" in contributing
    assert "https://developercertificate.org/" in contributing
    assert "release mirror" in contributing
    assert "crucible fuelcheck" in contributing


def test_security_has_reporting_trust_boundary_and_review_stub() -> None:
    security = (REPO / "SECURITY.md").read_text(encoding="utf-8")
    assert "## Report a vulnerability" in security
    assert "private vulnerability reporting" in security
    assert "## Threat model in one paragraph" in security
    assert ".forge.yaml" in security
    assert "linc-manifest.yaml" in security
    assert "safe_subprocess_run" in security
    assert "shell=False" in security
    assert "CRUCIBLE_USAGE_LOG" in security
    assert "0600" in security
    assert "## Independent review" in security
    stub = (
        "(The adversarial review summary is added here by the release runbook "
        "before the first public release.)"
    )
    assert stub not in security
    review_section = security.split("## Independent review", 1)[1]
    assert "### 2026-10-24 pre-release review" in review_section
    assert "### 2026-10-25 post-fix verification" in review_section
    assert "### Release candidate verification" in review_section
    sha256_hashes = re.findall(r"\b[0-9a-f]{64}\b", review_section)
    assert "16e5ec182a09f2c5b9dc65e9240342d7061fdb8154d87c79d77e8bf48563288c" in sha256_hashes
    assert "f38fe86a756dfbd46bed3fdc43d3770c2a6e21fc0f044c22657963d22df92668" in sha256_hashes


def test_changelog_headings_are_unreleased_then_0_2_0_then_the_0_1_0_baseline() -> None:
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[(.+?)\].*$", changelog, re.M)
    assert headings[:3] == ["Unreleased", "0.2.0", "0.1.0"]
    assert "## [0.2.0] - 2026-10-24" in changelog
    release_section = changelog.split("## [0.2.0]", 1)[1].split("\n## [0.1.0]", 1)[0]
    assert "### Added" in release_section
    assert "### Changed" in release_section
    assert "### Security" in release_section
    assert "(6 at the time of the 0.1.0 baseline)" in release_section


def test_version_is_0_2_0_everywhere() -> None:
    from linc_crucible import __version__
    from linc_crucible.cli import CRUCIBLE_BANNER

    assert _pyproject()["project"]["version"] == "0.2.0"
    assert __version__ == "0.2.0"
    assert f"v{__version__}" in CRUCIBLE_BANNER


def test_pyproject_project_urls_point_at_the_public_repo() -> None:
    urls = _pyproject()["project"]["urls"]
    assert set(urls) >= {"Homepage", "Repository", "Changelog", "Security"}
    assert urls["Homepage"] == PUBLIC_REPO
    assert urls["Repository"] == PUBLIC_REPO
    assert urls["Changelog"] == f"{PUBLIC_REPO}/blob/main/CHANGELOG.md"
    assert urls["Security"] == f"{PUBLIC_REPO}/blob/main/SECURITY.md"
    assert all(url.startswith(PUBLIC_REPO) for url in urls.values())


def test_public_docs_have_no_parent_relative_links() -> None:
    for filename in PUBLIC_DOC_FILES:
        path = REPO / filename
        assert path.is_file(), f"missing public doc file: {filename}"
        text = path.read_text(encoding="utf-8")
        targets = [a or b for a, b in _LINK_TARGET.findall(text)]
        assert not any("../" in t for t in targets), (
            f"{filename} has parent-relative link: {targets}"
        )
        assert "../" not in text, f"{filename} contains '../'"


# --- README for an outside user (REVIEW B6) ---------------------------------------------

_MIRROR_PARAGRAPH = (
    "This is a release mirror of LINC's private repository. The code is built by LINC's AI "
    "software factory under human review; each release lands as one commit. Issues are "
    "welcome; pull requests by arrangement — see "
    f"[CONTRIBUTING.md]({PUBLIC_REPO}/blob/main/CONTRIBUTING.md)."
)
_AUDIT_URL = (
    "https://lincinnovations.com/audit?utm_source=github&utm_medium=readme&utm_campaign=crucible"
)


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    assert f"\n{heading}\n" in text, f"README.md is missing '{heading}'"
    return text.split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_readme_has_release_mirror_paragraph() -> None:
    assert _flat(_MIRROR_PARAGRAPH) in _flat(_section(_readme(), "## How this repo works"))


def test_readme_trust_section_covers_commands_network_and_usage_log() -> None:
    trust = _flat(_section(_readme(), "## Trust"))
    assert ".forge.yaml" in trust
    assert "No telemetry" in trust
    assert "$LINC_HOME/mcp-calls.jsonl" in trust
    assert "CRUCIBLE_USAGE_LOG" in trust


def test_readme_names_every_usage_log_mode_and_redacted_default() -> None:
    trust = _section(_readme(), "## Trust")
    for mode in ("off", "redacted", "full"):
        assert f"`{mode}`" in trust, f"usage log mode {mode!r} not named"
    assert "`redacted` (default)" in trust
    config = _section(_readme(), "## Configuration")
    assert "| `CRUCIBLE_USAGE_LOG` | `redacted` |" in config


def test_readme_footer_has_utm_link_and_public_licence() -> None:
    readme = _readme()
    footer = f"Built by [LINC Innovations]({_AUDIT_URL}) — we diagnose flaky device test suites."
    assert footer in readme
    assert f"[Apache License 2.0]({PUBLIC_REPO}/blob/main/LICENSE)" in readme


def test_readme_quickstart_is_one_bash_fence_starting_with_fuelcheck() -> None:
    quickstart = _section(_readme(), "## Quickstart")
    fences = re.findall(r"```bash\n(.*?)```", quickstart, re.S)
    assert len(fences) == 1, f"expected one ```bash fence in Quickstart, got {len(fences)}"
    commands = [
        ln.strip() for ln in fences[0].splitlines() if ln.strip() and not ln.startswith("#")
    ]
    assert commands[0] == "crucible fuelcheck"
    assert "crucible assay --project myapp" in commands
    assert "claude mcp add linc-crucible -- crucible bellows" in commands
    assert any('"linc-nerve>=0.1,<0.2"' in c and "nerve daemon" in c for c in commands)


def test_readme_links_are_absolute_https_without_legacy_paths() -> None:
    readme = _readme()
    targets = [a or b for a, b in _LINK_TARGET.findall(readme)]
    assert targets, "README.md has no links"
    bad = [t for t in targets if not t.startswith("https://")]
    assert not bad, f"README.md has non-https links: {bad}"
    assert "../" not in readme
    assert "~/.crucible" not in readme


def test_readme_documents_ai_extra_and_fallback() -> None:
    """[test] documents the AI fallback and advertises no [ai] extra in README or CHANGELOG."""
    readme = _readme()
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "[ai]" not in readme
    assert "[ai]" not in changelog
    install = _flat(_section(readme, "## Install"))
    assert "linc-ai" in install
    assert "unpublished" in install
    assert "crucible_extract_assets" in install
    assert "evenly-spaced" in install
    assert "fallback_used: true" in install
    for cmd in ("pipx install linc-crucible", "uv tool install linc-crucible"):
        assert cmd in install


def test_pyproject_has_no_ai_extra_and_lists_linc_ai_only_in_dev() -> None:
    """[contract] tomllib-parsed pyproject has no `ai` extra and lists linc-ai only in `dev`."""
    data = _pyproject()
    project = data["project"]
    deps = project.get("dependencies", [])
    assert not any("linc-ai" in d for d in deps), f"linc-ai in dependencies: {deps}"
    opt_deps = project.get("optional-dependencies", {})
    assert "ai" not in opt_deps, f"pyproject still defines an 'ai' extra: {list(opt_deps)}"
    assert set(opt_deps) == {"dev"}, f"unexpected optional-dependencies keys: {list(opt_deps)}"
    uv_sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if uv_sources:
        assert "linc-ai" in opt_deps["dev"], f"linc-ai missing from dev extra: {opt_deps['dev']}"
        assert "linc-ai" in uv_sources, "tool.uv.sources must retain linc-ai"
    else:
        assert "linc-ai" not in opt_deps["dev"], (
            f"exported pyproject kept private linc-ai in dev extra: {opt_deps['dev']}"
        )


def test_readme_tool_table_says_which_tools_need_the_nerve_daemon() -> None:
    registry = sorted(t.name for t in anyio.run(server.list_tools))
    tools = _section(_readme(), f"## MCP Tools ({len(registry)})")
    rows = [ln for ln in tools.splitlines() if ln.startswith("| `crucible_")]
    needs = {ln.split("|")[1].strip().strip("`"): ln.split("|")[2].strip() for ln in rows}
    assert len(rows) == len(needs), f"duplicate tool rows in README: {rows}"
    assert sorted(needs) == registry, f"README tool table {sorted(needs)} != {registry}"
    assert needs["crucible_score"] == "no"
    assert needs["crucible_bvt"] == "yes"
    assert needs["crucible_usage_stats"] == "no"
    assert all(v.startswith(("yes", "no")) for v in needs.values()), needs


# Behaviour documented ahead of its release: each marker is dropped when the tracking change
# lands on main. The usage-log modes, fail-open DevicePolicy and linc-ai fallback are all on
# main, so no row carries the marker.
_NEXT_RELEASE = "from the next release"


def _block_containing(text: str, needle: str, sep: str) -> str:
    return _flat(next(b for b in text.split(sep) if needle in b)).lower()


def test_readme_documents_released_behaviour_without_a_marker() -> None:
    """[contract] rows for behaviour already on main must not say "from the next release"."""
    readme = _readme()
    assert _NEXT_RELEASE not in readme.lower()
    config = _section(readme, "## Configuration")
    for var in ("CRUCIBLE_USAGE_LOG", "CRUCIBLE_RUN_ID", "LINC_DEVICE_FLEET_FILE"):
        row = _block_containing(config, f"| `{var}` |", "\n")
        assert _NEXT_RELEASE not in row, f"{var} row promises a release that already shipped"
    state_row = _block_containing(config, "| `$LINC_HOME/mcp-calls.jsonl` |", "\n")
    assert _NEXT_RELEASE not in state_row, "the usage-log state row is still marked"
    assert "redacted by default" in state_row
    usage_log = _block_containing(_section(readme, "## Trust"), "A local usage log", "\n- ")
    assert _NEXT_RELEASE not in usage_log, "the usage-log trust bullet is still marked"
    assert "crucible_usage_log=off|redacted|full" in usage_log


def test_readme_and_security_document_allowlist_usage_log_without_overclaim() -> None:
    """[test] README.md and SECURITY.md accurately describe allowlisted redacted mode and
    do not claim that all 'parameter values are redacted'."""
    readme = _readme()
    security = (REPO / "SECURITY.md").read_text(encoding="utf-8")

    assert "parameter values are redacted" not in readme.lower()
    assert "parameter values are redacted" not in security.lower()

    trust = _flat(_section(readme, "## Trust"))
    assert "allowlisted parameter values" in trust
    assert "<redacted len=N>" in trust
    assert "<path>" in trust
    assert "0700" in trust
    assert "0600" in trust

    config = _section(readme, "## Configuration")
    state_row = _block_containing(config, "| `$LINC_HOME/mcp-calls.jsonl` |", "\n")
    assert "allowlisted parameter values" in state_row
    assert "<redacted len=n>" in state_row
    assert "<path>" in state_row

    flat_security = _flat(security)
    assert "allowlisted parameter values" in flat_security
    assert "<redacted len=N>" in flat_security
    assert "<path>" in flat_security
    assert "0700" in flat_security
    assert "0600" in flat_security


_GENERIC_TRACKER_ID = re.compile(
    r"(?i)\bissue\s+#[0-9]|linc-crucible#|linc-ops#|\breactor\s+[0-9a-f]{8}\b"
    r"|\bREVIEW\s+B\d+|\bIMP[- ][0-9a-f]{6,8}\b"
)


def test_src_has_no_generic_tracker_id_patterns() -> None:
    """[contract] shipped source files under src/ contain no internal tracker ID patterns."""
    hits: list[str] = []
    for path in sorted((REPO / "src").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _GENERIC_TRACKER_ID.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    assert hits == [], f"tracker ID patterns in src/: {hits}"


def test_security_and_readme_document_project_shell_and_trusted_nerve_peer() -> None:
    """[test] SECURITY.md and README.md explain sh -c project commands and trusted nerve peer."""
    security = _flat((REPO / "SECURITY.md").read_text(encoding="utf-8"))
    trust = _flat(_section(_readme(), "## Trust"))

    for label, text in (("SECURITY.md", security), ("README.md##Trust", trust)):
        assert "shlex.split" in text, f"{label} missing shlex.split"
        assert "sh -c" in text, f"{label} missing sh -c"
        assert "bash script.sh" in text, f"{label} missing bash script.sh"
        assert "trusted peer" in text, f"{label} missing trusted peer"
        assert "device list" in text, f"{label} missing device list"
        assert "element trees" in text, f"{label} missing element trees"
        assert "screenshot path" in text, f"{label} missing screenshot path"


def test_readme_install_states_python_3_13() -> None:
    """[test] ## Install states the Python 3.13+ prerequisite and --python 3.13 flag."""
    install = _flat(_section(_readme(), "## Install"))
    assert "Requires Python 3.13+" in install
    assert "python3 --version" in install
    assert "--python 3.13" in install


def test_readme_project_setup_precedes_quickstart() -> None:
    """[test] ## Project setup appears before ## Quickstart and documents gradlew + exit codes."""
    readme = _readme()
    setup_pos = readme.index("\n## Project setup\n")
    quickstart_pos = readme.index("\n## Quickstart\n")
    assert setup_pos < quickstart_pos, "## Project setup must precede ## Quickstart"

    setup = _flat(_section(readme, "## Project setup"))
    assert "project root" in setup
    assert "Gradle wrapper" in setup or "./gradlew" in setup
    assert "nothing measured" in setup and "exits `1`" in setup


def test_readme_has_no_projects_active_and_documents_local_store_dir() -> None:
    """[test] README never mentions projects/active and documents <project>/store/ inputs."""
    from linc_crucible.core.preflight import BRAND_BRIEF_REQUIRED

    readme = _readme()
    assert "projects/active" not in readme
    assert "<project>/store/store-listing.md" in readme
    assert "<project>/store/brand-brief.json" in readme
    for field in BRAND_BRIEF_REQUIRED:
        assert f"`{field}`" in readme, f"README missing brand-brief required field {field!r}"


def test_readme_troubleshooting_matches_deployer_nerve_unreachable_constant() -> None:
    """[test] Troubleshooting row text equals the deployer's unreachable message constant."""
    from linc_crucible import bvt_tools
    from linc_crucible.core.deployer import NERVE_UNREACHABLE_MSG

    assert bvt_tools.NERVE_UNREACHABLE_MSG == NERVE_UNREACHABLE_MSG
    troubleshooting = _section(_readme(), "## Troubleshooting")
    nerve_row = next(ln for ln in troubleshooting.splitlines() if "Nerve unreachable" in ln)
    symptom_cell = nerve_row.split("|")[1].strip()
    assert NERVE_UNREACHABLE_MSG in symptom_cell


def test_readme_cli_baseline_parity_claim_is_accurate() -> None:
    """[test] README accurately states crucible_baseline CLI surface vs MCP-only actions."""
    readme = _readme()
    assert "Every MCP tool is also a CLI command" not in readme
    tools_sec = _flat(_section(readme, "## MCP Tools (8)"))
    assert "crucible baseline" in tools_sec
    for sub in ("capture", "list", "prune"):
        assert sub in tools_sec
    for mcp_only in ("a11y", "latest", "promote"):
        assert mcp_only in tools_sec
    assert "MCP-only" in tools_sec
