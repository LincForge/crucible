# linc-crucible

Quality gates for mobile apps, answering one question: **is this app ready to ship?** crucible
builds an Android (Gradle) or iOS (xcodebuild) app and deploys it to devices through
[linc-nerve](https://github.com/LincForge/nerve), runs your lint and tests and folds them into a
7-dimension quality score with a ship gate at 85, checks app-store readiness (5 dimensions,
threshold 80%), runs YAML build-verification tests (BVT) against visual and accessibility-tree
baselines, and pulls ranked marketing screenshots out of a screen recording. It is one layer of a
4-layer stack: `linc-device-kit` (shared models) → `linc-nerve` (device I/O) → `linc-crucible`
(quality). crucible itself contains no device code; everything that touches a device goes through
the nerve daemon's HTTP API.

8 MCP tools | Python 3.13+ | Android + iOS

## Install

Requires Python 3.13+ (`python3 --version`); with pipx/uv, `--python 3.13`.

- `pipx install linc-crucible` (or `pipx install --python 3.13 linc-crucible`)
- `uv tool install linc-crucible` (or `uv tool install --python 3.13 linc-crucible`)
- `pip install linc-crucible` (inside a virtual environment)

AI-ranked frame picks need the unpublished `linc-ai` package; without it `crucible_extract_assets` uses evenly-spaced frames and reports `fallback_used: true`.

## Project setup

crucible discovers projects by scanning `$LINC_PROJECTS_DIR/*/linc-manifest.yaml` (default
`~/projects/*/linc-manifest.yaml`). Two files live in the project root:

- `linc-manifest.yaml` — identity. `identity.name` is the project name every command takes;
  `identity.package` is the Android applicationId / iOS bundle ID.
- `.forge.yaml` — holds the build/test/lint commands; the name is historical. Commands in
  `.forge.yaml` run from the project root, so the `./gradlew` example below requires a Gradle
  wrapper (`./gradlew`) in that directory. When neither `lint_command` nor `test_command` runs to a
  measurement (nothing measured), `crucible assay` reports `"status": "unmeasured"` and exits `1`.
  An empty or unparsed test run (0 parsed tests, even when `test_command` exits `0`) also counts
  as unmeasured (`pass_rate: null`, `ship_gate: false`; if nothing else measured, `assay` reports
  `"status": "unmeasured"` and exits `1`). Test counts come from stdout summaries and from JUnit
  XML under `**/build/test-results/` written during the run (the search skips dot directories,
  symlinks, `build` outputs other than `build/test-results`, `node_modules`, `Pods`,
  `vendor/bundle`, and the `src` directory of any module that has a `build.gradle(.kts)`; a project
  with more than 5,000 remaining directories reads `UNMEASURED: test-results search stopped at
  5000 directories`; XML files modified before the run
  starts or after it finishes, outside a 2-second slack window, are ignored). Because stale XML is
  not counted, a Gradle `UP-TO-DATE` test task that emits no stdout summary and rewrites no XML
  reads `UNMEASURED` — pass `--rerun-tasks` in `test_command` (for example
  `./gradlew test --rerun-tasks`) to force fresh XML.

```yaml
# ~/projects/myapp/linc-manifest.yaml
schema_version: 1
identity:
  name: myapp
  package: com.example.myapp
```

```yaml
# ~/projects/myapp/.forge.yaml
forge:
  build_command: "./gradlew :app:assembleDebug"
  test_command: "./gradlew test"
  lint_command: "./gradlew lint"
```

For `crucible preflight --project myapp`, place store metadata in `<project>/store/`:

- `<project>/store/store-listing.md` — markdown with headings `# Title`, `# Short Description`,
  `# Full Description`, `# Keywords`, `# Category`.
- `<project>/store/brand-brief.json` — JSON object with required fields `app_name`, `tagline`,
  `emotion`, `primary_color`, `accent_color`, `background_color`, `icon_concept`, `style`,
  `platforms`.

Apps with a phone and a watch module sharing one package ID can declare `modules:` in
`.forge.yaml` (`name`, `device_types`, `build_command`, `apk_pattern`) so each artifact is routed
to the right device type: `crucible pour --project myapp --module watchApp --device-type watch`.

## Quickstart

After [Project setup](https://github.com/LincForge/crucible#project-setup) above (`myapp` has
`linc-manifest.yaml` and `.forge.yaml` in `$LINC_PROJECTS_DIR/myapp`), the first four commands
need no daemon, device or Docker:

```bash
crucible fuelcheck
crucible assay --project myapp
crucible score myapp
crucible preflight --project myapp
# pour, bvt and baseline capture reach devices through the nerve daemon:
pipx install "linc-nerve>=0.1,<0.2" && nerve daemon &
claude mcp add linc-crucible -- crucible bellows
```

`crucible fuelcheck` reports which build tools, projects and services it can see and exits `0` on
a fresh machine. Seven of the 8 MCP tools map directly to CLI commands (`crucible_extract_assets` →
`crucible extract-assets`), while `crucible_baseline` is served by `crucible baseline` (`capture`,
`list`, `prune`; see [MCP Tools](https://github.com/LincForge/crucible#mcp-tools-8)); options are
flags, not positionals (`crucible pour --project myapp`), except `crucible score PROJECT
[--history] [--limit N]` and `crucible baseline`. Output is JSON on stdout; for text, put
`--human` before the command (`crucible --human score myapp`). Exit `0` success (`status: "partial"` stays exit `0` with `ship_gate: false` in the payload; `apk-audit` without `--strict` exits `0` for completed reports even when `summary.passed` is `false`), `1` tool error (a run that measured nothing, a BVT that did not run, or `apk-audit --strict` when `summary.passed` is `false` exits `1`), `2` usage error (including invalid `apk-audit` inputs). `crucible schema --json`
prints every command and option.

## MCP Tools (8)

Seven of the 8 MCP tools are reflected directly as CLI commands (`crucible_pour` → `crucible pour`).
`crucible_baseline` is served on the CLI by the hand-written `crucible baseline` group (`capture`,
`list`, `prune` for screenshot baselines); its `a11y` mode and `latest` / `promote` actions are
MCP-only.

| Tool | Needs nerve daemon | Description |
|------|--------------------|-------------|
| `crucible_pour` | yes (no with `build_only`) | Build, install and launch on devices (`launch_intent` deep-link URI, e.g. `myapp://settings`). Android/iOS, phone/watch routing, `build_only` and `skip_build` modes |
| `crucible_assay` | no (live a11y tree check uses it when running) | Quality analysis: lint, tests, optional security (MobSF), perf, static a11y, live a11y tree (`a11y_tree` honours the personal-device policy; `allow_personal=True` overrides). Modes: `quick` or `full` |
| `crucible_bvt` | yes | Run a YAML Build Verification Test flow on a device: scripted UI steps with screenshot, element and assert checks (`owner` / `--owner` or `NERVE_OWNER` honours the caller's nerve reservation lease) |
| `crucible_score` | no | Latest quality score or history. 7-dimension weighted composite, ship gate 85 |
| `crucible_preflight` | no | App-store readiness: code quality, store listing (`<project>/store/store-listing.md`), visual assets, brand brief (`<project>/store/brand-brief.json`), build artifacts. Threshold 80% |
| `crucible_baseline` | yes for `capture`; no for `latest`, `list`, `prune` | Golden baselines for screenshot and a11y-tree regression detection (`crucible baseline capture\|list\|prune` on the CLI; `a11y` mode, `latest` and `promote` are MCP-only) |
| `crucible_extract_assets` | no | Ranked marketing screenshots (with social/store variants) from a video recording |
| `crucible_usage_stats` | no | Usage analytics over the local MCP call log: per-tool calls, success rate, p50/p95 latency, dormant tools |

## APK audit

`crucible apk-audit <apk_path> [--assetlinks <path>] [--strict]` audits a standalone Android APK
offline without a project manifest, connected device, or nerve daemon (`crucible --human apk-audit`
for plain-text output; use `--` before dash-prefixed filenames).

- **What it checks:**
  - `archive` (pure-stdlib ZIP/DEX inspection; refused when the central directory exceeds 16 MiB
    or the archive declares or holds more than 50,000 entries):
    APK byte size, DEX count, compressed/uncompressed DEX bytes and ratio, uncompressed DEX storage
    (`dex_stored`), native ABIs (`abis`), and bundled sourcemaps under `assets/` or `res/` (`*.map`).
    Flags `unminified_dex_bloat` (`medium`) and `bundled_sourcemaps` (`low`).
  - `manifest` (via `aapt2 dump xmltree`, or `aapt` fallback): `package`, `version_code`,
    `version_name`, `min_sdk`, `target_sdk`, `debuggable`, `uses_cleartext_traffic`, `allow_backup`,
    `activities_missing_uimode_config`, and `autoverify_hosts`. Flags `debuggable_enabled` (`high`),
    `cleartext_traffic_enabled` (`high`), `backup_enabled` (`low`), `activity_recreates_on_uimode`
    (`info`), and `duplicate_android_attribute` (`high`: one element carries the same `android:`
    attribute twice, so the audit cannot tell which value Android applies).
  - `signing` (via `apksigner verify --print-certs`) & `assetlinks` (`--assetlinks`): extracts
    SHA-256 signing certificate digests (`cert_sha256s`) and optionally verifies them offline
    against a local Digital Asset Links `assetlinks.json`. Flags `assetlinks_package_missing`,
    `assetlinks_relation_missing`, `assetlinks_sha256_mismatch`, and `assetlinks_unavailable` (all
    `high`).
- **Required tools (`aapt2` and `apksigner`):** Resolved on `PATH` first (`aapt2` then `aapt`;
  `apksigner`), then under `$ANDROID_HOME/build-tools/*/` (or `$ANDROID_SDK_ROOT/build-tools/*/`
  when `ANDROID_HOME` is unset, picking the highest parsed numeric build-tools version tuple — so
  e.g. `35.0.0` beats `9.0.0`, preferring stable releases over pre-release suffixes and `aapt2`
  over `aapt` within the same version). `crucible fuelcheck` probes
  both and warns when either is absent. If either tool is missing (`status: "unavailable"`), `archive`
  checks still run, `summary.measured` is `false`, `summary.passed` is `false`, and `--human`
  prints `[UNMEASURED]`.
- **JSON fields:** Top-level keys `apk_path`, `sha256`, `archive`, `manifest`, `signing`,
  `assetlinks`, `findings` (`{"severity", "code", "detail"}`), and `summary` (`high`, `medium`,
  `low`, `info`, `measured`, `passed`). `summary.measured` is `true` only when both `manifest` and
  `signing` ran (`"ok"` or `"error"`) and any requested `--assetlinks` check completed;
  `summary.passed` is `true` only when `measured` is `true`, `high == 0`, `manifest.status == "ok"`,
  `signing.status == "ok"`, and `assetlinks.status` is `"skipped"` or `"matched"`.
- **Exit codes:**
  - Without `--strict`: exits `0` for any completed report (even when `high` findings, tool errors,
    or `measured: false` occur).
  - With `--strict`: exits `1` whenever `summary.passed` is `false` (any `high` finding, tool
    error, or `measured: false`); exits `0` only when `summary.passed` is `true`.
  - Invalid inputs (missing, directory, or corrupt APK; a central directory over 16 MiB or more
    than 50,000 entries; missing, oversized >1 MiB, or
    malformed `assetlinks.json`): exits `2` with a JSON validation envelope on stderr and empty
    stdout, regardless of `--strict`.
- **Accepted `assetlinks.json` format:** A UTF-8 JSON array (<= 1 MiB) of Digital Asset Links
  statements where `relation` is a list containing `"delegate_permission/common.handle_all_urls"`
  and `target` is an object with `"namespace": "android_app"`, `"package_name"` matching the APK
  package, and `"sha256_cert_fingerprints"` listing 64-hex or colon-separated SHA-256 digests:

```json
[
  {
    "relation": ["delegate_permission/common.handle_all_urls"],
    "target": {
      "namespace": "android_app",
      "package_name": "com.example.myapp",
      "sha256_cert_fingerprints": [
        "AA:11:BB:22:CC:33:DD:44:EE:55:FF:66:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33"
      ]
    }
  }
]
```

## Configuration

<!-- env-table:start -->
| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `LINC_HOME` | `~/.linc` | Root of crucible state (`$LINC_HOME/crucible/*`) and the MCP call log |
| `LINC_PROJECTS_DIR` | `~/projects` | Directory scanned for `*/linc-manifest.yaml` project discovery |
| `LINC_WORKSPACE` | `$LINC_HOME/projects/LINC` (or `~/projects/LINC` when `LINC_HOME` is unset) | Optional LINC workspace fallback read by `preflight` when `<project>/store/` is absent |
| `LINC_DEVICE_FLEET_FILE` | `$LINC_HOME/device-fleet.yaml` | Personal-device registry; devices marked `personal: true` are refused unless `allow_personal=True`. A missing file means no personal devices are registered; a file that exists but cannot be read or parsed refuses EVERY device operation unless `allow_personal=True` |
| `LINC_LOG_LEVEL` | `WARNING` | Level of the structured logs written to stderr |
| `LINC_BUILD_GATE` | *(unset)* | `1` wraps builds in `linc-build-gate` (host build-slot mutex) |
| `LINC_CRUCIBLE_BUILD_TIMEOUT` | `600` | Build timeout in seconds for `pour`; invalid values fall back to the default |
| `FORGE_PROJECTS_FILE` | `$LINC_HOME/.forge/projects.yaml` | Legacy project registry, read-only; manifests discovered on disk take precedence |
| `FORGE_RUN_ID` | `manual` | Deprecated session id, read when `CRUCIBLE_RUN_ID` is unset or empty |
| `CRUCIBLE_USAGE_LOG` | `redacted` | Local usage log mode: `off`, `redacted` or `full`; an unknown value means `redacted` (see [Trust](https://github.com/LincForge/crucible#trust)) |
| `CRUCIBLE_RUN_ID` | `FORGE_RUN_ID`, else `manual` | Session id stamped on usage log entries; a value outside `[A-Za-z0-9._-]{1,128}` is logged as `manual` |
| `CRUCIBLE_SCORES_DIR` | `$LINC_HOME/crucible/scores` | Quality score history |
| `CRUCIBLE_BASELINES_DIR` | `$LINC_HOME/crucible/baselines` | Visual and a11y-tree baselines |
| `CRUCIBLE_FLEET_FILE` | `$LINC_HOME/crucible/fleet.yaml` | Device fleet registry |
| `CRUCIBLE_IOS_TEAM_ID` | *(unset)* | Apple Developer Team ID for physical-iOS device builds |
| `CRUCIBLE_MOBSF_API_KEY` | *(unset)* | API key for the optional MobSF security scan (`assay` mode `full`) |
| `NERVE_OWNER` | *(unset)* | Default nerve reservation lease owner for `bvt` when `--owner` is not passed |
| `ANDROID_HOME` | *(unset)* | Android SDK root; `build-tools/*/aapt2` reads APK package/manifest metadata (for `pour` and `apk-audit`) and `build-tools/*/apksigner` verifies APK signing certificates (`apk-audit`, probed by `fuelcheck`) |
| `ANDROID_SDK_ROOT` | *(unset)* | Legacy Android SDK root, used when `ANDROID_HOME` is unset |
<!-- env-table:end -->

Generated from `linc_crucible.config_contract.ENV_VARS` (`crucible fuelcheck --env-table`); pinned to the variables read under `src/` by `tests/test_config_contract.py`.

State crucible writes between runs:

| Path | Contents |
|------|----------|
| `$LINC_HOME/crucible/scores/<project>/` | Quality score history (JSON) |
| `$LINC_HOME/crucible/baselines/<project>/` | Visual and a11y-tree baselines |
| `$LINC_HOME/crucible/assets/` | Extracted screenshots |
| `$LINC_HOME/crucible/fleet.yaml` | Device fleet registry |
| `$LINC_HOME/mcp-calls.jsonl` | Local MCP and CLI tool usage log; redacted by default (`CRUCIBLE_USAGE_LOG`): keeps tool name, timing, success and allowlisted parameter values, replaces every other parameter value with `<redacted len=N>` and absolute paths in `error` with `<path>` (dir `0700`, file `0600`) |

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Project not found" | No `linc-manifest.yaml` under `$LINC_PROJECTS_DIR` | Add the manifest to the project root, or point `LINC_PROJECTS_DIR` at its parent |
| "Nerve unreachable at http://localhost:8111 — start it with `nerve daemon`" | nerve daemon not running | `nerve daemon`, then `curl http://localhost:8111/health` |
| "No APK found" | Build not run | `crucible pour --project myapp`; check `build_command` in `.forge.yaml` |
| "No quality score found" | No assay yet | `crucible assay --project myapp` |
| "Latest/Newer quality score file(s) skipped (...)" | A score file in the store is corrupt or unreadable, so preflight will not fall back to an older score | `crucible assay --project myapp` writes a fresh score |
| "Quality score is dated in the future" | The score's recorded timestamp is ahead of the host clock (a hand-edited or restored score file) | Fix the clock (`sntp`/NTP), then `crucible assay --project myapp` |
| Build timeout (>10 min) | Cold build cache | Build once by hand, then pour with `skip_build` |

## Trust

- **Your project's commands run as you.** `assay` and `pour` run the `build_command`,
  `test_command` and `lint_command` from the project's `.forge.yaml` with your user's
  permissions. crucible never spawns a shell itself (`argv` list, `shell=False`, `shlex.split`)
  and refuses `|`, `;`, `&`, `$(` and backticks, but a project command that is itself a shell
  (`sh -c '…'`, `bash script.sh`) runs a shell because the project asked for it: the trust
  boundary is the project. Only point crucible at projects you trust. Details:
  [SECURITY.md](https://github.com/LincForge/crucible/blob/main/SECURITY.md).
- **No telemetry, no phone-home.** Nothing leaves the machine: crucible talks to your local nerve
  daemon, and to MobSF only if you run a `full` assay against one you started. The nerve daemon
  on loopback is a trusted peer: crucible acts on the device list, element trees and file paths
  (e.g. the screenshot path) nerve returns.
- **A local usage log, under your control.** Each MCP tool call — and each CLI command that
  invokes an MCP tool (`pour`, `assay`, `bvt`, `score`, `preflight`, `extract-assets`,
  `usage-stats`) — is appended to `$LINC_HOME/mcp-calls.jsonl` (default `~/.linc/mcp-calls.jsonl`)
  with the tool name, timing and success. Set `CRUCIBLE_USAGE_LOG=off|redacted|full`:
  - `redacted` (default) — keeps allowlisted parameter values (such as `project`, `mode`, `action`
    and boolean/numeric flags), replaces every other parameter value length-only (`<redacted len=N>`)
    and scrubs redacted values, host roots and absolute paths (`<path>`) from `error`; new state
    directories are created `0700` and the log file is created owner-only (`0600`).
  - `off` — nothing is written.
  - `full` — raw parameter values. Opt in only on a machine you trust.

## How this repo works

This is a release mirror of LINC's private repository. The code is built by LINC's AI software
factory under human review; each release lands as one commit. Issues are welcome; pull requests by
arrangement — see [CONTRIBUTING.md](https://github.com/LincForge/crucible/blob/main/CONTRIBUTING.md).

---

Built by [LINC Innovations](https://lincinnovations.com/audit?utm_source=github&utm_medium=readme&utm_campaign=crucible) — we diagnose flaky device test suites.

Licensed under the [Apache License 2.0](https://github.com/LincForge/crucible/blob/main/LICENSE).
