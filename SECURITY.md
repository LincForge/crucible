# Security

## Report a vulnerability

Use GitHub's **private vulnerability reporting** on this repository (Security → Report a
vulnerability). Please do not open a public issue for security problems. We aim to
acknowledge within 3 business days.

## Threat model in one paragraph

crucible runs on your machine, starts no server, and talks to the nerve daemon on loopback
`:8111` over HTTP. The nerve daemon on loopback is a trusted peer: crucible acts on the
device list, element trees, and file paths (e.g. the screenshot path) nerve returns. crucible
never spawns a shell itself (it splits commands with `shlex.split` into an `argv` list and runs
them with `shell=False` through `safe_subprocess_run`, which refuses `|`, `;`, `&`, `$(` and
backticks and kills the process group on timeout), but a project command in `.forge.yaml` or
`linc-manifest.yaml` that is itself a shell (`sh -c '…'`, `bash script.sh`) runs a shell as your
local user because the project asked for it: **the trust boundary is the project under test**, so
only point crucible at repositories you trust. No telemetry leaves the machine; local tool-call
usage logging is controlled by `CRUCIBLE_USAGE_LOG` (`off|redacted|full`, default `redacted`: keeps
tool name, timing, success, and allowlisted parameter values, replaces every other parameter value
with `<redacted len=N>` and absolute paths in `error` with `<path>`, creates new state directories
`0700` and the log file `0600` — see the configuration mode table in `README.md`).

## Independent review

### 2026-10-24 pre-release review

An adversarial security review and first-time user ("stranger") run conducted on 2026-10-24 evaluated the pre-release v0.2.0 candidate (private `linc-crucible` commit `7e9e981`, candidate wheel SHA-256 `16e5ec182a09f2c5b9dc65e9240342d7061fdb8154d87c79d77e8bf48563288c`). Testing used a separate session with no access to fix history, operating in-process against fakes for subprocesses and the nerve daemon alongside isolated CLI invocations with a local proxy listener and no device contact.

Across 45 adversarial attempts, 20 held and 18 security/reliability findings were identified (2 High, 5 Medium, 11 Low), all fixed before release:
- **Path traversal in baseline, a11y tree, and score stores (High, plus related Medium):** Unvalidated `project` and `screen_name` arguments in `crucible_baseline`, `TreeCaptureStore`, and `ScoreStore` allowed absolute paths or `..` segments to write or delete (`prune` with `keep=0`) directories outside the store root. Fixed by enforcing a strict `[A-Za-z0-9._-]` name allowlist (max 128 chars, never `.` or `..`) and verifying resolved-path containment inside the store root before every read, write, or delete.
- **Personal-device policy bypass in `crucible_assay(a11y_tree=True)` (High):** `crucible_assay` with `a11y_tree=True` queried the UI element tree of a named or auto-picked device without consulting `DevicePolicy` or honoring fail-closed registry states. Fixed by routing `a11y_tree` target selection through the same `DevicePolicy` gate as `crucible_pour` and `crucible_bvt`.
- **Personal-device alias and case-sensitivity bypass (Medium and Low):** A personal device registered by its USB hardware serial could be reached via its Wi-Fi ADB serial (`ip:port`) when `nerve_discover` reported both identifiers, and `crucible_bvt` did not casefold serial comparisons. Fixed by cross-checking all `nerve_discover` identifier fields (`serial`, `id`, `hardware_id`) against `DevicePolicy` and casefolding serial comparisons across every device-touching tool.
- **Environment proxy routing on loopback nerve and MobSF clients (Medium):** `httpx` clients honored `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`, allowing loopback nerve traffic and the MobSF `Authorization` header to be routed to an ambient proxy. Fixed by setting `trust_env=False` on all `httpx` clients.
- **Gradle option injection via `build_type` (Medium):** Whitespace or leading dashes in `build_type` could inject extra Gradle flags into `assemble<BuildType>` invocations. Fixed by enforcing `^[A-Za-z][A-Za-z0-9]{0,63}$` on `build_type`.
- **Sensitive paths and parameters in redacted usage logs (Low):** `CRUCIBLE_USAGE_LOG=redacted` logged free-text parameters (such as `device_info`) and absolute paths embedded in `error` strings, and created `$LINC_HOME` with default permissions. Fixed by switching `redacted` mode to a safe-parameter allowlist (replacing all other parameter values with `<redacted len=N>`), scrubbing absolute paths in `error` to `<path>`, normalizing `NaN`/`Infinity`, and creating state directories with mode `0700` and log files with mode `0600`.
- **Optional dependency and AI ranker hardening (Low):** Removed the unpublishable `[ai]` extra from package metadata, kept the internal AI helper in development dependencies only, and restricted AI asset-ranking tool permissions to read-only (`Read`).
- **Unbounded device-fleet YAML reads and alias expansion (Low):** Capped `device-fleet.yaml` reads at 1 MiB, bounded YAML node composition to 100,000 nodes with cycle detection on recursive aliases, and replaced recursive tree walks with iterative traversal.
- **Input bounds, malformed manifest handling, and gate exit codes (Low):** Added length, NUL-byte, and numeric bounds across all MCP tool inputs, handled malformed or non-string `.forge.yaml` commands without unhandled exceptions, failed `crucible assay` (code `UNMEASURED`) and `crucible bvt` (`ran: false`) with exit code `1` when nothing was measured or run, honored custom `launch_intent` values without falling back to private package prefixes, filtered personal serials out of `crucible fleet discover` by default, and replaced `opencv-python-headless` with `Pillow` + `numpy`.

Candidate wheel SHA-256 (pre-fix): `16e5ec182a09f2c5b9dc65e9240342d7061fdb8154d87c79d77e8bf48563288c`.

### 2026-10-25 post-fix verification

A second verification run on 2026-10-25 re-executed all 45 adversarial vectors and the stranger run against the fixed release candidate wheel (`f38fe86a756dfbd46bed3fdc43d3770c2a6e21fc0f044c22657963d22df92668`, built from private `linc-crucible` commit `f50cc98`).

Results: 0 High, 0 Medium, 0 Low reproduced. All 45 adversarial vectors and the stranger run passed cleanly.

### Release candidate verification

The reviewed v0.2.0 release candidate contains all fixes above:
- Source: private `linc-crucible` commit `f50cc98` (tree `be775263b096e9a416f9e39342629185df368bdb`). Commits after it and before the `v0.2.0` tag change only `SECURITY.md` and its test, not the package.
- Reviewed candidate wheel SHA-256: `f38fe86a756dfbd46bed3fdc43d3770c2a6e21fc0f044c22657963d22df92668`
- The wheel published on PyPI will have a different SHA-256 even though its contents are the same, because a wheel records each file's timestamp and the published build's timestamps differ from the candidate's.
- After `v0.2.0` is published, this section names the public `v0.2.0` tag, its commit, and the SHA-256 of the wheel PyPI serves. To check that hash: `pip download linc-crucible==0.2.0 --no-deps -d .` then `shasum -a 256 linc_crucible-0.2.0-py3-none-any.whl`.

### 2026-11-19 v0.3.0 pre-release review

The changes since v0.2.0 (the `mcp` 2.3 migration, `crucible apk-audit`, BVT lease owners and activity overrides, and the JUnit XML test counts) were reviewed adversarially before the v0.3.0 release in three rounds (an initial review and two post-fix verification passes), each by a separate session with no access to fix history. Every round ran against an installed candidate wheel with an isolated home directory, a fake nerve daemon, recording shims for `adb`, `aapt2`, `apksigner`, `gradlew` and `xcodebuild`, and proxies pointed at a closed port; no device or live daemon was contacted. No round found a High. The findings were fixed before release, except the Lows listed as known issues in the `[0.3.0]` section of the [changelog](https://github.com/LincForge/crucible/blob/main/CHANGELOG.md):
- **`apk-audit` false green (Medium, first round):** with `aapt2` or `apksigner` missing, or an `--assetlinks` check that could not run, `--strict` reported `passed: true` and exited 0 without checking the manifest or signature. Fixed by adding `summary.measured`, setting `passed` to false whenever a requested layer could not run, and printing `[UNMEASURED]`.
- **Preflight fell back past an unreadable score (Medium, second round):** when the newest score file was truncated or malformed (a non-atomic save interrupted mid-write), preflight silently used an older passing score. Fixed by writing scores atomically and failing preflight, naming the skipped files, whenever a newer score could not be read.
- **`apk-audit` input handling (Low, all rounds):** terminal escape and bidi characters from APK strings in `--human` output, APK paths starting with `-` passed to `aapt2`/`apksigner` as options, oversized integers in `assetlinks.json`, central directories parsed before the entry cap applied, `aapt2` output lines forged through Unicode line separators, and duplicate `android:` attributes. Fixed by escaping output, passing resolved absolute paths, bounding the central directory at 16 MiB and the declared and parsed entry counts at 50,000, splitting `aapt2` output on LF only, and flagging duplicate attributes as high.
- **Measurement honesty (Low, all rounds):** stale JUnit XML counted as a fresh test run, a FIFO in the score store hanging `crucible score` and preflight, a future-dated score hiding later runs, and a capped test-results search silently dropping modules. Fixed by counting only XML written during the run, refusing non-regular score files, ignoring scores dated ahead of the host clock, and reporting a search that hits its directory cap as `UNMEASURED`.
- **`crucible_bvt` (Low, first round):** an unconfined `output_dir`, and negative or non-finite step durations reaching `time.sleep`. Fixed by confining `output_dir` inside `$LINC_HOME/crucible` and validating durations at load.

After `v0.3.0` is published, this section names the public `v0.3.0` tag, its commit, and the SHA-256 of the wheel PyPI serves. To check that hash: `pip download linc-crucible==0.3.0 --no-deps -d .` then `shasum -a 256 linc_crucible-0.3.0-py3-none-any.whl`.
