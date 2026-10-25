# linc-crucible — Architecture

## Overview

Crucible is the quality analysis layer in the LINC 4-layer architecture: `device-kit` → `nerve` → `crucible` / `spectra`. Pure analysis — zero device code. All device interaction flows through NerveClient HTTP adapter at localhost:8111.

## Component Map

```
              ┌──────────────┐
              │  MCP Server   │  8 tools (pour, assay, bvt, score, preflight, baseline,
              └──────┬───────┘           extract_assets, usage_stats)
                     │
     ┌───────────┬───┴───────┬───────────┬────────────┐
     │           │           │           │            │
┌────▼───┐  ┌───▼────┐ ┌────▼─────┐ ┌───▼──────┐ ┌───▼────────┐
│Deployer│  │Assayer │ │Preflight │ │BVTRunner │ │usage_stats │
│ (pour) │  │(assay) │ │ Checker  │ │  (bvt)   │ │(MCP call   │
└────┬───┘  └───┬────┘ └──────────┘ └───┬──────┘ │ log JSONL) │
     │          │                       │        └────────────┘
     │    ┌─────┼─────┐            NerveAdapter
     │    │     │     │            VisualCompare
     │ Scorer  A11y  Store
     │        Engine Checks
     │    ┌────┴────┐
     │  TreeDiffer RuleChecker
     │    │
     │  TreeCapture
     │    │
┌────┼────┼────────┐
│    │    │         │
Nerve  Forge  Baseline
Client Reader Manager
(HTTP)  (YAML) (PNG+JSON)
```

## Key Abstractions

| Class | Purpose |
|-------|---------|
| `QualityScorer` | Weighted 0-100 composite across 7 dimensions. Dynamic weight normalization. Ship gate at 85. |
| `TreeCapture` | Versioned a11y tree baseline storage (JSON + metadata) |
| `TreeDiffer` | Structural diff between ElementNode lists. Penalty: missing=-10, role=-8, label=-5, interactivity=-3, added=-1 |
| `WCAGRuleChecker` | 6 a11y rules. Platform-aware (Android 48dp / iOS 44pt touch targets) |
| `NerveClient` | HTTP client to nerve :8111. Graceful degradation (returns [] on error) |
| `Deployer` | Build-install-launch pipeline. Single/multi-target routing |
| `Assayer` | Quality analysis pipeline: lint → tests → security → perf → a11y → score |
| `PreflightChecker` | 5-dimension store readiness (code 30%, listing 20%, assets 20%, brand 15%, build 15%) |
| `BaselineManager` | Versioned visual screenshot baselines (PNG + metadata) |
| `AssetExtractor` | Video → screenshots via blur detection + pHash dedup |
| `BVTRunner` | Runs YAML BVT flows step by step against a device through nerve |

## Data Flow

### Assay
```
crucible_assay(project)
  → ForgeProjectReader.get(project)
  → Assayer: lint → tests → security → perf → a11y (static) → a11y tree (live via nerve)
  → QualityScorer.compute() → persist to $LINC_HOME/crucible/scores/
  → return AssayReport
```

### Pour
```
crucible_pour(project)
  → NerveClient.discover() → find target devices
  → Deployer._build() → subprocess: gradle/xcodebuild
  → find_apk() → locate artifact
  → NerveClient.install() → NerveClient.launch()
  → return PourResult
```

## File Guide

| File | Purpose |
|------|---------|
| `mcp_server.py` | 8 MCP tool registrations, async wrappers, observability |
| `cli.py` | Click CLI reflected from the MCP tools (JSON on stdout, `--human` Rich rendering) |
| `core/deployer.py` | Build-install-launch orchestration |
| `core/assayer.py` | Quality analysis pipeline |
| `core/preflight.py` | App store readiness checker |
| `core/scorer.py` | Weighted composite scoring |
| `core/tree_capture.py` | A11y tree baseline storage |
| `core/tree_differ.py` | Structural element tree diff |
| `core/rule_checker.py` | WCAG accessibility rules (6) |
| `core/nerve_client.py` | HTTP client to nerve :8111 |
| `core/nerve_convert.py` | Android/iOS class → SemanticRole mapping |
| `core/baseline.py` | Visual screenshot baselines |
| `core/asset_extractor.py` | Video frame extraction + dedup |
| `core/forge_integration.py` | Read-only forge registry + manifest discovery |
| `core/bvt/` | BVT engine: YAML schema, runner, assertions, nerve adapter, visual compare |
| `core/usage_stats.py` | Usage analytics over the local MCP call log |
| `core/observability.py` | JSONL MCP call logging |

## Design Decisions

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Device access | NerveClient HTTP | Direct ADB/simctl | Zero device code — isolation from protocol changes |
| Visual regression | SSIM | Pixel-diff | Correlates with human perception, tolerates minor rendering differences |
| Scoring | Multi-dimensional weighted | Single pass/fail | Surfaces which dimension drags quality down; trend tracking |
| A11y rules | 6 WCAG-aligned | Full WCAG audit | Covers most common issues; extensible per-rule |
| Subprocess | shlex.split + shell=False | shell=True default | Prevents injection; fallback with audit warning |

## Extension Points

1. **New scoring dimension**: Add to `BASE_WEIGHTS` in scorer.py, pass value to `compute()`
2. **New WCAG rule**: Add check in `WCAGRuleChecker.check()`, append `RuleViolation`
3. **New platform mapping**: Add to `_ANDROID_EXACT`/`_IOS_EXACT` in nerve_convert.py
4. **New preflight dimension**: Add weight to `PREFLIGHT_WEIGHTS`, implement `_check_*` method
5. **New build system**: Extend `Deployer._build_for_platforms()`

## Failure Modes

| Failure | Impact | Recovery |
|---------|--------|---------|
| Nerve unreachable | Pour, a11y tree fail | Static analysis + scores still work |
| MobSF unreachable | Security dimension skipped | Score computed without security |
| Forge registry missing | "Project not found" | Add linc-manifest.yaml to project |
| Build timeout (>600s) | Pour fails | Use skip_build for iteration |
| Corrupt baseline | Load returns None | Warning logged, continues without baseline |

## Dependencies

| Package | Purpose |
|---------|---------|
| linc-device-kit | Shared models (ElementNode, Platform, DeviceType) |
| httpx | Nerve HTTP client, MobSF client |
| mcp | FastMCP server |
| av (PyAV) | Video frame decoding |
| opencv-python-headless | Blur detection |
| imagehash + Pillow | Perceptual hashing |
| structlog | Structured logging |

<!-- last verified: 2026-04-19 -->
