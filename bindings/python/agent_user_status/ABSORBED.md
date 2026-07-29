# agent-user-status → absorbed into KooshaPari/eyetracker (2026-07-28)

## Migration state: SPLIT-ABSORBED

The `agent-user-status` repo (33 branches, 318 files) split into two homes:

- Skill + MCP + iMessage CLI/lib → `KooshaPari/PhenoMCPServers`
- **Eye/gaze/webcam + macOS Swift + Python bindings → this repo (`eyetracker`)**

## What was absorbed here

### Python bindings (`bindings/python/agent_user_status/`)

| File | Purpose |
|---|---|
| `gaze_calibration.py` | Gaze calibration routines |
| `gaze_context.py` | Gaze context (screen, environment, fixations) |
| `gaze_drift_correction.py` | Long-running drift correction |
| `gaze_evaluation.py` | Accuracy / precision evaluation harness |
| `gaze_projection.py` | Screen coordinate projection |
| `eye_publish.py` | Eye-event publisher (IPC/JSON lines) |
| `eye_smoothing.py` | Eye-state smoothing filters |
| `eye_state_payload.py` | Eye-state payload schema |
| `webcam_eye_tracker.py` | Webcam-only eye tracker (no native deps) |
| `webcam_probe.py` | Webcam capability probe |
| `webcam_runtime.py` | Webcam tracker runtime loop |
| `webcam_support.py` | Webcam helpers (lighting, face detection) |
| `cursor_tracker.py` | Cursor-tracker companion daemon |
| `bootstrap_native.py` | Native binary bootstrap helper |

### Tests (`bindings/python/agent_user_status/tests/`)

`test_eye_publish.py`, `test_eye_smoothing.py`, `test_gaze_drift_correction.py`,
`test_gaze_evaluation.py`, `test_gaze_projection.py`, `test_webcam_eye_tracker.py`,
`test_webcam_probe.py`

## Status: MIGRATED — placeholder

The Python modules import `agent_user_status.optional_dependencies` and other
helpers that **stayed in the archived agent-user-status daemon**. Until a
follow-up PR lifts those helpers into this bindings package, the modules
import-cleanup is incomplete. They are tracked here as **migrated-placeholder**
in the repo-wide absorption record.

## macOS native Swift components

`AgentUserStatusApp.swift`, `EyeTrackerControls.swift`, `VisualGazeFilter.swift`,
`WindowTracking.swift`, and 7 other macOS native files in
`src/native/macos/` of the source repo were **NOT** absorbed in this commit:

- Swift Package Manager layout conflicts with this repo's UniFFI Rust bindings.
- They will land in a follow-up PR as `bindings/swift/agent-user-status/`
  once the Swift module structure is finalised.

## Source branches preserved

All 33 source branches preserved as historical refs:

```
refs/sources/agent-user-status/<branch>
```

including:

- `main`
- `feat/journey-impl`
- `user-status-next-dag-hardening`
- `chore/ssot-bundle-2026-06-20`
- `orch-v12-s2-017/tier-0-baseline`
- `pr-26`, `pr-26-tmp`
- `repo-governance-hardening`
- 26 `chore/*`, `ci/*`, `docs/*`, `fix/*` branches

## Source repo

`KooshaPari/agent-user-status` will be squashed to a 1-commit docket and
archived on 2026-07-28 after both absorptions land.

## Superseded by

- `KooshaPari/eyetracker` (this repo, for eye/gaze/Swift)
- `KooshaPari/PhenoMCPServers` (for skill + MCP + iMessage)

## Cross-references

- See `../../ABSORPTION_MANIFEST.md` for the repo-wide absorption record.
- See `bindings/python/agent_user_status/ABSORBED.md` for the migration details
  (this file).
