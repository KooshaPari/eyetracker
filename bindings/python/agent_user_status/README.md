# phenotype-eye-tracker-bindings

Python bindings for [KooshaPari/eyetracker](https://github.com/KooshaPari/eyetracker).
Absorbed from `KooshaPari/agent-user-status` on 2026-07-28 (see `ABSORBED.md`).

## Modules

| Module | Purpose |
|---|---|
| `gaze_calibration` | 9-point and 5-point calibration routines |
| `gaze_context` | Screen, environment, fixation context |
| `gaze_drift_correction` | Long-running drift correction |
| `gaze_evaluation` | Accuracy / precision evaluation harness |
| `gaze_projection` | Screen coordinate projection |
| `eye_publish` | Eye-event publisher (IPC / JSON lines) |
| `eye_smoothing` | Eye-state smoothing filters (Kalman, EWMA) |
| `eye_state_payload` | Eye-state payload schema (TypedDict) |
| `webcam_eye_tracker` | Webcam-only eye tracker (no native deps) |
| `webcam_probe` | Webcam capability probe |
| `webcam_runtime` | Webcam tracker runtime loop |
| `webcam_support` | Webcam helpers (lighting, face detection) |
| `cursor_tracker` | Cursor-tracker companion daemon |
| `bootstrap_native` | Native binary bootstrap helper |

## Installation

```bash
pip install phenotype-eye-tracker-bindings[cv]
```

## Status

**Migrated-placeholder** — see `ABSORBED.md`. The modules import helpers from
the original `agent_user_status` package layout (`optional_dependencies`,
`bootstrap_support`) that are not yet lifted into this package. A follow-up PR
will complete the package split.
