# Status

Last updated: 2026-06-20

## Build
✅ Release build clean: `cargo build --release --bin eyetracker` (4.56s incremental, 0 warnings, 0 errors)
✅ Binary verified end-to-end on FaceTime HD @ 1280x720@30 FPS (sub-40ms processing latency)

## Release
- **v0.1.0-alpha** published 2026-06-20 ([release](https://github.com/KooshaPari/eyetracker/releases/tag/v0.1.0-alpha))
- PR #64 (Rust port from Python) merged 2026-06-20 as commit `1e078df`

## Quality gates
- `cargo check --workspace --all-targets`: 0 errors, 0 warnings
- `cargo test --workspace`: 112 tests passing, 0 failing
- Branch protection: 1 reviewer required, no force-push
- CI workflows: cargo-deny, codeql, cargo-audit, pre-commit (weekly local via `governance/scripts/cargo-deny-org-weekly.sh`)

## Functional requirements (16/16 in scope)
- Calibration (5/5): 9-point grid, 1.5° drift tolerance, persistence, drift monitor + dismiss, multi-monitor
- Inference (4/4): ≤30ms latency, Kalman 2D smoothing, fixation/saccade classification
- Accessibility (2/2): Dwell-click (500ms), gaze-scroll (top/bottom 20%)
- Privacy (3/3): On-device, no default cloud, per-session recording consent
- Interop (2/2): UniFFI Swift + Kotlin bindings generated, FocalPoint NDJSON connector

## Cross-references
See `phenotype-org-governance/SUPERSEDED.md` for canonical authority.
See `phenotype-org-governance/CHANGELOG_2026_04_27.md` for prior sprint state.
