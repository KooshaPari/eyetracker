# EYE-SOTA-002 — Tobii Stream Engine Integration

## Goal

Add native, low-latency Tobii eye-tracker support to eyetracker via the
Tobii Stream Engine C API, with safe Rust bindings and a hardware
abstraction layer (HAL) trait that the existing `eyetracker-camera` crate
can implement alongside the macOS webcam backend.

## Scope

In-scope:
- New crate `eyetracker-tobii` (workspace member)
- `tobii-research-sys` FFI shim or `tobii` crate usage
- `GazeData` / `EyeTracker` traits in `eyetracker-core` extended
- Stream Engine: address-based enumeration, gaze subscription,
  coordinate mapping, timestamp alignment
- Example binary in `eyetracker-cli`: `eyetracker-cli tobii --list` and
  `eyetracker-cli tobii --record 10s`
- Documentation in `crates/eyetracker-tobii/README.md`

Out-of-scope (separate DAG units):
- Pupil Labs / ARKit / Vision OS adapters (EYE-SOTA-003..006)
- Web transport (EYE-SOTA-007)
- Calibration robustness (EYE-SOTA-008)

## SOTA edge

- Address-based discovery (not just USB enumeration) — supports network
  Stream Engine installs
- Sample-accurate timestamps (`system_time_stamp` + `device_time_stamp`
  cross-correlation) for video frame sync
- Gaze origin + direction vector, not just screen coordinates (needed
  for AR/VR downstream)
- Multi-tracker fusion via `eyetracker-fusion` later

## Acceptance criteria

- [ ] `cargo build -p eyetracker-tobii` clean
- [ ] `cargo test -p eyetracker-tobii` passes (mock + live if hardware)
- [ ] `cargo +nightly fmt --check` clean
- [ ] `cargo clippy -p eyetracker-tobii -- -D warnings` clean
- [ ] Example binary produces CSV at >=60Hz on reference hardware
- [ ] No new `unwrap()` in crate source
- [ ] `grade.sh --fast` for eyetracker moves fmt PASS, adds no FAILs

## Reference crates

- `tobii` (Rust wrapper, MIT/Apache-2.0)
- `tobii-research-sys` (raw FFI, optional)
- Tobii Stream Engine SDK C headers (proprietary, no redistribution)
