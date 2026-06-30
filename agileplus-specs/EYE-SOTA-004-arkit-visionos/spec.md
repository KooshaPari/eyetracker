# EYE-SOTA-004 — ARKit / Vision OS / WebXR Gaze Adapters

## Goal

Add eye-gaze adapters for consumer Apple platforms (iOS via ARKit,
visionOS via VisionOS Eye Tracking) and a WebXR device API
adapter, all surfacing through the unified `GazeSource` HAL.

## Scope

In-scope:
- New crate `eyetracker-platform` (workspace member) containing:
  - `arkit` feature: bridge to ARKit face-tracking Gaze API
    (iOS only via `objc2` + FFI)
  - `visionos` feature: bridge to `ARKit` + `RealityKit` eye
    tracking on visionOS
  - `webxr` feature: WASM-bindgen wrapper around the WebXR
    `EyeTracking` API (`immersive-vr` session with eye-tracking
    input)
- WGSL shader for visionOS gaze-rendering (pointer highlight)
- Example binaries (each gated by feature):
  - `eyetracker-cli webxr-demo` (web)
  - `eyetracker-cli ios-gaze-overlay` (iOS)

Out-of-scope:
- Android `androidx.eyetracking` (separate DAG: EYE-SOTA-006)
- HoloLens 2 MRTK eye gaze (separate DAG: EYE-SOTA-007)
- Real-time mesh-of-attention rendering

## SOTA edge

- Sub-frame latency budget: 8ms (60Hz head-locked reticle)
- Privacy: explicit user-consent gate on iOS / visionOS
  (`request_tracking_authorization()` before any sampling)
- Browser: `gated-features` query string (`requiredFeatures: eye-tracking`)
- Multi-modal fallback: when eye tracking denied, gracefully
  degrade to head-pointer + dwell-click

## Acceptance criteria

- [ ] `cargo build -p eyetracker-platform --features arkit` clean
- [ ] `cargo build -p eyetracker-platform --features visionos` clean
- [ ] `cargo build -p eyetracker-platform --features webxr` clean (wasm32-unknown-unknown)
- [ ] `cargo test -p eyetracker-platform` passes (mock providers)
- [ ] `cargo +nightly fmt --check` clean
- [ ] `cargo clippy -p eyetracker-platform -- -D warnings` clean
- [ ] No new `unwrap()` in crate source

## Reference

- ARKit Face Tracking: Apple Developer docs
- visionOS: `ARKit` + `RealityKit` eye-tracking sample
- WebXR Eye Tracking: `https://www.w3.org/TR/webxr-eye-tracking/`
- `wasm-bindgen` 0.2 for WebXR FFI
- `objc2` 0.5 for ARKit FFI

## Why this is its own WP

Apple platform code requires Xcode toolchain (Apple Clang,
SDK conditional compilation). Keeping it isolated from the rest
of eyetracker keeps CI happy on Linux runners and lets Apple
contributors iterate fast in their native toolchain.
