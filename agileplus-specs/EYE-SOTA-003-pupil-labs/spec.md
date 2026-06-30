# EYE-SOTA-003 — Pupil Labs Cloud + Companion Integration

## Goal

Add Pupil Labs eye-tracker support via the Pupil Cloud REST API and
Pupil Companion (LSL-over-network) protocol, with safe Rust
bindings and reuse of the `GazeSource` HAL trait introduced in
EYE-SOTA-002.

## Scope

In-scope:
- New crate `eyetracker-pupil` (workspace member)
- `pupil-cloud` feature: REST client to `api.pupil-labs.com`
  (recordings, events, gaze exports)
- `pupil-companion` feature: ZMQ subscriber for the local network
  broadcast (raw + surface gaze streams)
- Multi-source fusion stub: combine Tobii + Pupil into a single
  `GazeSource` (defer to EYE-SOTA-005 for production fusion)
- Example binary: `eyetracker-cli pupil --replay <recording-id>`

Out-of-scope:
- Pupil Cloud real-time streaming (uses Companion instead)
- ARKit/Vision OS (separate DAG: EYE-SOTA-004)
- Mobile pupil capture apps

## SOTA edge

- Time-series dedup at fusion boundary (Tobii + Pupil clocks
  drift ~5-15ms)
- 3D gaze origin in addition to screen coords (Pupil provides
  scene camera + world camera)
- Latency-aware fusion: pick freshest sample, not latest-arrived
- Privacy: opt-in only for Pupil Cloud; no auto-upload

## Acceptance criteria

- [ ] `cargo build -p eyetracker-pupil` clean
- [ ] `cargo test -p eyetracker-pupil` passes (mock REST + ZMQ)
- [ ] `cargo +nightly fmt --check` clean
- [ ] `cargo clippy -p eyetracker-pupil -- -D warnings` clean
- [ ] No new `unwrap()` in crate source
- [ ] `grade.sh --fast` for eyetracker: no regression

## Reference

- Pupil Cloud API: `https://api.pupil-labs.com/v1/docs`
- Pupil Companion: open-source, ZMQ-based (`tcp://*:50020`)
- Rust ZMQ: `zeromq` crate (libzmq-sys under the hood)
- HTTP: `reqwest` 0.12 (rustls-tls)
