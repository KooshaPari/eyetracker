# Threat Model — eyetracker

**Scope:** Local CLI/library processing gaze data from a laptop webcam on macOS.
No network service, no multi-tenant deployment.  
**Last reviewed:** 2026-06-30

---

## Trust Boundaries

```
[Webcam HW] → [camera crate (V4L2/AVFoundation)] → [inference crate]
                                                          ↓
                                                   [calibration store on disk]
                                                          ↓
                                         [CLI TUI / UniFFI/JNI consumer]
                                                          ↓
                                              [macOS CGEvent (mouse output)]
```

The only external inputs are:

1. Raw video frames from the webcam driver.
2. Calibration data loaded from `~/Library/Application Support/eyetracker/cal.bin`.
3. CLI flags provided by the local operator.

---

## Attack Catalog

### T-01 — Gaze Data Privacy Leak

**Threat:** A malicious process reads gaze coordinates from stdout/stderr or from
an IPC channel exposed by the CLI.

**Mitigations:**
- Default mode: gaze data is rendered in the TUI only; no file or socket output.
- `--csv` mode writes to stdout — operator is responsible for pipe destination.
- `FR-EYE-PRIVACY-001`: all inference is on-device.
- `FR-EYE-PRIVACY-002`: no default cloud upload.

**Residual risk:** Low for default usage; medium if `--csv` is piped to a
world-readable file.

### T-02 — Malicious ONNX Model File

**Threat:** An attacker replaces the bundled ONNX model with a crafted file
designed to exploit the ONNX Runtime parser.

**Mitigations:**
- `download-models.sh` should verify SHA-256 of downloaded models (TODO: implement).
- Model files should not be loaded from user-writable paths without checksum
  verification.

**Residual risk:** Medium until checksum verification is implemented.

### T-03 — Crafted Calibration File

**Threat:** An attacker writes a malformed `cal.bin` to trigger a panic or memory
corruption during deserialization (bincode).

**Mitigations:**
- Deserialization uses `bincode` with typed structs — no arbitrary code execution.
- A corrupted file produces a `bincode` error; the application falls back to
  uncalibrated mode.
- File path is user-specific (`~/Library/Application Support/…`) — only the local
  user can write it.

**Residual risk:** Low; panic on malformed data is not a security boundary crossing.

### T-04 — Unsafe FFI Exploitation

**Threat:** A bug in the `unsafe` CGEvent scroll call (`mouse.rs`) or the UniFFI
boundary allows memory corruption.

**Mitigations:**
- See `docs/SAFETY.md` for the full unsafe inventory and invariants.
- The unsafe block is macOS-only and isolated to `scroll_at`.
- UniFFI scaffolding wraps exported functions in `catch_unwind`.

**Residual risk:** Low while the invariants documented in `SAFETY.md` hold.

### T-05 — Screen Recording Capture

**Threat:** If screen recording is enabled for debugging, a secondary process reads
the recorded frames.

**Mitigations:**
- `FR-EYE-PRIVACY-003`: consent dialog required before any session data is
  persisted.
- Screen recording is off by default.

**Residual risk:** Low (user-initiated, explicit consent required).

### T-06 — Privilege Escalation via CGEvent

**Threat:** The `CGEvent` API (posting mouse clicks) could be abused if the
process is compromised to click through security dialogs.

**Mitigations:**
- macOS Accessibility permission is required to post CGEvents; the user grants
  this explicitly.
- `--no-mouse-output` flag disables all CGEvent posting.

**Residual risk:** Inherent to accessibility-category tools; matches threat level
of any assistive technology.

---

## Input Validation Strategy

| Input | Validation |
|---|---|
| CLI flags | Clap parser with typed fields + range checks (e.g., `dwell_ms` 200–1000). |
| Calibration file | `bincode::deserialize` with typed struct; error → fallback. |
| ONNX model file | Path is an embedded constant; future: SHA-256 gate in `download-models.sh`. |
| Camera frames | `image` crate decoding; malformed frames produce decode errors, not panics. |
| Gaze coordinates | Normalized to `[0, 1]` inside inference; clamped before CGEvent dispatch. |

---

## Out of Scope

- Network-level threats (no listener port, no HTTP API).
- Multi-tenant data isolation (single-user local process).
- Supply chain attacks on Rust crates (covered by `cargo-audit` + `cargo-deny`
  workflows and OpenSSF Scorecard).
