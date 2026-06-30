# Safety — Unsafe Block Inventory & FFI Invariants

This document catalogs every `unsafe` block in the codebase, the invariants that
justify them, and the review obligations for maintainers.

## Inventory

### `crates/eyetracker-cli/src/mouse.rs` — scroll event dispatch

**Location:** `mouse.rs` inside `#[cfg(target_os = "macos")] mod platform`, function
`scroll_at`.

**Why unsafe:** `CGEvent::new_scroll_event` is called via the `core-graphics` crate.
The binding is marked `unsafe` because the underlying `CGEventCreateScrollWheelEvent2`
C API requires the caller to ensure:

1. The `CGEventSource` passed is a valid, non-NULL pointer (guaranteed by the
   `CGEventSource::new(…).expect(…)` call immediately above).
2. The scroll unit enum value is a valid `kCGScrollEventUnitLine` / `kCGScrollEventUnitPixel`
   constant — enforced by the `ScrollEventUnit::LINE` type used here.
3. `delta` fits in `i32` — the cast from `i64` is bounded because `lines` is in
   `[-i32::MAX/10, i32::MAX/10]` at every call site.

**Audit status:** Reviewed 2026-06-30. No unsoundness found. Gate condition: macOS
only (`#[cfg(target_os = "macos")]`); non-macOS stub is pure safe Rust.

**Reviewer obligation:** Re-audit if `core-graphics` crate version is bumped or if
the `scroll_at` signature changes.

---

## FFI Invariants (UniFFI / JNI)

### UniFFI boundary (`crates/eyetracker-ffi`)

UniFFI-generated code handles marshalling between Rust and Swift/Kotlin. The
invariants that must hold at the FFI boundary are:

- **Pointer lifetimes:** All objects passed across the boundary are owned Arc<T>
  values; raw pointers are never exposed to callers.
- **Panic safety:** Rust panics must not unwind across the FFI boundary. Every
  exported function is wrapped in `std::panic::catch_unwind` (UniFFI's default
  scaffolding) or must be reviewed to confirm absence of panicking paths.
- **Thread safety:** All exported types implement `Send + Sync`; UniFFI scaffolding
  enforces this at compile time.
- **Null safety:** Nullable types are expressed as `Option<T>` on the Rust side;
  UniFFI maps these to Swift `Optional` / Kotlin nullable types automatically.

### JNI boundary (future — `FR-EYE-INTEROP-002`)

JNI bindings are planned but not yet implemented. When added:

- All JNI functions must be declared `extern "system"` with the correct signature.
- JVM object lifetimes must be managed via `JNIEnv::new_global_ref` where the
  object outlives the JNI call.
- Exception handling: check `JNIEnv::exception_check` after every JNI call that
  can throw.

---

## Review Checklist for New Unsafe Code

Before adding a new `unsafe` block, a reviewer must confirm:

- [ ] The invariant that justifies the block is documented here.
- [ ] The block is as small as possible (prefer `unsafe fn` wrappers over inline
  `unsafe` in business logic).
- [ ] A `// SAFETY:` comment in the source explains the invariant at the call site.
- [ ] MIRI or address-sanitizer has been run on the affected module (record result
  here with date).
