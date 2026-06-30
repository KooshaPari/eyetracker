# Agent Instructions for eyetracker

This repository is part of the [Phenotype](https://github.com/KooshaPari) ecosystem.

## Stack

Primary language: **Rust** (Cargo workspace; seven crates under `crates/`)

| Crate | Role |
|---|---|
| `eyetracker-domain` | Domain model types |
| `eyetracker-math` | Gaze/calibration geometry |
| `eyetracker-core` | Orchestration pipeline |
| `eyetracker-ffi` | UniFFI/JNI binding boundary |
| `eyetracker-camera` | Input adapter (webcam) |
| `eyetracker-inference` | ML inference + classification |
| `eyetracker-cli` | Operator/test harness CLI (ratatui TUI) |

Bindings under `bindings/` target Swift (UniFFI) and Kotlin (JNI) callers, but the source language is Rust throughout.

## Conventions

- Branches: `feat/*`, `fix/*`, `chore/*`, `docs/*` off `main`.
- Commits: Conventional Commits preferred (`feat:`, `fix:`, `chore:`, `docs:`).
- PRs: open ready-to-merge unless explicitly WIP. Squash-merge with branch deletion is the default.
- Quality gates: `cargo clippy --workspace -- -D warnings` + `cargo test --workspace --locked` must pass locally before pushing. CI is currently billing-blocked; do not block on CI status.
- Benchmarks: `cargo bench` via criterion in `crates/eyetracker-math/benches/`.

## Phenotype Org Policy

See `~/.claude/CLAUDE.md` (global) for the canonical Phenotype Org Cross-Project Reuse Protocol, billing constraints, and scripting language hierarchy.

## File Organization

Per the global `Phenotype/CLAUDE.md`, all docs except spec roots live under `docs/<category>/`. Spec roots (PRD, ADR, FUNCTIONAL_REQUIREMENTS, PLAN, USER_JOURNEYS) live at root.

## Security & Safety

- See `docs/SAFETY.md` for unsafe block inventory and FFI invariants.
- See `docs/THREAT_MODEL.md` for gaze privacy and input-validation attack surface.
- Report vulnerabilities via `SECURITY.md`.
