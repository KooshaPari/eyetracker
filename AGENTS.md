# Agent Instructions for eyetracker

This repository is part of the [Phenotype](https://github.com/KooshaPari) ecosystem.

## Stack

Primary language: **Kotlin**

## Conventions

- Branches: `feat/*`, `fix/*`, `chore/*`, `docs/*` off `main`.
- Commits: Conventional Commits preferred (`feat:`, `fix:`, `chore:`, `docs:`).
- PRs: open ready-to-merge unless explicitly WIP. Squash-merge with branch deletion is the default.
- Quality gates: lint + test must pass locally before pushing. CI is currently billing-blocked; do not block on CI status.

## Phenotype Org Policy

See `~/.claude/CLAUDE.md` (global) for the canonical Phenotype Org Cross-Project Reuse Protocol, billing constraints, and scripting language hierarchy.

## File Organization

Per the global `Phenotype/CLAUDE.md`, all docs except spec roots live under `docs/<category>/`. Spec roots (PRD, ADR, FUNCTIONAL_REQUIREMENTS, PLAN, USER_JOURNEYS) live at root.