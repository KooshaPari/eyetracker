# eyetracker ABSORPTION_MANIFEST

This repo is the canonical home for phenotype eye-tracking. Absorbed sources
are recorded here as **migrated**, **migrated-placeholder**, or **rejected**.

## Active absorptions

### agent-user-status → eyetracker (2026-07-28)
- Source: `KooshaPari/agent-user-status` (33 branches, 318 files)
- Status: **migrated-placeholder**
- What landed: `bindings/python/agent_user_status/{gaze,eye,webcam,cursor,bootstrap}*.py`
  and 7 unit tests
- What's outstanding: helpers from original `agent_user_status` package
  (`optional_dependencies`, `bootstrap_support`) not yet lifted.
- Source branches preserved as `refs/sources/agent-user-status/<branch>`.
- See `bindings/python/agent_user_status/ABSORBED.md` for details.
