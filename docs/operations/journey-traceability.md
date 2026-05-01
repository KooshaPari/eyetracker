# Journey Traceability

**Repo:** eyetracker: eye-tracking library
**Standard:** [phenotype-infra journey-traceability standard](https://github.com/kooshapari/phenotype-infra/blob/main/docs/governance/journey-traceability-standard.md)  
**Schema:** [phenotype-journeys Manifest schema](https://github.com/kooshapari/phenotype-journeys/blob/main/schema/manifest.schema.json)

## User-facing flows

- Primary library API usage patterns
- Configuration and initialization
- Metric/event emission and collection
- Error handling patterns

## Keyframe capture schedule

Keyframes should be captured for: initialization, metric emission, collection endpoints, error states.

## Icon set

`docs/operations/iconography/` — Fluent + Material SVG icons. See `SPEC.md`.

## Manifest location

Journey manifests: `docs/journeys/manifests/`  
Manifest schema: `manifest.schema.json` (from phenotype-journeys)

## CI Gate

`.github/workflows/journey-gate.yml` — **Stub, populate manifests to pass CI**
