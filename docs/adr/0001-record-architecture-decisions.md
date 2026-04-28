# ADR-0001: Record Architecture Decisions

## Status
Accepted

## Context
We need to record the architectural decisions made on this project to maintain shared understanding and avoid revisiting them as people come and go from the project.

## Decision
We will use Architecture Decision Records (ADRs) as described by Michael Nygard. Each ADR is a numbered Markdown file under `docs/adr/`.

## Consequences
- Every architectural decision is captured as a numbered, immutable Markdown file
- Decisions can be marked Accepted, Superseded, Deprecated, or Rejected
- New ADRs supersede prior ones via explicit cross-reference
- Bus-factor mitigation: rationale for past decisions is auditable

## Reference
- Michael Nygard, "Documenting Architecture Decisions": https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- joelparkerhenderson/architecture-decision-record: https://github.com/joelparkerhenderson/architecture-decision-record
