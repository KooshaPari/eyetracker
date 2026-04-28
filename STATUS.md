# Status

Last updated: 2026-04-27

## Build
TBD - GitHub Actions billing-blocked org-wide

## Quality gates (enrolled, awaiting billing for live runs)
- cargo-deny.yml: Monday 09:00 UTC cron + push/PR + workflow_dispatch
- codeql-rust.yml: Tuesday 04:17 UTC cron + push/PR + workflow_dispatch
- cargo-audit.yml: Wednesday 05:37 UTC cron + push/PR + workflow_dispatch
- pre-commit: client-side (cargo fmt + check + gitleaks)
- branch protection: 1 reviewer required, no force-push, dismiss stale

## Live verification
GitHub Actions billing-blocked. Local cargo-deny+audit weekly via `governance/scripts/cargo-deny-org-weekly.sh`.

## Cross-references
See `phenotype-org-governance/SUPERSEDED.md` for canonical authority.
See `phenotype-org-governance/CHANGELOG_2026_04_27.md` for current sprint state.
