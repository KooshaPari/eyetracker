#!/usr/bin/env bash
# check-cliff-template.sh
#
# Flags repos whose cliff.toml is the canonical pattern without
# a template-reference comment.
#
# Exit code 0 = OK (adopted or non-canonical)
# Exit code 1 = WARN (canonical pattern, not adopted)
# Exit code 2 = ERROR (no cliff.toml at all)
set -euo pipefail

if [ ! -f cliff.toml ]; then
  echo "::error::No cliff.toml file"
  exit 2
fi

# If Source comment is present, treat as adopted
if grep -qE '^# Source:.*phenotype-tooling.*cliff' cliff.toml; then
  echo "Adopted (Source comment present)"
  exit 0
fi

# Canonical pattern detection: check for the key markers
if grep -qF 'git-cliff.org/docs/configuration' cliff.toml \
   && grep -qF 'conventional_commits = true' cliff.toml \
   && grep -qF 'filter_unconventional = true' cliff.toml; then
  echo "::warning::Canonical cliff.toml pattern without template reference. See https://github.com/KooshaPari/phenotype-tooling/blob/main/docs/cliff-adoption.md"
  exit 1
fi

# Otherwise, OK (non-canonical, repo-specific cliff.toml)
echo "Non-canonical cliff.toml (OK, repo-specific)"
exit 0
