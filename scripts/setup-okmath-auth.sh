#!/bin/bash
# Configure this clone to always authenticate to GitHub as OkMath-Org,
# regardless of which account is active in `gh auth`.  Run once after
# cloning the repo.
#
# What this does:
#   - Marks scripts/credential-helper-okmath.sh executable.
#   - Overrides the global git credential helper (typically Git
#     Credential Manager) for THIS clone only, by clearing the helper
#     chain and pointing it at the local script.
#   - Verifies the helper can fetch a token.
#
# Idempotent: safe to run multiple times.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$repo_root" ]; then
    echo "error: not inside a git repository" >&2
    exit 1
fi

helper="$repo_root/scripts/credential-helper-okmath.sh"
if [ ! -f "$helper" ]; then
    echo "error: $helper not found" >&2
    exit 1
fi

chmod +x "$helper"

if ! command -v gh >/dev/null 2>&1; then
    echo "error: GitHub CLI (gh) not found in PATH" >&2
    exit 1
fi

if ! gh auth token --user OkMath-Org --hostname github.com >/dev/null 2>&1; then
    echo "error: gh has no stored token for OkMath-Org." >&2
    echo "  run \`gh auth login\` and authenticate as OkMath-Org first." >&2
    exit 1
fi

# Reset any previously configured local helpers, then install ours.
git config --local --unset-all credential.helper 2>/dev/null || true
# The first empty entry clears the chain inherited from the global
# config (otherwise GCM would still run first and supply a different
# account's token).  The second entry is our helper.
git config --local --add credential.helper ""
git config --local --add credential.helper "!\"\$(git rev-parse --show-toplevel)/scripts/credential-helper-okmath.sh\""

echo "configured credential helper:"
git config --local --get-all credential.helper | sed 's/^/  /'
echo
echo "verifying authentication..."
if git ls-remote origin HEAD >/dev/null 2>&1; then
    echo "  OK -- git operations on this clone now use the OkMath-Org token."
else
    echo "  FAILED -- git ls-remote did not authenticate." >&2
    exit 1
fi
