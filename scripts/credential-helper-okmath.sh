#!/bin/bash
# Git credential helper for the OkMathOrg/bouncing-cosmology repo.
#
# Always returns the OkMath-Org token from gh's keyring, without
# touching the globally active gh account.  This lets the maintainer
# work on other GitHub accounts (e.g. fameracecom for unrelated
# projects) in parallel terminals without having to `gh auth switch`
# before pushing here.
#
# Activation (once per clone):
#   bash scripts/setup-okmath-auth.sh
#
# Requirements:
#   - GitHub CLI (`gh`) installed
#   - `gh auth login` performed for the OkMath-Org account at least
#     once, so the token is stored in the keyring
case "$1" in
    get)
        token="$(gh auth token --user OkMath-Org --hostname github.com 2>/dev/null)"
        if [ -n "$token" ]; then
            echo "username=OkMath-Org"
            echo "password=$token"
        fi
        ;;
    store|erase)
        # No-op: credentials live in gh's keyring, not in our store.
        cat >/dev/null
        ;;
esac
