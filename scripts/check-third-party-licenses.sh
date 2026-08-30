#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo-about >/dev/null 2>&1; then
  echo "cargo-about 0.9.2 is required to verify third-party licenses" >&2
  exit 1
fi

generated=$(mktemp "${TMPDIR:-/tmp}/rust-spine-licenses.XXXXXX")
cleanup() {
  rm -f -- "$generated"
}
trap cleanup EXIT

cargo about generate about.hbs > "$generated"
if ! cmp -s THIRD_PARTY_LICENSES.html "$generated"; then
  echo "THIRD_PARTY_LICENSES.html is stale; regenerate it with:" >&2
  echo "  cargo about generate about.hbs > THIRD_PARTY_LICENSES.html" >&2
  exit 1
fi

echo "third-party license bundle is current"
