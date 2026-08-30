#!/usr/bin/env bash
set -euo pipefail

failed=0
release_files=()

while IFS= read -r -d '' file; do
  case "$file" in
    Cargo.lock|fuzz/Cargo.lock|scripts/check-release-hygiene.sh) continue ;;
  esac
  release_files+=("$file")
done < <(git ls-files -z --cached --others --exclude-standard)

check_files() {
  local label="$1"
  local pattern="$2"
  local matches
  matches=$(rg -l --no-messages -e "$pattern" -- "${release_files[@]}" || true)
  if [[ -n "$matches" ]]; then
    echo "$label found in:"
    echo "$matches"
    failed=1
  fi
}

check_files "absolute developer home path" '(/h[o]me/[^/[:space:]]+|/U[s]ers/[^/[:space:]]+|[A-Za-z]:\\U[s]ers\\[^\\[:space:]]+)'
check_files "private or tailnet IPv4 literal" '(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3})'
check_files "credential-shaped value" '(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xai-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|PRIVATE[[:space:]]KEY-----)'
check_files "credential embedded in URL" 'https?://[^/@[:space:]]+:[^/@[:space:]]+@'

if (( failed != 0 )); then
  exit 1
fi

echo "release hygiene checks passed"
