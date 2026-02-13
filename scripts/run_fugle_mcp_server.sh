#!/usr/bin/env bash
set -euo pipefail

# Prefer API_KEY expected by Fugle MCP server; fallback to app-level env var.
if [[ -z "${API_KEY:-}" && -n "${FUGLE_MARKETDATA_API_KEY:-}" ]]; then
  export API_KEY="${FUGLE_MARKETDATA_API_KEY}"
fi

# Fallback: key file path via env, then default iCloud key file.
if [[ -z "${API_KEY:-}" ]]; then
  key_file="${FUGLE_MARKETDATA_API_KEY_FILE:-${FUGLE_API_KEY_FILE:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/codexapp/fuglekey}}"
  if [[ -n "${key_file}" && -f "${key_file}" ]]; then
    key_text="$(cat "${key_file}")"
    key_text="${key_text#"${key_text%%[![:space:]]*}"}"
    key_text="${key_text%"${key_text##*[![:space:]]}"}"
    if [[ -n "${key_text}" ]]; then
      export API_KEY="${key_text}"
    fi
  fi
fi

if [[ -z "${API_KEY:-}" ]]; then
  echo "Missing API key. Set API_KEY/FUGLE_MARKETDATA_API_KEY or provide key file." >&2
  exit 1
fi

exec npx -y "https://github.com/fugle-dev/fugle-marketdata-mcp-server/releases/download/v0.0.1/fugle-marketdata-mcp-server-0.0.1.tgz"
