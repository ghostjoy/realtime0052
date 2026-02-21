#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/rollback_legacy_stack.sh [options]

Options:
  --sqlite-path <path>  SQLite path to use (default: $REALTIME0052_DB_PATH or iCloud/codexapp/market_history.sqlite3)
  --no-sync             Skip `uv sync`
  -h, --help            Show this help
EOF
}

expand_home() {
  local p="$1"
  if [[ "${p}" == "~"* ]]; then
    echo "${HOME}${p:1}"
  else
    echo "${p}"
  fi
}

path_scope() {
  local p="$1"
  local icloud_root="${HOME}/Library/Mobile Documents/com~apple~CloudDocs"
  if [[ "${p}" == "${icloud_root}"* ]]; then
    echo "iCloud"
  else
    echo "local"
  fi
}

ICLOUD_CODEXAPP_DIR="${HOME}/Library/Mobile Documents/com~apple~CloudDocs/codexapp"

default_sqlite_path() {
  if [[ -n "${REALTIME0052_DB_PATH:-}" ]]; then
    echo "${REALTIME0052_DB_PATH}"
  elif [[ -d "${ICLOUD_CODEXAPP_DIR}" ]]; then
    echo "${ICLOUD_CODEXAPP_DIR}/market_history.sqlite3"
  else
    echo "market_history.sqlite3"
  fi
}

SQLITE_PATH="$(default_sqlite_path)"
DO_SYNC="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sqlite-path)
      [[ $# -ge 2 ]] || { echo "[error] --sqlite-path needs a value"; exit 1; }
      SQLITE_PATH="$2"
      shift 2
      ;;
    --no-sync)
      DO_SYNC="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

SQLITE_PATH="$(expand_home "${SQLITE_PATH}")"

echo "[info] root        : ${ROOT_DIR}"
echo "[info] sqlite path : ${SQLITE_PATH}"
echo "[info] uv sync     : ${DO_SYNC}"
echo "[info] sqlite scope: $(path_scope "${SQLITE_PATH}")"

if [[ "${DO_SYNC}" == "true" ]]; then
  echo "[step] uv sync"
  uv sync
fi

if [[ ! -f "${SQLITE_PATH}" ]]; then
  echo "[warn] sqlite file not found yet: ${SQLITE_PATH}"
  echo "[warn] app will create a new db file if needed."
fi

export REALTIME0052_CONFIG_SOURCE="legacy_env"
export REALTIME0052_STORAGE_BACKEND="sqlite"
export REALTIME0052_DB_PATH="${SQLITE_PATH}"
export REALTIME0052_KLINE_RENDERER_LIVE="plotly"
export REALTIME0052_KLINE_RENDERER_REPLAY="plotly"
export REALTIME0052_BENCHMARK_RENDERER="plotly"

unset REALTIME0052_DUCKDB_PATH || true
unset REALTIME0052_PARQUET_ROOT || true
unset REALTIME0052_AUTO_MIGRATE_LEGACY_SQLITE || true

echo "[stack] config_source = ${REALTIME0052_CONFIG_SOURCE}"
echo "[stack] storage_backend = ${REALTIME0052_STORAGE_BACKEND}"
echo "[stack] db_path = ${REALTIME0052_DB_PATH}"
echo "[stack] renderer_live = ${REALTIME0052_KLINE_RENDERER_LIVE}"
echo "[stack] renderer_replay = ${REALTIME0052_KLINE_RENDERER_REPLAY}"
echo "[stack] renderer_benchmark = ${REALTIME0052_BENCHMARK_RENDERER}"

echo "[step] start app with legacy stack (legacy_env + sqlite + plotly)"
exec uv run streamlit run app.py
