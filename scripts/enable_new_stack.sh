#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/enable_new_stack.sh [options]

Options:
  --sqlite-path <path>   Legacy SQLite path (default: $REALTIME0052_DB_PATH or iCloud/codexapp/market_history.sqlite3)
  --duckdb-path <path>   DuckDB path (default: $REALTIME0052_DUCKDB_PATH or iCloud/codexapp/market_history.duckdb)
  --parquet-root <path>  Parquet root (default: $REALTIME0052_PARQUET_ROOT or iCloud/codexapp/parquet)
  --no-migrate           Skip SQLite -> DuckDB migration step
  -h, --help             Show this help
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

default_duckdb_path() {
  if [[ -n "${REALTIME0052_DUCKDB_PATH:-}" ]]; then
    echo "${REALTIME0052_DUCKDB_PATH}"
  elif [[ -d "${ICLOUD_CODEXAPP_DIR}" ]]; then
    echo "${ICLOUD_CODEXAPP_DIR}/market_history.duckdb"
  else
    echo "market_history.duckdb"
  fi
}

default_parquet_root() {
  if [[ -n "${REALTIME0052_PARQUET_ROOT:-}" ]]; then
    echo "${REALTIME0052_PARQUET_ROOT}"
  elif [[ -d "${ICLOUD_CODEXAPP_DIR}" ]]; then
    echo "${ICLOUD_CODEXAPP_DIR}/parquet"
  else
    echo ""
  fi
}

SQLITE_PATH="$(default_sqlite_path)"
DUCKDB_PATH="$(default_duckdb_path)"
PARQUET_ROOT="$(default_parquet_root)"
RUN_MIGRATION="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sqlite-path)
      [[ $# -ge 2 ]] || { echo "[error] --sqlite-path needs a value"; exit 1; }
      SQLITE_PATH="$2"
      shift 2
      ;;
    --duckdb-path)
      [[ $# -ge 2 ]] || { echo "[error] --duckdb-path needs a value"; exit 1; }
      DUCKDB_PATH="$2"
      shift 2
      ;;
    --parquet-root)
      [[ $# -ge 2 ]] || { echo "[error] --parquet-root needs a value"; exit 1; }
      PARQUET_ROOT="$2"
      shift 2
      ;;
    --no-migrate)
      RUN_MIGRATION="false"
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
DUCKDB_PATH="$(expand_home "${DUCKDB_PATH}")"
if [[ -z "${PARQUET_ROOT}" ]]; then
  PARQUET_ROOT="$(dirname "${DUCKDB_PATH}")/parquet"
else
  PARQUET_ROOT="$(expand_home "${PARQUET_ROOT}")"
fi

echo "[info] root         : ${ROOT_DIR}"
echo "[info] sqlite path  : ${SQLITE_PATH}"
echo "[info] duckdb path  : ${DUCKDB_PATH}"
echo "[info] parquet root : ${PARQUET_ROOT}"
echo "[info] migration    : ${RUN_MIGRATION}"
echo "[info] sqlite scope : $(path_scope "${SQLITE_PATH}")"
echo "[info] duckdb scope : $(path_scope "${DUCKDB_PATH}")"
echo "[info] parquet scope: $(path_scope "${PARQUET_ROOT}")"

echo "[step] uv sync"
uv sync

if [[ "${RUN_MIGRATION}" == "true" ]]; then
  if [[ -f "${SQLITE_PATH}" ]]; then
    echo "[step] migrate sqlite -> duckdb"
    uv run python scripts/migrate_sqlite_to_duckdb.py \
      --sqlite-path "${SQLITE_PATH}" \
      --duckdb-path "${DUCKDB_PATH}" \
      --parquet-root "${PARQUET_ROOT}"
  else
    echo "[warn] skip migration: sqlite source not found (${SQLITE_PATH})"
  fi
fi

export REALTIME0052_CONFIG_SOURCE="${REALTIME0052_CONFIG_SOURCE:-hydra}"
export REALTIME0052_STORAGE_BACKEND="${REALTIME0052_STORAGE_BACKEND:-duckdb}"
export REALTIME0052_DUCKDB_PATH="${REALTIME0052_DUCKDB_PATH:-${DUCKDB_PATH}}"
export REALTIME0052_PARQUET_ROOT="${REALTIME0052_PARQUET_ROOT:-${PARQUET_ROOT}}"
export REALTIME0052_DB_PATH="${REALTIME0052_DB_PATH:-${SQLITE_PATH}}"
export REALTIME0052_AUTO_MIGRATE_LEGACY_SQLITE="${REALTIME0052_AUTO_MIGRATE_LEGACY_SQLITE:-false}"
export REALTIME0052_KLINE_RENDERER_LIVE="${REALTIME0052_KLINE_RENDERER_LIVE:-plotly}"
export REALTIME0052_KLINE_RENDERER_REPLAY="${REALTIME0052_KLINE_RENDERER_REPLAY:-plotly}"
export REALTIME0052_BENCHMARK_RENDERER="${REALTIME0052_BENCHMARK_RENDERER:-plotly}"

echo "[stack] config_source = ${REALTIME0052_CONFIG_SOURCE}"
echo "[stack] storage_backend = ${REALTIME0052_STORAGE_BACKEND}"
echo "[stack] db_path = ${REALTIME0052_DUCKDB_PATH}"
echo "[stack] parquet_root = ${REALTIME0052_PARQUET_ROOT}"
echo "[stack] renderer_live = ${REALTIME0052_KLINE_RENDERER_LIVE}"
echo "[stack] renderer_replay = ${REALTIME0052_KLINE_RENDERER_REPLAY}"
echo "[stack] renderer_benchmark = ${REALTIME0052_BENCHMARK_RENDERER}"

echo "[step] start app with new stack (hydra + duckdb/parquet + plotly charts)"
exec uv run streamlit run app.py
