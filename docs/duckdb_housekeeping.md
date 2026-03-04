# DuckDB Housekeeping 用法

本文件說明 `scripts/duckdb_housekeeping.py` 的用途與參數。  
目標是控制 `market_snapshots` 表成長，並提供 DuckDB 維護入口。

## 1) 這個指令會做什麼

- 清理 `market_snapshots` 舊資料（依 `as_of` 與保留天數刪除）。
- 可對不同 `dataset_key` 設不同保留天數。
- 可選擇執行 DuckDB `CHECKPOINT` 與 `VACUUM`。
- 輸出 JSON 報告（刪除筆數、策略、執行時間）。

## 2) 這個指令不會做什麼

- 不會刪 `daily_bars` / `intraday_ticks` Parquet 檔。
- 不會刪回測結果表（如 `backtest_runs`、`rotation_runs`）。
- 不會改動交易規則或同步邏輯。

## 3) 基本語法

```bash
uv run python scripts/duckdb_housekeeping.py [options]
```

## 4) 參數說明

- `--db-path`
  - 指定 DuckDB 檔案路徑；未指定時使用專案預設路徑判定。
- `--parquet-root`
  - 指定 Parquet 根目錄（只用於回報資訊，不直接清理 Parquet）。
- `--default-keep-days`
  - 未在 `--keep` 指定的快照，統一保留天數；預設 `30`。
- `--keep dataset_key=days`
  - 為特定快照來源覆寫保留天數，可重複傳多次。
  - 範例：`--keep live_quote=5 --keep live_ohlcv=14`
- `--no-purge`
  - 不執行快照刪除，只跑維護動作（搭配 `--checkpoint` / `--vacuum`）。
- `--checkpoint`
  - 執行 DuckDB `CHECKPOINT`。
- `--vacuum`
  - 執行 DuckDB `VACUUM`（會隱含 checkpoint 流程）。

## 5) 常用範例

### A. 用預設策略清理快照（30 天）

```bash
uv run python scripts/duckdb_housekeeping.py
```

### B. 依快照類型分層保留

```bash
uv run python scripts/duckdb_housekeeping.py \
  --keep live_quote=5 \
  --keep live_ohlcv=14 \
  --keep twse_mi_index_allbut0999=60 \
  --default-keep-days 30 \
  --checkpoint
```

### C. 只做資料庫維護，不刪資料

```bash
uv run python scripts/duckdb_housekeeping.py --no-purge --vacuum
```

### D. 對指定資料庫路徑執行

```bash
uv run python scripts/duckdb_housekeeping.py \
  --db-path "/path/to/market_history.duckdb" \
  --parquet-root "/path/to/parquet" \
  --keep live_quote=3 \
  --checkpoint
```

## 6) 輸出結果（JSON）

輸出會包含：

- `db_path` / `parquet_root`
- `snapshot_housekeeping`
  - `removed_total`
  - `removed_by_dataset`
  - `applied_policy`
  - `default_keep_days`
  - `at`
- `duckdb_maintenance`
  - `actions`（例如 `["CHECKPOINT"]` 或 `["CHECKPOINT","VACUUM"]`）
  - `at`
  - `db_path`

## 7) 建議排程

- 交易時段高頻快照（如 `live_quote`）建議每日清一次。
- 低頻快照（如 `twse_mi_index_allbut0999`）可每週清一次。
- 若發現 DuckDB 檔案明顯膨脹，可加跑 `--vacuum`。
