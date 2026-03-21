# `export-tw-etf-super-table`

用途：同步 ETF 總表相關來源，並輸出超級大表 CSV。

## Help

```bash
uv run realtime0052 export-tw-etf-super-table --help
```

## 常用參數

- `--out PATH`：輸出檔案或目錄
- `--ytd-start YYYYMMDD`
- `--ytd-end YYYYMMDD`
- `--compare-start YYYYMMDD`
- `--compare-end YYYYMMDD`
- `--daily-lookback-days 14`
- `--force`

## 範例

```bash
uv run realtime0052 export-tw-etf-super-table --out ./tw_etf_super_export_latest.csv
uv run realtime0052 export-tw-etf-super-table --out ./exports/
uv run realtime0052 export-tw-etf-super-table --ytd-start 20260101 --ytd-end 20260321 --force
```

## 輸出

- 顯示 `ytd / compare / trade_date`
- 顯示 `path / rows / cols / run_id`
- 顯示各來源刷新摘要
- 最多列出前 10 筆 issues
