# 台股 ETF 單檔報表包

這份文件說明兩個新 CLI：

- `sync-tw-etf-constituents`
- `export-tw-etf-report`

兩者都不需要先執行 `uv run streamlit run app.py`。

## 1. 成分股掃描

```bash
uv run realtime0052 sync-tw-etf-constituents --help
```

預設會掃描全市場台股 ETF，把成分股快照寫進 DuckDB `market_snapshots`。

資料策略：

- 優先抓完整成分股 rows
- 若抓不到完整欄位，回退到簡版 constituent symbols
- 歷史快照保留在 `market_snapshots`
- 若最新內容與上一版完全相同，會記成 `unchanged`

## 2. 單檔報表包

```bash
uv run realtime0052 export-tw-etf-report --symbol 0052 --out ./reports/
```

報表包會輸出成資料夾，檔名全部帶 ETF 代碼，方便多檔混放時一眼辨識。

範例：

```text
tw_etf_report_0052_20260321/
  0052_summary.md
  0052_overview.csv
  0052_daily_market.csv
  0052_margin.csv
  0052_mis.csv
  0052_three_investors.csv
  0052_aum_track.csv
  0052_constituents.csv
  0052_indicators_snapshot.csv
  0052_indicators_timeseries.csv
  0052_indicators_summary.md
  0052_backtest.png
  0052_aum_track.png
  0052_constituent_heatmap.png
  0052_issues.csv
  0052_sync_log.json
  0052_sync_log.md
```

## 3. 回測區間規則

- `--backtest-start / --backtest-end` 會同時影響：
  - `0052_backtest.png`
  - `0052_indicators_timeseries.csv`
  - `0052_indicators_snapshot.csv`
  - `0052_indicators_summary.md`
  - `0052_constituent_heatmap.png`
- 若未指定，會使用可得全區間

## 4. 技術指標

目前報表包固定包含：

- `SMA 5 / 20 / 60`
- `EMA 12 / 26`
- `MACD / Signal / Hist`
- `RSI(14)`
- `Bollinger Bands`
- `VWAP`
- `ATR(14)`
- `OBV`
- `Stoch K / D`
- `MFI(14)`

輸出形式：

- `*_indicators_snapshot.csv`：最後一天快照
- `*_indicators_timeseries.csv`：完整時間序列
- `*_indicators_summary.md`：可讀摘要

## 5. Log 狀態

報表包與 constituent sync 都會產生 JSON + Markdown log。

常見狀態：

- `updated`：本次有新資料或新快照
- `unchanged`：內容和上一版相同
- `fallback`：回退到較舊但可用的日期
- `delayed`：資料存在，但日期落後請求日期
- `missing`：來源無資料
- `error`：同步或解析失敗

Log 會記錄：

- `dataset_name`
- `status`
- `requested_trade_date`
- `used_trade_date`
- `row_count_before`
- `row_count_after`
- `updated_rows`
- `source`
- `notes`
- `error`
