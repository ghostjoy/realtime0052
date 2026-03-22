# `export-tw-etf-report`

用途：輸出台股單一 ETF 的資料夾報表包。

這個指令不需要先啟動 `uv run streamlit run app.py`。

## Help

```bash
uv run realtime0052 export-tw-etf-report --help
```

## 常用參數

- `--symbol 0052`：必填，單一台股 ETF 代碼
- `--out ./reports/`：輸出根目錄
- `--sync-constituents / --no-sync-constituents`
- `--ytd-start YYYYMMDD`
- `--ytd-end YYYYMMDD`
- `--compare-start YYYYMMDD`
- `--compare-end YYYYMMDD`
- `--backtest-start YYYY-MM-DD`
- `--backtest-end YYYY-MM-DD`
- `--daily-lookback-days 14`
- `--force`
- `--log-dir ./logs/`
- `--heatmap-only`：只輸出成分股熱力圖 PNG

## 範例

```bash
uv run realtime0052 export-tw-etf-report --symbol 0052 --out ./reports/
uv run realtime0052 export-tw-etf-report --symbol 0052 --backtest-start 2023-01-01 --backtest-end 2026-03-21
uv run realtime0052 export-tw-etf-report --symbol 0052 --no-sync-constituents --out ./reports/
uv run realtime0052 export-tw-etf-report --symbol 0052 --heatmap-only --out ./0052_constituent_heatmap.png
```

## 報表包內容

目錄名稱會像：

```text
tw_etf_report_0052_20260321/
```

檔名都會帶 ETF 代碼，例如：

- `0052_summary.md`
- `0052_overview.csv`
- `0052_daily_market.csv`
- `0052_margin.csv`
- `0052_mis.csv`
- `0052_three_investors.csv`
- `0052_aum_track.csv`
- `0052_constituents.csv`
- `0052_indicators_snapshot.csv`
- `0052_indicators_timeseries.csv`
- `0052_indicators_summary.md`
- `0052_backtest.png`
- `0052_aum_track.png`
- `0052_constituent_heatmap.png`
- `0052_issues.csv`
- `0052_sync_log.json`
- `0052_sync_log.md`

## 說明

- 若未指定 `--backtest-start / --backtest-end`，回測、圖表與技術指標會使用可得全區間
- 預設會先執行一次台股 ETF 成分股掃描
- `--heatmap-only` 模式只輸出成分股熱力圖 PNG；`--out` 可直接指定 `.png` 檔名，或指定目錄後自動命名
- sync log 會記錄：
  - 哪些資料表有更新
  - 哪些資料表沒有更新
  - 各表實際使用日期
  - `AUM` 是否落後其他表
