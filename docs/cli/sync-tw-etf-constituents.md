# `sync-tw-etf-constituents`

用途：掃描台股 ETF 成分股，並把快照寫回 DuckDB `market_snapshots`。

這個指令不需要先啟動 `uv run streamlit run app.py`。

## Help

```bash
uv run realtime0052 sync-tw-etf-constituents --help
```

## 常用參數

- `--symbols 0050,0052`：只掃指定 ETF；未指定時掃全市場台股 ETF
- `--force`：強制遠端刷新後再比較
- `--max-workers 4`
- `--full-refresh`
- `--incremental`
- `--log-dir ./logs/`

## 範例

```bash
uv run realtime0052 sync-tw-etf-constituents
uv run realtime0052 sync-tw-etf-constituents --symbols 0050,0052 --full-refresh
uv run realtime0052 sync-tw-etf-constituents --symbols 00935 --log-dir ./artifacts/logs
```

## 輸出

- stdout 會顯示：
  - `etf_count / updated / unchanged / missing / errors`
  - JSON / Markdown log 路徑
- log 會記錄每檔 ETF 的：
  - `status`
  - `used_trade_date`
  - `row_count_before / row_count_after`
  - `source / notes / error`
