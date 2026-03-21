# `sync-twse-etf-daily`

用途：同步 TWSE 官方 ETF 日成交快照。

## Help

```bash
uv run realtime0052 sync-twse-etf-daily --help
```

## 常用參數

- `--start YYYY-MM-DD`
- `--end YYYY-MM-DD`
- `--lookback-days 7`：未指定日期時的回看視窗
- `--force`：強制重抓已覆蓋日期

## 範例

```bash
uv run realtime0052 sync-twse-etf-daily --lookback-days 5
uv run realtime0052 sync-twse-etf-daily --start 2026-03-01 --end 2026-03-20
uv run realtime0052 sync-twse-etf-daily --start 2026-03-01 --end 2026-03-20 --force
```

## 輸出

- 顯示 `start / end / latest`
- 顯示 `requested_days / synced_days / skipped_days / empty_days / saved_rows`
- 最多列出前 10 筆 issues
