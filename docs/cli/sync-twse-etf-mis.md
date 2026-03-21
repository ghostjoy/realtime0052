# `sync-twse-etf-mis`

用途：同步 TWSE MIS ETF 指標快照。

## Help

```bash
uv run realtime0052 sync-twse-etf-mis --help
```

## 常用參數

- `--start YYYY-MM-DD`
- `--end YYYY-MM-DD`
- `--force`：強制重抓最新覆蓋日

## 範例

```bash
uv run realtime0052 sync-twse-etf-mis
uv run realtime0052 sync-twse-etf-mis --start 2026-03-01 --end 2026-03-20
uv run realtime0052 sync-twse-etf-mis --end 2026-03-20 --force
```

## 輸出

- 顯示 `start / end / latest`
- 顯示 `requested_days / synced_days / skipped_days / empty_days / saved_rows`
- 最多列出前 10 筆 issues
