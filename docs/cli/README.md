# CLI Docs

本目錄整理 `realtime0052` 每個 CLI 指令的獨立用法說明。

除 `serve` 以外，其他 CLI 都可以在「沒有先執行 `uv run streamlit run app.py`」的情況下獨立使用。

可先用總覽 help：

```bash
uv run realtime0052 --help
```

各指令文件：

- [`sync`](./sync.md)
- [`sync-twse-etf-daily`](./sync-twse-etf-daily.md)
- [`sync-twse-etf-mis`](./sync-twse-etf-mis.md)
- [`sync-tw-etf-constituents`](./sync-tw-etf-constituents.md)
- [`export-tw-etf-super-table`](./export-tw-etf-super-table.md)
- [`export-tw-etf-report`](./export-tw-etf-report.md)
- [`export-etf-briefing`](./export-etf-briefing.md)
- [`backtest`](./backtest.md)
- [`chart-backtest`](./chart-backtest.md)
- [`bootstrap`](./bootstrap.md)
- [`info`](./info.md)
- [`serve`](./serve.md)
