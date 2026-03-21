# `sync`

用途：同步單一或多個標的的歷史日線資料。

## Help

```bash
uv run realtime0052 sync --help
```

## 常用參數

- `--market TW|US`：指定市場
- `--symbols 0050,0052`：逗號分隔標的
- `--days 60`：回補最近幾天
- `--mode backfill|min_rows|all`：同步模式
- `--min-rows 250`：`min_rows` 模式的最少列數
- `--max-workers 6`：平行 worker 數

## 範例

```bash
uv run realtime0052 sync --market TW --symbols 0050,0052 --days 60
uv run realtime0052 sync --market US --symbols SPY,QQQ --days 180
uv run realtime0052 sync --market TW --symbols 0050,00935 --mode min_rows --min-rows 250
```

## 輸出

- 列出 `market` 與 `symbols`
- 顯示 `synced / skipped / issues`
- 每個標的會顯示 `rows`、`source`、錯誤摘要
