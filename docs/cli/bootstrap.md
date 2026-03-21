# `bootstrap`

用途：批次預載本地市場資料與 metadata。

## Help

```bash
uv run realtime0052 bootstrap --help
```

## 常用參數

- `--scope tw|us|both`
- `--years 5`
- `--max-workers 6`
- `--sync-mode min_rows|backfill|all`
- `--min-rows 900`

## 範例

```bash
uv run realtime0052 bootstrap --scope tw --years 3
uv run realtime0052 bootstrap --scope both --years 5 --sync-mode min_rows --min-rows 900
```

## 輸出

- 顯示 `scope / years / metadata_upserted / total_symbols`
- 顯示 `synced_success / failed / issues`
- 最多列出前 10 筆 issues
