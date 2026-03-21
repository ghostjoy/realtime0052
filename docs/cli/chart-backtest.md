# `chart-backtest`

用途：輸出回測圖為 `PNG`，可給人看，也適合 AI workflow。

## Help

```bash
uv run realtime0052 chart-backtest --help
```

## 版型

- `single`：單一標的，`K線 + Equity + Benchmark`
- `combined`：多標的同圖，預設只畫 `Benchmark + 各標的 Buy and Hold`
- `split`：多標的逐檔各輸出一張圖

## 常用參數

- `--symbols 0050,0052`
- `--layout single|combined|split`
- `--market auto|TW|US`
- `--start YYYY-MM-DD`
- `--end YYYY-MM-DD`
- `--strategy ...`
- `--benchmark auto|twii|0050|006208|gspc|spy|qqq|dia`
- `--theme paper-light|soft-gray|data-dark`
- `--out PATH`
- `--out-dir DIR`

## 單標的參考圖標註

- `--reference-annotations`
- `--annotate-extrema`
- `--show-signals`
- `--show-fills`
- `--show-trade-path`
- `--show-end-marker`

## 多標的比較

- `--include-ew-portfolio`：在 `combined` 模式加入 EW portfolio 線
- 預設不加入 EW 線
- Benchmark 虛線固定保留

## 範例

```bash
uv run realtime0052 chart-backtest \
  --symbols 0050 \
  --layout single \
  --reference-annotations \
  --start 2023-01-01 \
  --end 2026-03-21

uv run realtime0052 chart-backtest \
  --symbols 0050,0052,00935,00735 \
  --layout combined \
  --market TW \
  --start 2023-01-01 \
  --end 2026-03-21

uv run realtime0052 chart-backtest \
  --symbols 0050,0052,00935,00735 \
  --layout combined \
  --market TW \
  --include-ew-portfolio \
  --start 2023-01-01 \
  --end 2026-03-21

uv run realtime0052 chart-backtest \
  --symbols 0050,SPY \
  --layout split \
  --market auto \
  --start 2023-01-01 \
  --end 2026-03-21 \
  --out-dir ./artifacts/charts
```

## 輸出

- 顯示 `layout / format / requested / exported`
- 列出每張輸出圖的檔案路徑
