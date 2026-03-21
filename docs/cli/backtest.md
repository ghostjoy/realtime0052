# `backtest`

用途：執行單一標的或多標的回測，輸出核心績效數字。

## Help

```bash
uv run realtime0052 backtest --help
```

## 常用參數

- `--symbol 0050` 或 `--symbol 0050,0052`
- `--market auto|TW|US`
- `--start YYYY-MM-DD`
- `--end YYYY-MM-DD`
- `--strategy buy_hold|sma_trend_filter|donchian_breakout|sma_cross`
- `--initial-capital 1000000`
- `--fee-rate` / `--sell-tax` / `--slippage`
- `--no-split-adjustment`
- `--no-total-return-adjustment`

## 範例

```bash
uv run realtime0052 backtest --symbol 0050 --market TW --strategy buy_hold
uv run realtime0052 backtest --symbol SPY --market US --start 2023-01-01 --end 2026-03-21
uv run realtime0052 backtest --symbol 0050,0052 --market TW --strategy sma_cross
```

## 輸出

- `mode`
- `total_return`
- `cagr`
- `mdd`
- `sharpe`
- `trades`
