# `export-etf-briefing`

一次輸出 ETF 戰報資料夾到本機目錄，內含：

- 台股 ETF 超級大表 CSV
- 依大表篩出的 `科技型 ETF` / `主動式 ETF` 名單 CSV
- `科技型 combined` 圖
- `主動式 combined` 圖
- 指定單檔 `single` 圖
- 額外 `split` 圖組
- `司馬懿` 風格 HTML 報告
- FB 可直接貼上的文案與預覽 HTML

此命令可在「沒有先執行 `uv run streamlit run app.py`」的情況下獨立使用。

## Help

```bash
uv run realtime0052 export-etf-briefing --help
```

## 常用參數

- `--out-root`
  輸出根目錄，預設 `~/Downloads/etf`
- `--start`
  回看起始日，預設 `2023-01-01`
- `--end`
  回看結束日，預設執行當天
- `--single-symbol`
  單檔觀察區的標的，預設 `00935`
- `--theme`
  圖表主題，支援 `paper-light / soft-gray / data-dark`
- `--news-window-days`
  近期公開資訊檢索視窗，預設 `14`
- `--include-extra-splits/--no-include-extra-splits`
  是否額外輸出 `科技型 / 主動式` 個別拆圖

## 輸出結構

命令會在 `out-root` 下面建立日期資料夾，例如：

```text
~/Downloads/etf/briefing_20260322/
```

裡面會包含：

```text
tw_etf_super_export_20260322.csv
tech_etf_from_super_export_20260322.csv
active_etf_from_super_export_20260322.csv
report_sima_yi.html
fb_post_sima_yi.txt
fb_post_sima_yi.html
manifest.json
news_sources.json
charts/
  tech_etf_combined_20230101_20260322.png
  active_etf_combined_20230101_20260322.png
  00935_single_20230101_20260322.png
  tech_etf_split/*.png
  active_etf_split/*.png
```

## 用法範例

```bash
uv run realtime0052 export-etf-briefing
```

```bash
uv run realtime0052 export-etf-briefing \
  --out-root ~/Downloads/etf \
  --start 2023-01-01 \
  --single-symbol 00935
```

```bash
uv run realtime0052 export-etf-briefing \
  --out-root ~/Downloads/etf \
  --start 2023-01-01 \
  --end 2026-03-22 \
  --single-symbol 00935 \
  --theme soft-gray \
  --news-window-days 14
```

## 規則說明

- `科技型 ETF` 與 `主動式 ETF` 都是直接從當次輸出的超級大表 CSV 反讀後篩選，不是另外一套來源。
- `combined` 圖固定疊 benchmark 虛線。
- `single` 圖用現有 `chart-backtest` 單檔輸出能力。
- HTML 主文不鋪陳來源清單，但 `news_sources.json` 仍保留可追溯的近期公開資訊結果。
- 若某一組 ETF 沒有足夠歷史資料可畫圖，會記錄到 `manifest.json` 的 `issues`。
