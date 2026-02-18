# AGENTS.md

本檔案提供給任何 LLM / coding agent 的快速執行規範。  
目標：讓代理在新環境下能快速理解此 repo 並安全地做修改。

## 1) 專案定位

- 專案名稱：`realtime0052`
- 主要技術：`Python + Streamlit + Plotly + SQLite`
- 核心用途：
  - 台股/美股即時與近即時資訊
  - 歷史資料同步與本地儲存
  - 策略回測、Benchmark 比較與視覺化回放
  - ETF 輪動策略回測（固定 ETF 池、月頻調倉）
  - ETF 專題頁（2025 Top10 / 2026 YTD Top10 / 2026 YTD 主動式 ETF）
  - 成分股熱力圖回測（`00910 / 00935 / 00993A / 0050 / 0052`）
  - SQLite 資料庫檢視與新手教學導引

## 2) 先讀哪些檔案

代理進入 repo 後，建議依序閱讀：

1. `README.md`（功能總覽與操作方式）
2. `PROJECT_CONTEXT.md`（架構與資料流）
3. `CHANGELOG.md`（最近改動）
4. `app.py`（UI 與主要流程）

## 3) 開發與驗證

常用指令：

```bash
uv run streamlit run app.py
uv run python -m unittest discover -s tests -v
uv run python -m compileall app.py
```

## 4) 資料與快取原則

- 歷史 OHLCV：預設儲存在 iCloud `~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/market_history.sqlite3`（可由 `REALTIME0052_DB_PATH` 覆蓋）
- 回測摘要與回放快取：`backtest_runs`、`backtest_replay_runs`
- 成分股快取：`universe_snapshots`（`00910 / 00935 / 00993A / 0050 / 0052`）
- 專題結果快取：`heatmap_runs`、`rotation_runs`、`bootstrap_runs`
- 原則：優先重用 SQLite；只做必要增量同步，避免重複打外部來源

## 5) 修改時注意事項

- 優先做「最小範圍、可驗證」的修改
- 維持現有中文介面文案風格
- 回測/圖表功能修改後，至少執行：
  - `compileall`
  - 單元測試
- 避免破壞既有 `session_state` key（Streamlit 常見錯誤來源）

## 6) UI 與主題規則（目前）

- 主題目前提供三套：
  - `日光白（Paper Light）`
  - `灰白專業（Soft Gray）`（預設）
  - `深色專業（Data Dark）`
- K 棒顏色採柔和低飽和、半透明填色

## 7) 不要做的事

- 不要清空或覆蓋使用者本地 SQLite 資料
- 不要引入需付費或需金鑰才能基本運作的強依賴
- 不要在未說明的情況下改動回測交易規則（T+1 成交）

## 8) 交付建議

若代理完成修改，請同步：

1. 更新 `CHANGELOG.md`（本 repo 已有 `pre-commit` 自動補記；若未啟用請先執行 `./scripts/setup_git_hooks.sh`）
2. 必要時更新 `README.md` 的功能描述
3. 回報測試結果與主要影響檔案
