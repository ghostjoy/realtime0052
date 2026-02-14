# PROJECT_CONTEXT.md

此檔案是給 LLM / 新接手工程師的「快速上下文」。  
目標：不用重跑整段對話，也能理解專案目前在做什麼。

## 1) 目前功能地圖

- `即時看盤`：台股/美股即時資訊、技術指標、建議文字
- `回測工作台`：歷史資料同步、本地回測、回放、Benchmark 比較
- `2025 前十大 ETF`：2025 全年區間前十大台股 ETF（卡片頁）
- `2026 YTD 前十大 ETF`：2026 年截至今日前十大台股 ETF（卡片頁）
- `ETF 輪動策略`：台股 6 檔 ETF 月頻輪動（大盤濾網 + 動能排名 + Top N 等權）
- `00935 熱力圖`：00935 成分股逐檔回測，對大盤做超額報酬熱力圖
- `0050 熱力圖`：0050 成分股逐檔回測，對大盤做超額報酬熱力圖
- `資料庫檢視`：直接查看 SQLite 各資料表內容與欄位
- `新手教學`：參數解釋、常見誤區、操作流程

## 2) 核心模組

- `app.py`
  - Streamlit UI 主流程與互動狀態管理
  - 各分頁渲染函式
- `services/market_data_service.py`
  - 多來源市場資料邏輯（含 fallback）
  - 台股即時來源：`Fugle WS -> TW MIS -> TW OpenAPI -> TPEx OpenAPI`
  - 台股日K來源：`Fugle Historical -> TW OpenAPI -> TPEx OpenAPI -> Yahoo`
  - Benchmark / 00935 成分股來源整合
- `storage/history_store.py`
  - SQLite schema 與歷史資料同步
  - 回測結果、成分股快取與台股即時 tick 持久化
- `backtest/*`
  - 回測核心邏輯、績效計算、walk-forward
  - `backtest/rotation.py`：ETF 輪動策略回測核心

## 3) 資料流（高層）

1. UI 設定區間/標的/策略
2. `HistoryStore` 先讀 SQLite，必要時增量同步
3. 回測引擎運算策略曲線
4. Benchmark 對齊同區間後比較
5. 視覺化（K 棒、資產曲線、熱力圖）

## 4) 關鍵資料表（SQLite）

- 預設 DB 路徑：`~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/market_history.sqlite3`（若無 iCloud 則回退 `market_history.sqlite3`；可由 `REALTIME0052_DB_PATH` 覆蓋）
- `instruments`
- `daily_bars`
- `intraday_ticks`（台股即時 tick；保留天數可由 `REALTIME0052_INTRADAY_RETAIN_DAYS` 調整，預設 1095 天）
- `sync_state`
- `backtest_runs`
- `universe_snapshots`（成分股清單快取）
- `heatmap_runs`（熱力圖最近一次結果）
- `rotation_runs`（ETF 輪動最近一次結果）

## 5) 既定行為與假設

- 回測交易規則：`T` 日訊號，`T+1` 開盤成交
- 回放只是視覺播放，不改變回測數值
- 成本模型：手續費 + 稅 + 滑價
- Benchmark 比較使用重疊區間

## 6) 近期重點改動（摘要）

- 新增相對大盤勝負與超額報酬顯示
- 回放預設起點改第 20 根 K，播放時位置同步移動
- 主題精簡為 `日光白 / 灰白專業`，預設灰白
- 新增 ETF 輪動分頁（固定 6 檔 ETF，月調倉）
- 新增 00935 成分股熱力圖分頁
- 成分股清單改為 SQLite 快取，避免反覆抓取
- 回測工作台新增「回測前自動補資料缺口」，可針對缺口標的增量回補
- 新增兩個 ETF 排行卡片頁：`2025 前十大 ETF` 與 `2026 YTD 前十大 ETF`（TWSE 快照區間報酬）

## 7) 常見故障點（已踩過）

- Streamlit `session_state` 在 widget 建立後再寫入同 key 會拋錯
- 外部來源有時只回 `close` 欄，需先標準化 OHLC
- 免費資料源會限流，需有 fallback 與快取
- Fugle WebSocket 需 API key；若未設定會自動退回 TW MIS / OpenAPI
- 新增 `scripts/run_fugle_mcp_server.sh` 與 `FUGLE_MCP_GUIDE.md`，供 Agent 直接連接 Fugle MCP server

## 8) 建議維護流程

1. 修改前先看 `CHANGELOG.md` 與本檔
2. 小步提交，避免一次改太多行為
3. 修改後跑：
   - `uv run python -m compileall app.py`
   - `uv run python -m unittest discover -s tests -v`
4. 更新 `CHANGELOG.md` 的 `Unreleased`
   - 若已啟用 `./scripts/setup_git_hooks.sh`，commit 前會自動補一筆
