# PROJECT_CONTEXT.md

此檔案是給 LLM / 新接手工程師的「快速上下文」。  
目標：不用重跑整段對話，也能理解專案目前在做什麼。

## 1) 目前功能地圖

- `即時看盤`：台股/美股即時資訊、技術指標、建議文字
- `回測工作台`：歷史資料同步、本地回測、回放、Benchmark 比較、DCA（期初+每月定投）績效比較
- `2026 YTD 前十大股利型、配息型 ETF`：2026 年截至今日前十大台股股利/配息型 ETF（卡片頁）
- `2026 YTD 前十大 ETF`：2026 年截至今日前十大台股 ETF（卡片頁）
- `共識代表 ETF`：以前10 ETF 成分股交集推導最具代表性的單一 ETF（附前3備選）
- `兩檔 ETF 推薦`：以共識代表（核心）+ 低重疊動能（衛星）輸出兩檔可執行組合
- `2026 YTD 主動式 ETF`：台股主動式 ETF 的 2026 YTD Buy & Hold 績效卡，含 Benchmark 對照圖（策略/基準/各ETF買進持有）
- `ETF 輪動策略`：台股 6 檔 ETF 月頻輪動（大盤濾網 + 動能排名 + Top N 等權）
- `00910 熱力圖`：新增全球成分股 YTD（Buy & Hold）分組熱力圖（國內/海外分開對齊基準），並保留台股子集合進階回測
- `00935 熱力圖`：00935 成分股逐檔回測，對大盤做超額報酬熱力圖
- `00993A 熱力圖`：00993A 成分股逐檔回測，對大盤做超額報酬熱力圖
- `0050 熱力圖`：0050 成分股逐檔回測，對大盤做超額報酬熱力圖
- `0052 熱力圖`：0052 成分股逐檔回測，對大盤做超額報酬熱力圖
- `00910 / 00935 / 00993A / 0050 / 0052`：頁首統一顯示官方編製/管理規則摘要（附來源連結）
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
  - Benchmark / ETF 成分股來源整合（`00910 / 00935 / 00993A / 0050 / 0052`）
- `services/benchmark_loader.py`
  - 台股 Benchmark 候選鏈、同步、split 調整與載入結果格式統一（熱力圖/輪動/比較卡共用）
- `services/backtest_runner.py`
  - 回測工作台資料準備（symbol bars/調整）與執行流程封裝
- `services/heatmap_runner.py`
  - 熱力圖回測前同步/資料準備與批次回測流程封裝
- `services/rotation_runner.py`
  - ETF 輪動頁資料準備、持有排名計算與 payload 組裝封裝
- `services/backtest_cache.py`
  - 回測 `run_key` 與回放簽章（`schema_version/source_hash`）共用產生器
- `services/sync_orchestrator.py`
  - 標的同步流程共用化（全量/缺口/最少K數）與錯誤收斂
- `services/bootstrap_loader.py`
  - 市場基礎資料預載與每日增量更新流程
  - 台股代碼 metadata 擷取（TWSE/TPEx）
- `state_keys.py`
  - 回測工作台 `session_state/widget key` 集中管理，降低 key 漂移與衝突
- `ui/shared/perf.py`
  - 頁面級分段耗時工具（`REALTIME0052_PERF_DEBUG=1`）
- `ui/shared/session_utils.py`
  - `session_state` 預設值初始化 helper
- `storage/history_store.py`
  - SQLite fallback store（回滾/相容用途）
- `storage/duck_store.py`
  - DuckDB + Parquet hybrid（目前預設）
  - DuckDB 儲存 metadata/回測快取；Parquet 儲存 `daily_bars`/`intraday_ticks`
- `backtest/*`
  - 回測核心邏輯、績效計算、walk-forward
  - `backtest/rotation.py`：ETF 輪動策略回測核心

## 3) 資料流（高層）

1. UI 設定區間/標的/策略
2. `DuckHistoryStore` 先讀 DuckDB/Parquet，必要時增量同步
3. 回測引擎運算策略曲線
4. Benchmark 對齊同區間後比較
5. 視覺化（K 棒、資產曲線、熱力圖）

## 4) 關鍵資料表（DuckDB + Parquet）

- 預設 DB 路徑：`~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/market_history.duckdb`（若無 iCloud 則回退 `market_history.duckdb`；可由 `REALTIME0052_DUCKDB_PATH` 覆蓋）
- 預設 Parquet 路徑：`~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/parquet`（可由 `REALTIME0052_PARQUET_ROOT` 覆蓋）
- `instruments`
- `sync_state`
- `symbol_metadata`（代碼名稱/交易所/產業 metadata）
- `backtest_runs`
- `backtest_replay_runs`（回測工作台完整回放快取）
- `universe_snapshots`（成分股清單快取）
- `heatmap_runs`（熱力圖最近一次結果）
- `rotation_runs`（ETF 輪動最近一次結果）
- `bootstrap_runs`（預載與增量更新任務紀錄）
- `daily_bars` / `intraday_ticks`：Parquet 分區資料（market/symbol）

## 5) 既定行為與假設

- 回測交易規則：`T` 日訊號，`T+1` 開盤成交
- 回放只是視覺播放，不改變回測數值
- 成本模型：手續費 + 稅 + 滑價
- 可選 `還原權息（Adj Close）`：有 `adj_close` 時優先採用，並避免與分割調整重複
- Benchmark 比較使用重疊區間

## 6) 近期重點改動（摘要）

- 新增相對大盤勝負與超額報酬顯示
- 回測/熱力圖/輪動三條路徑新增 runner 模組，將資料準備與執行邏輯從 `app.py` 抽離
- 回測/熱力圖/輪動頁新增可選效能分段資訊（`REALTIME0052_PERF_DEBUG=1`）
- 回測工作台改為輸入完成後自動回測；同條件優先讀取本地 DuckDB/Parquet 快取
- 回測回放快取新增 `schema_version/source_hash` 簽章，舊快取會自動重算避免相容性錯誤
- 回放預設改為完整區間且定位最後一根，`Reset` 可重播
- 前十大/主動式 ETF 卡片新增資料健康度欄位（`as_of/source/chain/fallback`）
- 新增 `共識代表 ETF` 分頁：以前10 ETF 成分股交集與權重覆蓋率，輸出單一代表 ETF 與備選
- 新增 `兩檔 ETF 推薦` 分頁：以共識代表 + 低重疊動能規則輸出核心/衛星兩檔，含海外限制與重疊門檻回退
- 主題提供 `日光白 / 灰白專業 / 深色專業`，預設灰白
- 新增 ETF 輪動分頁（固定 6 檔 ETF，月調倉）
- 新增/擴充 ETF 熱力圖分頁：`00910`（含全球成分股 YTD 分組）、`00935`、`00993A`、`0050`、`0052`
- 成分股清單改為本地 DuckDB 快取，避免反覆抓取
- 回測工作台新增「回測前自動補資料缺口」，可針對缺口標的增量回補
- 新增兩個 ETF 排行卡片頁：`2026 YTD 前十大股利型、配息型 ETF` 與 `2026 YTD 前十大 ETF`（TWSE 快照區間報酬）
- 新增 `資料庫檢視` 分頁可執行基礎資料預載與每日增量更新

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
