# CHANGELOG

這份檔案用來記錄專案的重要更新（功能、修正、文件）。

格式說明：
- `Added`：新功能
- `Changed`：行為或介面調整
- `Fixed`：錯誤修正
- `Docs`：文件更新

## [Unreleased]

### Added
- 新增 `symbol_metadata`（SQLite）：
  - 儲存 `symbol / market / name / exchange / industry / currency / source / updated_at`
  - 供公司名稱/產業查詢優先走本地資料，降低重複打外部 API
- 新增 `bootstrap_runs`（SQLite）：
  - 記錄資料預載與每日增量更新任務（範圍、狀態、成功/失敗數、摘要與錯誤）
- 新增 `services/bootstrap_loader.py`：
  - 提供台股 metadata 擷取、全量預載、每日增量更新共用流程
- 新增 `scripts/bootstrap_market_data.py`：
  - 可從 CLI 一次預建台股/美股核心資料（含 metadata + 歷史日K）
- 新增 `00935 熱力圖` 分頁：可批次比較 00935 成分股相對大盤的超額報酬，並用紅綠熱力圖顯示強弱。
- 新增 `0050 熱力圖` 分頁：可批次比較 0050 成分股相對大盤的超額報酬，並用紅綠熱力圖顯示強弱。
- 新增 `ETF輪動` 分頁：固定 `0050/0052/00935/0056/00878/00919` 的日K月頻輪動回測。
- 新增 `資料庫檢視` 分頁：可直接查看 `market_history.sqlite3` 的資料表總覽、欄位結構與分頁資料內容。
- 新增輪動引擎 `backtest/rotation.py`：
  - 大盤濾網（Benchmark > 60MA）
  - 動能加權分數（20/60/120）
  - 每月第一交易日評分、下一交易日開盤調倉
  - Top N 等權持有
- 新增 `rotation_runs`（SQLite）快取輪動回測結果，分頁載入可直接顯示上次結果。
- 新增 `universe_snapshots`（SQLite）用於快取成分股清單，避免每次切分頁都重抓。
- 新增成分股快取讀寫 API：
  - `save_universe_snapshot(...)`
  - `load_universe_snapshot(...)`
- 新增 repo 初始化腳本：`scripts/bootstrap_project_docs.py`，可一鍵建立 LLM 文件骨架。
- 新增 Prompt 範本文件：`PROMPT_TEMPLATES.md`（由初始化腳本建立）。
- 新增 `pre-commit` hook（`.githooks/pre-commit`）與 `scripts/auto_changelog.py`，可在 commit 前自動補 `CHANGELOG.md`。
- 新增 `scripts/setup_git_hooks.sh`，可一鍵設定 `core.hooksPath=.githooks`。
- 新增 Fugle MCP 啟動腳本：`scripts/run_fugle_mcp_server.sh`（支援 `FUGLE_MARKETDATA_API_KEY -> API_KEY` 映射）。
- 新增 `intraday_ticks`（SQLite）即時資料表與讀寫 API：
  - `save_intraday_ticks(...)`
  - `load_intraday_ticks(...)`
  - 支援 `REALTIME0052_INTRADAY_RETAIN_DAYS`（預設 1095 天）控管保留天數。
- Fugle API key 讀取新增 key file fallback：
  - 預設讀取 `~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/fuglekey`
  - 可用 `FUGLE_MARKETDATA_API_KEY_FILE` / `FUGLE_API_KEY_FILE` 覆蓋。
- 新增台股歷史 provider：`tw_fugle_rest`（Fugle Historical Candles 日K），支援長區間分段拉取後合併。
- 回測工作台新增 `回測前自動補資料缺口（推薦）`：
  - 先檢查本地區間覆蓋度
  - 僅對缺口標的做增量同步，降低「回測天數不夠」機率。
- 新增 `2025 前十大 ETF` 與 `2026 YTD 前十大 ETF` 兩個獨立卡片頁：
  - 以 TWSE `MI_INDEX` 全市場快照計算區間報酬率前十名
  - 顯示市值型/股利型/其他分類統計與期初期末收盤價
- 新增 `2026 YTD 主動式 ETF` 獨立卡片頁：
  - 以「代碼結尾 A + 名稱含主動」規則篩出台股主動式 ETF，顯示 2026 YTD Buy & Hold（復權版）績效排序
  - 新增 `Benchmark 對照卡`：同圖比較策略曲線（主動式ETF等權）、基準曲線與各ETF買進持有曲線
  - 每檔 ETF 曲線使用高對比、不相近配色，提升多檔比較辨識度
- 新增 `00910 熱力圖` 獨立卡片頁：
  - 可列出全部成分股清單
  - 可執行與 `00935` 同級的成分股熱力圖回測
  - 新增 `全球成分股 YTD（Buy & Hold）` 分組熱力圖：台股對台股基準、海外對海外基準，並以共同交易日對齊計算超額報酬
  - 新增 `00910 / 00935 / 0050` 頁首「官方指數編製標準摘要」與來源連結
- 新增 `00993A 熱力圖` 獨立卡片頁：
  - 可載入 `00993A` 成分股清單並執行相對大盤熱力圖回測
  - 沿用既有台股熱力圖參數（Benchmark / 策略 / 成本 / 同步選項）與結果呈現流程
- 新增 `backtest_replay_runs`（SQLite）：
  - 回測工作台可持久化完整回放資料（含曲線/訊號/交易）
  - 同條件可直接讀取快取，減少重複回測等待時間
- 新增熱力圖頁下方「成分股公司簡介」區塊：
  - `0050 / 00935`：顯示 `排名 / 代號 / 名稱 / 權重 / 產業 / 核心業務 / 產品面向`
  - `00910`：顯示全球成分股 `排名 / 市場 / 權重 / 公司簡介`
- 新增 ETF 成分股完整欄位擷取與解析（`get_etf_constituents_full`）：
  - 可回傳 `rank / symbol / name / market / weight_pct / shares / tw_code`
  - 可識別海外市場代碼（如 `.US/.JP/.KS`），供 00910 全球分組熱力圖與公司簡介使用

### Changed
- Auto: updated AGENTS.md, GIT_GITHUB_GUIDE.md, PROJECT_CONTEXT.md, PROMPT_TEMPLATES.md, README.md [id:c20efff0af]
#### Manual Summary
- `MarketDataService.get_tw_symbol_names/get_tw_symbol_industries` 改為「SQLite 優先、API 補抓、結果回寫 SQLite」，降低重複網路請求。
- `資料庫檢視` 分頁新增「市場基礎資料預載」操作區，並支援顯示最近任務摘要/錯誤。
- App 啟動後新增「每日一次」自動增量更新（有 seed symbols 時才執行），任務摘要回寫 `bootstrap_runs`。
- 新增共用資料健康度顯示（`as_of/source/source chain/degraded/fallback_depth`）到即時看盤、ETF 排行與 Benchmark 對照卡。
- 新增錯誤訊息分級 helper（error/warning/info），統一 ETF 排行與 Benchmark 卡片訊息語意。
- 回測回放快取新增 `schema_version + source_hash` 相容性檢查，不相容時自動重算。
- 抽出共用 helper：快照健康度、同步錯誤摘要、回測快取載入驗證流程。
- Benchmark 載入層重構並抽出 `services/benchmark_loader.py`，熱力圖/輪動/Benchmark 比較卡共用同一流程。
- 回測快取 key/簽章標準化：新增 `services/backtest_cache.py` 共用 `run_key` 與簽章組成。
- 回測工作台 state key 集中到 `state_keys.py`，降低 widget/session key 衝突。
- 回測回放 payload 升級為 `v3`（`meta/bars/results`），快取 schema 升級為 `3`，並維持舊版相容載入。
- 新增 `services/sync_orchestrator.py`，統一 `all/backfill/min_rows` 同步流程。
- 新增 `深色專業（Data Dark）` 主題，並統一 Benchmark 線條視覺樣式（主題色 + 虛線）。
- `回測工作台` 改為「輸入條件即自動回測」：移除手動執行按鈕、先讀 SQLite 快取、回放預設停在最後一根。
- `回測工作台` 新增 `還原權息計算（Adj Close）`，並避免與分割調整重複套用。
- 回測工作台「策略 vs 買進持有」比較表整併，新增 `比較區間` 欄位避免重複資訊。
- 熱力圖回測預設起始區間由 3 年調整為 5 年。
- `2026 YTD 主動式 ETF` 與 `2026 YTD 前十大 ETF` 強化比較能力：新增 `2025績效`、`贏輸台股大盤`、`台股大盤固定列`、`更新最新市況` 與 Benchmark 對照圖補強。
- 多標的比較圖的 Benchmark hover 顯示統一為代碼（例如 `00935`）。
- `0050` 公司簡介表改為依 `權重(%)` 排序；`0050/00935/00910` 熱力圖預設策略改為 `buy_hold`。
- `sync_symbol_history(...)` 補強增量判斷：若僅最新日尚未出新 K，改回報 `stale` 並不中斷流程。
- 台股資料鏈路升級：即時 `Fugle WS -> TW MIS -> TW OpenAPI -> TPEx OpenAPI`；日K同步新增 `TPEx OpenAPI` 後再回退 Yahoo。
- UI 導覽改為卡片式並改為單頁渲染，減少切頁等待；並新增設計協作 `design-tokens.json` 下載。
- 熱力圖/ETF 輪動新增「執行前同步最新日K」開關與多標的平行同步選項；回測頁啟動自動增量同步預設改為關閉。
- SQLite 預設路徑改為「優先 iCloud Drive、否則本地」，並支援 `REALTIME0052_DB_PATH` 覆蓋。
- `sync_symbol_history(...)` 起始日期邏輯補強：可正確回補更早區間資料，已覆蓋區間則維持增量同步。

#### Auto Tracked Files
- Auto: updated PROJECT_CONTEXT.md, README.md, app.py, providers/tw_fugle_rest.py, services/market_data_service.py, tests/test_active_etf_page.py, ... (+1) [id:695a12c9e1]
- Auto: updated PROJECT_CONTEXT.md, README.md, app.py, market_data_types.py, scripts/bootstrap_market_data.py, services/__init__.py, ... (+17) [id:54e33a04fe]
- Auto: updated .gitignore, PROJECT_CONTEXT.md, README.md, app.py, market_data_types.py, tests/test_active_etf_page.py [id:7909c33419]
- Auto: updated AGENTS.md, PROJECT_CONTEXT.md, README.md, app.py, storage/history_store.py, tests/test_history_store.py, ... (+1) [id:c89bc1cf9d]
- Auto: updated README.md, app.py, tests/test_active_etf_page.py [id:4d84559a83]
- Auto: updated PROJECT_CONTEXT.md, README.md, app.py, tests/test_active_etf_page.py [id:e2689afe76]
- Auto: updated app.py [id:13cce7fd07]
- Auto: updated AGENTS.md, PROJECT_CONTEXT.md, README.md, app.py, storage/history_store.py [id:8cdb1d9956]
- Auto: updated PROJECT_CONTEXT.md, README.md, app.py [id:2531f53664]
- Auto: updated README.md, app.py, services/market_data_service.py, tests/test_market_data_service.py [id:4b701c5149]
- Auto: updated PROJECT_CONTEXT.md, README.md, app.py, backtest/__init__.py, backtest/rotation.py, backtest/types.py, ... (+3) [id:3bd6d76b10]
- Auto: updated .githooks/pre-commit, AGENTS.md, PROJECT_CONTEXT.md, README.md, app.py, backtest/__init__.py, ... (+12) [id:d87b9ff71f]

### Fixed
- 修正 Fugle WebSocket 連線細節：改用官方 `.../stock/streaming` endpoint、`auth.data.apikey` 欄位，並修正微秒時間戳解析，避免 `year out of range` 導致即時報價失敗。
- 修正台股即時走勢來源覆蓋問題：當 Yahoo 1m 回傳空資料時，不再覆蓋 Fugle/MIS 即時 tick 聚合出的 K 線。
- 修正即時看盤名稱欄位空白問題：名稱缺值時改為 `name -> full_name -> 台股代號查名 -> symbol` 多層 fallback。
- 修正台股名稱對照缺漏：`get_tw_symbol_names` 新增 TPEx 上櫃名稱來源補齊（例如 `6510 -> 精測`）。
- 修正前十大 ETF 頁面篩選過嚴導致 `0052` 未納入問題：排除條件改為槓反/期貨/海外與債券商品，保留台股股票型 ETF。
- 修正 ETF 成分股 fallback 邏輯：未知代碼不再誤用 `00935` 成分股作為預設。
- 修正台股即時走勢偶發空白：日級報價來源（TW OpenAPI/TPEx）不再使用舊交易日時間戳作為即時 tick，並新增 SQLite 即時快取回補走勢。
- 改善台股即時走勢可讀性：當 tick 聚合 K 線過少（例如僅 1 根）時，自動改用 Fugle Historical `1m` 補齊圖表。
- 改善台股即時走勢補齊策略：不只空資料，當 K 數偏少時也會嘗試用本地 SQLite 快取補齊（僅在 K 數變多時套用）。
- 改善台股即時走勢體感：新增 `last-good 1m` 快取回退（Yahoo/Fugle 1m 暫時失敗時沿用最近成功資料）。
- 改善台股即時走勢穩定度：當 1m 即時資料仍不足時，自動改用 `日K尾段(260)` 顯示，避免僅剩 1 根 K。
- 修正台股即時走勢分割斷層：即時圖也套用已知分割調整（例如 `0052`），避免未復權造成價格斷崖。
- 即時走勢分割邏輯與回測對齊：改為 `known + auto-detect`，不侷限單一個股。
- 修正即時看盤漲跌/漲跌幅顯示：當來源缺 `prev_close` 時，改由日K/即時K自動回推計算，台股與美股皆適用。
- 修正即時看盤漲跌 fallback 的前收基準：改為依 quote 時間挑選正確前收（日K/分K），避免固定取 `[-2]` 造成偏移。
- 即時總覽新增 `漲跌計算依據` 顯示，便於判讀目前採用的來源與前收來源。
- 調整 `股癌風格(心法/檢核)` 建議輸出：新增「倉位建議區間」與中性市況小倉試單節奏，降低幾乎全程空手的體感。
- 修正台股日K走 Yahoo fallback 時未正規化代碼問題：4~6 碼代號自動改為 `.TW`（如 `0050 -> 0050.TW`），指數代碼（如 `^TWII`）維持原樣。
- 熱力圖與 ETF 輪動分頁新增同步錯誤可見提示：若部分標的或 Benchmark 同步失敗，UI 會顯示摘要警示且仍盡量使用本地可用資料。

### Docs
- 文件一致性整理：
  - `README.md` 修正 repo 路徑 `realtime_0052 -> realtime0052`
  - `AGENTS.md`、`PROJECT_CONTEXT.md` 同步完整分頁與功能範圍描述
  - `CHANGELOG.md` 的 `Unreleased/Changed` 改為 `Manual Summary + Auto Tracked Files`
  - `GIT_GITHUB_GUIDE.md` 改為可重複使用的通用模板
  - `PROMPT_TEMPLATES.md` 改為中文主體（附英文關鍵詞）
- 補齊 App 內 `新手教學`：新增完整分頁地圖、快取/更新按鈕邏輯、Top10/主動式/熱力圖/輪動判讀、常見誤解與建議上手順序。
- `README.md` 新增 `新手快速上手（10 分鐘）`，提供最短操作路徑與各分頁使用時機。
- 補充 `CHANGELOG` 維護說明：`Auto: updated ...` 條目只記錄 staged 檔案變動；本次已補登人工整理的功能摘要與行為說明（涵蓋 2026-02-15 前的遺漏項目）。
- 新增 `FUGLE_MCP_GUIDE.md`：整理 Fugle MCP Server 與 Agent 協作設定步驟與設定範本。
- `README.md` 補充 00935 熱力圖分頁與功能說明。
- `README.md` 補充「LLM 初始化自動化」使用說明。
- `README.md`、`AGENTS.md`、`PROJECT_CONTEXT.md` 補充「自動更新 CHANGELOG」啟用方式與行為。
- `README.md` 補上 `universe_snapshots` 資料表，並更新 `app.py` 說明為六分頁主流程。
- `README.md`、`PROJECT_CONTEXT.md` 更新分頁列表，加入 `資料庫檢視`。

## [0.3.0] - 2026-02-13

### Added
- 新增「策略是否贏過大盤」與「贏/輸大盤百分比」指標。
- 回放預設起點改為第 20 根 K 棒，降低初始空白感。

### Changed
- 預設主題改為灰白色。
- 主題選項精簡為 `日光白` / `灰白專業`。
- K 棒配色改為柔和低飽和半透明風格（粉紅 / 淡綠）。

### Fixed
- 修正 `ui_theme` 在 widget 建立後寫入 `session_state` 造成的 `StreamlitAPIException`。
- 修正即時資料欄位不完整時（缺 `open/high/low/close`）造成的 `KeyError`。

## [0.2.0] - 2026-02-13

### Added
- 新增多主題配色系統（後續已精簡）。
- 即時模式與回測畫面 UI 結構優化。
- 美股資料來源 fallback 強化（Yahoo/Stooq/快取）。
- 新增 `GIT_GITHUB_GUIDE.md` 新手文件。

## [0.1.0] - 2026-02-13

### Added
- 初始版本：即時看盤、回測工作台、SQLite 歷史資料與基本策略功能。
