# CHANGELOG

這份檔案用來記錄專案的重要更新（功能、修正、文件）。

格式說明：
- `Added`：新功能
- `Changed`：行為或介面調整
- `Fixed`：錯誤修正
- `Docs`：文件更新

## [Unreleased] - 2026-02-13

### Added
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
- 新增 `00910 熱力圖` 獨立卡片頁：
  - 可列出全部成分股清單
  - 可執行與 `00935` 同級的成分股熱力圖回測

### Changed
- 台股資料鏈路升級：即時改為 `Fugle WebSocket -> TW MIS -> TW OpenAPI -> TPEx OpenAPI`，日K同步新增 `TPEx OpenAPI`（短區間/最新日資料）後再回退 Yahoo。
- UI 導覽改為卡片式（取代原先分頁列），並新增 `即時看盤/回測工作台` 卡片化區段（即時行情卡、即時趨勢卡、績效卡、回放控制卡、回放圖卡）。
- 新增設計協作工具：可下載 `design-tokens.json`，方便與 Figma / Pencil 對齊色票與視覺 token。
- 前十大 ETF 排行頁補充資訊密度：加入樣本統計、分類說明、實際交易日區間與來源註記。
- 前十大 ETF 排行改為「復權後報酬」計算：區間報酬改用 `復權期初`（套用已知 split 事件）對比期末收盤。
- Auto: updated AGENTS.md, PROJECT_CONTEXT.md, README.md, app.py, storage/history_store.py [id:8cdb1d9956]
- Auto: updated PROJECT_CONTEXT.md, README.md, app.py [id:2531f53664]
- Auto: updated README.md, app.py, services/market_data_service.py, tests/test_market_data_service.py [id:4b701c5149]
- Auto: updated PROJECT_CONTEXT.md, README.md, app.py, backtest/__init__.py, backtest/rotation.py, backtest/types.py, ... (+3) [id:3bd6d76b10]
- Auto: updated .githooks/pre-commit, AGENTS.md, PROJECT_CONTEXT.md, README.md, app.py, backtest/__init__.py, ... (+12) [id:d87b9ff71f]
- SQLite 預設路徑改為「優先 iCloud Drive、否則本地」：iCloud 可用時使用 `~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/market_history.sqlite3`，並支援 `REALTIME0052_DB_PATH` 覆蓋。
- 主介面分頁調整為：`即時看盤 / 回測工作台 / 00935 熱力圖 / 新手教學`。
- 主介面分頁調整為：`即時看盤 / 回測工作台 / 00935 熱力圖 / 0050 熱力圖 / 新手教學`。
- 主介面分頁調整為：`即時看盤 / 回測工作台 / ETF輪動 / 00935 熱力圖 / 0050 熱力圖 / 新手教學`。
- 新手教學中補充「成分股快取」資料儲存位置說明。
- `00935 熱力圖` 回測結果新增公司名稱欄位（含熱力圖文字與 hover 顯示）。
- 維護檢查紀錄：完成 `compileall`、`unittest(40)`、分頁冒煙測試、SQLite 快取一致性檢查與文件對齊檢查。
- 熱力圖頁新增「結果怎麼看（越大/越小）」提示，補充 `勝負檔數/平均超額/最佳最差/Bars` 的判讀方式。
- 主畫面改為「單頁渲染」：切換頁面時只執行目前頁面內容，避免所有分頁同時 rerun 造成等待。
- 熱力圖/ETF 輪動新增「執行前同步最新日K（較慢）」開關（預設關閉），優先使用本地 SQLite。
- 多標的同步改為可平行處理（可切換關閉），改善多檔同步等待時間。
- 回測工作台「App 啟動時自動增量同步」預設改為關閉，並新增平行同步選項。
- 主介面分頁名稱由 `ETF輪動` 調整為 `ETF 輪動策略`，並同步更新該分頁按鈕/提示文案。
- `持有最久 ETF` 推薦由前兩名改為前三名（Top3）。
- `sync_symbol_history(...)` 起始日期邏輯調整：若使用者指定起點早於本地最早資料，會正確回補舊資料；若請求區間已在本地覆蓋範圍內，則維持增量向前同步。
- 台股歷史同步鏈路調整為：`Fugle Historical(日K) -> TW OpenAPI -> TPEx OpenAPI -> Yahoo`（有 key 時優先 Fugle），適用回測工作台與 00935/0050/ETF 輪動缺口補齊。

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
