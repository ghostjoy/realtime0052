# 台美股即時看盤 + 多來源資料 + 回測（Streamlit）

這個專案目前主要分頁如下：

- `即時看盤`：台股/美股即時與近即時走勢、技術指標、文字建議
- `回測工作台`：日K歷史下載、本地資料庫同步、策略回測、播放式回放、DCA（期初+每月定投）績效比較
- `2026 YTD 前十大股利型、配息型 ETF`：以 2026/01/01 至今區間計算前十大台股股利/配息型 ETF（卡片頁）
- `2026 YTD 前十大 ETF`：以 2026/01/01 至今區間計算前十大台股 ETF（卡片頁）
- `2025 後20大最差勁 ETF`：以 2025 全年區間「輸給台股大盤(%)」最多排序的後20名台股 ETF（排除復權事件標的，卡片頁）
- `共識代表 ETF`：以前10 ETF 成分股交集推導單一代表 ETF（附前3備選）
- `兩檔 ETF 推薦`：以「共識代表 + 低重疊動能」輸出核心/衛星兩檔組合（可切換海外限制與重疊門檻）
- `2026 YTD 主動式 ETF`：台股主動式 ETF 的 2026 YTD Buy & Hold 績效卡，含 Benchmark 對照圖（策略/基準/各ETF買進持有）
- `ETF 輪動策略`：台股多ETF（月頻）動能輪動策略回測與Benchmark對照
- `00910 熱力圖`：支援全球成分股 YTD（Buy & Hold）分組熱力圖（國內/海外分開對齊基準），並保留台股子集合進階回測
- `00935 熱力圖`：00935 成分股逐檔回測、相對大盤超額報酬熱力圖
- `00993A 熱力圖`：00993A 成分股逐檔回測、相對大盤超額報酬熱力圖
- `0050 熱力圖`：0050 成分股逐檔回測、相對大盤超額報酬熱力圖
- `0052 熱力圖`：0052 成分股逐檔回測、相對大盤超額報酬熱力圖
- `00910 / 00935 / 00993A / 0050 / 0052` 熱力圖頁：頁首統一顯示「官方編製/管理規則摘要」與來源連結
- `資料庫檢視`：查看 SQLite / DuckDB 內各資料表筆數、欄位結構與分頁資料
- `新手教學`：技術面與回測參數白話解釋、常見誤區、建議操作流程

> 免責聲明：僅供教育/研究，非投資建議。

## 相關文件

- 代理規範（給 LLM）：`AGENTS.md`
- 專案脈絡（給 LLM/新接手）：`PROJECT_CONTEXT.md`
- Prompt 範本：`PROMPT_TEMPLATES.md`
- GitHub 新手說明：`GIT_GITHUB_GUIDE.md`
- Fugle MCP 協作指南：`FUGLE_MCP_GUIDE.md`
- 更新紀錄：`CHANGELOG.md`

## LLM 初始化自動化

- 本 repo 內建初始化腳本（方法 2）：
  - `python3 scripts/bootstrap_project_docs.py --target .`
- 全域 Skill（方法 3）已建立於本機：
  - `~/.codex/skills/project-bootstrap`
  - 可用腳本：`python3 /Users/ztw/.codex/skills/project-bootstrap/scripts/bootstrap_repo_docs.py --target .`

## Commit 時自動更新 CHANGELOG

- 本 repo 提供 `pre-commit` hook：每次 `git commit` 前會自動執行 `scripts/auto_changelog.py`
- 功能：偵測本次 staged 檔案，將一筆 `Changed` 條目自動寫入 `CHANGELOG.md` 的 `Unreleased`
- 每台新電腦（每個 clone）需執行一次安裝：

```bash
./scripts/setup_git_hooks.sh
```

- 檢查是否啟用成功：

```bash
git config --get core.hooksPath
```

- 正常應顯示：`.githooks`

## 功能總覽

### 1) 多來源資料策略

- 美股：`Twelve Data -> Yahoo -> Stooq`（自動降級）
- 台股即時：`Fugle WebSocket -> TW MIS -> TW OpenAPI -> TPEx OpenAPI`（自動降級）
- 台股日K：`Fugle Historical（日K） -> TW OpenAPI(TWSE) -> TPEx OpenAPI(上櫃最新日資料) -> Yahoo`（自動降級；需 Fugle API key 才會啟用第一段）
- 每次顯示來源與資料品質（是否延遲、fallback depth、freshness）
- 主要卡片補上資料健康度：`as_of / source / source chain / fallback depth`
- 即時模式 UI：改為「即時總覽 / 即時走勢 / 側邊分析卡」版面，資訊密度更清楚
- 台股即時資料可批次落地到本地歷史儲存（DuckDB + Parquet），供後續查詢與回放擴充
- `Theme` 主題切換：可在側邊欄切換 `日光白` / `灰白專業` / `深色專業`

### 2) 自建歷史資料庫（DuckDB+Parquet 預設）

- 預設資料庫：
  - 若偵測到 iCloud Drive：`~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/market_history.duckdb`
  - 否則回退本地：`market_history.duckdb`
- `daily_bars` / `intraday_ticks` 以 Parquet（market/symbol 分區）儲存
- 表（DuckDB）：
  - `instruments`
  - `sync_state`
  - `symbol_metadata`（代碼/名稱/產業等快取）
  - `bootstrap_runs`（預載/增量更新任務紀錄）
  - `backtest_runs`
  - `backtest_replay_runs`
  - `universe_snapshots`
  - `heatmap_runs`
  - `rotation_runs`

```bash
# 可選：自訂 DuckDB 與 Parquet 路徑
export REALTIME0052_DUCKDB_PATH="/your/path/market_history.duckdb"
export REALTIME0052_PARQUET_ROOT="/your/path/parquet"
```

### 2.5) 基礎資料預載（降低第一次等待）

- `資料庫檢視` 分頁可直接執行：
  - `啟動基礎資料預載`（台股 + 美股核心）
  - `執行一次增量更新`
- CLI 版本：

```bash
uv run python scripts/bootstrap_market_data.py --scope both --years 5 --max-workers 6
```

- 預載流程會先建立 `symbol_metadata`，再批次同步 `daily_bars`，最後把任務摘要寫入 `bootstrap_runs`。

### 3) 回測（第一版）

- 粒度：日K
- 策略模板：
  - `buy_hold`（買進持有，適合近期短區間檢查）
  - `sma_cross`
  - `ema_cross`
  - `rsi_reversion`
  - `macd_trend`
- 交易規則：`T` 日收盤產生訊號，`T+1` 開盤成交（全倉進出）
- 成本模型：手續費 + 賣出稅 + 滑價（可在 UI 調整）
- 初始資產：可在回測頁直接輸入本金（例如 1,000,000 / 2,000,000）
- 實際投入起點：可選「沿用起始日 / 指定日期 / 指定第幾根K」
- 報表：總報酬、CAGR、MDD、Sharpe、勝率、交易明細、逐年報酬

### 3.5) Phase-2 回測增強

- 投組回測：多標的（逗號分隔）等權資金分配
- Walk-Forward：Train/Test 切分 + 參數搜尋（`sharpe/cagr/total_return/mdd`）
- Benchmark 比較：可手動選擇基準；`Auto` 預設台股 `^TWII` / 美股 `^GSPC`，失敗時自動改用代理標的（台股 `0050/006208`、美股 `SPY/QQQ/DIA`）
- 多標的比較：在 `Benchmark Comparison` 可加上各標的的 `Buy-and-Hold` 線（多條線）
- 回放副圖可同時顯示策略資產曲線與 Benchmark 資產曲線
- 買賣點顯示可切換：`訊號點（價格圖）` / `實際成交點（資產圖）`
- 回放視窗可切換：`固定視窗` / `完整區間`，並可調整 `生長位置（靠右%）`
- 可開啟 `顯示成交連線`，用垂直細線對照價格圖與資產圖的成交時刻
- 分項績效：投組中每一檔標的的獨立績效明細
- 分割調整（復權）：回測頁可開啟 `分割調整（復權）`，避免像 `0052` 分割後造成歷史曲線與報酬失真
- 還原權息（Adj Close）：回測頁可開啟 `還原權息計算（Adj Close）`，若資料有 `adj_close` 會優先套用，並避免與分割調整重複計算
- 回測頁可開啟 `回測前自動補資料缺口（推薦）`，僅同步缺口標的，降低「天數不足」機率
- 回測回放快取加入 `schema_version + source_hash` 參數簽章，避免載入不相容舊快取
- 回測回放 payload 升級為 `v3` 分層格式（`meta / bars / results`），並保留舊版 `v2` 載入相容
- 回測 `run_key` 與回放簽章改由共用模組產生（`services/backtest_cache.py`），降低 key 不一致風險
- 台股 Benchmark 載入鏈路改由共用 loader 處理（`services/benchmark_loader.py`），熱力圖/輪動/比較卡邏輯一致
- 回測同步流程改由 `services/sync_orchestrator.py` 共用（all/backfill/min_rows）
- 回測頁 widget/session key 改由 `state_keys.py` 管理，減少日後功能擴充時 key 衝突
- 新增執行層 runner：`services/backtest_runner.py`、`services/heatmap_runner.py`、`services/rotation_runner.py`，將資料準備/執行流程從 `app.py` 抽離
- 新增效能除錯開關：`REALTIME0052_PERF_DEBUG=1`（顯示頁面分段耗時）

### 3.6) 00935 成分股熱力圖回測

- 使用 00935 成分股（快取到 SQLite）逐檔回測，並與 Benchmark（`^TWII/0050/006208`）比較
- 熱力圖顯示相對大盤超額報酬：
  - 綠色：贏過大盤（贏越多越深）
  - 紅色：輸給大盤（輸越多越深）
- 每格標示檔號與超額報酬 `%`，並可查看表格明細（策略報酬 / 大盤報酬 / 差值）

### 3.7) 0050 成分股熱力圖回測

- 使用 0050 成分股（快取到 SQLite）逐檔回測，並與 Benchmark（`^TWII/0050/006208`）比較
- 熱力圖顯示相對大盤超額報酬：
  - 綠色：贏過大盤（贏越多越深）
  - 紅色：輸給大盤（輸越多越深）
- 每格標示代碼/公司名稱與超額報酬 `%`，並可查看表格明細

### 3.8) 台股多ETF輪動策略（日K）

- 固定標的池：`0050 / 0052 / 00935 / 0056 / 00878 / 00919`
- 大盤濾網：`Benchmark close > SMA60` 才允許持有，否則全數空手
- 動能分數：`0.2*20日報酬 + 0.5*60日報酬 + 0.3*120日報酬`
- 選股規則：每月第一個交易日評分，取前 `Top N`（預設 3）等權配置
- 持有與成交：月調倉；訊號後「下一交易日開盤」執行（避免前視偏誤）
- 風控：分數 `<= 0` 或跌破 `SMA60` 不納入當月排名
- 與 `Benchmark Equity`、`Buy-and-Hold (ETF池等權)` 同圖比較
- 調倉明細與成交紀錄可直接在分頁查看，並快取到 SQLite（`rotation_runs`）

### 4) 回放式視覺化

- `Play / Pause / Reset`
- 速度：`0.5x / 1x / 2x / 5x / 10x`
- 拖曳定位回放進度
- K線與資產曲線同步推進
- 預設直接顯示 `完整區間` 且定位在最後一根；要重播再按 `Reset`

### 5) 參數解釋（中文）

- `Slippage`：比例（小數），`0.0005 = 0.05%`，買賣雙邊都會套用
- `自動套用市場成本參數`：切換台/美股與標的時，會自動套用對應手續費、稅率與滑價預設
- `Walk-Forward`：Train 區挑參數，Test 區驗證穩健性
- `Train 比例`：例如 `0.70` 代表前 70% 訓練、後 30% 測試
- `參數挑選目標`：
  - `sharpe` 越高越好
  - `cagr` 越高越好
  - `total_return` 越高越好
  - `mdd` 越小越好
- `實際投入起點` 會影響回測結果；`回放位置` 只影響播放位置
- `buy_hold` 僅需至少 2 根 K；其餘策略建議至少 40 根 K；Walk-Forward 需至少 80 根 K
- `買賣點顯示`：
  - `訊號點（價格圖）` = 策略切換點
  - `實際成交點（資產圖）` = 回測成交點（T+1 開盤）
- `Equity`：策略資產曲線（總資產 = 現金 + 持倉市值）
- `Benchmark Equity`：同初始資產下的基準指數資產曲線
- `Benchmark` 可手動切換或關閉；若主來源限流，建議改用代理標的
- `Benchmark 比較`：同期間比較策略 vs 基準的總報酬、CAGR、MDD、Sharpe
- `策略 vs 買進持有`：比較策略報酬率與買進持有（單檔或等權投組）
- `指定區間報酬率`：可輸入起訖日期（如 2026-01-01 ~ 2026-02-11）比較策略與買進持有

## 新手快速上手（10 分鐘）

1. 先到 `回測工作台`，輸入單一標的 + `buy_hold`，系統會自動回測並顯示績效卡。
2. 再切到 `2026 YTD 前十大 ETF`，看 `YTD報酬(%) / 2025績效(%) / 贏輸台股大盤(%)`。
3. 再到 `共識代表 ETF` 看核心標的，接著用 `兩檔 ETF 推薦` 直接收斂成可執行組合。
4. 想看主動式 ETF 就到 `2026 YTD 主動式 ETF`，同樣可用 `更新最新市況` + Benchmark 對照卡。
5. 想看 ETF 內部強弱就到 `00910 / 00935 / 00993A / 0050 / 0052` 熱力圖，先按 `更新 成分股` 再執行回測。
6. 想看規則化組合再進 `ETF 輪動策略`，比較策略曲線、基準曲線與 ETF 池等權買進持有。
7. 任何頁面若看到舊結果，先按該頁 `更新最新市況` 或 `重新計算` 再判讀。

補充：完整名詞與操作流程請看 App 內 `新手教學` 分頁（含分頁地圖、快取邏輯、常見誤解與判讀範例）。

## 安裝

### 用 uv（建議）

```bash
cd /Users/ztw/codexapp/realtime0052
uv sync
```

## 環境變數

若要啟用美股主來源 Twelve Data，請設定：

```bash
export TWELVE_DATA_API_KEY="your_api_key"
```

未設定時，系統會自動從下一個來源（Yahoo）開始。

若要啟用 Fugle（台股即時 + 台股歷史日K優先），請設定：

```bash
export FUGLE_MARKETDATA_API_KEY="your_fugle_key"
```

若你不想每次 export，也可把 key 放到預設檔案（App 與 MCP 腳本都會自動讀）：

```text
~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/fuglekey
```

也可用環境變數改成其他檔案路徑：

```bash
export FUGLE_MARKETDATA_API_KEY_FILE="/your/path/fuglekey"
```

可選：若你要改走 MCP/代理 relay，可覆蓋 WS endpoint：

```bash
export FUGLE_WS_URL="wss://your-relay-or-proxy-endpoint"
```

若要把 Fugle MCP server 交給 Agent 使用（Claude / 其他 MCP client），可直接用本 repo 腳本作為 command：

```json
{
  "mcpServers": {
    "fugle-marketdata": {
      "command": "/Users/ztw/codexapp/realtime0052/scripts/run_fugle_mcp_server.sh"
    }
  }
}
```

更多細節請看：`FUGLE_MCP_GUIDE.md`

若要手動指定 DuckDB 路徑（會覆蓋預設 iCloud/本地判斷），可設定：

```bash
export REALTIME0052_DUCKDB_PATH="/your/path/market_history.duckdb"
export REALTIME0052_PARQUET_ROOT="/your/path/parquet"
```

若要改回 legacy env（除錯/回滾用）：

```bash
export REALTIME0052_CONFIG_SOURCE="legacy_env"
```

若要切換圖表渲染器（預設維持 Plotly，可安全回退）：

```bash
export REALTIME0052_KLINE_RENDERER_LIVE="lightweight"
export REALTIME0052_KLINE_RENDERER_REPLAY="lightweight"
export REALTIME0052_BENCHMARK_RENDERER="lightweight"
```

若要調整即時資料（`intraday_ticks`）保留天數（預設 1095 天），可設定：

```bash
export REALTIME0052_INTRADAY_RETAIN_DAYS="1095"
```

## SQLite -> DuckDB 一次性遷移（可回滾）

先保留舊 SQLite 檔案不動，再執行：

```bash
uv run python scripts/migrate_sqlite_to_duckdb.py \
  --sqlite-path "/your/path/market_history.sqlite3" \
  --duckdb-path "/your/path/market_history.duckdb" \
  --parquet-root "/your/path/parquet"
```

回滾方式：執行 `./scripts/rollback_legacy_stack.sh`，或手動把 `REALTIME0052_STORAGE_BACKEND=sqlite`。

## 執行

```bash
cd /Users/ztw/codexapp/realtime0052
uv run streamlit run app.py
```

一鍵啟用預設技術線（Hydra + DuckDB/Parquet + Plotly）：

```bash
cd /Users/ztw/codexapp/realtime0052
./scripts/enable_new_stack.sh
```

腳本會在啟動前列印目前技術線摘要（`config_source / storage_backend / renderer / 路徑是否 iCloud`）。
若偵測到 iCloud `codexapp` 目錄，腳本預設會直接使用 iCloud 路徑（DuckDB / Parquet / SQLite）。

一鍵回滾舊技術線（legacy_env + SQLite + Plotly）：

```bash
cd /Users/ztw/codexapp/realtime0052
./scripts/rollback_legacy_stack.sh
```

## 測試

```bash
cd /Users/ztw/codexapp/realtime0052
uv run python -m unittest discover -s tests -v
```

## 專案結構（主要）

- `app.py`：Streamlit UI 主流程（即時看盤 / 回測工作台 / 2026 YTD 前十大股利型、配息型 ETF / 2026 YTD 前十大 ETF / 2025 後20大最差勁 ETF / 共識代表 ETF / 兩檔 ETF 推薦 / 2026 YTD 主動式 ETF / ETF 輪動策略 / 00910 熱力圖 / 00935 熱力圖 / 00993A 熱力圖 / 0050 熱力圖 / 0052 熱力圖 / 資料庫檢視 / 新手教學）
- `services/market_data_service.py`：多來源 provider chain 與資料品質封裝
- `providers/`：各資料來源 adapter
- `storage/history_store.py`：SQLite schema、增量同步、回測紀錄
- `storage/duck_store.py`：DuckDB + Parquet hybrid store（與既有介面相容）
- `conf/`：Hydra YAML 設定（預設直接啟用 `hydra`）
- `config_loader.py`：Hydra/環境變數雙軌設定讀取器
- `backtest/`：策略模板、回測引擎、結果型別
- `tests/`：單元測試（資料鏈路/回測/儲存）
