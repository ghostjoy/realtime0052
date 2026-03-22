# 台美股即時看盤 + 多來源資料 + 回測（Streamlit）

這個專案目前主要分頁如下：

- `即時看盤`：台股/美股即時與近即時走勢、技術指標、文字建議
- `回測工作台`：日K歷史下載、本地資料庫同步、策略回測、播放式回放、DCA（期初+每月定投）績效比較
- `YTD股利型ETF`：以 2026/01/01 至今區間計算全部台股 `類型 = 股利型` ETF（卡片頁）
- `YTD top15 ETF`：以 2026/01/01 至今區間計算前15大台股 ETF（卡片頁）
- `2025 後20大最差勁 ETF`：以 2025 全年區間 `大盤超額(%)` 最低（輸給大盤最多）排序的後20名台股 ETF（排除復權事件標的，卡片頁）
- `共識代表 ETF`：以前10 ETF 成分股交集推導單一代表 ETF（附前3備選）
- `兩檔 ETF 推薦`：以「共識代表 + 低重疊動能」輸出核心/衛星兩檔組合（可切換海外限制與重疊門檻）
- `2026 YTD 主動式 ETF`：台股主動式 ETF 的 2026 YTD Buy & Hold 績效卡，依本專案 `類型 = 主動式` 建立母體，含 Benchmark 對照圖
- `台股 ETF 全類型總表`：台股 ETF 全名單與 2025/2026 YTD/大盤勝負比較，保留 `類型` 的語意分類（如科技型/股利型/市值型），並新增 `ETF 日成交資訊`、融資融券、MIS、三大法人與基金規模追蹤
- `ETF 輪動策略`：台股多ETF（月頻）動能輪動策略回測與Benchmark對照
- `00910 熱力圖`：支援全球成分股 YTD（Buy & Hold）分組熱力圖（國內/海外分開對齊基準），並保留台股子集合進階回測
- `00935 熱力圖`：00935 成分股逐檔回測、相對大盤超額報酬熱力圖
- `00735 熱力圖`：00735 成分股逐檔回測、相對大盤超額報酬熱力圖
- `00993A 熱力圖`：00993A 成分股逐檔回測、相對大盤超額報酬熱力圖
- `0050 熱力圖`：0050 成分股逐檔回測、相對大盤超額報酬熱力圖
- `0052 熱力圖`：0052 成分股逐檔回測、相對大盤超額報酬熱力圖
- `筆記本`：GitHub 風格 markdown 多筆筆記卡，左欄清單切換，中欄整合預覽/編輯，內容保存到 DuckDB
- `即時看盤`（台股）新增可選 `FinMind` 研究資料卡：可顯示公司基本資料、月營收、近期新聞與法人籌碼摘要
- `00910 / 00935 / 00993A / 0050 / 0052` 熱力圖頁：頁首統一顯示「官方編製/管理規則摘要」與來源連結
- `資料庫檢視`：查看 DuckDB 內各資料表筆數、欄位結構與分頁資料
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

- 本 repo 提供 `pre-commit` hook：每次 `git commit` 前會自動執行 `ruff check` + `ruff format` + `mypy` + `scripts/auto_changelog.py`
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
- `market_history.duckdb` 與 `parquet/` 為本地執行產物，repo 不追蹤；請用預載腳本或 CLI 建立
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
  - `tw_etf_daily_market`（TWSE 官方 ETF 日成交）
  - `tw_etf_mis_daily`（TWSE 官方 ETF MIS 指標）
  - `tw_etf_super_export_runs`（超級大表 CLI 匯出紀錄）
  - `market_snapshots`（外部來源快照、來源鏈路與新鮮度）

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

```bash
# 新增 CLI 入口（同步/回測/預載）
uv run realtime0052 info
uv run realtime0052 sync --market TW --symbols 0050,0052 --days 60
uv run realtime0052 sync-twse-etf-daily --start 2026-03-01 --end 2026-03-08
uv run realtime0052 sync-twse-etf-mis
uv run realtime0052 sync-tw-etf-constituents
uv run realtime0052 export-tw-etf-super-table --out ./tw_etf_super_export_latest.csv
uv run realtime0052 export-tw-etf-report --symbol 0052 --out ./reports/
uv run realtime0052 export-etf-briefing --out-root ~/Downloads/etf --start 2023-01-01 --single-symbol 00935
uv run realtime0052 backtest --symbol 0050 --market TW --strategy buy_hold
uv run realtime0052 chart-backtest --symbols 0050 --layout single --start 2024-01-01 --end 2026-03-20
uv run realtime0052 bootstrap --scope both --years 5
```

- 預載流程會先建立 `symbol_metadata`，再批次同步 `daily_bars`，接著同步 `tw_etf_daily_market` 與 `tw_etf_mis_daily`，最後把任務摘要寫入 `bootstrap_runs`。
- `export-tw-etf-super-table` 適合放進 `crontab`：會先同步主總表來源 + 官方 ETF 日成交 + 官方 ETF 融資融券 + 官方 MIS + 官方三大法人快取 + `基金規模追蹤`，再輸出 CSV，並把該次匯出摘要寫入 DuckDB `tw_etf_super_export_runs`。
- 超級大表 CSV 會額外帶入基金規模最近 `10` 個交易日原始欄位，以及 `1 / 5 / 10` 日規模變化摘要，方便直接看資金流入流出。
- `台股 ETF 全類型總表` 頁面上的官方日成交 / 融資融券 / MIS / 三大法人 / `基金規模追蹤` 區塊都讀本地快取，所以 `crontab` 跑完後，網頁下次開啟或重整時也會看到更新後的資料。

#### CLI `--help` 與圖表匯出

- 除 `serve` 以外，其他 CLI 都可以在沒有先啟動 `uv run streamlit run app.py` 的情況下獨立使用。
- 既有 CLI 都可直接看 `--help`：
  - `uv run realtime0052 --help`
  - `uv run realtime0052 export-tw-etf-super-table --help`
  - `uv run realtime0052 sync-tw-etf-constituents --help`
  - `uv run realtime0052 export-tw-etf-report --help`
  - `uv run realtime0052 export-etf-briefing --help`
  - `uv run realtime0052 backtest --help`
  - `uv run realtime0052 chart-backtest --help`
- 各指令的獨立 markdown 文件在 [`docs/cli/README.md`](./docs/cli/README.md)
- 單檔 ETF 報表包的完整說明在 [`docs/tw-etf-report-bundle.md`](./docs/tw-etf-report-bundle.md)
- `export-etf-briefing` 會一次輸出大表 CSV、科技型/主動式名單、combined/single 圖、HTML 戰報與 FB 文案，適合做每日簡報包。
- `chart-backtest` 會輸出 `PNG`，適合圖表、文件與 AI workflow。
- `sync-tw-etf-constituents` 會把台股 ETF 成分股快照寫進 DuckDB `market_snapshots`，並產生 JSON / Markdown log。
- `export-tw-etf-report` 會輸出單一 ETF 的資料夾報表包，內含單檔總表、MIS、三大法人、基金規模、成分股、技術指標、回測圖、熱力圖與 sync log。
- `single`：單一標的，預設輸出一張乾淨版 `K線 + Equity + Benchmark` 圖；若要接近回放參考圖，可加 `--reference-annotations`。
- `combined`：多標的同圖，改成單面板「Benchmark 虛線 + 等權策略線 + 各標的 Buy and Hold 線」的相對倍數比較圖。
- `combined` 可用 `--include-ew-portfolio` 額外加上 EW portfolio 線；預設不顯示。
- `split`：多標的輸入但逐檔各輸出一張 `PNG`。
- 所有模式都固定疊 Benchmark 虛線；台股預設 `^TWII`，美股預設 `^GSPC`，並沿用現有 fallback 邏輯。

```bash
# 單一標的
uv run realtime0052 chart-backtest \
  --symbols 0050 \
  --layout single \
  --start 2024-01-01 \
  --end 2026-03-20

# 單一標的，開啟參考圖標註
uv run realtime0052 chart-backtest \
  --symbols 0050 \
  --layout single \
  --reference-annotations \
  --start 2024-01-01 \
  --end 2026-03-20

# 多標的同圖
uv run realtime0052 chart-backtest \
  --symbols 0050,0052,006208 \
  --layout combined \
  --market TW \
  --start 2024-01-01 \
  --end 2026-03-20

# 多標的個別輸出
uv run realtime0052 chart-backtest \
  --symbols 0050,SPY \
  --layout split \
  --market auto \
  --start 2024-01-01 \
  --end 2026-03-20 \
  --out-dir ./artifacts/charts
```

#### `crontab` 範例：每日匯出超級大表

若要每天固定匯出到指定資料夾，建議把 `--out` 指到「目錄」而不是固定檔名；這樣 CLI 會沿用預設命名規則，自動輸出成 `tw_etf_super_export_<trade_day>.csv`，每天保留一份。

```cron
CRON_TZ=Asia/Taipei
5 17 * * * cd /Users/ztw/codexapp/realtime0052_minimax && /opt/homebrew/bin/uv run realtime0052 export-tw-etf-super-table --out "/Users/ztw/Library/CloudStorage/GoogleDrive-ghostjoy@gmail.com/My Drive/etf_report/"
```

- 上例會在每天 `17:05` 執行，輸出到 `Google Drive/etf_report/`
- 實際檔名會像 `tw_etf_super_export_20260318.csv`
- 若把 `--out` 寫成固定檔名，例如 `--out ./tw_etf_super_export_latest.csv`，則每次都會覆蓋同一個檔案
- `cron` 環境通常不會自動帶入互動式 shell 的 `PATH`，建議像上例一樣使用 `uv` 的絕對路徑
- 若想保留執行紀錄，可改寫成 `... >> /path/to/tw_etf_super_export.log 2>&1`

若本機另外有包一層 wrapper script（例如 `/Users/ztw/bin/export_tw_etf_super_table.sh`），也可以直接用該腳本：

```bash
# 最新交易日
/Users/ztw/bin/export_tw_etf_super_table.sh

# 指定單日（會輸出該交易日對應的超級大表）
/Users/ztw/bin/export_tw_etf_super_table.sh 2026-03-18

# 指定日期區間（逐日跑；非交易日會回退到前一個交易日，因此可能覆蓋成同一份交易日檔案）
/Users/ztw/bin/export_tw_etf_super_table.sh 2026-03-06 2026-03-18
```

- 單日/區間模式的核心是把目標日期傳給 `export-tw-etf-super-table --ytd-end <date>`
- 區間若包含週末或休市日，例如 `2026-03-08`，實際可能仍落到最近一個可交易日檔名，例如 `tw_etf_super_export_20260306.csv`

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

- 使用 00935 成分股（快取到 DuckDB）逐檔回測，並與 Benchmark（`^TWII/0050/006208`）比較
- 熱力圖顯示相對大盤超額報酬：
  - 綠色：贏過大盤（贏越多越深）
  - 紅色：輸給大盤（輸越多越深）
- 每格標示檔號與超額報酬 `%`，並可查看表格明細（策略報酬 / 大盤報酬 / 差值）

### 3.7) 0050 成分股熱力圖回測

- 使用 0050 成分股（快取到 DuckDB）逐檔回測，並與 Benchmark（`^TWII/0050/006208`）比較
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
- 調倉明細與成交紀錄可直接在分頁查看，並快取到 DuckDB（`rotation_runs`）

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
2. 再切到 `YTD top15 ETF`，看 `YTD績效(%) / 2025績效(%) / 大盤超額(%)`。
3. 再到 `共識代表 ETF` 看核心標的，接著用 `兩檔 ETF 推薦` 直接收斂成可執行組合。
4. 想看主動式 ETF 就到 `2026 YTD 主動式 ETF`，同樣可用 `更新最新市況` + Benchmark 對照卡。
5. 想看 ETF 內部強弱就到 `00910 / 00935 / 00735 / 00993A / 0050 / 0052` 熱力圖，先按 `更新 成分股` 再執行回測。
6. 想看規則化組合再進 `ETF 輪動策略`，比較策略曲線、基準曲線與 ETF 池等權買進持有。
7. 任何頁面若看到舊結果，先按該頁 `更新最新市況` 或 `重新計算` 再判讀。

補充：完整名詞與操作流程請看 App 內 `新手教學` 分頁（含分頁地圖、快取邏輯、常見誤解與判讀範例）。

## 安裝

### 用 uv（建議）

```bash
cd ~/codexapp/realtime0052_minimax
uv sync
```

## 環境變數

若要啟用美股主來源 Twelve Data，請設定：

```bash
export TWELVE_DATA_API_KEY="your_api_key"
```

未設定時，系統會自動從下一個來源（Yahoo）開始。

若你不想每次 export，也可把 key 放到預設檔案（會自動讀）：

```text
~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/twelvedatakey
```

也可用環境變數改成其他檔案路徑：

```bash
export TWELVE_DATA_API_KEY_FILE="/your/path/twelvedatakey"
```

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
      "command": "$PROJECT_DIR/scripts/run_fugle_mcp_server.sh"
    }
  }
}
```

更多細節請看：`FUGLE_MCP_GUIDE.md`

若要啟用 `FinMind`（台股研究資料增強：公司資料 / 月營收 / 新聞 / 法人籌碼），請設定：

```bash
export FINMIND_API_TOKEN="your_finmind_token"
```

若你不想每次 export，也可把 token 放到預設檔案（會自動讀）：

```text
~/Library/Mobile Documents/com~apple~CloudDocs/codexapp/finmindkey
```

也可用環境變數改成其他檔案路徑：

```bash
export FINMIND_API_TOKEN_FILE="/your/path/finmindkey"
```

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

## DuckDB 快照備份與回復

建立完整快照（DuckDB + Parquet）：

```bash
uv run python scripts/backup_duckdb_snapshot.py
```

回復指定快照（會覆蓋目標，需 `--force`）：

```bash
uv run python scripts/restore_duckdb_snapshot.py /path/to/snapshot_dir --force
```

執行快照清理與 DuckDB 維護（可選 CHECKPOINT/VACUUM）：

```bash
uv run python scripts/duckdb_housekeeping.py \
  --keep live_quote=5 \
  --keep live_ohlcv=14 \
  --keep twse_mi_index_allbut0999=60 \
  --default-keep-days 30 \
  --checkpoint
```

完整參數與情境範例請見：`docs/duckdb_housekeeping.md`

## 台股 ETF 管理費（可逐步補齊）

- 管理費資料檔：`conf/tw_etf_management_fees.json`
- 補資料工具：`scripts/manage_tw_etf_fees.py`
- App 端管理費設定會定期重新載入（約 2 分鐘快取），補完後通常不需重啟即可生效。

先看覆蓋率與待補名單（可同時輸出 CSV）：

```bash
uv run python scripts/manage_tw_etf_fees.py \
  --top 30 \
  --missing-csv conf/tw_etf_management_fees_missing.csv
```

單筆補值：

```bash
uv run python scripts/manage_tw_etf_fees.py --set 00900=0.30%起
```

批次補值（CSV 需有 `code/symbol/代碼/ETF代碼/基金代號` 與 `fee/management_fee/管理費/經理費` 欄位）：

```bash
uv run python scripts/manage_tw_etf_fees.py --input-csv /path/to/fees.csv
```

## 執行

```bash
cd ~/codexapp/realtime0052_minimax
uv run streamlit run app.py
```

一鍵啟用預設技術線（Hydra + DuckDB/Parquet + Plotly）：

```bash
cd ~/codexapp/realtime0052_minimax
./scripts/enable_new_stack.sh
```

腳本會在啟動前列印目前技術線摘要（`config_source / storage_backend / renderer / 路徑是否 iCloud`）。
若偵測到 iCloud `codexapp` 目錄，腳本預設會直接使用 iCloud 路徑（DuckDB / Parquet）。

## 測試

```bash
cd ~/codexapp/realtime0052_minimax
uv run python -m unittest discover -s tests -v
uv run pytest -q
uv run ruff check .
uv run mypy
```

## 專案結構（主要）

- `app.py`：Streamlit 啟動入口 + 全域樣式 + 頁面路由（其餘分頁渲染已逐步抽離）
- `ui/pages/live.py`：`即時看盤` 分頁主流程
- `ui/pages/backtest.py`：`回測工作台` 分頁主流程
- `ui/core/charts.py`：共用圖表渲染（Live / Indicator / Benchmark）
- `ui/core/health.py`：資料健康度與同步錯誤摘要 helper
- `ui/core/page_registry.py`：功能卡片導覽組裝與切頁 helper
- `services/market_data_service.py`：多來源 provider chain 與資料品質封裝
- `providers/`：各資料來源 adapter
- `storage/duck_store.py`：DuckDB + Parquet store（唯一儲存技術線）
- `conf/`：Hydra YAML 設定（預設直接啟用 `hydra`）
- `config_loader.py`：Hydra/環境變數雙軌設定讀取器
- `backtest/`：策略模板、回測引擎、結果型別
- `tests/`：單元測試（資料鏈路/回測/儲存）
