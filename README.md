# 台美股即時看盤 + 多來源資料 + 回測（Streamlit）

這個專案現在有三個主要分頁：

- `即時看盤`：台股/美股即時與近即時走勢、技術指標、文字建議
- `回測工作台`：日K歷史下載、本地資料庫同步、策略回測、播放式回放
- `新手教學`：技術面與回測參數白話解釋、常見誤區、建議操作流程

> 免責聲明：僅供教育/研究，非投資建議。

## 功能總覽

### 1) 多來源資料策略

- 美股：`Twelve Data -> Yahoo -> Stooq`（自動降級）
- 台股即時：`TW MIS -> TW OpenAPI`（自動降級）
- 台股日K：`TW OpenAPI/TWSE 官方歷史端點`，必要時可用 Yahoo 補
- 每次顯示來源與資料品質（是否延遲、fallback depth、freshness）
- 即時模式 UI：改為「即時總覽 / 即時走勢 / 側邊分析卡」版面，資訊密度更清楚
- `Theme` 主題切換：可在側邊欄切換多種配色（夜間與日間）

### 2) 自建歷史資料庫（SQLite）

- 預設資料庫：`market_history.sqlite3`
- 啟用 `WAL`、UPSERT 增量同步
- 表：
  - `instruments`
  - `daily_bars`
  - `sync_state`
  - `backtest_runs`

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

### 4) 回放式視覺化

- `Play / Pause / Reset`
- 速度：`0.5x / 1x / 2x / 5x / 10x`
- 拖曳定位回放進度
- K線與資產曲線同步推進

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

## 安裝

### 用 uv（建議）

```bash
cd /Users/ztw/codexapp/realtime_0052
uv sync
```

## 環境變數

若要啟用美股主來源 Twelve Data，請設定：

```bash
export TWELVE_DATA_API_KEY="your_api_key"
```

未設定時，系統會自動從下一個來源（Yahoo）開始。

## 執行

```bash
cd /Users/ztw/codexapp/realtime_0052
uv run streamlit run app.py
```

## 測試

```bash
cd /Users/ztw/codexapp/realtime_0052
uv run python -m unittest discover -s tests -v
```

## 專案結構（主要）

- `app.py`：Streamlit UI（即時看盤 + 回測工作台）
- `services/market_data_service.py`：多來源 provider chain 與資料品質封裝
- `providers/`：各資料來源 adapter
- `storage/history_store.py`：SQLite schema、增量同步、回測紀錄
- `backtest/`：策略模板、回測引擎、結果型別
- `tests/`：單元測試（資料鏈路/回測/儲存）
