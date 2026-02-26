# 效能專項計畫（回測 + 熱力圖，平衡新鮮度，目標降延遲 30%）

## 摘要
此專項鎖定已選方向：
- 優先頁面：`回測工作台` + `熱力圖`
- 取捨：`平衡模式`（維持資料新鮮度，不激進延長 TTL）
- 驗收：主要互動延遲（回測執行、熱力圖重算）相較基準降低約 30%

計畫分三段：`量測基線 -> 計算切分與快取 -> 驗收與保護欄`，不改交易規則與 session key 語義。

## 1. 目標與量測定義

### 1.1 KPI（主指標）
- `KPI-1` 回測頁：同條件重跑（symbol/strategy/date/cost 不變）互動到結果完成時間降低 30%
- `KPI-2` 熱力圖頁：同條件重算（含 benchmark 對齊）時間降低 30%
- `KPI-3` 冷啟不惡化：首次執行時間相較現況不退步超過 10%

### 1.2 測量點（統一）
在 `PerfTimer` 現有框架上固定打點名稱，便於前後版本對照：
- 回測頁：`input_parse`, `bars_load`, `sync_if_needed`, `benchmark_load`, `backtest_execute`, `render`
- 熱力圖頁：`universe_load`, `bars_prepare`, `benchmark_load`, `heatmap_compute`, `table_render`, `chart_render`
- 每次執行落地到可選 log（debug 模式）與 UI caption 摘要

## 2. 實作策略（決策完成）

### 2.1 回測路徑優化
1. 將輸入解析、資料準備、執行、視覺化切成明確函式邊界（維持既有 runner）
2. 對純函式計算輸出加 `st.cache_data`（以 `run_key + source_hash` 當 key）
   - `load_and_prepare_symbol_bars` 產物摘要（可序列化部分）
   - `execute_backtest_run` 的可序列化結果（metrics/series payload）
3. 對不可安全快取部分保留即時計算
   - 涉及 session 即時狀態、播放器進度、UI 暫態
4. Benchmark 載入走統一快取鍵
   - `market + choice + date_range + source_hash`
   - 若來源資料沒變，跳過重抓/重對齊

### 2.2 熱力圖路徑優化
1. `prepare_heatmap_bars` 分兩層
   - 層 A：symbol bars 取得與必要同步（可快取摘要）
   - 層 B：回測逐檔計算（可按 symbol 分段快取）
2. `compute_heatmap_rows` 改為可分塊執行
   - 以 symbol list chunk（例如 20/批）逐段計算與累積
   - 每段可中間落地進度與 partial cache（避免一次失敗全重來）
3. benchmark 對齊結果快取（單獨 key），避免每次逐檔重算都重建 benchmark series

### 2.3 Rerun 與 UI 互動成本控制
1. 明確區分參數變更與展示切換
   - 只有參數變更才觸發計算
   - 僅視覺切換（排序/欄位顯示/tab）不觸發重算
2. 把高頻 widget 區與重計算區解耦（必要處維持 `st.fragment`）
3. 對 `st.rerun()` 加守門條件，避免同一狀態重複 rerun

### 2.4 快取失效策略（平衡模式）
- 預設 TTL 保持目前等級，不整體拉長
- 以 `source_hash/schema_version/date_range` 做精準失效
- 資料有更新時只失效受影響 key，不全域清快取

## 3. 檔案與介面變更

### 3.1 會新增/調整的內部介面
- `services/backtest_runner.py`
  - 新增可序列化 payload helper（供 cache 使用）
- `services/heatmap_runner.py`
  - 新增 chunked compute 入口（例如 `compute_heatmap_rows_chunked(...)`）
- `ui/shared/perf.py`
  - 增加標準 step 常數（避免打點名稱漂移）
- `app.py` 或後續 `ui/pages/*`
  - 統一呼叫上述新介面與打點

### 3.2 不變更項目
- 不改回測交易規則（`T+1`）
- 不改資料表 schema（本輪）
- 不改既有 session key 命名

## 4. 測試與驗收

### 4.1 自動化測試
- 既有全量：
  - `uv run python -m unittest discover -s tests -v`
- 新增測試：
  - cache key 穩定性（同輸入同 key，資料變更 key 變）
  - heatmap chunked 與原本單次計算結果一致
  - benchmark 快取命中時不重抓資料（mock 驗證）
  - UI 參數未變時不重算（以函式呼叫次數驗證）

### 4.2 效能回歸測試（非功能）
新增 `scripts/perf_smoke.py`（只量測不改資料）：
- 固定 2 組回測案例、2 組熱力圖案例
- 輸出每步耗時與 total
- 與 baseline JSON 比較，判定是否達 `-30%` 目標

### 4.3 驗收情境
1. 回測頁同參數重跑，平均耗時下降 >=30%
2. 熱力圖頁同參數重算，平均耗時下降 >=30%
3. 首次冷跑不明顯惡化（<=10%）
4. 所有現有單元測試通過，功能結果數值一致

## 5. 執行順序（建議）
1. 建 baseline（加標準 PerfTimer 打點 + 收集現況）
2. 做回測快取與 benchmark 對齊快取
3. 做熱力圖 chunked compute + 分段快取
4. 補測試（功能一致性 + cache 行為）
5. 跑 perf smoke，比對 KPI，微調 TTL/切塊大小
6. 更新 README / PROJECT_CONTEXT / CHANGELOG 的效能章節

## 6. 假設與預設
- 預設資料儲存仍以既有 DuckDB/Parquet + 現行同步邏輯
- 不新增付費依賴、不引入背景工作佇列
- 不做激進新鮮度犧牲（平衡模式）
- 若某些頁面資料來源波動大，以結果一致優先於極限提速
