# AI Agent Runbook: ETF Briefing Output

這份文件是給 AI agent 的操作手冊。  
目標不是解釋功能，而是讓 agent 能穩定把 ETF briefing 輸出到指定目錄，並在卡住時知道怎麼排除。

適用目標：

- 匯出 `TW ETF 超級大表 CSV`
- 從大表篩出 `科技型 ETF / 主動式 ETF`
- 輸出 `combined / single / split` 圖
- 產出 `HTML` 報告與 `FB` 版本

## 1. 先決條件

- 不需要先啟動 `uv run streamlit run app.py`
- 優先使用：
  - `uv run realtime0052 export-tw-etf-super-table`
  - `uv run realtime0052 chart-backtest`
  - `uv run realtime0052 export-etf-briefing`
- 若只是要輸出 briefing，先不要做大範圍 `bootstrap`

## 2. 推薦執行順序

### 正常路徑

1. 先確認 CLI help 正常

```bash
uv run realtime0052 --help
uv run realtime0052 export-etf-briefing --help
uv run realtime0052 chart-backtest --help
```

2. 直接跑 briefing CLI

```bash
uv run realtime0052 export-etf-briefing \
  --out-root ~/Downloads/etf \
  --start 2023-01-01 \
  --single-symbol 00935
```

3. 驗證輸出目錄內至少有：

- `tw_etf_super_export_<trade_day>.csv`
- `tech_etf_from_super_export_<trade_day>.csv`
- `active_etf_from_super_export_<trade_day>.csv`
- `report_sima_yi.html`
- `fb_post_sima_yi.txt`
- `news_sources.json`
- `charts/tech_etf_combined_*.png`
- `charts/active_etf_combined_*.png`
- `charts/00935_single_*.png`

## 3. 這次實作遇到的真實卡點

### 卡點 A：全包命令卡在前段資料刷新

現象：

- `export-etf-briefing` 長時間無 stdout
- `briefing_YYYYMMDD/` 只建立目錄，或只寫出少量 CSV

常見原因：

- `export-tw-etf-super-table` 刷新官方來源時被網路或 DuckDB lock 拖住
- 本機 DuckDB 正被其他工作持有

排除方式：

1. 先看輸出目錄是否已寫出超級大表 CSV
2. 若還沒寫出，先不要一直重跑全包命令
3. 改成分段做：
   - 先單獨跑 `export-tw-etf-super-table`
   - 確認 CSV 已生成後，再做圖與報告

### 卡點 B：圖表前的預同步把任務拖死

現象：

- 大表 CSV 已經寫出
- 但後面圖檔遲遲不產生
- 程序長時間卡在大量歷史資料抓取

這次的教訓：

- 不要在 briefing orchestration 裡對整組 `科技型 / 主動式` 先做全量歷史補齊
- 對 20+ 檔科技型、10+ 檔主動式 ETF，回看三年區間，預同步很容易拖到不可控

建議策略：

- briefing 流程只用本地已有歷史資料判斷可畫標的
- 缺資料的標的記到 `issues`，不要讓整包 briefing 卡死
- 真要補資料，另外跑 `sync`，不要把補資料和產出 briefing 綁在同一個 blocking step

### 卡點 C：HTML 可能把 metadata 當成圖片路徑

現象：

- `img src` 裡出現 dict 內容，不是單純 `charts/*.png`

排除方式：

- 報告渲染前，統一把圖表輸出物轉成真正的 `path`
- 驗證 HTML 時直接搜：

```bash
rg "result&#x27;|symbols&#x27;" ~/Downloads/etf/briefing_*/report_sima_yi.html
```

正常狀況下，這個搜尋應該沒有結果。

### 卡點 D：ETF 代碼前導零被吃掉

現象：

- 表格裡出現 `52.00`、`935.00`

排除方式：

- 讀 CSV 時優先 `dtype=str`
- `代碼` 欄不要用數值格式化
- 對 `0052 / 00935 / 00988A` 這種欄位，一律當識別碼處理，不當數字處理

### 卡點 E：近期新聞會混入垃圾來源

現象：

- 搜到預測市場、入口網站二手轉貼、標題不相關內容

原則：

- 報告正文不要直接用原始搜尋結果
- 正文不列來源，但 sidecar 一定要保留 `news_sources.json`
- 允許 agent 在正文內用「臣已勘合近訊」這種語氣
- 不允許 agent 在正文內編造不存在的事件、日期、數字

建議做法：

1. 先收集近 14 天候選新聞
2. 再做人工或規則過濾
3. 若過濾後品質不夠，就在正文寫保守版：
   - `近十四日未見足以改寫盤勢的公開新變，宜守而不躁`

## 4. 推薦的 fallback 路徑

若 `export-etf-briefing` 卡住，請改走下面這條：

### Step 1：先拿大表

```bash
uv run realtime0052 export-tw-etf-super-table --out ~/Downloads/etf/
```

### Step 2：從大表篩清單

- 用 `類型 == 科技型`
- 用 `類型 == 主動式`

### Step 3：直接畫圖，不做預同步

```bash
uv run realtime0052 chart-backtest \
  --symbols <comma-separated-tech-symbols> \
  --layout combined \
  --market TW \
  --start 2023-01-01 \
  --end <today> \
  --theme soft-gray \
  --no-sync-before-run \
  --out <path>
```

主動式同理。  
`00935` 單檔圖則用 `--layout single`。

### Step 4：若全包命令失敗，但圖和 CSV 已齊

- 允許只重建：
  - `report_sima_yi.html`
  - `fb_post_sima_yi.txt`
  - `fb_post_sima_yi.html`
  - `news_sources.json`
  - `manifest.json`

也就是說：

- 不必因為最後報告生成失敗，就重跑整批 chart export

## 5. 報告文風規則

目標是《軍師聯盟》司馬懿，不是古文 cosplay。

### 要有的感覺

- 冷靜
- 克制
- 高判斷密度
- 先全局，後板塊，再個案
- 是在向主公密奏，不是在寫新聞稿

### 不要做的事

- 不要在正文列 Reuters / AP / forum 名稱清單
- 不要寫「根據某某來源」
- 不要亂寫市場陰謀論
- 不要把未驗證消息寫成既成事實

### 可以用的語氣

- `主公，臣觀今局`
- `臣以為`
- `可觀強，不可貪多`
- `不宜躁進`
- `先定局，後定策`

## 6. 驗證清單

輸出完成後，agent 應至少檢查：

1. `report_sima_yi.html` 能打開
2. HTML 內圖片路徑都是相對 `charts/...`
3. `fb_post_sima_yi.txt` 沒有 HTML tag
4. `news_sources.json` 存在
5. 主要圖檔存在且非空
6. `科技型 / 主動式` 的 CSV 非空
7. `00935_single_*.png` 存在

建議命令：

```bash
find ~/Downloads/etf/briefing_YYYYMMDD -maxdepth 2 -type f | sort
rg "result&#x27;|symbols&#x27;" ~/Downloads/etf/briefing_YYYYMMDD/report_sima_yi.html
```

## 7. 關鍵結論

AI agent 可以排除今天遇到的大多數問題，但前提是：

- 不把「同步全市場資料」和「輸出 briefing」強綁成同一條 blocking pipeline
- 遇到卡住時，知道要改走 `超級大表 -> 分組 CSV -> chart-backtest -> 報告重建` 的分段路徑
- 知道新聞正文要保守，追溯資訊放 sidecar，不要把報告寫成來源清單

若照這份 runbook 執行，agent 大多可以自己完成，不需要人工介入。
