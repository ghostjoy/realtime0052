# realtime0052 分階段優化計畫（架構 + Tooling + 文件）

## 摘要
本計畫採已確認策略：
- 架構：UI Modules（保留單一 `app.py` 入口）
- 節奏：兩階段重構
- 第一波搬遷頁面：`資料庫檢視`、`新手教學`、`熱力圖 Hub`
- Tooling：先落地 `Ruff + Pytest` 設定（保留現有 `unittest` 流程）

目標是先把頁面路由與低風險頁面抽離，建立可持續拆分框架，再逐步處理 `即時看盤`/`回測工作台` 等高耦合頁面，避免一次大搬遷造成回歸。

## 1. 架構重構（Phase 1 + Phase 2）

### 1.1 Phase 1（本輪可執行，低風險）

#### A. 新增 `ui/pages/` 模組化骨架
建立以下檔案（先不追求全部頁面搬完）：
- `ui/pages/__init__.py`
- `ui/pages/dashboard.py`
- `ui/pages/backtest.py`
- `ui/pages/heatmaps.py`
- `ui/pages/etf_cards.py`
- `ui/pages/settings.py`
- `ui/pages/registry.py`
- `ui/pages/types.py`（可選，但建議放頁面型別）

#### B. 定義統一頁面介面與註冊機制
在 `ui/pages/types.py`/`ui/pages/registry.py` 定義：
- `PageRenderer = Callable[[], None]`
- `PageCard`（`key`, `desc`）
- `build_page_cards(dynamic_cards_provider) -> list[PageCard]`
- `build_page_renderers(static_renderers, dynamic_renderers_provider) -> dict[str, PageRenderer]`

#### C. 先抽「頁面目錄 + 路由」，再搬第一波頁面實作
`app.py` 調整為：
- 保留：`st.set_page_config`、樣式注入、query/session 初始化、導航選擇
- 導入 `ui/pages/registry.py` 產生 `cards` 與 `page_renderers`
- 第一波將以下函式移入 `ui/pages/settings.py` / `ui/pages/heatmaps.py`：
  - `_render_db_browser_view`
  - `_render_tutorial_view`
  - `_render_heatmap_hub_view`
- 先保留 `app.py` 同名 wrapper（薄轉發）以維持既有測試相容，再逐步更新測試匯入點。

#### D. `session_state` 相容策略
- 所有既有 key 名稱不變（例如 `active_page`、熱力圖 hub keys）
- 只搬運程式碼位置，不改 key semantics
- 路由 fallback 保持現行行為（找不到 page 顯示錯誤訊息）

### 1.2 Phase 2（後續批次）
依風險由低到高搬遷：
1. ETF 卡片群頁（`etf_cards.py`）
2. 各熱力圖實作頁（`heatmaps.py`）
3. `即時看盤`（`dashboard.py`）
4. `回測工作台`（`backtest.py`，最後搬）

每批完成後都維持 `app.py` 只做入口、導航、session bootstrap。

## 2. 開發配置與工具鏈

### 2.1 `pyproject.toml` 單一真理（SoT）
調整方向：
- `dependencies`：維持 runtime 套件
- `[dependency-groups].dev`：加入 `pytest`, `ruff`
- 新增：
  - `[tool.ruff]`（`line-length`, `target-version`）
  - `[tool.ruff.lint]`（先用保守規則集，避免一次引爆舊碼）
  - `[tool.pytest.ini_options]`（`testpaths=["tests"]`, `python_files=["test_*.py"]`, `addopts="-v"`）

### 2.2 測試策略（漸進）
- 既有主流程仍以 `unittest` 為準（不破壞現況）
- 新增 `pytest` 作為統一入口的過渡層（可先僅收集與執行現有 tests）
- CI/文件同時提供：
  - `uv run python -m unittest discover -s tests -v`
  - `uv run pytest`

### 2.3 `requirements.txt` 同步規範
文件明確規定：
- `pyproject.toml` 是唯一來源
- `requirements.txt` 由命令生成，不手改：
  - `uv pip compile pyproject.toml -o requirements.txt`
- 建議在 CI 加「漂移檢查」步驟（重新 compile 後比對 diff）

## 3. 文件維護優化

### 3.1 README vs PROJECT_CONTEXT 去重
- `README.md`：保留對使用者的功能全覽與操作
- `PROJECT_CONTEXT.md`：
  - 精簡「功能地圖」為高層摘要
  - 明確引用 `README.md` 的功能清單
  - 強化「模組責任 / 資料流 / 快取與資料表 / 常見故障點」

### 3.2 `AGENTS.md` 新增 `Known Pitfalls`
新增章節（最少包含）：
- `st.session_state` 寫入後 rerun 的讀值時機
- widget key 唯一性與跨 rerun 穩定性
- `st.dataframe` vs `st.data_editor` 在大量資料的效能差異與選型
- 動態頁面 key（例如熱力圖 pin 卡）避免重複註冊
- query param 消費後清理，避免重複觸發路由

## 4. 公開介面/型別變更（重要）
雖然是內部重構，仍會新增可重用介面：
- `ui/pages/registry.py`
  - `build_page_cards(...)`
  - `build_page_renderers(...)`
- `ui/pages/{dashboard,backtest,heatmaps,etf_cards,settings}.py`
  - `register_pages(...)` 或 `render_*` 類型函式（固定簽名）
- `app.py`
  - `main()` 保留，但頁面映射由 registry 提供
  - 兼容 wrapper（暫存）直到測試完成遷移

## 5. 測試與驗收

### 5.1 自動化測試
每批至少執行：
1. `uv run python -m compileall app.py`
2. `uv run python -m unittest discover -s tests -v`
3. `uv run pytest -q`（新增後）

### 5.2 新增/調整測試案例
- 新增 `tests/test_page_registry.py`：
  - 靜態頁面註冊完整性
  - 動態熱力圖頁合併行為
  - unknown page fallback 行為
- 逐步把 `tests/test_active_etf_page.py` 對 `app` 的直接依賴改為新模組匯入（先保持相容，再遷移）

### 5.3 手動驗收（Streamlit smoke）
- 啟動後確認三頁可正常切換與 rerun：
  - `熱力圖總表`
  - `資料庫檢視`
  - `新手教學`
- 檢查 query drilldown 後 `active_page` 與 hub session payload 正確更新

## 6. 實施順序（決策完成版）
1. 建立 `ui/pages` 結構、型別、registry（不搬頁面）
2. `app.py` 改接 registry（功能不變）
3. 搬第一波三頁到 `ui/pages/*`，保留 `app.py` wrapper
4. 補測試（registry + 三頁 smoke 級單測）
5. 更新 `pyproject.toml` 的 ruff/pytest 設定
6. 更新 `README.md`（依賴 SoT/requirements 生成規範）
7. 精簡 `PROJECT_CONTEXT.md` 功能段並加 README 引用
8. 擴充 `AGENTS.md` 的 `Known Pitfalls`
9. 全量驗證與 changelog 記錄

## 7. 假設與預設（已鎖定）
- 保留單入口 `app.py`，不切到 Streamlit 原生多頁機制
- 不改任何交易規則（`T+1`）
- 不改既有 session keys 與資料表 schema
- 第一波只搬低耦合頁面，`回測工作台/即時看盤` 延後
- Tooling 先上 `Ruff + Pytest`，`Black` 不在本輪強制導入
