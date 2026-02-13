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
- 新增 `universe_snapshots`（SQLite）用於快取成分股清單，避免每次切分頁都重抓。
- 新增成分股快取讀寫 API：
  - `save_universe_snapshot(...)`
  - `load_universe_snapshot(...)`

### Changed
- 主介面分頁調整為：`即時看盤 / 回測工作台 / 00935 熱力圖 / 新手教學`。
- 新手教學中補充「成分股快取」資料儲存位置說明。

### Docs
- `README.md` 補充 00935 熱力圖分頁與功能說明。

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
