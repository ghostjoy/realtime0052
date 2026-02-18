# PROMPT_TEMPLATES.md

此檔案提供本 repo 可重複使用的 Prompt 範本（中文為主，附英文關鍵詞）。

## 1) 實作新功能（Implement a feature）
```
先讀 `AGENTS.md` 與 `PROJECT_CONTEXT.md`。
請實作：<功能描述>。
限制條件：<constraints>。
驗證方式：執行相關測試並回報結果。
若有行為變更，更新 `CHANGELOG.md`（Unreleased）。
```

## 2) 修正錯誤（Fix a bug）
```
先讀 `AGENTS.md` 與 `PROJECT_CONTEXT.md`。
問題現象：<symptom>。
預期行為：<expected>。
請先找 root cause，再做最小修正，最後用測試驗證。
若有行為變更，更新 `CHANGELOG.md`（Unreleased）。
```

## 3) 更新文件（Update documentation）
```
請同步 `README.md`、`PROJECT_CONTEXT.md`、`AGENTS.md`、`CHANGELOG.md` 與目前程式行為。
文案保持精簡、準確、可執行。
若修正的是教學命令，請驗證命令與路徑可直接使用。
```

## 4) 新機器 / 新對話快速接手（Onboarding）
```
請先閱讀 `README.md`、`AGENTS.md`、`PROJECT_CONTEXT.md`、`CHANGELOG.md`。
然後整理：
1) 專案在做什麼
2) 主要架構與資料流
3) 已知風險與常見故障點
4) 建議下一步（含最小可驗證動作）
```
