# Git / GitHub 新手快速筆記

這份文件提供本 repo 可重複使用的 Git 基本流程，避免綁定單次對話情境。

## 1) 三個核心概念

- `commit`：一次版本快照（存檔點）
- `branch`：指向某個 commit 的指標名稱（例如 `main`、`feature/<topic>`）
- `remote`：遠端倉庫（通常是 `origin`）

## 2) 常用檢查指令

```bash
# 目前分支
git branch --show-current

# 本地分支與遠端追蹤狀態
git branch -vv

# 工作區是否乾淨
git status --short

# 最近提交（簡版）
git log --oneline -n 5
```

## 3) `fast-forward` 是什麼？

`fast-forward` = 分支指標直接往前移，不新增 merge commit。

適用情境：
- `main` 沒有額外分叉
- 只想把本地 `main` 對齊 `origin/main`

## 4) 同步本地 `main`（安全版）

```bash
git checkout main
git fetch origin
git merge --ff-only origin/main
```

若成功，表示本地 `main` 已線性對齊遠端最新版本。

## 5) 推送當前分支到遠端 `main`（明確指定）

```bash
git push origin HEAD:main
```

這個指令代表：把你「目前所在分支」的最新 commit，推到遠端 `main`。

## 6) 建議工作流（新手安全版）

1. 從 `main` 切出功能分支：`feature/<topic>`
2. 開發完成後先檢查：`git status`
3. commit 並 push 功能分支
4. 合併到遠端 `main`（PR 或你既有流程）
5. 回到本地 `main` 做 `fast-forward` 同步

## 7) 範例（占位符）

- 分支：`feature/ui-layout-polish`
- 新 commit：`<new-hash>`
- 舊 `main`：`<old-hash>`

只要遠端 `origin/main` 指到 `<new-hash>`，你本地 `main` 仍在 `<old-hash>`，就可以用 `merge --ff-only` 直接追上。
