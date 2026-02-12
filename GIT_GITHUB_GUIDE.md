# Git / GitHub 新手快速筆記

這份文件是針對你目前專案（`ghostjoy/realtime0052`）整理的可回看版本。

## 1) 先理解三個東西

- `commit`：一次版本快照（可想成存檔點）
- `branch`：指向某個 commit 的「指標名稱」（例如 `main`、`ui-layout-polish`）
- `remote`：遠端倉庫（通常叫 `origin`，也就是 GitHub 上那份）

## 2) 你剛剛的實際狀態（範例）

當時狀態大意是：

- 本地 `ui-layout-polish` 指到新 commit（例如 `e215f5b`）
- 遠端 `origin/main` 也被推到同一個 commit
- 本地 `main` 還停在舊 commit（例如 `8c0e472`）

所以 GitHub 上其實已經是最新，但你本機的 `main` 還沒跟上。

## 3) `Fast-forward` 是什麼？

`fast-forward` = 分支指標「直接往前移」，**不新增 merge commit**。

重點：

- 不是「只更新主功能，不更新 branch 功能」
- 也不是「刪掉 branch 內容」
- 它只是把某個分支（例如本地 `main`）移到較新的 commit

## 4) 為什麼會想用 `fast-forward`？

- 歷史更乾淨（沒有多餘 merge 節點）
- 可讀性高（線性 history）
- 在「main 沒分叉」的情況下最自然

## 5) 常見指令對照（這次會用到）

```bash
# 看目前在哪個分支
git branch --show-current

# 看本地分支與遠端追蹤狀態
git branch -vv

# 看工作區有沒有未提交
git status --short

# 把目前分支的 HEAD 推到遠端 main
git push origin HEAD:main
```

## 6) 把本地 `main` 同步到遠端最新（Fast-forward）

```bash
git checkout main
git fetch origin
git merge --ff-only origin/main
```

如果成功，表示本地 `main` 直接移到 `origin/main` 最新點。

## 7) 一個好記的心智模型

- `main` / `ui-layout-polish`：只是「書籤名稱」
- commit：實際內容
- `push`：把你本地書籤指向的 commit，更新到遠端書籤
- `fast-forward`：書籤往前移，不創造新節點

## 8) 新手建議工作流（安全版）

1. 在功能分支開發（例如 `ui-layout-polish`）
2. `git status` 確認乾淨後 commit
3. push 到 GitHub
4. 確認遠端 `main` 是否已更新
5. 最後把本地 `main` fast-forward 到最新

---

如果你要，我可以再補一份「圖解版」（ASCII 圖）放在同檔案，幫你一眼看懂分支移動。
