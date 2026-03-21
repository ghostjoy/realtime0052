# `serve`

用途：啟動本機 Streamlit Web UI。

## Help

```bash
uv run realtime0052 serve --help
```

## 範例

```bash
uv run realtime0052 serve
```

## 行為

- 內部會執行 `python -m streamlit run app.py`
- 適合本機開發或手動驗證 UI
- 這是長駐程序，會持續佔用終端機直到停止
