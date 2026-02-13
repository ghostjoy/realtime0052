# Fugle MCP 協作設定指南

這份文件提供本專案與 `fugle-marketdata-mcp-server` 的最小可行設定，讓你在不同 MCP client（例如 Claude Desktop / 其他支援 MCP 的 Agent）使用同一組配置。

## 1) 先決條件

- 已安裝 Node.js（含 `npx`）
- 已申請 Fugle MarketData API Key
- 本 repo 已有腳本：`scripts/run_fugle_mcp_server.sh`

## 2) 設定 API Key（本機）

建議把 key 放在 shell 環境變數，不要寫進 repo：

```bash
export FUGLE_MARKETDATA_API_KEY="your_fugle_key"
```

可放進 `~/.zshrc`：

```bash
echo 'export FUGLE_MARKETDATA_API_KEY="your_fugle_key"' >> ~/.zshrc
source ~/.zshrc
```

> `scripts/run_fugle_mcp_server.sh` 會自動把 `FUGLE_MARKETDATA_API_KEY` 映射為 Fugle MCP 需要的 `API_KEY`。

## 3) 在 MCP Client 設定 server

### 通用設定（推薦）

把 command 指向 repo 內腳本：

```json
{
  "mcpServers": {
    "fugle-marketdata": {
      "command": "/Users/ztw/codexapp/realtime0052/scripts/run_fugle_mcp_server.sh"
    }
  }
}
```

### 若 client 不會繼承 shell 環境

在設定中顯式帶入 `API_KEY`：

```json
{
  "mcpServers": {
    "fugle-marketdata": {
      "command": "/Users/ztw/codexapp/realtime0052/scripts/run_fugle_mcp_server.sh",
      "env": {
        "API_KEY": "your_fugle_key"
      }
    }
  }
}
```

## 4) 快速驗證

你可以先在終端直接啟動（成功會進入 MCP stdio wait 狀態）：

```bash
cd /Users/ztw/codexapp/realtime0052
FUGLE_MARKETDATA_API_KEY="your_fugle_key" ./scripts/run_fugle_mcp_server.sh
```

## 5) 安全建議

- 不要把 API key commit 到 git。
- 若 key 曾貼在聊天紀錄或公開位置，建議到 Fugle 後台立即重發新 key。
- 多台機器共用時，優先用各機器環境變數管理，不要寫死在設定檔。

## 6) 參考

- Fugle LLM 開發頁：<https://developer.fugle.tw/docs/data/build-with-llm/>
- Fugle `llms.txt`：<https://developer.fugle.tw/llms.txt>
- Fugle MCP Server repo：<https://github.com/fugle-dev/fugle-marketdata-mcp-server>
