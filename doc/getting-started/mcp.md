# Agent 与 MCP

IntentDSL 提供两个独立的 stdio MCP 服务。公开接入说明与客户端配置放在仓库的 [`mcp/`](https://github.com/Tianchen-ai/Intent/tree/main/mcp)，传输实现集中在 `python/intent/mcp/`。

## 语言手册

```bash
intent-manual
```

手册服务只读，提供语法、类型、操作与接口规则的查询；使用 `search` 找概念、`api` 查声明、`read` 读规则。它不执行程序，不提供完整算法、评测题解法、reference 源码或针对性调优建议。公开声明也不意味着所有 target 已实现。

## 编译工具

```bash
intent-compiler-mcp
```

编译服务提供公开 API 发现、环境检查、编译已有用户 `.py` 文件、IR 优化及分页读取产物/日志。它与 CLI 使用同一编译实现，必须由客户端显式启用。加载模块会执行普通顶层 Python host code；生成源码、native materialization、kernel launch 与数值验证分别发生，MCP 编译不自动执行 kernel。

## 客户端接入

安装 `manual` extra 后，在支持 MCP 的客户端中配置已安装环境的绝对可执行路径：

```json
{
  "mcpServers": {
    "intent-manual": {
      "command": "/absolute/path/to/venv/bin/intent-manual"
    }
  }
}
```

只读手册不需要 GPU SDK 或 PyTorch。编译服务需要可启动的 Intent 编译器，所选后端的 prerequisites 由环境查询说明。完整双语接入及编译服务配置见[仓库 MCP 指南](https://github.com/Tianchen-ai/Intent/tree/main/mcp)。
