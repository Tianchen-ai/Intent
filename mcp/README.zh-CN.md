# IntentDSL MCP 服务

[English](README.md) · [工具书](https://tianchen-ai.github.io/Intent/getting-started/mcp/)

IntentDSL 提供两个 stdio MCP 服务。实现集中在
[python/intent/mcp](../python/intent/mcp)，本目录提供接入说明和
[客户端配置](servers.json)。两个服务都随 IntentDSL wheel 安装，不需要另一个包或仓库。

| 命令 | 工具 | 用途 |
|---|---|---|
| `intent-manual` | `search`、`api`、`read` | 查询公开语法、类型、语言合同和 kernel/host 接口，不执行程序、不提供完整算法解法。 |
| `intent-compiler-mcp` | `describe`、`environment`、`compile`、`generate_from_ir`、`optimize`、`materialize`、`read_artifact` | 对明确提供的程序/IR 路径调用与 CLI 相同的公开编译流程。需要编译工具时启用。 |

手册语料来自安装包中的语言文档与公开 Python 声明。编译服务复用
[tools/compilation.py](../python/intent/tools/compilation.py) 和共享 worker，不维护第二套编译器。

## 安装

从[公开 preview](https://github.com/Tianchen-ai/Intent/releases/tag/preview) 下载 Linux wheel，将 `INTENT_WHEEL` 设为实际本地 `.whl` 路径：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "${INTENT_WHEEL}[manual]"
.venv/bin/intent-manual --help
.venv/bin/intent-compiler-mcp --help
```

`manual` extra 安装两个 MCP 服务共同需要的依赖。查询手册和公开 API 不需要 GPU 或 provider SDK；生成后端源码和物化 native 程序各有自己的前置条件，见[安装指南](../environment/README.zh-CN.md)。

从源码安装时，准备好 LLVM/MLIR 构建依赖后，可使用现有脚本选择后端：

```bash
python3 environment/install.py --backend triton --venv .venv-triton --examples
```

该命令也会安装 MCP extra。GitHub preview 附件可公开下载；目前尚未发布到 PyPI，请使用上述 wheel 或源码安装路线。

## 接入客户端

支持 `mcpServers` JSON 格式的客户端，可复制 [servers.json](servers.json) 中的条目，并把 command 换成安装环境中命令的绝对路径。只需查询语言时，使用 `intent_manual` 条目即可。

客户端启动命令后，通过 stdin/stdout 通信，不需要端口、API key 或后台服务。使用 TOML 等其它格式的客户端，同样配置这两个命令和空参数列表。

```json
{
  "mcpServers": {
    "intent_manual": {
      "command": "/absolute/path/to/.venv/bin/intent-manual",
      "args": []
    },
    "intent_compiler": {
      "command": "/absolute/path/to/.venv/bin/intent-compiler-mcp",
      "args": []
    }
  }
}
```

## 使用工具

用 `api(name="intent.compile")` 或 `api(name="I.domain")` 查询准确声明。通过 `search(query=..., kind=...)` 找到规则 ID，再用 `read(id=...)` 阅读。手册只包含合同与最小语法片段，完整程序保存在 [examples](../examples/README.md)。

需要编译时，先用 `describe` 查询公开接口，用 `environment` 检查前置条件。`compile` 需要已有 `program_path` 和 kernel 名称；`kir` 阶段不需要 target 或设备，`shared` 和 `provider` 阶段必须指定 target。导入 Python 程序会执行普通顶层 host 代码；返回的产物路径可以用 `read_artifact` 查看。

源码生成、native 编译与 kernel 执行是不同步骤。这些工具不启动选中的 kernel，也不证明数值正确或性能达标。真实调用继续使用公开 runtime 或既有完整 program 入口。

IntentDSL 当前使用仍受支持的 Python SDK v1 系列（`mcp>=1.28,<2`）；客户端传输细节见[官方 SDK v1 文档](https://py.sdk.modelcontextprotocol.io/v1/)。
