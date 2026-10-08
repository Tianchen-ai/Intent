# IntentDSL MCP servers

[中文](README.zh-CN.md) · [User handbook](https://tianchen-ai.github.io/Intent/en/getting-started/mcp/)

IntentDSL provides two stdio MCP servers. Their implementation lives in
[python/intent/mcp](../python/intent/mcp); this directory contains connection
instructions and the [client configuration](servers.json). Both servers ship
inside the IntentDSL wheel, so they do not need a separate package or checkout.

| Command | Tools | Purpose |
|---|---|---|
| `intent-manual` | `search`, `api`, `read` | Read public syntax, types, language contracts and kernel/host interfaces. It does not execute programs or provide complete algorithm solutions. |
| `intent-compiler-mcp` | `describe`, `environment`, `compile`, `generate_from_ir`, `optimize`, `materialize`, `read_artifact` | Work with explicitly supplied program/IR paths through the same public pipeline as the CLI. Enable this server when compilation tools are wanted. |

The manual corpus comes from the installed language documents and public Python
declarations. The compiler server uses [tools/compilation.py](../python/intent/tools/compilation.py)
and its shared worker; it does not maintain a second compiler implementation.

## Install

Download the Linux wheel from the [public preview](https://github.com/Tianchen-ai/Intent/releases/tag/preview). Set `INTENT_WHEEL` to its actual local `.whl` path:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "${INTENT_WHEEL}[manual]"
.venv/bin/intent-manual --help
.venv/bin/intent-compiler-mcp --help
```

The `manual` extra installs the dependencies for both servers. Reading the
manual and public API does not require a GPU or a provider SDK. Provider source
generation and native materialization have their own prerequisites, described
in the [installation guide](../environment/README.md).

From a source checkout with the LLVM/MLIR build prerequisites, the existing
installer can also set up a selected backend:

```bash
python3 environment/install.py --backend triton --venv .venv-triton --examples
```

This command installs the MCP extra too. GitHub preview artifacts are public
downloads; there is no PyPI release yet, so use the wheel or source route above.

## Connect a client

Copy the server entries from [servers.json](servers.json) into a client that
accepts the `mcpServers` configuration format. Replace each command with the
absolute path inside the environment where IntentDSL is installed. If only
language lookup is needed, use the `intent_manual` entry.

The client launches the commands and communicates over stdin/stdout; no port,
API key or background service is needed. Other clients may use TOML or another
configuration format: keep the same command and empty argument list.

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

## Use the tools

Start with `api(name="intent.compile")` or `api(name="I.domain")` for an exact
declaration. Use `search(query=..., kind=...)` to find rule IDs, then
`read(id=...)` to read them. The manual contains contracts and minimal syntax,
with full programs kept in [examples](../examples/README.md).

For compiler work, use `describe` to discover public signatures and
`environment` to inspect prerequisites. `compile` requires an existing
`program_path` and kernel name. Stage `kir` needs no target or device; `shared`
and `provider` require an explicit target. Importing a Python program executes
its ordinary top-level host code. Returned artifact paths can be inspected with
`read_artifact`.

Source generation, native compilation and kernel execution are distinct.
These tools do not launch the selected kernel or establish numerical or
performance correctness. Actual calls use the public runtime or the existing
complete program entry.

IntentDSL currently uses the supported Python SDK v1 line (`mcp>=1.28,<2`).
See the [official SDK v1 documentation](https://py.sdk.modelcontextprotocol.io/v1/)
for client transport details.
