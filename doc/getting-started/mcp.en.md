# Agents and MCP

IntentDSL provides two separate stdio MCP servers. Public connection instructions and client configurations live in the repository's [`mcp/`](https://github.com/Tianchen-ai/Intent/tree/main/mcp) directory; transport implementations are grouped in `python/intent/mcp/`.

## Language manual

```bash
intent-manual
```

The manual is read-only. It exposes syntax, types, operation and interface rules: `search` finds concepts, `api` retrieves declarations and `read` reads rules. It does not execute programs or provide complete algorithms, evaluation solutions, reference source or targeted tuning advice. Public declarations do not imply implementation on every target.

## Compiler tools

```bash
intent-compiler-mcp
```

The compiler server provides public API discovery, environment inspection, compilation of existing user-supplied `.py` files, IR optimization and paged artifact/log reading. It uses the same compiler implementation as the CLI and must be enabled explicitly by the client. Loading a module executes ordinary top-level Python host code. Source generation, native materialization, kernel launch and numerical validation are separate; an MCP compile call does not automatically launch a kernel.

## Connect a client

After installing the `manual` extra, configure the installed environment's absolute executable path in an MCP client:

```json
{
  "mcpServers": {
    "intent-manual": {
      "command": "/absolute/path/to/venv/bin/intent-manual"
    }
  }
}
```

The read-only manual needs no GPU SDK or PyTorch. Compiler tools need a working Intent compiler; environment discovery explains the selected backend's prerequisites. The [repository MCP guide](https://github.com/Tianchen-ai/Intent/tree/main/mcp) contains complete bilingual instructions and compiler-server configurations.
