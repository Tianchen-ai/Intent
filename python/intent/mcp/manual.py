"""Read-only stdio MCP transport for the installed Intent manual."""
import argparse
import json
from pathlib import Path

from ..tools.manual import Manual, installed_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Intent author manual MCP (requires mcp>=1.28,<2)")
    parser.add_argument("--corpus", type=Path,
                        help="Use an explicit frozen public material JSON instead of the installed manual")
    arguments = parser.parse_args()
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        parser.error("Install IntentDSL with its 'manual' extra (from a checkout: pip install '.[manual]').")

    manual = Manual(json.loads(arguments.corpus.read_text(encoding="utf-8"))
                    if arguments.corpus is not None else installed_corpus())
    host_instructions = (
        "Call api(name='intent.compile') or api(name='intent.generate') for the current public host interfaces, "
        "and search(query='Target', kind='api') for target declarations. "
        if arguments.corpus is None else
        "Use api(name=...) for the host interface declared by this explicit corpus. "
    )
    server = FastMCP("intent_manual", instructions=(
        "Read-only Intent language and kernel/host contracts. "
        "Start with read(id='doc/dsl/authoring.md'). "
        + host_instructions +
        "Call api(name='I.domain') for exact DSL declarations and rule IDs; "
        "read(id=...) for syntax, types, semantics and callable interface rules. "
        "read(id=..., section=...) accepts an exact section title or its published heading line. "
        "Use search(query=..., kind=...) to find names and IDs. "
        "No execution or task answers."
    ))
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    for method in (manual.search, manual.api, manual.read):
        server.add_tool(method, annotations=annotations)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
