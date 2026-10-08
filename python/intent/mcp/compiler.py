"""Explicitly enabled compilation tools, separate from the read-only manual."""
import argparse
from pathlib import Path
from ..tools.compilation import describe as describe_interface, read_artifact as read_artifact_file
from ..tools.requests import request_async


def main() -> None:
    parser = argparse.ArgumentParser(description="Intent compiler tools over stdio MCP")
    parser.parse_args()
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        raise SystemExit("Install IntentDSL with its 'manual' extra to use the MCP servers.") from error

    server = FastMCP("intent_compiler", instructions=(
        "Use only existing Python program, IR or saved program paths explicitly supplied by the user. "
        "Loading that module executes its top-level Python host code. "
        "These tools use the public Intent pipeline and do not themselves launch the selected kernel. "
        "Generated/materialized does not mean numerical or performance validation. "
        "Use describe to discover public signatures and environment to inspect prerequisites; "
        "use read_artifact to page through returned source, IR and diagnostic files. "
        "Use the separate intent_manual server for language contracts."
    ))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def compile(program_path: str, kernel: str, target: str | None = None,
                      target_options: dict | None = None, constexprs: dict | None = None,
                      compiler: str | None = None, tuning_config: str | None = None,
                      materialize: bool = False, stage: str = "provider",
                      target_facts: dict | None = None, export_directory: str | None = None,
                      options: dict | None = None) -> dict:
        """Compile an existing .py file and report stages/artifacts.

        stage='kir' needs no target, provider SDK or device. 'shared' and
        'provider' need a target; only 'provider' permits materialize=True.
        target_facts accepts the compiler's explicit target object for offline
        generation. export_directory saves source, final IR and metadata for a
        later host; target_options select the local runtime if materializing.
        The file's ordinary top-level Python host code executes normally.
        options accepts numerics ('source' or 'relaxed_normalization'),
        online_reduction (bool), and optimization_remarks (bool). Source includes
        the language's FMA/reduction permissions, not bitwise reproducibility.
        Relaxed normalization permits normalized-summary rescaling and movement
        of low-precision weight casts. Valid-member scores and all values entering
        the moment contraction must be finite, including zero-weight terms;
        zero times infinity is not an inactive access. Masked accesses keep fills.
        It can change rounding/underflow/overflow but does not enable global fast
        math or FTZ. The online switch controls optimization, not permission.
        """
        return await request_async("compile", dict(
            program=str(Path(program_path).expanduser().absolute()), kernel=kernel,
            target=target, target_options=target_options,
            constexprs=constexprs, compiler=compiler, tuning_config=tuning_config,
            materialize=materialize, stage=stage, target_facts=target_facts,
            export_directory=export_directory, options=options))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def generate_from_ir(ir_file: str, name: str, target: str,
                               input_stage: str = "shared", target_options: dict | None = None,
                               compiler: str | None = None, materialize: bool = False,
                               target_facts: dict | None = None, export_directory: str | None = None) -> dict:
        """Generate provider source from explicit existing KIR or shared IR.

        Supply the input stage, target and diagnostic program name explicitly.
        The whole module is compiled; name does not select a kernel. Callable
        entries and candidates are determined by the IR and its metadata.
        Shared input retains its physical program/configuration and must agree
        with the selected target's capabilities. Its compile options are preserved
        from IR, not reselected by this tool. This does not launch a kernel.
        """
        return await request_async("generate_from_ir", dict(
            ir_file=ir_file, name=name, target=target, input_stage=input_stage,
            target_options=target_options, compiler=compiler, materialize=materialize,
            target_facts=target_facts, export_directory=export_directory))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def materialize(program_directory: str, target: str, target_options: dict | None = None) -> dict:
        """Load a generated program from an explicit directory and bind a matching local runtime.

        Target capabilities must agree with the saved compiler facts. This may
        compile/load native code, but neither recompiles KIR nor launches a kernel.
        """
        return await request_async("materialize", dict(
            directory=program_directory, target=target, target_options=target_options))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def optimize(ir_file: str, pipeline: str, optimizer: str | None = None) -> dict:
        """Run a standard MLIR pipeline on an existing IR file supplied by the user.

        The installed intent-opt applies the requested passes and reports output
        and diagnostic paths. Pass prerequisites belong to the input/current IR;
        this does not materialize or launch a kernel or establish correctness.
        """
        return await request_async("optimize", dict(ir_file=ir_file, pipeline=pipeline, optimizer=optimizer))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    async def environment(target: str | None = None, target_options: dict | None = None,
                          compiler: str | None = None, target_facts: dict | None = None) -> dict:
        """Inspect base compiler/KIR, or a selected provider's prerequisites.

        Omit target to check the compiler without any provider SDK or device.
        Explicit facts skip local SDK/device probing. Otherwise inspect only the
        selected runtime dependencies. This does not establish numerical
        correctness or device execution, or support for a particular program.
        """
        return await request_async("environment", dict(
            target=target, target_options=target_options, compiler=compiler, target_facts=target_facts))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    async def describe(target: str | None = None) -> dict:
        """Discover public API, compile options and backend constructor fields.

        Declarations are read from the installed Python API. This neither probes
        a device nor claims that the selected native compiler supports a program.
        """
        return describe_interface(target)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    async def read_artifact(path: str, offset: int = 0, limit: int = 16000) -> dict:
        """Read an explicit source, IR, metadata or log path returned by compilation.

        This reads UTF-8 text only. offset and limit count characters, with limit
        at most 64000. Continue at next_offset until eof. No program is executed.
        """
        return read_artifact_file(path, offset=offset, limit=limit)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
