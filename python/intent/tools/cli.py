from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .backends import BACKENDS
from .compilation import (compile_request, describe, doctor, generate_ir_request,
                          materialize_request, optimize_request, read_artifact)


def _assignments(parser, values: list[str]) -> dict:
    result = {}
    for item in values:
        name, separator, encoded = item.partition("=")
        if not separator or not name.isidentifier() or name in result:
            parser.error(f"expected a distinct NAME=JSON assignment, got {item!r}")
        try:
            result[name] = json.loads(encoded)
        except json.JSONDecodeError as error:
            parser.error(f"invalid JSON value for {name}: {error}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up an Intent environment, inspect its tools, compile a kernel or transform existing IR")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("doctor", "Check the base compiler, or selected backend dependencies and target facts"),
                            ("compile", "Compile a Python file:kernel or importable.module:kernel"),
                            ("generate-ir", "Generate provider source from existing KIR or shared IR")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--target", choices=BACKENDS, required=name == "generate-ir")
        command.add_argument("--target-option", action="append", default=[], metavar="NAME=JSON")
        command.add_argument("--compiler", help="Intent compiler override")
        command.add_argument("--target-facts", type=Path,
                             help="JSON object containing the compiler's target facts; generate without probing a local device or SDK")
        command.add_argument("--json", action="store_true", help="Write a structured result")
    compile_parser = commands.choices["compile"]
    compile_parser.add_argument("program", metavar="PROGRAM:KERNEL")
    compile_parser.add_argument("--constexpr", action="append", default=[], metavar="NAME=JSON")
    compile_parser.add_argument("--tuning-config")
    compile_parser.add_argument("--numerics", choices=("source", "relaxed_normalization"),
                                help="Numerical permission: source retains the language contract; relaxed_normalization permits summary rescaling and requires finite valid scores and all values entering the moment contraction, including zero-weight terms")
    compile_parser.add_argument("--online-reduction", choices=("true", "false"),
                                help="Consider online reduction restructuring; enabling it does not grant numerical permission")
    compile_parser.add_argument("--optimization-remarks", choices=("true", "false"),
                                help="Emit optimization decision remarks into compiler diagnostics")
    compile_parser.add_argument("--stage", choices=("kir", "shared", "provider"), default="provider",
                                help="Stop at verified KIR, shared physical IR, or generated provider source")
    compile_parser.add_argument("--materialize", action="store_true", help="Also create the callable; never launch the kernel")
    compile_parser.add_argument("--export", dest="export_directory", help="Save the generated program in a new portable directory")
    generate_parser = commands.choices["generate-ir"]
    generate_parser.add_argument("ir_file", metavar="INPUT.mlir")
    generate_parser.add_argument("--input-stage", choices=("kir", "shared"), default="shared")
    generate_parser.add_argument("--name", required=True, help="Diagnostic program identifier; does not select a kernel")
    generate_parser.add_argument("--materialize", action="store_true", help="Also create the callable without launching")
    generate_parser.add_argument("--export", dest="export_directory", help="Save the generated program in a new portable directory")
    materialize_parser = commands.add_parser("materialize", help="Load a saved generated program and bind an explicitly selected runtime")
    materialize_parser.add_argument("directory")
    materialize_parser.add_argument("--target", choices=BACKENDS, required=True)
    materialize_parser.add_argument("--target-option", action="append", default=[], metavar="NAME=JSON")
    materialize_parser.add_argument("--json", action="store_true")
    optimize_parser = commands.add_parser("optimize", help="Run a standard MLIR pass pipeline on existing IR")
    optimize_parser.add_argument("ir_file", metavar="INPUT.mlir")
    optimize_parser.add_argument("--pipeline", required=True, help="Standard MLIR pass pipeline")
    optimize_parser.add_argument("--optimizer", help="intent-opt executable override")
    optimize_parser.add_argument("--json", action="store_true", help="Write a structured result")
    describe_parser = commands.add_parser("describe", help="Discover installed public API and target fields without probing SDKs or devices")
    describe_parser.add_argument("--target", choices=BACKENDS)
    describe_parser.add_argument("--json", action="store_true")
    setup_parser = commands.add_parser("setup", help="Install one backend's Python dependencies in this Python environment")
    setup_parser.add_argument("--target", choices=BACKENDS, required=True)
    setup_parser.add_argument("--torch-index-url", help="Override the selected route's PyTorch wheel index")
    setup_parser.add_argument("--json", action="store_true")
    read_parser = commands.add_parser("read-artifact", help="Read an explicit source, IR, metadata or log file without execution")
    read_parser.add_argument("path")
    read_parser.add_argument("--offset", type=int, default=0, help="Unicode character offset")
    read_parser.add_argument("--limit", type=int, default=16000, help="Maximum characters, from 1 to 64000")
    read_parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "setup":
        from .installation import setup_backend

        result = setup_backend(arguments.target, torch_index_url=arguments.torch_index_url)
    elif arguments.command == "describe":
        result = describe(arguments.target)
    elif arguments.command == "read-artifact":
        result = read_artifact(arguments.path, offset=arguments.offset, limit=arguments.limit)
    elif arguments.command == "optimize":
        result = optimize_request(arguments.ir_file, arguments.pipeline, optimizer=arguments.optimizer)
    else:
        options = _assignments(parser, arguments.target_option)
        facts = None
        if arguments.command != "materialize" and arguments.target_facts is not None:
            try:
                facts = json.loads(arguments.target_facts.read_text(encoding="utf-8"))
                if not isinstance(facts, dict):
                    raise ValueError("target facts must be a JSON object")
            except (OSError, UnicodeError, ValueError) as error:
                parser.error(str(error))
        if arguments.command == "materialize":
            result = materialize_request(arguments.directory, arguments.target, target_options=options)
        elif arguments.command == "doctor":
            result = doctor(arguments.target, target_options=options, compiler=arguments.compiler, target_facts=facts)
        elif arguments.command == "generate-ir":
            result = generate_ir_request(arguments.ir_file, arguments.name, arguments.target,
                                         input_stage=arguments.input_stage, target_options=options,
                                         compiler=arguments.compiler, materialize=arguments.materialize,
                                         target_facts=facts, export_directory=arguments.export_directory)
        else:
            program, separator, kernel = arguments.program.rpartition(":")
            if not separator or not program:
                parser.error("program must be a Python file:kernel or importable.module:kernel")
            compile_options = {}
            if arguments.numerics is not None:
                compile_options["numerics"] = arguments.numerics
            for name in ("online_reduction", "optimization_remarks"):
                value = getattr(arguments, name)
                if value is not None:
                    compile_options[name] = value == "true"
            result = compile_request(program, kernel, arguments.target, target_options=options,
                                     constexprs=_assignments(parser, arguments.constexpr),
                                     compiler=arguments.compiler, tuning_config=arguments.tuning_config,
                                     materialize=arguments.materialize, stage=arguments.stage,
                                     target_facts=facts, export_directory=arguments.export_directory,
                                     options=compile_options or None)
    if arguments.json or arguments.command == "describe":
        print(json.dumps(result, indent=2))
    elif arguments.command == "setup":
        print(f"{arguments.target}: {result['status']} in {result['python']}")
        if "diagnostic" in result:
            print(result["diagnostic"]["message"], file=sys.stderr)
        print(result["scope"])
    elif arguments.command == "doctor":
        print(f"{arguments.target or 'base compiler'}: {result['status']}")
        for check in result["checks"]:
            detail = check["diagnostic"]["message"] if "diagnostic" in check else check.get("detail")
            print(f"  {check['name']}: {check['status']} — {detail}")
        print(result["scope"])
    elif arguments.command == "read-artifact":
        if result["status"] == "read":
            print(result["text"], end="")
            if not result["eof"]:
                print(f"\nContinue with --offset {result['next_offset']}", file=sys.stderr)
        else:
            print(f"Stage: {result['stage']}\n{result['diagnostic']['message']}", file=sys.stderr)
    else:
        if arguments.command == "optimize":
            print(f"{arguments.ir_file}: {result['status']} with {arguments.pipeline}")
        elif arguments.command == "generate-ir":
            print(f"{arguments.ir_file}: {result['status']} for {arguments.target}")
        elif arguments.command == "materialize":
            print(f"{arguments.directory}: {result['status']} for {arguments.target}")
        else:
            scope = arguments.target or arguments.stage
            print(f"{arguments.program}: {result['status']} for {scope}")
        if "diagnostic" in result:
            print(f"Stage: {result['stage']}\n{result['diagnostic']['message']}", file=sys.stderr)
        for name, path in result.get("files", {}).items():
            print(f"  {name}: {path}")
        if "export_directory" in result:
            print(f"  exported program: {result['export_directory']}")
        if result.get("program_stdout"):
            print(result["program_stdout"], file=sys.stderr, end="")
        if result.get("program_stderr"):
            print(result["program_stderr"], file=sys.stderr, end="")
        if arguments.command == "compile":
            print("The tool did not launch the selected kernel; module-level Python code executes normally.")
        else:
            print("The tool did not launch a kernel.")
        print("Compilation does not establish numerical correctness.")
    if result["status"] in {"error", "unavailable"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
