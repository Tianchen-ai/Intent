#!/usr/bin/env python3
"""Install an Intent wheel or build from source, with one selected backend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
# This declaration module uses only the standard library. Loading it does not
# import Intent or require any backend before the environment has been created.
DECLARATIONS = runpy.run_path(str(REPOSITORY / "python/intent/tools/backends.py"))
BACKENDS = DECLARATIONS["BACKENDS"]


def run(*command: str | Path, env: dict[str, str]) -> None:
    arguments = [str(value) for value in command]
    print("+ " + shlex.join(arguments), flush=True)
    subprocess.run(arguments, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--venv", type=Path, help="Default: .venv-BACKEND")
    parser.add_argument("--wheel", type=Path, help="Install an existing wheel without an LLVM/MLIR build SDK")
    parser.add_argument("--mlir-dir", type=Path, help="Source build: MLIR CMake package directory")
    parser.add_argument("--llvm-dir", type=Path, help="Source build: LLVM CMake package directory")
    parser.add_argument("--runtime-notices", type=Path, help="Source build: JSON library-to-notice mapping for custom SDKs")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--examples", action="store_true", help="Install portable NumPy/BF16/FP8 inputs for the thirty public programs")
    parser.add_argument("--torch-index-url", help="Override the selected backend's PyTorch wheel index")
    parser.add_argument("--provider-compiler", type=Path, help="Existing Mojo, Weft, or BANG C compiler")
    parser.add_argument("--weft-source-dir", type=Path)
    parser.add_argument("--weft-binary-dir", type=Path)
    parser.add_argument("--neuware", type=Path, help="Existing NeuWare SDK for BANG C")
    args = parser.parse_args()
    selected = BACKENDS[args.backend]
    if sys.platform != "linux":
        parser.error("the packaged backend setup currently supports Linux")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if (args.weft_source_dir or args.weft_binary_dir) and args.backend != "weft":
        parser.error("Weft source/build flags require --backend weft")
    if args.neuware and args.backend != "bangc":
        parser.error("--neuware requires --backend bangc")
    if args.torch_index_url and selected["torch"] is None:
        parser.error("this backend does not install PyTorch")
    if args.wheel is not None:
        if any((args.mlir_dir, args.llvm_dir, args.runtime_notices, args.build_dir,
                args.weft_source_dir, args.weft_binary_dir)):
            parser.error("--wheel cannot be combined with source-build SDK, build, notice, or Weft source flags")
        args.wheel = args.wheel.expanduser().resolve()
        if not args.wheel.is_file() or args.wheel.suffix != ".whl":
            parser.error(f"Wheel file not found: {args.wheel}")
    else:
        args.mlir_dir = (args.mlir_dir or Path(os.environ.get("INTENT_MLIR_DIR", "/usr/lib/llvm-20/lib/cmake/mlir"))).expanduser().resolve()
        args.llvm_dir = (args.llvm_dir or Path(os.environ.get("INTENT_LLVM_DIR", "/usr/lib/llvm-20/lib/cmake/llvm"))).expanduser().resolve()
        for directory, filename in ((args.mlir_dir, "MLIRConfig.cmake"), (args.llvm_dir, "LLVMConfig.cmake")):
            if not (directory / filename).is_file():
                parser.error(f"{filename} not found in {directory}; select an LLVM/MLIR 20 build SDK or --wheel")
    options = {}
    extra_cmake = []
    if args.backend == "mojo":
        compiler = args.provider_compiler or os.environ.get("INTENT_MOJO")
        if compiler is None:
            parser.error("Mojo must already be installed; provide --provider-compiler or INTENT_MOJO")
        resolved = shutil.which(os.path.expanduser(str(compiler)))
        if resolved is None:
            parser.error(f"Mojo compiler is not executable: {compiler}")
        options["compiler"] = str(Path(resolved).resolve())
    elif args.backend == "weft":
        if args.wheel is None:
            if args.weft_source_dir is None or args.weft_binary_dir is None:
                parser.error("Weft source builds require --weft-source-dir and --weft-binary-dir matching the MLIR SDK")
            source, build = args.weft_source_dir.expanduser().resolve(), args.weft_binary_dir.expanduser().resolve()
            for path in (source / "include/Weft/Dialect/Kernel/IR/KernelDialect.h",
                         build / "include/Weft/Dialect/Kernel/IR/KernelOps.h.inc"):
                if not path.is_file():
                    parser.error(f"Weft source/build is incomplete: {path}")
            extra_cmake = [f"--config-settings=cmake.define.INTENT_WEFT_SOURCE_DIR={source}",
                           f"--config-settings=cmake.define.INTENT_WEFT_BINARY_DIR={build}"]
        if args.provider_compiler and shutil.which(os.path.expanduser(str(args.provider_compiler))) is None:
            parser.error(f"Weft compiler is not executable: {args.provider_compiler}")
    elif args.backend == "bangc":
        if args.neuware is None:
            parser.error("BANG C requires an existing SDK selected by --neuware")
        sdk = args.neuware.expanduser().resolve()
        compiler = args.provider_compiler or sdk / "bin/cncc"
        resolved = shutil.which(os.path.expanduser(str(compiler)))
        if resolved is None or not (sdk / "lib64/libcnrt.so").is_file():
            parser.error("NeuWare must supply an executable cncc and lib64/libcnrt.so")
        options.update(neuware=str(sdk), compiler=str(Path(resolved).resolve()))
    elif args.provider_compiler or args.neuware or args.weft_source_dir or args.weft_binary_dir:
        parser.error("external compiler/SDK flags apply only to their selected CPU or MLU backend")
    destination = (args.venv or Path(f".venv-{args.backend}")).expanduser().resolve()
    if destination.exists() and not (destination / "pyvenv.cfg").is_file():
        parser.error(f"{destination} exists but is not a virtual environment")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("INTENT_COMPILER", None)
    env.pop("INTENT_OPTIMIZER", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    if not destination.exists():
        run(sys.executable, "-m", "venv", destination, env=env)
    python = destination / "bin/python"
    interpreter = json.loads(subprocess.check_output(
        [str(python), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"], text=True, env=env))
    if not (3, 10) <= tuple(interpreter) <= selected["python_max"]:
        parser.error("the public backend dependency routes currently require Python 3.10–3.12")
    run(python, "-m", "pip", "install", "--upgrade", "pip", env=env)
    extras = "[manual,examples]" if args.examples else "[manual]"
    if args.wheel is not None:
        run(python, "-m", "pip", "install", str(args.wheel) + extras, env=env)
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        build = (args.build_dir or cache / "intentdsl/install-build" / args.backend / sys.implementation.cache_tag).resolve()
        if args.runtime_notices is not None:
            extra_cmake.append("--config-settings=cmake.define.INTENT_RUNTIME_NOTICES=" + str(args.runtime_notices.expanduser().resolve()))
        run(python, "-m", "pip", "install", str(REPOSITORY) + extras,
            "--config-settings=cmake.define.MLIR_DIR=" + str(args.mlir_dir),
            "--config-settings=cmake.define.LLVM_DIR=" + str(args.llvm_dir),
            "--config-settings=build-dir=" + str(build), *extra_cmake, env=env)
    command = [str(destination / "bin/intent"), "setup", "--target", args.backend]
    if args.torch_index_url:
        command.extend(("--torch-index-url", args.torch_index_url))
    run(*command, env=env)
    print(f"Installed {args.backend}. Activate: source {shlex.quote(str(destination / 'bin/activate'))}")
    if args.backend == "weft":
        run(python, "-c", "import sys; from intent.compiler.toolchain import compiler_info; "
            "facts = compiler_info(); "
            "print(facts); "
            "facts['providers']['weft']['available'] or "
            "sys.exit('The installed Intent compiler was built without Weft support')", env=env)
        print("Weft is available for source generation. Supply your actual vector_bits/workers to intent doctor/compile;")
        print("native RISC-V execution additionally requires an explicit deployment and matching external toolchain.")
    else:
        command = [str(destination / "bin/intent"), "doctor", "--target", args.backend]
        for key, value in options.items():
            command.extend(("--target-option", f"{key}={json.dumps(value)}"))
        run(*command, env=env)
        if options:
            print("Use the same --target-option settings in subsequent CLI calls, or the corresponding Target constructor arguments.")
    print(f"Manual MCP: {destination / 'bin/intent-manual'}")
    print(f"Optional compiler MCP: {destination / 'bin/intent-compiler-mcp'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
