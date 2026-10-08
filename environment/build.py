"""Build and install-check an Ubuntu 22.04 x86-64 distribution.

This recipe builds a wheel from its sdist and uses the existing public tools on
the installed package. It neither installs provider SDKs nor executes kernels.
Auditwheel verifies and repairs the wheel for manylinux_2_35_x86_64 before
installation. The recipe does not publish artifacts or claim reproducibility.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
from itertools import count
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
AUTHOR_SOURCE = PurePosixPath("examples/kernels/normalization/softmax.py")
COMMANDS = count(1)


def outside_source(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPOSITORY or REPOSITORY in resolved.parents:
        raise ValueError(f"Build and distribution paths must be outside the source tree: {resolved}")
    return resolved


def run(*command: str | Path, cwd: Path, env: dict[str, str],
        output: Path | None = None) -> str:
    arguments = [str(value) for value in command]
    print("+ " + shlex.join(arguments), flush=True)
    logs = cwd / "commands"
    logs.mkdir(exist_ok=True)
    record = logs / f"{next(COMMANDS):02d}"
    record.with_suffix(".command").write_text(shlex.join(arguments) + "\n", encoding="utf-8")
    log_path = record.with_suffix(".log")
    print(f"  log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        if output is None:
            with subprocess.Popen(arguments, cwd=cwd, env=env, text=True,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
        else:
            # Keep machine-readable stdout separate from diagnostics. These
            # public tool requests are short; compiler subprocess logs remain
            # available at the artifact paths reported in their JSON responses.
            with output.open("w", encoding="utf-8") as destination:
                returncode = subprocess.run(arguments, cwd=cwd, env=env,
                                            stdin=subprocess.DEVNULL,
                                            stdout=destination, stderr=log).returncode
            print(f"  output: {output}", flush=True)
    record.with_suffix(".status").write_text(str(returncode) + "\n", encoding="utf-8")
    if returncode:
        raise subprocess.CalledProcessError(returncode, arguments)
    return output.read_text(encoding="utf-8") if output is not None else ""


def one_artifact(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} artifact in {directory}, found {len(matches)}")
    return matches[0]


def read_source_distribution(archive: Path, destination: Path) -> tuple[str, list[str]]:
    """Keep the original production definition and package facts from the sdist."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise ValueError("The source distribution must have one top-level directory")
        root = roots.pop()

        def contents(relative: str | PurePosixPath) -> bytes:
            member = source.getmember(f"{root}/{relative}")
            if not member.isfile():
                raise ValueError(f"Expected a regular source-distribution file: {relative}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read source-distribution file: {relative}")
            with stream:
                return stream.read()

        # setuptools-scm uses PKG-INFO when rebuilding outside the Git checkout.
        metadata = BytesParser().parsebytes(contents("PKG-INFO"))
        version = metadata["Version"]
        if metadata["Name"] != "intentdsl" or not version:
            raise ValueError("The source distribution has no IntentDSL package identity")
        if not contents("LICENSE"):
            raise ValueError("The source distribution does not contain the project license")
        destination.write_bytes(contents(AUTHOR_SOURCE))
        manual = []
        for member in members:
            relative = PurePosixPath(member.name).relative_to(root)
            if relative.parts and (relative.parts[0] in {
                    ".codegraph", ".agents", ".codex", "AGENTS.md",
                    "experiments", "report", "tasks", "assets"}
                    or str(relative) == "environment/cutile.txt"):
                raise ValueError(f"Internal material is present in the source distribution: {relative}")
            if (member.isfile() and len(relative.parts) == 3
                    and relative.parts[0] == "doc"
                    and relative.parts[1] in {"dsl", "programming-model"}
                    and relative.suffix == ".md"):
                manual.append("intent/_manual/" + str(relative))
        if not manual:
            raise ValueError("The source distribution does not contain the public manual")
        return version, manual


def check_wheel_contents(wheel: Path, version: str, manual: list[str]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [name for name in archive.namelist()
                          if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("The wheel must contain exactly one package metadata file")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        if metadata["Name"] != "intentdsl" or metadata["Version"] != version:
            raise ValueError("The wheel package identity differs from its source distribution")
        if (metadata["License-Expression"] != "Apache-2.0"
                or "LICENSE" not in metadata.get_all("License-File", [])):
            raise ValueError("The wheel does not declare the project license and license file")
        license_file = metadata_files[0].removesuffix("METADATA") + "licenses/LICENSE"
        for name in ("intent/_bin/intent-compile", "intent/_bin/intent-opt",
                     "intent/_bin/third-party/libraries.json", license_file, *manual):
            if archive.getinfo(name).file_size == 0:
                raise ValueError(f"The wheel has an empty runtime resource: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Distribution destination outside the checkout; existing files are never replaced")
    parser.add_argument("--work-dir", type=Path,
                        help="New directory outside the checkout for build environments, logs and compiler artifacts")
    parser.add_argument("--mlir-dir", type=Path,
                        default=Path(os.environ.get("INTENT_MLIR_DIR", "/usr/lib/llvm-20/lib/cmake/mlir")))
    parser.add_argument("--llvm-dir", type=Path,
                        default=Path(os.environ.get("INTENT_LLVM_DIR", "/usr/lib/llvm-20/lib/cmake/llvm")))
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--runtime-notices", type=Path)
    parser.add_argument("--weft-source-dir", type=Path)
    parser.add_argument("--weft-binary-dir", type=Path)
    args = parser.parse_args()

    if (sys.platform != "linux" or platform.machine() != "x86_64"
            or sys.implementation.name != "cpython" or not (3, 10) <= sys.version_info[:2] <= (3, 12)):
        parser.error("this distribution recipe requires Linux x86_64 and CPython 3.10–3.12; other platforms can use a manual source build")
    system = platform.freedesktop_os_release()
    if system.get("ID") != "ubuntu" or system.get("VERSION_ID") != "22.04":
        parser.error("this distribution recipe requires Ubuntu 22.04; it does not restrict manual source builds")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if bool(args.weft_source_dir) != bool(args.weft_binary_dir):
        parser.error("Weft requires both --weft-source-dir and --weft-binary-dir")
    tools = {}
    for name in ("cmake", "ninja", "gcc", "g++"):
        executable = shutil.which(name)
        if executable is None:
            parser.error(f"Required source-build tool is unavailable: {name}")
        tools[name] = str(Path(executable).resolve())
    sdk = {}
    for name, directory in (("MLIR", args.mlir_dir), ("LLVM", args.llvm_dir)):
        directory = directory.expanduser().resolve()
        if not (directory / f"{name}Config.cmake").is_file():
            parser.error(f"{name}Config.cmake not found in {directory}")
        sdk[name] = directory

    output = outside_source(args.output_dir)
    if args.work_dir is None:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        parent = outside_source(cache / "intentdsl/distribution-build")
        parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="build-", dir=parent))
    else:
        work = outside_source(args.work_dir)
        work.mkdir(parents=True, exist_ok=False)
    if work == output or work in output.parents or output in work.parents:
        parser.error("--output-dir and --work-dir must be separate, non-nested directories")
    print(f"Build workspace (retained on failure): {work}", flush=True)
    print(f"Baseline: {system['PRETTY_NAME']}; {platform.machine()}; Python {platform.python_version()}; glibc {platform.libc_ver()[1]}", flush=True)

    env = dict(os.environ)
    for name in tuple(env):
        if (name in {"PYTHONPATH", "PYTHONHOME", "INTENT_COMPILER", "INTENT_OPTIMIZER",
                     "INTENT_CACHE_DIR", "CMAKE_ARGS", "CMAKE_TOOLCHAIN_FILE", "CC", "CXX",
                     "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"}
                or name.startswith(("SKBUILD_", "SETUPTOOLS_SCM_PRETEND_"))):
            del env[name]
    env.update(PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), CMAKE_GENERATOR="Ninja")
    build_environment = work / "build-environment"
    run(sys.executable, "-m", "venv", build_environment, cwd=work, env=env)
    build_python = build_environment / "bin/python"
    env["PATH"] = str(build_environment / "bin") + os.pathsep + env["PATH"]
    run(build_python, "-m", "pip", "install", "--upgrade", "pip", "build>=1.2,<2",
        "twine>=6,<7", "auditwheel>=6.8,<7", "patchelf>=0.17", cwd=work, env=env)
    source_output, wheel_output = work / "sdist", work / "wheel"
    run(build_python, "-m", "build", "--sdist", "--outdir", source_output,
        REPOSITORY, cwd=work, env=env)
    source_archive = one_artifact(source_output, "*.tar.gz")
    author_source = work / "softmax.py"
    version, manual = read_source_distribution(source_archive, author_source)

    settings = {
        "build-dir": str(work / "native-build"),
        "cmake.build-type": "Release",
        "cmake.define.MLIR_DIR": str(sdk["MLIR"]),
        "cmake.define.LLVM_DIR": str(sdk["LLVM"]),
        "cmake.define.CMAKE_C_COMPILER": tools["gcc"],
        "cmake.define.CMAKE_CXX_COMPILER": tools["g++"],
    }
    if args.runtime_notices is not None:
        settings["cmake.define.INTENT_RUNTIME_NOTICES"] = str(args.runtime_notices.expanduser().resolve(strict=True))
    if args.weft_source_dir is not None:
        settings["cmake.define.INTENT_WEFT_SOURCE_DIR"] = str(args.weft_source_dir.expanduser().resolve(strict=True))
        settings["cmake.define.INTENT_WEFT_BINARY_DIR"] = str(args.weft_binary_dir.expanduser().resolve(strict=True))
    run(build_python, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", wheel_output,
        *(f"--config-settings={name}={value}" for name, value in settings.items()),
        source_archive, cwd=work, env=env)
    wheel = one_artifact(wheel_output, "*.whl")
    check_wheel_contents(wheel, version, manual)
    run(build_python, "-m", "auditwheel", "show", wheel, cwd=work, env=env)
    portable_output = work / "repaired-wheel"
    run(build_python, "-m", "auditwheel", "repair", "--plat", "manylinux_2_35_x86_64",
        "--only-plat", "--wheel-dir", portable_output, wheel, cwd=work, env=env)
    wheel = one_artifact(portable_output, "*.whl")
    check_wheel_contents(wheel, version, manual)
    run(build_python, "-m", "auditwheel", "show", wheel, cwd=work, env=env)
    run(build_python, "-m", "twine", "check", "--strict", source_archive, wheel,
        cwd=work, env=env)

    # No source imports, explicit compiler, SDK loader path or prior cache may
    # make the installed wheel's own runtime checks appear to work.
    installed = work / "installed"
    run(sys.executable, "-m", "venv", installed, cwd=work, env=env)
    runtime_env = {name: value for name, value in env.items()
                   if name not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "INTENT_MLIR_DIR", "INTENT_LLVM_DIR"}}
    runtime_env["XDG_CACHE_HOME"] = str(work / "cache")
    runtime_env["PATH"] = str(installed / "bin") + os.pathsep + os.defpath
    run(installed / "bin/python", "-m", "pip", "install", str(wheel) + "[manual]",
        cwd=work, env=runtime_env)
    intent = installed / "bin/intent"
    diagnosis = json.loads(run(intent, "doctor", "--json", cwd=work, env=runtime_env,
                               output=work / "doctor.json"))
    compiler = next(check["detail"] for check in diagnosis["checks"]
                    if check["category"] == "compiler")
    for provider in compiler["providers"].values():
        for filename in provider["profiles"]:
            resource = Path(filename).resolve(strict=True)
            resource.relative_to(installed)
            if not resource.is_file() or resource.stat().st_size == 0:
                raise ValueError(f"The installed compiler resource is missing or empty: {resource}")
    run(intent, "describe", "--json", cwd=work, env=runtime_env, output=work / "describe.json")
    run(installed / "bin/intent-manual", cwd=work, env=runtime_env)
    run(installed / "bin/intent-compiler-mcp", cwd=work, env=runtime_env)
    lowered = json.loads(run(intent, "compile", f"{author_source}:stable_softmax", "--stage", "kir",
                             "--json", cwd=work, env=runtime_env, output=work / "kir.json"))
    run(intent, "optimize", lowered["files"]["kernel.mlir"],
        "--pipeline", "builtin.module(canonicalize,cse)", "--json",
        cwd=work, env=runtime_env, output=work / "optimize.json")

    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / artifact.name for artifact in (source_archive, wheel)]
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"Refusing to replace a distribution: {destination}")
    for artifact, destination in zip((source_archive, wheel), destinations):
        shutil.copy2(artifact, destination)
    print(json.dumps({"sdist": str(destinations[0]), "wheel": str(destinations[1]),
                      "work_directory": str(work), "installed_environment": str(installed),
                      "platform": "manylinux_2_35_x86_64",
                      "scope": "Installed compiler, optimizer, public declarations, manual packaging and original softmax KIR; no provider execution"}, indent=2))


if __name__ == "__main__":
    main()
