from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
from typing import Callable


class _UncacheableToolchain(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolchainSnapshot:
    identity: tuple[object, ...] | None
    reason: str | None
    _refresh: Callable[[], "ToolchainSnapshot"] | None = field(default=None, repr=False, compare=False)

    def check_unchanged(self) -> None:
        if self.identity is None:
            return
        current = self._refresh()
        if current.identity != self.identity:
            detail = current.reason or "toolchain files or dependency resolution changed"
            raise RuntimeError(f"Mojo toolchain changed during native compilation: {detail}")


# Recognize console-script trampolines, not arbitrary programs with a Python
# shebang. Distribution ownership is independently checked by the interpreter.
_LAUNCHERS = tuple(ast.dump(ast.parse(source)) for source in (
    """import sys
from mojo._entrypoints import exec_mojo
if __name__ == '__main__':
    if sys.argv[0].endswith('-script.pyw'):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(exec_mojo())
""",
    """import re
import sys
from mojo._entrypoints import exec_mojo
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])
    sys.exit(exec_mojo())
""",
    """import sys
from mojo._entrypoints import exec_mojo
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(exec_mojo())
""",
))

_PROBE = r"""
import sys
from pathlib import Path
launcher = Path(sys.argv[1]).resolve()
sys.path[0] = str(launcher.parent)
import importlib.metadata as metadata
import json
relevant_environment = set(json.loads(sys.argv[2]))

def inspect_installation():
    owners = []
    for distribution in metadata.distributions():
        entries = [entry for entry in distribution.entry_points
                   if entry.group == 'console_scripts' and entry.name == 'mojo'
                   and entry.value == 'mojo._entrypoints:exec_mojo']
        if not entries:
            continue
        files = {str(path): Path(distribution.locate_file(path)).resolve()
                 for path in distribution.files or ()}
        if launcher in files.values():
            owners.append((distribution, files))
    if len(owners) != 1:
        return {'reason': 'selected launcher is not owned by one official Mojo console-entry distribution'}
    distribution, files = owners[0]
    import mojo._entrypoints as entrypoints
    import mojo._package_root as package_root
    import mojo.run as run
    for module in (entrypoints, package_root, run):
        relative = module.__name__.replace('.', '/') + '.py'
        if Path(module.__file__).resolve() != files.get(relative):
            return {'reason': 'Mojo launcher helpers are shadowed outside their owning distribution'}
    if not callable(getattr(run, '_sdk_default_env', None)) or not callable(getattr(run, '_mojo_env', None)):
        return {'reason': 'installed Mojo launcher does not expose its SDK environment resolver'}
    defaults = run._sdk_default_env()
    if not defaults:
        return {'reason': 'Mojo development/runfiles installation does not expose an SDK dependency root'}
    modules = []
    for module in list(sys.modules.values()):
        for attribute in ('__file__', '__cached__'):
            path = getattr(module, attribute, None)
            if path and Path(path).is_file():
                modules.append(str(Path(path).absolute()))
    metadata_files = [str(path) for name, path in files.items()
                      if '.dist-info/' in name and
                      name.rsplit('/', 1)[-1] in ('METADATA', 'entry_points.txt', 'RECORD')]
    search = [Path(path).absolute() for path in sys.path if path]
    startup = [str(path) for directory in search if directory.is_dir()
               for path in directory.glob('*.pth')]
    effective = {name: value for name, value in run._mojo_env().items()
                 if name in relevant_environment or
                 name.startswith(('MODULAR_', 'LD_', 'MOJO_', 'KGEN_', 'LLVM_'))}
    return {'defaults': defaults, 'modules': sorted(set(modules)),
            'metadata': metadata_files, 'python': sys.executable,
            'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
            'search': [str(path) for path in search], 'startup': startup,
            'environment': effective}

print(json.dumps(inspect_installation()))
"""

_ENVIRONMENT = (
    "PATH", "HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
    "PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONUSERBASE", "PYTHONSAFEPATH",
    "VIRTUAL_ENV", "LANG", "LC_ALL", "LC_CTYPE", "GLIBC_TUNABLES",
    "MODULAR_CACHE_DIR", "MODULAR_ENABLE_CACHE_LOGGING",
    "MODULAR_ENABLE_AFFINITY", "MODULAR_THREAD_BUSY_WAIT_US",
)
_EXTERNAL_SEARCH = (
    "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "RUNFILES_DIR",
    "RUNFILES_MANIFEST_FILE", "BUILD_WORKSPACE_DIRECTORY", "BAZEL_TEST", "TEST_TMPDIR",
)


def _file_identity(path: Path) -> tuple[object, ...]:
    try:
        link = path.lstat()
    except FileNotFoundError:
        return str(path), None
    target = path.resolve(strict=True)
    status = target.stat()
    return (str(path), str(target),
            (link.st_dev, link.st_ino, link.st_mode, link.st_size, link.st_mtime_ns, link.st_ctime_ns),
            (status.st_dev, status.st_ino, status.st_mode, status.st_size,
             status.st_mtime_ns, status.st_ctime_ns))


def _configuration_paths(environment: dict[str, str]) -> tuple[Path, ...]:
    # Config::findModularFile searches these Linux paths. MODULAR_HOME,
    # MODULAR_DERIVED_PATH and runfiles overrides are rejected before this call.
    home = environment.get("HOME")
    roots = []
    if home is not None:
        roots.append(Path(home) / ".modular")
    xdg = environment.get("XDG_CONFIG_HOME")
    if xdg is not None:
        roots.append(Path(xdg) / "modular")
    elif home is not None:
        roots.append(Path(home) / ".config/modular")
    roots.extend((Path("/opt/modular"), Path("/etc/modular")))
    return tuple(root.absolute() / "modular.cfg" for root in roots)


def _tree_paths(root: Path) -> list[Path]:
    paths = []
    def fail(error: OSError) -> None:
        raise error
    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail):
        base = Path(directory)
        paths.append(base)
        for name in directories:
            if (base / name).is_symlink():
                raise _UncacheableToolchain(f"Mojo SDK contains an unsupported directory symlink: {base / name}")
        paths.extend(base / name for name in files)
    return paths


def _elf_dependencies(executable: Path) -> tuple[bool, Path | None]:
    """Read the native ELF's own dependency declarations before invoking ldd."""
    with executable.open("rb") as source:
        header = source.read(64)
        if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01":
            raise _UncacheableToolchain(f"Mojo dependency inspection requires a Linux ELF64 executable: {executable}")
        offset, = struct.unpack_from("<Q", header, 32)
        entry_size, count = struct.unpack_from("<HH", header, 54)
        if entry_size < 56:
            raise _UncacheableToolchain(f"Invalid ELF program headers in {executable}")
        dynamic = []
        interpreter = None
        for index in range(count):
            source.seek(offset + entry_size * index)
            program_header = source.read(56)
            if len(program_header) != 56:
                raise _UncacheableToolchain(f"Incomplete ELF program headers in {executable}")
            kind, = struct.unpack_from("<I", program_header)
            begin, = struct.unpack_from("<Q", program_header, 8)
            length, = struct.unpack_from("<Q", program_header, 32)
            if kind == 2:
                dynamic.append((begin, length))
            elif kind == 3:
                source.seek(begin)
                interpreter = Path(source.read(length).rstrip(b"\0").decode("utf-8"))
        for begin, length in dynamic:
            source.seek(begin)
            entries = source.read(length)
            if len(entries) != length or length % 16:
                raise _UncacheableToolchain(f"Incomplete ELF dynamic entries in {executable}")
            for index in range(0, length, 16):
                kind, = struct.unpack_from("<Q", entries, index)
                if kind == 1:
                    return True, interpreter
                if kind == 0:
                    break
        return False, interpreter


def _shared_libraries(executables: tuple[Path, ...], environment: dict[str, str]) -> list[Path]:
    selected = shutil.which("ldd", path=environment.get("PATH", os.defpath))
    if selected is None:
        raise FileNotFoundError("ldd is required to resolve official Mojo toolchain shared libraries")
    paths = [Path(selected).absolute()]
    for executable in executables:
        dynamic, interpreter = _elf_dependencies(executable)
        if interpreter is not None:
            paths.append(interpreter)
        if not dynamic:
            continue
        result = subprocess.run([selected, str(executable)], env={**environment, "LC_ALL": "C"},
                                text=True, capture_output=True, check=False)
        if result.returncode:
            raise _UncacheableToolchain(
                f"Shared-library dependency inspection failed for {executable} "
                f"(exit {result.returncode}): {result.stderr}{result.stdout}"
            )
        for line in result.stdout.splitlines():
            text = line.strip()
            if not text or text.startswith("linux-vdso.so."):
                continue
            match = re.fullmatch(r"(?:\S+ => )?(/.+) \(0x[0-9a-fA-F]+\)", text)
            if match is None:
                raise _UncacheableToolchain(f"Cannot resolve shared-library dependency of {executable}: {text}")
            paths.append(Path(match.group(1)))
    return paths


def runtime_dependencies(library: Path, environment: dict[str, str]) -> list[Path]:
    """Resolve the actual native library's loader mapping, including on cache hits."""
    return _shared_libraries((library,), environment)


def _startup_dependencies(launcher: Path, environment: dict[str, str]) -> list[Path]:
    # Compiler startup can dlopen optional plugins that are absent from its
    # DT_NEEDED list. Ask the real loader for its successful and failed probes;
    # reconstructing rpath, hwcaps and system search order would miss shadows.
    # This diagnostic environment belongs only to the metadata query.
    result = subprocess.run([str(launcher), "build", "--print-effective-target"],
                            env={**environment, "LD_DEBUG": "libs"},
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise _UncacheableToolchain(
            f"Mojo startup dependency discovery failed (exit {result.returncode}): {result.stderr}"
        )
    paths = []
    searched = False
    for line in result.stderr.splitlines():
        if "trying file=" not in line and "calling init:" not in line:
            continue
        match = re.fullmatch(r"\s*\d+:\s*(trying file=|calling init:)\s*(.+)", line)
        if match is None or not Path(match.group(2)).is_absolute():
            raise _UncacheableToolchain(f"Unrecognized Mojo loader dependency record: {line}")
        searched |= match.group(1) == "trying file="
        # Preserve the loader's lookup path, including symlink components. The
        # file snapshot separately records its currently resolved identity.
        paths.append(Path(match.group(2)))
    if not searched:
        raise _UncacheableToolchain("Mojo loader did not expose dependency search records")
    return paths


def _resolve_toolchain(executable: str, declared_modules: tuple[str, ...],
                       environment: dict[str, str]) -> ToolchainSnapshot:
    if declared_modules != ("std", "max"):
        return ToolchainSnapshot(None, "native source has no supported compiler-declared std/max dependency closure")
    environment = dict(environment)
    launcher = Path(executable).absolute()
    with launcher.open("rb") as source:
        header = source.readline(4096)
        if not header.startswith(b"#!/"):
            return ToolchainSnapshot(None, "selected compiler is not an official Python console launcher")
        interpreter_text = header[2:].decode("utf-8").strip()
        if any(character.isspace() for character in interpreter_text):
            return ToolchainSnapshot(None, "compiler launcher uses an unsupported interpreter command")
        interpreter = Path(interpreter_text)
        script = source.read().decode("utf-8")
    try:
        syntax = ast.dump(ast.parse(script))
    except SyntaxError:
        return ToolchainSnapshot(None, "compiler launcher is not a supported official console trampoline")
    if syntax not in _LAUNCHERS:
        return ToolchainSnapshot(None, "compiler launcher contains custom code outside the official console trampoline")
    result = subprocess.run([str(interpreter), "-c", _PROBE, str(launcher),
                             json.dumps(_ENVIRONMENT + _EXTERNAL_SEARCH)], env=environment,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        return ToolchainSnapshot(None,
            f"Official Mojo SDK discovery failed (exit {result.returncode}): {result.stderr}")
    try:
        installation = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return ToolchainSnapshot(None, f"Official Mojo SDK discovery produced invalid JSON: {error}")
    if not isinstance(installation, dict):
        return ToolchainSnapshot(None, "Official Mojo SDK discovery did not produce an object")
    if "reason" in installation:
        return ToolchainSnapshot(None, installation["reason"])
    defaults = installation["defaults"]
    effective = installation["environment"]
    sdk = Path(defaults["MODULAR_MOJO_MAX_PACKAGE_ROOT"]).resolve(strict=True)
    driver = Path(defaults["MODULAR_MOJO_MAX_DRIVER_PATH"])
    imports = Path(defaults["MODULAR_MOJO_MAX_IMPORT_PATH"])
    linker = sdk / "bin/lld"
    for path in (driver, imports, linker):
        if not path.resolve(strict=True).is_relative_to(sdk):
            return ToolchainSnapshot(None, "official Mojo SDK paths escape the reported package root")
    for name, value in effective.items():
        if name in _EXTERNAL_SEARCH or name.startswith(("LD_", "MOJO_", "KGEN_", "LLVM_")):
            return ToolchainSnapshot(None, f"toolchain dependency closure does not cover environment override {name}")
        if name.startswith("MODULAR_") and name not in _ENVIRONMENT:
            if name not in defaults or value != defaults[name]:
                return ToolchainSnapshot(None, f"toolchain dependency closure does not cover environment override {name}")
    configs = _configuration_paths(effective)
    for path in configs:
        if _file_identity(path)[1] is not None:
            return ToolchainSnapshot(None, f"toolchain dependency closure does not cover Modular configuration {path}")
    if _file_identity(Path("/etc/ld.so.preload"))[1] is not None:
        return ToolchainSnapshot(None, "toolchain dependency closure does not cover /etc/ld.so.preload")
    for name in declared_modules:
        package = imports / f"{name}.mojoc"
        if not package.is_file():
            raise FileNotFoundError(f"official Mojo SDK is missing declared native package: {package}")
    paths = _tree_paths(sdk)
    paths.extend((launcher, interpreter, Path(installation["python"]), Path(__file__),
                  Path(installation["prefix"]) / "pyvenv.cfg", Path("/etc/ld.so.cache"),
                  Path("/etc/ld.so.preload"), *configs))
    paths.extend(Path(path) for field in ("modules", "metadata", "search", "startup") for path in installation[field])
    paths.extend(_shared_libraries((driver, linker, interpreter.resolve(strict=True)), effective))
    paths.extend(_startup_dependencies(launcher, environment))
    identity = (tuple(sorted(defaults.items())),
                tuple((name, effective.get(name)) for name in _ENVIRONMENT),
                tuple(_file_identity(path) for path in sorted(set(paths))))
    return ToolchainSnapshot(
        identity, None, lambda: resolve_toolchain(executable, declared_modules, environment),
    )


def resolve_toolchain(executable: str, declared_modules: tuple[str, ...],
                      environment: dict[str, str]) -> ToolchainSnapshot:
    """Prove a closed official SDK for compiler-declared native imports.

    An unavailable identity disables only persistent binary reuse. Its reason
    preserves discovery failures for diagnostics; the caller still invokes the
    selected compiler with the original environment. Native linker inputs are
    recorded separately from the actual link depfile.
    """
    try:
        return _resolve_toolchain(executable, declared_modules, environment)
    except (_UncacheableToolchain, OSError, UnicodeError, KeyError) as error:
        return ToolchainSnapshot(None, f"Mojo native cache dependency discovery failed: {error}")
