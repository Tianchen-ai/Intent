"""Install the compiler tools' shared Linux runtime closure into their wheel.

CMake supplies resolved ELF dependencies. This script preserves their SONAMEs,
collects notices from the owning Debian packages or an explicit notice map, and
changes only the staged copies. It does not determine a manylinux platform tag.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess


# These remain requirements of the host Linux platform. In particular, never
# ship a private glibc or loader alongside libraries built against the host ABI.
_PLATFORM_LIBRARIES = re.compile(
    r"(?:ld-linux[^/]*|ld64\.so(?:\.[0-9]+)*|"
    r"lib(?:c|m|mvec|pthread|dl|rt|util|resolv|anl|BrokenLocale|thread_db|"
    r"nss_[A-Za-z0-9_]+|stdc\+\+|gcc_s)\.so(?:\.[0-9]+)*)\Z"
)
_COMMON_LICENSE = re.compile(r"/usr/share/common-licenses/[A-Za-z0-9.+_-]+")


@dataclass(frozen=True)
class Library:
    source: Path
    soname: str
    notices: tuple[Path, ...]
    package: str | None


def command(*arguments: str | Path) -> str:
    return subprocess.check_output([str(value) for value in arguments], text=True).strip()


def read_notice_map(filename: str) -> dict[Path, tuple[Path, ...]]:
    if not filename:
        return {}
    source = Path(filename).expanduser().resolve(strict=True)
    entries = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(entries, dict):
        raise ValueError("Runtime notices must map library paths to lists of notice files")
    result = {}
    for library, files in entries.items():
        if not isinstance(files, list) or not files or any(not isinstance(value, str) for value in files):
            raise ValueError(f"Runtime notices for {library!r} must be a non-empty list of paths")
        def resolve(value: str) -> Path:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else source.parent / path).resolve(strict=True)
        path = resolve(library)
        if path in result:
            raise ValueError(f"Runtime notice map names the same library more than once: {path}")
        result[path] = tuple(resolve(value) for value in files)
    return result


def package_notices(library: Path) -> tuple[str, tuple[Path, ...]]:
    query = shutil.which("dpkg-query")
    if query is None:
        raise RuntimeError(f"No notice mapping for {library}; set INTENT_RUNTIME_NOTICES for this SDK")
    owners = set()
    # Prefer ownership of the resolved library itself. Development packages can
    # own symlinks to a runtime package's file; those are not its copyright owner.
    # A merged-/usr system may instead register the same regular file under /lib.
    for pattern in (str(library), f"*/{library.name}"):
        lookup = subprocess.run([query, "--search", pattern],
                                text=True, capture_output=True, check=False)
        if lookup.returncode not in (0, 1):
            raise RuntimeError(f"Cannot query ownership of {library}: {lookup.stderr}")
        for line in lookup.stdout.splitlines():
            owner, separator, filename = line.partition(": ")
            path = Path(filename)
            if (separator and path.is_file() and not path.is_symlink()
                    and path.samefile(library)):
                owners.update(owner.split(", "))
        if owners:
            break
    if len(owners) != 1:
        raise RuntimeError(f"No unique package owner for {library}; set INTENT_RUNTIME_NOTICES")
    package = owners.pop()
    files = [Path(line) for line in command(query, "--listfiles", package).splitlines()]
    notices = {path.resolve() for path in files if path.name == "copyright" and path.is_file()}
    # Some packages own a doc-directory symlink instead of its individual files.
    for path in files:
        if path.parent.name == "doc" and path.name == package.split(":")[0]:
            copyright_file = path / "copyright"
            if copyright_file.is_file():
                notices.add(copyright_file.resolve())
    if not notices:
        raise RuntimeError(f"Package {package} provides no copyright file for {library}; set INTENT_RUNTIME_NOTICES")
    return package, tuple(sorted(notices))


def complete_notices(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    pending = list(paths)
    contents = set()
    while pending:
        path = pending.pop().resolve(strict=True)
        if path in contents:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Third-party notice is not a non-empty file: {path}")
        contents.add(path)
        # Debian copyright files can refer to common license texts instead of
        # embedding them. Ship those exact referenced texts as well.
        pending.extend(Path(name) for name in _COMMON_LICENSE.findall(path.read_text(encoding="utf-8")))
    return tuple(sorted(contents))


def bundle(executables: tuple[Path, ...], dependencies: Path, patchelf: str, notices: str) -> None:
    executables = tuple(path.resolve(strict=True) for path in executables)
    directories = {path.parent for path in executables}
    if len(directories) != 1 or any(not path.is_file() for path in executables):
        raise ValueError("Compiler tools must be installed as files in one shared runtime directory")
    runtime_directory = executables[0].parent
    supplied = read_notice_map(notices)
    libraries = {}
    platform = set()
    for filename in dependencies.read_text(encoding="utf-8").splitlines():
        if not filename:
            continue
        source = Path(filename).resolve(strict=True)
        soname = command(patchelf, "--print-soname", source)
        if not soname or Path(soname).name != soname:
            raise ValueError(f"Runtime library requires a plain SONAME: {source}: {soname!r}")
        if _PLATFORM_LIBRARIES.fullmatch(soname):
            platform.add(soname)
            continue
        if soname in libraries:
            if source != libraries[soname].source:
                raise ValueError(f"Conflicting runtime libraries for {soname}: {source}, {libraries[soname].source}")
            continue
        package = None
        files = supplied.get(source)
        if files is None:
            package, files = package_notices(source)
        libraries[soname] = Library(source, soname, complete_notices(files), package)

    # Check the actual needed names before copying. Absolute dependencies and
    # unresolved aliases must not silently fall back to the SDK at runtime.
    for source in (*executables, *(item.source for item in libraries.values())):
        for needed in command(patchelf, "--print-needed", source).splitlines():
            if needed not in libraries and not _PLATFORM_LIBRARIES.fullmatch(needed):
                raise ValueError(f"Runtime closure does not provide {needed!r}, needed by {source}")

    destination = runtime_directory / "lib"
    notice_root = runtime_directory / "third-party"
    destination.mkdir(exist_ok=True)
    notice_root.mkdir(exist_ok=True)
    records = []
    for soname, library in sorted(libraries.items()):
        target = destination / soname
        shutil.copyfile(library.source, target)
        target.chmod(0o755)
        subprocess.run([patchelf, "--set-rpath", "$ORIGIN", str(target)], check=True)
        directory = notice_root / soname
        directory.mkdir(exist_ok=True)
        installed = []
        for index, source in enumerate(library.notices):
            target_notice = directory / f"{index + 1:02d}-{source.name}"
            shutil.copyfile(source, target_notice)
            installed.append(str(target_notice.relative_to(runtime_directory)))
        records.append({"soname": soname, "source": str(library.source),
                        "package": library.package, "notices": installed})
    for executable in executables:
        subprocess.run([patchelf, "--set-rpath", "$ORIGIN/lib", str(executable)], check=True)
    (notice_root / "libraries.json").write_text(
        json.dumps({"libraries": records, "platform_libraries": sorted(platform)}, indent=2) + "\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, action="append", required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--patchelf", required=True)
    parser.add_argument("--notices", default="")
    args = parser.parse_args()
    bundle(tuple(args.executable), args.dependencies, args.patchelf, args.notices)


if __name__ == "__main__":
    main()
