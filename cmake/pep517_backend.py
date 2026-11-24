"""Out-of-tree PEP 517 build backend for IntentDSL sdist and self-contained native wheels."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ALLOWED_DIRS = (
    "cmake",
    "frontend",
    "levels",
    "exits",
    "runtime",
    "include",
    "agent",
    "doc",
)
ROOT_FILES = (
    "CMakeLists.txt",
    "product.json",
    "pyproject.toml",
    "README.md",
    "AGENTS.md",
)


def _product_meta(root: Path) -> tuple[str, str, str]:
    decl = json.loads((root / "product.json").read_text(encoding="utf-8"))
    return decl["name"], decl["version"], decl["compilerinfra"]


def _pkg_info(name: str, version: str, ci_ver: str, readme: str) -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        "Summary: Multi-target native JIT compiler for typed Python tensor kernels and programs",
        "Requires-Python: >=3.10",
        f"Requires-Dist: compilerinfra=={ci_ver}",
        "Description-Content-Type: text/markdown",
        "",
        readme,
    ]
    return "\n".join(lines)


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def _write_dist_info(dist_info: Path, name: str, version: str, ci_ver: str, readme: str, tag: str) -> None:
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_pkg_info(name, version, ci_ver, readme), encoding="utf-8")
    (dist_info / "WHEEL").write_text(
        "\n".join(
            [
                "Wheel-Version: 1.0",
                "Generator: intentdsl-pep517",
                "Root-Is-Purelib: false",
                f"Tag: {tag}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[console_scripts]\n"
        "intentdsl = intentdsl.cli:main\n"
        "intentdsl-compile = intentdsl.cli:compile_main\n",
        encoding="utf-8",
    )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    root = Path.cwd().resolve()
    name, version, ci_ver = _product_meta(root)
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    plat = os.environ.get("INTENTDSL_WHEEL_PLATFORM", f"linux_{platform.machine()}")
    tag = f"py3-none-{plat}"
    dist_name = f"{name}-{version}.dist-info"
    dist_info = Path(metadata_directory) / dist_name
    _write_dist_info(dist_info, name, version, ci_ver, readme, tag)
    return dist_name


def build_sdist(sdist_directory, config_settings=None):
    root = Path.cwd().resolve()
    name, version, ci_ver = _product_meta(root)
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    archive_base = f"{name}-{version}"
    out_path = Path(sdist_directory) / f"{archive_base}.tar.gz"

    def _skip(p: Path) -> bool:
        return any(
            part == "__pycache__" or part.startswith(".") or part.endswith(".egg-info")
            for part in p.parts
        ) or p.suffix in {".pyc", ".pyo", ".so", ".o", ".a"}

    with tarfile.open(out_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        pkg_bytes = _pkg_info(name, version, ci_ver, readme).encode("utf-8")
        ti = tarfile.TarInfo(f"{archive_base}/PKG-INFO")
        ti.size = len(pkg_bytes)
        ti.mode = 0o644
        tf.addfile(ti, io.BytesIO(pkg_bytes))

        for fname in ROOT_FILES:
            fpath = root / fname
            if fpath.is_file():
                tf.add(fpath, arcname=f"{archive_base}/{fname}", recursive=False)

        for dname in ALLOWED_DIRS:
            dpath = root / dname
            if not dpath.is_dir():
                continue
            for item in sorted(dpath.rglob("*")):
                rel = item.relative_to(root)
                if item.is_file() and not _skip(rel):
                    tf.add(item, arcname=f"{archive_base}/{rel.as_posix()}", recursive=False)

    return out_path.name


def _c_env() -> dict[str, str]:
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    return env


def _max_glibc_version(binary: Path) -> tuple[int, int]:
    objdump = shutil.which("objdump") or shutil.which("readelf")
    if not objdump:
        return (0, 0)
    cmd = [objdump, "-T", str(binary)] if objdump.endswith("objdump") else [objdump, "--dyn-syms", "-W", str(binary)]
    out = subprocess.check_output(cmd, text=True, errors="replace", env=_c_env())
    versions = [(0, 0)]
    for m in re.finditer(r"GLIBC_(\d+)\.(\d+)", out):
        versions.append((int(m.group(1)), int(m.group(2))))
    return max(versions)


def _needed_libs(binary: Path) -> list[str]:
    readelf = shutil.which("readelf")
    if readelf:
        out = subprocess.check_output([readelf, "-d", str(binary)], text=True, errors="replace", env=_c_env())
        return re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", out)
    objdump = shutil.which("objdump")
    if objdump:
        out = subprocess.check_output([objdump, "-p", str(binary)], text=True, errors="replace", env=_c_env())
        return re.findall(r"NEEDED\s+(\S+)", out)
    return []


def _runpaths(binary: Path) -> list[str]:
    readelf = shutil.which("readelf")
    if not readelf:
        return []
    out = subprocess.check_output([readelf, "-d", str(binary)], text=True, errors="replace", env=_c_env())
    return re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]+)\]", out)


def _audit_wheel_binaries(binaries: list[Path], plat_tag: str) -> None:
    allow_shared_mlir = os.environ.get("INTENTDSL_ALLOW_SHARED_MLIR", "") == "1"
    m = re.fullmatch(r"manylinux_(\d+)_(\d+)_\w+", plat_tag)
    ceiling = (int(m.group(1)), int(m.group(2))) if m else None
    for b in binaries:
        needed = _needed_libs(b)
        if not allow_shared_mlir:
            leaked = [
                lib
                for lib in needed
                if lib.startswith(("libMLIR", "libLLVM", "libzstd", "libxml2", "libedit", "libicu"))
            ]
            if leaked:
                raise RuntimeError(
                    f"{b.name} dynamically links {leaked}; build against a static MLIR/LLVM CompilerInfra SDK "
                    "(COMPILERINFRA_LINK_MLIR_DYLIB=OFF) so wheel users do not require an LLVM dev SDK"
                )
        if b.suffix == ".so":
            rpaths = _runpaths(b)
            if rpaths:
                raise RuntimeError(f"{b.name} retains build-host RPATH/RUNPATH: {rpaths}")
        if ceiling is not None:
            req = _max_glibc_version(b)
            if req > ceiling:
                raise RuntimeError(
                    f"{b.name} requires GLIBC_{req[0]}.{req[1]}, exceeding declared wheel tag {plat_tag}"
                )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    root = Path.cwd().resolve()
    name, version, ci_ver = _product_meta(root)
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    plat = os.environ.get("INTENTDSL_WHEEL_PLATFORM", f"linux_{platform.machine()}")
    tag = f"py3-none-{plat}"

    cmake = os.environ.get("CMAKE_EXECUTABLE") or shutil.which("cmake")
    if not cmake:
        raise FileNotFoundError("cmake >= 3.22 is required to build intentdsl from source")

    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "16")

    with tempfile.TemporaryDirectory(prefix="intentdsl-wheel-build-") as tmp:
        tmp_path = Path(tmp)
        build_dir = tmp_path / "build"
        stage_dir = tmp_path / "stage"
        wheel_root = tmp_path / "wheel"
        wheel_root.mkdir()

        cfg_cmd = [
            cmake,
            "-S",
            str(root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DINTENTDSL_PORTABLE_BINARY=ON",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DCMAKE_INSTALL_PREFIX={stage_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        ninja = shutil.which("ninja")
        if ninja:
            cfg_cmd.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"])

        for var in ("CMAKE_PREFIX_PATH", "CompilerInfra_DIR", "MLIR_DIR", "LLVM_DIR", "dnnl_DIR"):
            val = os.environ.get(var, "").strip()
            if val:
                cfg_cmd.append(f"-D{var}={val}")
        extra = os.environ.get("INTENTDSL_CMAKE_ARGS", "").strip()
        if extra:
            cfg_cmd.extend(extra.split())

        subprocess.run(cfg_cmd, check=True)
        subprocess.run([cmake, "--build", str(build_dir), "-j", str(jobs)], check=True)
        subprocess.run([cmake, "--install", str(build_dir)], check=True)

        libdir = next(
            (stage_dir / d for d in ("lib", "lib64") if (stage_dir / d / f"lib{name}.so").exists()),
            None,
        )
        if libdir is None:
            raise FileNotFoundError(f"lib{name}.so missing from CMake install stage {stage_dir}")

        real_so = (libdir / f"lib{name}.so").resolve(strict=True)
        compile_bin = (stage_dir / "bin" / f"{name}-compile").resolve(strict=True)

        strip_tool = shutil.which("strip")
        if strip_tool:
            subprocess.run([strip_tool, "--strip-debug", str(real_so)], check=False)
            subprocess.run([strip_tool, "--strip-debug", str(compile_bin)], check=False)

        _audit_wheel_binaries([real_so, compile_bin], plat)

        # 1. Python package & offline bilingual manual
        pkg_src = root / "frontend" / "python" / name
        pkg_dst = wheel_root / name
        pkg_dst.mkdir(parents=True)
        for py in sorted(pkg_src.rglob("*.py")):
            rel = py.relative_to(pkg_src)
            out_py = pkg_dst / rel
            out_py.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py, out_py)

        man_dst = pkg_dst / "_manual"
        man_dst.mkdir(parents=True)
        for lang in ("zh", "en"):
            mfile = root / "doc" / f"manual.{lang}.md"
            if mfile.is_file():
                shutil.copy2(mfile, man_dst / mfile.name)

        # 2. Bundled native library, assets, compiler driver, and public ABI headers
        bundled = pkg_dst / "_bundled"
        (bundled / "lib").mkdir(parents=True)
        (bundled / "bin").mkdir(parents=True)
        shutil.copy2(real_so, bundled / "lib" / f"lib{name}.so")
        (bundled / "lib" / f"lib{name}.so.python").write_text("python3\n", encoding="utf-8")
        shutil.copytree(libdir / f"{name}.assets", bundled / "lib" / f"{name}.assets")
        shutil.copy2(compile_bin, bundled / "bin" / f"{name}-compile")
        os.chmod(bundled / "bin" / f"{name}-compile", 0o755)
        if (stage_dir / "include" / name).is_dir():
            shutil.copytree(stage_dir / "include" / name, bundled / "include" / name)

        # Fix RPATH on bundled intentdsl-compile if stage_dir used lib64
        patchelf = shutil.which("patchelf")
        if patchelf:
            subprocess.run([patchelf, "--set-rpath", "$ORIGIN/../lib", str(bundled / "bin" / f"{name}-compile")], check=False)
            subprocess.run([patchelf, "--remove-rpath", str(bundled / "lib" / f"lib{name}.so")], check=False)

        dist_info = wheel_root / f"{name}-{version}.dist-info"
        _write_dist_info(dist_info, name, version, ci_ver, readme, tag)

        records = []
        for f in sorted(wheel_root.rglob("*")):
            if not f.is_file():
                continue
            rel_posix = f.relative_to(wheel_root).as_posix()
            data = f.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            records.append((rel_posix, f"sha256={digest}", str(len(data))))

        record_rel = f"{name}-{version}.dist-info/RECORD"
        rec_buf = io.StringIO()
        writer = csv.writer(rec_buf, lineterminator="\n")
        for row in records:
            writer.writerow(row)
        writer.writerow((record_rel, "", ""))
        (wheel_root / record_rel).write_text(rec_buf.getvalue(), encoding="utf-8")

        whl_name = f"{name}-{version}-{tag}.whl"
        whl_path = Path(wheel_directory) / whl_name
        with zipfile.ZipFile(whl_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(wheel_root.rglob("*")):
                if not f.is_file():
                    continue
                rel_posix = f.relative_to(wheel_root).as_posix()
                zi = zipfile.ZipInfo(rel_posix)
                zi.compress_type = zipfile.ZIP_DEFLATED
                mode = stat.S_IMODE(f.stat().st_mode)
                if f.parent.name == "bin" or mode & 0o111:
                    zi.external_attr = (stat.S_IFREG | 0o755) << 16
                else:
                    zi.external_attr = (stat.S_IFREG | 0o644) << 16
                zf.writestr(zi, f.read_bytes())

        return whl_name
