#!/usr/bin/env python3
"""Repeatable out-of-tree release builder & clean-room verifier for V2-6.R."""

from __future__ import annotations

import argparse
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


def _parse_simple_pyproject(toml_path: Path) -> dict:
    """Parse the PEP 621 subset used by compilerinfra/python and compilerinfra/agent on Python >=3.10."""
    try:
        import tomllib  # Python 3.11+

        return tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except ImportError:
        pass
    text = toml_path.read_text(encoding="utf-8")

    def _str_field(section: str, key: str) -> str:
        m = re.search(rf"(?ms)^\[{re.escape(section)}\].*?^{re.escape(key)}\s*=\s*\"([^\"]*)\"", text)
        return m.group(1) if m else ""

    def _list_field(section: str, key: str) -> list[str]:
        m = re.search(rf"(?ms)^\[{re.escape(section)}\].*?^{re.escape(key)}\s*=\s*\[(.*?)\]", text)
        return re.findall(r"\"([^\"]+)\"", m.group(1)) if m else []

    scripts = {}
    sm = re.search(r"(?ms)^\[project\.scripts\]\s*\n(.*?)(?=^\[|\Z)", text)
    if sm:
        for line in sm.group(1).splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=\s*\"([^\"]+)\"", line)
            if m:
                scripts[m.group(1)] = m.group(2)

    return {
        "project": {
            "name": _str_field("project", "name"),
            "version": _str_field("project", "version"),
            "description": _str_field("project", "description"),
            "requires-python": _str_field("project", "requires-python") or ">=3.10",
            "license": _str_field("project", "license"),
            "license-files": _list_field("project", "license-files"),
            "dependencies": _list_field("project", "dependencies"),
            "scripts": scripts,
        },
        "tool": {
            "setuptools": {
                "packages": _list_field("tool.setuptools", "packages"),
                "py-modules": _list_field("tool.setuptools", "py-modules"),
            }
        },
    }


def _build_pure_python_pkg(src_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build sdist and py3-none-any wheel without requiring setuptools>=61 on the host."""
    meta = _parse_simple_pyproject(src_dir / "pyproject.toml")
    proj = meta["project"]
    st = meta.get("tool", {}).get("setuptools", {})
    name: str = proj["name"]
    dist_norm = name.replace("-", "_")
    version: str = proj["version"]
    desc: str = proj.get("description", "")
    req_py: str = proj.get("requires-python", ">=3.10")
    deps: list[str] = proj.get("dependencies", [])
    scripts: dict[str, str] = proj.get("scripts", {})
    packages: list[str] = st.get("packages", [])
    py_modules: list[str] = st.get("py-modules", [])
    license_files = sorted({p.relative_to(src_dir) for pattern in proj.get("license-files", [])
                            for p in src_dir.glob(pattern) if p.is_file()})

    meta_lines = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"Summary: {desc}",
        f"Requires-Python: {req_py}",
        *([f"License-Expression: {proj['license']}"] if proj.get("license") else []),
        *[f"License-File: {p.as_posix()}" for p in license_files],
        *[f"Requires-Dist: {d}" for d in deps],
        "",
    ]
    pkg_info = "\n".join(meta_lines)

    # 1. Build sdist (.tar.gz)
    sdist_path = out_dir / f"{dist_norm}-{version}.tar.gz"
    base = f"{dist_norm}-{version}"
    with tarfile.open(sdist_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        b_info = pkg_info.encode("utf-8")
        ti = tarfile.TarInfo(f"{base}/PKG-INFO")
        ti.size = len(b_info)
        ti.mode = 0o644
        tf.addfile(ti, io.BytesIO(b_info))
        tf.add(src_dir / "pyproject.toml", arcname=f"{base}/pyproject.toml", recursive=False)
        for path in license_files:
            tf.add((src_dir / path).resolve(), arcname=f"{base}/{path.as_posix()}", recursive=False)
        for mod in py_modules:
            tf.add(src_dir / f"{mod}.py", arcname=f"{base}/{mod}.py", recursive=False)
        for pkg in packages:
            for item in sorted((src_dir / pkg).rglob("*")):
                if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}:
                    rel = item.relative_to(src_dir).as_posix()
                    tf.add(item, arcname=f"{base}/{rel}", recursive=False)

    # 2. Build wheel (py3-none-any)
    whl_path = out_dir / f"{dist_norm}-{version}-py3-none-any.whl"
    with tempfile.TemporaryDirectory(prefix=f"whl-{dist_norm}-") as tmp:
        wroot = Path(tmp)
        for mod in py_modules:
            shutil.copy2(src_dir / f"{mod}.py", wroot / f"{mod}.py")
        for pkg in packages:
            for item in sorted((src_dir / pkg).rglob("*")):
                if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}:
                    dst = wroot / item.relative_to(src_dir)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dst)

        dist_info = wroot / f"{dist_norm}-{version}.dist-info"
        dist_info.mkdir(parents=True)
        for path in license_files:
            dest = dist_info / "licenses" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_dir / path, dest)
        (dist_info / "METADATA").write_text(pkg_info, encoding="utf-8")
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: compilerinfra-release\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            encoding="utf-8",
        )
        if scripts:
            ep = ["[console_scripts]"] + [f"{k} = {v}" for k, v in scripts.items()] + [""]
            (dist_info / "entry_points.txt").write_text("\n".join(ep), encoding="utf-8")

        records = []
        for f in sorted(wroot.rglob("*")):
            if not f.is_file():
                continue
            rel_posix = f.relative_to(wroot).as_posix()
            data = f.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            records.append((rel_posix, f"sha256={digest}", str(len(data))))
        rec_rel = f"{dist_norm}-{version}.dist-info/RECORD"
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for r in records:
            writer.writerow(r)
        writer.writerow((rec_rel, "", ""))
        (wroot / rec_rel).write_text(buf.getvalue(), encoding="utf-8")

        with zipfile.ZipFile(whl_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(wroot.rglob("*")):
                if not f.is_file():
                    continue
                zi = zipfile.ZipInfo(f.relative_to(wroot).as_posix())
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (stat.S_IFREG | 0o644) << 16
                zf.writestr(zi, f.read_bytes())

    return sdist_path, whl_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory outside source tree")
    parser.add_argument("--compilerinfra-root", type=Path, default=None, help="Optional CompilerInfra repo root")
    parser.add_argument("--sdk-prefix", type=Path, default=None, help="Installed static CompilerInfra 0.5.2 SDK prefix")
    parser.add_argument("--llvm-prefix", type=Path, default=None, help="LLVM/MLIR 20.1.8 prefix")
    parser.add_argument("--onednn-prefix", type=Path, default=None, help="Optional oneDNN >=3.13 prefix for header build")
    parser.add_argument("--platform-tag", default=f"linux_{platform.machine()}", help="Wheel platform tag")
    parser.add_argument("--jobs", type=int, default=16, help="Parallel CMake build jobs")
    parser.add_argument("--verify-clean-sdist", action="store_true", help="Rebuild wheel from sdist in a clean venv and verify")
    args = parser.parse_args()

    product_root = Path(__file__).resolve().parents[1]
    out_dir = args.output_dir.resolve()
    if out_dir.is_relative_to(product_root):
        raise ValueError(f"--output-dir must be outside the product source tree: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compilerinfra_root:
        ci_root = args.compilerinfra_root.resolve()
        _build_pure_python_pkg(ci_root / "python", out_dir)
        _build_pure_python_pkg(ci_root / "agent", out_dir)

    sys.path.insert(0, str(product_root / "cmake"))
    import pep517_backend as backend  # type: ignore

    env_backup = dict(os.environ)
    try:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
        os.environ["INTENTDSL_WHEEL_PLATFORM"] = args.platform_tag
        prefixes = [str(p.resolve()) for p in (args.sdk_prefix, args.llvm_prefix, args.onednn_prefix) if p]
        if prefixes:
            existing = os.environ.get("CMAKE_PREFIX_PATH", "")
            os.environ["CMAKE_PREFIX_PATH"] = ";".join(prefixes + ([existing] if existing else []))

        cwd = Path.cwd()
        try:
            os.chdir(product_root)
            sdist_name = backend.build_sdist(str(out_dir))
            whl_name = backend.build_wheel(str(out_dir))
        finally:
            os.chdir(cwd)
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    print(f"Built sdist: {out_dir / sdist_name}")
    print(f"Built wheel: {out_dir / whl_name}")

    if args.verify_clean_sdist:
        ci_ver = json.loads((product_root / "product.json").read_text(encoding="utf-8"))["compilerinfra"]
        ci_wheels = sorted(out_dir.glob(f"compilerinfra-{ci_ver}-*.whl"))
        if not ci_wheels:
            raise FileNotFoundError(f"Pass --compilerinfra-root so compilerinfra=={ci_ver} wheel is available for clean venv test")
        with tempfile.TemporaryDirectory(prefix="intentdsl-clean-sdist-") as tmp:
            tmp_path = Path(tmp)
            venv_dir = tmp_path / "venv"
            rebuilt_dir = tmp_path / "rebuilt"
            rebuilt_dir.mkdir()
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            vpy = venv_dir / "bin" / "python"
            vpip = venv_dir / "bin" / "pip"

            clean_env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("PYTHONPATH", "PYTHONHOME", "INTENTDSL_LIBRARY")
            }
            clean_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
            clean_env["INTENTDSL_WHEEL_PLATFORM"] = args.platform_tag
            prefixes = [str(p.resolve()) for p in (args.sdk_prefix, args.llvm_prefix, args.onednn_prefix) if p]
            if prefixes:
                clean_env["CMAKE_PREFIX_PATH"] = ";".join(prefixes)

            subprocess.run([str(vpip), "install", "--no-index", "--find-links", str(out_dir), str(ci_wheels[0])], env=clean_env, check=True)
            subprocess.run(
                [str(vpip), "wheel", "--no-build-isolation", "--no-deps", str(out_dir / sdist_name), "-w", str(rebuilt_dir)],
                env=clean_env,
                check=True,
            )
            product_version = json.loads((product_root / "product.json").read_text())["version"]
            rebuilt_whl = next(rebuilt_dir.glob(f"intentdsl-{product_version}-*.whl"))
            subprocess.run([str(vpip), "install", "--no-deps", str(rebuilt_whl)], env=clean_env, check=True)

            check_code = """
import json, intentdsl as I
assert "_bundled" in str(I.library_path()), I.library_path()
lay = I.layout()
assert set(lay["exits"]) == {"triton", "cutile", "mojo", "weft", "bangc"}
setup_rep = I.setup()
assert setup_rep["ok"] and setup_rep["verified_assets"] > 0
zh = I.manual("components", lang="zh")
en = I.manual("components", lang="en")
assert "compilerinfra-dev" in zh and "compilerinfra-dev" in en
sk = I.skills()
assert "agent/skills/intentdsl-development.md" in sk
print("CLEAN_SDIST_VERIFIED:", json.dumps({"library": str(I.library_path()), "assets": setup_rep["verified_assets"], "skills": list(sk)}))
"""
            subprocess.run([str(vpy), "-c", check_code], cwd=tmp_path, env=clean_env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
