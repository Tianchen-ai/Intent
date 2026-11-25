"""Offline installation diagnostics, toolchain setup checks, skills, and bilingual manual."""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _probe_triton_and_weft():
    """Detect installed triton flavor (GPU triton vs CPU triton-weft) without crashing."""
    info = {
        "triton_package": None,
        "is_weft_triton": False,
        "weft_core": False,
        "coexistence_note": (
            "GPU triton and CPU triton-weft share the top-level 'triton' module; "
            "install them in separate virtual environments."
        ),
    }
    try:
        import weft_core  # type: ignore

        info["weft_core"] = True
        info["weft_core_version"] = getattr(weft_core, "__version__", "installed")
    except Exception:
        pass
    try:
        import triton  # type: ignore

        info["triton_package"] = getattr(triton, "__version__", "installed")
        try:
            import triton.backends.weft.target as wt  # type: ignore

            info["is_weft_triton"] = True
            info["weft_capabilities"] = wt.native_capabilities()
        except Exception:
            info["is_weft_triton"] = False
    except Exception as exc:
        info["triton_import_error"] = str(exc)
    return info


def _probe_backend(name: str, compiler, native_doctor: dict, tri_info: dict) -> dict:
    if name == "triton":
        cuda_ok, devs, err = False, [], ""
        try:
            import torch  # type: ignore

            cuda_ok = bool(torch.cuda.is_available())
            if cuda_ok:
                for i in range(torch.cuda.device_count()):
                    cap = torch.cuda.get_device_capability(i)
                    devs.append({"index": i, "name": torch.cuda.get_device_name(i), "sm": cap[0] * 10 + cap[1]})
        except Exception as exc:
            err = str(exc)
        has_gpu_triton = bool(tri_info["triton_package"]) and not tri_info["is_weft_triton"]
        missing = []
        if not cuda_ok:
            missing.append("CUDA device / torch with CUDA" + (f" ({err})" if err else ""))
        if not has_gpu_triton:
            missing.append(
                "GPU triton package (current 'triton' is triton-weft)"
                if tri_info["is_weft_triton"]
                else "triton Python package"
            )
        return {
            "ready": cuda_ok and has_gpu_triton,
            "devices": devs,
            "triton_version": tri_info["triton_package"] if has_gpu_triton else None,
            "coexistence_note": tri_info["coexistence_note"],
            "missing": missing,
        }

    if name == "cutile":
        cuda_ok, devs, ct_ok, nvcc = False, [], False, shutil.which("nvcc")
        try:
            import torch  # type: ignore

            cuda_ok = bool(torch.cuda.is_available())
            if cuda_ok:
                for i in range(torch.cuda.device_count()):
                    cap = torch.cuda.get_device_capability(i)
                    devs.append({"index": i, "name": torch.cuda.get_device_name(i), "sm": cap[0] * 10 + cap[1]})
        except Exception:
            pass
        try:
            import cuda.tile  # type: ignore  # noqa: F401

            ct_ok = True
        except Exception:
            ct_ok = False
        missing = []
        if not cuda_ok:
            missing.append("CUDA device / torch with CUDA")
        if not ct_ok:
            missing.append("cuda-tile (cuda.tile) Python package")
        if not nvcc:
            missing.append("nvcc on PATH")
        return {
            "ready": cuda_ok and ct_ok,
            "devices": devs,
            "cuda_tile": ct_ok,
            "nvcc": nvcc or "",
            "missing": missing,
        }

    if name == "mojo":
        mojo_bin = shutil.which(os.environ.get("INTENT_MOJO", "mojo"))
        gcc_bin = shutil.which("gcc")
        gomp_ok = False
        if gcc_bin:
            try:
                gomp = subprocess.check_output([gcc_bin, "-print-file-name=libgomp.so"], text=True).strip()
                gomp_ok = os.path.isabs(gomp) and os.path.isfile(gomp)
            except Exception:
                gomp_ok = False
        arch_ok = platform.machine() == "x86_64"
        missing = []
        if not arch_ok:
            missing.append("x86_64 host CPU")
        if not mojo_bin:
            missing.append("mojo compiler (set INTENT_MOJO or add mojo to PATH)")
        if not (gcc_bin and gomp_ok):
            missing.append("gcc with GNU OpenMP (libgomp.so)")
        return {
            "ready": arch_ok and bool(mojo_bin) and bool(gcc_bin) and gomp_ok,
            "mojo": mojo_bin or "",
            "gcc": gcc_bin or "",
            "libgomp": gomp_ok,
            "optional_dependencies": native_doctor.get("mojo", {}),
            "missing": missing,
        }

    if name == "weft":
        cc_bin = shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc") or shutil.which("clang")
        scope = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        caps = tri_info.get("weft_capabilities") or {}
        ready = bool(tri_info["weft_core"] and tri_info["is_weft_triton"] and cc_bin)
        missing = []
        if not tri_info["weft_core"]:
            missing.append("weft_core package")
        if not tri_info["is_weft_triton"]:
            missing.append("triton-weft package")
        if not cc_bin:
            missing.append("C compiler (cc / gcc / clang)")
        if caps and scope and set(caps.get("cpus", ())) != set(scope):
            ready = False
            missing.append("CPU affinity mismatch with Weft native discovery")
        return {
            "ready": ready,
            "arch": platform.machine(),
            "cpu_affinity": scope,
            "cc": cc_bin or "",
            "target": caps.get("target") if caps else None,
            "coexistence_note": tri_info["coexistence_note"],
            "missing": missing,
        }

    if name == "bangc":
        ext = {}
        try:
            mod = compiler.runtime("bangc", "binding.target")
            if hasattr(mod, "external_dependencies"):
                ext = mod.external_dependencies()
        except Exception as exc:
            ext = {"error": str(exc)}
        if not ext:
            cncc = shutil.which("cncc")
            nodes = sorted(glob.glob("/dev/cambricon_dev*"))
            cnrt_ok = False
            try:
                ctypes.CDLL("libcnrt.so")
                cnrt_ok = True
            except OSError:
                pass
            ext = {
                "cncc": {"available": bool(cncc), "path": cncc or "", "external": True},
                "cnrt": {"available": cnrt_ok, "external": True},
                "driver": {"available": bool(nodes), "devices": nodes, "external": True},
            }
        torch_mlu_ok, mlu_count = False, 0
        try:
            import torch  # type: ignore
            import torch_mlu  # type: ignore  # noqa: F401

            torch_mlu_ok = hasattr(torch, "mlu") and bool(torch.mlu.is_available())
            if torch_mlu_ok:
                mlu_count = int(torch.mlu.device_count())
        except Exception:
            pass
        cncc_ok = bool(ext.get("cncc", {}).get("available"))
        cnrt_ok = bool(ext.get("cnrt", {}).get("available"))
        missing = []
        if not cncc_ok:
            missing.append("Cambricon cncc compiler on PATH (external dependency)")
        if not cnrt_ok:
            missing.append("Cambricon CNRT runtime libcnrt.so on LD_LIBRARY_PATH (external dependency)")
        if not torch_mlu_ok:
            missing.append("torch_mlu with an accessible local MLU device")
        return {
            "ready": cncc_ok and cnrt_ok and torch_mlu_ok and mlu_count > 0,
            "mlu_device_count": mlu_count,
            "optional_dependencies": native_doctor.get("bangc", {}),
            "external_dependencies": ext,
            "missing": missing,
        }

    raise ValueError(f"unknown backend: {name}")


def run_doctor(*, backend=None, library=None) -> dict:
    from compilerinfra import Compiler

    from . import __version__, library_path

    lib = library_path(library)
    pkg_file = Path(__file__).resolve()
    with Compiler(lib) as compiler:
        desc = compiler.describe()
        native_doc = compiler.doctor()
        all_exits = list(desc["exits"])
        targets = [backend] if backend else all_exits
        for t in targets:
            if t not in desc["exits"]:
                raise ValueError(f"unknown exit: {t}")
        tri_info = _probe_triton_and_weft()
        backends = {t: _probe_backend(t, compiler, native_doc, tri_info) for t in targets}

    return {
        "package": {
            "name": desc["product"]["name"],
            "version": __version__,
            "python_module": str(pkg_file.parent),
            "library": str(lib),
            "bundled": "_bundled" in lib.parts,
            "platform": f"{platform.system().lower()}_{platform.machine()}",
            "glibc": platform.libc_ver()[1],
        },
        "optional_dependencies": native_doc,
        "backends": backends,
        "ready_backends": [k for k, v in backends.items() if v.get("ready")],
    }


def run_setup(*, backend=None, library=None) -> dict:
    from compilerinfra import Compiler

    from . import library_path
    from .examples import relu

    doc = run_doctor(backend=backend, library=library)
    lib = library_path(library)
    verified_assets = 0
    with Compiler(lib) as compiler:
        desc = compiler.describe()
        assets_root = lib.parent / (desc["product"]["name"] + ".assets")
        for entry in desc["assets"]:
            p = (assets_root / entry["path"]).resolve(strict=True)
            if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"corrupted installed asset: {entry['path']}")
            verified_assets += 1
        root_bytes = compiler.root_ir(relu.source)

    hints = []
    b_map = doc["backends"]
    if "bangc" in b_map and not b_map["bangc"]["ready"]:
        hints.append(
            "For MLU (bangc): export PATH=/usr/local/neuware/bin:$PATH "
            "LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH"
        )
    if "mojo" in b_map and not b_map["mojo"]["ready"] and platform.machine() == "x86_64":
        hints.append("For Mojo (mojo): export INTENT_MOJO=/path/to/mojo (optional: ONEDNN_ROOT=/path/to/onednn)")
    if "cutile" in b_map and not b_map["cutile"]["ready"]:
        hints.append("For cuTile (cutile): install cuda-tile and ensure CUDA Toolkit bin (nvcc) is on PATH")

    return {
        "ok": True,
        "verified_assets": verified_assets,
        "frontend_smoke_bytes": len(root_bytes),
        "doctor": doc,
        "hints": hints,
    }


def read_skills(*, library=None) -> dict[str, str]:
    from compilerinfra import Compiler

    from . import library_path

    lib = library_path(library)
    out = {}
    with Compiler(lib) as compiler:
        desc = compiler.describe()
        assets_root = lib.parent / (desc["product"]["name"] + ".assets")
        for entry in desc["assets"]:
            if entry.get("kind") in {"frontend", "runtime"}:
                continue
            rel = entry["path"]
            if rel.endswith(".md"):
                p = (assets_root / rel).resolve(strict=True)
                data = p.read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError(f"skill checksum mismatch: {rel}")
                out[rel] = data.decode("utf-8")
    return out


def read_manual(topic=None, *, lang="zh") -> str:
    if lang not in ("zh", "en"):
        raise ValueError("lang must be 'zh' or 'en'")
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "_manual" / f"manual.{lang}.md",
        pkg_dir.parents[2] / "doc" / f"manual.{lang}.md",
    ]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"packaged manual.{lang}.md not found")
    text = path.read_text(encoding="utf-8")
    if not topic:
        return text
    key = topic.strip().lower()
    sections = text.split("\n## ")
    matched = [sections[0]]
    for sec in sections[1:]:
        header = sec.splitlines()[0].lower()
        if key in header:
            matched.append("## " + sec)
    if len(matched) == 1:
        raise ValueError(
            f"unknown manual topic {topic!r}; choose from: install, language, model, components"
        )
    return "\n\n".join(matched)
