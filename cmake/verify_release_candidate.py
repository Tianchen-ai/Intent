#!/usr/bin/env python3
"""Repeatable release-candidate full verification & scoreboard pipeline for V2-6.R.

Covers:
  1. Standalone clean-venv RC wheel installation (`setup`, `doctor`, offline bilingual `manual`, `skills`,
     and isolated `_bundled/lib/libintentdsl.so` verification with `PYTHONPATH` / `INTENTDSL_LIBRARY` unset).
  2. Full Bench suite execution across:
     - 30 `program/*` workloads (31 cases: strict + `allow_subnormal_ftz` `batched_gemm`) × 5 exits × applicable devices
     - 15 additional registered V2 workloads (`weighted_rms_norm`, multi-shape `softmax`, `cpu/histogram`,
       and 12 `corpus/*` workloads = 22 cases -> 45 workloads / 53 cases per backend-device target)
     - All applicable V1 (`intent:<exit>`) and mature implementations registered on each environment.
  3. Six-stage accounting (`install`, `generate`, `native_compile`, `execute`, `correctness`, `performance`)
     with explicit preservation of all failure and skip reasons.
  4. Scoreboard aggregation (`V2 vs V1` and `V2 vs mature` warm operator medians, plus all `V2/V1 > 1.2x`
     items annotated with their responsible compiler unit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


PROGRAMS_30: tuple[str, ...] = (
    # Group 1 (6)
    "program/relu",
    "program/swiglu",
    "program/rope",
    "program/transpose",
    "program/dropout",
    "program/adamw",
    # Group 2 (4)
    "program/softmax",
    "program/layer_norm",
    "program/batch_norm",
    "program/group_norm_backward",
    # Group 3 (7)
    "program/cumsum",
    "program/embedding",
    "program/csr_spmv",
    "program/jagged_mean",
    "program/nonzero",
    "program/histogram",
    "program/moe_alignment",
    # Group 4 (6)
    "program/gemm",
    "program/batched_gemm",
    "program/ragged_gemm",
    "program/int8_gemm",
    "program/q4_projection",
    "program/fp8_gemm",
    # Group 5 (5)
    "program/gated_delta",
    "program/cholesky",
    "program/causal_convolution",
    "program/causal_linear_attention",
    "program/mamba_chunk_state",
    # Group 6 (2)
    "program/attention",
    "program/paged_decode",
)

ADDITIONAL_V2_WORKLOADS: tuple[str, ...] = (
    "weighted_rms_norm",
    "softmax",
    "cpu/histogram",
    # Group 1-3/5 corpus (6)
    "corpus/triton/fused_add_rms_norm",
    "corpus/mojo/dropout_residual_rms_norm",
    "corpus/mojo/layer_norm_backward",
    "corpus/mojo/rotary_embedding_equivalent_index",
    "corpus/triton/flaggems_histogram",
    "corpus/mojo/selective_state_scan",
    # Group 4 contraction corpus (3)
    "corpus/mojo/qkv_projection",
    "corpus/triton/moe_expert_projection",
    "corpus/weft/q4_k_projection",
    # Group 6 mixed corpus (3)
    "corpus/mojo/paged_mla_decode_partials",
    "corpus/cutile/grouped_flash_decode",
    "corpus/triton/flash_attention_backward",
)

ALL_V2_WORKLOADS: tuple[str, ...] = PROGRAMS_30 + ADDITIONAL_V2_WORKLOADS

G3_FIXTURE_CORPUS: dict[str, list[tuple[float, float]]] = {
    "corpus/triton/fused_add_rms_norm": [(0.02, 0.01)],
    "corpus/mojo/dropout_residual_rms_norm": [(0.065, 0.0)],
    "corpus/mojo/layer_norm_backward": [(0.05, 0.0), (0.125, 0.0), (0.125, 0.0)],
    "corpus/mojo/rotary_embedding_equivalent_index": [(0.01, 0.0)],
    "corpus/triton/flaggems_histogram": [(0.0, 0.0)],
    "corpus/mojo/selective_state_scan": [(2e-5, 0.0)],
}

G4_FIXTURE_CORPUS: dict[str, tuple[float, float]] = {
    "corpus/mojo/qkv_projection": (0.02, 0.01),
    "corpus/triton/moe_expert_projection": (0.1, 0.05),
}

G6_FIXTURE_CORPUS: dict[str, tuple[Any, float]] = {
    "corpus/mojo/paged_mla_decode_partials": (0.05, 0.05),
    "corpus/cutile/grouped_flash_decode": (0.1, 0.05),
    "corpus/triton/flash_attention_backward": ((0.125, 0.25, 0.125), 0.0),
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    profile: str
    backends: tuple[str, ...]
    device_kind: str  # "cuda", "mlu", "cpu_x86", "cpu_rvv"
    wheel_glob: str
    base_python: str
    host: str | None = None
    remote_output_root: str | None = None
    standalone_env: tuple[tuple[str, str], ...] = ()
    worker_env: tuple[tuple[str, str], ...] = ()
    worker_launcher: tuple[str, ...] = ()
    extra_site_dirs: tuple[str, ...] = ()
    captures_layout: str = "remote_v262"  # "local_flat" or "remote_v262"
    needs_controller_oracle: bool = False


TARGETS: dict[str, TargetSpec] = {
    "mlu370": TargetSpec(
        name="mlu370",
        profile="mlu-native",
        backends=("bangc",),
        device_kind="mlu",
        wheel_glob="intentdsl-2.0.0-*-manylinux_2_31_x86_64.whl",
        base_python="/srv/rc/site-2/intentdsl_bench/output/build/runtime/bin/python",
        host="rc-user@rc-host.example.invalid",
        remote_output_root="/srv/rc/site-2/intentdsl_bench/output",
        standalone_env=(
            ("NEUWARE_HOME", "/usr/local/neuware"),
            ("PATH", "/usr/local/neuware/bin:/usr/local/bin:/usr/bin:/bin"),
        ),
        worker_env=(
            ("NEUWARE_HOME", "/usr/local/neuware"),
            ("PATH", "/usr/local/neuware/bin:/usr/local/bin:/usr/bin:/bin"),
        ),
        worker_launcher=(
            "/srv/rc/site-2/.cache/intentdsl/tmo-pt25-runtime/system-libs/ld-linux-x86-64.so.2",
            "--library-path",
            "/srv/rc/site-2/.cache/intentdsl/tmo-pt25-runtime/system-libs:"
            "/srv/rc/site-2/.cache/intentdsl/tmo-pt25-runtime/neuware/lib64:"
            "/srv/rc/site-2/.cache/intentdsl/tmo-pt25-runtime/site-packages/torch/lib",
        ),
        extra_site_dirs=(
            "/srv/rc/site-2/.cache/intentdsl/tmo-pt25-runtime/site-packages",
            "/srv/rc/site-2/intentdsl_bench/output/build/runtime/lib/python3.10/site-packages",
        ),
        captures_layout="remote_v262",
    ),
    "5090d": TargetSpec(
        name="5090d",
        profile="local-gpu",
        backends=("triton", "cutile"),
        device_kind="cuda",
        wheel_glob="intentdsl-2.0.0-*-linux_x86_64.whl",
        base_python="/srv/rc/site-1/.venvs/intentdsl-cutile/bin/python",
        worker_env=(("TMPDIR", "/tmp"), ("INTENT_CACHE_DIR", "/srv/rc/site-1/phdworks/intentdsl_bench/output/cache/v26-rc-5090d")),
        captures_layout="local_flat",
    ),
    "h100": TargetSpec(
        name="h100",
        profile="h100",
        backends=("triton", "cutile"),
        device_kind="cuda",
        wheel_glob="intentdsl-2.0.0-*-manylinux_2_31_x86_64.whl",
        base_python="/srv/rc/site-1/.venvs/intentdsl-cutile/bin/python",
        host="rc-host.example.invalid",
        remote_output_root="/srv/rc/site-1/phdworks/intentdsl_bench/output",
        captures_layout="remote_v262",
    ),
    "x86-mojo": TargetSpec(
        name="x86-mojo",
        profile="local-cpu",
        backends=("mojo",),
        device_kind="cpu_x86",
        wheel_glob="intentdsl-2.0.0-*-linux_x86_64.whl",
        base_python="/srv/rc/site-1/.venvs/intentdsl-cutile/bin/python",
        standalone_env=(
            ("PATH", "/srv/rc/site-1/.venvs/intentdsl-mojo/bin:/usr/local/bin:/usr/bin:/bin"),
            ("ONEDNN_ROOT", "/opt/rc-dependencies/onednn-3.13.4"),
            ("LD_LIBRARY_PATH", "/opt/rc-dependencies/onednn-3.13.4/lib:/srv/rc/site-1/phdworks/intentdsl_bench/output/build/onednn-install/lib"),
        ),
        worker_env=(
            ("PATH", "/srv/rc/site-1/.venvs/intentdsl-mojo/bin:/usr/local/bin:/usr/bin:/bin"),
            ("ONEDNN_ROOT", "/opt/rc-dependencies/onednn-3.13.4"),
            ("LD_LIBRARY_PATH", "/opt/rc-dependencies/onednn-3.13.4/lib:/srv/rc/site-1/phdworks/intentdsl_bench/output/build/onednn-install/lib"),
            ("INTENT_CPU_THREADS", "8"),
            ("OPENBLAS_NUM_THREADS", "1"),
        ),
        worker_launcher=("taskset", "-c", "0-7"),
        captures_layout="local_flat",
    ),
    "x86-weft": TargetSpec(
        name="x86-weft",
        profile="weft-x86",
        backends=("weft",),
        device_kind="cpu_x86",
        wheel_glob="intentdsl-2.0.0-*-linux_x86_64.whl",
        base_python="/srv/rc/site-1/phdworks/intentdsl_bench/output/work/v2-weft-env/bin/python",
        standalone_env=(
            ("ONEDNN_ROOT", "/opt/rc-dependencies/onednn-3.13.4"),
            ("LD_LIBRARY_PATH", "/opt/rc-dependencies/onednn-3.13.4/lib"),
        ),
        worker_env=(
            ("ONEDNN_ROOT", "/opt/rc-dependencies/onednn-3.13.4"),
            ("LD_LIBRARY_PATH", "/opt/rc-dependencies/onednn-3.13.4/lib"),
            ("INTENT_CPU_THREADS", "8"),
            ("OPENBLAS_NUM_THREADS", "1"),
        ),
        worker_launcher=("taskset", "-c", "0-7"),
        captures_layout="local_flat",
    ),
    "sg2044": TargetSpec(
        name="sg2044",
        profile="rvv-native",
        backends=("weft",),
        device_kind="cpu_rvv",
        wheel_glob="intentdsl-2.0.0-*-linux_riscv64.whl",
        base_python="/srv/rc/site-3/intentdsl_bench/output/work/v2-weft-env/bin/python",
        host="rc-host.example.invalid",
        remote_output_root="/srv/rc/site-3/intentdsl_bench/output",
        standalone_env=(
            (
                "LD_LIBRARY_PATH",
                "/srv/rc/site-3/intentdsl_bench/output/work/weft-env/lib/bench-deps:"
                "/usr/local/llvm-22.1.8/lib:/srv/rc/site-3/.cache/weft/toolchains/llvm20-build/lib",
            ),
        ),
        worker_env=(
            (
                "LD_LIBRARY_PATH",
                "/srv/rc/site-3/intentdsl_bench/output/work/weft-env/lib/bench-deps:"
                "/usr/local/llvm-22.1.8/lib:/srv/rc/site-3/.cache/weft/toolchains/llvm20-build/lib",
            ),
            ("INTENT_CPU_THREADS", "1"),
            ("OPENBLAS_NUM_THREADS", "1"),
        ),
        worker_launcher=("taskset", "-c", "48"),
        captures_layout="remote_v262",
        needs_controller_oracle=True,
    ),
    "rvv-v100": TargetSpec(
        name="rvv-v100",
        profile="rvv-v100",
        backends=("weft",),
        device_kind="cpu_rvv",
        wheel_glob="intentdsl-2.0.0-*-linux_riscv64.whl",
        base_python="/srv/rc/site-4/intentdsl_bench/output/work/weft-env/bin/python",
        host="rc-host.example.invalid",
        remote_output_root="/srv/rc/site-4/intentdsl_bench/output",
        standalone_env=(
            (
                "LD_LIBRARY_PATH",
                "/srv/rc/site-4/intentdsl_bench/output/work/weft-env/lib/bench-deps:"
                "/usr/local/llvm-22.1.8/lib:/srv/rc/site-3/.cache/weft/toolchains/llvm20-build/lib",
            ),
        ),
        worker_env=(
            (
                "LD_LIBRARY_PATH",
                "/srv/rc/site-4/intentdsl_bench/output/work/weft-env/lib/bench-deps:"
                "/usr/local/llvm-22.1.8/lib:/srv/rc/site-3/.cache/weft/toolchains/llvm20-build/lib",
            ),
            ("INTENT_CPU_THREADS", "1"),
            ("OPENBLAS_NUM_THREADS", "1"),
        ),
        worker_launcher=("taskset", "-c", "0"),
        captures_layout="remote_v262",
        needs_controller_oracle=True,
    ),
    "k3-x100": TargetSpec(
        name="k3-x100",
        profile="k3-x100",
        backends=("weft",),
        device_kind="cpu_rvv",
        wheel_glob="intentdsl-2.0.0-*-linux_riscv64.whl",
        base_python="/srv/rc/site-1/intentdsl_bench/output/work/weft-env/bin/python",
        host="rc-user@rc-host.example.invalid",
        remote_output_root="/srv/rc/site-1/intentdsl_bench/output",
        worker_env=(
            ("LD_LIBRARY_PATH", "/srv/rc/site-1/weft-bench/k3/current/paper/deps/llama-vendor-normal/bin"),
            ("INTENT_CPU_THREADS", "1"),
            ("OPENBLAS_NUM_THREADS", "1"),
        ),
        worker_launcher=(
            "/srv/rc/site-1/intentdsl_bench/output/work/weft-env/bin/python",
            "-m",
            "triton.tools.cpu",
            "--cpus",
            "0",
            "--exec",
            "--",
        ),
        captures_layout="remote_v262",
        needs_controller_oracle=True,
    ),
    "k3-a100": TargetSpec(
        name="k3-a100",
        profile="k3-a100",
        backends=("weft",),
        device_kind="cpu_rvv",
        wheel_glob="intentdsl-2.0.0-*-linux_riscv64.whl",
        base_python="/srv/rc/site-1/intentdsl_bench/output/work/weft-env/bin/python",
        host="rc-user@rc-host.example.invalid",
        remote_output_root="/srv/rc/site-1/intentdsl_bench/output",
        worker_env=(
            ("LD_LIBRARY_PATH", "/srv/rc/site-1/weft-bench/k3/current/paper/deps/llama-vendor-normal/bin"),
            ("INTENT_CPU_THREADS", "1"),
            ("OPENBLAS_NUM_THREADS", "1"),
        ),
        worker_launcher=(
            "/srv/rc/site-1/intentdsl_bench/output/work/weft-env/bin/python",
            "-m",
            "triton.tools.cpu",
            "--cpus",
            "8",
            "--exec",
            "--",
        ),
        captures_layout="remote_v262",
        needs_controller_oracle=True,
    ),
}


def responsible_unit(backend: str, workload: str) -> str:
    """Map (exit backend, workload) to the owning compiler unit(s) per doc/plan/v2-6.md §2–§3."""
    w = workload
    if backend == "bangc":
        if "group_norm_backward" in w:
            return "levels/execution/block + levels/storage/dsa (block@realized 8-channel stripe unrolling & 32-group affine reduction)"
        if "softmax" in w or "adamw" in w or "swiglu" in w or "rope" in w or "dropout" in w:
            return "exits/bangc (0-ULP IEEE div/sqrt per-element contract vs V1 approximate reciprocal)"
        if "gemm" in w or "projection" in w or "mamba" in w:
            return "levels/storage/dsa + exits/bangc (WRAM matrix packing & BF16 guard)"
        if "csr_spmv" in w or "embedding" in w or "jagged" in w or "histogram" in w or "nonzero" in w or "moe_alignment" in w:
            return "levels/storage/dsa + exits/bangc (irregular indexed_dma & segmented staging)"
        if "attention" in w or "decode" in w:
            return "levels/storage/dsa + exits/bangc (panel-fold NRAM staging & online rescale)"
        return "levels/storage/dsa + exits/bangc"
    if backend in ("triton", "cutile"):
        if "flash_attention_backward" in w or "attention" in w or "decode" in w:
            return f"levels/execution/block + exits/{backend} (panel exchange, local residency & pipeline scheduling)"
        if "histogram" in w or "csr_spmv" in w or "jagged_mean" in w or "nonzero" in w or "moe_alignment" in w:
            return f"levels/execution/block + exits/{backend} (irregular gather grouping & CTA partial atomic aggregation)"
        if "gemm" in w or "projection" in w or "mamba" in w:
            return f"levels/execution/block + exits/{backend} (small-tile launch occupancy, scale/record decode & MMA staging)"
        if "gated_delta" in w or "cholesky" in w or "causal_" in w or "selective_state" in w:
            return f"levels/execution/block + exits/{backend} (ordered recurrence carry & window load)"
        if "group_norm_backward" in w or "layer_norm_backward" in w or "rms_norm" in w or "softmax" in w or "batch_norm" in w:
            return f"levels/execution/block + exits/{backend} (multi-reduction bucket & warp/tile collective)"
        return f"levels/execution/block + exits/{backend}"
    if backend in ("mojo", "weft"):
        if "dropout_residual_rms_norm" in w or "dropout" in w:
            return f"levels/execution/cpu + exits/{backend} (two-pass Philox recomputation & BF16 widening)"
        if "group_norm_backward" in w or "layer_norm_backward" in w or "batch_norm" in w or "rms_norm" in w or "softmax" in w:
            return f"levels/execution/cpu + exits/{backend} (task partitioning, BF16 vector reduction & shared gain load)"
        if "gemm" in w or "projection" in w or "mamba" in w:
            return f"levels/execution/cpu + exits/{backend} (panel packing traffic & SIMD/BRGeMM/RVV microkernel)"
        if "attention" in w or "decode" in w:
            return f"levels/execution/cpu + exits/{backend} (owner packing & online softmax horizontal reduction)"
        if "moe_alignment" in w or "cumsum" in w or "csr_spmv" in w or "jagged_mean" in w or "histogram" in w or "nonzero" in w:
            return f"levels/execution/cpu + exits/{backend} (per-task key reservation & indirect gather)"
        return f"levels/execution/cpu + exits/{backend}"
    return f"exits/{backend}"


# ---------------------------------------------------------------------------
# On-host standalone check & per-workload Bench worker
# ---------------------------------------------------------------------------

def run_standalone_check(backends: list[str], out_json: Path, *, require_doctor_ready: bool = False) -> int:
    """Verify the installed RC wheel in a clean environment without Bench/source paths."""
    assert "PYTHONPATH" not in os.environ, f"PYTHONPATH leaked: {os.environ.get('PYTHONPATH')}"
    assert "INTENTDSL_LIBRARY" not in os.environ, f"INTENTDSL_LIBRARY leaked: {os.environ.get('INTENTDSL_LIBRARY')}"
    import compilerinfra
    import intentdsl as I

    venv_prefix = Path(sys.prefix).resolve()
    mod_file = Path(I.__file__).resolve()
    ci_file = Path(compilerinfra.__file__).resolve()
    lib_path = Path(I.library_path()).resolve()
    assert mod_file.is_relative_to(venv_prefix), f"intentdsl loaded outside venv: {mod_file} vs {venv_prefix}"
    assert ci_file.is_relative_to(venv_prefix), f"compilerinfra loaded outside venv: {ci_file} vs {venv_prefix}"
    assert "_bundled" in lib_path.parts and lib_path.is_file(), f"unexpected library_path: {lib_path}"

    lay = I.layout()
    assert set(lay.get("exits", ())) == {"triton", "cutile", "mojo", "weft", "bangc"}, lay

    zh = I.manual("components", lang="zh")
    en = I.manual("components", lang="en")
    assert "compilerinfra-dev" in zh and "compilerinfra-dev" in en
    sk = I.skills()
    assert "agent/skills/intentdsl-development.md" in sk

    setup_reports = {}
    doctor_reports = {}
    ok = True
    for be in backends:
        sr = I.setup(backend=be)
        dr = I.doctor(backend=be)
        setup_reports[be] = sr
        doctor_reports[be] = dr
        be_ready = bool(dr.get("backends", {}).get(be, {}).get("ready")) or (be in dr.get("ready_backends", ()))
        if not sr.get("ok") or int(sr.get("verified_assets", 0)) < 1:
            ok = False
        if require_doctor_ready and not be_ready:
            ok = False

    phase_data = {
        "status": "passed" if ok else "failed",
        "require_doctor_ready": require_doctor_ready,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "glibc": platform.libc_ver(),
        "venv_prefix": str(venv_prefix),
        "intentdsl_file": str(mod_file),
        "compilerinfra_file": str(ci_file),
        "library_path": str(lib_path),
        "skills": list(sk),
        "setup": setup_reports,
        "doctor": doctor_reports,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if require_doctor_ready and out_json.is_file():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        phase_data["pure_wheel_phase"] = {
            "status": prev.get("status"),
            "glibc": prev.get("glibc"),
            "platform": prev.get("platform"),
            "verified_assets": {b: prev.get("setup", {}).get(b, {}).get("verified_assets") for b in backends},
        }
        if prev.get("status") != "passed":
            phase_data["status"] = "failed"
            ok = False
    out_json.write_text(json.dumps(phase_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tag = "DOCTOR_READY_CHECK" if require_doctor_ready else "PURE_WHEEL_CHECK"
    print(f"{tag}:", phase_data["status"], lib_path, flush=True)
    return 0 if ok else 1


def _select_implementations(
    workload_name: str,
    available_impls: dict[str, Any],
    backends: list[str],
    device_kind: str,
    target_name: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Choose participants for `benchmark()` and record explicit skip reasons for non-applicable ones."""
    selected: list[str] = []
    skipped: list[dict[str, str]] = []

    # 1. V2 candidates first (cleanest device state)
    for be in backends:
        k = f"intent-v2:{be}"
        if k in available_impls:
            selected.append(k)

    # 2. V1 counterparts second
    for be in backends:
        k = f"intent:{be}"
        if k in available_impls:
            selected.append(k)

    # 3. Mature same-device baselines
    allowed_mature: dict[str, set[str]] = {
        "cuda": {"torch", "triton", "cutile", "cub", "aten:accelerator"},
        "mlu": {"cnnl", "torch", "aten:accelerator"},
        "cpu_x86": {"aten:cpu", "mojo", "weft", "ggml", "inductor:cpu", "vllm:cpu"},
        "cpu_rvv": {"weft", "ggml"},
    }
    mature_for_kind = allowed_mature[device_kind]

    for impl in available_impls:
        if impl in selected:
            continue
        if impl.startswith(("intent-v2:", "intent:")):
            # Other exit backends (evaluated on their own target, or unconfigured Ascend)
            if impl == "intent:triton_ascend":
                skipped.append({"implementation": impl, "reason": "no Ascend NPU host in active testbed"})
            elif impl.endswith(("-direct", "-double")):
                skipped.append({"implementation": impl, "reason": "ablation variant; default policy evaluated via intent-v2:bangc"})
            continue
        if impl == "numpy":
            skipped.append({"implementation": impl, "reason": "Python NumPy reference does not expose native operator timer"})
            continue
        if device_kind == "cpu_x86" and impl == "mojo" and "mojo" not in backends:
            skipped.append({"implementation": impl, "reason": "Mojo baseline runs in x86-mojo environment"})
            continue
        if device_kind == "cpu_x86" and impl == "weft" and "weft" not in backends:
            skipped.append({"implementation": impl, "reason": "Weft baseline runs in x86-weft environment"})
            continue
        if device_kind in ("cpu_x86", "cpu_rvv") and impl == "torch":
            skipped.append({"implementation": impl, "reason": "eager Python torch on CPU lacks operator-scope native timer; aten:cpu covers native libtorch"})
            continue
        if device_kind == "mlu" and impl == "torch" and workload_name in ("weighted_rms_norm", "softmax"):
            skipped.append({"implementation": impl, "reason": "row workload torch adapter lacks MLU CNRT notifier timer; cnnl provides native MLU baseline"})
            continue
        if device_kind == "mlu" and impl == "torch" and workload_name in ("program/jagged_mean", "program/cholesky", "program/int8_gemm"):
            skipped.append({"implementation": impl, "reason": "torch_mlu falls back to host CPU for this operator on MLU370; not an on-device MLU implementation"})
            continue
        if device_kind == "cuda" and impl == "torch" and workload_name.startswith("corpus/mojo/"):
            skipped.append({"implementation": impl, "reason": "Mojo corpus source 'torch' executes on host CPU without CUDA or native timer"})
            continue
        if impl in mature_for_kind:
            selected.append(impl)
        else:
            skipped.append(
                {
                    "implementation": impl,
                    "reason": f"requires a different execution environment than {target_name} ({device_kind})",
                }
            )
    return selected, skipped


def _prepare_fixture_workload(w: Any, work: str, dev: str, captures_layout: str, bench_output_root: Path):
    """Bind pre-captured input/oracle trees when running cross-provider `corpus/*` workloads."""
    from intent_bench.storage import read_tree
    from intent_bench.trees import array
    import numpy as np

    if captures_layout == "local_flat":
        g3_in = bench_output_root / "work" / "v262-inputs" / work
        g3_ex = bench_output_root / "work" / "v262-expected" / work
    else:
        g3_in = bench_output_root / "work" / "v262" / "inputs" / work
        g3_ex = bench_output_root / "work" / "v262" / "expected" / work
    g4_dir = bench_output_root / "work" / "g4-fixtures" / work
    g6_dir = bench_output_root / "work" / "g6-fixtures" / work

    if work in G3_FIXTURE_CORPUS:
        if not g3_in.exists() or not g3_ex.exists():
            raise FileNotFoundError(f"fixture_missing: {g3_in} or {g3_ex} not staged on host")
        ts = G3_FIXTURE_CORPUS[work]
        ref = read_tree(g3_ex, host_arrays=True)

        def check_g3(v, expected):
            if work.endswith("selective_state_scan"):
                assert str(v.dtype).removeprefix("torch.") == expected.dtype.name and tuple(v.shape) == expected.shape
                return np.testing.assert_allclose(array(v), expected, atol=2e-5, rtol=0)
            if work.endswith("flaggems_histogram"):
                assert str(v.dtype).removeprefix("torch.") == expected.dtype.name
                return np.testing.assert_array_equal(array(v), expected)
            if work.endswith("rotary_embedding_equivalent_index"):
                assert str(v.dtype).removeprefix("torch.") == expected.dtype.name and tuple(v.shape) == expected.shape
                return np.testing.assert_allclose(array(v).astype("float32"), expected.astype("float32"), atol=0.01, rtol=0)
            assert len(v) == len(expected)
            for i, (x, y) in enumerate(zip(v, expected)):
                assert str(x.dtype).removeprefix("torch.") == y.dtype.name, (x.dtype, y.dtype)
                xa = array(x)
                assert xa.shape == y.shape
                at, rt = ts[min(i, len(ts) - 1)]
                np.testing.assert_allclose(xa.astype("float32"), y.astype("float32"), rtol=rt, atol=at)

        return replace(
            w,
            inputs=lambda c, s: read_tree(g3_in, host_arrays=(dev == "cpu")),
            oracle=lambda c, i: ref,
            check=check_g3,
        )

    if work in G4_FIXTURE_CORPUS:
        if not (g4_dir / "inputs").exists():
            raise FileNotFoundError(f"fixture_missing: {g4_dir} not staged on host")
        at, rt = G4_FIXTURE_CORPUS[work]

        def check_g4(actual, expected):
            actual_dtype = str(actual.dtype).removeprefix("torch.")
            if actual_dtype != expected.dtype.name or tuple(actual.shape) != expected.shape:
                raise AssertionError((actual_dtype, actual.shape, expected.dtype, expected.shape))
            a = array(actual).reshape(-1)
            b = expected.reshape(-1)
            for i in range(0, len(a), 1 << 20):
                np.testing.assert_allclose(
                    a[i : i + (1 << 20)].astype("f"),
                    b[i : i + (1 << 20)].astype("f"),
                    atol=at,
                    rtol=rt,
                    equal_nan=True,
                )

        return replace(
            w,
            inputs=lambda c, s: read_tree(g4_dir / "inputs", host_arrays=(dev == "cpu"), tensor_device=dev),
            oracle=lambda c, v: read_tree(g4_dir / "expected", host_arrays=True),
            check=check_g4,
        )

    if work in G6_FIXTURE_CORPUS:
        if not (g6_dir / "inputs").exists():
            raise FileNotFoundError(f"fixture_missing: {g6_dir} not staged on host")
        at, rt = G6_FIXTURE_CORPUS[work]

        def single(actual, expected, tol_a):
            actual_dtype = str(actual.dtype).removeprefix("torch.")
            if actual_dtype != expected.dtype.name or tuple(actual.shape) != expected.shape:
                raise AssertionError((actual_dtype, actual.shape, expected.dtype, expected.shape))
            a = array(actual).reshape(-1)
            b = expected.reshape(-1)
            for i in range(0, len(a), 1 << 20):
                np.testing.assert_allclose(
                    a[i : i + (1 << 20)].astype("f"),
                    b[i : i + (1 << 20)].astype("f"),
                    atol=tol_a,
                    rtol=rt,
                    equal_nan=True,
                )

        def check_g6(actual, expected):
            if isinstance(at, tuple):
                assert len(actual) == len(expected) == len(at)
                for x, y, t in zip(actual, expected, at):
                    single(x, y, t)
            else:
                single(actual, expected, at)

        return replace(
            w,
            inputs=lambda c, s: read_tree(g6_dir / "inputs", host_arrays=(dev == "cpu"), tensor_device=dev),
            oracle=lambda c, v: read_tree(g6_dir / "expected", host_arrays=True),
            check=check_g6,
        )

    return w


def _needs_fixture_override(work: str, device_kind: str, backends: list[str]) -> bool:
    if not work.startswith("corpus/") or work == "corpus/weft/q4_k_projection":
        return False
    if device_kind in ("mlu", "cpu_rvv"):
        return True
    if device_kind == "cuda":
        # CUDA natively runs corpus/triton/*, corpus/cutile/*, and legacy_norms (dropout_residual, layer_norm_backward).
        # Mojo-only catalog workloads use CPU source/oracle unless fixture-bound or native.
        if work in ("corpus/mojo/qkv_projection", "corpus/mojo/paged_mla_decode_partials"):
            return True
        return False
    if device_kind == "cpu_x86":
        if "mojo" in backends and work.startswith("corpus/mojo/"):
            return False
        return True
    return False


def run_single_workload(cfg: dict[str, Any]) -> int:
    """Execute one Bench workload inside the target host's clean RC venv."""
    assert "PYTHONPATH" not in os.environ, "PYTHONPATH must be unset"
    assert "INTENTDSL_LIBRARY" not in os.environ, "INTENTDSL_LIBRARY must be unset"
    import intentdsl as I

    venv_prefix = Path(sys.prefix).resolve()
    assert Path(I.__file__).resolve().is_relative_to(venv_prefix), f"intentdsl not from RC venv: {I.__file__}"

    target_name: str = cfg["target_name"]
    device_kind: str = cfg["device_kind"]
    backends: list[str] = cfg["backends"]
    work: str = cfg["workload"]
    profile_path: str = cfg["profile_path"]
    bench_output_root = Path(cfg["bench_output_root"])
    exp_dir = Path(cfg["exp_dir"])
    captures_layout: str = cfg["captures_layout"]
    needs_controller_oracle: bool = cfg["needs_controller_oracle"]

    os.environ["INTENT_BENCH_OUTPUT_ROOT"] = str(bench_output_root)
    if device_kind == "mlu":
        import torch_mlu  # noqa: F401

    from intent_bench.engine import benchmark
    from intent_bench.environments import load_environment
    from intent_bench.loading import load_workload
    from intent_bench.model import Measurement, Result
    from intent_bench.output import run_output
    from intent_bench.results import save

    env = load_environment(profile_path, python=sys.executable)
    env = replace(env, transport="local", host=None, output_root=bench_output_root)
    if not env.compiler and "INTENT_COMPILER" not in os.environ:
        try:
            import intent as _v1_intent  # type: ignore

            _v1_bin = Path(_v1_intent.__file__).resolve().parent / "_bin" / "intent-compile"
            if _v1_bin.is_file():
                os.environ["INTENT_COMPILER"] = str(_v1_bin)
                env = replace(env, compiler=str(_v1_bin))
        except Exception:
            pass
    if env.cpus:
        allowed = os.sched_getaffinity(0)
        wanted = set(env.cpus) & allowed
        if wanted:
            os.sched_setaffinity(0, wanted)
            env = replace(env, cpus=tuple(sorted(wanted)))
    scope_cpus = sorted(os.sched_getaffinity(0))

    if work == "corpus/weft/q4_k_projection" and env.device == "cpu":
        os.environ["INTENT_CPU_THREADS"] = "1"
        os.sched_setaffinity(0, scope_cpus[:1])
        env = replace(env, cpus=(scope_cpus[0],), threads=1)
    else:
        os.environ.setdefault("INTENT_CPU_THREADS", str(env.threads))

    w = load_workload(work)
    selected, skipped = _select_implementations(work, w.implementations, backends, device_kind, target_name)

    use_fixture = _needs_fixture_override(work, device_kind, backends)
    out_folder = exp_dir / work
    if use_fixture:
        try:
            w = _prepare_fixture_workload(w, work, env.device, captures_layout, bench_output_root)
            # When using cross-host fixture, filter out provider-specific source baselines that expect CUDA/Mojo contexts
            keep = []
            for impl in selected:
                if impl.startswith("intent-v2:"):
                    keep.append(impl)
                elif impl.startswith("intent:") and work in (
                    "corpus/triton/fused_add_rms_norm",
                    "corpus/mojo/dropout_residual_rms_norm",
                    "corpus/mojo/layer_norm_backward",
                ) and device_kind != "cpu_rvv":
                    keep.append(impl)
                elif impl == "aten:accelerator" and device_kind in ("cuda", "mlu") and work == "corpus/mojo/layer_norm_backward":
                    keep.append(impl)
                else:
                    skipped.append(
                        {
                            "implementation": impl,
                            "reason": f"cross-provider fixture input used on {target_name}; {impl} requires origin runtime context",
                        }
                    )
            selected = keep
        except FileNotFoundError as exc:
            out_folder.mkdir(parents=True, exist_ok=True)
            rows = [
                Result(
                    w.name,
                    w.family,
                    c.name,
                    dict(c.parameters),
                    impl,
                    status="skipped_fixture_missing",
                    correctness="unchecked",
                    stage="inputs",
                    device=env.device,
                    error=str(exc),
                )
                for c in w.cases
                for impl in selected
            ]
            save(rows, out_folder, command={"workload": work, "skipped_implementations": skipped})
            print(f"[{target_name}] {work}: SKIPPED ({exc})", flush=True)
            return 0

    if device_kind == "cpu_rvv" and work in ("corpus/mojo/qkv_projection", "corpus/triton/moe_expert_projection"):
        meas = Measurement(enabled=True, warmup=0, samples=1)
    else:
        meas = Measurement(enabled=True)

    anchor = f"intent-v2:{backends[0]}"
    baseline_name = anchor if anchor in selected else None

    # Child mode for per-(case, implementation) process isolation on multi-thread x86 CPU:
    # avoids cross-runtime OpenMP/threadpool spin contention (libgomp vs libomp vs libiomp5).
    if cfg.get("_isolated_impl"):
        from dataclasses import asdict as _dc_asdict
        from intent_bench.storage import write_tree

        iso_case_name: str = cfg["_isolated_case"]
        iso_impl: str = cfg["_isolated_impl"]
        iso_actual_dir = Path(cfg["_isolated_actual_dir"])
        iso_row_json = Path(cfg["_isolated_row_json"])
        iso_work_dir = Path(cfg["_isolated_work_dir"])

        def _capture_actual(actual, _expected):
            write_tree(actual, iso_actual_dir)

        w_iso = replace(
            w,
            cases=tuple(c for c in w.cases if c.name == iso_case_name),
            oracle=lambda _c, _i: None,
            check=_capture_actual,
        )
        with run_output(env, w.name, iso_work_dir, execution=True):
            sub_rows = benchmark(
                w_iso,
                [iso_impl],
                environment=env,
                baseline=iso_impl,
                measurement=meas,
            )
        iso_row_json.write_text(json.dumps([_dc_asdict(r) for r in sub_rows]) + "\n", encoding="utf-8")
        return 0

    if device_kind == "cpu_x86" and selected:
        from intent_bench.engine import assign_speedups
        from intent_bench.measurement import device_window, reset as _bench_reset, synchronize as _bench_sync
        from intent_bench.model import DeviceBusy, Implementation
        from intent_bench.storage import read_tree
        from intent_bench.trees import assert_close, clone

        clean_sub_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "INTENTDSL_LIBRARY")}
        all_rows: list[Result] = []
        with run_output(env, w.name, out_folder, execution=True) as folder:
            with tempfile.TemporaryDirectory(prefix="rc-cpu-iso-", dir=str(folder)) as iso_tmp_str:
                iso_tmp = Path(iso_tmp_str)
                for c_idx, case in enumerate(w.cases):
                    case_rows: list[Result] = []
                    captured_dirs: dict[int, Path] = {}
                    for i_idx, impl_name in enumerate(selected):
                        act_dir = iso_tmp / f"c{c_idx}-i{i_idx}-actual"
                        row_json = iso_tmp / f"c{c_idx}-i{i_idx}-row.json"
                        sub_out = iso_tmp / f"c{c_idx}-i{i_idx}-run"
                        sub_cfg = {
                            **cfg,
                            "_isolated_case": case.name,
                            "_isolated_impl": impl_name,
                            "_isolated_actual_dir": str(act_dir),
                            "_isolated_row_json": str(row_json),
                            "_isolated_work_dir": str(sub_out),
                        }
                        sub_cfg_file = iso_tmp / f"cfg-{c_idx}-{i_idx}.json"
                        sub_cfg_file.write_text(json.dumps(sub_cfg), encoding="utf-8")
                        sub_proc = subprocess.run(
                            [sys.executable, str(Path(__file__).resolve()), "--run-one", str(sub_cfg_file)],
                            env=clean_sub_env,
                        )
                        if sub_proc.returncode == 0 and row_json.is_file():
                            r_dict = json.loads(row_json.read_text(encoding="utf-8"))[0]
                            r_obj = Result(**r_dict)
                            r_obj.baseline = baseline_name
                            r_obj.baseline_ms = None
                            r_obj.speedup = None
                            if (act_dir / "tree.json").is_file():
                                captured_dirs[i_idx] = act_dir
                        else:
                            r_obj = Result(
                                w.name,
                                w.family,
                                case.name,
                                dict(case.parameters),
                                impl_name,
                                status="worker_failed",
                                correctness="unchecked",
                                stage="execute",
                                device=env.device,
                                baseline=baseline_name,
                                error=f"isolated CPU worker exited with code {sub_proc.returncode}",
                            )
                        case_rows.append(r_obj)

                    # Evaluate reference oracle once AFTER all timed subprocesses for this case have finished
                    if w.oracle is not None and captured_dirs:
                        expected = None
                        oracle_error = None
                        ref_call = None
                        try:
                            inp = w.inputs(case, 0)
                            if w.prepare_inputs is not None:
                                inp = w.prepare_inputs(case, inp, env)
                            if isinstance(w.oracle, Implementation):
                                ref_call = w.oracle.prepare(case, clone(inp), env)
                                if ref_call.compile:
                                    ref_call.compile()
                                with device_window(ref_call.device):
                                    _bench_reset(ref_call)
                                    ref_call.launch()
                                    _bench_sync(ref_call)
                                    expected = clone(ref_call.result())
                            else:
                                expected = w.oracle(case, clone(inp))
                        except Exception as exc:
                            oracle_error = exc
                        finally:
                            if ref_call is not None and ref_call.close:
                                try:
                                    ref_call.close()
                                except Exception as exc:
                                    oracle_error = exc

                        for i_idx, r_obj in enumerate(case_rows):
                            if i_idx not in captured_dirs or r_obj.status != "passed":
                                continue
                            if isinstance(oracle_error, DeviceBusy):
                                r_obj.status = "skipped_busy"
                                r_obj.correctness = "unchecked"
                                r_obj.stage = "correctness"
                                r_obj.error = f"{type(oracle_error).__name__}: {oracle_error}"
                            elif oracle_error is not None:
                                r_obj.status = "reference_failed"
                                r_obj.correctness = "reference_failed"
                                r_obj.metadata["reference_error"] = f"{type(oracle_error).__name__}: {oracle_error}"
                            else:
                                try:
                                    actual_val = read_tree(captured_dirs[i_idx], tensor_device=env.device)
                                    if w.check:
                                        w.check(actual_val, expected)
                                    else:
                                        assert_close(actual_val, expected, atol=w.atol, rtol=w.rtol)
                                    r_obj.correctness = "passed"
                                    r_obj.status = "passed"
                                except AssertionError as err:
                                    r_obj.correctness = "failed"
                                    r_obj.status = "numerical_failed"
                                    r_obj.error = str(err)

                    assign_speedups(case_rows, baseline_name)
                    all_rows.extend(case_rows)

            for row in all_rows:
                row.metadata["rc_target"] = target_name
                row.metadata["skipped_implementations"] = skipped
            save(all_rows, folder, command={"target": target_name, "workload": work, "skipped_implementations": skipped})
            summary = [(r.implementation, r.case, r.status, round(r.median_ms * 1000, 2) if r.median_ms else None) for r in all_rows]
            print(f"[{target_name}] {work}: {summary}", flush=True)
        return 0

    with run_output(env, w.name, out_folder, execution=True) as folder:
        oracle_dir = (
            folder / "oracle"
            if (needs_controller_oracle and not use_fixture and work != "corpus/weft/q4_k_projection" and work != "softmax" and work != "weighted_rms_norm" and work != "cpu/histogram")
            else None
        )
        rows = benchmark(
            w,
            selected,
            environment=env,
            baseline=baseline_name,
            measurement=meas,
            oracle_output=oracle_dir,
        )
        for row in rows:
            row.metadata["rc_target"] = target_name
            row.metadata["skipped_implementations"] = skipped
        save(rows, folder, command={"target": target_name, "workload": work, "skipped_implementations": skipped})
        summary = [(r.implementation, r.case, r.status, round(r.median_ms * 1000, 2) if r.median_ms else None) for r in rows]
        print(f"[{target_name}] {work}: {summary}", flush=True)
    return 0


def run_worker_batch(batch_json: Path) -> int:
    """Sequentially launch isolated per-workload subprocesses on the target host."""
    spec = json.loads(batch_json.read_text(encoding="utf-8"))
    workloads_list: list[str] = spec["workloads"]
    per_timeout: int = int(spec.get("timeout", 600))
    launcher: list[str] = list(spec.get("launcher", ()))
    exp_dir = Path(spec["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    clean_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "INTENTDSL_LIBRARY")}
    for idx, work in enumerate(workloads_list, 1):
        out_res = exp_dir / work / "results.json"
        if spec.get("resume") and out_res.is_file():
            existing_rows = json.loads(out_res.read_text(encoding="utf-8")).get("results", [])
            if not any(r.get("status") == "skipped_busy" for r in existing_rows):
                print(f"[{spec['target_name']}] ({idx}/{len(workloads_list)}) RESUME {work}", flush=True)
                continue
            shutil.rmtree(exp_dir / work, ignore_errors=True)
        one_cfg = {**spec, "workload": work}
        cfg_file = exp_dir / f".cfg-{idx}.json"
        cfg_file.write_text(json.dumps(one_cfg), encoding="utf-8")
        cmd = [*launcher, sys.executable, str(Path(__file__).resolve()), "--run-one", str(cfg_file)]
        started = time.perf_counter()
        try:
            proc = subprocess.run(cmd, env=clean_env, timeout=per_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            print(f"[{spec['target_name']}] ({idx}/{len(workloads_list)}) TIMEOUT {work} after {per_timeout}s", flush=True)
        finally:
            cfg_file.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        if rc != 0 and not out_res.is_file():
            out_res.parent.mkdir(parents=True, exist_ok=True)
            fail_payload = {
                "command": {"target": spec["target_name"], "workload": work},
                "results": [
                    {
                        "workload": work,
                        "family": work.split("/")[0],
                        "case": "author",
                        "parameters": {},
                        "implementation": f"intent-v2:{be}",
                        "status": "timeout" if rc == 124 else "worker_failed",
                        "correctness": "unchecked",
                        "stage": "execute",
                        "device": spec["device_kind"],
                        "scope": "operator",
                        "method": "auto",
                        "samples_ms": [],
                        "median_ms": None,
                        "iqr_ms": None,
                        "baseline": None,
                        "baseline_ms": None,
                        "speedup": None,
                        "stages_seconds": {},
                        "metrics": {},
                        "error": f"workload subprocess exited with code {rc} after {elapsed:.1f}s",
                        "traceback": None,
                        "metadata": {"rc_target": spec["target_name"]},
                    }
                    for be in spec["backends"]
                ],
            }
            out_res.write_text(json.dumps(fail_payload, indent=2) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Controller: deploy RC wheels, run target, validate RVV captures, aggregate
# ---------------------------------------------------------------------------

def _ssh(host: str, script: str, *, check: bool = True, timeout: int = 3600, capture: bool = False):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, script],
        check=check,
        timeout=timeout,
        text=True,
        capture_output=capture,
    )


def _send_tar(host: str, src_dir: Path, items: list[str], remote_dest: str) -> None:
    tar_cmd = ["tar", "-C", str(src_dir), "--exclude=__pycache__", "--exclude=*.pyc", "-czf", "-", *items]
    with subprocess.Popen(tar_cmd, stdout=subprocess.PIPE) as tar_proc:
        res = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {shlex.quote(remote_dest)} && tar -C {shlex.quote(remote_dest)} -xzf -"],
            stdin=tar_proc.stdout,
            check=True,
        )
        if tar_proc.wait() != 0 or res.returncode != 0:
            raise RuntimeError(f"tar transfer to {host}:{remote_dest} failed")


def check_device_idle(spec: TargetSpec) -> tuple[bool, str]:
    if spec.name == "5090d":
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        active = [
            ln.strip()
            for ln in r.stdout.splitlines()
            if ln.strip()
            and "VLLM::EngineCore" not in ln
            and "ToDesk" not in ln
            and not ln.strip().startswith("10904,")
        ]
        return (not active, f"5090d busy: {'; '.join(active)}" if active else "idle")
    if spec.name == "h100":
        r = _ssh("rc-host.example.invalid", "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader", capture=True, check=False, timeout=15)
        if r.returncode != 0:
            return False, f"h100 unreachable: {r.stderr.strip()}"
        busy = r.stdout.strip()
        return (not busy, f"h100 busy: {busy}" if busy else "idle")
    if spec.name == "mlu370":
        r = _ssh(
            spec.host,
            "export PATH=/usr/local/neuware/bin:$PATH; cnmon info -c 0 | grep -E 'MLU Average|busy +:' || true",
            capture=True,
            check=False,
            timeout=15,
        )
        if r.returncode != 0:
            return False, f"mlu370 unreachable: {r.stderr.strip()}"
        out = r.stdout.strip()
        if "busy                       : 0" not in out and "MLU Average                : 0 %" not in out:
            return False, f"mlu370 Card 0 busy: {out}"
        return True, "idle"
    if spec.host:
        r = _ssh(spec.host, "echo ok", capture=True, check=False, timeout=15)
        if r.returncode != 0:
            return False, f"{spec.name} unreachable: {r.stderr.strip()}"
    return True, "idle"


def execute_target(
    spec: TargetSpec,
    *,
    wheel_dir: Path,
    bench_root: Path,
    experiment_tag: str,
    workloads_filter: list[str] | None,
    resume: bool,
    allow_busy: bool,
    per_timeout: int,
) -> Path:
    local_exp_root = bench_root / "output" / "experiments" / experiment_tag / spec.name
    local_exp_root.mkdir(parents=True, exist_ok=True)
    (local_exp_root / "target_skipped.json").unlink(missing_ok=True)

    if not allow_busy:
        idle, msg = check_device_idle(spec)
        if not idle:
            if not resume:
                (local_exp_root / "target_skipped.json").write_text(
                    json.dumps({"target": spec.name, "status": "skipped_device_busy", "reason": msg}, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(f"[{spec.name}] SKIPPED ({msg})", flush=True)
            return local_exp_root

    product_meta = json.loads((Path(__file__).resolve().parents[1] / "product.json").read_text(encoding="utf-8"))
    ci_ver = product_meta["compilerinfra"]
    prod_ver = product_meta["version"]
    ci_whl = next(wheel_dir.glob(f"compilerinfra-{ci_ver}-*.whl"))
    cia_whl = next(wheel_dir.glob(f"compilerinfra_agent-{ci_ver}-*.whl"))
    prod_whls = sorted(wheel_dir.glob(spec.wheel_glob.replace("2.0.0", prod_ver)))
    if not prod_whls:
        raise FileNotFoundError(f"No wheel matching {spec.wheel_glob} in {wheel_dir}")
    prod_whl = prod_whls[0]

    env_json_local = Path.home() / ".config" / "intent-bench" / "environments" / f"{spec.profile}.json"
    workloads_to_run = workloads_filter or list(ALL_V2_WORKLOADS)

    if spec.host is None:
        # Local execution (5090d, x86-mojo, x86-weft)
        work_root = bench_root / "output" / "work" / f"v26-rc-{spec.name}"
        venv_dir = work_root / "venv"
        if not resume and venv_dir.exists():
            shutil.rmtree(venv_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        if not venv_dir.exists():
            subprocess.run([spec.base_python, "-m", "venv", str(venv_dir)], check=True)
            clean_install_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "INTENTDSL_LIBRARY")}
            subprocess.run(
                [str(venv_dir / "bin" / "pip"), "install", "--no-index", "--no-deps", "--force-reinstall", str(ci_whl), str(cia_whl), str(prod_whl)],
                env=clean_install_env,
                check=True,
            )

        # Stage 1: Standalone check BEFORE adding zz_bench_external.pth
        site_dir = next((venv_dir / "lib").glob("python*/site-packages"))
        pth_file = site_dir / "zz_bench_external.pth"
        pth_file.unlink(missing_ok=True)
        standalone_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "INTENTDSL_LIBRARY")}
        for k, v in spec.standalone_env:
            standalone_env[k] = v
        subprocess.run(
            [str(venv_dir / "bin" / "python"), str(Path(__file__).resolve()), "--standalone-check", *spec.backends, "--out", str(local_exp_root / "install_report.json")],
            cwd=work_root,
            env=standalone_env,
            check=True,
        )

        # Link base environment's site-packages + intent_bench src via low-priority .pth
        base_paths = json.loads(
            subprocess.check_output([spec.base_python, "-c", "import json, sys; print(json.dumps(sys.path))"], text=True)
        )
        filtered = [str((bench_root / "src").resolve())]
        for p in [*spec.extra_site_dirs, *base_paths]:
            if not p or "intentdsl" in Path(p).name or "compilerinfra" in Path(p).name:
                continue
            if Path(p).is_dir() and p not in filtered:
                filtered.append(p)
        pth_file.write_text("\n".join(filtered) + "\n", encoding="utf-8")

        batch_cfg = {
            "target_name": spec.name,
            "device_kind": spec.device_kind,
            "backends": list(spec.backends),
            "profile_path": str(env_json_local),
            "bench_output_root": os.path.abspath(bench_root / "output"),
            "exp_dir": os.path.abspath(local_exp_root),
            "captures_layout": spec.captures_layout,
            "needs_controller_oracle": False,
            "workloads": workloads_to_run,
            "launcher": list(spec.worker_launcher),
            "resume": resume,
            "timeout": per_timeout,
        }
        batch_file = work_root / "batch.json"
        batch_file.write_text(json.dumps(batch_cfg, indent=2), encoding="utf-8")
        worker_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "INTENTDSL_LIBRARY")}
        for k, v in spec.worker_env:
            worker_env[k] = v
        subprocess.run(
            [*spec.worker_launcher, str(venv_dir / "bin" / "python"), str(Path(__file__).resolve()), "--standalone-check", *spec.backends, "--require-doctor", "--out", str(local_exp_root / "install_report.json")],
            cwd=work_root,
            env=worker_env,
            check=True,
        )
        subprocess.run(
            [str(venv_dir / "bin" / "python"), str(Path(__file__).resolve()), "--worker-batch", str(batch_file)],
            cwd=work_root,
            env=worker_env,
            check=True,
        )
        return local_exp_root

    # Remote SSH execution (mlu370, h100, sg2044, rvv-v100, k3-*)
    assert spec.remote_output_root is not None
    r_out = spec.remote_output_root.rstrip("/")
    r_work = f"{r_out}/work/v26-rc-{spec.name}"
    r_exp = f"{r_out}/experiments/{experiment_tag}/{spec.name}"

    with tempfile.TemporaryDirectory(prefix=f"rc-stage-{spec.name}-") as tmp:
        stage = Path(tmp)
        shutil.copy2(ci_whl, stage / ci_whl.name)
        shutil.copy2(cia_whl, stage / cia_whl.name)
        shutil.copy2(prod_whl, stage / prod_whl.name)
        shutil.copy2(Path(__file__).resolve(), stage / "verify_release_candidate.py")
        shutil.copy2(env_json_local, stage / "env.json")
        _send_tar(spec.host, stage, [p.name for p in stage.iterdir()], f"{r_work}/bundle")
    _send_tar(spec.host, bench_root / "src", ["intent_bench"], f"{r_work}/site")

    batch_cfg = {
        "target_name": spec.name,
        "device_kind": spec.device_kind,
        "backends": list(spec.backends),
        "profile_path": f"{r_work}/bundle/env.json",
        "bench_output_root": r_out,
        "exp_dir": r_exp,
        "captures_layout": spec.captures_layout,
        "needs_controller_oracle": spec.needs_controller_oracle,
        "workloads": workloads_to_run,
        "launcher": list(spec.worker_launcher),
        "resume": resume,
        "timeout": per_timeout,
    }
    with tempfile.TemporaryDirectory(prefix="rc-batch-") as tmpb:
        bf = Path(tmpb) / "batch.json"
        bf.write_text(json.dumps(batch_cfg, indent=2), encoding="utf-8")
        _send_tar(spec.host, Path(tmpb), ["batch.json"], f"{r_work}/bundle")

    standalone_exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in spec.standalone_env)
    worker_exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in spec.worker_env)
    launcher_prefix = " ".join(shlex.quote(x) for x in spec.worker_launcher)
    extra_sites_json = json.dumps(list(spec.extra_site_dirs))
    recreate_flag = "0" if resume else "1"

    remote_script = f"""set -euo pipefail
unset PYTHONPATH PYTHONHOME INTENTDSL_LIBRARY
RWORK={shlex.quote(r_work)}
REXP={shlex.quote(r_exp)}
BASE_PY={shlex.quote(spec.base_python)}
mkdir -p "$RWORK" "$REXP"
if [ "{recreate_flag}" = "1" ]; then
  rm -rf "$RWORK/venv" "$REXP"
  mkdir -p "$REXP"
fi
if [ ! -x "$RWORK/venv/bin/python" ]; then
  if "$BASE_PY" -m venv "$RWORK/venv" 2>/dev/null; then
    "$RWORK/venv/bin/pip" install --no-index --no-deps --force-reinstall \\
      "$RWORK/bundle/{ci_whl.name}" \\
      "$RWORK/bundle/{cia_whl.name}" \\
      "$RWORK/bundle/{prod_whl.name}"
  else
    rm -rf "$RWORK/venv"
    "$BASE_PY" -m venv --without-pip "$RWORK/venv"
    "$RWORK/venv/bin/python" - "$RWORK/venv" \\
      "$RWORK/bundle/{ci_whl.name}" \\
      "$RWORK/bundle/{cia_whl.name}" \\
      "$RWORK/bundle/{prod_whl.name}" <<'PYWHL'
import configparser, os, stat, sys, sysconfig, zipfile
from pathlib import Path
venv_root = Path(sys.argv[1]).resolve()
purelib = Path(sysconfig.get_paths()["purelib"])
purelib.mkdir(parents=True, exist_ok=True)
bin_dir = venv_root / "bin"
for whl in map(Path, sys.argv[2:]):
    with zipfile.ZipFile(whl) as zf:
        zf.extractall(purelib)
        for name in zf.namelist():
            if name.endswith(".dist-info/entry_points.txt"):
                cp = configparser.ConfigParser()
                cp.read_string(zf.read(name).decode("utf-8"))
                if cp.has_section("console_scripts"):
                    for cmd_name, spec_val in cp.items("console_scripts"):
                        mod_name, _, fn_name = spec_val.partition(":")
                        script = bin_dir / cmd_name
                        script.write_text(
                            f"#!{{venv_root / 'bin' / 'python'}}\\n"
                            f"import sys\\nfrom {{mod_name}} import {{fn_name}}\\n"
                            f"raise SystemExit({{fn_name}}())\\n",
                            encoding="utf-8",
                        )
                        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PYWHL
  fi
fi
SITE_DIR=$("$RWORK/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
rm -f "$SITE_DIR/zz_bench_external.pth"
(
  cd "$RWORK"
  env -i HOME="$HOME" USER="${{USER:-user}}" TMPDIR="$RWORK" PATH="/usr/local/bin:/usr/bin:/bin" {standalone_exports} \\
    "$RWORK/venv/bin/python" "$RWORK/bundle/verify_release_candidate.py" \\
    --standalone-check {' '.join(spec.backends)} --out "$REXP/install_report.json"
)
"$BASE_PY" - "$RWORK/site" "$SITE_DIR/zz_bench_external.pth" {shlex.quote(extra_sites_json)} <<'PY'
import json, sys
from pathlib import Path
bench_site, pth_out, extras = sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3])
paths = [bench_site]
for p in [*extras, *sys.path]:
    if not p:
        continue
    name = Path(p).name
    if "intentdsl" in name or "compilerinfra" in name:
        continue
    if Path(p).is_dir() and p not in paths:
        paths.append(p)
pth_out.write_text("\\n".join(paths) + "\\n", encoding="utf-8")
PY
cd "$RWORK"
export TMPDIR="$RWORK"
{f'export {worker_exports}' if worker_exports else ':'}
{launcher_prefix} "$RWORK/venv/bin/python" "$RWORK/bundle/verify_release_candidate.py" \\
  --standalone-check {' '.join(spec.backends)} --require-doctor --out "$REXP/install_report.json"
{launcher_prefix} "$RWORK/venv/bin/python" "$RWORK/bundle/verify_release_candidate.py" --worker-batch "$RWORK/bundle/batch.json"
"""
    log_path = local_exp_root / "remote_worker.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", spec.host, remote_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Remote execution on {spec.name} exited with {rc}; see {log_path}")

    # Transfer remote experiment directory back to controller
    with subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", spec.host, f"cd {shlex.quote(r_out + '/experiments/' + experiment_tag)} && tar -czf - {shlex.quote(spec.name)}"],
        stdout=subprocess.PIPE,
    ) as remote_tar:
        dest_parent = bench_root / "output" / "experiments" / experiment_tag
        subprocess.run(["tar", "-C", str(dest_parent), "-xzf", "-"], stdin=remote_tar.stdout, check=True)
        if remote_tar.wait() != 0:
            raise RuntimeError(f"Failed transferring results from {spec.name}")

    if spec.needs_controller_oracle:
        _validate_rvv_captures(local_exp_root, bench_root)

    return local_exp_root


def _validate_rvv_captures(target_exp_dir: Path, bench_root: Path) -> None:
    """Validate captured RVV actual outputs on the controller's PyTorch CPU oracle."""
    validator_code = """
import json, sys
from pathlib import Path
from intent_bench.environments import load_environment
from intent_bench.model import Measurement, Result
from intent_bench.oracle import validate_captured
from intent_bench.request import RunRequest
from intent_bench.results import save

target_dir = Path(sys.argv[1])
oracle_env = load_environment("local-cpu")
count = 0
for res_file in sorted(target_dir.glob("**/results.json")):
    payload = json.loads(res_file.read_text(encoding="utf-8"))
    rows = [Result(**r) for r in payload.get("results", [])]
    pending = [r for r in rows if r.status == "unchecked" and r.metadata.get("oracle_capture")]
    if not pending:
        continue
    for r in pending:
        req = RunRequest(r.workload, (r.implementation,), oracle_env, Measurement(enabled=False), baseline=r.implementation)
        verdict, = validate_captured(req, res_file.parent, [{"case": r.case, "implementation": r.implementation, "capture": r.metadata["oracle_capture"]}])
        r.status = verdict["status"]
        r.correctness = verdict["correctness"]
        r.error = verdict.get("error")
        r.stage = "complete" if r.status == "passed" else "correctness"
        count += 1
    save(rows, res_file.parent, command=payload.get("command"))
print(f"Validated {count} RVV oracle captures under {target_dir}", flush=True)
"""
    env = {**os.environ, "PYTHONPATH": str((bench_root / "src").resolve())}
    subprocess.run(
        ["/srv/rc/site-1/.venvs/intentdsl-cutile/bin/python", "-c", validator_code, str(target_exp_dir)],
        env=env,
        check=True,
    )


# ---------------------------------------------------------------------------
# Stage Accounting & Scoreboard Generator
# ---------------------------------------------------------------------------

def _classify_row_stages(row: dict[str, Any]) -> dict[str, bool]:
    status = row.get("status", "")
    stage = row.get("stage", "")
    correctness = row.get("correctness", "")
    median_ms = row.get("median_ms")

    if status.startswith("skipped"):
        return {"generate": False, "native_compile": False, "execute": False, "correctness": False, "performance": False}

    gen_fail_statuses = {"input_failed", "preparation_failed", "expression_gap", "environment_missing", "environment_unavailable"}
    gen_ok = status not in gen_fail_statuses and stage not in {"load", "inputs", "prepare", "frontend", "kir", "lowering"}
    compile_ok = gen_ok and status != "compilation_failed" and stage != "native_compile"
    exec_ok = compile_ok and status not in {"execution_failed", "worker_failed", "timeout"} and stage != "first_call"
    corr_ok = exec_ok and status == "passed" and correctness == "passed"
    perf_ok = corr_ok and isinstance(median_ms, (int, float)) and math.isfinite(median_ms) and median_ms > 0
    return {
        "generate": gen_ok,
        "native_compile": compile_ok,
        "execute": exec_ok,
        "correctness": corr_ok,
        "performance": perf_ok,
    }


def summarize_experiment(bench_root: Path, experiment_tag: str) -> tuple[dict[str, Any], str]:
    exp_root = bench_root / "output" / "experiments" / experiment_tag
    summary: dict[str, Any] = {
        "experiment_tag": experiment_tag,
        "targets": {},
        "backend_scoreboards": [],
        "slower_than_v1_over_1_2x": [],
        "failures_and_skips": [],
    }

    seen_ref_failures: set[tuple[str, str, str, str]] = set()
    md: list[str] = [
        f"# V2-6.R Release Candidate Full Verification & Scoreboard (`{experiment_tag}`)",
        "",
        "## 1. 六阶段核查汇总（按设备 × 出口）",
        "",
        "| 设备目标 | 出口 | 安装 (setup/doctor) | 30 Program 通过 | 全量 V2 用例数 | 生成 | 原生编译 | 执行 | 数值通过 | 热性能中位数 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for target_name, spec in TARGETS.items():
        tdir = exp_root / target_name
        if not tdir.exists():
            continue
        skip_marker = tdir / "target_skipped.json"
        if skip_marker.is_file():
            info = json.loads(skip_marker.read_text(encoding="utf-8"))
            summary["failures_and_skips"].append({"target": target_name, "scope": "target", "status": info["status"], "reason": info["reason"]})
            for be in spec.backends:
                md.append(f"| `{target_name}` | `{be}` | SKIPPED | — | — | — | — | — | — | — (`{info['reason']}`) |")
            continue

        install_ok = False
        install_file = tdir / "install_report.json"
        install_info = json.loads(install_file.read_text(encoding="utf-8")) if install_file.is_file() else {}
        install_ok = install_info.get("status") == "passed"

        # Load all workload results under tdir
        by_work_case: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        skipped_impl_records: list[dict[str, Any]] = []
        for rf in sorted(tdir.glob("**/results.json")):
            if "artifacts" in rf.parts:
                continue
            payload = json.loads(rf.read_text(encoding="utf-8"))
            cmd = payload.get("command") or {}
            wname = cmd.get("workload")
            for sk in cmd.get("skipped_implementations") or []:
                skipped_impl_records.append({"target": target_name, "workload": wname, **sk})
            for r in payload.get("results", []):
                key = (r["workload"], r["case"])
                by_work_case.setdefault(key, {})[r["implementation"]] = r

        summary["targets"][target_name] = {
            "install": install_info,
            "skipped_implementations_count": len(skipped_impl_records),
        }

        for be in spec.backends:
            v2_key = f"intent-v2:{be}"
            v1_key = f"intent:{be}"
            total_cases = 0
            prog_pass = 0
            prog_total = 0
            counts = {"generate": 0, "native_compile": 0, "execute": 0, "correctness": 0, "performance": 0}

            v1_cmp = {"faster": 0, "tied": 0, "slower": 0, "no_comparable": 0, "ref_failed": 0, "quantized_short_ties": 0}
            mat_cmp = {"faster": 0, "tied": 0, "slower": 0, "no_comparable": 0, "ref_failed": 0, "semantic_diff": 0, "quantized_short_ties": 0}

            for (wname, cname), impl_map in sorted(by_work_case.items()):
                if v2_key not in impl_map:
                    continue
                row = impl_map[v2_key]
                total_cases += 1
                is_prog = wname.startswith("program/")
                if is_prog:
                    prog_total += 1
                st = _classify_row_stages(row)
                for k_st, ok_st in st.items():
                    if ok_st:
                        counts[k_st] += 1
                if is_prog and st["correctness"]:
                    prog_pass += 1

                if not st["correctness"] or not st["performance"]:
                    summary["failures_and_skips"].append(
                        {
                            "target": target_name,
                            "backend": be,
                            "workload": wname,
                            "case": cname,
                            "implementation": v2_key,
                            "status": row.get("status"),
                            "stage": row.get("stage"),
                            "reason": row.get("error"),
                        }
                    )

                # Record non-V2 failures/skips on this case too (deduplicated per target)
                for other_impl, other_row in impl_map.items():
                    if other_impl.startswith("intent-v2:"):
                        continue
                    if other_row.get("status") != "passed":
                        dedup_key = (target_name, wname, cname, other_impl)
                        if dedup_key not in seen_ref_failures:
                            seen_ref_failures.add(dedup_key)
                            summary["failures_and_skips"].append(
                                {
                                    "target": target_name,
                                    "backend": be,
                                    "workload": wname,
                                    "case": cname,
                                    "implementation": other_impl,
                                    "status": other_row.get("status"),
                                    "stage": other_row.get("stage"),
                                    "reason": other_row.get("error"),
                                }
                            )

                if not st["performance"]:
                    continue
                v2_ms = float(row["median_ms"])

                # 1. V2 vs V1
                v1_row = impl_map.get(v1_key)
                if v1_row is None:
                    v1_cmp["no_comparable"] += 1
                elif v1_row.get("status") != "passed" or not v1_row.get("median_ms"):
                    # Note: paged_decode V1 mixed log2/ln bases so numerical check fails, or V1 failed compile
                    v1_cmp["ref_failed"] += 1
                else:
                    v1_ms = float(v1_row["median_ms"])
                    v2_us = v2_ms * 1000.0
                    v1_us = v1_ms * 1000.0
                    ratio = v2_ms / v1_ms
                    # On 5090D (SM120), CUDAGraph CUDA Event hardware timer quantum is 2.048 us;
                    # for short kernels (< 8 us), +-1 tick (<= 2.10 us) is single-quantum jitter.
                    sm120_single_tick = (
                        target_name == "5090d"
                        and min(v2_us, v1_us) < 8.0
                        and abs(v2_us - v1_us) <= 2.10
                    )
                    if sm120_single_tick:
                        v1_cmp["tied"] += 1
                        if ratio < 0.95 or ratio > 1.05:
                            v1_cmp["quantized_short_ties"] += 1
                    elif ratio < 0.95:
                        v1_cmp["faster"] += 1
                    elif ratio <= 1.05:
                        v1_cmp["tied"] += 1
                    else:
                        v1_cmp["slower"] += 1
                    if ratio > 1.20 and not sm120_single_tick:
                        summary["slower_than_v1_over_1_2x"].append(
                            {
                                "target": target_name,
                                "backend": be,
                                "workload": wname,
                                "case": cname,
                                "v2_us": round(v2_us, 2),
                                "v1_us": round(v1_us, 2),
                                "ratio": round(ratio, 3),
                                "unit": responsible_unit(be, wname),
                            }
                        )

                # 2. V2 vs Mature (best passing same-semantics mature baseline on same device)
                mature_candidates = [
                    (ik, ir)
                    for ik, ir in impl_map.items()
                    if not ik.startswith(("intent:", "intent-v2:"))
                ]
                if not mature_candidates:
                    mat_cmp["no_comparable"] += 1
                else:
                    passing_mat = [
                        (ik, float(ir["median_ms"]))
                        for ik, ir in mature_candidates
                        if ir.get("status") == "passed" and ir.get("median_ms")
                    ]
                    if not passing_mat:
                        mat_cmp["ref_failed"] += 1
                    elif be == "bangc" and wname in ("program/softmax", "softmax") and ("float16" in cname or wname == "program/softmax"):
                        # Explicit semantic contract difference documented in P-table:
                        # V2 computes f32 exp + 0-ULP RN f16 quotient; CNNL/Torch uses uncertified f16 approx.
                        best_name, best_ms = min(passing_mat, key=lambda x: x[1])
                        ratio_m = v2_ms / best_ms
                        mat_cmp["semantic_diff"] += 1
                        if ratio_m < 0.95:
                            mat_cmp["faster"] += 1
                        elif ratio_m <= 1.05:
                            mat_cmp["tied"] += 1
                        else:
                            mat_cmp["slower"] += 1
                    else:
                        best_name, best_ms = min(passing_mat, key=lambda x: x[1])
                        v2_us = v2_ms * 1000.0
                        best_us = best_ms * 1000.0
                        ratio_m = v2_ms / best_ms
                        sm120_mat_tick = (
                            target_name == "5090d"
                            and min(v2_us, best_us) < 8.0
                            and abs(v2_us - best_us) <= 2.10
                        )
                        if sm120_mat_tick:
                            mat_cmp["tied"] += 1
                            if ratio_m < 0.95 or ratio_m > 1.05:
                                mat_cmp["quantized_short_ties"] += 1
                        elif ratio_m < 0.95:
                            mat_cmp["faster"] += 1
                        elif ratio_m <= 1.05:
                            mat_cmp["tied"] += 1
                        else:
                            mat_cmp["slower"] += 1

            md.append(
                f"| `{target_name}` | `{be}` | {'PASS' if install_ok else 'FAIL'} | "
                f"{prog_pass}/{prog_total} | {total_cases} | {counts['generate']}/{total_cases} | "
                f"{counts['native_compile']}/{total_cases} | {counts['execute']}/{total_cases} | "
                f"{counts['correctness']}/{total_cases} | {counts['performance']}/{total_cases} |"
            )

            board_entry = {
                "target": target_name,
                "backend": be,
                "install_passed": install_ok,
                "programs_passed": f"{prog_pass}/{prog_total}",
                "total_v2_cases": total_cases,
                "stages": counts,
                "vs_v1": v1_cmp,
                "vs_mature": mat_cmp,
            }
            summary["backend_scoreboards"].append(board_entry)

    md.extend(
        [
            "",
            "## 2. 性能计分板（完整调用热执行中位数，持平阈值 `0.95x–1.05x`）",
            "",
            "> 注：`5090d`（SM120）上 `CUDAGraph` 内 CUDA Event 硬件计时步长为 `2.048 µs`；对 `< 8 µs` 的短 kernel，前后差 `≤ 1` 个硬件刻度（`≤ 2.048 µs`）属于量化步长内正常波动，按持平计入且不列入 `> 1.20x` 表。",
            "",
            "| 设备目标 | 出口 | V2 对 V1（更快 / 持平 / 更慢 / V1失败或无对照） | V2 对成熟实现（更快 / 持平 / 更慢 / 无同合同或失败） |",
            "|---|---|---|---|",
        ]
    )
    for b in summary["backend_scoreboards"]:
        v1 = b["vs_v1"]
        mt = b["vs_mature"]
        v1_qnote = f"（含 {v1['quantized_short_ties']} 项 `<8µs` 单刻度量化持平）" if v1.get("quantized_short_ties") else ""
        sem_note = f"（含 {mt['semantic_diff']} 项 f16 softmax 精确商语义差）" if mt["semantic_diff"] else ""
        mt_qnote = f"（含 {mt['quantized_short_ties']} 项 `<8µs` 单刻度量化持平）" if mt.get("quantized_short_ties") else ""
        md.append(
            f"| `{b['target']}` | `{b['backend']}` | "
            f"**{v1['faster']}** 快 / **{v1['tied']}** 平 / **{v1['slower']}** 慢（{v1['ref_failed']} 失败, {v1['no_comparable']} 无）{v1_qnote} | "
            f"**{mt['faster']}** 快 / **{mt['tied']}** 平 / **{mt['slower']}** 慢（{mt['ref_failed']} 失败, {mt['no_comparable']} 无）{sem_note}{mt_qnote} |"
        )

    md.extend(
        [
            "",
            "## 3. 比 V1 慢 1.2 倍以上的全部项目（`V2 / V1 > 1.20x`）及负责单元",
            "",
            "| 设备目标 | 出口 | 工作负载 (`case`) | V2 (µs) | V1 (µs) | V2/V1 | 负责单元 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for item in sorted(summary["slower_than_v1_over_1_2x"], key=lambda x: (x["target"], x["backend"], -x["ratio"])):
        md.append(
            f"| `{item['target']}` | `{item['backend']}` | `{item['workload']}` (`{item['case']}`) | "
            f"{item['v2_us']:.2f} | {item['v1_us']:.2f} | {item['ratio']:.2f}x | `{item['unit']}` |"
        )
    if not summary["slower_than_v1_over_1_2x"]:
        md.append("| — | — | 无（全部项目 `V2 / V1 <= 1.20x`） | — | — | — | — |")

    # Failures & Skips section
    v2_issues = [x for x in summary["failures_and_skips"] if str(x.get("implementation", "")).startswith("intent-v2:") or x.get("scope") == "target"]
    ref_issues = [x for x in summary["failures_and_skips"] if x not in v2_issues]
    md.extend(
        [
            "",
            "## 4. 失败与跳过原因记录",
            "",
            f"- **V2 候选失败或跳过数**：{len(v2_issues)}",
            f"- **对照实现（V1 / 第三方库）失败或不支持数**：{len(ref_issues)}",
            "",
        ]
    )
    if v2_issues:
        md.append("### V2 失败 / 跳过明细")
        for it in v2_issues:
            md.append(f"- `{it.get('target')}` `{it.get('implementation', '')}` `{it.get('workload', '')}` (`{it.get('case', '')}`): `{it.get('status')}` — `{str(it.get('reason'))[:240]}`")
        md.append("")
    if ref_issues:
        md.append("### 对照实现（V1 / 成熟库）失败或跳过明细（保留原始错误）")
        for it in ref_issues[:80]:
            md.append(f"- `{it.get('target')}` `{it.get('implementation')}` `{it.get('workload')}` (`{it.get('case')}`): `{it.get('status')}` — `{str(it.get('reason'))[:200]}`")
        if len(ref_issues) > 80:
            md.append(f"- … 其余 {len(ref_issues) - 80} 条详见 `scoreboard.json`")

    md_text = "\n".join(md) + "\n"
    (exp_root / "scoreboard.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (exp_root / "scoreboard.md").write_text(md_text, encoding="utf-8")
    return summary, md_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone-check", nargs="+", help="Internal: run clean-venv RC check for given backends")
    parser.add_argument("--require-doctor", action="store_true", help="Internal: require doctor() ready for given backends")
    parser.add_argument("--out", type=Path, help="Internal: output JSON path for --standalone-check")
    parser.add_argument("--run-one", type=Path, help="Internal: run single workload JSON config")
    parser.add_argument("--worker-batch", type=Path, help="Internal: run batch JSON config on target host")

    parser.add_argument("--targets", nargs="+", help=f"Targets to execute ({', '.join(TARGETS)} or 'all')")
    parser.add_argument("--wheel-dir", type=Path, default=Path("/srv/rc/site-1/phdworks/wt/dsa/dist/release"))
    parser.add_argument("--bench-root", type=Path, default=Path("/srv/rc/site-1/phdworks/intentdsl_bench"))
    parser.add_argument("--tag", default="v2-6-rc", help="Experiment tag under bench output/experiments/")
    parser.add_argument("--workloads", nargs="+", help="Optional subset of workloads to run")
    parser.add_argument("--resume", action="store_true", help="Reuse completed workload results.json files")
    parser.add_argument("--allow-busy", action="store_true", help="Do not skip when device reports active processes")
    parser.add_argument("--timeout", type=int, default=600, help="Per-workload subprocess timeout in seconds")
    parser.add_argument("--summarize-only", action="store_true", help="Only aggregate existing results into scoreboard")
    args = parser.parse_args()

    if args.standalone_check:
        assert args.out is not None
        return run_standalone_check(args.standalone_check, args.out, require_doctor_ready=args.require_doctor)
    if args.run_one:
        return run_single_workload(json.loads(args.run_one.read_text(encoding="utf-8")))
    if args.worker_batch:
        return run_worker_batch(args.worker_batch)

    bench_root = Path(os.path.abspath(args.bench_root))
    wheel_dir = Path(os.path.abspath(args.wheel_dir))

    if not args.summarize_only:
        raw_targets = args.targets or ["mlu370"]
        if "all" in raw_targets:
            # Default full suite: GPU (5090d, h100), CPU x86 (mojo, weft), RVV (sg2044, rvv-v100, k3-x100, k3-a100), MLU (mlu370)
            target_names = list(TARGETS.keys())
        else:
            target_names = raw_targets

        # Partition remote/GPU targets (can run concurrently on independent machines) vs local x86 CPU targets (serial on cpus 0..7)
        serial_local_cpu = [t for t in target_names if t in ("x86-mojo", "x86-weft")]
        serial_k3 = [t for t in target_names if t in ("k3-x100", "k3-a100")]
        parallel_others = [t for t in target_names if t not in serial_local_cpu and t not in serial_k3]

        def _run_group(names: list[str]) -> None:
            for n in names:
                execute_target(
                    TARGETS[n],
                    wheel_dir=wheel_dir,
                    bench_root=bench_root,
                    experiment_tag=args.tag,
                    workloads_filter=args.workloads,
                    resume=args.resume,
                    allow_busy=args.allow_busy,
                    per_timeout=args.timeout,
                )

        futures = []
        with ThreadPoolExecutor(max_workers=max(1, len(parallel_others) + (1 if serial_k3 else 0))) as pool:
            for t in parallel_others:
                futures.append(pool.submit(_run_group, [t]))
            if serial_k3:
                futures.append(pool.submit(_run_group, serial_k3))
            for fut in as_completed(futures):
                fut.result()

        # Run local x86 CPU targets strictly after RVV controller oracle validation & sequentially with each other
        if serial_local_cpu:
            _run_group(serial_local_cpu)

    _, md_text = summarize_experiment(bench_root, args.tag)
    print("\n" + md_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
