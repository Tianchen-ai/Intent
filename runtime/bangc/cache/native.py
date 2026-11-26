"""Compile and retain immutable, per-row CNCC products."""

import ctypes as C
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter


def _cnnl_bridge_source():
    local = Path(__file__).resolve().parents[1] / "kernels" / "cnnl_bridge.cpp"
    if local.is_file():
        return local
    raise FileNotFoundError(f"missing BANG C CNNL bridge source {local}")


def _compile_cnnl_bridge_object(base: Path, sdk: Path) -> Path:
    src = _cnnl_bridge_source()
    digest = hashlib.sha256(
        src.read_bytes() + b"\0" + str(sdk).encode("utf-8")
    ).hexdigest()[:16]
    obj = base / f"cnnl_bridge-{digest}.o"
    if obj.is_file():
        return obj
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or "c++"
    with NamedTemporaryFile(
        prefix="cnnl_bridge-", suffix=".o", dir=base, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [
                cxx,
                "-std=c++14",
                "-O2",
                "-fPIC",
                "-c",
                f"-I{sdk}/include",
                str(src),
                "-o",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"CNNL host bridge compile failed: {proc.stderr or proc.stdout}"
            )
        os.replace(tmp_path, obj)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return obj


class Native:
    def __init__(self, source, row, abi, *, cnnl=False, fp_contract="off"):
        base = (
            Path(
                os.environ.get("INTENT_CACHE_DIR", Path.home() / ".cache" / "intentdsl")
            )
            / "bangc"
        )
        base.mkdir(parents=True, exist_ok=True)
        self.directory = TemporaryDirectory(prefix="row-", dir=base)
        work = Path(self.directory.name)
        src = work / "kernel.mlu"
        src.write_text(source)
        sdk = Path(os.environ.get("NEUWARE_HOME", "/usr/local/neuware"))
        lib = work / "kernel.so"
        extra_objs = [str(_compile_cnnl_bridge_object(base, sdk))] if cnnl else []
        extra_libs = ["-lcnnl", f"-Wl,-rpath,{sdk}/lib64"] if cnnl else []
        cmd = [
            str(sdk / "bin" / "cncc"),
            f"--neuware-path={sdk}",
            "--bang-mlu-arch=mtp_372",
            "-std=c++14",
            "-O2",
            "-ffp-contract=" + fp_contract,
            "-shared",
            "-fPIC",
            *(
                f"-DINTENT_TILE_{k}={row[v]}"
                for k, v in (
                    ("R", "rows"),
                    ("W", "width"),
                    ("P", "pitch"),
                    ("BYTES", "bytes"),
                )
            ),
            str(src),
            *extra_objs,
            "-o",
            str(lib),
            f"-L{sdk}/lib64",
            "-lcnrt",
            *extra_libs,
        ]
        start = perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"CNCC row {row['id']}: {proc.stderr or proc.stdout}")
        self.seconds = perf_counter() - start
        self.module = C.CDLL(str(lib))
        self._cleanup = getattr(self.module, "intent_cnnl_cleanup", None)
        if self._cleanup is not None:
            self._cleanup.argtypes = []
            self._cleanup.restype = None
        self.launch = self.module.intent_enqueue
        types = []
        for arg in abi:
            if arg["dims"]:
                types += [C.c_void_p, *([C.c_int64] * (2 * len(arg["dims"])))]
            else:
                types.append({"i32":C.c_int32,"f32":C.c_float,"u64":C.c_uint64}[arg["dtype"]])
        self.launch.argtypes = [
            *types,
            C.c_int64,
            C.c_int64,
            C.c_int64,
            C.c_int,
            C.c_void_p,
        ]
        self.launch.restype = C.c_int

    def close(self):
        if getattr(self, "_cleanup", None) is not None:
            self._cleanup()
            self._cleanup = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
