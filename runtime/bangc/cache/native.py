"""Compile and retain immutable, per-row CNCC products."""

import ctypes as C
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter


class Native:
    def __init__(self, source, row, abi):
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
        cmd = [
            str(sdk / "bin" / "cncc"),
            f"--neuware-path={sdk}",
            "--bang-mlu-arch=mtp_372",
            "-std=c++14",
            "-O2",
            "-ffp-contract=off",
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
            "-o",
            str(lib),
            f"-L{sdk}/lib64",
            "-lcnrt",
        ]
        start = perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"CNCC row {row['id']}: {proc.stderr or proc.stdout}")
        self.seconds = perf_counter() - start
        self.module = C.CDLL(str(lib))
        self.launch = self.module.intent_enqueue
        types = []
        for arg in abi:
            if arg["dims"]:
                types += [C.c_void_p, *([C.c_int64] * (2 * len(arg["dims"])))]
            else:
                types.append(C.c_int32 if arg["dtype"] == "i32" else C.c_float)
        self.launch.argtypes = [
            *types,
            C.c_int64,
            C.c_int64,
            C.c_int64,
            C.c_int,
            C.c_void_p,
        ]
        self.launch.restype = C.c_int
