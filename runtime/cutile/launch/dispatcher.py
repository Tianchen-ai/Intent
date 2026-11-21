"""cuTile 1.6 dispatcher adapter. Only prepare_native may invoke the compiler."""

from threading import RLock
from time import perf_counter
import cuda.tile as ct
from cuda.tile._cext import default_tile_context
from cuda.tile.compilation import CallingConvention, KernelSignature, mangle_kernel_name


class BoundKernel(ct.kernel):
    def __init__(self, fn, hints, arch):
        super().__init__(fn, **hints)
        self.arch = arch
        self.native = {}
        self.lock = RLock()
        self.compiles = 0
        self.reuses = 0

    def replace_hints(self, **options):
        # SDK exhaustive_search asks for a variant even when no hints change.
        if any(getattr(self._compiler_options, k) != v for k, v in options.items()):
            raise ValueError("tuner requested an undeclared hint change")
        return self

    def prepare_native(self, args):
        sig = KernelSignature.from_kernel_args(
            self, args, CallingConvention.cutile_python_v1()
        )
        key = (default_tile_context, mangle_kernel_name("kernel", sig))
        with self.lock:
            hit = key in self.native
            start = perf_counter()
            if not hit:
                self.compiles += 1
                self.native[key] = super()._compile(
                    sig, default_tile_context, self.arch
                )
            return {
                "compiler_ms": 1000 * (perf_counter() - start),
                "dispatcher_hit": hit,
            }

    def _compile(self, signature, context, compute_capability=None):
        if compute_capability is not None and tuple(compute_capability) != self.arch:
            raise RuntimeError("cuTile dispatcher architecture changed")
        with self.lock:
            key = (context, mangle_kernel_name("kernel", signature))
            if key not in self.native:
                raise RuntimeError("cuTile requested an unprepared native signature")
            self.reuses += 1
            return self.native[key]
