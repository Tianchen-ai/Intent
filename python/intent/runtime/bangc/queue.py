"""An explicitly owned local CNRT queue for authored multi-kernel calls."""
import ctypes
from .buffer import runtime


class Queue:
    def __init__(self, *, device=0, neuware="/usr/local/neuware"):
        self.device = device
        self.runtime = runtime(neuware)
        self.runtime.select(device)
        self.handle = ctypes.c_void_p()
        self.runtime.invoke("cnrtQueueCreate", ctypes.byref(self.handle))

    def synchronize(self):
        if not self.handle.value:
            raise RuntimeError("CNRT queue is closed")
        self.runtime.select(self.device)
        self.runtime.invoke("cnrtQueueSync", self.handle)

    def close(self):
        if self.handle.value:
            self.synchronize()
            self.runtime.invoke("cnrtQueueDestroy", self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
