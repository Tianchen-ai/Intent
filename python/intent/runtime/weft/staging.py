"""Host storage adapter; ordinary JIT still accepts existing native buffers."""
from .buffer import Buffer


class HostStorage:
    copies_host = False

    def accepts(self, value):
        return isinstance(value, Buffer)

    def allocate(self, shape, element):
        return Buffer.empty(tuple(shape), element)

    def upload(self, host, element):
        return memoryview(host)

    def download(self, allocation):
        return allocation

    def view(self, allocation, shape, strides, offset, element):
        return Buffer(allocation, shape=shape, dtype=element, strides=strides, offset=offset)

    def snapshot(self, value):
        owner = memoryview(value.storage_span.obj if isinstance(value, Buffer) else value.obj).cast("B")
        saved = bytes(owner)
        def restore():
            owner[:] = saved
        return id(owner.obj), restore

    def release(self, value):
        pass  # The borrowed host allocation is held by the buffer's memoryview.

    def prepare_fill(self, value, scalar):
        import numpy as np
        from ..native_call import NativeMemoryFill
        from ..sequence import PreparedInitialization
        from ..storage_types import numpy_dtype
        if not self.accepts(value):
            raise TypeError("Weft initialization requires its public CPU Buffer")
        if not value.canonical:
            raise NotImplementedError("native Weft initialization requires a contiguous destination")
        encoded = np.asarray(scalar, dtype=numpy_dtype(value.dtype)).reshape(()).tobytes()
        native = NativeMemoryFill(value.pointer, value.nbytes // len(encoded), encoded,
                                  (value,), lambda: None)
        return PreparedInitialization(native.launch, (value,), native)
