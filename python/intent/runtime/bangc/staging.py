"""MLU host transfer and allocation owned by the BANG C component."""
from .buffer import DeviceBuffer, DeviceView, ELEMENT_BYTES


class HostStorage:
    copies_host = True
    contiguous_views = False

    def __init__(self, device, neuware):
        self.device, self.neuware = device, neuware

    def accepts(self, value):
        return isinstance(value, (DeviceBuffer, DeviceView))

    def allocate(self, shape, element):
        return DeviceBuffer(tuple(shape), element, device=self.device, neuware=self.neuware)

    def upload(self, host, element):
        width = ELEMENT_BYTES[element]
        return DeviceBuffer.from_host(host, shape=(host.nbytes // width,), dtype=element,
                                      device=self.device, neuware=self.neuware)

    def view(self, allocation, shape, strides, offset, element):
        return allocation.view(tuple(shape), strides, offset=offset)

    def download(self, allocation):
        import numpy as np
        return np.frombuffer(allocation.to_host(), dtype=np.uint8)

    def snapshot(self, value):
        owner = value.owner if isinstance(value, DeviceView) else value
        saved = owner.to_host()
        return id(owner), lambda: owner.copy_from_host(saved)

    def release(self, value):
        if isinstance(value, DeviceBuffer):
            value.close()

    def prepare_fill(self, value, scalar):
        import ctypes
        from math import prod
        import numpy as np
        from ..sequence import PreparedInitialization
        from ..storage_types import numpy_dtype
        if not self.accepts(value) or value.device != self.device:
            raise ValueError("initialization must use the selected MLU device")
        if value.strides != tuple(prod(value.shape[i + 1:]) for i in range(len(value.shape))):
            raise NotImplementedError("MLU initialization requires contiguous storage")
        encoded = np.asarray(scalar, dtype=numpy_dtype(value.dtype)).reshape(()).tobytes()
        uniform = len(set(encoded)) == 1
        host = None if uniform else ctypes.create_string_buffer(encoded * prod(value.shape))

        def enqueue(queue):
            if queue.runtime is not value.runtime or queue.device != value.device:
                raise ValueError("initialization and queue must share their MLU device and runtime")
            if not value.pointer or not queue.handle.value:
                raise RuntimeError("initialization refers to closed MLU storage or queue")
            value.runtime.select(value.device)
            if value.nbytes:
                if uniform:
                    value.runtime.invoke("cnrtMemsetAsync", value.pointer, encoded[0], value.nbytes, queue.handle)
                else:
                    value.runtime.invoke("cnrtMemcpyAsync", value.pointer, ctypes.addressof(host),
                                         value.nbytes, queue.handle, 0)

        def launch():
            from .queue import Queue
            with Queue(device=self.device, neuware=self.neuware) as queue:
                enqueue(queue)
                queue.synchronize()

        return PreparedInitialization(launch, (value, host), _enqueue=enqueue)
