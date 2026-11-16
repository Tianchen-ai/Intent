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
