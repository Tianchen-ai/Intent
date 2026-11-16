"""Host storage adapter; ordinary JIT still accepts existing native buffers."""
from .buffer import Buffer, ELEMENT_BYTES


class HostStorage:
    copies_host = False
    contiguous_views = True

    def accepts(self, value):
        return isinstance(value, Buffer)

    def allocate(self, shape, element):
        return Buffer.empty(tuple(shape), element)

    def upload(self, host, element):
        return memoryview(host)

    def download(self, allocation):
        return allocation

    def view(self, allocation, shape, strides, offset, element):
        from math import prod
        width = ELEMENT_BYTES[element]
        return Buffer(allocation[offset * width:(offset + prod(shape)) * width], shape=shape, dtype=element)

    def snapshot(self, value):
        owner = memoryview(value.storage.obj).cast("B") if isinstance(value, Buffer) else memoryview(value.obj).cast("B")
        saved = bytes(owner)
        def restore():
            owner[:] = saved
        return id(owner.obj), restore

    def release(self, value):
        pass  # The borrowed host allocation is held by the buffer's memoryview.
