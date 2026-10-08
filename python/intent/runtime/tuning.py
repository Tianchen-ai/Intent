from __future__ import annotations

from dataclasses import dataclass
from .torch_views import view_byte_span


@dataclass
class _StorageSpan:
    start: int
    end: int
    entries: list[tuple[int, object, int, int]]


class TuningState:
    """Private writable allocations with the invocation's actual view aliases."""

    def __init__(self, views: tuple, writable: tuple[bool, ...]):
        import torch

        spans = []
        for index, view in enumerate(views):
            start, end = view_byte_span(view)
            if start == end:
                continue
            spans.append(_StorageSpan(start, end, [(index, view, start, end)]))
        spans.sort(key=lambda span: span.start)
        groups: list[_StorageSpan] = []
        for span in spans:
            if groups and span.start < groups[-1].end:
                groups[-1].end = max(groups[-1].end, span.end)
                groups[-1].entries.extend(span.entries)
            else:
                groups.append(span)

        arguments = list(views)
        self._copies: list[tuple[object, object, tuple[int, ...]]] = []
        for group in groups:
            if not any(writable[index] for index, _, _, _ in group.entries):
                continue
            device = group.entries[0][1].device
            # Retain low address alignment bits as well as relative byte offsets.
            padding = group.start % 256
            scratch = torch.empty(group.end - group.start + padding,
                                  dtype=torch.uint8, device=device)
            copied = set()
            members = tuple(index for index, _, _, _ in group.entries)
            for index, view, start, end in group.entries:
                storage = view.untyped_storage()
                size = end - start
                offset = start - group.start + padding
                if (start, size) not in copied:
                    original = torch.empty(0, dtype=torch.uint8, device=device).set_(
                        storage, start - storage.data_ptr(), (size,), (1,))
                    self._copies.append((original, scratch[offset:offset + size], members))
                    copied.add((start, size))
                byte_offset = view.data_ptr() - group.start + padding
                arguments[index] = torch.empty(0, dtype=view.dtype, device=device).set_(
                    scratch.untyped_storage(), byte_offset // view.element_size(),
                    view.shape, view.stride())
        self.views = tuple(arguments)
        self.reset()

    def reset(self) -> None:
        for original, scratch, _ in self._copies:
            scratch.copy_(original)

    def restore(self, readable: tuple[bool, ...] | None = None) -> None:
        for original, scratch, members in self._copies:
            if readable is None or any(readable[index] for index in members):
                original.copy_(scratch)

    def arguments(self, arguments: tuple) -> tuple:
        self.reset()
        return self.views + arguments[len(self.views):]
