"""Author-owned multi-kernel calls, shared by usage examples and experiments.

These classes compose already compiled kernels. They do not choose a target,
change an algorithm, time a call, or load a reference implementation.
"""

from dataclasses import dataclass
from typing import ClassVar
from intent.runtime import PreparedSequence

from intent.examples.kernels.routing.moe_align import EXPERTS, PADDED_ROUTES, ROUTES, TOKENS, TOP_K
from .native import empty_like


@dataclass
class Histogram:
    mutable_inputs: ClassVar[tuple[int, ...]] = ()
    partials: object
    merge: object
    chunk: int = 8192

    def prepare(self, samples, *, outputs=None):
        from intent.examples.kernels.statistics.histogram import BINS
        partitions = (samples.shape[0] + self.chunk - 1) // self.chunk
        # The workspace belongs to this author program and is fully overwritten
        # by its first kernel; no fill or hidden initialization launch is needed.
        allocate = getattr(self.partials, "empty_like", empty_like)
        workspace = allocate(samples, (partitions, BINS), dtype="i32")
        first = self.partials.prepare(samples, outputs=(workspace,))
        second = self.merge.prepare(first.result(), outputs=outputs)
        return PreparedSequence((first, second), outputs=second.result(), owners=(workspace,))

    def run(self, samples):
        call = self.prepare(samples)
        call.launch()
        return call.result()


@dataclass
class GroupNormBackward:
    mutable_inputs: ClassVar[tuple[int, ...]] = ()
    dx: object
    affine: object

    def run(self, x, grad_y, weight, mean, rstd, inverse_group_elements):
        grad_x = self.dx.run(x, grad_y, weight, mean, rstd, inverse_group_elements)
        grad_weight, grad_bias = self.affine.run(x, grad_y, mean, rstd)
        return grad_x, grad_weight, grad_bias

    def into(self, x, grad_y, weight, mean, rstd, grad_x, grad_weight, grad_bias,
             inverse_group_elements):
        self.dx(x, grad_y, weight, mean, rstd, grad_x, inverse_group_elements)
        self.affine(x, grad_y, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias

    def prepare(self, x, grad_y, weight, mean, rstd, inverse_group_elements):
        dx = self.dx.prepare(x, grad_y, weight, mean, rstd, inverse_group_elements)
        affine = self.affine.prepare(x, grad_y, mean, rstd)
        return PreparedSequence((dx, affine), outputs=(dx.result(), *affine.result()))


@dataclass
class PagedDecode:
    mutable_inputs: ClassVar[tuple[int, ...]] = ()
    partials: object
    merge: object

    def run(self, q, key_cache, value_cache, page_offsets, page_indices,
            lengths, split_offsets, scale):
        lse, output = self.partials.run(q, key_cache, value_cache, page_offsets,
                                       page_indices, lengths, split_offsets, scale)
        return self.merge.run(output, lse)

    def into(self, q, key_cache, value_cache, page_offsets, page_indices,
             lengths, split_offsets, partial_lse, partial_output, output, scale):
        self.partials(q, key_cache, value_cache, page_offsets, page_indices,
                      lengths, split_offsets, partial_lse, partial_output, scale)
        self.merge(partial_output, partial_lse, output)
        return output

    def prepare(self, q, key_cache, value_cache, page_offsets, page_indices,
                lengths, split_offsets, scale):
        partials = self.partials.prepare(q, key_cache, value_cache, page_offsets,
                                         page_indices, lengths, split_offsets, scale)
        # result() returns allocated output handles without launching or reading
        # their contents. The merge is bound to those exact intermediate views.
        partial_lse, partial_output = partials.result()
        merge = self.merge.prepare(partial_output, partial_lse)
        return PreparedSequence((partials, merge), outputs=merge.result())


@dataclass
class ChunkGatedDelta:
    mutable_inputs: ClassVar[tuple[int, ...]] = ()
    partition: object
    recurrence: object

    def prepare(self, query, key, value, gate, beta, scale):
        partition = self.partition.prepare(query, key, value, gate, beta, scale)
        recurrence = self.recurrence.prepare(*partition.result())
        return PreparedSequence((partition, recurrence), outputs=recurrence.result())

    def run(self, *arguments):
        call = self.prepare(*arguments)
        call.launch()
        return call.result()


@dataclass
class MoEAlignment:
    mutable_inputs: ClassVar[tuple[int, ...]] = ()
    count: object
    prefix: object
    scatter: object
    mark: object

    @staticmethod
    def _validate_ids(ids):
        if len(ids.shape) != 2 or ids.shape[1] != TOP_K or ids.shape[0] > TOKENS:
            raise ValueError(f"MoEAlignment requires at most {TOKENS} tokens with top-k {TOP_K}")

    def _workspace(self, ids):
        allocate = getattr(self.count, "empty_like", empty_like)
        counts = allocate(ids, (EXPERTS,), dtype="i32")
        cursors = allocate(ids, (EXPERTS,), dtype="i32")
        sorted_routes = allocate(ids, (PADDED_ROUTES,), dtype="i32")
        return counts, cursors, sorted_routes

    def run_into(self, ids, counts, cursors, sorted_routes):
        call = self._prepare_into(ids, counts, cursors, sorted_routes)
        call.launch()
        return call.result()

    def run(self, ids):
        counts, cursors, sorted_routes = self._workspace(ids)
        return self.run_into(ids, counts, cursors, sorted_routes)

    def prepare(self, ids):
        self._validate_ids(ids)
        counts, cursors, sorted_routes = self._workspace(ids)
        return self._prepare_into(ids, counts, cursors, sorted_routes)

    def _prepare_into(self, ids, counts, cursors, sorted_routes):
        self._validate_ids(ids)
        initializations = tuple(self.count.prepare_fill(buffer, value)
                               for buffer, value in ((counts, 0), (cursors, 0), (sorted_routes, ROUTES)))
        count = self.count.prepare(ids, counts)
        prefix = self.prefix.prepare(counts)
        offsets, total_padded = prefix.result()
        scatter = self.scatter.prepare(ids, offsets, cursors, sorted_routes)
        mark = self.mark.prepare(offsets)
        return PreparedSequence((*initializations, count, prefix, scatter, mark),
            outputs=(sorted_routes, mark.result(), total_padded), owners=(counts, cursors))
