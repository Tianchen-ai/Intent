"""Author-owned multi-kernel calls, shared by usage examples and experiments.

These classes compose already compiled kernels. They do not choose a target,
change an algorithm, time a call, or load a reference implementation.
"""

from dataclasses import dataclass

from kernels.routing.moe_align import EXPERTS, PADDED_ROUTES, ROUTES, TOKENS, TOP_K
from .native import empty_like, fill


@dataclass
class GroupNormBackwardCall:
    dx: object
    affine: object

    def compile(self):
        self.dx.compile()
        self.affine.compile()

    def launch(self):
        self.dx.launch()
        self.affine.launch()

    def result(self):
        return self.dx.result(), *self.affine.result()


@dataclass
class GroupNormBackward:
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
        return GroupNormBackwardCall(dx, affine)


@dataclass
class PagedDecodeCall:
    partials: object
    merge: object

    def compile(self):
        self.partials.compile()
        self.merge.compile()

    def launch(self):
        self.partials.launch()
        self.merge.launch()

    def result(self):
        return self.merge.result()


@dataclass
class PagedDecode:
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
        return PagedDecodeCall(partials, merge)


def reset_moe_workspace(counts, cursors, sorted_routes):
    fill(counts, 0)
    fill(cursors, 0)
    fill(sorted_routes, ROUTES)


@dataclass
class MoEAlignmentCall:
    count: object
    prefix: object
    scatter: object
    mark: object
    sorted_routes: object
    expert_blocks: object
    total_padded: object
    counts: object
    cursors: object

    def compile(self):
        self.count.compile()
        self.prefix.compile()
        self.scatter.compile()
        self.mark.compile()

    def launch(self):
        reset_moe_workspace(self.counts, self.cursors, self.sorted_routes)
        self.count.launch()
        self.prefix.launch()
        self.scatter.launch()
        self.mark.launch()

    def result(self):
        return self.sorted_routes, self.expert_blocks, self.total_padded


@dataclass
class MoEAlignment:
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
        self._validate_ids(ids)
        reset_moe_workspace(counts, cursors, sorted_routes)
        self.count.run(ids, counts)
        offsets, total_padded = self.prefix.run(counts)
        self.scatter.run(ids, offsets, cursors, sorted_routes)
        expert_blocks = self.mark.run(offsets)
        return sorted_routes, expert_blocks, total_padded

    def run(self, ids):
        counts, cursors, sorted_routes = self._workspace(ids)
        return self.run_into(ids, counts, cursors, sorted_routes)

    def prepare(self, ids):
        self._validate_ids(ids)
        counts, cursors, sorted_routes = self._workspace(ids)
        count = self.count.prepare(ids, counts)
        prefix = self.prefix.prepare(counts)
        offsets, total_padded = prefix.result()
        scatter = self.scatter.prepare(ids, offsets, cursors, sorted_routes)
        mark = self.mark.prepare(offsets)
        return MoEAlignmentCall(count, prefix, scatter, mark,
                                sorted_routes, mark.result(), total_padded, counts, cursors)
