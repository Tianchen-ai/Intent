"""Contractions and normalization inside independent query or split owners."""

import intentdsl as I


@I.kernel(numerics="relaxed_normalization")
def attention(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    y: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
):
    for batch in I.parallel(q.dim(0)):
        for head in I.parallel(q.dim(1)):
            for query in I.parallel(q.dim(2)):
                kv = head // 4
                query_vector = I.cast(q[batch, head, query, :], I.f32)
                log2_scale = scale * I.f32(1.4426950408889634)
                peak = -I.inf
                for token in I.fold(k.dim(2)):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    score = I.sum(query_vector * key) * log2_scale
                    peak = I.maximum_num(peak, score if query >= token else -I.inf)
                mass = I.f32(0)
                weighted = I.zeros(v.dim(3), I.f32)
                for token in I.fold(k.dim(2)):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    score = I.sum(query_vector * key) * log2_scale
                    weight = I.exp2(score - peak, approximate=True, flush_to_zero=True) if query >= token else I.f32(0)
                    mass = mass + weight
                    factor = I.cast(I.cast(weight, I.bf16), I.f32)
                    weighted = weighted + I.cast(v[batch, kv, token, :], I.f32) * factor
                inverse = I.f32(0) if k.dim(2) == 0 else I.fdiv(I.f32(1), mass, approximate=True, flush_to_zero=True)
                y[batch, head, query, :] = I.cast(weighted * inverse, I.bf16)


@I.kernel
def merge_splits(
    partial: I.In[I.f32, ("B", "H", "S", "D")],
    lse: I.In[I.f32, ("B", "H", "S")],
    y: I.Out[I.f16, ("B", "H", "D")],
):
    for batch in I.parallel(lse.dim(0)):
        for head in I.parallel(lse.dim(1)):
            peak = -I.inf
            for split in I.fold(lse.dim(2)):
                peak = I.maximum_num(peak, lse[batch, head, split])
            weighted = I.zeros(partial.dim(3), I.f32)
            mass = I.f32(0)
            for split in I.fold(lse.dim(2)):
                weight = I.exp(lse[batch, head, split] - peak)
                weighted = weighted + partial[batch, head, split, :] * weight
                mass = mass + weight
            y[batch, head, :] = I.cast(weighted * I.fdiv(I.f32(1), mass, approximate=True, flush_to_zero=True), I.f16)


@I.kernel
def merge_log2_splits(
    partial: I.In[I.f32, ("B", "H", "S", "D")],
    lse: I.In[I.f32, ("B", "H", "S")],
    y: I.Out[I.f16, ("B", "H", "D")],
):
    for b in I.parallel(lse.dim(0)):
        for h in I.parallel(lse.dim(1)):
            peak = -I.inf
            for s in I.fold(lse.dim(2)):
                peak = I.maximum_num(peak, lse[b, h, s])
            mass = I.f32(0)
            vector = I.zeros(partial.dim(3), I.f32)
            for s in I.fold(lse.dim(2)):
                weight = I.exp2(lse[b, h, s] - peak, approximate=True, flush_to_zero=True)
                mass = mass + weight
                vector = vector + partial[b, h, s, :] * weight
            y[b, h, :] = I.cast(vector * I.fdiv(I.f32(1), mass, approximate=True, flush_to_zero=True), I.f16)


@I.kernel(numerics="relaxed_normalization")
def paged_partials(
    q: I.In[I.f16, ("B", "H", "D")],
    keys: I.In[I.f16, ("P", "PS", "KV", "D")],
    values: I.In[I.f16, ("P", "PS", "KV", "V")],
    pages: I.In[I.i32, ("BP",)],
    table: I.In[I.i32, ("Entries",)],
    lengths: I.In[I.i32, ("B",)],
    cuts: I.In[I.i32, ("Cuts",)],
    lse: I.Out[I.f32, ("B", "H", 8)],
    partial: I.Out[I.f32, ("B", "H", 8, "V")],
    scale: I.f32,
):
    for batch in I.parallel(q.dim(0)):
        for head in I.parallel(q.dim(1)):
            for split in I.parallel(lse.dim(2)):
                job = batch * lse.dim(2) + split
                kv = head // 4
                page_size = keys.dim(1)
                origin = I.cast(pages[batch], I.index)
                start = (I.cast(cuts[job], I.index) - origin) * page_size
                limit = (I.cast(cuts[job + 1], I.index) - origin) * page_size
                length = I.cast(lengths[batch], I.index)
                stop = limit if limit < length else length
                query = I.cast(q[batch, head, :], I.f32)
                log2_scale = scale * I.f32(1.4426950408889634)
                peak = -I.inf
                for pos in I.fold(start, stop):
                    page = I.cast(table[origin + pos // page_size], I.index)
                    key = I.cast(keys[page, pos % page_size, kv, :], I.f32)
                    peak = I.maximum_num(peak, I.sum(query * key) * log2_scale)
                weighted = I.zeros(values.dim(3), I.f32)
                mass = I.f32(0)
                for pos in I.fold(start, stop):
                    page = I.cast(table[origin + pos // page_size], I.index)
                    key = I.cast(keys[page, pos % page_size, kv, :], I.f32)
                    weight = I.exp2(I.sum(query * key) * log2_scale - peak, approximate=True, flush_to_zero=True)
                    mass = mass + weight
                    value = I.cast(values[page, pos % page_size, kv, :], I.f32)
                    weighted = weighted + I.cast(I.cast(weight, I.f16), I.f32) * value
                inv = (
                    I.fdiv(I.f32(1), mass, approximate=True, flush_to_zero=True)
                    if start < stop
                    else I.f32(0)
                )
                partial[batch, head, split, :] = weighted * inv
                lse[batch, head, split] = peak + I.log(mass) * I.f32(1.4426950408889634) if start < stop else -I.inf


@I.program
def paged_decode(q, keys, values, pages, table, lengths, cuts, scale):
    logs = I.workspace(I.f32, (q.dim(0), q.dim(1), 8))
    partial = I.workspace(I.f32, (q.dim(0), q.dim(1), 8, values.dim(3)))
    y = I.output(I.f16, (q.dim(0), q.dim(1), values.dim(3)))
    paged_partials(q, keys, values, pages, table, lengths, cuts, logs, partial, scale)
    merge_log2_splits(partial, logs, y)
    return y
