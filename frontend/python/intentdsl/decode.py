"""Head-group and page-table projections with author-owned split workspaces."""
import intentdsl as I
from .attention import merge_splits


@I.kernel
def grouped_partials(q: I.In[I.bf16, ("B", "H", 1, "D")],
                     keys: I.In[I.bf16, ("B", "KV", "T", "D")],
                     values: I.In[I.bf16, ("B", "KV", "T", "V")],
                     logs: I.Out[I.f32, ("B", "H", "S")],
                     partial: I.Out[I.bf16, ("B", "H", "S", "V")], scale: I.f32):
    for b in I.parallel(q.dim(0)):
        for h in I.parallel(q.dim(1)):
            for s in I.parallel(logs.dim(2)):
                kv = h // 4
                start = s * 256
                end = start + 256 if start + 256 < keys.dim(2) else keys.dim(2)
                query = I.cast(q[b, h, 0, :], I.f32)
                factor = scale * I.f32(1.4426950408889634)
                peak = -I.inf
                for t in I.fold(start, end):
                    key = I.cast(keys[b, kv, t, :], I.f32)
                    peak = I.maximum_num(peak, I.sum(query * key) * factor)
                mass = I.f32(0)
                vector = I.zeros(values.dim(3), I.f32)
                for t in I.fold(start, end):
                    key = I.cast(keys[b, kv, t, :], I.f32)
                    p = I.exp2(I.sum(query * key) * factor - peak)
                    mass = mass + p
                    vector = vector + I.cast(values[b, kv, t, :], I.f32) * I.cast(I.cast(p, I.bf16), I.f32)
                inv = I.f32(1) / mass if start < end else I.f32(0)
                partial[b, h, s, :] = I.cast(vector * inv, I.bf16)
                logs[b, h, s] = peak + I.log(mass) * I.f32(1.4426950408889634) if start < end else -I.inf


@I.kernel
def merge_grouped(partial: I.In[I.bf16, ("B", "H", "S", "D")],
                  logs: I.In[I.f32, ("B", "H", "S")],
                  out: I.Out[I.bf16, ("B", "H", 1, "D")]):
    for b in I.parallel(logs.dim(0)):
        for h in I.parallel(logs.dim(1)):
            peak = -I.inf
            for s in I.ordered(logs.dim(2)):
                peak = I.maximum_num(peak, logs[b, h, s])
            total = I.f32(0)
            vector = I.zeros(partial.dim(3), I.f32)
            for s in I.ordered(logs.dim(2)):
                w = I.exp2(logs[b, h, s] - peak)
                total = total + w
                vector = vector + I.cast(partial[b, h, s, :], I.f32) * w
            out[b, h, 0, :] = I.cast(vector * (I.f32(1) / total), I.bf16)


@I.program
def grouped_flash_decode(q, k, v, scale):
    logs = I.workspace(I.f32, (q.dim(0), q.dim(1), I.ceildiv(k.dim(2), 256)))
    partial = I.workspace(I.bf16, (q.dim(0), q.dim(1), I.ceildiv(k.dim(2), 256), v.dim(3)))
    out = I.output(I.bf16, (q.dim(0), q.dim(1), 1, v.dim(3)))
    grouped_partials(q, k, v, logs, partial, scale)
    merge_grouped(partial, logs, out)
    return out


@I.kernel
def latent_partials(q: I.In[I.f16, ("B", "H", "C")], qr: I.In[I.f16, ("B", "H", "R")],
                    latent: I.In[I.f16, ("P", "PS", "KV", "C")], rope: I.In[I.f16, ("P", "PS", "KV", "R")],
                    pages: I.In[I.i32, ("BP",)], table: I.In[I.i32, ("Slots",)], lengths: I.In[I.i32, ("B",)],
                    logs: I.Out[I.f32, ("B", "H", 8)], partial: I.Out[I.f32, ("B", "H", 8, "C")], scale: I.f32):
    for b in I.parallel(q.dim(0)):
        for h in I.parallel(q.dim(1)):
            for s in I.parallel(logs.dim(2)):
                kv = h // 128
                length = I.cast(lengths[b], I.index)
                pitch = (length + 7) // 8
                lo = s * pitch
                hi = lo + pitch if lo + pitch < length else length
                offset = I.cast(pages[b], I.index)
                width = latent.dim(1)
                query = I.cast(q[b, h, :], I.f32)
                position = I.cast(qr[b, h, :], I.f32)
                peak = -I.inf
                for t in I.fold(lo, hi):
                    p = I.cast(table[offset + t // width], I.index)
                    content = I.cast(latent[p, t % width, kv, :], I.f32)
                    key_pos = I.cast(rope[p, t % width, kv, :], I.f32)
                    peak = I.maximum_num(peak, (I.sum(query * content) + I.sum(position * key_pos)) * scale)
                mass = I.f32(0)
                vector = I.zeros(latent.dim(3), I.f32)
                for t in I.fold(lo, hi):
                    p = I.cast(table[offset + t // width], I.index)
                    content = I.cast(latent[p, t % width, kv, :], I.f32)
                    key_pos = I.cast(rope[p, t % width, kv, :], I.f32)
                    score = (I.sum(query * content) + I.sum(position * key_pos)) * scale
                    w = I.exp(score - peak)
                    mass = mass + w
                    vector = vector + content * I.cast(I.cast(w, I.f16), I.f32)
                inv = I.f32(1) / mass if lo < hi else I.f32(0)
                partial[b, h, s, :] = vector * inv
                logs[b, h, s] = peak + I.log(mass) if lo < hi else -I.inf


@I.program
def paged_mla_decode_partials(q, qr, latent, rope, pages, table, lengths, scale):
    logs = I.workspace(I.f32, (q.dim(0), q.dim(1), 8))
    partial = I.workspace(I.f32, (q.dim(0), q.dim(1), 8, q.dim(2)))
    out = I.output(I.f16, (q.dim(0), q.dim(1), q.dim(2)))
    latent_partials(q, qr, latent, rope, pages, table, lengths, logs, partial, scale)
    merge_splits(partial, logs, out)
    return out
