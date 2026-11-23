import intentdsl as I
from .attention import attention, merge_splits, paged_decode
from .attention_grad import flash_attention_backward
from .decode import grouped_flash_decode, paged_mla_decode_partials
from .recurrences import causal_convolution, causal_linear_attention, cholesky, gated_delta, selective_state_scan
from .matrix import (gemm, batched_gemm, ragged_gemm, int8_gemm, q4_projection, mamba_chunk_state,
                     fp8_gemm, qkv_projection, expert_projection)

@I.kernel
def count_routes(ids:I.In[I.i32,(4096,2)], counts:I.InOut[I.i32,(64,)]):
    for r in I.parallel(ids.dim(0)):
        I.atomic_add(counts,ids[r,:],I.cast(1,I.i32),order="relaxed",in_bounds=True)

@I.kernel
def mark_route_blocks(offsets:I.In[I.i32,(65,)], blocks:I.InOut[I.i32,(128,)]):
    for expert in I.parallel(offsets.dim(0)):
        I.fill_ranges(blocks,offsets,expert,unit=128,disjoint_in_bounds=True)

@I.kernel
def aligned_offsets(counts:I.In[I.i32,(64,)], offsets:I.Out[I.i32,(65,)], total:I.Out[I.i32,(1,)]):
    for e in I.parallel(counts.dim(0)):
        n = counts[e]+I.cast(127,I.i32)
        padded = (n >> I.cast(7,I.i32)) << I.cast(7,I.i32)
        prefix = I.cumsum(padded,axis=0)
        I.write_offsets(offsets,total,prefix)

@I.kernel
def place_routes(ids:I.In[I.i32,(4096,2)], offsets:I.In[I.i32,(65,)], cursors:I.InOut[I.i32,(64,)], routes:I.InOut[I.i32,(16320,)]):
    for r in I.parallel(ids.dim(0)):
        key=ids[r,:]
        ticket=I.atomic_add(cursors,key,I.cast(1,I.i32),order="relaxed",in_bounds=True)
        dst=I.gather(offsets,key,in_bounds=True)+ticket
        I.scatter_unique(routes,dst,I.linear_index(ids,r),in_bounds=True)

@I.program
def moe_alignment_into(ids, expert_blocks:I.InOut[I.i32,(128,)]):
    counts=I.workspace(I.i32,(64,))
    cursors=I.workspace(I.i32,(64,))
    offsets=I.workspace(I.i32,(65,))
    routes=I.output(I.i32,(16320,))
    total=I.output(I.i32,(1,))
    I.fill(counts,0)
    I.fill(cursors,0)
    I.fill(routes,8192)
    count_routes(ids,counts)
    aligned_offsets(counts,offsets,total)
    place_routes(ids,offsets,cursors,routes)
    mark_route_blocks(offsets,expert_blocks)
    return routes,expert_blocks,total

@I.kernel
def nonzero(values:I.In[I.f32,("M","N")], indices:I.Out[I.i32,("M","N")], counts:I.Out[I.i32,("M",)]):
    for r in I.parallel(values.dim(0)):
        keep = values[r,:] != I.cast(0.0,I.f32)
        rank = I.cumsum(I.cast(keep,I.i32),axis=0)
        I.scatter_unique(indices[r,:], rank, I.column_index(values,r), valid=keep,fill=-1,count=counts[r])

@I.kernel
def count_partitions(samples:I.In[I.f32,("N",)], counts:I.Out[I.i32,("P",256)]):
    for partition in I.parallel(counts.dim(0)):
        counts[partition,:] = I.reduce_by_key(I.cast(I.chunk(samples,partition,8192),I.i32),1,
                                               size=counts.dim(1),invalid="skip",combine=I.add,init=0)

@I.kernel
def merge_counts(partials:I.In[I.i32,("P",256)], bins:I.Out[I.i32,(256,)]):
    for column in I.parallel(bins.dim(0)):
        bins[column] = I.sum_columns(partials)

@I.program
def histogram(samples):
    parts=I.workspace(I.i32,(I.ceildiv(samples.dim(0),8192),256))
    bins=I.output(I.i32,(256,))
    count_partitions(samples,parts)
    merge_counts(parts,bins)
    return bins

@I.kernel
def count_index_parts(samples:I.In[I.i64,("N",)], counts:I.Out[I.i32,("P","B")]):
    for part in I.parallel(counts.dim(0)):
        counts[part,:] = I.reduce_by_key(I.chunk(samples,part,8192),1,
                                         size=counts.dim(1),invalid="skip",combine=I.add,init=0)

@I.kernel
def merge_counts_f32(parts:I.In[I.i32,("P","B")], output:I.Out[I.f32,("B",)]):
    for k in I.parallel(output.dim(0)):
        output[k] = I.cast(I.sum_columns(parts),I.f32)

@I.program
def histogram_from_indices(samples, bin_count):
    parts=I.workspace(I.i32,(I.ceildiv(samples.dim(0),8192),bin_count))
    result=I.output(I.f32,(bin_count,))
    count_index_parts(samples,parts)
    merge_counts_f32(parts,result)
    return result

@I.kernel
def cumsum(values: I.In[I.f32, ("M","N")], prefix: I.Out[I.f32, ("M","N")]):
    for row in I.parallel(values.dim(0)):
        prefix[row,:] = I.cumsum(values[row,:], axis=0)

@I.kernel
def embedding(table:I.In[I.bf16,("V","D")], ids:I.In[I.i64,("T",)], output:I.Out[I.bf16,("T","D")]):
    for token in I.parallel(output.dim(0)):
        output[token,:] = I.gather_row(table,ids,token,in_bounds=True)

@I.kernel
def csr_spmv(offsets:I.In[I.i32,(32769,)], indices:I.In[I.i32,("Z",)], data:I.In[I.f32,("Z",)],
             x:I.In[I.f32,(32768,)], y:I.Out[I.f32,(32768,)]):
    for row in I.parallel(y.dim(0)):
        y[row] = I.ordered_sum(I.segment(offsets,row),lambda p:data[p]*x[indices[p]])

@I.kernel
def jagged_mean(values:I.In[I.f32,("T","D")], offsets:I.In[I.i32,(513,)], output:I.Out[I.f32,(512,"D")]):
    for row in I.parallel(output.dim(0)):
        total = I.ordered_sum(I.segment(offsets,row),lambda p:values[p,:])
        output[row,:] = total / I.segment_length(offsets,row)


@I.kernel
def softmax(x: I.In[I.f16, ("M", "N")], out: I.Out[I.f16, ("M", "N")]):
    for r in I.parallel(x.dim(0)):
        v = I.cast(x[r, :], I.f32)
        m = I.reduce(v, axis=0, combine=I.max_num, init=I.neg_inf(I.f32))
        w = I.exp(v - m)
        out[r, :] = I.cast(w / I.sum(w, axis=0, acc=I.f32), I.f16)


@I.kernel
def softmax_f32(x: I.In[I.f32, ("M", "N")], out: I.Out[I.f32, ("M", "N")]):
    for r in I.parallel(x.dim(0)):
        v = x[r, :]
        m = I.reduce(v, axis=0, combine=I.max_num, init=I.neg_inf(I.f32))
        w = I.exp(v - m)
        out[r, :] = w / I.sum(w, axis=0, acc=I.f32)


@I.kernel
def rms_norm(
    x: I.In[I.f32, ("M", "N")],
    gain: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
    inv_c: I.f32,
    eps: I.f32,
):
    for r in I.parallel(x.dim(0)):
        u = x[r, :]
        s = I.sum(u * u, axis=0, acc=I.f32)
        out[r, :] = (u * I.rsqrt(s * inv_c + eps)) * gain[:]


@I.kernel
def relu(x: I.In[I.f16, ("M", "N")], out: I.Out[I.f16, ("M", "N")]):
    for row in I.parallel(x.dim(0)):
        out[row, :] = I.maximum(x[row, :], I.cast(0.0, I.f16))


@I.kernel
def layer_norm(x: I.In[I.f32, ("R", "C")], gain: I.In[I.f32, ("C",)],
               offset: I.In[I.f32, ("C",)], out: I.Out[I.f32, ("R", "C")],
               inv_width: I.f32, epsilon: I.f32):
    for row in I.parallel(x.dim(0)):
        v = x[row, :]
        mean = I.sum(v, axis=0, acc=I.f32) * inv_width
        second = I.sum(v * v, axis=0, acc=I.f32) * inv_width
        variance = second - mean * mean
        out[row, :] = (v - mean) * I.rsqrt(variance + epsilon) * gain[:] + offset[:]


@I.kernel
def fused_add_rms_norm(x: I.In[I.bf16, ("R", "C")],
                       residual: I.In[I.bf16, ("R", "C")],
                       weight: I.In[I.bf16, ("C",)],
                       out: I.Out[I.bf16, ("R", "C")],
                       residual_out: I.Out[I.bf16, ("R", "C")],
                       inv_width: I.f32, epsilon: I.f32, offset: I.f32):
    for row in I.parallel(x.dim(0)):
        rounded = I.cast(I.cast(x[row, :], I.f32) + I.cast(residual[row, :], I.f32), I.bf16)
        v = I.cast(rounded, I.f32)
        scale = I.rsqrt(I.sum(v * v, axis=0, acc=I.f32) * inv_width + epsilon)
        gain = I.cast(weight[:], I.f32) + offset
        out[row, :] = I.cast(v * scale * gain, I.bf16)
        residual_out[row, :] = rounded


@I.kernel
def batch_norm_training(x: I.In[I.f16, ("B","C","S")], weight: I.In[I.f32, ("C",)],
               bias: I.In[I.f32, ("C",)], moving_mean: I.InOut[I.f32, ("C",)],
               moving_var: I.InOut[I.f32, ("C",)], y: I.Out[I.f16, ("B","C","S")],
               mean: I.Out[I.f32, ("C",)], inv_std: I.Out[I.f32, ("C",)],
               epsilon: I.f32, momentum: I.f32):
    for ch in I.parallel(x.dim(1)):
        samples = I.cast(x[:,ch,:], I.f32)
        count, center, spread = I.welford(samples)
        n = I.cast(count,I.f32)
        variance = spread / n
        inv = I.rsqrt(variance + epsilon)
        previous_mean = moving_mean[ch]
        previous_var = moving_var[ch]
        gain = weight[ch]
        offset = bias[ch]
        one = I.cast(1.,I.f32)
        moving_mean[ch] = (one-momentum)*previous_mean + momentum*center
        moving_var[ch] = (one-momentum)*previous_var + momentum*variance*n/(n-one)
        y[:,ch,:] = I.cast((samples-center)*inv*gain+offset,I.f16)
        mean[ch] = center
        inv_std[ch] = inv


@I.program
def batch_norm(x, weight, bias, moving_mean, moving_var, epsilon, momentum):
    y = I.output(I.f16,(x.dim(0),x.dim(1),x.dim(2)))
    mean = I.output(I.f32,(x.dim(1),))
    inv = I.output(I.f32,(x.dim(1),))
    batch_norm_training(x,weight,bias,moving_mean,moving_var,y,mean,inv,epsilon,momentum)
    return y,mean,inv,moving_mean,moving_var


@I.kernel
def merge_affine_gradients(weight_partials: I.In[I.bf16, ("P", "F")],
                           bias_partials: I.In[I.bf16, ("P", "F")],
                           dw: I.Out[I.f32, ("F",)], db: I.Out[I.f32, ("F",)]):
    for feature in I.parallel(weight_partials.dim(1)):
        weight_terms = I.cast(weight_partials[:, feature], I.f32)
        bias_terms = I.cast(bias_partials[:, feature], I.f32)
        dw[feature] = I.sum(weight_terms, axis=0, acc=I.f32)
        db[feature] = I.sum(bias_terms, axis=0, acc=I.f32)


@I.kernel
def layer_gradient_rows(x: I.In[I.bf16,(4096,"N")], dy: I.In[I.bf16,(4096,"N")],
                        gain: I.In[I.bf16,("N",)], mean: I.In[I.f32,(4096,)],
                        inv: I.In[I.f32,(4096,)], dx: I.Out[I.bf16,(4096,"N")],
                        wp: I.InOut[I.bf16,(128,"N")], bp: I.InOut[I.bf16,(128,"N")],
                        inverse_width: I.f32):
    for bucket in I.parallel(wp.dim(0)):
        col = I.domain(0,x.dim(1))
        weight_sum = wp[bucket,col]
        bias_sum = bp[bucket,col]
        weight = I.cast(gain[col],I.f32)
        for round_id in I.static_range(32):
            value = I.cast(x[128*round_id+bucket,col],I.f32)
            grad = I.cast(dy[128*round_id+bucket,col],I.f32)
            rs = inv[128*round_id+bucket]
            normalized = (value-mean[128*round_id+bucket])*rs
            weighted = weight*grad
            center = I.sum(weighted,axis=0,acc=I.f32)*inverse_width
            projection = I.sum(normalized*weighted,axis=0,acc=I.f32)*inverse_width
            dx[128*round_id+bucket,col] = I.cast((weighted-center-normalized*projection)*rs,I.bf16)
            weight_term = I.cast(grad*normalized,I.bf16)
            weight_sum = I.cast(I.cast(weight_sum,I.f32)+I.cast(weight_term,I.f32),I.bf16)
            bias_sum = I.cast(I.cast(bias_sum,I.f32)+grad,I.bf16)
        wp[bucket,col] = weight_sum
        bp[bucket,col] = bias_sum


@I.program
def layer_norm_backward(x, dy, gain, mean, inv, inverse_width):
    dx = I.output(I.bf16,(x.dim(0),x.dim(1)))
    wp = I.workspace(I.bf16,(128,x.dim(1)))
    bp = I.workspace(I.bf16,(128,x.dim(1)))
    dw = I.output(I.f32,(x.dim(1),))
    db = I.output(I.f32,(x.dim(1),))
    I.fill(wp,0)
    I.fill(bp,0)
    layer_gradient_rows(x,dy,gain,mean,inv,dx,wp,bp,inverse_width)
    merge_affine_gradients(wp,bp,dw,db)
    return dx,dw,db


@I.kernel
def group_input_gradient(x: I.In[I.f16, ("B","C","S")],
                         dy: I.In[I.f16, ("B","C","S")], gain: I.In[I.f16, ("C",)],
                         mean: I.In[I.f16, ("B","G")], inv: I.In[I.f16, ("B","G")],
                         dx: I.Out[I.f16, ("B","C","S")], inv_count: I.f32):
    for batch, group in I.parallel((x.dim(0), mean.dim(1))):
        position = I.domain(0,x.dim(2))
        total = I.cast(0,I.f32)
        projection = I.cast(0,I.f32)
        center = I.cast(mean[batch,group],I.f32)
        inverse = I.cast(inv[batch,group],I.f32)
        for h in I.static_range(8):
            normalized = (I.cast(x[batch,8*group+h,position],I.f32)-center)*inverse
            upstream = I.cast(dy[batch,8*group+h,position],I.f32)*I.cast(gain[8*group+h],I.f32)
            total = total+I.sum(upstream,axis=0,acc=I.f32)
            projection = projection+I.sum(normalized*upstream,axis=0,acc=I.f32)
        for h in I.static_range(8):
            normalized = (I.cast(x[batch,8*group+h,position],I.f32)-center)*inverse
            upstream = I.cast(dy[batch,8*group+h,position],I.f32)*I.cast(gain[8*group+h],I.f32)
            dx[batch,8*group+h,position] = I.cast(inverse*(upstream-total*inv_count-normalized*projection*inv_count),I.f16)


@I.kernel
def group_affine_gradient(x: I.In[I.f16, ("B","C","S")], dy: I.In[I.f16, ("B","C","S")],
                          mean: I.In[I.f16, ("B","G")], inv: I.In[I.f16, ("B","G")],
                          dw: I.Out[I.f16, ("C",)], db: I.Out[I.f16, ("C",)]):
    for group in I.parallel(mean.dim(1)):
        batch = I.domain(0,x.dim(0))
        position = I.domain(0,x.dim(2))
        center = I.cast(mean[batch,group],I.f32)
        inverse = I.cast(inv[batch,group],I.f32)
        for h in I.static_range(8):
            normalized = (I.cast(x[batch,8*group+h,position],I.f32)-center)*inverse
            upstream = I.cast(dy[batch,8*group+h,position],I.f32)
            weight_sum = I.sum(normalized*upstream,axis=0,acc=I.f32)
            bias_sum = I.sum(upstream,axis=0,acc=I.f32)
            dw[8*group+h] = I.cast(weight_sum,I.f16)
            db[8*group+h] = I.cast(bias_sum,I.f16)


@I.program
def group_norm_backward(x, dy, gain, mean, inv, inv_count):
    dx = I.output(I.f16,(x.dim(0),x.dim(1),x.dim(2)))
    dw = I.output(I.f16,(x.dim(1),))
    db = I.output(I.f16,(x.dim(1),))
    group_input_gradient(x,dy,gain,mean,inv,dx,inv_count)
    group_affine_gradient(x,dy,mean,inv,dw,db)
    return dx,dw,db


@I.kernel
def transpose(x: I.In[I.f16, ("M", "N")], out: I.Out[I.f16, ("N", "M")]):
    for row in I.parallel(x.dim(0)):
        out[:, row] = x[row, :]


@I.kernel
def swiglu(
    gate: I.In[I.bf16, ("M", "N")],
    up: I.In[I.bf16, ("M", "N")],
    out: I.Out[I.bf16, ("M", "N")],
):
    for row in I.parallel(gate.dim(0)):
        x = I.cast(gate[row, :], I.f32)
        one = I.cast(1.0, I.f32)
        sigmoid = one / (one + I.exp(I.cast(0.0, I.f32) - x))
        rounded = I.cast(x * sigmoid, I.bf16)
        out[row, :] = rounded * up[row, :]


@I.kernel
def dropout(
    x: I.In[I.f16, ("M", "N")],
    out: I.Out[I.f16, ("M", "N")],
    seed: I.i32,
    probability: I.f32,
    scale: I.f32,
):
    for row in I.parallel(x.dim(0)):
        h = I.linear_index(x, row) * I.cast(1103515245, I.i32) + seed
        a = h ^ (h >> I.cast(16, I.i32))
        b = a ^ (a << I.cast(8, I.i32))
        c = b ^ (b >> I.cast(4, I.i32))
        u = I.cast(c & I.cast(2147483647, I.i32), I.f32) / I.cast(2147483647.0, I.f32)
        out[row, :] = I.where(
            u > probability, x[row, :] * I.cast(scale, I.f16), I.cast(0.0, I.f16)
        )


@I.kernel
def adamw(
    g: I.In[I.f32, ("N",)],
    p: I.InOut[I.f32, ("N",)],
    m: I.InOut[I.f32, ("N",)],
    v: I.InOut[I.f32, ("N",)],
    lr: I.f32,
    beta1: I.f32,
    beta2: I.f32,
    correction1: I.f32,
    correction2: I.f32,
    eps: I.f32,
    decay: I.f32,
):
    for i in I.parallel(g.dim(0)):
        one = I.cast(1.0, I.f32)
        grad = g[i]
        mean = beta1 * m[i] + (one - beta1) * grad
        square = beta2 * v[i] + ((one - beta2) * grad) * grad
        step = (mean / correction1) / (I.sqrt(square / correction2) + eps) + decay * p[
            i
        ]
        updated = p[i] - lr * step
        m[i] = mean
        v[i] = square
        p[i] = updated


@I.kernel
def dropout_norm_forward(
    x: I.In[I.bf16, ("M", "N")],
    residual: I.In[I.bf16, ("M", "N")],
    gain: I.In[I.bf16, ("N",)],
    normalized: I.Out[I.bf16, ("M", "N")],
    summed: I.Out[I.bf16, ("M", "N")],
    seed: I.u64,
    keep: I.f32,
    scale: I.f32,
    inv_n: I.f32,
    eps: I.f32,
    bias: I.f32,
):
    for row in I.parallel(x.dim(0)):
        uniform = I.uniform(seed, I.member_counter(x, row))
        active = I.cast(keep > uniform, I.f32)
        value = I.cast(
            I.cast(x[row, :], I.f32) * active * scale + I.cast(residual[row, :], I.f32),
            I.bf16,
        )
        z = I.cast(value, I.f32)
        inv = I.rsqrt(I.sum(z * z, axis=0, acc=I.f32) * inv_n + eps)
        y = I.cast(z * inv * (I.cast(gain[:], I.f32) + bias), I.bf16)
        normalized[row, :] = y
        summed[row, :] = value


@I.kernel
def dropout_norm_backward(
    x: I.In[I.bf16, ("M", "N")],
    residual: I.In[I.bf16, ("M", "N")],
    gain: I.In[I.bf16, ("N",)],
    dy: I.In[I.bf16, ("M", "N")],
    carry: I.In[I.bf16, ("M", "N")],
    dx: I.Out[I.bf16, ("M", "N")],
    dr: I.Out[I.bf16, ("M", "N")],
    seed: I.u64,
    keep: I.f32,
    scale: I.f32,
    inv_n: I.f32,
    eps: I.f32,
    bias: I.f32,
):
    for row in I.parallel(x.dim(0)):
        uniform = I.uniform(seed, I.member_counter(x, row))
        active = I.cast(keep > uniform, I.f32)
        rounded = I.cast(
            I.cast(x[row, :], I.f32) * active * scale + I.cast(residual[row, :], I.f32),
            I.bf16,
        )
        z = I.cast(rounded, I.f32)
        wdy = I.cast(dy[row, :], I.f32) * (I.cast(gain[:], I.f32) + bias)
        inv = I.rsqrt(I.sum(z * z, axis=0, acc=I.f32) * inv_n + eps)
        dot = I.sum(wdy * z, axis=0, acc=I.f32) * inv_n
        dz = wdy * inv - z * (inv * inv * inv) * dot + I.cast(carry[row, :], I.f32)
        dx[row, :] = I.cast(dz * active * scale, I.bf16)
        dr[row, :] = I.cast(dz, I.bf16)


@I.program
def dropout_residual_rms_norm(x, residual, gain, dy, carry, seed, keep, scale, inv_n, eps, bias):
    y = I.output(I.bf16, (x.dim(0), x.dim(1)))
    summed = I.output(I.bf16, (x.dim(0), x.dim(1)))
    dx = I.output(I.bf16, (x.dim(0), x.dim(1)))
    dr = I.output(I.bf16, (x.dim(0), x.dim(1)))
    dropout_norm_forward(x, residual, gain, y, summed, seed, keep, scale, inv_n, eps, bias)
    dropout_norm_backward(x, residual, gain, dy, carry, dx, dr, seed, keep, scale, inv_n, eps, bias)
    return y, summed, dx, dr


@I.kernel
def rope(q: I.InOut[I.f16, ("B",32,"S",128)], k: I.InOut[I.f16, ("B",8,"S",128)],
         cosine: I.In[I.f16,(1,"S",128)], sine: I.In[I.f16,(1,"S",128)]):
    for b,g,t,j in I.parallel((q.dim(0),8,q.dim(2),64)):
        c = cosine[0,t,j]
        s = sine[0,t,j]
        for h in I.static_range(4):
            lo = q[b,g*4+h,t,j]
            hi = q[b,g*4+h,t,j+64]
            q[b,g*4+h,t,j] = lo*c-hi*s
            q[b,g*4+h,t,j+64] = hi*c+lo*s
        a = k[b,g,t,j]
        z = k[b,g,t,j+64]
        k[b,g,t,j] = a*c-z*s
        k[b,g,t,j+64] = z*c+a*s


@I.kernel
def rotary_embedding_equivalent_index(x: I.In[I.f16,("R",128)], c: I.In[I.f16,("U",64)],
                                      s: I.In[I.f16,("U",64)], y:I.Out[I.f16,("R",2,64)]):
    for u,h,j in I.parallel((x.dim(0)//32,32,64)):
        left = x[u*32+h,j]
        right = x[u*32+h,j+64]
        cosine = c[u,j]
        sine = s[u,j]
        y[u*32+h,0,j] = left*cosine-right*sine
        y[u*32+h,1,j] = right*cosine+left*sine
