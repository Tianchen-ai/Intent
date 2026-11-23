"""Causal attention gradients from saved natural logarithms and forward output."""
import intentdsl as I


@I.kernel
def output_inner_product(y: I.In[I.f16, ("B", "H", "T", "D")],
                         dy: I.In[I.f16, ("B", "H", "T", "D")],
                         delta: I.Out[I.f32, ("B", "H", "T")]):
    for b in I.parallel(y.dim(0)):
        for h in I.parallel(y.dim(1)):
            for t in I.parallel(y.dim(2)):
                delta[b,h,t] = I.sum(I.cast(y[b,h,t,:], I.f32) * I.cast(dy[b,h,t,:], I.f32))


@I.kernel
def key_value_grad(q: I.In[I.f16, ("B","H","Q","D")], k: I.In[I.f16, ("B","H","K","D")],
                   v: I.In[I.f16, ("B","H","K","D")], dy: I.In[I.f16, ("B","H","Q","D")],
                   lse: I.In[I.f32, ("B","H","Q")], delta: I.In[I.f32, ("B","H","Q")],
                   dk: I.Out[I.f16, ("B","H","K","D")], dv: I.Out[I.f16, ("B","H","K","D")], scale: I.f32):
    for b in I.parallel(k.dim(0)):
        for h in I.parallel(k.dim(1)):
            for row in I.parallel(k.dim(2)):
                key = I.cast(k[b,h,row,:], I.f32)
                val = I.cast(v[b,h,row,:], I.f32)
                grad_k = I.zeros(k.dim(3), I.f32)
                grad_v = I.zeros(v.dim(3), I.f32)
                factor = scale * I.f32(1.4426950408889634)
                for t in I.fold(row,q.dim(2)):
                    query = I.cast(q[b,h,t,:], I.f32)
                    grad = I.cast(dy[b,h,t,:], I.f32)
                    p = I.exp2(I.sum(key * query) * factor - lse[b,h,t] * I.f32(1.4426950408889634))
                    ds = p * (I.sum(val * grad) - delta[b,h,t])
                    grad_k = grad_k + query * I.cast(I.cast(ds,I.f16),I.f32)
                    grad_v = grad_v + grad * I.cast(I.cast(p,I.f16),I.f32)
                dk[b,h,row,:] = I.cast(grad_k * scale,I.f16)
                dv[b,h,row,:] = I.cast(grad_v,I.f16)


@I.kernel
def query_grad(q: I.In[I.f16, ("B","H","Q","D")], k: I.In[I.f16, ("B","H","K","D")],
               v: I.In[I.f16, ("B","H","K","D")], dy: I.In[I.f16, ("B","H","Q","D")],
               lse: I.In[I.f32, ("B","H","Q")], delta: I.In[I.f32, ("B","H","Q")],
               dq: I.Out[I.f16, ("B","H","Q","D")], scale: I.f32):
    for b in I.parallel(q.dim(0)):
        for h in I.parallel(q.dim(1)):
            for row in I.parallel(q.dim(2)):
                query = I.cast(q[b,h,row,:],I.f32)
                grad = I.cast(dy[b,h,row,:],I.f32)
                normalizer = lse[b,h,row] * I.f32(1.4426950408889634)
                adjustment = delta[b,h,row]
                result = I.zeros(q.dim(3),I.f32)
                end = row+1 if row<k.dim(2) else k.dim(2)
                for t in I.fold(end):
                    key = I.cast(k[b,h,t,:],I.f32)
                    val = I.cast(v[b,h,t,:],I.f32)
                    p = I.exp2(I.sum(query * key) * (scale * I.f32(1.4426950408889634)) - normalizer)
                    ds = p * (I.sum(grad * val) - adjustment)
                    result = result + key * I.cast(I.cast(ds,I.f16),I.f32)
                dq[b,h,row,:] = I.cast(result * scale,I.f16)


@I.program
def flash_attention_backward(q, k, v, y, dy, lse, scale):
    delta = I.workspace(I.f32,(q.dim(0),q.dim(1),q.dim(2)))
    dq = I.output(I.f16,(q.dim(0),q.dim(1),q.dim(2),q.dim(3)))
    dk = I.output(I.f16,(k.dim(0),k.dim(1),k.dim(2),k.dim(3)))
    dv = I.output(I.f16,(v.dim(0),v.dim(1),v.dim(2),v.dim(3)))
    output_inner_product(y,dy,delta)
    key_value_grad(q,k,v,dy,lse,delta,dk,dv,scale)
    query_grad(q,k,v,dy,lse,delta,dq,scale)
    return dq,dk,dv
