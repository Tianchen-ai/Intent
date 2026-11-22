"""Author definitions for floating, integral and record-format contractions."""
import intentdsl as I


@I.kernel
def mamba_chunk_state(basis:I.In[I.bf16,("B","L","G","N")],
                      x:I.In[I.bf16,("B","L","H","P")],
                      dt:I.In[I.f32,("B","H","C","S")],
                      decay:I.In[I.f32,("B","H","C","S")],
                      state:I.Out[I.f32,("B","C","H","P","N")]):
    state[:,:,:,:,:] = I.window_contract(
        x[:,:,:,:],basis[:,:,:,:],dt[:,:,:,:],decay[:,:,:,:],
        head_group=4,acc=I.f32,
        rhs=lambda basis, step, current, last: I.cast(
            I.cast(basis,I.f32)*(I.exp(I.minimum(last-current,I.f32(0)))*step), I.bf16))


@I.kernel
def gemm(lhs: I.In[I.f16, ("M","K")], rhs: I.In[I.f16, ("K","N")],
         out: I.Out[I.f16, ("M","N")]):
    result = I.contract(lhs[:,:], rhs[:,:], axes=((1,), (0,)), acc=I.f32)
    out[:,:] = I.cast(result, I.f16)


@I.kernel
def batched_gemm(lhs: I.In[I.bf16, ("B","M","K")], rhs: I.In[I.bf16, ("B","K","N")],
                 out: I.Out[I.bf16, ("B","M","N")]):
    total = I.contract(lhs[:,:,:], rhs[:,:,:], axes=((2,), (1,)), batch=((0,), (0,)), acc=I.f32)
    out[:,:,:] = I.cast(total, I.bf16)


@I.kernel
def ragged_gemm(tokens: I.In[I.f16, ("T","K")], ends: I.In[I.i32, ("E1",)],
                banks: I.In[I.f16, ("E","K","N")], out: I.Out[I.f16, ("T","N")]):
    total = I.contract(tokens[:,:], banks[:,:,:], axes=((1,), (1,)), batch=((), (0,)),
                       offsets=ends[:], acc=I.f32)
    out[:,:] = I.cast(total, I.f16)


@I.kernel
def int8_gemm(lhs: I.In[I.i8, ("M","K")], rhs: I.In[I.i8, ("K","N")],
              bias: I.In[I.i32, ("N",)], out: I.Out[I.i32, ("M","N")]):
    partial = I.contract(lhs[:,:], rhs[:,:], axes=((1,), (0,)), acc=I.i32)
    out[:,:] = partial + bias[:]


@I.kernel
def q4_projection(records: I.In[I.u8, ("N","G",144)], x: I.In[I.f32, ("G",256)],
                  out: I.Out[I.f32, ("N",)]):
    packed = I.quantize_q8k(x[:,:])
    out[:] = I.q4k_dot(records[:,:,:], packed)


@I.kernel
def fp8_gemm(a: I.In[I.f8e4m3fn, ("M","G",32)], sa: I.In[I.u8, ("M","G")],
             b: I.In[I.f8e4m3fn, ("G",32,"N")], sb: I.In[I.u8, ("G","N")],
             out: I.Out[I.f32, ("M","N")]):
    out[:,:] = I.scaled_contract(a[:,:,:], sa[:,:], b[:,:,:], I.transpose(sb[:,:], (1,0)), acc=I.f32)


@I.kernel
def qkv_projection(x: I.In[I.f16, ("M","K")], weights: I.In[I.f16, ("P","K","N")],
                   out: I.Out[I.f16, ("P","M","N")]):
    total = I.contract(x[:,:], weights[:,:,:], axes=((1,), (1,)), batch=((), (0,)), acc=I.f32)
    out[:,:,:] = I.cast(total, I.f16)


@I.kernel
def expert_projection(x: I.In[I.bf16, ("T","K")], offsets: I.In[I.i32, ("E1",)],
                      routes: I.In[I.i32, ("R",)], weights: I.In[I.bf16, ("E","N","K")],
                      out: I.Out[I.bf16, ("T",2,"N")]):
    total = I.contract(x[:,:], weights[:,:,:], axes=((1,), (2,)), batch=((), (0,)),
                       offsets=offsets[:], routes=routes[:], acc=I.f32)
    out[:,:,:] = I.cast(total, I.bf16)
