"""Group 7 DSA-authored operators (quantized format descriptors + depthwise conv2d) and GEMM layout variant."""

import intentdsl as I


@I.kernel
def q6k_gemv(
    records: I.In[I.u8, ("N", "G", 210)],
    x: I.In[I.f32, ("G", 256)],
    out: I.Out[I.f32, ("N",)],
):
    packed = I.quantize_q8k(x[:, :])
    out[:] = I.quant_contract(records[:, :, :], packed, format="q6_k", acc=I.f32)


@I.kernel
def mxfp4_gemm(
    a: I.In[I.u8, ("M", "G", 16)],
    sa: I.In[I.u8, ("M", "G")],
    b: I.In[I.u8, ("G", 16, "N")],
    sb: I.In[I.u8, ("G", "N")],
    out: I.Out[I.f32, ("M", "N")],
):
    out[:, :] = I.quant_contract(
        a[:, :, :],
        sa[:, :],
        b[:, :, :],
        I.transpose(sb[:, :], (1, 0)),
        format="mxfp4",
        acc=I.f32,
    )


@I.kernel
def awq_int4_gemm(
    x: I.In[I.f16, ("M", "K")],
    qweight: I.In[I.i32, ("K", "NP")],
    scales: I.In[I.f16, ("G", "N")],
    qzeros: I.In[I.i32, ("G", "NP")],
    out: I.Out[I.f16, ("M", "N")],
):
    acc = I.quant_contract(
        x[:, :], qweight[:, :], scales[:, :], qzeros[:, :], format="awq_int4", acc=I.f32
    )
    out[:, :] = I.cast(acc, I.f16)


@I.kernel
def depthwise_conv2d(
    x: I.In[I.f32, ("B", "C", "H", "W")],
    weight: I.In[I.f32, ("C", 3, 3)],
    bias: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("B", "C", "H", "W")],
):
    for b in I.parallel(out.dim(0)):
        for c in I.parallel(out.dim(1)):
            for oh in I.parallel(out.dim(2)):
                for ow in I.parallel(out.dim(3)):
                    h_lim = x.dim(2)
                    w_lim = x.dim(3)
                    acc = I.f32(0)
                    for k in I.fold(9):
                        kh = k // 3
                        kw = k % 3
                        ih1 = oh + kh
                        iw1 = ow + kw
                        in_h = 1 if ih1 >= 1 else 0
                        in_h2 = in_h if ih1 <= h_lim else 0
                        in_w = in_h2 if iw1 >= 1 else 0
                        valid = in_w if iw1 <= w_lim else 0
                        sample = (
                            x[b, c, ih1 - 1, iw1 - 1]
                            if valid == 1
                            else I.f32(0)
                        )
                        acc = acc + sample * weight[c, kh, kw]
                    out[b, c, oh, ow] = acc + bias[c]


@I.kernel
def gemm_nt(
    lhs: I.In[I.f16, ("M", "K")],
    rhs: I.In[I.f16, ("N", "K")],
    out: I.Out[I.f16, ("M", "N")],
):
    total = I.contract(lhs[:, :], rhs[:, :], axes=((1,), (1,)), acc=I.f32)
    out[:, :] = I.cast(total, I.f16)
