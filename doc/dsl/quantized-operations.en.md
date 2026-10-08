# Quantized operations

## 1. Closed operations

```python
records = I.quantize(values, format=I.quant.q8_k)
result = I.quantized_dot(lhs, rhs, lhs_format=I.quant.q4_k,
                        rhs_format=I.quant.q8_k, acc_dtype=I.f32)
```

Quantize accepts f32 `[G,256]` and returns u8 `[G,292]` retaining G identity. Quantized_dot accepts u8 `[G,144]` and `[G,292]`, returning rank-zero f32 tensor. G agrees; logical dot length is256*G, with no implicit batch/matrix axes/initial accumulation/partial-record padding. G=0 returns empty records or f32 +0 without reads.

Both are independent pure canonical operations. Formats are closed typed schemas, not scalar dtypes/provider strings/implementation keys; no programmable public Encoding exists. Ordinary tensor/view/buffer semantics carry input/output. Authors compose full quantized GEMV/GEMM without whole-kernel operations. Explicit INT4/INT2 decoding + ordinary contract remains legal.

## 2. Record formats

Each record covers256 logical elements with little-endian bytes, low bits first and no implicit padding. Offsets are record-relative.

| Format | Fields | Byte spans and type |
|---|---|---|
| Q4_K,144 bytes | d,dmin | `[0,2)`,`[2,4)`, IEEE f16 |
| Q4_K | sc[8],m[8] | shared `[4,16)`, unsigned6-bit each |
| Q4_K | q[256] | `[16,144)`, unsigned4-bit |
| Q8_K,292 bytes | ds | `[0,4)`, IEEE f32 |
| Q8_K | q[256] | `[4,260)`, two's-complement i8 |
| Q8_K | bsum[16] | `[260,292)`, little-endian two's-complement i16 |

For Q4_K bytes a, r=0 for sc and1 for m:

```text
field_r[i]   = a[4+4*r+i] & 63                         (0 <= i < 4)
field_r[4+t] = ((a[12+t] >> (4*r)) & 15)
               | ((a[4+4*r+t] >> 6) << 4)             (0 <= t < 4)
packed       = a[16+32*(i//64)+(i%32)]                 (0 <= i < 256)
q[i]         = (packed >> (0 if i%64 < 32 else 4)) & 15
```

Legal Q8_K records require bsum[j]=sum(q[16*j:16*j+16]). This is a data contract, not inferred permission from field names. Runtime does not implicitly quantize/repack/scan to repair inputs. Carrier alignment does not follow from a format's name; compiler/runtime must establish actual alignment/contiguity/resource requirements.

## 3. Dot numerics

Decode each record's fields by signedness and widen before products; never truncate in narrow input types:

```text
S_b = sum(s=0..7, sc_b[s] * sum(i=0..31, wq_b[32*s+i] * xq_b[32*s+i]))
T_b = sum(s=0..7, m_b[s] * (bsum_b[2*s] + bsum_b[2*s+1]))
p_b = R32(f32(d_b) * f32(S_b))
n_b = R32(f32(dmin_b) * f32(T_b))
v_b = R32(ds_b * R32(p_b - n_b))
result = f32 add-reduction of v_b in increasing record order
```

R32 is one nearest-even f32 round; f32 conversion uses it too. Legal records satisfy |S_b|≤30965760 and |T_b|≤2064384, exactly representable in i32. Internal partial organization may vary but exact statistics cannot saturate/overflow through narrow products.

Records may use order-preserving parenthesization, not permutations. Preserve within-record conversions/float boundaries without implicit FMA/FTZ/fast math or replacement by elementwise f32 dequantization + ordinary dot. Ordinary special-value rules apply with no added finite-only contract.

## 4. Q8_K quantization numerics

Process256-element blocks independently. Pick the signed max/min extreme with larger absolute value, choosing minimum on ties. All-zero blocks produce ds=+0,q=0,bsum=0. Otherwise:

```text
inverse = R32(-127.0 / extreme)
ds      = R32(1.0 / inverse)
q[i]    = saturate_i8(round_nearest_even(R32(values[i] * inverse)))
bsum[j] = exact_sum(q[16*j:16*j+16])
```

Inputs must be finite f32; inverse/ds/scaled values of nonzero blocks must remain finite and representable. Out-of-domain inputs are invalid, not converted from NaN/overflow to zero. Saturation is [-128,127]; exact bsum is i16. This is not truncation-toward-zero cast.

Invocation-local SSA/buffers may share quantization across dots. Compiler forms legal fusion/materialization/lifetime; expert implementations encapsulate local statistics/conversion/packing. Do not repeat shared preparation per consumer or silently add lossy quantization to ordinary f32 matmul.
