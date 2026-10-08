# 量化计算

## 1. 闭合操作

```python
records = I.quantize(values, format=I.quant.q8_k)
result = I.quantized_dot(lhs, rhs, lhs_format=I.quant.q4_k,
                        rhs_format=I.quant.q8_k, acc_dtype=I.f32)
```

`quantize` 接受 f32 `[G,256]`，返回 u8 `[G,292]`，保留 G identity。
`quantized_dot` 接受 u8 `[G,144]` 与 `[G,292]`，返回 rank-0 f32 tensor。
双方 G 必须相等，logical 点积长度为 `256*G`；不隐含 batch、矩阵轴、初始
accumulator 或 partial-record padding。G 可以为零：量化返回 empty records，点积返回
f32 `+0`，不读取输入。

两者都是独立、pure canonical operations。格式是 closed typed schema，不是 scalar
dtype、provider 字符串或 implementation key；没有可编程 Encoding public type。
输入与结果保持 ordinary tensor semantics，读取和存储使用普通 views/buffers。
完整量化 GEMV/GEMM 由作者组合量化、点积、控制与 effects，不增加 whole-kernel op。
普通 packed INT4/INT2 的显式 decode 与 ordinary contract 路径仍然合法。

## 2. Record 格式

每条 record 对应 256 个 logical elements，little-endian bytes、低位优先 bits，
无隐式 padding。以下 byte offsets 相对当前 record。

| 格式 | 字段 | Byte span 与类型 |
|---|---|---|
| Q4_K，144 bytes | d、dmin | `[0,2)`、`[2,4)`，IEEE f16 |
| Q4_K | sc[8]、m[8] | 共用 `[4,16)`，各 unsigned 6-bit |
| Q4_K | q[256] | `[16,144)`，unsigned 4-bit |
| Q8_K，292 bytes | ds | `[0,4)`，IEEE f32 |
| Q8_K | q[256] | `[4,260)`，two's-complement i8 |
| Q8_K | bsum[16] | `[260,292)`，little-endian two's-complement i16 |

令 a 为 Q4_K bytes，r=0 代表 sc，r=1 代表 m：

```text
field_r[i]   = a[4+4*r+i] & 63                         (0 <= i < 4)
field_r[4+t] = ((a[12+t] >> (4*r)) & 15)
               | ((a[4+4*r+t] >> 6) << 4)             (0 <= t < 4)
packed       = a[16+32*(i//64)+(i%32)]                 (0 <= i < 256)
q[i]         = (packed >> (0 if i%64 < 32 else 4)) & 15
```

Q8_K 合法 records 要求 `bsum[j] = sum(q[16*j:16*j+16])`。这是数据合同，
不是由字段名推断的优化许可；runtime 不通过隐式量化、repack 或全数据扫描修复输入。
普通 carrier view 的对齐不是格式名自动赋予的事实；目标实现提出的对齐、连续性和
资源需求必须由 compiler/runtime 明确兑现。

## 3. 点积数值

每个 record b 的字段先按各自 signedness 解码并提升，不在窄输入类型截断乘积：

```text
S_b = sum(s=0..7, sc_b[s] * sum(i=0..31, wq_b[32*s+i] * xq_b[32*s+i]))
T_b = sum(s=0..7, m_b[s] * (bsum_b[2*s] + bsum_b[2*s+1]))
p_b = R32(f32(d_b) * f32(S_b))
n_b = R32(f32(dmin_b) * f32(T_b))
v_b = R32(ds_b * R32(p_b - n_b))
result = f32 add-reduction of v_b in increasing record order
```

`R32` 是一次 nearest-even f32 rounding，`f32` 是同样舍入的数值转换。
合法 records 下 `abs(S_b)<=30965760`、`abs(T_b)<=2064384`，i32 精确承载。
内部 partial 可以换组织但必须保留精确统计，不能饱和或窄乘法溢出。
记录间允许保持 logical order 的 parenthesization，不允许任意 permutation。
记录内保持上述转换与浮点边界，不隐式 FMA、FTZ、fast-math，也不替换成逐元素
f32 解量化后 ordinary dot。特殊值遵循普通浮点规范，不另设 finite-only 输入条件。

## 4. Q8_K 量化数值

每个 256-element block 独立处理：取最大/最小值，选绝对值较大的 signed extreme，
绝对值相等时选 minimum。全零 block 生成 ds=+0、q=0、bsum=0；否则：

```text
inverse = R32(-127.0 / extreme)
ds      = R32(1.0 / inverse)
q[i]    = saturate_i8(round_nearest_even(R32(values[i] * inverse)))
bsum[j] = exact_sum(q[16*j:16*j+16])
```

输入必须是有限 f32；非零 block 的 inverse、ds 与 scaled values 必须是可表示的
有限结果。域外输入非法，不默默将 NaN/overflow 变为零。饱和范围为 `[-128,127]`，
bsum 精确保存为 i16；这不是普通 float→integer 向零截断 cast。

量化结果可以是 invocation-local SSA/buffer，被多个点积共享；compiler 决定合法的
融合、物化和 lifetime，专业实现封装局部统计、转换与 packing。不能在每个消费者
重复整个共享准备，也不能给普通 f32 matmul 隐式添加有损量化。
