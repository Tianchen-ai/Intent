# 作者语言与 KIR

**V2-1 设计提案。** 下列代码是作者表面的规范样例，可供评审和后续实现对照。V2-2 优先实现稳定 softmax 和 weighted RMSNorm 所需的子集。文中的 `import intentdsl as I` 是建议的公共导入名。`I.jit(definition, backend=...)` 创建本机可复用调用对象。

## 从作者选择到语义

| 作者决定／表面 | 前端写入的 KIR | 消费者、检查及与 V1 的关系 |
|---|---|---|
| `@I.kernel` 声明一次可见调用；`@I.fn` 为有类型的纯 helper；`@I.program` 描述显式多次调用 | 可引用的 kernel/fn 符号与 block 参数；program 的调用、依赖、workspace 和输出由主机对象持有 | 捕获保留符号、文件行及闭包常量；kernel 不递归、不隐式 launch；fn 可特化/内联但不改变类型、顺序。沿用 V1 前两种语义，新增 program 的声明式表面 |
| `I.In/Out/InOut[T, shape]`；标量参数；`I.Constexpr[T]` | 可读、写或读写的 ranked view，标量 SSA 或编译常量 | `("R","C")` 在同一签名内建立维度相等关系，整数维度为约束。独立 view 默认可别名；Out 表达写入责任，不授权假设 noalias。沿用 |
| shape/stride、view、逻辑成员 | view 含 base、offset、shape、元素 stride；domain/slice/reindex 含来源轴与成员关系 | `x.dim(i)` 读取运行时维度；`x[r, :]` 读出逻辑行的值，赋值左侧为写。切片保留绝对成员坐标与 step。省略 stride 表示绑定时读取，不自动假设连续；可选显式 stride 等式和 noalias 是调用前置条件。沿用语义，简化整轴写法 |
| `I.parallel(n)`、`I.ordered(n)`、if/else、循环 carry | 前者是不排序的不相交工作；后者为固定迭代顺序的 region、迭代参数、yield | `n` 表示 [0,n)；也接受显式 domain。parallel 内跨迭代冲突须由唯一写、合并或 atomic 解释。ordered 保留循环携带的真实依赖和 effects，不预设并行组合律。沿用，表面显式区分 |
| 普通读写、`gather(view, indices, valid, fill)`、scatter_unique、scatter_reduce、atomic | 访问关系、active predicate、fill/写入效果；间接关系保留索引值依赖 | inactive 读不访问地址，返回显式 fill；inactive 写不生效。unique 要求有效域无碰撞；scatter_reduce 保留重复贡献。atomic 保留地址、op、返回旧值及内存次序；CAS 返回 old/success。同步的物理 scope 由后端兑现。沿用 |
| `reduce/sum`、`scan`、`reduce_by_key` | 轴、typed combine、identity、acc/result 类型；scan 含方向及 inclusive；按键归约含键域/有效性 | reduce 允许结合交换，scan 允许同一逻辑前缀内保序重括号化；严格左折叠用 ordered。按键归约输出 SSA：每个有效键必须在范围内，空 bin 有 identity。它将 V1 histogram 的计数语义落到可复用的操作上 |
| `contract`（显式 A/B 轴对、批维、acc dtype；提供 matmul 简写） | contract 保留自由轴、对约轴、batch 对齐、累加类型、可选初值 | 用户表达收缩和 epilogue 的先后；向量核、矩阵核、packing 及物理布局不进入 KIR。无对约成员返回初值。沿用 |
| `region_fold`、`region_scan` | 源轴、summary/identity、summarize/combine 纯 region；scan 另含 initial、apply 和 emit | 作者选择可分段算法：summarize(A++B)=combine(summarize(A),summarize(B))，identity 双侧中立；scan 的相邻段等价于传递 incoming state 的连续 emit。允许保序分段，不能排列段或观察物理段号。沿用 |
| 标量、tensor、tuple/record，局部状态、计数器随机数 | 逐字段类型，reshape/broadcast 等成员变换；词法生命周期 buffer；由 seed 和逻辑坐标产生 RNG | tuple 可作 helper 结果、Welford 或 recurrence 的 carry，字段可不同形状。buffer 需在 active 读前已初始化；跨 kernel 状态属 program。随机序列不依赖物理线程。沿用 |
| 量化、格式转换、scaled/sparse contraction | 闭合格式、carrier、每组 scale/元数据关系、解码/舍入点、acc 类型；原始整数位运算也可表达格式 | 保留 Q4_K/Q8_K record 组关系与独立舍入点，E4M3FN 与 E8M0 scale 不折为普通整型，INT8 累加到 i32。解码分块与供数复用留给子节点。沿用 |
| `workspace/output`、有序 kernel calls、主机输出 tuple | 根 view ABI 与主机 program 的 typed 调用签名相接，完整调用序列保留符号、形状表达式、初始化与依赖 | workspace 每次绑定有所有者，完成前不可释放；异步出口在同一 stream/queue 上维持作者顺序。Bench 消费真实 prepared sequence。新表面替代逐程序访问私有子对象 |

类型包括 bool、带符号的 64 位逻辑 index、有/无符号 8/16/32/64 位整数、f16/bf16/f32/f64、具有确定编码的 FP8。张量有逻辑 rank 与 extent identity；物理 lane 宽度与字节填充不是元素类型。标量 literal 由期望类型定型。沿用 V1：普通 binary 两个已定型操作数的 dtype 不同时须显式 cast，比较返回 bool；结构化运算使用指定的 acc dtype 或操作规定的固定提升。支持 view offset 与非连续读，负 stride 仅在完整地址范围可绑定时使用。无法支持的 specialization 应由相应出口诊断。

## 五个作者程序

以下样例共享上述定义。只使用参数、有限常量、纯 helper 和可捕获的控制。单 kernel 可直接交给 jit；program 体中的 kernel 名表示在同一显式 backend 上按序调用。

### 1．Stable softmax

```python
import intentdsl as I

@I.kernel
def softmax(x: I.In[I.f16, ("R", "C")], y: I.Out[I.f16, ("R", "C")]):
    for r in I.parallel(x.dim(0)):
        row = I.cast(x[r, :], I.f32)
        peak = I.reduce(row, axis=0, combine=I.max_num, init=I.neg_inf(I.f32))
        mass = I.exp(row - peak)
        y[r, :] = I.cast(mass / I.sum(mass, axis=0, acc=I.f32), I.f16)
```

作者选择逐行、先减最大值、f32 中归一化、最后一次窄化。max_num 忽略单侧 NaN；有 NaN 的 active 行仍会由 exp/sum 传播。全负无穷行按浮点减法产生 NaN。零列行没有输出成员；禁止以填充 lane 的值改变归约或制造访存。

### 2．Weighted RMSNorm

```python
import intentdsl as I

@I.kernel
def rms(x: I.In[I.f32, ("R", "C")], gain: I.In[I.f32, ("C",)],
        dst: I.Out[I.f32, ("R", "C")], inv_c: I.f32, eps: I.f32):
    for r in I.parallel(x.dim(0)):
        u = x[r, :]
        moment = I.sum(u * u, axis=0, acc=I.f32) * inv_c
        dst[r, :] = (u * I.rsqrt(moment + eps)) * gain[:]
```

保留 V1 的运行时 inv_c 与 eps；调用方可传 1/C，编译器不能自行用 1/C 代换任意实参。与 softmax 共享行覆盖、尾部、归约和行内值复用，gain 是沿列的额外读。

### 3．GEMM 与 epilogue

```python
import intentdsl as I

@I.fn
def biased_relu(s, bias):
    return I.maximum(s + bias, I.f32(0))

@I.kernel
def projection(a: I.In[I.f16, ("M", "K")], b: I.In[I.f16, ("K", "N")],
               bias: I.In[I.f32, ("N",)], out: I.Out[I.f16, ("M", "N")]):
    sums = I.contract(a[:, :], b[:, :], axes=((1,), (0,)), acc=I.f32)
    out[:, :] = I.cast(biased_relu(sums, bias[:]), I.f16)
```

未列作对约或 batch 的轴按左再右的顺序成为输出自由轴，bias 按末轴广播。零 K 的和为 f32 零，仍需执行 bias 与激活；maximum 传播 NaN。只有完整累加值进入 epilogue。

### 4．严格有序的前缀累加

```python
import intentdsl as I

@I.kernel
def ordered_prefix(src: I.In[I.f32, ("B", "T")], out: I.Out[I.f32, ("B", "T")]):
    for b in I.parallel(src.dim(0)):
        carry = I.f32(0)
        for t in I.ordered(src.dim(1)):
            carry = carry + src[b, t]
            out[b, t] = carry
```

这是一个严格左折叠示例。选择 `I.scan(..., inclusive=True)` 则保留每个 prefix 的成员和相对顺序、允许合法重括号化，两份程序的浮点许可不同。T=0 时没有读写。

### 5．两阶段 histogram

```python
import intentdsl as I

@I.kernel
def count_parts(x: I.In[I.f32, ("N",)], tmp: I.Out[I.i32, ("P", "B")],
                span: I.Constexpr[I.index]):
    I.require(span > 0)
    I.require(tmp.dim(1) > 0)
    I.require(tmp.dim(0) == I.ceildiv(x.dim(0), span))
    for p in I.parallel(tmp.dim(0)):
        lo = p * span
        keys = I.cast(x[lo:I.min_index(lo + span, x.dim(0))], I.i32)
        keep = (keys >= 0) & (I.cast(keys, I.index) < tmp.dim(1))
        tmp[p, :] = I.reduce_by_key(keys, I.i32(1), size=tmp.dim(1),
                                    valid=keep, combine=I.add, init=I.i32(0))

@I.kernel
def merge_parts(tmp: I.In[I.i32, ("P", "B")], out: I.Out[I.i32, ("B",)]):
    out[:] = I.sum(tmp[:, :], axis=0, acc=I.i32)

@I.program
def histogram(x, bins: I.Constexpr[I.index] = 256,
              span: I.Constexpr[I.index] = 8192):
    I.require((span > 0) & (bins > 0))
    tmp = I.workspace(I.i32, (I.ceildiv(x.dim(0), span), bins))
    out = I.output(I.i32, (bins,))
    count_parts(x, tmp, span)
    merge_parts(tmp, out)
    return out
```

保持 V1 程序的 f32 样本、先向零截断到 i32、再判 bin 范围；float→integer 的合法输入须有限且截断结果可表示。按键归约将标量 1 广播到所有有效键，i32 计数按整数宽度回绕。P=0 时 merge 写出 B 个零，不可仅跳过两阶段就返回未初始化的 out。span 决定作者可见的 partials 形状与第二次调用输入；内部 tile 不得改变它。`I.require` 只用于可由 shape、运行时标量或 constexpr 判定的用户前置条件，违反时在相关执行前诊断。

## 根 IR 合同

| 类型／操作簇（拟议名称） | 生产者 → 消费者 | 不变量 |
|---|---|---|
| view、tensor、index、domain/relation、record、buffer | 前端 → KIR 检查器与执行模型 | rank/type 可判定；符号 identity、来源轴、base/stride/offset 和参数方向各自可追踪；record 保持字段顺序 |
| kernel、call/yield、parallel、ordered、if | 捕获与 helper 特化 → 依赖/控制分析 | 支配关系成立；carry/result 类型及形状一致；同一 kernel 内 control 不代表额外 launch |
| slice、index_map、read/write、gather/scatter、atomic | 访问构造 → 有效域、alias/effect 分析 | 成员投影与谓词分别表达；只对 active 成员要求地址合法，写入的唯一性／冲突语义明确 |
| 标量算术、cast/bitcast、广播、reshape、compare、select | 统一类型推导 → canonicalizer | 转换、求值精度、值与 mask 的形状关系显式；select 的值选择不能代替 masked load 的访问抑制 |
| reduce、scan、reduce_by_key、contract、region_fold/scan | 结构化 builder → execution lowering | 包含轴、identity、combine、dtype 和空域定义；ordered 与可重结合 op 的许可不同 |
| quantize、quantized_dot、scaled/sparse contract、RNG | 格式或算法 intrinsic → typed pattern | 格式和元数据关系可验证；RNG 由显式 seed/counter 定址；保留独立舍入和状态顺序 |

**imported → canonical。** 入口校验签名和基本类型、有限的纯 helper 调用关系、SSA/region 合法性、源位置。规范化将整轴切片/索引简写变为同一成员 relation，将 Python 循环变量变为 region arguments，补齐隐式合法广播与 promotion，规范化可证明等价的控制和无副作用表达式。canonical verifier 要求无待解释的 Python 语法或设备属性，所有动态等式均有来源或显式 guard。它给出可证明的 alias/effect 信息，未知须保守建模。两种作者写法若仍有不同的舍入边界、alias 或 effects，保留差异。

**数值许可。** 默认 source。cast 明示目的格式；f16/bf16 普通窄化 RN-even。max 与 max_num 分开；argmax 的 NaN/tie 和空域合同属于操作本身。普通 source 浮点表达式仅沿用 V1 已允许的乘加融合，不混入任意重结合、TF32 或隐式 FTZ。由操作选择 reduce/scan/region 的代数自由度；quant record 内指定独立 R32 的步骤不得跨越。近似指令许可附着在对应操作，调用级在线归约许可附着在 kernel 数值属性，记录适用对象和有限输入等前提。允许重写 ≠ 选择重写。KIR 提供语义 pattern，block/cpu 的入口 pass 根据实际执行成本调用；它必须在仍有 KIR 结构时完成改写与 verifier 检查，不跨越作者的 kernel 边界。

具体保留 V1 的 `source` 与 `relaxed_normalization` 区别。后一模式仅许可合格的 max/mass/weighted-moment 按连续段重缩放；要求有效 score 及实际参与加权收缩的 value 有限。它不能泛化为所有 softmax 或其他浮点运算的 fast-math 开关。设备无关的必要前提由用户显式选择模式承担，前端不额外扫描 tensor。相同输入下用哪一种合法实现由子节点决定。

**Region 的验证接口。** fold 的 `summarize(slice, logical_coords) → S`、`combine(S, S) → S` 和 identity 同 schema；scan 另外有 `apply(S, incoming) → state` 与 `emit(slice, incoming) → prefix_values`。对相邻 A/B，B 的 incoming 为 apply(summarize(A), incoming)，emit 的结果沿原轴拼合。空 fold 返回 identity，空 scan 返回空输出与初态。verifier 检查纯度、值/轴和输出关系，组合律是作者选择原语时的前提。带最大值的浮点 summary 若用负无穷作为初值，需用有效位或等价表示避免在空段计算负无穷减负无穷。

## 对 V1 30 个登记程序的语言覆盖设计

此表按 V1 的程序标识核对语言需要，不是 V2 运行结果。weighted RMSNorm 是额外的首路线代表；30 项中登记的是 layer_norm。

| V1 登记项 | 必须能表达的结构／不能丢的事实 |
|---|---|
| relu、swiglu、transpose、dropout、rope、adamw | 元素映射、轴置换、按坐标随机数、多输入/多输出、成对原地更新与优化器状态 |
| softmax、layer_norm、batch_norm、group_norm_backward | 多归约、权重广播、Welford record、可变统计量、梯度与显式多次调用 |
| cumsum、causal_linear_attention、gated_delta、mamba_chunk_state、causal_convolution | prefix、矩阵状态、ordered recurrence、contract 与沿源轴的邻域边界 |
| embedding、csr_spmv、jagged_mean、nonzero、moe_alignment、histogram | 间接地址、分段/空段、索引生成、计数/前缀/散射、确定冲突语义及 workspace |
| gemm、batched_gemm、ragged_gemm、int8_gemm、q4_projection、fp8_gemm | 批维、变化的分组 extent、不同累加 dtype、压缩 record 与分组 scales |
| attention、paged_decode、cholesky | 因果及分页掩码、softmax 状态与部分结果合并、三角域内有序原地状态 |

**实现次序。** 首路线只要求二维动态 view、连续/受支持的非连续访问、行并行、f16/f32 转换、逐元素算术、行 max/sum、显式输出和标量参数。其余构造的类型与顺序在设计中有位置；实现时以各自的真实作者程序接通，未接通的合法构造报告 lowering 缺失。
