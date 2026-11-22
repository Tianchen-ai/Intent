# 作者语言与 KIR

**V2-1b 设计合同。** 下列代码是作者表面的规范样例，可供评审和后续实现对照。V2-2 优先实现稳定 softmax 和 weighted RMSNorm 所需的子集。文中的 `import intentdsl as I` 是建议的公共导入名。`I.jit(definition, backend=...)` 创建本机可复用调用对象。

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
| `workspace/output`、显式 fill、有序 kernel calls、主机输出 tuple | 根 view ABI 与主机 program 的 typed 调用签名相接，完整调用序列保留符号、形状表达式、初始化与依赖 | workspace 每次绑定有所有者，完成前不可释放；异步出口在同一 stream/queue 上维持作者顺序。Bench 消费真实 prepared sequence。新表面替代逐程序访问私有子对象 |

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

### 归一化 program 的所有权与部分和

训练 BatchNorm 以通道为 owner。每次调用返回归一化输出、保存的均值与逆标准差、
更新后的 running mean 与 running variance。Welford 摘要包含一个整数计数和两个
FP32 字段，投影到 owner 的统计量每通道发布一次。

GroupNorm 反向的第一个调用按 batch 和 group 归约空间成员，第二个调用独立归约
仿射参数梯度。具名成员域保留源轴的仿射映射。结果投影到 owner 时，verifier
要求地址与全部归约轴无关；绑定器根据真实 stride 检查输出完整覆盖与无碰撞。

已登记的 LayerNorm 反向输入有 4096 行。作者指定 128 个部分和桶，每个桶按行号
升序接收 32 个贡献。两份 BF16 workspace 在每次 program 调用中先独立清零；
第一项调用同时写入输入梯度并逐次舍入部分和，第二项调用按特征以 FP32 合并。
目前这个作者 kernel 将 32 次更新写成静态循环；其他行数需要相应的作者分组。
完整 program 的观察保留两次 fill 和两次调用的顺序，按成员比较时由同一份
绑定提供实参、输出与 workspace。

| 类型／操作簇（拟议名称） | 生产者 → 消费者 | 不变量 |
|---|---|---|
| view、tensor、index、domain/relation、record、buffer | 前端 → KIR 检查器与执行模型 | rank/type 可判定；符号 identity、来源轴、base/stride/offset 和参数方向各自可追踪；record 保持字段顺序 |
| kernel、call/yield、parallel、ordered、if | 捕获与 helper 特化 → 依赖/控制分析 | 支配关系成立；carry/result 类型及形状一致；同一 kernel 内 control 不代表额外 launch |
| slice、index_map、read/write、gather/scatter、atomic | 访问构造 → 有效域、alias/effect 分析 | 成员投影与谓词分别表达；只对 active 成员要求地址合法，写入的唯一性／冲突语义明确 |
| 标量算术、cast/bitcast、广播、reshape、compare、select | 统一类型推导 → canonicalizer | 转换、求值精度、值与 mask 的形状关系显式；select 的值选择不能代替 masked load 的访问抑制 |
| reduce、scan、reduce_by_key、contract、region_fold/scan | 结构化 builder → execution lowering | 包含轴、identity、combine、dtype 和空域定义；ordered 与可重结合 op 的许可不同 |
| quantize、quantized_dot、scaled/sparse contract、RNG | 格式或算法 intrinsic → typed pattern | 格式和元数据关系可验证；RNG 由显式 seed/counter 定址；保留独立舍入和状态顺序 |

**imported → canonical。** 入口校验签名和基本类型、有限的纯 helper 调用关系、SSA/region 合法性、源位置。规范化将整轴切片/索引简写变为同一成员 relation，将 Python 循环变量变为 region arguments，补齐隐式合法广播与 promotion，规范化可证明等价的控制和无副作用表达式。canonical verifier 要求无待解释的 Python 语法或设备属性，所有动态等式均有来源或显式 guard。它给出可证明的 alias/effect 信息，未知须保守建模。两种作者写法若仍有不同的舍入边界、alias 或 effects，保留差异。

**数值许可。** 默认 source。cast 明示目的格式；f16/bf16 普通窄化 RN-even。max 与 max_num 分开；argmax 的 NaN/tie 和空域合同属于操作本身。普通 source 浮点表达式仅沿用 V1 已允许的乘加融合，不混入任意重结合、TF32 或隐式 FTZ。由操作选择 reduce/scan/region 的代数自由度；quant record 内指定独立 R32 的步骤不得跨越。近似指令许可附着在对应操作，调用级在线归约许可附着在 kernel 数值属性，记录适用对象和有限输入等前提。允许重写 ≠ 选择重写。KIR 单元提供唯一的匹配、数值前提检查和替换构造。block/cpu 的入口在降低 KIR 前据实际成本选择是否调用同一接口；匹配或许可不满足就保留源表达式。改写结果仍须通过 KIR verifier，且保持作者的 kernel 边界。

具体保留 V1 的 `source` 与 `relaxed_normalization` 区别。后一模式仅许可合格的 max/mass/weighted-moment 按连续段重缩放；要求有效 score 及实际参与加权收缩的 value 有限。它不能泛化为所有 softmax 或其他浮点运算的 fast-math 开关。设备无关的必要前提由用户显式选择模式承担，前端不额外扫描 tensor。相同输入下用哪一种合法实现由子节点决定。

## 访问、成员与 effects 的验证合同

下列规则沿用 V1 的数据与数值语义，作为 V2 `kir@canonical` 以及绑定器的共同依据。IR verifier 检查类型、关系来源和 effects；绑定时可读取 shape、stride、allocation 或显式 guard。依赖实际 tensor 内容的条件注明为数据前提。仅有合法的类型或一次恰好无碰撞的运行，都不能给优化增加证明。

### 切片与 ragged 成员

- `v[4:12:2]` 的源成员坐标依次为 4、6、8、10；读出的值 `t` 形状为 [4]，`t[1]` 对应 `v[6]`。用 `v.offset + 6*v.stride` 得到元素地址偏移。verifier 将 source relation 与结果 extent 分开，后续索引按结果 extent 检查，不能再次叠加起点或步长。
- G 个组的 offsets 是有至少 G+1 个元素的一维整数 view，访问 g 要求 `0 ≤ g < G`。把相邻两个值无损转成 index 得到 lo、hi；数据须满足 `0 ≤ lo ≤ hi ≤ 成员源长度`。组内 t 的逻辑域为 [0,hi−lo)，其成员源坐标为 lo+t。若给出间接表，则成员源是该表，以其在 lo+t 处的值访问最终 view；仅对活跃成员要求映射值落在目标轴内。KIR 保留 offsets 读取、slice、可选 gather 之间的 SSA 关系。
- lo=hi 时无成员，不访问索引表或 payload，不产生散射。归约返回作者声明的 identity；mean 等后续运算按作者的空段表达式取值。直接散射可由相邻前缀区间不重叠、组内注入映射、目标视图地址无碰撞共同证明唯一性。间接映射即使有界也可能重复；要么有可追踪的唯一来源（如无回绕的 atomic-add 旧值分配的不交槽位，另验容量），要么由作者满足 `scatter_unique` 的活跃地址唯一前提。普通散射不能靠 offsets 自动升级为唯一散射。有意重复的更新须用合并或 atomic。

### 作者 fill、物理尾部与按键归约

在逻辑域内，masked read 对假 predicate 返回作者指定的、类型和形状可广播的 fill；真 predicate 才计算需要解引用的地址。比如读出的一个逻辑元素是 fill=5，同一 SSA 值送给 sum 和 max 时，两者均须看到 5。在逻辑域之外，尾片段补出的 lane 对两者均无贡献。实现可屏蔽该 lane，或在 sum 消费处用 0、在 max 消费处用相应的负无穷 identity。不得把任一消费者的 identity 写回共享读值。写和原子更新在无效地址上均无 effect。

`reduce_by_key(keys, values, size=B, valid=mask, combine, init)` 产生长度 B 的结果。verifier 要求 B 非负、整数或 index 键、布尔 mask，values 与 mask 可广播到键域，init 是累加类型的标量 identity，combine 是闭合的纯二元操作。在 histogram 中，作者显式以键范围计算 mask。只对 mask 为真的位置要求 `0 ≤ key < B`；活跃越界违反前提，mask 为假时即使键越界也不寻址、不贡献。一个 bin 中每个有效位置恰贡献一次，包括重复键。选用此原语即允许对该 bin 按其 combine 数值合同结合、交换；不得省去重复值或引入额外精度许可。没有有效贡献的 bin 取 init，空输入亦然。需要按输入次序更新的算法使用有序循环。

### 原子与别名

| 操作 | 可用 order | 结果与 verifier 条件 |
|---|---|---|
| load | relaxed、acquire | 返回地址元素类型的值；只需读权限 |
| store | relaxed、release | 无结果；要求写权限和同类型写值 |
| RMW | relaxed、acquire、release、acq_rel | 要求读写权限、合法的 kind 与元素类型；更新不可分割，返回更新前的同类型值 |
| CAS | 成功 order 同 RMW | expected/desired 与元素同类型，要求读写权限；返回旧值与 bool success。成功为 relaxed/acquire/release/acq_rel 时，失败分别为 relaxed/acquire/relaxed/acquire。失败只读，不发布写入 |

order 是操作属性，拒绝表外组合；一个逻辑原子地址上存在单一修改顺序。匹配的 acquire/release 可以建立 happens-before。实际同步范围须覆盖该地址的逻辑参与者，不能据此假定不同地址有全局顺序或所有实例同时驻留。

- 绑定器保留 allocation 身份及每个 view 的 dtype、元素 offset、shape、带符号 stride。相同 alias group 必须同 allocation；不同 group 或不同参数名都不足以证明互斥。显式 noalias 的 allocation 须与其他实参不重叠，且不得同时声明 alias group。判断具体读写重叠时使用实际字节地址集合；只知包围区间相交时须保守。
- 零 stride 可用于广播读。若两个活跃并行成员经零 stride 或不同 `InOut` 子视图写到相同地址，普通唯一写非法，即使写入值相同。严格有序的部分重叠读写保持原先后，合并或原子遵守各自合同。后端优化须保留读写依赖；无法处理此绑定时给出出口诊断。
- 首次调优按底层 allocation 隔离状态。恢复单位覆盖所有可能改写或通过别名观察的初始字节；保留所有活跃视图相对同一底层存储的 offset 和 stride，包含只读实参所见的重叠区域。每一试跑前还原初态，保持别名关系，按原序列执行本次试跑范围内的 fill 和调用；等该行全部异步访问完成才能复用试跑空间。丢弃试跑的输出，最终只在用户实参上提交一次选中行。无法隔离外部 effects 时按单行执行或诊断。例：`a=buf[0:8]` 和 `b=buf[4:12]` 的重叠四项在试跑中也须是同四项。

## Region 的 source 与 captures

`region_fold/region_scan` 分别有 source 元组与 captures 元组。所有 source 在所选轴有同一 extent，随同一组连续、保序切点切分；需绝对坐标时将坐标作为 source 成员传入。captures 是支配该 op 的值：运行时标量或 tensor 按位置作为显式 SSA operands，不参与切片；constexpr 保留为特化事实。每个分段使用相同的 captures 类型、形状和值。回调不得读取隐式可变状态、段号或调度选择，不得产生外部 effects。

| 签名（C 表示同一组 captures） | verifier 检查及作者保证 |
|---|---|
| summarize(sources 切片…, C…) → S；combine(S,S) → S | 参数数量、顺序和类型匹配，identity 同 S；相邻段的摘要可按源顺序组合，identity 双侧中立。空 fold 返回 identity |
| apply(S,state) → state；emit(sources 切片…,incoming,C…) → 结果切片 | state schema 稳定，identity 对状态无作用，先后 apply 与依序 combine 后 apply 等价；右段的 incoming 为左段摘要作用后的状态。拼合 emit 的结果必须保留原轴成员，终态等于完整摘要作用于初态。空 scan 返回空输出与初态 |

代数等式是作者使用这些原语的前提，verifier 检查签名、纯度和成员关系。分页 attention 将 K、V、key 的逻辑 token 坐标及有效位作为 source；query、query 的坐标和 scale 是 captures。partial 写 f32 输出片段与 f32 log₂-sum-exp；空 split 分别写零和 −∞。merge 在前一调用完成后读回同一对 workspace，执行作者写出的成员选择与运算。

| 作者的 merge | 部分 split 为空 | 某输出行全部 split 为空 |
|---|---|---|
| 沿用 V1 的 `splitk_attention_f32_to_f16_reduce` | 按原式对所有 split 求最大值、`exp(lse−maximum)`、分母与加权和，保留原有指数和窄化。若最大值有限，空 split 的权重为零 | 有 split 且 LSE 全为 −∞ 时，减法得到 NaN，结果各分量为 f16 NaN。lowering 保留这一行为 |
| 作者显式选择跳过空 split，并定义空结果为零 | 作者从成员域或显式传递的有效位构造 mask，最大值、权重和加权和只使用有效成员。在归一化前按是否存在有效成员分支 | 作者分支直接写出各分量 +0，不对空域执行除法。额外传递有效位时，由作者同步声明 partial 与 merge 的接口 |

第二种写法是另一份作者程序的合同。split 轴长度本身为零时，显式的“无有效成员”分支同样写 +0；沿用原式时，空求和为零、空收缩分子为零，最后仍执行零乘零分母的倒数，输出 NaN。有效成员也可能产生非有限值；只有作者的数据合同保证标记无歧义时，才能用 LSE 的值判空。编译器不能根据算子名、零 partial 或 −∞ 哨兵给第一种写法加跳过逻辑。

## 量化记录与块缩放

以下是闭合格式的语义。verifier 校验 carrier dtype、rank、轴关系和记录长度；字节内容约束属于格式数据前提。解码、无损整数统计及浮点舍入点均须在降低后保留。

| 格式／逻辑长度 | 字段位置（相对记录首字节，半开区间，多字节小端） |
|---|---|
| Q4_K／256 | 144 字节。`[0,2)`、`[2,4)` 是 f16 的 d、dmin；`[4,16)` 打包 8 对六位无符号 scale 和 minimum；`[16,144)` 打包 256 个无符号四位权重 |
| Q8_K／256 | 292 字节。`[0,4)` 是 f32 的 ds；`[4,260)` 是 256 个补码 i8；`[260,292)` 是 16 个补码 i16，每个等于前述量值连续 16 项的精确和 |

**Q4_K 位序。** 对 t=0…3，取三个无符号字节 L=record[4+t]、H=record[8+t]、P=record[12+t]。第 t 组的 sc、m 分别取 L、H 的低六位；第 t+4 组的 sc 以 L 的高两位为高位、P 的低四位为低位，m 以 H 的高两位为高位、P 的高四位为低位。权重 i 使用字节 `16+32*(i//64)+(i%32)`，i%64<32 取低半字节，其余取高半字节。

**Q8_K 编码。** 输入 f32[G,256]，输出 u8[G,292]。每条记录选绝对值最大的带符号量 e，并列取较小值。全零时 ds=+0，所有量值和分组和为 0。其余记录以 RN-even 的独立 f32 舍入 R32 计算 `v=R32(−127/e)`、`ds=R32(1/v)`；第 i 个字节是 `R32(x[i]*v)` 取最近偶整数后饱和到 [−128,127] 的补码。随后对落盘的 i8 值精确求各组 i16 和。编码要求输入、v、ds 和乘法结果均为有限值。

**Q4_K × Q8_K。** 两个 u8 carrier 的形状分别为 [G,144] 与 [G,292]，G 必须相同。第 j 个 32 元素组先求四位权重与有符号量值的精确整数点积。令 S 为这八个点积乘各自 sc 后的和，T 为每组两项 i16 分组和之和乘各自 m 后的和；二者用 i32 精确表示。分别做 `p=R32(f32(d)*R32(S))` 和 `n=R32(f32(dmin)*R32(T))`，该记录贡献 `R32(ds*R32(p−n))`。记录间 f32 求和可保序重括号化；记录内不得跨上述舍入点融合或隐式 FTZ。G=0 的点积为 rank-0 f32 +0。格式不补不完整的 256 元素记录。

**FP8 块缩放。** 对 group_size=32 的 E4M3FN 形式，左、右 carrier 的逻辑形状分别为 [M,G,32]、[G,32,N]。同一个 k 的组为 g=k//32，组内 t=k%32；左值 A[m,g,t] 先乘左 scale[m,g]，右值 B[g,t,n] 先乘右 scale[n,g]，再按这两个对约轴配对乘加至 f32。两侧 scale 是 u8 的 E8M0 编码，0…254 解码为 2^(code−127)，255 为 NaN。作者若提供 [G,N] 排列的右 scale，须显式转为 [N,G]。verifier 检查 scale 分别绑定左自由轴 M、右自由轴 N 及同一组轴 G；尺寸碰巧相等不能授权交换轴。

## program 的静态调用合同

行摘要 `I.welford(samples)` 消费一维 f32 有效成员。返回的 SSA 元组依次为
`(count: i32, mean: f32, m2: f32)`，要求有效成员数可用 i32 表示。
空域返回三个零。合并相邻的非空摘要时，令 `delta = mean_b - mean_a`，
`n = count_a + count_b`，`d = delta * float(count_b) / float(n)`；
输出为 `(n, mean_a + d, m2_a + m2_b + delta * float(count_a) * d)`。
空摘要合并时保留另一项。支持按顺序解包或以 `.count/.mean/.m2` 取字段。

捕获结果按顺序保存 `workspace/output` 声明、显式 `I.fill(view, value)`、kernel 调用及返回的输出结构。kernel 符号、调用结构和输出个数在捕获完成时确定；地址、维度以及由维度求出的工作区尺寸可延迟到 prepare 求值。例如 MoE 可在四次调用前记录三次有序 fill，多输出可返回原有视图的 tuple。fill 要求目标可写、值符合目标元素类型，覆盖指定视图的逻辑域，是作者调用序列中的可见写入。

验证每个使用者的输入由调用实参、初始化或此前写入供给，依赖顺序覆盖读写冲突。新 workspace 未承诺初值；prepare 只绑定和分配，compile 只构建，fill 随 launch 执行。workspace 从分配到最后一次相关读写完成前保持有效；异步出口按队列依赖延迟释放，多输出的所有权随返回对象保留。根据设备计数选择下一次 host 调用或在捕获时读取 tensor 内容均超出此合同；数据相关的算法循环、分支和提前终止在 kernel IR 中表达。

## 对 V1 30 个登记程序的语言覆盖设计

### MoE 的预分配输出调用

产品入口 `moe_alignment_into(ids, expert_blocks)` 接受 4096×2 的 i32 expert ID 和调用方提供的 128 项 i32 缓冲。每个 ID 属于 [0,64)。程序分配 counts、cursors、offsets、routes 和 total，两次清零计数器，并把 routes 填成 8192。count、prefix、scatter、mark 按声明顺序执行。返回值为 routes、原 expert_blocks 和 total。

mark 只写 expert_blocks 的前 total/128 项。调用方可以重用同一个输出缓冲，未覆盖的尾部保留调用前的内容。直接通过 `intentdsl.jit(examples.moe_alignment_into, backend=...).run(ids, expert_blocks)` 执行时，工作区、三个 fill 和四个 kernel 均由产品管理。

此表按 V1 的程序标识核对语言需要，不是 V2 运行结果。weighted RMSNorm 是额外的首路线代表；30 项中登记的是 layer_norm。

| V1 登记项 | 必须能表达的结构／不能丢的事实 |
|---|---|
| relu、swiglu、transpose、dropout、rope、adamw | 元素映射、轴置换、按坐标随机数、多输入/多输出、成对原地更新与优化器状态 |
| softmax、layer_norm、batch_norm、group_norm_backward | 多归约、权重广播、Welford record、可变统计量、梯度与显式多次调用 |
| cumsum、causal_linear_attention、gated_delta、mamba_chunk_state、causal_convolution | prefix、矩阵状态、ordered recurrence、contract 与沿源轴的邻域边界 |

### 独立成员内的有序程序

`kir.owner_space` 的操作数与 `axes` 逐项确定成员域。区域的块参数是成员坐标。输出地址必须保留每个独立坐标，同一 InOut 的读写使用相同投影。前端将固定顺序的循环携带值写成 `scf.for` 的迭代参数和 yield。依赖谓词保护的读取放在 `scf.if` 的相应分支内。一次整轴读取产生 rank-one SSA 值，广播、求和和循环状态都保留成员轴的尺寸来源。

`I.window_contract` 在每个 batch、chunk 和 head 上计算独立的矩阵。输入布局为 `[B,L,H,P]` 与 `[B,L,G,N]`，系数视图为 `[B,H,C,S]`。调用时检查 `L=C*S` 和 `H=G*head_group`。右操作数的纯标量区域接收当前 basis、对应系数、当前 prefix 和 chunk 末项，区域必须返回 BF16。两个 BF16 面板在 S 轴做 FP32 累加，发布 `[B,C,H,P,N]`。没有跨 chunk 的状态依赖。
| embedding、csr_spmv、jagged_mean、nonzero、moe_alignment、histogram | 间接地址、分段/空段、索引生成、计数/前缀/散射、确定冲突语义及 workspace |
| gemm、batched_gemm、ragged_gemm、int8_gemm、q4_projection、fp8_gemm | 批维、变化的分组 extent、不同累加 dtype、压缩 record 与分组 scales |
| attention、paged_decode、cholesky | 因果及分页掩码、softmax 状态与部分结果合并、三角域内有序原地状态 |

**实现次序。** 首路线只要求二维动态 view、连续/受支持的非连续访问、行并行、f16/f32 转换、逐元素算术、行 max/sum、显式输出和标量参数。其余构造的类型与顺序在设计中有位置；实现时以各自的真实作者程序接通，未接通的合法构造报告 lowering 缺失。
