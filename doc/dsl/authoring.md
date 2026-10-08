# 作者速查

本页是 [core.md](core.md) 与[数值规则](types-numerics-and-effects.md)的使用入口，不定义第二套语义。MCP 的 `search` 找概念/API/诊断，`api` 查当前声明及返回规则，`read` 读语言规则与接口说明。公开声明不等于所有 target 已支持；没有性能测量不能宣称高效。

完整 callable 的算法组织由作者选择。在满足任务的数值、effects 与外部接口合同的前提下，作者可以自行设计内部 kernel interfaces、逻辑分组和中间 tensor shape；这些内容无须出现在任务签名中。内部 kernel 之间实际传递的 tensor interface 同样是可观察语义，由 compiler 保持。Compiler 为每个已声明的 kernel 形成物理执行程序。

## 从 Triton 算法到逻辑域

可以把 Intent 理解为 Triton 式 kernel 算法的逻辑域表达：保留算法的独立工作、逻辑分组、局部结果与阶段依赖，把物理 tile、线程布局和流水线配置交给 compiler。它不是把完整算子交给库或 compiler 自动选择算法的接口。

- 先确定各阶段的输入、输出、参与计算的成员和依赖，再写 kernel。Triton 的 `program_id` 既可区分算法上的局部汇总，也可仅用于给独立结果分配物理 tiles：前者保留相应逻辑分组与中间接口，后者由 tensor 的自由轴或 `I.parallel` 表达独立坐标即可，不要求逐个 program 复刻分组。
- Triton 的向量表达式对应逻辑 tensor 运算。核对 dtype、成员与数值合同后，`tl.sum` 对应 `I.reduce.sum`，`tl.dot` 对应 `I.matmul`/`I.contract`，`tl.associative_scan` 对应 `I.scan`；无需逐元素展开成普通循环。外层 `I.parallel` 不会使内层普通循环并行；structured operation 产生的 tensor values 可以由满足无冲突 effects 的独立逻辑点消费。
- Tensor 运算与 structured operation 保留的自由轴本身就表达独立结果坐标；compiler 在保持 control/effects 的前提下分块和映射这些轴。无需仅为复刻 Triton 的物理 `BLOCK_M/BLOCK_N` 而额外建立固定大小的 logical subregions。算法需要的独立成员集合、局部汇总或跨 kernel 接口仍须显式保留；完整归约的 reduction axis 不因省略物理 tile 而成为自由轴。
- 外部结果是 scalar，不意味着应把整个输入放进一个全轴归约。`I.reduce` 处理传入 value 的指定轴；全域 value 的归约仍有全局依赖。对 producer 或后续 consumer 写 `I.parallel`，不会替这项归约建立分组或跨 kernel 汇总。
- 若所选 Triton 算法包含多个 kernels，翻译到 Intent 时保留阶段、跨 kernel tensors 和 host 调用顺序。Kernel 内的多个表达式、helper 或 `I.buffer` 都不能替代这些阶段。分组大小、局部结果接口可以由作者选择；compiler 不会补出源程序缺失的阶段。
- 以完整 callable 的并行工作量、数据读写和所有 launches 的总成本选择组织，不以源码最短或 kernel 最少为目标。归约、prefix 和 ordered loop 各自的成员、顺序、dtype 与数值合同必须保持。

## 类型、literal 与 shape

程序使用 `import intent` 和 `import intent.language as I`；kernel/helper 分别用 `@intent.kernel`、`@intent.fn` 声明，装饰器不在 `I` 命名空间。`I.In/I.Out/I.InOut` 描述外部 views，必须同时给出 dtype 和 shape，例如 `I.In[I.f32, ("M", "N")]`；rank-0 view 的 shape 写 `()`。`I.f32` 等描述 scalar dtype；Python literal 可按上下文实例化，但两个不同 dtype 的 runtime values 必须显式 `I.cast`。例如先把 bf16 输入 cast 到 f32，再和 f32 累加器计算，最后 cast 回输出 dtype。

Host 的 `I.dtype(name)` 按公开 dtype 的 canonical 名称查找相同 token，例如 `I.dtype("f32")` 得到 `I.f32`，`I.dtype("i64")` 得到 `I.i64`。名称不使用 Torch 的 `float32`、`int64` 拼写；直接使用 `I.f32`、`I.i64` 等 token 同样有效。

View 注解的第三个可选参数是 `I.constraints(...)`，例如 `I.Out[I.f32, ("N",), I.constraints(noalias=True)]`。不同 view 默认可以 alias；`noalias=True` 声明该 view 的底层 allocation 与其它 view 不重叠，是调用方必须满足的前置条件，不能仅凭 `Out` 访问方向推断。相同 `alias="group"` 的 views 必须来自同一底层 allocation，可以具有不同 offset、shape，也可以是不相交的子视图；不同 group 不推出 allocation 不相交。`alias` 不能与 `noalias=True` 同时指定；这些字段不改变读写方向、shape 或 dtype。某个 target 尚不支持重叠可写 views 时须明确报告其执行能力限制，不能把它解释为语言的默认 noalias 合同。

Rank-0 view 使用空索引 tuple：`value = view[()]` 读取 scalar，`view[()] = value` 写入 scalar。读写仍遵守 view 的访问方向、dtype 和先写后读规则。

完整切片的数量与 rank 一致：一维 view 使用 `view[:]`，二维 view 使用 `view[:, :]`。每个 `:` 消费一个已有轴，不创建额外轴；一维 view 上的 `view[:, :]` 因索引轴数超过 rank 而不合法。Tensor value 的切片遵守同一轴数规则。

Literal 首次形成 runtime value 时若没有 expected dtype，Python `bool/int/float` 分别采用 `bool/i64/f64`。需要 f32 的循环状态可用 `I.cast(1.0, I.f32)` 初始化；后续使用不会反向改变它的 dtype。正/负无穷常量使用 `I.inf` / `-I.inf`，可写 `I.cast(I.inf, I.f32)` 指定 dtype。

`I.exp2(x, approximate=True)`、`I.tanh(x, approximate=True)` 和 `I.fdiv(x, y, approximate=True)` 是已有的显式近似浮点接口；省略 `approximate` 时保持普通运算。非默认模式要求 operands/result 为 `I.f32`，`approximate` 必须是 constexpr bool。`exp2/fdiv` 还接受独立的 `flush_to_zero`，默认 `False`，仅在近似模式下可启用。各接口的误差界限、特殊值和 subnormal 规则见[显式近似数学](types-numerics-and-effects.md#51-显式近似数学)；这是逐操作数值合同，不改变相邻运算、归约精度或求值顺序。

`I.select` 的 bool 条件不提供数值分支的 expected dtype。两个分支都写成 literal 时，不要从生成条件的 tensor 推断结果 dtype；例如需要 f32 符号值时写 `I.cast(I.select(mask, -1.0, 1.0), I.f32)`。已经产生的 runtime value 不会因后续与 f32 相乘而重新实例化。

Tensor 和 view 有 `.shape`；scalar、tuple、record、domain 没有统一 `.shape`。`I.full(shape, fill, dtype)` 产生 tensor value，不分配跨 kernel workspace。`I.dot(lhs, rhs, acc_dtype=...)` 必须显式指定累加 dtype，只接受两个 rank-1 tensor，返回 rank-0 tensor `[]`，不是 rank-1 `[1]` 或一个 Python number。Scalar 和 rank-0 tensor 是不同类型；需要将 scalar 放入 rank-0 tensor 的分支或 carry schema 时，可用 `I.full((), value, dtype=...)` 显式构造。Pointwise scalar broadcast 由 frontend 显式表达。

`I.zeros`、`I.full` 和 indexed read 的结果都是不可变 tensor SSA values。下标赋值和 `I.store` 的写入目标必须是可写 view 或 logical buffer，不能原地修改这些 tensor values。Tensor carry 通过计算新 tensor 并重新绑定名字更新；逐地址写入则使用显式存储。

名字重绑定可以携带完整 tensor value，无须为了每次状态更新而写回存储。下面 `values` 与 `delta` 是同 dtype、同 shape 的 tensors，`active` 是同 shape 的 bool tensor，`count` 是 scalar index；每轮 `state` 的 schema 保持不变，物理存储由 compiler 决定：

```python
state = values
for step in I.domain(0, count):
    state = I.select(active, state + delta, state)
```

`I.buffer` 返回可变存储对象，buffer 名字本身不是 tensor value。把其中内容用于赋值右值、pointwise 或 reduce 时，先通过下标读取为 scalar/tensor；例如 `value = state[:, :]` 读取二维 buffer 的完整内容。

### Comparison operators（比较运算）

Scalar/tensor 比较使用 Python 运算符 `==`、`!=`、`<`、`<=`、`>`、`>=`，结果 dtype 为 `I.bool`。操作数按普通 literal/dtype 与尾部 broadcast 规则对齐，结果保留广播后的 shape；这些运算符不是 `I.eq` 等 intrinsic 调用。

## Domain、index 与 broadcast

`rows = I.domain(0, M)` 是 logical coordinate domain，不是整数 tensor，也不是归约的 axis 编号。Domain 本身不参与数值算术或比较；需要坐标值时先使用 `I.indices(rows)` 产生 logical index tensor。坐标和 domain 循环变量使用 `I.index`，与 runtime `I.i64` 不同；参与坐标算术的 runtime 步幅、偏移等先显式转为 `I.index`，循环变量参与浮点运算前则转为相应浮点 dtype。`x[rows, columns]` 读取完整二维 logical region；`I.reduce.sum(value, axis=1)` 的 `1` 指 value 的第二个 axis。

Pointwise 按尾部对齐，允许 scalar/size-one broadcast。`[M]` 与 `[M,N]` 相加不会自动按行匹配：先 `I.reshape(row_value, (M, 1))`；`[N]` 可以直接广播到 `[M,N]`。两个未知 extent 不是因为都 dynamic 就兼容。

同样，`[C]` value 要对应 `[B,C,H,W]` 的第二个轴时，先变成 `[1,C,1,1]`；直接使用 `[C]` 仍只会与最后的 `W` 轴对齐。Broadcast 不依据变量名或数学用途选择轴。

Kernel annotation 使用符号维度时，`reshape`、`full` 等 shape 表达继续引用该符号或对应的 `value.shape`；固定输入本次恰为某个尺寸，不使这个符号等同于该尺寸的字面量。

`I.transpose(value, permutation)` 显式重排；permutation 的长度必须等于输入 rank，包含 `0..rank-1` 的每个轴且仅一次。省略 permutation 时反转全部轴顺序。`I.reshape` 保持 row-major element order，不是任意 data permutation。Domain/subregion 与 integer coordinate tensor 不可互换；索引关系、有效范围和 fill 必须来自作者实际表达。

Domain 索引按资源索引顺序形成读取结果的 tensor axes，赋值仍按 positional axes 对齐，不按 domain 变量名自动换轴。例如两个等长 domains 下，`output[rows, columns] = input[columns, rows]` 不表示矩阵转置；应显式转置读取的 tensor value，或构造具有所需对应关系的坐标 tensor。不同 subregions 的动态长度也不会因本次输入碰巧等长而成为同一 extent；需要使用已成立的 shape relation，或在共同输出 domain 上表达坐标映射。

`view[region]` 返回由该区域成员组成的 tensor value。随后索引这个 value 时使用从 0 开始的局部位置，范围由 value 自身的 shape 决定，不沿用原 view 的绝对坐标或完整长度。

多个 tensor indices 按 broadcast 规则形成共同的索引 shape，坐标逐位置配对，不自动形成 Cartesian product。对二维 `x`，两个 `[K]` 坐标 tensor 的 `x[r, c]` 结果是 `[K]`；需要 `[M,N]` 坐标组合时，可写 `x[I.reshape(r, (M, 1)), I.reshape(c, (1, N))]`。`[M]` 与 `[1,N]` 仍在末轴比较 `M` 和 `N`，不表示 `[M,N]`。

所有 tensor indices 共同贡献一份 broadcast shape，每个 domain index 另外引入一个独立 logical axis。Tensor index 的 size-one 轴也属于这份 shape，不会因旁边有 domain 而自动消失。索引赋值的右值必须能 broadcast 到该索引表达式的结果 shape，目标 view 不会替右值隐式降维。

`I.gather(source, index, valid=True, fill=0)` 使用与 `source[index]` 相同的索引关系。`valid` 必须为 bool，`fill` 与 source 同 dtype，两者均须能广播到索引结果的 shape，不能扩大它。无效成员不读取 source，而返回 fill；省略 fill 时使用 source dtype 的零，bool 使用 `False`。

## Reduce、tuple 与 helpers

普通 `I.reduce` 声明 combine 可结合、可交换，允许并行重结合与重排；不承诺输入顺序或逐 bit 重现。`I.scan` 保留各个 logical prefix 的成员顺序，只允许保序重结合；严格顺序累加使用普通 loop。这些是不同的 operation 合同。

`I.reduce.sum/max/any/all` 返回归约后的 values，没有 `keepdim`；axis 是编译期确定的整数或非空整数 tuple，tuple 可同时归约多轴。若所有轴都被归约，结果是 scalar，而非零维 tensor。补 size-one 轴时，tensor value 使用 `I.reshape`；scalar 不能 reshape，可用 `I.full` 构造 tensor，或按赋值的广播规则直接写出。

例如已有 rank-2 tensor `value` 时，下面只展示归约后的 shape 恢复；第二行把 `[M]` 变为 `[M,1]`，以便后续按行 broadcast 到 `[M,N]`：

```python
reduced = I.reduce.sum(value, axis=1)
expanded = I.reshape(reduced, (value.shape[0], 1))
```

`I.cumsum(value, axis=1)` 对 shape 为 `[M,N]` 的 value 分别计算每一行的前缀，返回同 shape 的 tensor；未选中的轴保持独立，不在不同行之间传递状态。它不会先 flatten 输入。Scan 的成员来自传入的 tensor；读取 subregion 后再 scan，只包含该 subregion 的成员。`I.scan` 与 `I.cummax` 使用相同的 axis 规则。

`I.arg_reduce.max(value, axis=...)` 沿一个 tensor axis 归约，返回 `(values, indices)`，不能把整对结果当 indices。Indices 的 dtype 为 `I.index`，表示输入 tensor 被归约轴内从 0 开始的位置，不自动返回底层 domain/subregion 的绝对坐标。完整归约时两项均为 scalar；保留轴时两项均为保留这些轴的 tensor。分别核对两个 component 的 dtype；写入不同 dtype 的输出前必须显式 `I.cast`，包括 `I.i64` 与 `I.index` 之间，不能因它们都使用 64 bits 就视为同一类型。

Generic `I.reduce(value, axis=..., identity=..., combine=helper)` 的 identity、两组 combine 参数和返回值必须具有删除归约轴后的同一 schema。例如 `[M,N]` 沿 `1` 归约得到 `[M]`，identity 可写成 `I.full((M,), 0.0, dtype=I.f32)`。这里检查的是进入 combine 的 dtype；builtin 的 `acc_dtype` 或默认 widening 会先转换 source，不要求原 external view 与 accumulator 同 dtype。

Python tuple 与 `I.record(field=value, ...)` 是结构化 products，不要求各 component 同 dtype/shape，但每个 component 必须与对应 identity/combine/result 一致。Tuple 静态解构，record 用 `.field`；都不直接成为 host-visible kernel return。

`@intent.fn` 是 typed kernel helper，不是任意 Python 调用；普通 Python `abs/math.*` 不会自动变成 DSL。先查当前 API，使用 `I.abs`、`I.sqrt`、`I.asin`、`I.lgamma`、`I.log1p`、`I.erfc`、`I.i0` 等已声明数学入口，不猜 `I.keepdim` 等未声明名字。Runtime captures 显式传参，structured combine 必须 pure。

Helper 可以没有返回值；函数体自然结束与裸 `return` 等价，都保留已经执行的 reads/effects。有值返回仍须满足 runtime 分支的返回 schema 一致性。

## Control、effects 与多个 kernels

普通 `for/while` 保持顺序与 loop carry；carry 的初值与每轮更新必须保持 dtype、rank 和逻辑 shape，循环体内的 broadcast 不会改变初始 schema。Tensor carry 可按目标 shape 初始化，例如 `state = I.full(value.shape, 0.0, I.f32)`；`I.cast(0.0, I.f32)` 初始化的是 scalar。`I.parallel(domain)` 表达独立无序点，不允许 carry。Tensor predicate 使用 `I.select`，不控制 statement `if`。`Out` 进入 kernel 时未定义，读取前必须先定义；不能用 InOut 掩盖未定义读取。

`I.parallel` 可以嵌套，内层可以读取外层词法作用域的不可变 tensor values。外层各点内部产生的值分别属于各自的逻辑迭代；内层读取这些值不会把它们变成所有外层点共同更新的全局状态。

Runtime `if` 之后读取的名字，必须在分支前已有定义，或在每个正常继续执行的分支中赋值。不同条件之间的逻辑蕴含不会自动建立名字的定义；先初始化共同状态，再在分支中更新。

Kernel/helper 中的 Python `range` 是 domain 遍历的 shorthand，遵循正步长合同。即使边界都是常量，循环变量的 dtype 仍是 `I.index`，不是可按浮点上下文实例化的 Python literal；参与浮点计算前须显式 `I.cast`。逆序遍历使用正向 ordinal 并显式计算反向坐标，具体规则见 [core.md 的 domains 章节](core.md#3-domains-与-source-derived-subregions)。

`I.mask(value, predicate, fill)` 等价于 `I.select(predicate, value, fill)`：predicate 为真保留 value，为假使用 fill。它只选择值，不抑制写入；对写入坐标使用 `I.mask` 会把未选中成员写到 fill 指定的地址。`I.store` 和下标赋值没有 `mask` 或 `valid` 参数。条件写入使用 scalar 条件下的 structured control，或只遍历实际参与写入的 domain/subregion；普通并行写入仍须满足目标地址不冲突的合同。

例如已有同 shape 的一维 tensor values `values`、`active`（bool）和 `destinations`（index）时，可以逐个独立逻辑点表达条件写入。`out` 是可写 view；所有 active 点的目标地址须在界内且互不冲突：

```python
for i in I.parallel(I.domain(0, values.shape[0])):
    if active[i]:
        out[destinations[i]] = values[i]
```

一个 kernel 不自动拆成多个 launches。多个 kernels 由 host 分别编译、显式调用；跨 kernel tensors 的分配与生命周期由 host 管理。Kernel 数量与算法编排由作者定义，各 kernel 内的物理分块、布局与 target 配置由 compiler 形成。

算法定义的逻辑分组数量、边界和中间 tensor shape 可以显式表达，不属于硬件 block size。`I.parallel` 表达无序独立的逻辑 points，source-derived subregion 表达成员集合；两者都不指定线程块数量，也不产生额外 launch。跨 kernel 的中间 tensor 由 host 分配，作为显式 view 参数传递：使用 `Out` 表达输出、`In` 表达只读输入、`InOut` 表达对已有内容的读写。`I.buffer` 是 kernel-local 状态，不能跨 kernel 传递。

### Host 编译与调用

Public 调用为 `intent.compile(kernel, compiler=..., target=..., constexprs=...)`，返回 artifact。显式调用 `artifact(...)` 按 kernel 声明顺序传入全部 runtime 参数，`Out` 保留在声明位置；`artifact.run(...)` 只省略 `Out`，其余 views 与 scalars 保持原顺序，由 runtime 分配并返回输出。例如声明顺序为 `A: In, B: Out, scale: f32` 时，调用为 `artifact(A, B, scale)` 或 `artifact.run(A, scale)`。`Constexpr` 在编译时绑定，不传入这两种 runtime 调用。

`artifact.run(...)` 与 prepared call 的 `result()` 只返回声明的 `Out`：零个返回 `None`，一个返回该输出对象，多个按声明顺序返回 tuple。`InOut` 始终由调用方传入，通过原对象观察修改，不加入返回值。`artifact(...)`、`artifact.launch(...)` 与 prepared call 的 `launch()` 只执行并返回 `None`。`artifact.prepare(..., outputs=(...))` 按 `Out` 声明顺序绑定显式输出；省略 `outputs` 则分配输出。Prepare 不执行 kernel；`result()` 只取得输出容器，不执行或同步。CPU task join、GPU 当前 stream 和 MLU queue 的完成规则仍由相应 runtime 定义。

Host callable 可以用普通 Python `for/while/if` 编排已经编译的 artifacts，包括重复调用同一个 artifact。每次调用都是一次 kernel invocation，runtime scalars 可随调用变化；这与在 `@intent.kernel` 内写循环不同，也不要求在 host 循环中重新编译。`Constexpr` 的绑定仍属于编译阶段。

编译与 launch 分开。`intent.generate` 只生成 source/IR；target 在 host 选择，例如 `intent.targets.TritonTarget()`，不能在 kernel 查询设备或选择 warp/tile。

标量 constexpr 使用 Python 类型注解，例如 `STEP: I.Constexpr[int]`、`EPS: I.Constexpr[float]`、`ENABLED: I.Constexpr[bool]`。`I.f32` 等是 runtime scalar dtype 描述符。Host 已确定、用于选择硬件无关算法变体或成员关系的参数可以在编译时绑定；runtime 参数不会因为调用时恰好取固定值，就自动成为 Intent 编译期常量。需要支持不同变体时，由 host 选择相应 specialization。

`constexprs` 绑定 kernel 签名中声明的 `I.Constexpr[...]` 参数。View shape 的静态 extent 使用非负整数，如 `1`；字符串必须是合法符号名，如 `"M"`、`"K"`，`"1"` 不是整数 extent。注解中的符号名不会绑定 Python 局部变量；使用前写 `M, K = input.shape` 读取这些 extents，这也不会声明同名 constexpr 参数。只有动态 shape 的 kernel 无需把本次输入尺寸传入 `constexprs`。

同一 kernel 的 view shape 注解中，相同符号名表示同一 logical extent，调用时这些维度必须相等；不同符号名不声明相等关系。需要表达相等维度时复用符号名，不能依赖本次输入碰巧具有相同大小。

评测中的 `build(context)` 只是上述调用的薄适配：在 build 内分别 `context.compile("name", kernel)`，返回一个 host callable，在 callable 中分配中间 tensors 并调用 artifacts。它同样接受 `constexprs`，例如签名中的 `ENABLED: I.Constexpr[bool]` 可以通过 `context.compile("name", kernel, constexprs={"ENABLED": True})` 绑定，launch 时不再传入该参数。它不改变 DSL，也不要求整个任务只能写一个 kernel。

## 诊断的含义

`unknown intrinsic` 应先核对当前公开名字；`axis must be an integer` 检查是否误传 domain；dtype mismatch 检查 runtime operands 的显式 cast；shape mismatch 检查尾部对齐和真实 index relation。不得通过换输出 shape、忽略 tuple component 或放大容差“修复”任务。

公开规则允许、但 lowering 报 `NotImplementedError` 的程序是实现缺口，不自动成为作者错误。`intent.binary ... schema` 或 `make_tuple ... wrong type` 等非法 compiler IR 需要保留完整原程序与诊断交给 compiler 开发侧，不能反向改写语言规则。MCP 查询只提供原文和声明，不执行程序或判断数值正确性。
