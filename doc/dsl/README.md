# Intent DSL

## 1. 语言表面

Intent DSL 使用受限 Python AST 表达一个硬件无关 kernel algorithm。Python 提供可编程语法；正式语义由 canonical Kernel IR 定义，不等同于任意 Python 执行。

target 在编译调用中选择，不是 DSL value、类型、`Constexpr` 或控制条件。本文档定义最终作者表面，不以当前 frontend、KIR 或 provider 是否已经实现为依据，也不记录实现进度与性能状态。

## 2. 构造的归属

### Canonical algorithm semantics

这些内容必须在 canonical Kernel IR 中拥有唯一、可验证的表示：

- kernel interface、runtime/constexpr 参数与 helper calls；
- domain、source-derived subregion、logical index、compositional coordinate provenance 与 indexed relation；
- structured control、unordered parallel iteration 与 loop carry；
- typed tensor/value operations，包括 tuple/record、full、broadcast、reshape、transpose 与 join；
- generic reduce、scan、region fold/scan、contract、scaled contract、sparse contract 与 histogram；
- external/logical-buffer read/write、unique/reduction scatter 与 atomic operations；
- stateless counter-based RNG。

`I.Enum`是constexpr-only surface type，在specialization时归一为closed canonical enum，不进入runtime ABI。

### 具名计算与 surface shorthand

公开 DSL 以作者熟悉的具名计算为常用入口；通用 structured constructs 与 `@intent.fn` 继续支持算法组合。公开名字不必成为新的 canonical operation：下列写法由 frontend 完整、机械地归一到上面的唯一 canonical path：

- `dot/matvec/vecmat/matmul`、`outer`、`scaled_matmul/sparse_matmul`；
- `reduce.sum/max/any/all`、`cumsum/cummax` 与 `arg_reduce.max`；
- Python `range`、slice、条件表达式、`break`、`continue`；
- `zeros`、常用 pointwise helper 与 value-level `mask`；
- `indices`、endpoint、`ragged(...)`、`members(...)` 等 relation helpers；
- ordinary indexing/assignment 与明确的 gather/scatter convenience methods；
- `sparse_contract_2to4` 等具体 format convenience spelling。

### Author libraries

softmax、logsumexp、Welford、attention、normalization、MoE、量化 GEMM 等完整算法由作者使用基础 constructs、`@intent.fn` 与显式多个 kernels组合。它们不是 language intrinsic，也不由 compiler 用 whole-operator matcher替换成模板。

### 不属于 DSL

- `auto("TILE")`、`state_stream` 与所有 `partition` forms；
- public `ordered` 或独立 RaggedOp/MembersOp；
- program id、grid、block、warp、thread、lane、hart；
- physical tile、vector length、register/shared/TMEM/local storage；
- pointer spelling、physical mask/padding、copy instruction、layout、pipeline 与 barrier；
- physical atomic scope；
- provider/target/device/ISA capability branch；
- autotune configuration 与 winner。

这些排除项的算法能力并未被删除：

- reduce、scan、region fold/scan与ordered loop承接原`state_stream`中分别属于element aggregation、region homomorphism和strict recurrence的语义；
- part domain、boundary arithmetic、source slicing与partial tensor承接可观察 partitioning；
- offsets、subregion与indexed relation承接 ragged关系；
- indexed read→SSA value→write承接逻辑 copy；
- physical program根据 target补回 blocking、storage、copy、atomic scope与execution mapping。

## 3. Operation 进入 canonical language 的判据

判断一项构造时：

1. 抽掉 target spelling、physical participant、tile、storage与instruction，写出它的 logical values/control/effects语义；
2. 尝试用已有 canonical constructs展开；
3. 若展开完全机械且不改变结果、顺序、shape、effects、alias、数值与接口语义，专用 canonical op不成立；
4. 若展开必须选择 reduction order、中间物化、snapshot、collision behavior、format decode或其它 realization，该 structured semantics必须first-class保留；
5. GPU、CPU、RVV只用于检验同一语义是否成立；某个 target有无同名 primitive不决定公共 op是否存在。

“理论上能用标量循环模拟”不是机械展开。若模拟会丢失 associativity、prefix、paired-axis relation、metadata interpretation、collision或atomicity，compiler之后只能从任意代码猜回结构，说明该语义应当保留。

## 4. 文档

- [`core.md`](core.md)：作者构造、canonical operations与surface shorthand；
- [`types-numerics-and-effects.md`](types-numerics-and-effects.md)：类型、数值、views、buffers、memory effects与RNG；
- [`examples/`](examples/README.md)：理想化完整 DSL 示例。
