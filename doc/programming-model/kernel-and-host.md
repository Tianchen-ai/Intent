# Kernel 与 Host 边界

## 1. Kernel definition

一个 `@intent.kernel` 定义一份独立的 logical kernel computation。target 由编译调用在 source 之外选择；对给定 specialization、interface 与 target，一次编译产生一个 target kernel artifact。host/runtime 决定何时以及如何 launch 该 artifact。

kernel：

- 通过 `In`、`Out`、`InOut` views 与 runtime scalars 接收参数；
- 不通过 Python return 把 device tensor 返回给 host；
- 可以包含硬件无关的 control、state、structured operations 与 effects；
- 不查询 target、设备资源、provider capability 或 launch configuration；
- 不分配或发射隐藏的第二个 kernel。

编译器不得把一个 kernel 自动裂成多次 host-visible invocations，也不得创建对 host 不可见的跨 kernel workspace 与同步。

## 2. Multi-kernel algorithm

作者可以选择一个或多个 kernels 实现同一外部 callable 合同，并须保持任务规定的数值、effects 与接口语义。内部 kernel interfaces、逻辑分组与中间 tensor shape 可由作者设计，无须由外部调用方给定。这里的可观察接口也包括 wrapper 内部实际传入各 kernel 的 views 和 scalars。

选择多个 kernels 时，作者显式定义多个 `@intent.kernel`，并由普通 Python host code：

1. 选择算法 specialization 与编译 target；
2. 分配输出和中间 tensors；
3. 建立 invocation 顺序或并发关系；
4. 管理跨 kernel 可见状态与生命周期。

Intent 逐个编译 kernel；Python wrapper 是 multi-kernel orchestration 的权威。compile、artifact creation 与 launch 是不同阶段，不能在 Kernel IR 中混为一个 operation。

## 3. Helper function

`@intent.fn` 是 kernel 内的硬件无关 helper：

- 参数与结果是 typed DSL values；
- 可以返回 scalar、tensor、tuple 或 record；
- 可以包含和调用点相同语义的 read/write effects，effects 从函数体得到；
- runtime values 必须显式成为参数；只允许捕获不可变 `Constexpr`；
- 不允许递归、host dispatch、kernel launch、target query 或 provider selection。

helper 是否 inline、是否成为 target-local device function，是实现问题，不改变调用语义。

Helper call同样不改变typed index relation。来自`I.indices`、subregion或indexed relation的coordinate values通过helper parameters/results后仍具有同一source identity、axis mapping与SSA provenance；是否inline不能让它们退化为无来源的integer tensor。

## 4. Runtime 与 `Constexpr`

runtime 参数在每次 invocation 时取值，可以参与数据计算和 runtime control flow。

`Constexpr[T]` 在 specialization 时绑定，只能控制硬件无关的算法变体，例如：

- causal 与 non-causal；
- 是否应用某个 activation；
- 算法定义中的 group count、sparse format 或数值模式。

`Constexpr` 不得查询或选择：

- target/provider、GPU 型号、ISA 或 capability；
- TMA/MMA/layout；
- warp、stage、tile 或 memory space；
- autotune configuration。

被选择的硬件无关分支进入 canonical Kernel IR；target-specific selection 发生在其后。

## 5. Kernel 内控制流

kernel 内允许：

- runtime `if`，条件来自 scalar runtime value；
- algorithmic constexpr `if`；
- ordered `for/while`、loop-carried values、`break` 与 `continue`；
- unordered `parallel` iteration。

tensor predicate 使用 value-level `select`，不作为 structured `if` 条件。独立 `stop` operation 不存在；终止由 loop condition、`break` 或 domain/subregion endpoint 表达。

这些控制都属于同一个 kernel，不能被误解成 multi-kernel dispatch。

## 6. Interface 与 kernel-local resource

external view 的作者可见语义是 element dtype、logical shape、runtime element strides、读写方向、bounds 与必要 alias relation。作者不手算 target pointer arithmetic，也不把 alignment、contiguity、TMA eligibility 或 vector width 当作算法参数。

kernel-local logical buffer 是算法状态。作者定义 shape、dtype、初始化与读写关系，不指定它最终位于 SSA、register、stack、shared/local memory 或其它 target storage。若中间 tensor 必须跨 kernel 存活，它由 host 显式分配，不属于 kernel-local buffer。

同一 external allocation 在多个 kernels 或 host/device 之间的并发访问，由 Python wrapper/runtime 的 invocation 与 synchronization 语义定义；单个 kernel body 不通过 physical atomic scope 隐式扩大自己的参与者集合。
