# Intent 编程模型

## 1. 定位

Intent 是结构化的算子 kernel DSL。作者定义一份硬件无关的 logical kernel algorithm；编译器在不改变该算法的前提下，为选定 target 形成执行组织、值表示、访问、存储和目标语言程序。

Intent 既不是图编译器，也不是完整算子库选择器。它的编译单位是一个 kernel definition 的 specialization，不是模型图、Python 调用树、provider 模板或 whole-operator 名称。

## 2. Target 由编译调用选择

Triton、cuTile、CPU、RISC-V/RVV 等 target 是编译调用的外部输入。它们不是 DSL value、`Constexpr`、类型、函数参数或算法分支。

```text
Intent source + specialization + external target selection
    -> target artifact
```

规格可以说明某项算法语义怎样投影到不同 target，但作者 source 与 canonical Kernel IR 不携带 provider、设备型号、ISA 或 capability identity。同一项 canonical operation 可以在一个 target 上直接映射、在另一个 target 上展开，或因目标确实无法兑现而被拒绝；这些结果不改变 operation 本身的定义。

## 3. 作者写下什么

作者拥有所有会改变程序可观察结果的算法事实：

- 输入、输出、可变参数以及必要的 shape、bounds 与 alias semantics；
- logical domains、由作者边界得到的 source-derived subregions 和 indexed relations；
- 硬件无关的 `if`、ordered `for/while`、unordered `parallel`、状态转移与停止条件；
- 张量索引、广播、形状变换和数值表达式；
- reduce、scan、region fold/scan、contract、scaled contract、sparse contract 与 histogram 的完整操作语义；
- indexed read/write、collision reduction、atomic、logical buffer 与 deterministic RNG；
- 一个算法使用一个 kernel 还是由 Python wrapper 编排多个 kernels。

这些内容构成 canonical Kernel IR 的唯一算法权威。编译器可以分析和引用它们，但不能静默替换成另一种算法。

## 4. 作者不写什么

下面这些信息只有在 target 与 physical program 被选择后才有意义：

- program id、launch grid、thread block、warp、lane、hart 或 CPU vector lane；
- execution tile、向量长度、线程分块与目标相关的静态 block shape；
- 寄存器、shared memory、TMEM、local memory 等存储层级；
- pointer tensor、物理地址宽度、尾块 mask 与 neutral padding；
- copy instruction、MMA variant、layout、swizzle、pipeline、barrier 与物理 atomic scope；
- Triton/cuTile API form、autotune candidates 与 winner；
- provider capability 或设备型号分支。

作者可用`I.assume_in_bounds`声明typed index relation的范围，并用external-view annotation声明`alias/noalias`。语言没有一般布尔precondition或优化hint；其它输入关系必须由类型、普通operations或未来被真实kernel逼出的closed typed construct表达。

## 5. Logical subregion 不是 physical tile

domain 是一组有序 logical coordinates。source-derived subregion 是由作者边界、输入 relation 或算法 metadata 从 source domain 取得的逻辑连续区间。它们不归属于 program、block、warp、线程或 vector register，长度可以在运行时确定，也不要求二次幂。

subregion 会改变 body 读取的逻辑成员，因此属于算法。compiler 为执行选择的 physical tile 不改变作者 tensor shape，也不会伪装成 logical subregion进入 Kernel IR。

`I.indices`、subregion与indexed relation产生的coordinate values即使经过helper call、slice、broadcast、reshape或transpose，仍保留source identity与typed coordinate provenance。这些facts让compiler能从作者已写下的logical predicate证明physical all-valid/all-invalid ranges，但不赋予compiler改变logical members的权力。

作者可以表达“对变长序列的有效前缀做归约”或“第 `p` 份读取 `[begin_p,end_p)`”；不表达“每个 CTA 处理 128 个 token”“CPU 每次处理 16 个元素”或“RVV 使用当前 VL”。

Region fold/scan中的source slice不是作者创建的ordinary subregion，也不能逃逸成value、shape或ABI。它是structured operation内部受homomorphism约束的parametric slice：operation对所有合法连续segmentation定义同一结果，compiler只为当前physical program绑定extent。作者只能消费被同步切片的tensor components和absolute source coordinates，不能观察segment identity、数量或chosen extent。该受限语义不赋予ordinary loop任意重新分段的权限。

## 6. 程序式与张量式语义并存

Intent 不是纯表达式图。一个 kernel body 可以同时包含：

- 完整 domain/subregion 上的张量运算；
- 具有 SSA merge 的条件控制；
- ordered loop 与 loop-carried state；
- unordered parallel iteration；
- first-class reduce、scan、homomorphic region operations 与 contraction；
- indexed access、external view 与 logical mutable buffer effects。

张量式构造不会消除程序顺序；程序式控制也不会迫使作者写机器执行单位。编译器必须同时保持 logical tensor flow 与 control/effect semantics。

## 7. 可观察语义与实现自由

可观察语义包括：

- 输出值、输出 shape 与 dtype；
- external effects、atomic ordering、collision 与 alias behavior；
- ordered control、loop state 与停止条件；
- accumulator、rounding、approximation 与 structured-operation reassociation；
- host 可见的 kernel 数量、调用顺序与中间 tensor interface。

在保持这些语义的前提下，编译器可以为每个 target 分别引入 execution coordinates、blocking loops、vectorization、value materialization、access forms、storage placement 和 target primitives。GPU、CPU 与 RVV 不要求共享一套改名后的 GPU physical topology；它们共享的是算法事实与可复用分析。

## 8. 规格组成

- [`kernel-and-host.md`](kernel-and-host.md) 定义 kernel、helper、specialization 与 host orchestration；
- [`logical-program.md`](logical-program.md) 定义 domain、subregion、control、state、structured operations、relations 与 effects；
- [`../dsl/`](../dsl/README.md) 定义 Python DSL 表面。
