# CPU executable programs

## 1. 编程模型与编译边界

CPU 是直接从 immutable canonical KIR 构造的非 SIMT execution family。一次调用包含显式 tasks；每个 task 拥有 workset、captures 和输出责任，内部执行顺序控制与 shaped computations，调用返回前完成依赖与 join。Task coordinate 不是固定 hart/thread ID，block 是计算范围，不是硬件寄存器或 matrix tile。

```text
canonical KIR
    → shared CPU task/block program / analyses / passes
    → 目标实现选择、需求协调与微程序展开
        ├─ Mojo legalization / serialization → Mojo / LLVM
        └─ Weft legalization → Canonical Weft IR → Weft compiler
```

CPU 与 GPU 共享作者语义及可复用分析，保持相近的 structured-program 抽象层，但不共享 GPU program id、warp/lane 或 fragment ownership。Provider 与 hardware 分开选择；Mojo/Weft 是 provider，x86/RVV 与可选 matrix extension 是硬件能力，不改变 KIR 算法、dtype、控制和 effects。

## 2. 共同 CPU IR

CPU dialect 拥有本 family 的 typed ABI、任务、轴/访问关系、资源、数值义务和 verifier；可复用 `func/arith/math/scf/memref/vector/linalg`，不重复包装标准算术，也不以标准 dialect 合法代替 CPU 语义检查。

当前程序必须独立保存：

- scalar/shaped values 的 dtype、logical axes、result/broadcast relations；
- views、访问坐标、有效域、fill、读写方向与 alias；
- task worksets、captures、coverage、唯一写入和完成边界；
- ordered control、blocking loops、bounds、tails 与 carries；
- reduce/contract 等 structured operations 的输入、轴、identity/combine、accumulator 与完整数值合同；
- 本层形成的 reuse、materialization、资源 owner、初始化、lifetime 与实际参数 binding。

完整的 structured operation 不要求展开成 scalar loops。共同 IR 保留计算含义与外围连接，不穷尽微程序内部的解码、寄存器布局或数值实现树，也不把 shaped extent 等同于 SIMD width。必要的 logical format/scale 信息来自 DSL/KIR，不从 provider shape 或 kernel 名猜测。

Construction 后只变换当前程序，不回读 KIR 重建执行结构。标准 operations、provider-local extensions 与后续展开不形成另一份 planning IR；origin、analysis cache 或选择记录不能与 IR 共同解释执行。CPU IR/analysis/transforms、provider implementation 与 runtime 按各自职责组织。

## 3. 编译器与实现作者的职责

| 责任 | 所属边界 |
|---|---|
| 算法、调用接口、可观察数值与格式语义 | Intent 作者及 DSL/KIR |
| task partition、外层 blocking、跨操作 fusion/reuse、访问与 lifetime | 共同 CPU passes |
| 计算块内部的 decode、子块、局部累加、转换和阶段组织 | 专家编写的目标 implementation，在该计算块语义内 |
| 实现所需的外围数据供应、共享表示和资源 | implementation 提需求，CPU/provider passes 协调并显式形成 |
| 机器 layout、指令、寄存器分配与调度 | 对应 provider-local lowering 或外部 compiler |

专业实现可以是参数化的 microkernel/微程序，含局部循环、scratch 和多个 operations；不必是一条指令，也不要求普通 Intent DSL 能逐句复述。实现作者提供复杂 realization，compiler 负责适用性、绑定和组合，不要求通用 pass 自动发明所有高性能结构。

按作用范围而不是优化名字划分责任：跨消费者 panel 的存在与 lifetime 由外围协调；仅供局部实现的 packing/fragment 由实现或下层 compiler 负责。共享层不能在查询实现需求前无条件固定其向量宽度、微块和内部 packing。通用 vector/reduction 变换可复用，但不是所有 provider 必经的最低公共表示。

## 4. 可编程目标 lowering

每个 implementation 必须明确：

1. 实现的 operation/计算区域语义、operand/result relation 与数值合同；
2. dtype、format、shape/tail、访问、capability 与资源的适用条件；
3. 对外层 block、输入表示、数据供应、输出形式和 scratch 的需求；
4. 实际消费的有限参数及合法域；
5. 将当前 operands、regions 和 bindings 实例化为真实 IR 的展开过程。

正式 transformation 查询有限适用实现，协调需求，再固定该候选的实现与参数并展开。布局/供应规划与展开消费同一选择；局部资源与外围 effects 必须连接到当前程序。实现身份可用于编译分派，不得替代输入语义或变成按 kernel 名选择整算子程序。

实现可用 IRBuilder、结构化宏或目标 DSL helper 编写；目标 source 通过其 frontend 形成 IR，导入时显式绑定 captures、axes/symbols、domain、输出和 lifetime，不靠文本拼接重建结构。普通操作的直接映射也是合法 lowering，不强制每个 add/load 都经过复杂微程序。

展开结果继续参与 legalization、analysis 和 verifier；不能只留一个 operation 加旁路 recipe 给 runtime/emitter 解释。可编程展开与整算子库旁路不同，也不因函数或类名含 `Emitter` 就变成终端代码发射。

## 5. Access、数值与资源不变量

ABI 保存 element type、rank、静态/动态 extent identity、offset/strides、access 与 alias，以及已定义的输入格式；native lowering 按该接口展开 pointer/descriptor/scalar 参数。所选实现需要的 contiguity、alignment、disjointness 必须显式成立并由实际 view facts 兑现，不暗中 copy、假定 noalias 或覆盖尚需读取的快照。

Task partition 依据 workset、dependence 和唯一写入形成。资源保存大小、对齐、owner、初始化与 lifetime；forwarding/rematerialization 保持坐标、dominance、effect order 与来源稳定性。微程序不能重复外围共享的准备工作，或隐藏跨调用 repack、workspace 与缓存。

数值语义遵循 [Types、Numerics 与 Effects](../dsl/types-numerics-and-effects.md)。输入量化格式、scale/group/zero-point、转换和 accumulator 合同在选实现前确定；不同格式不是可任意互换的 tuning 候选。格式未被语言定义时不能仅加 provider 名称冒充支持，也不把完整 Weft Encoding/Level surface 提升为 Intent 作者接口。

专业实现必须保持所承接计算的数值合同与外部 effects；其内部 partial、widen/narrow、FMA 或重结合只能使用合同允许的自由。实现作者承担语义保持义务，compiler 检查适用条件，不在每次编译重新证明任意量化代数式。普通算术不隐式继承 provider 的 fast-math/FTZ；FP environment 必须保持所需语义及调用方状态。

Scalar/vector/matrix 协作不默认引入独立执行上下文。同步 CPU 指令流使用 SSA、effects 和资源 lifetime；异步 engine、queue 或跨上下文同步只能由真实能力与完整依赖语义引入。

## 6. Capabilities、bindings 与选优

Capabilities 以 typed facts 描述实际消费者所需的 dtype、资源、memory/vector/matrix 能力；不匹配设备名称，不把固定 f32 宽度或 AVX/RVV 条件作为整个 CPU family 的定义，也不预建无人消费的硬件字段。

共同参数约束 task grain、外层 block 和跨块组织；实现参数约束局部微块、vector/replica/unroll 等；外部 compiler 参数归其消费者。跨边界参数只有一个 binding owner，通过明确约束关联，不能各选一个值。有限配置与对应实现/变换相邻，保留 override，不构建任意程序树的笛卡尔积。

CPU经验配置每行明确保存`shared`五个外层绑定、`local`实现参数和`implementations`实现集合。集合描述不同typed计算种类的实际实现，不是优先级或失败回退顺序。每个需要实现的计算必须有唯一适用成员，再检查原类型、布局、资源、数值和供数合法性；没有或多个匹配均拒绝该行。绑定写入当前operation的既有`ImplementationAttr`。一行至多形成一个完整候选，不按registry追加未指定的实现组合。

每个候选固定实现和完整参数，形成同语义的具体程序；可按明确策略或实测选择合法候选，不用 tuning 代替缺失的实现。无合法实现时诊断，不在 serialization 或执行失败后隐式换算法。Artifact/winner cache 分离，身份包含实际依赖的 specialization、view facts、provider/hardware、implementation 定义与 bindings。

## 7. Provider 与验证边界

Mojo 路径由选定实现形成其需要的循环、SIMD、数学操作、资源与 native ABI；Mojo/LLVM 继续负责机器 lowering、指令选择、寄存器分配和调度。共同 CPU IR 不必事先采用 Mojo 的向量表示。

Weft 的外部输入为 Canonical Weft IR。简单 structured operations 可直接转换，专业实现可由 Weft 作者程序实例化；adapter 保持宿主 task 的接口、坐标、控制和 lifetime，不引入隐式 hart/grid。RVV/IME layout、fragment、machine packing 与资源/指令实现继续归 Weft compiler，不强迫 Weft 先走 Mojo SIMD 展开。

两种 provider 可停在不同抽象层。只有确需独立 legality 和后续 consumer 的目标结构才增加 local extension，不为表面对称建立重复 IR。

Analyses 从当前程序重算 coverage、axis/access、dependence、reuse、lifetime 和 eligibility，相关改写后失效。每个完整 transformation group 声明输入条件、IR 改写、语义保持和 analysis 失效，并验证后置条件。Verifier 只检查，不修程序；serializer 只拼写已合法化的程序，不选择实现或临时创建 loop、scratch、packing 与参数。

## 8. Artifact 与调用边界

Runtime 只绑定已声明资源并执行任务，host-visible 调用返回前完成 join。初始化、编译、加载与普通调用分离；不执行 Python/Torch 算法替代品，也不增加隐藏 kernel invocation。

生成 source/IR 与 native materialization 分开，前者不等于可调用 artifact。Native 算子计时包含该次调用所需 packing、内部 materialization、任务派发与同步，排除编译、调优、加载及外部输出分配；实现覆盖、测量结果与性能门槛不进入本规格。
