# IntentDSL V2 产品架构

**状态**　V2-2 已实现 Python → KIR → Block → Triton 及三个关联 runtime。下图同时给出已定的后续分支；CPU、Weft 与 Mojo 的边界细化见 [V2-3 CPU 设计](design/cpu.md)，目前处于设计评审阶段。

**设计依据**　CompilerInfra `3ddeef2` 的 `lib/Assembly.cpp`、`include/compilerinfra/Decision.h` 和安装包的 Python 绑定决定了本设计可用的阶段顺序、策略接口和候选加载行为。语义对照为原版 IntentDSL `b0a19990` 的 `doc/dsl/types-numerics-and-effects.md`、`doc/dsl/quantized-operations.md` 与 `python/intent/examples/registry.py`。V1 的 Triton 经验行见 `lib/Target/Triton/Transforms/Configuration/TuningProfiles.json`；V2 建议的行值须在 V2 的产物上重新判断。

## 编译对象与职责

| 生产者 | 交付给谁 | 独占的决定 |
|---|---|---|
| 作者 | Python 前端 | 逻辑成员及访问、并行或顺序、聚合代数、状态、数值精度、可见的 kernel 数量与调用顺序 |
| Intent | 所选下游编译器 | 根据真实类型、效应及出口能力，组织实例或任务、遍历、复用、片段和供数；把这些选择兑现成程序 |
| 下游工具链 | 出口 runtime | 线程/lane 到元素的内部布局、指令、最终寄存器分配和机器码；返回原生符号、实际资源及加载条件 |
| 用户／调用环境 | 本机 JIT | 显式指定出口；buffer 决定设备。CPU affinity、线程预算与硬件事实分开。流/队列来自调用上下文。 |

一个 `kernel` 是一个可调用的作者算法。一个 `program` 持有一组 kernel 符号以及有序调用、workspace 和输出关系。每次通过 SDK 编译一份根 IR 和一个出口；完整 program 依照作者调用图引用这些产物。本机设备探测和原生编译发生在执行机器上。agent 与普通 JIT 打开同一产品库，按同一 ABI 执行。

## 编译树

```text
frontend/python
  algorithm/kir             imported → canonical
    ├─ execution/block      formed → decided → realized → normalized
    │    ├─ block@normalized → triton
    │    ├─ block@normalized → cutile
    │    └─ block@realized → storage/dsa     formed → supplied
    │                             └─ dsa@supplied → bangc
    └─ execution/cpu        formed → tasked → buffers
         ├─ cpu@tasked → weft
         └─ cpu@buffers → mojo
```

两个 execution 节点都从 `kir@canonical` 接出。algorithm 级负责算法语义，execution 级负责计算和工作归属，storage 级负责显式存储、搬运与引擎依赖。目录中分别使用 `levels/<level>/<node>`、`exits/<exit>`、`runtime/<owner>`。边界是 verifier 接受的实际 IR；子节点的 lowering 只读取挂接点已有的表示。

| 节点@边界 | 谁生成／谁消费 | 通过 verifier 后的保证 |
|---|---|---|
| kir@imported | Python 捕获与根输入校验 → KIR 规范化 | kernel/纯函数符号、参数方向、标量/视图类型、逻辑轴、区域参数、yield 及源位置齐备；高层索引语法可暂留 |
| kir@canonical | KIR 规范化 → block 或 cpu | 索引、类型提升、广播、显式 cast 与 effects 唯一可读；有序依赖、空域 identity、数值许可和别名约束保真 |
| block@formed | 父到子 lowering → 结构选择 | 已有结构化块计算、坐标映射与待分配的工作域；推导唯一写、依赖、形状守卫和参数可行域；控制拓扑与参数行组尚未选定 |
| block@decided | block 策略 → block realizer | 同一结构家族联合选择控制拓扑与完整行组；每行对所有关联区域的结构参数赋值一致 |
| block@realized | block realizer → dsa 或 block 规范化 | SSA 中已有实例到输出的映射、循环、片段、mask、累加与依赖；未兑现的结构决定为零。参数可在有限行域中取值 |
| block@normalized | block 公共规范化 → triton、cutile | 尾部、广播、标量/片段转换、循环参数和结果投影具有统一形式，GPU 语言出口可以直接消费 |
| cpu@formed | KIR 到值表示、CPU 分区家族 → task 构造 | 值、轴、读写与 carry 保留；已选的分区成为工作域关系；计算块不预设 SIMD 宽度，task 不绑定线程或 hart |
| cpu@tasked | CPU 分工 → weft 或 CPU 物化 | task captures、输出 owner、阶段完成与 join 明确；计算仍为值。需要共享准备、packing 或存储的值携带待兑现需求 |
| cpu@buffers | CPU 物化 → mojo | 必需存储有 owner、初始化、对齐、寿命及供数顺序。可保留为 SSA 的值继续为 SSA，线程由唯一调度方控制 |
| dsa@formed | block@realized 单次转换 → 供数规划 | 保留实例覆盖及行 ID；选择一个共同供数拓扑，逐行检查抽象空间预算和事件条件，留下可兑现的行 |
| dsa@supplied | DSA 供数与寿命 realizer → bangc | 搬运及等待、计算与复用次序、容量与重叠约束可检查；物理地址和厂商存储类待出口绑定 |

## 出口合同

以下均为出口自己的有序边界，最后一个边界通过后才 emit。下游接收一份包含入口、参数及有限配置的产品产物；编译成功、加载、执行和数值正确分别观察。

| 出口与挂接点 | 私有边界及保证 | emit 的主载荷 → 实际消费者 |
|---|---|---|
| triton ← block@normalized | @legal：Triton 可表达的 grid、指针/值类型、掩码、原生归约等已选定；@callable：constexpr、Config 行与调用签名闭合 | 参数化 Triton Python JIT kernel → 本机 Triton GPU 编译器 |
| cutile ← block@normalized | @legal：已选择 cuTile 的 tile load/store、tile 计算和边界机制；@callable：原生签名、tile 常量和启动映射闭合 | CUDA Tile Python kernel → 本机 cuTile 工具链 |
| weft ← cpu@tasked | 进入首边界前，私有 realizer 在值程序中消费 CPU 存储需求；@legal：task 计算与 host 调用、join 分开；@callable：两侧签名、入口、完成点闭合 | 每个静态 task 定义一份 `.weftir`，各交给 `triton.compile`；另发射一份 host 程序，与这些 native kernel 绑定 |
| mojo ← cpu@buffers | @legal：选择目标循环、SIMD 或矩阵实现；@realized：终验展开后的计算、尾部、存储与线程完成点；@callable：固定已终验实现的导出函数与实参顺序 | Mojo 模块源 → 本机 Mojo 编译器与原生库加载器 |
| bangc ← dsa@supplied | @bound：选定微核合同、实际存储类和 scratch 需求，核对物理容量、SDK 保留量及行预算；@legal：兑现地址、packing、搬运、同步与 BANG C primitive；@callable：CNCC 入口和队列 ABI 完整 | BANG C 源 → 本机 CNCC、CNRT。微核承接有类型的计算区域，外层仍负责跨区域遍历与复用 |

下游的合法性受安装版本约束。缺少目标操作、工具链或有效行，诊断给出出口、位置和失败阶段；不得改变 dtype、成员集合或出口来绕过错误。

## 决策归属

「决定 → 兑现」是同一节点或出口的两个边界，可以同名。策略返回一个完整 MLIR 属性，realizer 写入真实程序并消费该属性。对相关联的参数只接受同一合法行。下表列出 V2-1b 的决定归属；block 的承载 op 和约束见 [Block 设计](design/block.md)。

| 决定的所有者及对象 | 边界 | 读取的事实 → 唯一输出责任 |
|---|---|---|
| kir：等价及受许可重写 | imported → canonical 做无条件规范化；有条件重写由 execution 入口选用 | 作者数值模式、成员关系、舍入与 effects → 合法改写的匹配、证明条件与构造方法。是否值得应用由正在消费 KIR 的执行模型决定 |
| block：结构拓扑与完整参数行（一个家族） | decided → realized | formed 提供覆盖、轴、依赖、活跃值和合法参数域；出口能力提供经验行的公共投影 → 联合选择归属、分块、驻留/流式、分段与收缩组织，在 IR 兑现 grid、循环、片段、状态及守卫；原生载荷按行 ID 留给出口 |
| cpu：任务划分 | formed → formed | 输出映射、carry、共享读取、可用核集合与线程预算 → 生成分区关系；tasked 的必需 lowering 据此生成任务、捕获和 join，随后存储策略可以检查实际 task |
| cpu：共享准备与物化 | tasked → buffers | 值的寿命、实现输入表示、别名、工作集和 scratch 预算 → 必需的存储与供数。weft 在中途接出，必须自己兑现 |
| dsa：供数、搬运与复用 | formed → supplied | block 中已有的访问、参与者、可见性、目标容量及并发约束 → 存储寿命、事件与等待 |
| bangc：物理空间与微核 | bound → legal | MLU 资源事实、ABI、格式、scratch/尾部及指令支持 → 目标微核、地址和原生 primitive |
| 语言出口：原生映射 | Weft/Triton/cuTile：legal → callable；Mojo：legal → realized，之后在 callable 绑定 ABI。必要合法化可用普通 pass | 已兑现的执行结构、下游版本和设备能力 → 目标 DSL 操作、签名、行的原生参数 |

接出点早于决定时，家族尚未运行。接出点位于决定与兑现之间时，接收方须有私有 realizer，在首边界前兑现；缺失导致该分支及后代无效。等于兑现点时，接收方直接消费已兑现的程序。本树只有 weft 位于决定区间内：`cpu@tasked` 已作存储决定，`cpu@buffers` 才兑现。dsa 从 `block@realized` 接出，此时所有 Block 结构决定已经兑现；DSA 直接接收参数化的实例、片段和循环。它在 `dsa@formed → @supplied` 作出的供数决定有自己的 realizer，无需为接出 Block 登记私有 realizer。

**CPU 的次序约束。** SDK 在一个边界内先运行必需 lowering，再运行 decide 和 materialize。任务分区放在 formed 同边界兑现；tasked 的必需 lowering 已能按分区建立 task，存储策略此时读取的确实是带 captures/join 的值程序。若把任务家族也延迟到 tasked 的 materialize，另一个家族的 decide 会先读到旧表示。weft 的私有 realizer 处理已决定的表示需求：需要共享准备时形成 Weft 能消费的显式准备/完成关系；可保持为值时保留结构化值。不能靠丢弃 pending 属性通过 verifier。

### Weft 的 host 与 task ABI

一个作者 kernel 在 `cpu@tasked` 可有多个 task 定义；同一 task 的不同工作区间复用同一份原生代码。weft 的私有 realizer 先落实该路径所需的共享准备与寿命。出口为每个定义发射一份 Canonical Weft 输入，各自经 `triton.compile` 得到一个 native kernel；另发射可原生构建的 host 程序。两侧在 `weft@callable` 核对下列合同。

| 接口 | 调用者提供 | 返回与所有权 |
|---|---|---|
| runtime → host | 已解析的 host 入口，按作者签名排列的视图和标量，CPU 执行集合、线程预算及有寿命的 workspace | runtime 加载已绑定 task 入口的 host 产物，调用 host 一次；host 在本次所有 task 完成后同步返回状态，错误沿同一入口交回 |
| host → task | 按 ABI 版本、符号与函数签名绑定的原生入口；本次任务的逻辑区间、captures（含基址、元素 offset、各轴 extent/元素 stride 和 dtype 的视图，或有类型的标量）、分配给该任务的输出切片和对齐的 scratch | 每次调用只处理它的工作区间。返回即该次原生计算完成；host 收集完成通知，在 join 后向依赖任务发布写入。scratch 和共享准备由 host 持有至最后一个消费者完成 |

例如第一阶段的多个 task 实例写互斥 partials，host 等它们全部完成后才调用读取 partials 的第二阶段。host 独占任务划分的派发、join、错误收束和最终完成点；失败时等待已发出的工作结束，再释放共享存储。作者 `program` 在更外层决定各个 kernel 的调用次序。CPU launcher 执行 host 指定的任务，`runtime/weft` 只加载并调用 host。

## 运行时与本机 JIT

| 单元 | 生产／消费的合同 | 运行时保证 |
|---|---|---|
| runtime/kir | 作者 signature、视图、shape 等式、调用图 → 绑定后的参数帧与序列 | 显式 backend；输入一致的 device；输出结构、alias/InOut、临时量和释放点由作者签名/编排决定 |
| runtime/block | block 的公开 ABI：实例域及有界 grid 公式、符号参数槽 → 具体尺寸与选中行对应的 grid | 一个逻辑实例的覆盖在编译期确定；运行时仅求值。triton、cutile 共享；bangc 可消费其通用实例映射 |
| runtime/triton、runtime/cutile | 各自出口 ABI、kir/block 的绑定 → 本机模块、编译选项、参数与当前 stream | 各自加载和 launch，尊重对应框架的图捕获、流及资源寿命 |
| runtime/weft | kir 参数帧与 weft host ABI → 已加载的 host 入口 | 每个作者调用进入 host 一次，持有实参至同步返回 |
| runtime/mojo | kir 参数帧与 mojo 的 CPU ABI → 函数、affinity 和 scratch | 遵循该入口声明的任务执行方式，持有资源至完成 |
| runtime/bangc | kir/block 合同、DSA 供数 ABI 及 bangc 的 launch/地址槽 → CNRT queue 调用 | 核对机器架构与容量，管理全局 workspace 和在途对象；片上地址在出口确定 |

树上只设上述七个 runtime。CPU/DSA 可发布仅含数据或类型的 ABI 头供后代读取。运行时之间可沿祖先共享辅助代码，各 runtime 不链接编译单元或 Core。

1. **定义与捕获**　`intent.kernel/fn/program` 记录定义、源码位置与闭包常量。安装包的薄 Python 入口持有 SDK Compiler；候选安装内的前端生成根 KIR 字节。捕获不查询设备。
2. **prepare**　给定定义、显式出口和本机 buffers，检查 shape/dtype/device/alias，绑定 constexpr 和输出，分配作者声明的 workspace。获取 device/CPU 集及工具链事实；由出口将快照绑定到 typed services。可做必要 Intent lowering；不执行、调优或同步设备。
3. **compile**　沿唯一父链作 Intent lowering 并 emit，若 prepare 已完成则复用。出口的 native builder 在目标机编译适用行；Weft 在此编译 task 和 host 并核对两者的链接签名。出口 runtime 加载可调用产物。原生编译与首次调优有不同的失败阶段；不可把首次执行藏入 compile。
4. **launch**　首次需选择时，只在同一控制和供数拓扑下、已声明且守卫成立的行中测量，恢复 InOut、别名关系及累积初态；对真实用户状态提交一次选中实现。单一行无需试跑；无法隔离试跑外部 effects 的调用须选定安全单行或报告原因。GPU/MLU 按原 stream/queue 异步；CPU 完成后返回。捕获图前完成编译与选行，热调用不重新探测或读表。
5. **缓存**　Intent/原生编译缓存以定义、出口、有效目标/工具链、数值选项、constexpr、必要的 shape/stride 类和配置版本区分。winner 独立按有效候选集合、设备及性能相关参数区分。绑定对象单独持有本次地址、输出与生命周期；更换 buffer 重新绑定。数据内容与指针地址不进入通用 IR。缓存在候选安装间按实际构建与配置隔离。
6. **开发与验证**　同一产品库支持配置派生和源码派生；普通 JIT 仅留编译 ID，显式观察或 development profile 才记录 IR 变化。Bench 拿完整 prepared call、输入及同设备成熟对照做数值和完整调用计时。Bench 不补 ABI、不替作者分解 program。

## 对 V1 七个问题的取舍

| 总纲问题 | V2 的防线 | 后续核对位置 |
|---|---|---|
| 1 两个 GPU 出口重复组织程序 | 公共访问、广播、尾部及循环形态在 block@normalized 完成；私有合法化只选择目标 primitive | 对比两个出口的输入检查点 |
| 2 CPU 提前 bufferize | cpu@tasked 保留值和 task；Weft 接收方有私有 realizer，Mojo 走完整 buffers 边界 | 同一根程序的两个 CPU 挂点 |
| 3 按候选重复 KIR/Block、厂商内存上浮 | dsa 只消费一份带参数的 realized Block；容量事实由出口服务提供，物理存储类及微核只在 bangc 出口出现 | DSA 路径、真实 IR 与地址校验 |
| 4 大 pass 再装配 pipeline | 必需 lowering 在对应单元，独立可选改写挂具名边界；相互依赖的取舍用一个完整策略值 | 真实登记位置与 explain |
| 5 数值改写与收益判断混在一起 | KIR 定义变换及出处，block/cpu 接收 KIR 时依据目标和成本调用其接口，然后降低；禁用仍有合法源路径 | 比较相同数值模式下的 KIR 输入与子节点输出 |
| 6 资源及候选散落 | 实际事实／调用预算／估算使用量分别建模。出口声明硬件事实与经验表，节点查询接口，runtime 只实测合法行 | 目标快照、参数行、选中行 |
| 7 范围漂移 | 本轮树包含五个出口；Ascend 不列为可选出口，不预建其单元 | 安装清单与产品登记 |

## 产品边界与框架接口

- SDK 的 `TargetServices`、`ExitImplementation.capabilities(options)`、`FamilyBinding` 足以登记上述单元。设备从运行调用取得，建议以无地址的能力快照派生并缓存目标 Compiler。打开产品库可能装配多个出口；能力回调须能处理尚未绑定设备，不能因选 Triton 而导入全部厂商 SDK。
- SDK 的 compile/emit 交换字节。V2 要定义并版本化自己的产物封套（入口源、公开 ABI、参数域及守卫），发射一次，normal JIT 与 Bench 共用。框架 configuration candidate、某次编译、一个 kernel 的 tuning row 是三个不同身份。
- SDK 不代替产品实现 provider-native build、调优、launch 或 workspace 寿命。包级装饰器、Torch 等外部对象到作者定义的桥接也由产品负责。若必须给 SDK 增加接口，先提交能说明缺口的调用序列供评审，不另建编译驱动。
- worker 的作者输入当前走字节接口，不能假设能跨进程取得任意 Python 闭包。建议薄入口把作者源码、引用的纯 helper 及可序列化常量组成一个输入包，位置保留到定义。安装内前端解析受支持的 Python AST；普通模型对象和 tensor 留在绑定端。直接提供根 IR 可复用同一 compile，语义许可必须已在根中显式化。
- v0.2 不负责决定多个候选是否等价、为不同调用解释 KIR 语义或隔离第三方代码的任意动态导入。V2 的跨单元 Python 调用使用候选加载器取得模块对象；第三方接受显式模块/限定名。若产品集成需要宿主预先绑定的裸名加载回调同样隔离，仍须向框架负责人提出。

## 已裁决的产品行为

| 事项 | 编译与执行合同 |
|---|---|
| 调优范围 | 一次编译选定一种控制拓扑；需显式供数的路径还固定供数与等待拓扑。runtime 仅比较组内完整行。比较两种拓扑时从同一父编译器派生，使用相同作者输入和数值合同。 |
| 动态宽度 | 以有 guard 的上界类专化，流式实现覆盖开放宽度。编译键只包含该实现实际依赖的 shape 类和 stride 类；命中时重新验证 guard。零宽度按作者的输出域、identity 和 epilogue 执行。 |
| 数值许可 | KIR 拥有同一套匹配、数值前提及替换构造；block/cpu 在入口根据事实决定是否采用，随后检查同一规则的结果。 |
| program | 前端静态捕获调用、fill、返回结构和 workspace 的使用关系。维度表达式运行时求值；支持多个输出。设备数据决定的控制流在 kernel 内表达。 |
