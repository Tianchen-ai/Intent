# IntentDSL V2 产品架构

**状态**　V2-1 设计，交付评审。树形及挂接点已定；下表的 IR 合同、出口边界名和 ABI 是产品提案。仓库中的 `product.json` 目前只登记脚手架前端。本文的图描述目标产品，不代表已有可执行后端。

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
| block@formed | 父到子 lowering → 结构选择 | 已有结构化块计算、坐标映射与待分配的工作域；具有一个合法的、有限的参数行域 |
| block@decided | block 策略 → block realizer | 每个适用区域有一份可验证的完整决定；每个引用的参数在同一行中有定义 |
| block@realized | block realizer → dsa 或 block 规范化 | SSA 中已有实例到输出的映射、循环、片段、mask、累加与依赖；未兑现的结构决定为零。参数可在有限行域中取值 |
| block@normalized | block 公共规范化 → triton、cutile | 尾部、广播、标量/片段转换、循环参数和结果投影具有统一形式，GPU 语言出口可以直接消费 |
| cpu@formed | KIR 到值表示、CPU 分区家族 → task 构造 | 值、轴、读写与 carry 保留；已选的分区成为工作域关系；计算块不预设 SIMD 宽度，task 不绑定线程或 hart |
| cpu@tasked | CPU 分工 → weft 或 CPU 物化 | task captures、输出 owner、阶段完成与 join 明确；计算仍为值。需要共享准备、packing 或存储的值携带待兑现需求 |
| cpu@buffers | CPU 物化 → mojo | 必需存储有 owner、初始化、对齐、寿命及供数顺序。可保留为 SSA 的值继续为 SSA，线程由唯一调度方控制 |
| dsa@formed | block@realized 单次转换 → 供数规划 | 实例的覆盖和参数关系保留，显式形成参与者组、抽象存储可见域、生产/消费事件 |
| dsa@supplied | DSA 供数与寿命 realizer → bangc | 搬运及等待、计算与复用次序、容量与重叠约束可检查；物理地址和厂商存储类待出口绑定 |

## 出口合同

以下均为出口自己的有序边界，最后一个边界通过后才 emit。下游接收一份包含入口、参数及有限配置的产品产物；编译成功、加载、执行和数值正确分别观察。

| 出口与挂接点 | 私有边界及保证 | emit 的主载荷 → 实际消费者 |
|---|---|---|
| triton ← block@normalized | @legal：Triton 可表达的 grid、指针/值类型、掩码、原生归约等已选定；@callable：constexpr、Config 行与调用签名闭合 | 参数化 Triton Python JIT kernel → 本机 Triton GPU 编译器 |
| cutile ← block@normalized | @legal：已选择 cuTile 的 tile load/store、tile 计算和边界机制；@callable：原生签名、tile 常量和启动映射闭合 | CUDA Tile Python kernel → 本机 cuTile 工具链 |
| weft ← cpu@tasked | 进入首边界前，私有 realizer 在值程序中消费 CPU 存储需求；@legal：Canonical Weft 可表达整个 task 阶段与 join；@callable：参数和 CPU driver 启动合同齐备 | Canonical Weft `.weftir` → Triton–Weft 的 `triton.compile` 与 CPU launcher |
| mojo ← cpu@buffers | @legal：循环、SIMD 或实际矩阵实现、scratch 与任务 ABI 对应；@callable：导出函数符号和实参顺序固定 | Mojo 模块源 → 本机 Mojo 编译器与原生库加载器 |
| bangc ← dsa@supplied | @bound：选定微核合同、实际存储类和 scratch 需求，核对物理容量、SDK 保留量及行预算；@legal：兑现地址、packing、搬运、同步与 BANG C primitive；@callable：CNCC 入口和队列 ABI 完整 | BANG C 源 → 本机 CNCC、CNRT。微核承接有类型的计算区域，外层仍负责跨区域遍历与复用 |

下游的合法性受安装版本约束。缺少目标操作、工具链或有效行，诊断给出出口、位置和失败阶段；不得改变 dtype、成员集合或出口来绕过错误。

## 决策归属

「决定 → 兑现」是同一节点或出口的两个边界，可以同名。策略返回一个完整 MLIR 属性，realizer 写入真实程序并消费该属性。对相关联的参数只接受同一合法行。下表为 V2-1 推荐拆法；block 的承载 op 和约束见 [Block 设计](design/block.md)。

| 决定的所有者及对象 | 边界 | 读取的事实 → 唯一输出责任 |
|---|---|---|
| kir：等价及受许可重写 | imported → canonical 做无条件规范化；有条件重写由 execution 入口选用 | 作者数值模式、成员关系、舍入与 effects → 合法改写的匹配、证明条件与构造方法。是否值得应用由正在消费 KIR 的执行模型决定 |
| block：经验行域 | formed → formed | 结构、数值合同、适用 shape 类及出口表 → 一个有限联合参数域 |
| block：工作归属、访问分块、归约驻留/流式、fold/scan 分段、收缩分块 | decided → realized | 轴、依赖、外部访问、活跃值、同一参数域与目标能力 → 工作映射及真实循环、片段、状态和复用 |
| cpu：任务划分 | formed → formed | 输出映射、carry、共享读取、可用核集合与线程预算 → 生成分区关系；tasked 的必需 lowering 据此生成任务、捕获和 join，随后存储策略可以检查实际 task |
| cpu：共享准备与物化 | tasked → buffers | 值的寿命、实现输入表示、别名、工作集和 scratch 预算 → 必需的存储与供数。weft 在中途接出，必须自己兑现 |
| dsa：供数、搬运与复用 | formed → supplied | block 中已有的访问、参与者、可见性、目标容量及并发约束 → 存储寿命、事件与等待 |
| bangc：物理空间与微核 | bound → legal | MLU 资源事实、ABI、格式、scratch/尾部及指令支持 → 目标微核、地址和原生 primitive |
| 各语言出口：原生映射 | legal → callable，必要合法化也可用普通 pass | 已兑现的执行结构、下游版本和设备能力 → 目标指令级 DSL 操作、签名、行的原生参数 |

接出点早于决定时，家族尚未运行。接出点位于决定与兑现之间时，接收方须有私有 realizer，在首边界前兑现；缺失导致该分支及后代无效。等于兑现点时，接收方直接消费已兑现的程序。dsa 不经过 block@normalized，设计中不留跨越 realized 的 block 结构决定。

**CPU 的次序约束。** SDK 在一个边界内先运行必需 lowering，再运行 decide 和 materialize。任务分区放在 formed 同边界兑现；tasked 的必需 lowering 已能按分区建立 task，存储策略此时读取的确实是带 captures/join 的值程序。若把任务家族也延迟到 tasked 的 materialize，另一个家族的 decide 会先读到旧表示。weft 的私有 realizer 处理已决定的表示需求：需要共享准备时形成 Weft 能消费的显式准备/完成关系；可保持为值时保留结构化值。不能靠丢弃 pending 属性通过 verifier。

## 运行时与本机 JIT

| 单元 | 生产／消费的合同 | 运行时保证 |
|---|---|---|
| runtime/kir | 作者 signature、视图、shape 等式、调用图 → 绑定后的参数帧与序列 | 显式 backend；输入一致的 device；输出结构、alias/InOut、临时量和释放点由作者签名/编排决定 |
| runtime/block | block 的公开 ABI：实例域及有界 grid 公式、符号参数槽 → 具体尺寸与选中行对应的 grid | 一个逻辑实例的覆盖在编译期确定；运行时仅求值。triton、cutile 共享；bangc 可消费其通用实例映射 |
| runtime/triton、runtime/cutile | 各自出口 ABI、kir/block 的绑定 → 本机模块、编译选项、参数与当前 stream | 各自加载和 launch，尊重对应框架的图捕获、流及资源寿命 |
| runtime/weft、runtime/mojo | kir 参数帧与各自 CPU/task ABI → 函数、捕获、affinity 和 scratch | 每次调用只由一方派发任务，完成及 join 后返回；保留原生调用所需的所有权引用 |
| runtime/bangc | kir/block 合同、DSA 供数 ABI 及 bangc 的 launch/地址槽 → CNRT queue 调用 | 核对机器架构与容量，管理全局 workspace 和在途对象；片上地址在出口确定 |

树上只设上述七个 runtime。CPU/DSA 可发布仅含数据或类型的 ABI 头供后代读取。运行时之间可沿祖先共享辅助代码，各 runtime 不链接编译单元或 Core。

1. **定义与捕获**　`intent.kernel/fn/program` 记录定义、源码位置与闭包常量。安装包的薄 Python 入口持有 SDK Compiler；候选安装内的前端生成根 KIR 字节。捕获不查询设备。
2. **prepare**　给定定义、显式出口和本机 buffers，检查 shape/dtype/device/alias，绑定 constexpr 和输出，分配作者声明的 workspace。获取 device/CPU 集及工具链事实；由出口将快照绑定到 typed services。可做必要 Intent lowering；不执行、调优或同步设备。
3. **compile**　沿唯一父链作 Intent lowering 并 emit，若 prepare 已完成则复用。出口 runtime 在目标机编译适用行的原生版本并加载，返回已准备的调用。原生编译与首次调优有不同的失败阶段；不可把首次执行藏入 compile。
4. **launch**　首次需选择时，只在预先声明且守卫成立的行中测量，恢复 InOut、别名关系及累积初态；对真实用户状态提交一次选中实现。单一行无需试跑；无法隔离试跑外部 effects 的调用须选定安全单行或报告原因。GPU/MLU 按原 stream/queue 异步；CPU 完成后返回。捕获图前完成编译与选行，热调用不重新探测或读表。
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

## 产品边界与框架接口（待评审）

- SDK 的 `TargetServices`、`ExitImplementation.capabilities(options)`、`FamilyBinding` 足以登记上述单元。设备从运行调用取得，建议以无地址的能力快照派生并缓存目标 Compiler。打开产品库可能装配多个出口；能力回调须能处理尚未绑定设备，不能因选 Triton 而导入全部厂商 SDK。
- SDK 的 compile/emit 交换字节。V2 要定义并版本化自己的产物封套（入口源、公开 ABI、参数域及守卫），发射一次，normal JIT 与 Bench 共用。框架 configuration candidate、某次编译、一个 kernel 的 tuning row 是三个不同身份。
- SDK 不代替产品实现 provider-native build、调优、launch 或 workspace 寿命。包级装饰器、Torch 等外部对象到作者定义的桥接也由产品负责。若必须给 SDK 增加接口，先提交能说明缺口的调用序列供评审，不另建编译驱动。
- worker 的作者输入当前走字节接口，不能假设能跨进程取得任意 Python 闭包。建议薄入口把作者源码、引用的纯 helper 及可序列化常量组成一个输入包，位置保留到定义。安装内前端解析受支持的 Python AST；普通模型对象和 tensor 留在绑定端。直接提供根 IR 可复用同一 compile，语义许可必须已在根中显式化。
- v0.2 不负责决定多个候选是否等价、为不同调用解释 KIR 语义或隔离第三方代码的任意动态导入。V2 的跨单元 Python 调用使用候选加载器取得模块对象；第三方接受显式模块/限定名。若产品集成需要宿主预先绑定的裸名加载回调同样隔离，仍须向框架负责人提出。

## 本轮待裁决

| 选择 | 推荐方案 | 替代方案及代价 |
|---|---|---|
| 调优跨不跨控制拓扑 | 一份参数化程序只容纳一种拓扑，运行时只选完整行；不同拓扑由策略选择或派生比较 | 同时编译多个异构 body 交给运行时，可能找到更快 winner，但需额外的代码尺寸、冷编译和等价性合同 |
| 何时专化动态 extent | 首路线优先按真实上界/连续性类专化，所有假设都有 guard；用通用流式组兜住合法动态范围 | 按精确宽度编译，静态简化更多，缓存与首次编译次数增加 |
| 数值改写与模型采用的位置 | 在子节点开始降低前，调用 KIR 提供的可选变换与合法性条件，保留源路径 | 在 canonical 前为多个模型产生多份等价 body，增加公共 IR 大小及后续选择责任 |
| 两阶段调用的作者表面 | 静态可捕获的 program 声明 workspace 与调用，支持宿主常量/形状算术；普通 Python 可组合独立 JIT 调用 | 任意 Python 回调作为可编译 program，会妨碍独立 worker 重放及完整原生计时 |
