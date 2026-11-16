# Intent Compiler

## 1. 编译对象

Intent compiler 的输入是已经完成 specialization、surface desugaring 与 canonical verification 的 Kernel IR。KIR 是作者算法的唯一权威；compiler 不根据 kernel 名称选择模板，也不把一个 kernel 自动拆成多个 launches。

compiler 的职责是从这份算法建立所选 execution family 可执行的 physical program：

```text
canonical Intent KIR
    ├─ Block executable program
    │    ├─ GPU mapping → Triton / cuTile
    │    ├─ Ascend mapping → Triton–Ascend
    │    └─ DSA storage / engine realization → BANG C
    └─ CPU executable program
         ├─ Mojo legalization / serialization
         └─ Canonical Weft IR → Triton–Weft JIT
```

Block、CPU 与 DSA 复用 KIR semantics 及 shape、index、effect、dependence、reuse 等分析接口。设备身份、执行模型与编译 provider 独立，具体边界见[执行模型与组件](execution-models.md)。Mojo 与 Triton–Weft 消费共同 CPU 程序的不同降低阶段；RVV 是 CPU 硬件能力，不是另一套 Intent execution family。

## 2. Block 编译边界

Block pipeline 必须产生一份共同、完整、唯一的 executable block program。它显式包含：

- program-instance space 与 logical workset mapping；
- program 内 structured control、physical loops 与 loop carries；
- scalar/fragment SSA、accumulator、materialization 与 buffer lifetimes；
- compositional coordinate provenance、predicate/index sets、access coordinates、validity、fill、collision 与 atomic effects；
- physical reduce、scan、region fold/scan、contract、scaled/sparse contract 与其它 local structured operations；
- compile-time physical parameters对types、loops、accesses与launch的直接约束。

任何影响执行的决定都必须成为当前 IR 的 type、operation、region、operand、result、attribute或def-use。Analysis cache、diagnostic origin和search declaration可以独立存在，但不能与executable function共同解释程序。Coordinate provenance由canonical KIR relation派生后，必须物化为current physical coordinate/access mappings，不能只存在origin side table。

## 3. 与 Triton 两层 IR 的关系

Triton TTIR 已经是一份作者写好的GPU block program；TTGIR进一步加入distributed encodings、layout conversions、MMA/shared/TMEM、pipeline与vendor extensions。Intent KIR比TTIR更高，因为Intent作者没有预写program id、static block tensors与pointer/mask program。

因此Intent需要补出与高质量provider source同等完整的block program，但不复制TTGIR或NVIDIA/AMD lower compiler：

```text
Intent KIR
    → shared executable GPU IR
    → provider source
    → external provider frontend/lower compiler
```

共同GPU IR不包含warp/lane distribution、register/shared/TMEM layout、MMA instruction、TMA lowering、software-pipeline schedule或machine ISA。它保留足以让provider compiler继续完成这些工作的program、fragment、access与structured-operation facts。

## 4. 术语

- **execution model**：Block、CPU task 与 DSA storage/engine 等具有明确执行不变量的程序表示；它不等于设备类别或 provider，同一设备可以由不同 provider 在不同降低深度接管；
- **physical program structure**：program-space mapping、loop nesting、control、value/access graph、structured-operation skeleton 与 resource ownership 的总和；
- **execution group**：同一个 physical kernel body 内共同拥有一组 connected computations/effects 的内部 dispatch region；它不是第二个 kernel、artifact 或 launch；
- **logical ownership extent**：一个 physical program instance 负责的 logical instances 集合或连续范围；
- **provider form**：同一共同 operation/access 在某个 source provider 中的可选表示，例如 Triton pointer access 与 tensor descriptor access；
- **target artifact**：外部 provider compiler 由一个 physical kernel specialization 产生、可由 host 启动一次的 executable entry；
- **target capability/resource limit**：所选 provider 与 hardware target 可用的 operation、dtype、grid 和资源约束；它们参与 legality，不进入作者 KIR。

## 5. Provider 与 hardware target

Triton、cuTile是GPU source providers；NVIDIA/AMD及SM/gfx版本是hardware targets。两者是正交维度。

CPU 同样区分 Mojo/Weft provider 与 x86/RISC-V hardware。共同 CPU IR 保留 task/block 与 structured compute；target lowering 可查询实现需求、选择并实例化专家编写的微程序，而非仅做 API 映射或调用整算子库旁路。具体向量宽度、AMX/IME 等能力参与实现合法性，职责与参数边界由 [CPU 规格](cpu-program-ir.md)统一定义。

Provider surface若只是API spelling不同，直接从共同GPU IR确定性序列化。只有共同IR无法无损表达、且需要多个passes或独立legality的真实target-local structure，才增加extension operations。Extension扩展同一当前program，不复制一份完整leaf program。

SM90、SM100、SM120或不同gfx版本不建立独立IR。Physical module携带target capability；passes、verifiers与lowering patterns按feature predicates形成不同program。作者KIR不包含device/provider分支。

## 6. 唯一 executable authority

KIR在physical construction期间保持immutable。Conversion产生所选family的独立physical program后，后续passes只改当前IR；provider lowering和serializer不得回到KIR、shape metadata或role名称重建ownership、range、access、validity、workspace或control。

KIR origin只用于语义保持验证、诊断和追踪。Physical program本身必须能够独立verify与serialize；执行不依赖旁边的KIR clone或decision records。

## 7. 规格组成

- [`execution-models.md`](execution-models.md)：设备、模型、目标实现、资源与扩充责任；
- [`kir-to-block.md`](kir-to-block.md)：KIR authority、logical worksets与第一份完整Block program；
- [`block-program-ir.md`](block-program-ir.md)：共同executable Block IR的types、operations与invariants；
- [`passes-and-analyses.md`](passes-and-analyses.md)：analysis、transformation、verification与semantic-preservation；
- [`physical-parameters.md`](physical-parameters.md)：compile-time physical parameters、candidate legality与下层tuning；
- [`target-lowering.md`](target-lowering.md)：provider extensions、architecture features、serialization与外部compiler边界。
- [`cpu-program-ir.md`](cpu-program-ir.md)：CPU task/block 编程模型、共同 passes、可编程目标实现与 Mojo/Weft compiler 边界。
