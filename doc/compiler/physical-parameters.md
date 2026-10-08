# Physical Parameters 与 Tuning

## 1. 参数不是旁表中的名字

Physical parameter是在target compile time绑定、直接参与当前GPU program的typed symbol。例如BM/BN/BK必须同时约束fragment types、program-space extent、loops、access coordinates与validity，而不是只出现在search declaration或generated variable name中。

每个parameter declaration至少定义：

- stable symbol；
- semantic role，例如logical ownership extent、reduction chunk、scan/region segment、group size或provider option；
- integer/bool/enum kind；
- finite candidate domain或可验证constraints；
- 它直接影响的types、operations与launch fields；
- target/provider legality requirements。

Physical parameter不进入KIR，也不成为作者DSL参数。

## 编译期候选数据输入

`intent.compile(..., tuning_config=path)`与`compile_shared_gpu(..., tuning_config=path)`接受有限JSON配置覆盖；CLI对应`--tuning-config <path>`。不传时读取所选provider随compiler分发的完整配置表。表与provider配置职责相邻，编译器在配置物化前读取一次；调表不需要重编译C++，但必须重新编译kernel artifact。launch与autotune不再读取该文件。

一行配置同时指定shared粒度偏好和provider选项，组织方式对应Triton的完整`Config`列表。覆盖文件使用当前所选provider的命名空间`triton`或`cutile`，包含已有结构family到完整配置行数组的映射。例如Triton配置：

```json
{
  "triton": {
    "contraction_narrow": [
      [128, 256, 64, 1, 128, 1, 8, 8, 3, 1, 0],
      [64, 128, 32, 1, 128, 1, 8, 4, 4, 1, 0]
    ]
  }
}
```

显式提供的family整组替换默认行；未提供的family继续使用默认数据。Compiler依据当前typed program的计算与遍历结构选择一个family，所有参数消费同一行，不能从各参数的独立family拼行。不存在按kernel名称、source或registry选择配置的规则，也不能在JSON中写条件或算法。文件不可读、未知字段/命名空间/family、错误列数/类型、超出列类型的值、重复行或空候选表直接诊断。

每行前七列依次为`ownership_m, ownership_n, reduction, reduction_outer, scan, traversal_workers, traversal_group`。它们是按role消费的相关粒度偏好，不是对任意shape强制生效的parameter binding：投影选择typed domain内不超过请求值的最大值，无此值时选择domain最小值；程序已固定的维度保持固定。轴关系与资源约束可以确定性地调整同一行的绑定或拒绝该行，不能追加新的候选。相同规则用于默认表和覆盖表。最终tuple明确保存实际值，不能把profile原值冒充最终binding。

Pointwise ownership若由当前reduce的轴关系证明为保留行轴，消费同行的`ownership_m`行预算；pointwise遍历与其它列轴继续消费对应的原role偏好。因此同一完整行能表达少量保留行与较大的遍历/reduction chunk，不依赖另加候选。该映射不重解释contraction自身的M/N，也不改logical axis或source顺序。

Triton行随后四列为`NUM_WARPS, NUM_STAGES, NUM_CTAS, USE_TENSOR_DESCRIPTOR`；cuTile随后五列为`CUTILE_CTAS, CUTILE_WORKER_WARPS, CUTILE_OCCUPANCY, CUTILE_ACCESS_FORM, CUTILE_LOAD_POLICY`。Boolean列使用0/1，其余列为正整数。Provider选项以正式`ParameterAttr`声明，同一行的shared/provider绑定沿既有`ConfigurationSetAttr`保持关联；shared阶段可以保存尚未被provider结构消费的选项。Provider逐完整行过滤可证明非法的组合，删除没有实际消费者的选项并去重；不把tile与launch、访存、pipeline选项再次相乘，也不按轴补端点或反推未提供的搭配。没有影响程序的stage差异可以归并，实际流水线的stage选择保持原行。cuTile resident数由同一行的CTA/occupancy绑定导出，不与其它行交换。若当前程序没有合法候选则编译失败，不换算法、不退回默认表。外部provider compiler继续负责其独有的机器资源约束。

## 2. Parameter expressions

Fragment shapes与compile-time loop steps可以使用由常量和physical parameters组成的typed integer expressions。Runtime logical extents仍是SSA values；二者不能用字符串混合。

典型关系包括：

```text
grid_m = ceil_div(M, BM)
m_coord = program_m * BM + range(0, BM)
m_valid = m_coord < M
fragment_type = fragment<f16, [BM, BK]>
```

这里M是runtime logical extent，BM/BK是compile-time physical parameters。Grid和validity显式连接两类values。

## 3. Structural decisions 与 parameter decisions

Shared passes先确定一份physical program structure，例如program mapping、loop nesting、value/access graph与structured-op skeleton。Parameters绑定该程序中的granularity和provider compile options。

不建立任意physical-program Cartesian product或图级structural autotuner。若一种选择会改变program structure且下层无法由参数自行形成，它由typed compiler pass作出并写入IR；若下层已把该选择作为compile-time config并能实测winner，Intent只提供合法parameter domain。

`num_warps`、`num_stages`、`num_ctas`虽然会驱动Triton TTGIR结构变化，仍属于Triton provider parameters：Intent提供与当前program兼容的经验配置和可证明constraints，由provider调优器选winner。Parameter domain是所提供配置列的取值范围，不授权生成这些列的笛卡尔积；每条完整经验配置最多物化一条候选，过滤与去重可以减少数量。

## 4. Candidate 与 instantiation

一个candidate是所有required physical/provider parameters的concrete binding。Candidate instantiation必须得到一份完整、合法的provider program；不存在“候选只改字符串、程序仍靠emitter解释”的状态。

Legality分两级：

1. Intent验证可证明约束，例如positive/static extent、shape关系、operation capability、grid限制、已知资源上限和provider surface requirements；
2. provider compiler验证其拥有的layout、register、shared-memory、instruction与pipeline约束。

Intent删除确定非法的候选，但不复制下层完整resource allocator。下层编译失败若来自无法预知的机器资源，可以作为该candidate无效；不能反向改变KIR或静默使用另一算法。

## 5. Provider autotuner

Provider autotuner对剩余candidates分别编译、benchmark并选择winner。Winner是runtime/tuning artifact，不写回canonical KIR或shared GPU IR，也不变成设备型号分支。

调优试跑不属于作者的一次可观察invocation。Runtime依据external view的读写方向和实际allocation alias关系维护trial state；需要原始内容的`InOut`、读写alias和atomic状态在每次试跑前保持同一初始内容，不能只在候选之间恢复。Trial参数保留shape、dtype、strides、offset与alias关系，不把每个view独立clone成互不alias的输入。Winner只对调用者参数执行一次；调优失败也不把试跑效果留在调用者状态中。

Triton路径应生成一份参数角色明确的kernel与真实`Config`集合，让Triton autotuner选择BM/BN/BK、num_warps、num_stages、num_ctas及被允许的provider forms。cuTile使用其实际支持的tuning入口；没有下层tuner时，provider runtime可以对同一已声明candidate set实测选择。

## 6. Provider form candidates

Pointer/descriptor、cuTile gather spelling等local form只有满足下列条件时才进入candidate surface：

- 两种形式实现同一shared GPU access/operation；
- 每种形式都能在serialization前独立legalize；
- 下层或provider runtime能真实编译和测量；
- 选择不会改变KIR algorithm、kernel数量或计时scope。

若form改变operands、types或control，它必须先成为current program中的provider-local extension/compile-time branch，不能只在serializer里用字符串条件切换。

## 7. Cache 与 specialization identity

Compiled-candidate identity至少包含：

- canonical kernel specialization与public ABI；
- runtime shape/dtype specialization key；
- selected provider与hardware target；
- concrete physical/provider parameter bindings；
- compiler/provider code identity与relevant options。

Autotune winner cache另外以runtime tuning key索引候选timings。Compiled artifact cache与winner cache是不同层次，不能把winner写进IR冒充编译决定。

Intent 的编译产物默认保存到 `$XDG_CACHE_HOME/intentdsl/`，未设置 XDG 路径时使用 `~/.cache/intentdsl/`；`INTENT_CACHE_DIR` 可指定根目录。每项保存输入 KIR、目标 IR、provider source 与接口 metadata，`intent.generate` 和 `intent.compile` 的结果通过 `cache_directory` 暴露其位置。相同编译输入复用生成结果；compiler 文件身份、target/capability options、默认 profiles 或显式 tuning 配置内容改变时不复用旧项。并发请求只在相同 cache entry 上串行，成功产物完整后才发布；失败保留输入与诊断，但不作为成功结果复用。

这层缓存对应 Intent 的 source/IR 编译调用。Triton、cuTile 等下层仍使用各自的原生 JIT cache，负责 runtime specialization、具体候选与 native binary 的身份和复用；实验目录不另存生成的源码或中间 IR。

## 8. Baseline 对照边界

比较generated与手写provider source的算子性能时，以算法、输入shape、外部dtype与明确的调用和计时范围为基础。双方可各自调优，不要求先对齐完整candidate集合或穷举全部候选才允许测量。舍入位置、近似数学和中间精度的细微差异应注明，不一概阻断同算法的性能比较；这不放松compiler对Intent语义的保持要求。

若进一步把性能差距归因于compiler program quality，则需要控制双方相关parameter roles、候选预算和计时范围，或明确说明这些因素的影响。相同调优winner不是比较目标，候选调优耗时也不是算子执行时间；ABI表示、辅助输出或布局转换的差异应明确计入或排除相应测量范围，不隐藏额外工作。

Tuning只选择参数和已声明local forms，不能掩盖缺失的program mapping、access graph或structured realization。若generated依赖更大的search space才弥补结构缺口，该问题仍属于compiler IR/passes。

## 9. DSA完整配置

BANG C采用相同的完整经验列表原则，使用其独立执行模型的参数，不复用GPU参数角色。默认表位于provider配置模块；`tuning_config`的`bangc.block`整组替换列表。每行依次为`tile, tile_m, tile_n, tile_k, region_tile, tasks, local_bytes`，由既有`dsa.ConfigurationAttr`验证。配置不作为hardware target身份或作者DSL参数；`local_bytes`是配置使用的存储预算，不改变所选architecture的实际容量。

每行从同一canonical KIR独立形成一份完整、已绑定的DSA物理程序。标准nested ModuleOp保存这些候选，各候选沿原shared/provider passes和verifiers处理；公共ABI、数值许可和architecture保持一致。Source范围、entry与候选配置由serializer机械发布，runtime不根据配置重建程序。

单个合法候选直接调用；多个候选分别native编译、真实计时择优并缓存。试跑按allocation owner保存初始内容和alias关系，保留view的dtype、offset和strides，每次计时前恢复状态；winner只对原调用参数执行一次。编译失败记录到对应候选，全部无法编译时诊断；执行失败保留原错误并终止，不靠重试其它候选掩盖错误。完整配置列表也不能掩盖DSA实现或供数结构的缺失。
