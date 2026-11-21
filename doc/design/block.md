# Block 表示、Triton 出口与首次调用

**状态**　V2-1 待审设计。第一条路径是 `kir@canonical → block@formed → @decided → @realized → @normalized → triton@legal → @callable → emit`。本文件给出不变量和实现责任。表中的配置值是设计示例，尚无本版真机成绩。

## 表示合同

| 对象 | 谁生产 → 谁消费 | 在 IR 中可检查的性质 |
|---|---|---|
| program / instance domain | 父到子 lowering 建立逻辑工作域；工作归属 realizer 建立具体 grid → 出口及 runtime/block | grid 是 shape、constexpr 和选中行的纯整数函数；同一实例标识在程序和 launch 中含义相同。对唯一写，活跃实例输出构成不交叠覆盖；原子/合并写以对应 effects 证明合法 |
| coordinate / relation | KIR 的域、slice、间接索引 → block 访问与遍历 | 清楚区分源坐标、实例原点、片段内坐标、线性地址；slice 保留原轴偏移。访问组合 stride/offset 只出现一次；有证明才收窄地址算术 |
| fragment | 结构化运算/访问 realizer → 算术、collective、store | SSA 值具有元素类型、逻辑轴及片段范围；标量归约结果可按显式轴广播。相同元素数量不意味着相同成员关系。warp/lane 分配由下游决定 |
| bounded access | relation + active domain → load/store/gather/atomic legalize | tail mask 与作者 predicate 取交；inactive read 的 fill 对应实际消费者（sum 与 max 不同），inactive write 不发出；masked-off 地址无需解引用。NaN、Inf 不能靠乘零掩盖 |
| buffer / accumulator / loop | KIR 局部状态或 block 复用决定 → 后续 block/dsa lowering | 有作用域、所有者、初始化、更新次序及结束点。普通 SSA 值可以一直是值。loop 的 carry/yield 与 dtype 固定；存储到寄存器、共享空间还是本地 scratch 属具体兑现责任 |
| config domain / parameter | 出口提供有类型的完整经验行，block 选择适用域 → block program 和出口 | 行的身份是相关选择的最小单位。实际遍历、坐标、片段和传输必须用到它的参数；检查点仅凭当前模块及声明的入口 ABI 即可继续降低 |

### 四个边界

1. **formed**　必需 lowering 消费 canonical KIR，建立 `block.program`、逻辑输出域、坐标关系与结构化 reduce/scan/contract。作者的 ordered loop、mask、state 和 effects 仍在程序中。模型可在消费 KIR 前选择调用 KIR 的受许可模式；没有许可则使用源结构。到此边界时，配置域家族已经选择并兑现一个行集合，工作归属仍是逻辑的。
2. **decided**　策略在待调度的有类型 op/region 上写入决定，包含选择的方法、参数引用及适用条件。行域、轴和 guard 必须一致，不能先固定 tile，再用另一行的 warp 数“修好”资源预算。联合合法性覆盖所有剩余行；报告空集或明确不支持的技术条件。
3. **realized**　决定变为实例映射、具体遍历/循环、片段类型、valid access、累加与供数。编译期参数仍可来自有限行域；不得残留待读取的 DecisionFamily carrier。可用此检查点直接建立 DSA 表示；无需先转 GPU 语言。
4. **normalized**　完成两个 GPU 语言出口共同需要的规范化：显式尾部有效域和 fill、仿射坐标与 stride 组合、标量/片段广播、静态轴顺序、循环携带状态和结果投影、可证明安全的冗余 load 消除。保持 dtype/NaN/effects。这里只定共同形态；Triton 的 power-of-two 片段、cuTile 专有 tile 和厂商指令在各自出口处理。

## 可登记的决策家族

节点内普通等价清理使用 MLIR pass。下面的家族才需要 Strategy 选择与 Realizer。决定通过领域 op 上的单个结构化属性读写；实现策略名和行 ID 不可成为匹配某个作者程序的依据。

| 家族／承载对象 | 决定 → 兑现 | 输入事实 | 实际变换与复用 |
|---|---|---|---|
| 有限配置域／program | formed → formed | 输出域类别、访问和规约/收缩签名、数值许可、shape 类、出口能力及经验表 | 选一组合法完整行及一个共享的参数 schema，建立 typed config-domain。无可调参数时为合法单行。它是为后续家族绑定参数的前置条件 |
| 工作归属／结果域 | decided → realized | 唯一写或合并 effects、行间独立性、工作量及行内 rows_per_instance | 形成 grid、实例原点和每实例输出行数，多行末实例按真实 extent 裁剪。适用 softmax、norm、逐行投影 |
| 访问分块／成员 relation | decided → realized | 共用坐标、stride、连续性、间接映射、tail、行的访问宽度 | 兑现每次读取的成员、掩码及普通遍历步长；负责地址计算，计算域的专有归约/收缩轴由其家族负责 |
| 归约驻留或流式／reduce 区域 | decided → realized | axis 长度类、归约代数、消费者数量、活跃字节、可用 collective 及完整行预算 | 可保留一行片段做多次运算；或沿约简轴多次遍历，以局部摘要更新状态。softmax、RMSNorm 无须同一个 pass 对整个函数做名称匹配 |
| fold/scan 分段／region op | decided → realized | 有序源轴、summarize/combine/apply/emit 合同、identity、状态大小、有限段宽 | 生成连续段、summary carry、前缀与 emit，必要时保留串行段循环；不跨独立实例假设全局 barrier。适用线性 attention、显式 summary 的 recurrence |
| 收缩分块／contract op | decided → realized | M/N/K 与 batch 关系、稀疏/量化格式、acc 类型、packing 及目标 primitive 要求 | 形成自由轴分块、K 遍历、累加与最终 epilogue。与输出归属共同引用同一行，不能再有第二组 M/N/K 参数 |

同一 op 同时涉及多个约束时，联合候选包含整个相关向量。例如某行给出 M/N/K、供数粒度、原生选项，需要同时满足工作集及 primitive 的要求。归属、访问、计算各写自己的结构；共享的是同一行引用与同一坐标分析。会互相废掉决定的改写必须合为同一 realizer，或用独立锚点并保留其他 carrier。不得依赖同一边界各家族恰好按某个名字先后运行。

**与 SDK 的次序吻合。** 同一边界先运行必需 lowering、贡献，再做选择和物化，最后验 IR。配置域在 formed 就完成物化，decided 的策略可读到它。realized 的结构改写消费上一边界已完成的决定。出口接收时只剩普通、可继续编译的产品 IR。

## 配置不是六个调参循环

**答复评审问题：允许 IR 保留多个 tuning 行；每个决策家族仍只做一次结构选择。** 提议用一份参数化程序配一个有界参数域。配置行与结构策略分两步工作。

| 时刻 | 生产者 → 消费者 | 唯一事实来源与拒绝条件 |
|---|---|---|
| 声明 | 出口组件 → typed capability service | 经验表按架构、计算/访问结构、dtype、适用范围登记。每一行由共同的结构列和一个有类型的原生选项元组组成；列名、单位、合法取值由 schema 定义 |
| 选择域 | formed 的 config 策略 → config-domain realizer | 先选适用于当前执行结构的表组，再作整行过滤和完全等价的去重。已知不合法的行删除；依赖运行时值的条件保留为纯 guard。不同拓扑分到不同组，一次编译选一组 |
| 形成程序 | block 策略/realizer → realized 检查点 | 参数用 SSA 或可折叠的有类型参数 op 表达。例如 `row0 = instance_x * row.rows`、`col = pass_index * row.width + offset`。每个合法行代入后产生闭合程序，原生选项也来自该行。IR 仅保存表的选中子集，不在旁边维护第二份计划 |
| 发射 | triton/cutile 等出口 → native builder | 一次输出参数化源码、row id→constexpr/原生选项的投影、ABI、grid 表达式和 guard。不同候选的外部实参、计算结果及作者调用数相同。原生编译可按行特化同一份源码 |
| 选择 winner | 出口 runtime → 热 launch | prepare 依据真实绑定筛除不适用行；compile 生成适用原生版本。首次 launch 试跑已声明的行，按原调用语义恢复副作用，仅提交一次 winner。缓存以硬件、工具链、数值合同、实参性能类及**完整候选集合**区分 |

**行间差异的限度。** 可改变每实例行数、片段宽度、循环步长及原生 launch 选项；所有行共享控制拓扑、effect 次序和 ABI。例如，单片段驻留组要求对每个有效行满足 `N <= width`；流式组的所有行允许有界循环。第一版不在一张表中放“额外一次 launch”“换算法”或任意多份完整 IR。需要另一种拓扑时，由 typed 策略按当前上下文选择另一组；agent 可派生编译器比较两种策略。不会为每个行组合重走一遍 KIR → block → dsa。DSA 直接将参数化 Block 转为参数化供数结构，目标原生特化仍归下游。

**形状与失败。** 实参的逻辑维度仍是动态 SSA。若选择驻留路径必须知道 N 的上界，prepare 可产生可验证的 shape 类（如 N 在 513–1024）；该约束进入编译输入或显式 specialization 及缓存键。未绑定上界时选择合法的流式组。守卫失败的配置不得发射用户调用；合法行皆为空时给出失败事实与范围。用户替换整组候选仍过相同的合法性检查。不得进行无界 sweep 或自由拼列。

**检查点切片（语义示意，尚非解析器语法）。** 下列是驻留组的一份程序。行参数在原生特化时是常量，R、N、offset 和 strides 仍取本次绑定。第二阶段的 max 和 sum 各有自己的有效域，不能把用于 max 的填充值直接计入 sum。

```text
config domain C     r0=(rows:1,width:1024,warps:4,stages:1)
                    r1=(rows:2,width:1024,warps:8,stages:1)
instance domain     [0, ceildiv(R, C.rows))
row, col            pid * C.rows + range(C.rows), range(C.width)
active              (row < R) and (col < N)                 // outer product
v                   read(x, offset + row*s0 + col*s1, active) → f32 fragment
peak                reduce.max_num(v, over=col, valid=active, identity=-inf)
w                   exp(v - broadcast_col(peak)) on active members
total               reduce.add(w, over=col, valid=active, identity=0)
write               y[row,col] = narrow_f16(w / broadcast_col(total)) on active
```

例如选择 r1 且 R=3、N=1000，grid 为 2，最后一实例只有一行可写。将相同 row id 投影为两组分离的调用参数：编译器消费 `ROWS=2, WIDTH=1024`，Triton 消费 `num_warps=8, num_stages=1`。artifact 对同一 row id 引用这两组值；不允许单独覆盖某个投影然后沿用旧 grid 或旧 winner。

### 两个首路线 kernel 如何消费同一份知识

稳定 softmax 的驻留路径：实例覆盖 r 行 → masked read 并升到 f32 → 沿列求 max_num → 同一片段计算 exp、sum → masked write 并窄化。一条流式路径可以重读输入，按原式分别求最大值、分母、输出；填充成员必须从 sum/max 的成员集中消失或使用相应 identity。它不能仅凭“适用流式”改成需要额外数值许可的在线合并。RMSNorm 复用同一行映射、tail 和驻留/重读分析，但计算是 sum-of-squares，再乘运行时 inv_c、加 eps、rsqrt、乘权并写回。两者均由结构识别；gain 的列广播和别名约束是 RMSNorm 的额外输入事实。

## Triton 出口

**能力合同。** 本机调用解析设备 ordinal、架构、warp 事实、每块资源上限、工具链版本和所需特性。未探测到的可选能力为 unknown；不得填入另一代 GPU 的数值。上限、调用预算和 pass 的估算分别记录。经验表不以实时 SM 空闲率、可用显存或历史赢家作为硬件事实。链接时无设备的声明查询与真实调用时绑定快照应可区分。

**表的形式。** 下表演示两项独立登记如何服务同一个“列连续、f32 累加、最多 1024 列、行内驻留”的结构；元组顺序为 (rows_per_instance, width, num_warps, num_stages)。这些是待评审的初始行，未声称达到目标性能。

| 目标键 | 完整候选（示意） | 共同核验 |
|---|---|---|
| Hopper / sm90（H100） | (1,1024,4,1)，(2,1024,8,1) | 支持 dtype/op；width 与 N、tail 及 Triton 的静态片段约束相容；整行工作集符合目标预算 |
| Blackwell / sm120（5090D） | (1,1024,4,1)，(2,1024,4,1) | 使用本机 sm120 的真实上限独立过滤；不能从 H100 的估算推出本机寄存器分配或 winner |

两表允许有相同的行；是否需要不同的映射由目标事实和真实运行判断。普通 strided 访问不套用连续访问表的假设。架构专有 descriptor 等只在提供相应能力、符合访问约束且有完整登记行时选择。

- **triton@legal** 接收 normalized Block，为已形成的访问、广播、归约、循环和收缩选择等价的 Triton 表示。显式处理 predication、最大值的 NaN 规则、累加类型和指针步长。若原生 primitive 合同不符，在此合法化或报告不支持。
- **triton@callable** 固定 kernel 名/入口、运行时参数顺序、constexpr 槽、候选选项、grid 输出维数、设备兼容条件。验证每个 row id 可还原唯一 constexpr 与原生 config，grid 和源码使用相同的分块常量。
- **emit** 只拼写已形成的结构。原生工具链可以获得 `triton.Config` 形式的有限候选；产品为同一入口指定唯一的选行所有者。首选 runtime/triton 驱动显式编译与计时，调用 Triton 的 JIT/原生 launcher；若复用其 autotuner，必须能履行相同的 prepare/compile/恢复合同，不得叠第二层 autotune。
- **分工实例** 输入一行、N=1000，用 width=1024 的行实例时只读写 1000 个成员；当 R 不是 rows_per_instance 的倍数时再屏蔽末实例中的整行。下游决定向量元素落在哪个 warp/lane 和实际指令，Intent 的 IR 中应能读出实例覆盖与谓词。

## 运行时交接

| 公开合同与生产者 | 谁读取 | 数据与不变量 |
|---|---|---|
| kir 单元的 KernelSignature / program 序列 | runtime/kir、各出口绑定器 | 参数方向、dtype、rank、shape 等式、元素 stride/offset、constexpr、别名及返回结构、依赖和 workspace 寿命。IR 中无运行时地址。prepare 由本次 buffer 建立 address frame 并保持所有权 |
| block 单元的 InstanceMapping | runtime/block 与继承该接口的出口 runtime | grid(rank≤目标可表达范围)、每轴 extent 的纯整数公式、符号槽与 row id；参数溢出、负维度、零工作分开处理。零输出可免 launch；零约简维且有输出仍须生成 identity 或执行 epilogue |
| triton 单元的 CallableABI | runtime/triton | 源入口、ABI 版本、指针/标量及 constexpr 对应、有效行和原生选项、stream/device、目标工具链识别信息。运行时不能猜字段、补循环或改写源文件 |
| native handle / prepared call | JIT 调用者、独立 Bench | prepare 仅绑定，必要时 lower；compile 编译原生行但不计时；launch 选行并提交；result 暴露原输出容器。已绑定对象持有参数、workspace、原生模块和队列依赖 |

**首路线实施的停止点。** 安装目录外通过 `jit(..., backend="triton")` 运行两个作者程序；能分别在 formed、decided、realized、normalized 和出口检查点读出同一结构的变化。用独立 Bench 在 5090D、H100 测数值、冷编译、首次选行和热调用。上述均属 V2-2，设计文档没有把它们记作已验证。
