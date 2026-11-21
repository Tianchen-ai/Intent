# Block 表示、Triton 出口与首次调用

**状态**　V2-1b 设计裁决。第一条路径是 `kir@canonical → block@formed → @decided → @realized → @normalized → triton@legal → @callable → emit`。本文件给出不变量和实现责任。表中的配置值是设计示例，尚无本版真机成绩。

## 表示合同

| 对象 | 谁生产 → 谁消费 | 在 IR 中可检查的性质 |
|---|---|---|
| program / instance domain | 父到子 lowering 建立逻辑工作域；联合结构 realizer 按归属建立具体 grid → 出口及 runtime/block | grid 是 shape、constexpr 和选中行的纯整数函数；同一实例标识在程序和 launch 中含义相同。对唯一写，活跃实例输出构成不交叠覆盖；原子/合并写以对应 effects 证明合法 |
| coordinate / relation | KIR 的域、slice、间接索引 → block 访问与遍历 | 清楚区分源坐标、实例原点、片段内坐标、线性地址；slice 保留原轴偏移。访问组合 stride/offset 只出现一次；有证明才收窄地址算术 |
| fragment | 结构化运算/访问 realizer → 算术、collective、store | SSA 值具有元素类型、逻辑轴及片段范围；标量归约结果可按显式轴广播。相同元素数量不意味着相同成员关系。warp/lane 分配由下游决定 |
| bounded access | relation + active domain → load/store/gather/atomic legalize | 实际访存限于逻辑成员与作者 predicate 的交集。逻辑成员上 predicate 为假时，各消费者读同一个作者 fill。超出逻辑域的物理 lane 不贡献，可在各自 collective 入口分别用 identity 表示；两种屏蔽都不解引用无效地址 |
| buffer / accumulator / loop | KIR 局部状态或 block 复用决定 → 后续 block/dsa lowering | 有作用域、所有者、初始化、更新次序及结束点。普通 SSA 值可以一直是值。loop 的 carry/yield 与 dtype 固定；存储到寄存器、共享空间还是本地 scratch 属具体兑现责任 |
| config domain / parameter | 出口提供整行的公共投影，block 联合选择拓扑与行组 → block program 和出口 | 行的身份是相关选择的最小单位。实际遍历、坐标、片段和传输必须用到它的参数；检查点仅凭当前模块及声明的入口 ABI 即可继续降低 |

### 四个边界

1. **formed**　必需 lowering 建立逻辑实例域、访问关系、结构化计算、状态与 effects。分析可推导连续性、有效成员、驻留所需空间、可容纳的宽度和合法参数域；这些事实允许多种拓扑同时可行。此处没有选中的拓扑、参数行或配置家族。
2. **decided**　结构策略针对同一程序的相关区域一次选出控制拓扑及该拓扑的完整行组。整份 kernel 程序有一个联合决定锚点，涵盖相关的有类型区域。各区域可有不同计算结构，但行 ID 在整份程序中取同一赋值；共享坐标、状态和资源的操作须共同满足这份决定。每行覆盖全部关联参数，守卫可求值，行组非空。
3. **realized**　联合 realizer 把决定变为实例映射、循环、片段、有效访问、累加和依赖。参数引用已进入相应操作；保留参数域、行 ID 与守卫，待兑现的结构决定为零。DSA 在这里接收当前程序。
4. **normalized**　两个 GPU 语言出口共用的规范形式包括尾部有效域、坐标与 stride 的组合、标量广播、循环 carry 和结果投影。消除冗余读须满足 alias/effect 条件，保留作者 fill、dtype 和 NaN 语义。各出口负责其原生 tile 和指令约束。

## 联合决策家族

| 登记家族 | 决定 → 物化 | 读取的事实 | 写入的 IR |
|---|---|---|---|
| Block 结构拓扑与完整参数行 | `block@decided → block@realized` | 输出所有权与 effects、轴/stride/有效域、归约代数、fold/scan 的摘要与顺序、收缩轴与累加精度、活跃值、shape guard、出口能力和经验行的公共列 | 决定时写一种控制拓扑和一组相关参数赋值。物化为实例/grid 及每实例行数、访问分块和掩码、驻留或流式循环、分段和 carry、M/N/K 分块及 epilogue，保留有类型的行参数引用 |

工作归属、访问、归约、分段和收缩是同一决定的组成部分。比如 attention 的 K 遍历同时约束打分收缩、在线状态和权重乘 V，策略须为三个消费者选出相容的分段和尺寸。realizer 可分工构造这些操作；任何一步都不能重新选行或改拓扑。

**混合区域如何扩充。** 对同一段 attention，收缩组件从 QK 和 PV 提取轴、输入表示、尾块与累加器要求，归约组件从归一化区域提取成员顺序、活跃状态和数值许可；各组件返回约束以及对完整行的提议，均只读 IR。纯收缩、纯归约和混合策略可调用相同组件，每种模式的分析与提议独立维护。匹配整个区域的联合策略对齐 K 的成员坐标、共享的参数行 ID 及状态寿命，核对同时存活的资源总量，最终向结构家族提交一份拓扑和一组整行。SDK 的策略链按序匹配，应让覆盖整个混合区域的策略先匹配；子模式匹配成功本身无权写入决定。联合提议不成立时报告约束或返回未匹配，后继策略仍须覆盖整个区域。

**同一 op 上的多个决定。** 各家族用 SDK 的 `DecisionFamily::read/write/verify` 在锚点上读写和检查自己的有类型属性，以登记的 family ID 区分。策略只提出值；校验后由框架调用该家族的 write。verifier 检查每个 ID 至多一项、锚点的 operand/result 与适用区域匹配、共享行 ID 有定义，联合约束对每行成立。物化者只消费自己的项。若替换锚点，必须将其他仍有效的项连同 SSA/区域引用映射到语义对应的新锚点并重新验证；丢失、重复或无法映射即在该边界报错。两个决定若要求共同选择、共同改写或依赖隐含的执行先后，应并入同一家族，或先在确实独立的锚点上分开。独立贡献不能用覆盖整个属性字典来更新自己的决定。

## 参数行、守卫与调优

| 阶段 | 交付的内容及校验 |
|---|---|
| 出口声明 | 出口拥有按架构和计算/访存结构登记的经验表，逐行同时提供两类列。公共列有 block schema，如每实例行数、宽度、M/N/K 尺寸、段长；原生列由该出口定义。每条完整行在本表版本内有唯一 ID。 |
| block 查询与选择 | 借助框架的目标能力服务取得公共投影及约束；出口可据本机事实检查私有选项。formed 只推导事实。decided 策略一次选择一个拓扑与匹配的整组行，整行过滤并记录不符合的约束。禁止跨行拼列。 |
| 检查点与出口 | realized 内的 `row.rows`、`row.width` 等直接决定 grid、坐标与循环。检查点按表身份、版本及行 ID 携带选中行的不透明原生载荷，供出口恢复；恢复时核对 schema 版本，不重新读取经验表或按位置配对。只有对应出口解释载荷，按 ID 核对一对一关联。过滤或重排行时保留其 ID；核验该载荷能实现此行承诺的公共参数与拓扑。 |
| 编译与执行 | 出口发射一份参数化程序、完整行、守卫与 ABI，可按行原生特化；prepare 过滤绑定不适用的行，compile 原生编译，首次 launch 只在剩余完整行中调优。赢家标识绑定整组候选、设备及实际依赖的性能类；每次新绑定重验守卫。 |

一次编译内，所有行的控制拓扑、供数拓扑、effect 次序、ABI 相同，循环界限和资源尺寸可以随行变化。比较驻留和流式、单缓冲和双缓冲等拓扑，用派生编译器编译同一输入，各自独立测量。runtime 不在拓扑间试跑。

**动态宽度。** 例如驻留实现依赖 `512 < N <= 1024`、内轴 stride 为 1，编译缓存键记录这个上界类和该 stride 类。N=600 和 1000 可共享编译结果，仍按各自 N 裁边；N=1025 或 stride 改变必须重新分类。开放宽度保留合法的流式实现，它的键只记录实际依赖的条件。N=0 时屏蔽所有列访问；输出也是空域可免启动，有非空的归约输出则写 identity 并执行作者 epilogue。guard 失败不会给已有产物传入越界实参；若目标没有适用行，诊断包含被拒行与条件。

**驻留行归约的检查点示意。** 下面只列共享表示的列。对同一组行，Triton 自己持有的原生载荷可分别为四个 warp 和八个 warp；另一个出口只需提供自己的整行及公共投影。

```text
public rows     r0=(rows:1,width:1024), r1=(rows:2,width:1024)
grid            ceildiv(R, row.rows)
coordinates     i = pid * row.rows + range(row.rows), j = range(row.width)
members         (i < R) & (j < N)
v               read_f32(x, offset + i*s0 + j*s1, members)
peak            max_num(v, axis=j, valid=members, identity=-inf)
w               exp(v - broadcast(peak)) on members
total           sum(w, axis=j, valid=members, identity=0)
output          write_f16(y, i, j, w / broadcast(total), valid=members)
```

## 两行从 Block 到 DSA 与 BANG C

以独立输入/输出的 f32 逐元素 `y=x+c` 为例，M=3、N=1100，内轴连续。block 选择按行组归属、沿列流式的同一循环。已兑现的片段为 `f32[R,W]`，grid 为 `ceildiv(M,R)`，每次活跃域为该片段与实际行列域的交集。A、B 是贯穿三层的两行；另列 C 说明剔除位置。没有为行复制 KIR 或重新规划实例归属。

`dsa@formed` 选择一种供数拓扑：两个交替的输入槽和一个输出槽，每槽容量 `F=align_up(4*R*W,128)` 字节。为便于核算，假定本例抽象存储映射到一类物理空间，容量 24 KiB，SDK 保留 4 KiB，本次供数预算 16 KiB，有效上限为 min(20,16) KiB。这些数是设计示例。表中区间单位为 KiB，均相对本实例的私有基址。

| 行与 Block 覆盖 | DSA 分配及结果 | BANG C 物化 |
|---|---|---|
| A：R=1、W=512；3 个实例，各遍历 3 块 | F=2；输入 [0,2)、[2,4)，输出 [4,6)，峰值 6 | 在 bound 核对真实空间、对齐和 primitive 需求，legal 绑定为 NRAM 对应区间；原生按 A 的尺寸发射搬运、计算与等待 |
| B：R=2、W=512；2 个实例，各遍历 3 块；末实例 1 行有效 | F=4；输入 [0,4)、[4,8)，输出 [8,12)，峰值 12 | 同一供数程序按 B 的尺寸绑定，末块按源及目标行距逐行搬运/写回 76 列，末实例只访问一行 |
| C（剔除示例）：R=2、W=1024；Block 流式遍历合法 | F=8，三槽需 24，超过 16。单输入槽加一输出槽虽只需 16，却会改变供数与等待拓扑。DSA 在 formed 拒绝整行，记录行 ID、所需容量、预算与原因 | supplied 及 bangc 都只收到 A、B |

**两行的等待证明。** 在 `dsa@supplied`，每个实例、每个有效块 k 都执行相同的事件图。覆盖输入槽 `k%2` 之前，若 k≥2，等 `compute_done(k−2)`；计算 k 前等 `read_done(k)`，若 k>0，还等 `write_done(k−1)` 释放唯一的输出槽；写回 k 前等 `compute_done(k)`。因此读下一块可与当前计算重叠。尾声等最后一次写回再复用或释放空间，零块不发事件；序言不等尚未产生的事件。A、B 的 k 均为 0、1、2，最后一次预取和写回都裁到 76 列。bangc 将每条依赖映射到相应的原生完成点；若目标不支持已选搬运或等待，记录能力缺失，不能私自切换为另一供数拓扑。

**剔除归属。** 控制拓扑不符归 block@decided，供数拓扑或其预算不符归 dsa@formed。仅到 bangc@bound 才知道的物理对齐、原生 scratch 等不满足时，由出口剔除该行并记录实际约束。任何一级过滤到零行都在该处报告失败；runtime 只接收全部检查通过的行。

## 两个首路线 kernel 如何消费同一份知识

稳定 softmax 的驻留路径：实例覆盖 r 行 → masked read 并升到 f32 → 沿列求 max_num → 同一片段计算 exp、sum → masked write 并窄化。一条流式路径可以重读输入，按原式分别求最大值、分母、输出；物理尾部的伪成员必须从 sum/max 中排除或使用各自的 identity；作者逻辑域内的 fill 继续参与计算。它不能仅凭“适用流式”改成需要额外数值许可的在线合并。RMSNorm 复用同一行映射、tail 和驻留/重读分析，但计算是 sum-of-squares，再乘运行时 inv_c、加 eps、rsqrt、乘权并写回。两者均由结构识别；gain 的列广播和别名约束是 RMSNorm 的额外输入事实。

## Triton 出口

**能力合同。** 本机调用解析设备 ordinal、架构、warp 事实、每块资源上限、工具链版本和所需特性。未探测到的可选能力为 unknown；不得填入另一代 GPU 的数值。上限、调用预算和 pass 的估算分别记录。经验表不以实时 SM 空闲率、可用显存或历史赢家作为硬件事实。链接时无设备的声明查询与真实调用时绑定快照应可区分。

**表的形式。** 下表演示两项独立登记如何服务同一个“列连续、f32 累加、最多 1024 列、行内驻留”的结构；元组顺序为 (rows_per_instance, width, num_warps, num_stages)。这些行只用于说明公共投影与私有载荷的对应。

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
