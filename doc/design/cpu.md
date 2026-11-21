# CPU 任务程序与本机出口

**V2-3 设计，四项实施选择已批准。** 本文给出 softmax 和 weighted RMSNorm 的实施合同，以及收缩、共享准备和有序状态接入同一模型的位置。基线是产品 `b01f659` 和 CompilerInfra `3ddeef2`。下文的 IR 名称表示拟议的有类型操作；候选行是首轮设计值，尚无 V2 CPU 编译或计时结果。

## 1. 一份值程序，两个交接点

```text
kir@canonical → cpu@formed → cpu@tasked ─┬→ [weft 私有存储兑现] → weft@legal → @callable → emit
                                       └→ cpu@buffers → mojo@legal → @realized → @callable → emit
```

| 边界及生产者 | 检查点中必须存在 | verifier 拒绝什么／下一消费者 |
|---|---|---|
| `cpu@formed`，KIR → CPU 值 lowering，加任务分区策略与同边界 realizer | 输入输出 view 带原有方向、shape、stride 和别名合同；中间计算为 shaped SSA。parallel 工作域、切片、归约、cast、ordered carry 和 effects 保真。分区决定已经写成工作集分组、task 成员域、域内计算块及结果映射；保留完整合法行及 shape guard。 | 唯一写的活跃输出须恰好覆盖一次；间接映射须有唯一性依据或明确合并/原子语义。归约轴上的切分须满足代数与数值许可。拒绝跨依赖的任意切分及绑定某个核号的 task。task 构造读取这些分区关系。 |
| `cpu@tasked`，任务 lowering，然后存储策略 | 每个 task 有逻辑区间、显式 captures、输出责任、阶段依赖和完成点；跨 task 的值通过具名结果或共享准备的值传递。task body 保留结构化计算、只读值和有序状态。存储决定标在需要兑现的值/区域上。 | task body 无隐式外部可变 captures；并发写切片互斥或有已声明冲突处理。跨阶段使用由生产者完成和 join 支配。verifier 检查存储需求的类型、生产点、读者和作用域；外部 view 本来就可寻址，内部值仍未统一转成 buffer。Weft 在此接出。 |
| `cpu@buffers`，CPU 存储 realizer | 需有地址的状态才形成 buffer/view、初始化、生产与释放；每个共享块在可见范围内生成一次。SSA 中可完成的中间运算继续保留。并发 task 各自获得隔离的私有空间。 | 所有待兑现决定均已消费；逐行检查大小和对齐、支配初始化、无越界、无并发生命期冲突、最后一次读取完成后方可复用。CPU verifier 只检查公共类型、效果和合同。Mojo 消费该检查点。 |

**三种粒度。** 工作集是一组值得共同使用某个输入窗口的逻辑工作；计算块是一次实现能处理的有界成员集合；task 是能独立派发、完成和等待的责任范围。以 M 行为例，任务 t 拥有行 `[tR,min((t+1)R,M))`，其内部可按宽度 C 遍历列，一个 task 可执行多个计算块。R 与 C 均不决定线程数。完整输出行的 task 独立；若共享准备产生多个 task 的输入，先完成准备，再向消费者发布只读值；若有跨 task 部分和，必须显式表达 partial 的归属、join 以及后继归并。这些内部阶段归同一个作者 kernel，不能把作者可见的多次调用合并。

分区 realizer 也要固定准备的生产/消费拓扑：若一组完整行依赖共享 packing，形成一个逻辑准备结果及其消费者，task 构造把该依赖落到阶段上。存储策略随后才决定怎样保存、放在哪里和何时复用。不得在 Weft 私有兑现或 Mojo 物化时给一个原本独立的 task 图偷偷插入新的并发阶段；需要另一种任务图时，回到分区策略选另一种实现。

## 2. 决定、能力服务与 packing

| 家族及锚点 | 决定 → 物化 | 读取的事实 | 写入程序的内容 |
|---|---|---|---|
| CPU 分区与计算实现，整个相关计算区域 | `formed → formed` | KIR 的成员/effects、可并行轴与 carry、消费链、实现对数值/输入的要求、线程及工作集预算、出口经验行 | 一种任务/遍历拓扑与一组完整行；工作集、区间与输出映射，子块轴与尾部、实现引用及其输入要求。tasked 的必需 lowering 形成显式 task、捕获、依赖和 join。 |
| CPU 供数与物化，含共享生产者及其读者的任务区域 | `tasked → buffers`；Weft 接出时私有兑现 | 实际 task 与捕获，已绑定实现、读写别名、复用次数、最后使用与完成点、并发峰值、行预算 | 每个值保持 SSA、task 私有、或指定阶段生产并共享；写明 packing 的输入/结果关系、初始化位置、对齐、容量、复用槽和释放条件。所选供数拓扑在组内唯一。 |
| 出口原生实现 | Weft `legal → callable`；Mojo `legal → realized` | 当前有类型计算、行 ID、出口私有载荷、真实可用工具链与原生能力 | Weft 的单实例 task、host 调用与 ABI；Mojo 的循环、向量/标量尾部、局部实现与导出体。必需的单一路径合法化直接用普通 pass，无选择时无须新 family。 |

**次序与可插拔性。** SDK 先执行边界 lowering，再 decide/materialize。分区在 formed 当场兑现；tasked 先生成任务图，存储策略才能读取真实并发关系。行归约、收缩、scan 各维护只读分析和提议；attention 等混合区域由一个覆盖全区的策略组合提议后提交一份决定。不按 kernel 名选择。默认链按节点配置；可选优化关闭时仍可选择满足作者合同的通用实现。形状、类型或数值条件不符返回未匹配；声称适用却交出非法行属于组件错误。两家族共享稳定行 ID，第二家族可因预算拒绝整行，不能重选第一家族的拓扑。比较不同拓扑用派生编译器。

**服务由 CPU 声明、出口实现。** 产品在 CPU 单元公开头中规定查询参数和返回类型，经 SDK 的 typed target service 取得实现。查询输入是当前 op/region 的类型、数值许可、访问和块域；出口返回有限的实现引用、完整参数行、输入表示要求、尾部支持、scratch 需求和合法条件。硬件上限、软件预算、某行估算用量是三个量。描述布局或在 kir 停止时不要求机器已绑定。每条被采用的选择都须能从当前 IR 和匹配的能力合同继续兑现，不能靠旁置计划解释执行。

以 M×N×K 收缩为例：

1. 分区策略查到实现可接受的元素/累加类型、M/N/K 子块、B 的 panel 关系、尾部及舍入约束。`formTile` 接口以 tile 原点、有效 extent、操作数切片、原累加值和行引用为输入。在选中后，由 realizer 调用它构造公共 CPU 计算块，返回更新后的有类型累加值。在查询阶段只描述需求。
2. 两个输出 task 读同一 B panel 时，若源 view 在读者完成前不会被写且并发预算足够，供数策略选一次 packing，在准备完成后共享。若 B 与可写参数可能重叠，先检查原效果顺序；无法保持时不得外提。私有 scratch 可在先后完成的块间复用；并行任务计入多个活跃副本。即便每项单独能装入，峰值合计超预算也拒绝该行。
3. 公共 IR 保留输入成员关系、对齐、布局要求和数值合同，公共 verifier 检查 tile 覆盖与供数一致性。AMX tile type、RVV LMUL、具体打包指令由出口私有兑现。若 primitive 会隐式 FTZ，而作者要求保持次正规数，只有符合合同的实现或语义等价的显式分支才合法。V1 的 `verifyCPUProgram` 直接识别 AMX op/type，V2 在出口实现登记处验证这些专有性质。

例如一块 B 被 task A/B 共读，准备 P 写 packed B，P 完成后 A/B 可并发，二者 join 后方可重用该槽；A 的私有累加区与 B 的私有累加区不得覆盖。该行的预算下界包含一个共享 panel 和两个同时活跃的私有区，另计原生实现要求的对齐及额外 scratch。若实际执行预算只允许一个活跃实例，按串行寿命计算。保持并发拓扑却按一个私有副本估算属于非法行。Weft 和 Mojo 都从同一条生产与使用关系导出这一检查。

## 3. Weft：在值交接点接管，host 唯一派发

**入口私有 realizer。** `cpu@tasked` 包含存储决定但尚未经过 `cpu@buffers`。SDK 在 Weft 首个 lowering 前调用该家族的出口私有 realizer。它把可保留的值映射为结构化表达，把确需共享的准备落实为 host 拥有的存储、准备 task、传入后继 task 的 view 及释放依赖；必须有地址的私有 scratch 绑定到本次任务实例。它兑现既定准备阶段、行、容量与最后完成点，不引入新的调度拓扑。每项决定消费一次。私有 realizer 缺失时仅该出口无效。

| 出口边界 | 可检查的产物 |
|---|---|
| `weft@legal` | 每个静态 task 定义对应一个可表示为 Canonical Weft 的计算体。有类型的 host 程序显式包含准备、任务区间、派发、join 和退出。跨 task 的储存与状态全部有 owner；不遗留 CPU 待兑现属性。 |
| `weft@callable` | 每个任务符号的调用点与签名、运行时形状和 stride 的槽序、workspace 及调度预算已确定。host/task 接口可与下游真实编译返回的 ABI 逐项配对。 |
| emit 与 native build | 同组发射参数化 host 源和每个 task 定义一份 `.weftir`。出口 native builder 在执行机按需要特化整行，每份输入各经一次适用的 `triton.compile` 生成一个 native kernel，取得 `single_instance_abi`；校验符号、类型与参数顺序，再编译/链接 host。动态任务实例复用相同定义。 |

**两条 ABI。** 对外，一个有版本的同步 host 入口接收按作者签名排列的已绑定 views、标量、选中行及执行预算，返回成功或具体错误；基址/元素 offset、extent、元素 stride 和 dtype 的换算只做一次。对内，host 按导出的单实例声明传入基址、标量捕获、显式工作区间、输出切片及必要 scratch；动态 shape/stride 按该声明平铺，检查有符号值、宽度与 `size_t` 可表示性。无法表达的负 stride 在绑定边界诊断。下游的单实例 C 符号返回 `void`，其内建 PID 为零、grid 为一。因此全局行号来自 host 显式传入的逻辑区间，不能用下游 PID 恢复。版本和签名不符时在加载/原生编译阶段报错。

host 独占任务派发与收束，允许它调用已有的并行执行库。每个任务只作单实例调用；不再进入 Triton 的多实例 grid 派发。一次 task 正常返回构成该次同步计算的完成；host 在全部依赖完成后发布读者，最终 join 后才返回。绑定、工作者初始化或调度失败时停止继续派发、收束在途工作、释放共享资源并返回状态；不能把单实例的 void 当成错误码。runtime 只加载完整 host 产物及其任务依赖，launch 进入 host 一次。

Triton–Weft 的导出物同时描述了单实例和 grid 接口。本路线校验前者的符号及声明，复制/持有必要的原生模块和依赖直至最后一次 host 返回。出口发现时不导入设备 SDK；真正编译时要求当前环境的 Triton 提供 Weft backend 和 `.weftir` 入口，缺失时说明所需安装，保持 GPU Triton 所在环境的 driver 状态。

### 本机事实与首轮经验行

下表的事实来自指定执行环境或注明的历史记录；每次绑定仍由正式 native-target 接口校验。示例行写作 **(每 task 的行数 R，循环每块列数 C)**，单位为逻辑成员。行来自各出口的有限经验表，供相同结构的行归约使用；它们不是测得的 winner。每个公共行 ID 可关联出口私有选项，不向公共 CPU 方言暴露特定 ISA 参数。

| 环境 | 事实与选表条件 | 首轮同拓扑行，未调优 |
|---|---|---|
| x86-64 Weft | 本机查询为 SysV、x86-64-v4、固定向量 512 bit，RVV 的 `vlen_bits` 为 0。原生矩阵能力另查执行集合与权限；这些 f32 行归约只要求向量计算。其他 x86 层级按实际能力另选。 | (1,64)、(4,128) |
| SG2044 | 本轮查询为 RV64、LP64D、VLEN 128，含 Zvfh；native matrix 列表为空。VLA 尾部按剩余元素处理。 | (1,32)、(4,64) |
| K3 X100 | Weft 的 `k3-x100` 档为 RV64/LP64D、VLEN 256，未选择矩阵扩展；Bench 档选择 CPU 0。本轮 SSH 超时。实际接入按所选核组重新查询 ABI、VLEN 和可执行扩展。 | (1,64)、(4,128) |
| K3 A100 | 同机不同核组，历史目录为 RV64/LP64D、VLEN 1024，含部分使用 `spacemit-ime2` 的既有矩阵案例。本轮未实时复核。特殊核注册、绑定及逐核检查走 Weft 正式 CPU 执行支持；矩阵 fragment 只按实际授权启用。 | (1,128)、(2,256) |
| RVV V100 | 本轮查询为 RV64、LP64D、VLEN 256，含 Zvfh，native matrix 列表为空；现有 Bench 档取单核。独立于 K3 选表，即使两者 VLEN 相等。 | (1,64)、(2,128) |

选表读取授权 CPU 集合中各核的共同 ISA、运行时实际 VLEN、缓存和扩展可执行条件；线程数来自调用预算。跨异构核组时，先要求选择同质执行集合。硬件型号和线程瞬时位置均不足以证明可执行性。任务粒度与机器向量宽度可独立变化，Weft 负责实际 RVV 指令布局、LMUL 及寄存器分配。

工作区预算按调用配置与可用上限过滤。此次 SG2044、V100 探测都返回 64 KiB 的数据 cache 信息，该值不能直接作为某次调用可独占的 scratch 容量。需要特殊线程注册的核心，必须由 Weft 支持的执行作用域在进入目标集合时完成注册与核验；同一 host 派发出的工作者须满足该行实际使用的扩展要求。

## 4. Mojo：在 buffers 之后实现原生程序

| 出口边界 | 实际变换与检查 |
|---|---|
| `mojo@legal` | 接收有 owner 和寿命的 CPU 存储；为结构化计算选择已声明的 Mojo 实现、显式循环和向量化轴。保留签名、行与 guard，以及 masked read、尾部和归约的数值前提；按访问关系决定连续向量载入还是受支持的 strided/scalar 路径。 |
| `mojo@realized` | 在该出口兑现 SIMD 运算、水平归约、尾部、packing 消费、局部矩阵 primitive 与线程执行体；这里承担 V1 realized 的终验。检查每个实际 primitive 的类型、ISA/OS 许可、初始化与所有退出路径的 join；拒绝未展开的计算/存储决定或 serializer 尚需补出的循环。 |
| `mojo@callable`、emit | 只绑定原生入口与参数槽，验证已终验的计算体仍闭合。emit 输出可编译的 Mojo 模块源、有版本的 C ABI 和整行选项；所需的本地辅助库列为真实链接依赖。下游是执行主机的 `mojo build --emit shared-lib`，产物为可加载的共享库及同步入口。 |

**能力。** V2-3 Mojo 以 Linux x86-64 为首个执行环境。由 `mojo build --print-effective-target` 取 triple、CPU、features，与本进程可执行核集合取交。本轮本机返回 `x86_64-unknown-linux-gnu / sapphirerapids`。AVX2 与 AVX-512 分表，例如 f32 行归约各提议 (1,64)/(4,64) 与 (1,64)/(4,128)，可选择的 8/16 个 f32 lane 只在出口私有选项中解释。缺少 f16 转换或精确数值实现的行不得采用。矩阵能力分别描述输入、累加、子块、供数和次正规数语义；AMX 使用还需 OS 许可及实际执行线程满足条件。Mojo 编译并发度不计入应用线程预算。Mojo 导出函数拥有本次计算的并行区间与 join；若用 OpenMP，只由该入口建立并行域。

## 5. 绑定、缓存和 winner

| 阶段 | 两出口共同遵守的行为 |
|---|---|
| prepare | 复用 `runtime/kir` 的参数方向、符号维度等式、输出分配、别名及所有权；目标单元要求 CPU 内存，再解析当前有效执行范围、工具链和预算。用当前绑定建立实参帧，必要时 lowering，不执行任务或试跑。Mojo/Weft 均只读自己的 ABI 和祖先合同，不依赖 runtime/block 或彼此。 |
| compile | 由所选出口的 native builder 生成适用行的全部必需产物。Weft 必须同时取得 task 模块与可调用 host，Mojo 必须取得 ABI 相符的共享库。失败保留实际阶段、符号及工具链诊断；同一编译身份只发布完整可加载的一组文件。 |
| launch / result | 调用选中行的原生入口，所有工作完成后返回。返回绑定时的输出容器。供 Bench 使用的公开原生调用描述包含函数地址、typed slots、所有权对象和完成语义；Bench 可对完整入口计时。 |
| 调优 | 同一输入、控制与供数拓扑内比较完整行，计时包含共享准备、派发、计算及 join。试跑在隔离的用户状态上进行，重建 alias 关系，每轮恢复到同一初态；真实状态只执行 winner 一次。不能隔离的副作用选择合法单行或诊断。跨拓扑比较走派生编译器。 |

编译缓存按候选库/配置、作者定义、出口和工具链版本、有效 ISA/ABI/向量及实现能力、实际依赖的形状/stride 类和完整行组隔离。Weft 可复用下游的 task 编译缓存；host 另绑定同一组 task 的真实 ABI/产物。数据相关的 packed 内容默认只活在一次调用内。winner 单独按有效行集合、设备/CPU 集、线程预算及性能相关工作量和访问类复用；缓存中不保存用户地址。普通热调用复用能力快照，只需核对有效作用域和 guard。新地址重新绑定；若 prepare 后 affinity 或预算已变，旧 prepared 入口须在执行前诊断并要求重新 prepare。相同能力类且无相关预算依赖时可以复用机器码，不能直接复用跨核组的 winner。普通 JIT 默认关闭观察。

两出口均通过新 runtime 单元下的 `binding.target.Target` 接现有调用协议。specialize 返回编译身份与本次绑定参数，options 只上送已探测的目标事实；产品配置给出的线程上限仍归调用方。materialize 读取版本化产物，其顶层作者 ABI 与入口签名相等；bind 持有当前内存。prepare 可装配描述，显式 compile 才建齐适用的原生行，首次 launch 需要时调用 compile 并选行，单行直接执行。发现权限或工具链变化时失效或诊断；不得把候选失败记作通过。

## 6. 两个作者程序逐边界走读

| 检查点 | Stable softmax | Weighted RMSNorm |
|---|---|---|
| kir@canonical → cpu@formed | 一个逻辑行的 f16 输入升 f32，先 max_num，再 exp、sum、除法及一次窄化。分区行独立，保持作者求值和 NaN 规则。每个 t 拥有 R 行，列块 C 留在行内。 | f32 输入、同列 gain、标量 inv_c 和 eps。独立输出行，归约为平方和；完整归约值才能进入 rsqrt 和加权阶段。inv_c 保留实参。 |
| cpu@tasked | Task 捕获 x/out 视图和行区间，仅写自己的行。max 与 sum 各有私有 f32 摘要；选择保留 exp 片段或重算时检查活跃范围和许可。跨行只在作者 kernel 完成时 join。 | Task 捕获 x、gain、inv_c、eps、输出切片。平方和及逆标准差为私有值。共读 gain 只是只读 capture；连续未变换的 gain 无需新增共享 pack。末行组按实际 M 截断。 |
| cpu@buffers | 示例选择重读输入：第一遍求最大值，第二遍累加 exp 求分母，第三遍按同一表达式重算分子并写输出；无整张矩阵临时量。若选保留 exp 的拓扑，需私有行空间并检查并发副本总预算。 | 每个 task 的归约状态可始终为 SSA；在归约完成后重读 x、广播系数并乘 gain。只在实现确需地址时形成 scratch。 |
| weft（直接从 tasked） | 私有 realizer 保留值归约或兑现被选中的行 scratch。每个行组 task 定义一份含归约与逐成员运算的 `.weftir`；host 按行区间实例化并 join。 | 同一 host 调度方式，task ABI 再携带 gain 及两个标量；同一 task 完成平方和与缩放，host 不重建计算。 |
| mojo（从 buffers） | legal 选定连续行上的向量读取/转 f32；realized 明确 max、exp、sum 的遍历及 mask 或标量尾；callable 导出含同步完成点的函数。 | 明确向量平方累加、水平归约和 gain 的列读，出口生成第二次遍历及尾部；入口返回前完成所有行。 |

例如 M=3、N=70、R=2、C=64 时，两项程序的两个 task 分别负责 [0,2)、[2,3)，每行尾块只有 6 个活跃成员。物理填充不贡献 max/sum，不凭空读取 gain。N=0 时这两项程序都没有输出成员；对有非空输出的其他归约必须另外履行 identity 和 epilogue。跨行全局 barrier、跨 task scratch 及行分母共享均不由这两个算法推得。

## 7. 加法成本与实施前裁决

| 现有位置 | 接入时所需变化 |
|---|---|
| 产品 `product.json` 与 `cmake/default.json` | 前者在既有 execution 级登记 cpu、两个出口和对应 runtime；后者增 CPU 的有序策略链/realizer 选择。各新单元自己的 CMake 完成链接。顶层 CMake 及 CompilerInfra 公共驱动无须按后端枚举。 |
| `levels/algorithm/kir`、`levels/execution/block`、`exits/triton`、`runtime/block`、`runtime/triton` | 两个相同作者程序已由 V2-2 产生 canonical KIR。CPU 接在此处，预计这些单元的实现文件改动数为零。收缩举例刻画未来模型扩展；若日后新增作者语言构造，其完整前端/KIR 定义单列为语言工作。 |
| `frontend/python/intentdsl/compiler.py` 与公开 JIT | 当前已按出口名装载 `binding.target`，调用 specialize/options/materialize/bind 协议。两个新出口可按该协议返回 CPU 产物（保留顶层 `abi`），shape 元组携带目标需要的调用形状。首轮沿进程执行集合和出口配置取得线程预算，现有入口无须逐出口修改。 |
| `runtime/kir/binding/views.py` | Torch CPU tensor 路径可直接复用。实际文件无条件导入 Torch，只接受 Torch tensor、以 Torch 分配输出；SG2044 和 RVV V100 的已配置 Weft Python 本轮均有 NumPy、无 Torch。若覆盖这两处现有环境，建议在该文件把现有存储操作分派给本单元的 Torch/NumPy 绑定组件，保持原接口，做到仅在需要时导入相应库。依赖 shape/dtype/元素 stride、写权限、底层 allocation、偏移与所有权，不得为适配静默拷贝。首轮对共享可写 storage 延续现有诊断。此项是输入容器能力扩充，与 Weft 特判无关。 |

**已批准的四项实施选择**（以下保留选型时的依据）：

1. **无 Torch 的 CPU 输入。** 扩充共享绑定以直接接受 NumPy 数组，或首轮限定 Torch 环境。倾向前者，保证已配置的 SG2044/V100 可以从普通 JIT 直接调用。暂不要求通用 DLPack/全部 Python buffer 格式。
2. **首轮任务划分。** 行归约先支持完整行归属；或同时引入小 M、超宽 N 的跨 task 两阶段归约。倾向先做行间分工；后者保留在同一分区/供数家族，以派生比较再启用，归约顺序另核。
3. **共享 packing 寿命。** 限于一次调用；或为作者显式冻结的权重增加跨调用 prepared-weight 所有权接口。倾向前者，消除跨消费者重复准备，同时避免裸地址复用已修改输入。
4. **Weft host 绑定时机。** native build 读取 task 单实例导出声明再编译并链接 typed host bridge；或 loader 在加载时核验并填充有类型的函数表，由 host 间接调用。倾向前者，让签名错误止于 compile，热路径可直接调用。两案均由 host 唯一调度，task 均经 `triton.compile`。

**依据与证据边界。** 框架阶段次序和接出规则取自 CompilerInfra `3ddeef2` 的 `lib/Assembly.cpp`。V1 `b0a19990` 对照位置为 `include/Intent/Dialect/CPU/Transforms/Implementation/Implementation.h`、`lib/Dialect/CPU/Analysis/PhysicalProgram.cpp`、`lib/Target/Mojo/Transforms/Passes.cpp`、`lib/Target/Mojo/Serialization/Serializer.cpp` 与 `python/intent/runtime/weft/compilation.py`。Triton–Weft `c4fa4a9bc` 的 `python/triton/backends/weft/compiler.py`、`third_party/weft/python/weft_core/launcher.py`、`python/triton/runtime/weft/artifact.py` 分别定义单任务编译、单实例声明和导出物。本轮读取这些实现与目标查询结果；经验行、两出口的 V2 ABI、性能及数值仍待实施验证。
