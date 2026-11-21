# cuTile 出口与本机 JIT

**V2-4 行程序已实现。** 首批作者程序为 stable softmax 和 weighted RMSNorm。下表记录发行配置。两个行程序已经在 SM90 和 SM120 通过本机编译、数值检查与 Bench 图重放测量。本机和 H100 环境均查到 cuTile Python 1.6.0、tileiras 13.3.36。出口的目标是实际输入所在的 CUDA 设备。

## 1. 入口、边界与产物

| 边界 | 生产者与消费者 | 离开边界必须成立 |
|---|---|---|
| `block@normalized` | Block → Triton 或 cuTile | 单一控制拓扑；完整行 ID、公共参数和不透明载荷；形状 guard 与实例映射；访问的源 view、成员坐标与有效域；有代数和轴含义的归约、循环 carry/yield、必要广播。行参数在实际 grid、片段和循环中生效。 |
| `cutile@legal` | cuTile legalize → 出口终验 | 每个访问都有 array、轴映射、tile 形状、索引单位、边界行为和已选访问形式。每个 collective 有轴、累加类型、identity、NaN 与舍入规则。流式循环的初值、递推和结束点已确定。原生 gather、MMA 等超出能力时在所属操作给出诊断。 |
| `cutile@callable` | 出口 ABI lowering → emitter 与 runtime | 已固定入口符号、参数方向与槽序、数组/标量约束、索引位宽、编译期常量槽、调用约定和三维 grid 表达式。逐行核对公共参数、guard、出口载荷及其 ID。循环、访问、同步均可从当前 IR 发射。无法兑现的行给出原因，全部失败则停止。 |
| emit | serializer → 本机 cuTile | 版本化产物包含可导入的 `@ct.kernel` 源码、作者 ABI、native signature、行与 guard、grid、行到原生 hints 的绑定。serializer 逐项翻译已选操作；加载器按产物版本校验。 |

首轮沿用线性实例空间 `(ceildiv(M,R),1,1)` 和 `ct.bid(0)`。实例 i 拥有行区间 `[iR,min((i+1)R,M))`。驻留拓扑处理一个 W 宽片段；流式拓扑沿现有 `block.tiles` 逐片段处理列。零行、或这两项程序的零宽输出，由绑定返回空输出并跳过原生 launch。未来多轴工作归属必须先在 Block 中形成可求值的映射。

## 2. 设备能力与候选行

由 `exits/cutile` 登记 Block 已有的 `RowCapacity` 服务，输入为架构、绑定宽度及本机工具链能力。每项经验行是整体。W 取有限序列 1、2、4、…、4096，对于基址或外轴字节 stride 未满足 tensor-map 对齐类、或内轴非连续的绑定，上限收至 256，避免宽静态片段在原生编译器中膨胀。每个允许宽度精确登记下列两项；存储 ID 在同一设备表及版本内唯一。Block 选择一个宽度类和一种拓扑，并把所选行写进 IR。以下配置对两个作者程序通用。

| 表与行 | 公共列 (R,W) | 不透明载荷（建议起点） | 用意 |
|---|---|---|---|
| H100 / H0 | (1,W) | num_ctas=1，occupancy=2，opt_level=3 | 单行归属，保留并发实例 |
| H100 / H1 | (2,W) | num_ctas=1，occupancy=1，opt_level=3 | 合并两行，给较大的片段更多片上余量 |
| 5090D / B0 | (1,W) | num_ctas=1，occupancy=1，opt_level=3 | 单行归属，作为低资源起点 |
| 5090D / B1 | (2,W) | num_ctas=1，occupancy=2，opt_level=3 | 在相同遍历中比较行组与驻留提示 |

例如 N=1000 选择 W=1024、guard 为 513…1024 的驻留行组。默认 N>4096 时选择 W=4096 的流式行组。显式派生流式策略仍使用 Block 发布的 guard。调优只能在已选组内改变整行。对同一设备上 H/B 系列各自使用的 row ID，恢复检查点时还须匹配出口、能力表和载荷，不能仅凭数组下标重建配置。

| 能力 | H100（SM90） | 5090D（SM120） | 编译合同 |
|---|---|---|---|
| 架构与资源 | Hopper，Tile IR 的 SM90 支持始于 13.3 | Blackwell，SM120 支持始于 13.1 | 按实际 device ordinal 查询架构、grid 上限、片上资源和驱动，并核对 tileiras 支持的目标。编译资源需求与原生编译/加载诊断共同决定合法性。 |
| tiled array / TMA | 检查规则矩形和 TMA 描述条件 | 单独检查 SM120 的工具链与访问约束 | TMA 只作为合格 tile load/store 的许可。检查基址/偏移对齐、各轴 extent、字节 stride、元素格式及 tile 形状；具体阈值取 CUDA tensor-map 及该版本 SDK 的约束。合法的普通 tile 访问可发射 `allow_tma=False`。间接 gather 不带此提示。 |
| MMA | 独立登记 Hopper 原生矩阵能力 | 独立登记 SM120 能力 | 表按输入/累加 dtype、M/N/K、对齐和数学许可查询。首选后续验证 f16×f16→f32、bf16×bf16→f32。TF32、FP8 快速累加和次正规数处理需要语义许可与原生支持。首轮两个行程序均不需 MMA。 |
| CTA / occupancy | 首批两行均用单 CTA | 首批两行均用单 CTA | SDK 1.6 的 num_ctas 取 1/2/4/8/16，occupancy 取 1…32，均须通过目标核验。它们是下游资源提示；程序中的工作归属和 grid 由 Block 决定。多 CTA/cluster 要有独立能力与合法实例同期执行的依据才扩充。 |
| 索引 | 经范围证明可用 i32，否则 i64 | 同规则，按本设备的签名编译 | 分别验证 shape、stride、tile 坐标、乘加中间量、末端地址和 grid。使用 i64 时同步升级标量算术与数组 index dtype（`ArrayAnnotation`）；地址计算保持合法的指针宽度。无法表示的绑定在 prepare 拒绝。 |

GPU 型号只提供能力查询的起点。winner 随设备与绑定选择；`allow_tma=True` 也不证明生成了 TMA 指令。[NVIDIA Tile IR 架构支持](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html) · [cuTile 性能提示](https://docs.nvidia.com/cuda/cutile-python/performance.html)

R、W 在编译单行时作为静态 tile 常量，两轴都是正的 2 的幂；动态的是绑定时的 M、N。对于 i64 路径，先将 block ID 和索引操作数提升，再进行乘法。CTA hint 不参与 grid 求值。SM120 的经验表独立于 SM100；不能把 Blackwell 数据中心型号的矩阵指令或集群配置直接套在 5090D 上。

## 3. 合法化与两个程序的走读

| Block 已作出的决定 | cuTile 如何兑现 |
|---|---|
| 规则矩形、活跃成员、物理尾部 | array view 保留实际指针起点、dtype、shape、元素 stride 和所有者。逻辑元素坐标转换为 tile 下标时须证明整除关系；例如第 i 行组、第 j 列块直接用 `ct.load(X,(i,j),shape=(R,W),...)`。不再把 stride 乘第二次。尾读使用有界 load 和已定义 padding，尾写只覆盖合法成员。任意作者 predicate 需要保留其显式 fill。 |
| 非矩形或数据相关访问 | 将已知坐标与有效 mask 明确映射为 `ct.gather` 的 indices、mask、padding_value；写对应 scatter 或原子。重复地址必须服从作者的 combine、原子或唯一性合同。SDK 不支持某种有序原子时在 legal 诊断。 |
| 行归约、广播与有序循环 | 在消费归约前处理非成员 identity（sum 为 0，max_num 带作者初值）；分别选择 `ct.sum`、具有匹配 NaN 行为的 `ct.max` 和显式 reshape/broadcast。保留各自的 f32 累加、除法、rsqrt 和末次 cast。流式版本逐个翻译已有 sweep，并保留前一 sweep 的完整结果。 |
| 已分块的收缩 | 将两个片段的 M/N/K 轴、累加器、尾块与 epilogue 匹配到 `ct.mma`。丢弃不满足数值许可的 primitive；累加器初始化、跨 K 的迭代和结果归属均从 Block 取得。 |

**Softmax。** 对活跃行读 f16，升 f32；先算 max_num，再减最大值、exp、sum、除法，最后窄化。行尾的无效 lane 在两个归约处使用各自的 identity。任一有效输入为 NaN 时，按作者 exp/sum 链传播；不得以统一的 load fill 改写这条链。

**Weighted RMSNorm。** 对 X 的行片段做 f32 平方和，乘实际 inv、加 eps、rsqrt；将结果沿列广播。rank-one gain 用列坐标读 W 个元素并沿行复用。对 gain 的 slice，物理地址由它自己的 offset/stride 产生。对 X 和 Out 各用自己的二维视图。两个程序都保持一个作者 kernel，在调用方当前 stream 返回。

## 4. Runtime、原生编译与调优

| 阶段 | 调用方式及可核对结果 |
|---|---|
| prepare | `runtime/kir` 检查参数方向、维度、dtype、别名和输出；`runtime/block` 对已发布的 guard 与 grid 求值；`runtime/cutile` 将实参绑定成 CUDA array 和标量槽。按输入设备发现能力。动态 shape 只对实际依赖的类特化，例如索引宽度及对齐/stride 影响的访问类。此时可 lowering，不执行作者调用。 |
| compile | 对每个合法行按实际数组约束、行常量、编译 hints、SM 和工具链完成原生 cubin 编译，核对入口符号及调用约定；相同签名可共享。保留逐行编译时间、命中与失败原因。SDK 的 `compilation.export_kernel` 能做指定签名的无执行 cubin 导出。导出成功本身不证明普通 dispatcher 会复用 cubin。 |
| 首次 launch | 一个可执行行直接启动。多行把**已编译的这一组 row ID** 作为 `exhaustive_search(search_space=...)` 的完整输入；`grid_fn` 使用 mapping，`args_fn` 绑定试跑槽，`hints_fn` 只翻译该行载荷。成功结果的 `best.config` 必须对应输入行。保存每行失败；没有可用行即报错。搜索会执行 kernel，结束后仍须在真实输出上启动 winner 一次。 |
| 热调用与捕获 | winner 缓存与原生缓存分开；winner 键包含设备、拓扑、合法行组、性能相关 shape/stride 类。重用实际 launch 所需的同一带 hints 的 dispatcher/已加载函数；新 buffer 重新绑定。每次 launch 在实际输入设备上取得**调用方当前 stream**，保活输入、输出和必要工作区至该 stream 完成。capture 前完成原生编译、模块加载、调优与预热；capture 中遇到冷路径明确诊断。热 launch 不加全设备同步。 |

**怎样保证只编译一次。** SDK 1.6 的搜索器会在内部调用 `replace_hints`，而新的 kernel 对象拥有自己的 JIT 缓存。建议把实际签名解析、按原生选项复用 dispatcher、以及 compile-only 结果交接封装在版本化的 `runtime/cutile` adapter；以原生编译调用数检验 compile、search、winner launch 三者使用同一组产物。可在该版本窄适配 dispatcher 的编译回调，以真实 `(cubin, symbol, calling convention)` 完成交接。升级 SDK 时必须验证此接口。若选全公开 export + CUDA Driver 加载路径，则需要实现已导出调用约定，并以同一加载入口计时；不能再交给另一个惰性 dispatcher 重编译。

**状态恢复。** 首批两项只有只读输入与整写 Out，调优使用私有等形状/stride 输出，保留原 buffer 的别名关系。未来涉及 InOut、原地更新或跨实参别名时，每次试跑须从同一初始状态开始。SDK 的一次 benchmark 可能在内部重复调用，同一次 args_fn 返回一份快照不足以保证此合同。不能控制每次恢复时，使用现有安全单行或另行实现逐次恢复的计时适配，并明确记录选择。无需改变 Bench 的计时边界。

私有试跑 buffer 必须满足与真实绑定相同的 dtype、对齐类与编译签名。原生缓存按候选、源码、实际签名、hints、目标及工具链隔离；winner 另按实际工作量和访问类复用。编译预算耗尽时保存失败行及 tileiras/ptxas 的原始诊断，热缓存成绩不能覆盖冷启动错误。

V1 的 `runtime/cutile/program.py` 将 compile 和 launch 分开，`exhaustive_search` 位于首次 launch。V2 沿用这种阶段划分，搜索范围由当前 Block 程序限定。不同控制拓扑的比较走派生编译器。源码、Tile IR、cubin、成功加载、数值通过和热执行是分别可观察的结果。[SDK 调优 API](https://docs.nvidia.com/cuda/cutile-python/performance.html)

## 5. 加法成本与公共规范化

V1 两份 `lib/Target/{Triton,CuTile}/Transforms/Legalize.cpp` 都调用短窗口归约兑现、执行组 lowering、收缩形状规范化、workspace lowering、整数折叠，以及值外提或公共值消除。结果依赖出口作者再次排列公共变换。V2 的入口是**已验证的同一份** `block@normalized` 检查点，出口只能登记自己的 `legal → callable`。

| 位置 | V2-4 实施预估 |
|---|---|
| `exits/cutile`、`runtime/cutile` 与产品登记 | 新增这两个单元，在已有编译树登记，增加对应安装依赖与说明。合法化、发射、SDK 适配均归出口。 |
| `levels/execution/block` | 两项算子的行表、guard、实例映射、归约和显式行广播可以原样使用。当前 normalized 里对 rank-one gain 的 `block.load` 仍同时传 rows、columns，结果为 rank-two 片段。**它选择列轴并复制到各行的语义需要有共同的可验证依据。** 建议按下节裁决在既有 Block 单元内补齐。 |
| `exits/triton` | 原生参数、映射和后端流水保持现状。如采用公共访问投影接口，预计仅改读取该投影的访问 lowering/发射点并复验同一地址表达式。 |
| `runtime/block`、`runtime/triton`、公共 JIT | 首轮预计各 0 个文件；既有 mapping 足以求出三个 grid 分量。cuTile 的签名、hints 和缓存均在自己的 runtime 消费。 |

当前 `Normalize.cpp` 实际显式化的是二维算术中的行广播。不能据此声称收缩、workspace、任意 gather 或有序 collective 已具备公共规范形。以后增加相关程序时，跨出口共享的仿射访问投影、有效域/归约维对应、循环 carry 归一化均应在 Block 内有真实生产者和 verifier，再交各出口；array view 到 SDK 操作和 TMA 许可属于 cuTile。[Block 规范化](../../levels/execution/block/transforms/Normalize.cpp) · [Triton 访问合法化](../../exits/triton/legalize/Legalize.cpp)

例如当前 `block.reduce(value, active, algebra)` 已携带中和尾部所需的三个量。Triton 可显式生成 neutralize，cuTile 可用相同有效域生成 select 后接原生 collective。若某种变换会改变 active 的来源或重排共同的归约遍历，须在 Block 规范化并验证；单个出口只选择目标语言的等价拼写。

## 6. 已采纳的三项设计选择

1. **rank-one 访问投影**可将现有行方言的 rank→轴合同写成公共查询并在 normalized 校验，或现在就新增通用轴映射属性。首选前者，覆盖 gain 的列投影；新增真正多义的访问时再引入显式映射。
2. **原生编译交接**可采用版本化 SDK dispatcher adapter，或采用公开 cubin 导出加独立 Driver launcher。首选前者，沿用 SDK 调用约定与当前 stream，并把对版本的依赖集中在一处。
3. **首批访问形式**可先对已证明矩形的行访问采用有界 tile load/store 并允许合法的 TMA，或统一先走 masked gather/scatter。首选前者；只对需任意 mask/不规则地址的访问走索引路径。按设备复测编译成本和完整算子性能，再调整各自的两项经验行。
