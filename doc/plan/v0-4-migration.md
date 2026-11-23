# v0.4 出口迁移与 V2-6.P 性能收口

产品起点为 `4fed147` 的五个出口。沿用现有树：KIR 表达算法，Block 和 CPU 决定工作归属、分块与复用，DSA 从 Block 的已兑现边界接出，负责显式供数。Triton/cuTile 从 Block 接出；Weft 从 CPU 的值任务边界接出并兑现存储，Mojo 从 buffers 接出。库支持通过出口能力加入候选，不增加新出口。下表是后续实施计划，本轮只调整 Bench 数学裁判与范围说明。

## 一、现有 B 类代码逐处去向

调查 6 的人工归类以 `285c70e` 为快照。B 包含“重写整个可辨认区域”和“发射时补写执行结构”，不能据此推断所有成段匹配都应删除。审计 G6 又指出两处跨层决定。下表按当前产品的源码相对路径和符号定位；`IR` 表示由节点或出口 legalize 形成可验证、可继续变换的结构；`目标操作` 表示按合同登记的不可分调用；`删除` 表示替代路径接管后移除。一个旧函数同时处理多件事时按职责分列。

### 共同节点（随第一个消费出口落地）

| 现有位置及行为 | 去向 | 替代后的归属与保证 |
|---|---|---|
| KIR `ir/Normalized.cpp` 的双 fold 匹配、Block `analysis/ScorePanel.cpp` 将整段 attention 固定为四条计算 | IR | KIR 检查因果语义、缩放底数和数值许可，Block 显式形成 QK、mask、归一化状态和 PV；由值和效果检查依赖，允许插入转换、改变供数或独立替换任一收缩。 |
| 同处硬编码的“五参数／恰四操作”发射条件、按位置重抓原参数 | 删除 | 改写以 use-def 及语义前提配对；未匹配的合法 region 继续按普通 fold 降低。 |
| Block `transforms/ColumnPrefixes.cpp` 的三角列更新匹配 | IR | 保存列循环、对角值、读后写及外积更新。允许后续改变独立行的分块，GPU 出口实际遍历该循环。 |
| 公共头 `block/ir/{OwnerSource.h,FoldSource.inc,PairSource.inc}` 的 token 循环、mask、carry、片段拼接 | IR | Block 形成分段循环、有效域、slice/contract 和 yield；GPU 物理轴与目标语法在各自出口 legalize。 |
| 上述公共头根据语言选择 `tl.*` 或 `ct.*` | 删除 | 同一个 Block 检查点由两个出口独立消费，公共节点没有下游语言字符串。 |
| G6 审计：`block/decisions/strategies/Ordered.cpp` 把短逻辑片段公共补到 16 行 | IR | Block 写实际 owner 与有效行；具体 MMA 的最小形状由出口提供，在出口 legalize 形成 pad、mask 和有效写回，短行可选另一实现。 |
| CPU 公共头 `cpu/ir/Pack.h` 的 prefix_scatter 和 `IndexedEffects.h` 的领号匹配 | IR | 将谓词、局部前缀、有效计数、原子预留、旧值及唯一 scatter 连接成区域。只删除已证明被替代的计算，保留未匹配的可观察写和空行计数。 |
| CPU `decisions/MatrixPanels.cpp` 用 packing 标签拼 host，涵盖 GEMM、INT8、Q4、FP8、Mamba | IR | 在值任务上声明输入表示与共享消费者，形成 per-call 的分配、pack、尾部、join 和 release。Weft 私有 realizer 在早出口兑现；Mojo 消费 buffers，保留栈／堆决定。 |
| CPU 转置的专用 host 遍历（调查 6 第 1 组），以及 emitter 临时恢复遍历次序 | IR | 结构候选为源行、目的行、二维 tile；模型取舍写入 CPU task 的循环、stride、输出归属，导出 host 按其分工。 |
| G6 审计：`cpu/decisions/Ordered.cpp` 只给 MLA 窗口容量；Mojo `emit/OwnerBatch.inc` 现场选保存／重放值 | IR | CPU 写出保存哪个 SSA 值、窗口作用域、分配寿命、写入、读取和超窗时的重算分支；Weft 和 Mojo 消费同一决定。以 1024/1025 窗口交界验证。 |
| DSA `decisions/Ordered.cpp`、`dsa/ir/ScoreSupply.h` 见 score_panel 即固定十二槽 | IR | 从当前值、完成依赖及可见范围分配实际片段，容量按行核对。分数和转置缓冲构成真实 alloc/use/wait，允许独立重用。 |

### 各出口（顺序也是迁移顺序）

| 出口内源码及行为 | 去向 | 接管方式 |
|---|---|---|
| Triton `emit/Ordered.cpp` 的 Cholesky、scorePanel 模板 | 删除 | 消费上述列循环及 attention region。Triton legalize 形成 dot、条件、地址、carry 和 store 并登记逐 op 打印。 |
| Triton `emit/Matrix.cpp` 的 GEMM/batch/ragged、INT8、Q4、FP8、窗口收缩 | IR | tile 循环、间接路由、格式访问、scale 分派及 epilogue 在 legalize 成形。 |
| 同处 MMA、整数 dot、合法精度的 dot 变体 | 目标操作 | 绑定下游 dot 实现；登记输入／累加类型、几何、目标与数值条件。 |
| Triton `emit/Exit.cpp` 的 private counts、纵向 merge、区间 fill、scatter 写回 | IR | 清零、分片循环、合并、位宽及写轴显式。复用 `atomic_bins` 已有子 region；把 bin_add 内剩余的 load/地址/effect 拆到正确位置。 |
| cuTile `emit/Ordered.cpp` 的 Cholesky、score panel 整核输出 | 删除 | 沿用同一 Block 区域，在 cuTile 映射 view、边界 mask、有效矩阵片段。 |
| cuTile `emit/Matrix.cpp` 的矩阵访问、记录解码、窗口、MX 与慢路分派 | IR | 由格式和 capability 生成 tile load/gather、循环、转换和写回，保留两侧独立分组轴。 |
| 同处 `mma` / `mma_scaled` | 目标操作 | 区分架构、dtype、块形状、累加及 scale 编码。 |
| cuTile `emit/Kernel.cpp` 的 `private_bin_store`、key_counts、sum_columns、fill_ranges | IR | 实例私有或全局计数由 Block 选；归零、分片、投影、原子、同步和合并在出口 IR 可见。原生 Tile IR 计数原语仅在树外扩展试验，采用前评审产物 ABI。 |
| Mojo `emit/Ordered.cpp` 的 score_microtile 和 `OwnerBatch.inc` 重新识别 SSA/分配缓存的分支 | 删除 | 逐 op 消费 CPU 的计算、窗口与重算决定，分数可驻留的条件由 CPU 模型解释。 |
| Mojo `emit/Source.cpp` 的 pack_members、private_bins、indexed_copy、copy、领号、range fill、纵向求和及 Welford 在 yield 才补写的 lane 更新 | IR | legalize 形成循环、scan/写入、向量主体与尾段。Welford 显示 count、mean、M2 的 combine 与空域 identity。 |
| Mojo `emit/Matrix.cpp` 的 Q4、INT8、普通面板与按行写回 | IR | packing/前处理、外层遍历和 epilogue 保持可见，小面板的指令选择绑定下行目标操作。 |
| 同处向量整数点积、FMA、可用的 x86 BRGeMM | 目标操作 | 以 SIMD/dot 或显式外调实现一个收缩，保留数值与 layout 前提。 |
| Weft `emit/Task.cpp` 的 reserve_tickets、pack_members、private_bins、vertical_sum 与 fill_ranges 模板 | IR | legalize 形成任务起始地址、局部存储、初始化、循环、原子、发布及 join；只在已有 Canonical Weft 中拼写。 |
| Weft `emit/Matrix.cpp` 按空 K、reduction-packets、Q4、普通面板生成整段程序 | IR | 零 K 的 identity 和 epilogue、记录迭代、打包布局及物化位置都写入 IR；下游 `.weftir` 保留 dot。保留已逐 SSA 处理的 record/replay，删除重复匹配。 |
| Weft 的点积与 vendor BRGeMM 实现 | 目标操作 | x86 的实现选择交给下游 triton-weft 的 dot lowering；RVV 消费它的向量/矩阵能力。 |
| BANG C `emit/ScorePanel.h` 与 `emit/Ordered.cpp` 按 marker 发整段 attention | 删除 | DSA 保留 QK、归一化、PV 间的 DMA、等待及存储依赖，BANG C legalize 绑定到物理空间。 |
| BANG C `emit/Matrix.cpp` 的矩阵模板与 writeback | IR | 按行特化合法尺寸，形成两侧供数、量化、累加、路由或普通 epilogue 与写回。 |
| BANG C 矩阵内核、局部 collective 及 `emit/RowMath.inc` 的除法快慢路 | 目标操作 | 出口内建或 `runtime/bangc` 微核，声明片上 scratch、内部同步、IEEE 边界与回退成本；保留严格除法对歧义 lane 的结果。 |
| BANG C `emit/Kernel.cpp` 的压缩、直方图、分段、成员搬运、地址和写回模板 | IR | 任务所有权、跨片前缀和合并在 DSA；pitch、NRAM/WRAM/SRAM、分段载入、scatter 及真实完成点在出口。已有单双缓冲的 await/drain 继续遍历子 IR。 |

规模依据是调查时三个公共节点约 640 行 B 类、五出口约 2,100 行 B 类，共约 2,740 行**待检查和替代的旧逻辑**。其中共同模板按来源计一次，历史分组的增删不再相加。实际工程为三个节点的共享变换、五次出口的 IR／ABI／原生接通、一个下游 x86 lowering 及下述库和量化工作；行数没有计入新方言、运行时或下游改动。

## 二、量化：从格式推导访问和解码

KIR 持有作者可见的逻辑值与数值合同。可复用的**物理存储类型描述**归能被 CPU、Block、DSA 共同使用的祖先（KIR 单元公开的 storage type）；DSA 的 `storage` 级只是一个分支，CPU 不从它导入格式。类型由相应产品组件登记，出口查询同一份描述。框架只提供目标操作合同。描述给出逻辑轴到记录／组／成员的映射、块与超块长度、记录字节数、端序、字段偏移、scale/min 位宽与跨字节拼接、nibble 次序、有符号性、scale dtype、分组轴及尾部规则。解码先用有宽度的 IR 索引定位记录，再形成 load、extract、移位、扩展、转换和算术。硬件 primitive 可以替代等价的子图。

| 首批格式 | 由描述推导的关键事实 |
|---|---|
| 普通 i8 | 有符号 i8 × i8、i32 累加，无 scale/min。带分组 scale 的整数格式独立描述。 |
| Q4_K × Q8_K | Q4_K 每 256 项 144 B，8 个 32 项组，共用 f16 的 d/dmin 及 12 B 的六位 scale/min 字段；按 64 项一组的前后半区取低／高 nibble。Q8_K 每 256 项 292 B，有 f32 ds、256 个 i8 及 16 个 i16 小计。按首个最大绝对值编码、最近偶数舍入；组内整数点积与偏移修正后按规定的 FP32 舍入点合成。 |
| E4M3FN × E8M0 | 每 32 项一个 scale，A 的逻辑轴是 [M,G]，B 是 [N,G]，接受显式转置形成的 view；E8M0 零码和 255 的特殊解释保留。scale 与指针都有独立 stride，索引乘加使用已验证位宽。 |

**模型候选。** ① 解码后做浮点收缩，计解码、工作集和面板复用；② 在格式允许时做整数点积和 scale/min 修正，计解包、转换和局部 dot；③ 在作者显式允许且能证明权重不可变时离线重排为实现需要的布局，再使用合法 dot。记录预打包来源、格式版本、原布局/ISA，变更后重建；普通可写／别名权重只允许调用内准备。候选各自证明数值等价，例如 Q4_K 直接逐元素反量化再点积可能跨越规定的舍入点，应淘汰。FP8 不能仅凭位宽当整数点积。K-quant/MX 的特殊语义由产品处理，内层满足条件时才接 BRGeMM。Bench 继续独立按规范判定字节、分组轴、间接地址、次正规数和 NaN。[记录格式与数值步骤](../design/kir.md)

## 三、按后端迁移，同阶段收口性能

| 次序 | 一次交付要接通的真实路径 | V2-6.P 同期处理的重点 |
|---|---|---|
| 1 Triton | 先让 Block 的 fold、列更新、mask 和 contraction 能逐 op 打印；保留当前 Triton 源、ABI、guard 和配置行格式。验证合法终端检查点可直接发射，并注入窄 stride 查边界拒绝。 | H100 softmax、转置、GQA、flash backward；5090D 对照同一 Block 程序及成熟 Triton。 |
| 2 cuTile | 复用第一步的 Block 结构，出口绑定目标几何、array view 和 MMA。两出口都接通后删公共双语 printer。 | H100 RMSNorm、backward、分页 MLA 的 KV 复用和冷编译；私有直方图用扩展包与现出口/CUB 测完整调用。 |
| 3 Mojo | CPU 在值上分工，形成可遍历的任务、窗口、准备与存储。native 与 x86 vendor 共享外层循环及同步／异步 host 合同。 | 转置、Welford、因果 GQA、分页 MLA、Mamba chunk；量化检查 CPU host 的实际 packed bytes。 |
| 4 Weft | 从 `cpu@tasked` 接管，私有 realizer 兑现供数，产出 typed host、task 和 bridge；对接 triton-weft 的 x86/RVV JIT。两 CPU 出口迁完再移除旧 host 模板。 | x86 因果 GQA 与矩阵；SG2044、RVV V100 的跨步访问、量化和向量尾段。用现有 oracle 检验实际输出。 |
| 5 BANG C | 从 `block@realized` 经 DSA 决定槽、寿命和完成点，绑定 MLU370 存储和已有微核，逐 op 发射 `.mlu`；核对 CNCC、CNRT queue、完成句柄。 | softmax 先做设备分段计时，区分提交、供数与严格除法；RMSNorm、矩阵供数、attention 和反向冷构建分别对照 CNNL 或已有同设备实现。 |

每波先交付一条可从同一检查点对照的路径，再迁本出口表中的剩余项。以既有 Bench 的同设备、同输入、同数值许可的完整调用看 V1 与成熟实现，区分冷编译、首次选行与热执行。用两个不同程序验证共有分析；日常只查受影响项，波末跑已接通的 1–6 组矩阵。新旧产物可用 derive/compare 检查，较小回退记回既有 P 表；超过约两倍先定位并修。保留图重放、CPU 原生区间、状态恢复和已验证的地址／空域行为。待五个出口都无 module 回调消费者，再由母仓移除废弃 API。[既有性能读数和待办](v2-6.md)

按可独立评审的改动估算，Triton 连同首批 Block/格式工作约 4–6 批，cuTile 约 3–4 批，Mojo 连同 CPU 公共任务和存储约 4–6 批，Weft 连同下游改造约 3–5 批，DSA/BANG C 约 4–6 批，共约 18–27 批。每批对应上表的一类真实计算及其原生调用，不以文件数计。最大变数是 Weft task 的外部符号链接、数值规则限制可用的 BRGeMM 类型，以及供数重写后的 MLU 真机性能。

### CPU vendor 落地条件

- **x86 的准备工作。** 系统头为 oneDNN 2.2.4；工作区 3.13.4 的构建缓存是 `ONEDNN_EXPERIMENTAL_UKERNEL=OFF`。迁移时用独立安装前缀以 ON 重建 3.13.4。doctor 同时探测头的功能宏、动态库版本及实际导出的 BRGeMM create/finalize/generate/execute/destroy 符号；用真实 tile 调用核对输入 packing、scratch 和线程级 ISA 权限。仅找到头文件不足以启用。
- **Mojo。** 节点给出外层分块和共享准备；出口在 IR 绑定一个有类型的 BRGeMM 调用，Mojo 通过 `external_call` 接产品 C ABI。描述符和 JIT 内核按实现与形状复用，入口对实参和 scratch 负责；需要 AMX context 时遵循执行线程的设置与释放，数值合同禁止的 FTZ 或重结合会排除此候选。
- **Weft。** 已获准修改下游 triton-weft。在 x86 dot/contract lowering 兑现实现绑定、ABI 和必要 packing 要求，调用 ukernel；Weft host 继续只调度 task 和 join。以同一个 Canonical Weft 输入分别走 native/vendor，核对 task IR、真实库调用及原生结果；没有在 host 里替换整个 dot 所在算法的旁路。
- **SG2044。** 2026-11-23 在板上只读实查：`pkg-config --modversion openblas` 为 0.3.25，`pkg-config --cflags --libs openblas` 返回 CBLAS 头目录和 `-lopenblas`；`libopenblas.so` 导出 `cblas_sgemm`，实际 f32 的 [1,2]·[3,4] 得 11。系统另有 64 位整数接口变体和本地 GGML 构建。动态链接器缓存未登记 oneDNN/libxsmm。OpenBLAS 整矩阵调用只能作 library 候选或对照，没有据此宣称板上有 BRGeMM ukernel。vendor 默认下继续使用合法的 RVV native 候选；将来需新探测其 ABI、线程行为和数值能力才可登记另一实现。

**已批准。** 首次迁移以 Triton 检验共享 Block、以第二个 GPU 出口检验它确实可复用；CPU vendor 首先接 oneDNN 对应的合法小集合。整库只在显式 library 档和完整 region/effects 匹配时参加，与 native/vendor 一起计入 packing、scratch、同步和初始化成本。离线权重重排等作者冻结语义明确后开放。
