# Triton 与 cuTile 性能设计（SM90 / SM120）

本文档按“原理 → 候选 → 代价项 → 机器参数 → 验证与回滚闭环”记录 `levels/execution/block`、`exits/triton` 与 `exits/cutile` 在 NVIDIA Hopper（H100，`sm90`）与 Blackwell（RTX 5090D，`sm120`）上的性能模型与迭代路径。

## 1. 原理

GPU 执行性能由三层显式决定共同支配：

1. **CTA 数据归属与片上复用**：
   - 矩阵收缩按输出驻留块 `(BM, BN, BK)` 切分，配合硬件异步 2D/ND 张量描述符搬运（Triton `tl.make_tensor_descriptor` 与 cuTile `ct.load/ct.store(..., allow_tma=True)`）直接喂入 Tensor Core（`tl.dot` / `tl.dot_scaled`、`ct.mma` / `ct.mma_scaled`）。
   - 有序与注意力类程序在 `block.owner_space` 中区分单 owner 向量化面板（`tokens-S`）、多 owner 协同矩阵面板（`pair-WxS`）及无跨迭代依赖尾轴写循环分条向量化；在线 `(max, sum)` 单趟重缩放在满足 `reassoc` 授权与 `proveValueReplay` 无写冲突证明时消去二次全局读。
   - 行与集体归约在满足寄存器文件与片上容量预算时优先采用 `resident-R` 整行驻留（复用跨行广播权重与位置编码），超预算时切为 `stream-R` 分片归约；直方图在 Triton 上优先走片上 `private-merge`（`tl.histogram`），在无片上原子原语的 `tileiras` 上走 `global-atomic` 分区工作集。
2. **数值合同与硬件快路守卫**：
   - 算术与超越函数遵循 MLIR `FastMathFlags` 与函数级 `denormal-fp-math` 属性：默认严格走 `div_rn` / `sqrt_rn` / IEEE `exp` 及被掩码通道非有限值修复（`block.repair_masked_nan`）；当作者显式授予 `afn` / `arcp` / `nsz` 时，出口发射 SFU `exp2`、近似倒数及无冗余掩码的 TMA→MMA 直连路径。
   - MX8/E8M0 块缩放 FP8 收缩在 `scale_dispatch` 指数安全窗 `[87, 163]` 内走原生块缩放 Tensor Core（Triton `tl.dot_scaled`、SM120 cuTile `ct.mma_scaled`），含 `NaN` 或极端指数时回退至精确 E8M0 解码累加。
3. **32 位窄地址与计数循环证明**：
   - 经 `proveNarrowAddress` 证明有符号范围不溢出的张量坐标与多轴偏移分解降为 32 位整数算术，消除 GPU 上高昂的 64 位整数整除/取模序列；经 `proveTailAccess` 证明对齐无尾块及 `countedPanel` 证明行程不溢出的面板循环降为 32 位计数循环以启用软件流水。

## 2. 候选空间（Decision Families）

| 决策族 | 位置 | 枚举候选 | 影响范围 |
|---|---|---|---|
| `levels/execution/block::contraction` | `block@decided` → `block@realized` | 目标合法 `(BM, BN, BK)` 输出驻留块及出口 `num_warps` / `num_stages` 或 `occupancy` 行 | 稠密、批处理、路由、窗口、MX8 FP8 与 Q4_K 收缩 |
| `levels/execution/block::ordered-owners` | `block@decided` → `block@realized` | `pair-WxS`（协同矩阵折叠）、`tokens-S`（单 owner 贡献面板）、`shared-W`、`prefetch{4,16}-W`、`resident-W`、`columns-1`、`owners-W` | 注意力（因果/滑窗/ALiBi/分页解码）、交叉熵、扫描递推、Cholesky 等 |
| `levels/execution/block::structure` | `block@decided` → `block@realized` | `resident-R` / `stream-R`、`private-merge` / `global-atomic`、`source` / `destination` / `grid-tile` / `column-panels` / `segment-*` | 逐元素、转置、行归约、Welford、前缀和、直方图、分段归约 |
| `exits/triton::narrow-proven-addresses` | `triton@callable` | `proveNarrowAddress` 32 位坐标窄化 vs 64 位回退（cuTile 在 `@callable` 同步执行同类窄化） | Tile 访存与多轴步长分解 |

## 3. 代价项与局部估算器

- **节点级代价模型**：
  - `levels/execution/block::tile-time`：计入有效读写字节（扣除跨 `R` 行广播输入的片上复用）、并发上限下的波次量化（`waveQuant >= 1`）、尾块扇区利用率、子 CTA 线程束欠饱和惩罚、原子冲突与前缀交换步数。
  - `levels/execution/block::matrix-time`：结合矩阵块在 `M/N` 上的尾块浪费、驻留波次与出口矩阵发射估算。
  - `levels/execution/block::owner-issue`：比较单 owner 与协同 `pair-WxS` 的面板共享访存、片上缓存容量与 CTA 寄存器溢出字节。
- **出口局部估算器**：
  - `exits/triton::matrix-issues` / `exits/cutile::matrix-issues`：按 Tensor Core 指令粒度估算收缩发射步数。
  - `exits/triton::collective-steps` / `exits/cutile::collective-steps`：估算跨线程束归约、前缀扫描与直方图合并代价。

## 4. 机器参数与校准来源

校准文件：`exits/{triton,cutile}/capabilities/{sm90,sm120}.calibration.json`。
分别在 **NVIDIA H100 80GB HBM3（`sm90`，132 SM）** 与 **NVIDIA GeForce RTX 5090 D（`sm120`，170 SM）** 上通过独立合成微基准实测采集：连续读写带宽（`read_1`、`write_1`）、跨步读写带宽（`read_16`、`write_16`）、片上共享带宽（`read_1_shared_bw`）、内核启动与同步延迟（`launch`、`sync`、`dma_in_latency`）及算术吞吐（`compute`）。

## 5. 验证程序与迭代闭环

- **校准与留出集**：在 `rms_norm`、`layer_norm`、`count_partitions`、`dropout_norm_forward` 上核对参数行校准；在未参与拟合的留出集（非对齐 `transpose 353×701`、`rms_norm 257×2048`、`softmax 512×1536`、`swiglu 384×1536`）上通过 `compilerinfra.performance.accuracy` 检验 `selected_over_best`、成对排序准确率与 Top-k 召回率（`intentdsl_bench/output/experiments/v052-gpu-readiness/{5090d,h100}/`）。
- **单步派生与回滚验证**：通过 `compilerinfra_dev` 在开发工作区执行 `derive → adopt → revert`，核验 `inspect parity`（Triton `9/9` 实现、cuTile `sm120` `8/9` / `sm90` `7/9`，`0 todo`）与 `inspect readiness`（`revert.verified = true` 且与当前安装库指纹一致），每次路径变更同步通过第 1–6 组全量 30 个程序及第 7 组泛化算子数值回归。
