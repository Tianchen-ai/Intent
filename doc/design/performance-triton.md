# Triton 出口性能设计（SM90 / SM120）

本文档按“原理 → 候选 → 代价项 → 机器参数 → 验证程序”记录 `levels/execution/block` 与 `exits/triton` 在 NVIDIA Hopper（H100，`sm90`）与 Blackwell（RTX 5090 D，`sm120`）上的性能模型、类型化合同与迭代路径。

## 1. 原理

Triton 出口的执行吞吐由四项显式 IR 决定支配：

1. **CTA 数据归属、异步张量描述符（TMA）与双 Warpgroup 流水**：
   - 稠密、批处理、路由与窗口矩阵收缩按 `(BM, BN, BK)` 切分输出驻留块，绑定 `{num_warps, num_stages}` 参数行；对满足 16B 起始与步长对齐的张量视图，在 `triton_src.tile_program` 入口块外提并去重 `triton_src.make_desc`（`tl.make_tensor_descriptor`），由 `desc_read` / `desc_write` 直接驱动 TMA 异步搬运与 WGMMA / MMA 指令流水，消除循环内逐迭代描述符构造栅栏。
   - Hopper `sm90` 在 `OwnerMachine` 中开放双 warpgroup（`maxWarpgroups = 2`，`num_warps = 8`）协同矩阵面板（如 `pair-128x64`），按实占寄存器预算平衡双 warpgroup 计算吞吐与共享内存双缓冲驻留。
2. **有序区域向量化与单趟融合**：
   - `block.owner_space` 统一比较单 owner 宽贡献面板（`tokens-S`，标量折叠扩展至 `4096`）、多 owner 协同矩阵面板（`pair-WxS`）、`K` 槽有序选择链并行选优归约、下三角前缀内积的寄存器驻留列消元代换（`O(N^2)` 串行访存依赖化为 `N` 步无回读向量消元）、以及有界桶计数直方图 / 前缀扫描 / 稳定秩散射面板化。
   - 当作者授予 `reassoc` 且 `proveValueReplay` 证明两次遍历间无别名写冲突时，在线 `(max, sum)` 重缩放融合为单趟扫描；在 `afn` 许可下将循环不变量 `scale * log2e` 外提至循环外直接发射硬件 SFU `exp2`。
3. **行程序驻留、条带折叠与片上原子**：
   - 行与集体归约在满足寄存器文件预算时选 `resident-R` 整行驻留（跨 `R` 行复用广播权重与位置表），超预算时切为 `stream-R` 分片归约；等距同构投影折叠为 `triton_src.stripe_loop`；256-bin 直方图支持 `private-merge`（片上 `tl.histogram` + 纵向归并）与 `global-atomic` 双候选择优。
4. **数值合同与范围证明**：
   - 默认严格履行 `div_rn` / `sqrt_rn` / IEEE `exp` 及被掩码通道非有限值修复；MX8/E8M0 块缩放 FP8 在指数安全窗 `[87, 163]` 内走 `triton_src.scaled_mma`（`tl.dot_scaled`），异常指数回退精确解码；`exits/triton::narrow-proven-addresses` 基于 `proveNarrowAddress` 将有界 64 位坐标与步长分解窄化为 32 位算术。

## 2. 候选空间（Decision Families）

| 决策族 | 位置 | 枚举候选 | 影响范围 |
|---|---|---|---|
| `levels/execution/block::contraction` | `block@decided` → `block@realized` | 目标合法 `(BM, BN, BK)` 驻留块及 `{num_warps, num_stages}` 参数行（含 `scaled_contract`、`window_contract`、`q4k_dot`） | `gemm`、`batched_gemm`、`ragged_gemm`、`int8_gemm`、`fp8_gemm`、`q4_projection`、`mamba_chunk_state` |
| `levels/execution/block::ordered-owners` | `block@decided` → `block@realized` | `pair-WxS`（至 `128×64`）、`tokens-S`（至 `4096`）、`scatter-WxS`（可结合二维散射累加 `tl.atomic_add`）、`prefetch{4,16}-W`、`shared-W`、`resident-W`、`columns-1`、`owners-W` | 注意力全族、`cross_entropy`、`topk_rows`、`index_add`、`log_softmax_kl`、`retention_recurrence`、`gated_delta`、`triangular_solve`、`radix_pass`、`cholesky` |
| `levels/execution/block::structure` | `block@decided` → `block@realized` | `resident-R` / `stream-R`（1–8 warps）、`private-merge` / `global-atomic`、`source` / `destination` / `grid-tile` / `column-panels` / `segment-*` | 逐元素、`transpose`、`softmax`、`layer_norm`、`qk_norm_rope`、`histogram`、`embedding_bag`、`jagged_mean` |
| `exits/triton::narrow-proven-addresses` | `triton@callable` | 经 `proveNarrowAddress` 证明有符号不溢出的 `i32` 地址窄化 vs `i64` 保持 | Tile 指针寻址与多轴展平坐标分解 |

## 3. 代价项与局部估算器

- **节点级代价模型**：
  - `levels/execution/block::tile-time`：建模有效读写字节（按 `1/rows` 分摊跨行广播只读张量）、并发上限下的波次量化（`waveQuant >= 1`）、单 CTA `16K` 寄存器溢出上限、活跃线程束欠饱和惩罚及原子冲突步数。
  - `levels/execution/block::matrix-time`：结合 `(BM, BN, BK)` 尾块利用率、SM 驻留波次、L2/共享面板字节与局部矩阵发射代价。
  - `levels/execution/block::owner-issue`：解耦折叠体内外向量宽度（`foldBodyLanes`），计入循环携带状态与预取输入实占寄存器字数（`6144` 字预算）、CTA 内 L1 合并共享切片带宽、双 warpgroup 片段容量、下三角列消元步数及单 owner 面板化步数。
- **出口局部估算器与类型化合同**：
  - `exits/triton::matrix-issues`：按 Tensor Core 指令粒度估算 `native-dot`、`scaled-mma` 与 `row-collective` 发射步数。
  - `exits/triton::collective-steps`：估算跨 warp 树归约、包容扫描（`tl.cumsum`）、Welford 归并及片上直方图代价。
  - 类型化合同注册于 `exits/triton/ir/Implementations.cpp`（覆盖 `native-dot`、`scaled-mma`、`row-collective`、`moments-reduce`、`moments-merge`、`inclusive-scan`、`philox-uniform`、`cta-histogram` 等，parity `9/9` 全部实现）。

## 4. 机器参数与校准来源

校准文件：`exits/triton/capabilities/sm90.calibration.json`（H100 80GB HBM3，`132` SM）与 `exits/triton/capabilities/sm120.calibration.json`（RTX 5090 D，`170` SM），由 `exits/triton/capabilities/Rows.cpp` 按运行期 CUDA 设备架构自动匹配。
参数均由独立微基准实测拟合，包含连续/跨步全局读写带宽（`read_1`、`write_1`、`read_16`、`write_16`）、片上共享内存带宽（`read_1_shared_bw`）、内核启动与同步开销（`launch`、`sync`、`dma_in_latency`）及算术吞吐（`compute`）。

## 5. 验证程序与迭代闭环

- **模型准确度（`model-accuracy`）**：在留出集（非对齐 `transpose 353×701`、`rms_norm 257×2048`、`softmax 512×1536`、`swiglu 384×1536`）上通过 `compilerinfra.performance.accuracy` 持续核验 `selected_over_best`、成对排序准确率与 Top-k 召回率（`intentdsl_bench/output/experiments/v052-gpu-readiness/{5090d,h100}/`）。
- **代表性验证程序**：第 1–6 组全部 30 个 `program/*`（`gemm`、`fp8_gemm`、`q4_projection`、`attention`、`paged_decode`、`gated_delta`、`histogram` 等）与第 7 组泛化算子（`sliding_window_attention`、`alibi_attention`、`cross_entropy`、`topk_rows`、`qk_norm_rope`、`attention_unscaled_max`、`retention_recurrence`、`triangular_solve`、`radix_pass` 等），通过 `derive → compare → measure → adopt → revert` 闭环验证。
