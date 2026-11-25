# cuTile 出口性能设计（SM90 / SM120）

本文档按“原理 → 候选 → 代价项 → 机器参数 → 验证程序”记录 `levels/execution/block` 与 `exits/cutile` 在 NVIDIA Hopper（H100，`sm90`）与 Blackwell（RTX 5090 D，`sm120`）上的性能模型、类型化合同与迭代路径。

## 1. 原理

cuTile 出口面向 `cuda.tile` / `tileiras` 编译器，其执行吞吐由四项显式 IR 决定支配：

1. **结构化矩形视图（`rect_read` / `rect_write`）与硬件 TMA 门控**：
   - 凡经 `proveTailAccess` 证明对齐无越界且满足 16B 步长对齐的 2D–5D 矩阵面板、owner 矩形、行/条带切片及跨 owner 共享的连续 2 幂 `slice_read`，均在 `exits/cutile/legalize` 中降为结构化 `ct_src.rect_read` / `ct_src.rect_write`（`ct.load` / `ct.store`），避免通用散射/聚集 `ct.gather` / `ct.scatter` 的逐通道地址展开开销。
   - 多行驻留与收缩面板在单 tile `≤ 16 KiB` 且非单行（`rows > 1`）时开启 `allow_tma=True`；单行（`rows = 1`）块及紧接流水化 TMA 读循环后的 4D 写回自动关闭冗余 TMA，规避共享内存往返开销与 `tileiras 1.6` mbarrier 同步限制。
2. **Tensor Core 块缩放收缩与架构分级能力**：
   - 稠密、转置右操作数（`transpose_rhs` → `ct.transpose`）、批处理、路由、窗口与分组打包权重（`GroupedWeightRhs` 紧凑 `rect_read` + 多维重塑移位解包）收缩降为 `ct.mma` 并绑定 `{num_ctas, occupancy, opt_level}` 参数行；Blackwell `sm120` 在 MX8/E8M0 安全指数窗内启用原生硬件 FP8（`e4m3`）与打包 FP4（`e2m1`）`ct_src.scaled_mma`（`ct.mma_scaled`），Hopper `sm90` 因硬件无块缩放 MMA 原语，按能力表自动降为安全窗内 `BF16` `ct.mma`。
3. **Tile-IR 循环形态合法化与展开预算控制**：
   - `exits/cutile/legalize/Views.cpp` 在编译期显式判定每个 `scf.for` 的 `ct_src.loop_kind`（`range_const`、`range_i32` 或 `while`），使有符号 32 位有界循环生成原生 `range(...)` 以启用 `tileiras` 软件流水。
   - 针对 `tileiras` 中每个 tile 访存携带独立 CTA 描述符的特性，`OwnerMachine` 将分组预取上界设为 `maxGroupReads = 4`（配合 `ArrayPanels` 连续 1D `rect_read` + 寄存器广播），对大标量无别名空间开放 `owners-256` / `owners-512`，并与 Triton 共享在线 `(max, sum)` 单趟重缩放、有序选择链并行归约、下三角寄存器列消元及有界桶计数/扫描/稳定秩散射面板化。
4. **数值合同、寄存器预算精确半精度折叠与有界地址收窄**：
   - 除法与指数严格区分 MLIR `FastMathFlags`（`arcp` / `afn`）与函数级 `denormal-fp-math`（仅在显式允许次正规冲零时附带 `ct_src.ftz` / `flush_to_zero=True`，绝不将 `nsz` 误作 FTZ）。
   - 当面板 FP32 中间转换通道数超出 `OwnerMachine::residentRegisters` 预算（如 `sm120` `24,576` 字）且整数码经区间分析证明落在目标半精度精确整数域（`f16` `[-2048, 2048]`、`bf16` `[-128, 128]`）时，`Views.cpp` 将 `truncf(mulf(sitofp(x → f32), extf(s → f32)))` 零误差折叠为半精度 `mulf(sitofp(x → half), s)` 以消除寄存器溢出；同时结合 `proveNarrowAddress` 与有界余数区间将 `ct_src.tile_address` 收窄至 `ct.int32`。

## 2. 候选空间（Decision Families）

| 决策族 | 位置 | 枚举候选 | 影响范围 |
|---|---|---|---|
| `levels/execution/block::contraction` | `block@decided` → `block@realized` | 架构合法 `(BM, BN, BK)` 块及 `{num_ctas, occupancy, opt_level}` 行（`sm120` 含 FP8/FP4 `scaled-mma`） | `gemm`、`gemm_nt`、`batched_gemm`、`ragged_gemm`、`int8_gemm`、`fp8_gemm`、`mxfp4_gemm`、`awq_int4_gemm`、`q4_projection`、`q6k_gemv`、`mamba_chunk_state` |
| `levels/execution/block::ordered-owners` | `block@decided` → `block@realized` | `pair-WxS`、`tokens-S`（至 `4096`）、`scatter-WxS`（可结合 1D 合并散射累加 `ct.atomic_add`）、`prefetch4-W`、`shared-W`、`resident-W`、`columns-1`、`owners-W`（大标量空间至 `512`） | 注意力全族、`cross_entropy`、`topk_rows`、`index_add`、`log_softmax_kl`、`retention_recurrence`、`gated_delta`、`triangular_solve`、`radix_pass`、`depthwise_conv2d`、`cholesky` |
| `levels/execution/block::structure` | `block@decided` → `block@realized` | `resident-R` / `stream-R`、`global-atomic` 分区直方图（`tileiras` 无片上原子原语）、`source` / `destination` / `grid-tile` / `column-panels` / `segment-*` | 逐元素、`transpose`、`softmax`、`layer_norm`、`qk_norm_rope`、`histogram`、`embedding_bag`、`jagged_mean` |

## 3. 代价项与局部估算器

- **节点级代价模型**：与 Triton 共用 `levels/execution/block::{tile-time, matrix-time, owner-issue}`，通过 `OwnerMachine`（`sm90` / `sm120` 寄存器文件比例、共享内存上限、`maxGroupReads = 4`）与 `RowCapacity`（`storeSectorBytes = 32`、独立列宽 `{8..128}`、无片上共享桶原子）向公共代价模型注入目标约束。
- **出口局部估算器与类型化合同**：
  - `exits/cutile::matrix-issues`：按 `ct.mma` / `ct.mma_scaled` 及 `row-collective`（`warpRowCap = 4` 对齐 `occupancy = 4` 行片段容量，超出时按每 SM 共享内存带宽计入跨 warp 归约代价）估算矩阵收缩发射代价。
  - `exits/cutile::collective-steps`：估算 `ct.sum` / `ct.max` / `ct.cumsum`、Welford 归并及全局原子分区归并步数。
  - 类型化合同注册于 `exits/cutile/ir/Contracts.cpp`（覆盖 `native-dot`、`scaled-mma`、`row-collective`、`moments-reduce`、`moments-merge`、`inclusive-scan`、`philox-uniform`、`workgroup-atomic-bins`；`sm120` parity `8/9`、`sm90` `7/9`，仅硬件或 `tileiras` 原语客观不支持项标为 `unsupported`）。

## 4. 机器参数与校准来源

校准文件：`exits/cutile/capabilities/sm90.calibration.json`（H100 80GB HBM3，`132` SM）与 `exits/cutile/capabilities/sm120.calibration.json`（RTX 5090 D，`170` SM），由 `exits/cutile/capabilities/Rows.cpp` 按运行期 CUDA 设备属性自动加载。
各项带宽（`read_1`、`write_1`、`read_16`、`write_16`、`read_1_shared_bw`）、启动同步延迟（`launch`、`sync`、`dma_in_latency`）与计算吞吐（`compute`）均由同设备独立合成微基准实测校准。

## 5. 验证程序与迭代闭环

- **模型准确度（`model-accuracy`）**：在未参与校准的留出形状（`transpose 353×701`、`rms_norm 257×2048`、`softmax 512×1536`、`swiglu 384×1536`）上通过 `compilerinfra.performance.accuracy` 检验 `selected_over_best` 与成对排序准确率（`intentdsl_bench/output/experiments/v052-gpu-readiness/{5090d,h100}/`）。
- **代表性验证程序**：覆盖第 1–6 组全部 30 个 `program/*` 与第 7 组泛化算子，通过 `derive → compare → measure → adopt → revert` 与 [test_gpu_audit_mig.py](file:///srv/rc/site-1/phdworks/intentdsl_bench/tests/test_gpu_audit_mig.py) 最小重现集持续验证。
