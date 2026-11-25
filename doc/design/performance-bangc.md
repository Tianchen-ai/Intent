# BANG C 与 DSA 性能设计（MLU370）

本文档按“原理 → 候选 → 代价项 → 机器参数 → 验证程序”记录 `levels/storage/dsa` 与 `exits/bangc` 在寒武纪 MLU370（`mtp_372`）上的性能模型与迭代路径。

## 1. 原理

MLU370 没有硬件统一缓存替内核自动重排数据，执行性能完全由以下三项显式决定支配：

1. **片上存储归属与预算。** 每个 IPU 核使用私有 NRAM（768 KiB，预算上限 `min(2/3 * NRAM, 512 KiB)`）和 WRAM（512 KiB）。同簇的 4 个 IPU 共用 4 MiB SRAM。稠密收缩的完整块在 SRAM 为 A、B 各分配两个槽。MPU 同步拷入下一对面板时，IPU 从当前槽取数并发射 64 通道 WRAM 卷积，簇屏障完成后才交换槽位。分组量化收缩可以只双缓冲左面板，右面板由各核独立解码。块缩放收缩则由 IPU 协作解码，写入单份 A、B 共享面板，经过簇屏障再消费。`bangc@callable` 中的循环、地址选择、搬运和同步给出各自的执行顺序。`quant.mlu` 实现 NRAM 面板内的查表、缩放和排列。
2. **DMA 供数与计算重叠**：1D/2D `__memcpy_async` 可将 GDRAM↔NRAM 或 GDRAM↔SRAM 传输与向量/矩阵计算重叠；但双缓冲会使输入与写回槽占用翻倍，缩小单步宽度或批大小，因此深度 1 与深度 2 必须根据实际搬运体积与计算强度比较。
3. **严格 IEEE-754 合同与快速路径边界**：
   - `div` / `sqrt` / `f16` 商由无条件对称精确残差端点认证保证 0 ULP（含跨多行批处理的 `group_div_scalar_rn` 与无拷贝双精度残差 `BANG_TWODIFF_SE`），快路走向量指令，未过认证的极少数通道走标量精确修复；
   - `exp` / `exp2` 在安全指数窗内走 `__bang_active_exphp`，窗外走标量 `expf` / `exp2f` 修复；
   - BF16 矩阵收缩在默认 `denormal-fp-math=ieee,ieee` 下先由 `bf16_panel_ok`（单次分块按位或归并 `__bang_count`）检查正规指数窗 `[127-48, 127+46]`，通过则走 WRAM 矩阵引擎，否则回退至 NRAM/WRAM 精确 IEEE 点积；当函数显式授予次正规冲零许可时直接走 WRAM 矩阵引擎。

## 2. 候选空间（Decision Families）

| 决策族 | 位置 | 枚举候选 | 影响范围 |
|---|---|---|---|
| `levels/execution/block::structure` | `block@decided` | `resident` / `stream`、行宽 `W ∈ [1..4096]`、行组 `R ∈ {1,2,16,32,64}`、`private-merge` 直方图 | 任务切分与片上驻留拓扑 |
| `levels/storage/dsa::supply` | `dsa@formed` → `dsa@supplied` | 深度 `depth ∈ {1, 2}`、共享只读 1D 向量驻留 `persistent`、异步写回 `async_writeback`、临时片段区间复用 `reuse_temporaries`、批大小 `batch ∈ {1,2,4}`（含 `indexed_tile_read` 索引行收集批处理），以及 `library` 档位下的外部库供数 `library-cnnl` | 行与归约程序的 NRAM 槽位、DMA 流水与整算子库调用 |
| `levels/storage/dsa::panel-supply` | `dsa@formed` → `dsa@supplied` | 对 block 已选的块形比较供数方式与参与核数。`private` 为各核独立取数，`shared-tile` 为多核共同计算一个块，`shared-lhs` 为相邻列块复用左面板。后两者由 BANG C 根据矩阵访问、类型、簇宽与 SRAM 容量提供。`library` 档还可选 `library-cnnl`。 | 稠密、批处理、路由、窗口、融合尾声与量化收缩 |
| `levels/storage/dsa::ordered-supply` | `dsa@formed` → `dsa@supplied` | `panel-fold`（片上双遍 QK/PV WRAM 面板折叠）、`packed`（多 owner 打包流 `pack ∈ {2,4,8}`）、`general`（通用标量/向量流） | 注意力、线性注意力、Mamba/Gated Delta 等有序程序 |
| `exits/bangc::narrow-proven-addresses` | `bangc@callable` | 经 `proveNarrowAddress` 白名单证明不溢出的 32 位有符号坐标乘步长 vs 64 位回退 | GDRAM/NRAM 地址计算吞吐 |

### 外部库档位（`exits.bangc.reuse = "library"` 默认）

MLU370 没有类似 oneDNN BRGeMM 微核的厂商微核接口，CNNL 是唯一可复用的厂商整算子库实现。因此产品配置（`cmake/default.json`）与 `exits/bangc` 目标能力仅对 `bangc` 出口将默认复用档位设为 `library`（其余出口 `triton`、`cutile`、`mojo`、`weft` 保持 `vendor`；顶层显式传入 `derive({"reuse": "vendor"})` 或 `"native"` 时仍按框架规则统一覆盖所有出口）。

`product.json` 将寒武纪 CNNL（`libcnnl.so` `>=1.15,<3`）登记为 `exits/bangc` 拥有的可选依赖 `cnnl`（探测钩子 `exits/bangc::cnnl-library` 校验动态库版本与全部 26 个 C API 符号，集成至 `doctor`）。`exits/bangc` 注册 5 项 `Reuse::Library` / `ImplementationSource::External` 类型化合同，默认严格 `source` 数值语义保持不变，仅在作者数值许可与布局前提满足时与原生候选同场按代价竞选（`explain` 显式展示选中 `library-cnnl` 或被数值前提拒绝的原因）：
- `exits/bangc::cnnl-matmul-f16`（`cnnlMatMul_v2`，FP32 累加，`ALLOW_TF32=0`，严格保留 FP16 次正规数；默认选中大形状稠密 GEMM/QKV，带融合尾声或量化解码的算子自动走 `Union4` 原生路径）；
- `exits/bangc::cnnl-matmul-ftz`（BF16/FP32，要求函数级 `denormal-fp-math` 冲零许可；未授许可时自动走带 `bf16_panel_ok` 守卫的原生路径）；
- `exits/bangc::cnnl-transpose`（`cnnlTranspose_v2`，8/16/32 位逐位精确二维转置）；
- `exits/bangc::cnnl-softmax`（`cnnlSoftmaxForward_v2`，要求 `fastmath` `afn,arcp,reassoc`；默认无 `fastmath` 时仍走 0-ULP 原生路径）；
- `exits/bangc::cnnl-layer-norm`（`cnnlLayerNormForward`，要求 `fastmath` `afn,reassoc`；默认无 `fastmath` 时仍走 0-ULP 原生路径）。

## 3. 代价项与局部估算器

- **节点级代价模型**：
  - `levels/storage/dsa::pipeline-time`（v2）：按实际 rank-2 向量算子数（区分 per-row rank-1 标量算子）、每步同步 DMA 往返延迟（`dma_in_sync_latency` / `dma_out_sync_latency`）、按活跃核数受限的片外带宽屋顶线（`min(dma_*_single_bw, consumers * dma_*_bw / active_participants)`）、不参与循环重叠的一次性 `persistent` 共享向量预载（`shared_prologue`），以及与计算实际重叠的异步预取池（仅覆盖每步异步取数，不把尾同步写回或预载计入重叠）综合估算完整调用耗时。
  - `levels/storage/dsa::matrix-supply-cost`：比较各 `(BM, BN, BK)` 面板的 GDRAM→NRAM→WRAM 搬运次数、核间负载均衡与尾块填充浪费。
  - `levels/storage/dsa::owner-supply-cost`：比较 `panel-fold` 的片上分数重放 + WRAM 吞吐、`packed` 流的连续向量化收益与 `general` 流的逐步访存开销。
- **出口局部估算器**：
  - `exits/bangc::dma`（v2）：分离每步连续/分组 DMA 与一次性 `persistent` 共享读字节数及请求数，统计融合原生原语（如 `div + cast(f16)` 覆盖的 2 个 rank-2 算子）及校准向量归约/超越函数耗时。
  - `exits/bangc::matrix-transfer`：估算 NRAM 转置、`NRAM2WRAM` 滤波装载、`__bang_conv` 矩阵发射及 BF16 指数窗守卫开销。

## 4. 机器参数与校准来源

校准文件：`exits/bangc/capabilities/mlu370.calibration.json`（配置名 `mlu370`，版本 `5`）。
所有参数均在 MLU370（`mtp_372`，`cncc v4.7.2`，16 block tasks 与单核对比，CNRT notifier）上通过独立合成微基准实测采集：
- 启动与同步固定开销：`launch`、DMA `sync_io` 壁垒、以及 128−16 差分拟合的每步同步读写往返延迟（`dma_in_sync_latency`、`dma_out_sync_latency`）；
- 分工作集（64 KiB / 1 MiB / 8 MiB / 256 MiB）连续及跨步读写带宽、`16` 核与单核流式 DMA 读写带宽（`dma_in_bw` / `dma_out_bw`、`dma_in_single_bw` / `dma_out_single_bw`）；
- 向量超越函数、归约与 WRAM 卷积吞吐。

## 5. 验证程序与迭代闭环

短面板回归复用 `group7/mxfp4_gemm` 的作者程序、输入生成器和数值裁判，取 `M=512, N=4096, K=32`。此前选中簇级块后，单个缩放组只解码出 32 个 FP16 值，合法化落到私有实现并超出 NRAM 预算。协作实现现在在 SRAM 初始化时为每个 WRAM 行预留并清零填充部分，随后用带行距的 DMA 写入解码出的有效列。K 可以含一个或奇数个缩放组。

- **校准与留出划分**：微基准独立拟合 `mlu370.calibration.json`；使用 `softmax`、`weighted_rms_norm`、`layer_norm`、`fused_add_rms_norm`、`transpose`、`gemm`、`batched_gemm`、`attention` 等跨类别程序检验候选排序与留出集 `model-accuracy`（所选/最佳比与成对排序准确率）。
- **单步派生与回滚纪律**：针对慢路径通过 `compilerinfra-dev derive` 派生候选或调整局部估算器，先在同一测量协议下核对 `check_protocol` 与第 1–6 组数值回归，再执行 `adopt`；若留出集退化或完整调用未获益，立即通过 `revert` 原子恢复至已验证安装指纹。
