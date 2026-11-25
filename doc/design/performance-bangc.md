# BANG C 与 DSA 性能设计（MLU370）

本文档按“原理 → 候选 → 代价项 → 机器参数 → 验证程序”记录 `levels/storage/dsa` 与 `exits/bangc` 在寒武纪 MLU370（`mtp_372`）上的性能模型与迭代路径。

## 1. 原理

MLU370 没有硬件统一缓存替内核自动重排数据，执行性能完全由以下三项显式决定支配：

1. **片上存储归属与预算**：每个 IPU 核具备独立的私有 NRAM（768 KiB，预算上限 `min(2/3 * NRAM, 512 KiB)`）与矩阵滤波空间 WRAM（512 KiB）。NRAM 承载流式输入槽、归约暂存与写回缓冲；WRAM 承载 64 通道卷积滤波块（`__bang_conv` / `__bang_conv_partial`）。
2. **DMA 供数与计算重叠**：1D/2D `__memcpy_async` 可将 GDRAM↔NRAM 传输与向量/矩阵计算重叠；但双缓冲会使输入与写回槽占用翻倍，缩小单步宽度或批大小，因此深度 1 与深度 2 必须根据实际搬运体积与计算强度比较。
3. **严格 IEEE-754 合同与快速路径边界**：
   - `div` / `sqrt` / `f16` 商由无条件对称精确残差端点认证保证 0 ULP，快路走向量指令，未过认证的极少数通道走标量精确修复；
   - `exp` / `exp2` 在安全指数窗内走 `__bang_active_exphp`，窗外走标量 `expf` / `exp2f` 修复；
   - BF16 矩阵收缩在默认 `denormal-fp-math=ieee,ieee` 下先由 `bf16_panel_ok` 检查正规指数窗 `[127-48, 127+46]`，通过则走 WRAM 矩阵引擎，否则回退至 NRAM IEEE 点积；当函数显式授予次正规冲零许可时直接走 WRAM 矩阵引擎。

## 2. 候选空间（Decision Families）

| 决策族 | 位置 | 枚举候选 | 影响范围 |
|---|---|---|---|
| `levels/execution/block::structure` | `block@decided` | `resident` / `stream`、行宽 `W ∈ [1..4096]`、行组 `R ∈ {1,2,16,32,64}`、`private-merge` 直方图 | 任务切分与片上驻留拓扑 |
| `levels/storage/dsa::supply` | `dsa@formed` → `dsa@supplied` | 深度 `depth ∈ {1, 2}`、共享只读 1D 向量驻留 `persistent`、异步写回 `async_writeback`、临时片段区间复用 `reuse_temporaries`、批大小 `batch`，以及 `library` 档位下的外部库供数 `library-cnnl` | 行与归约程序的 NRAM 槽位、DMA 流水与整算子库调用 |
| `levels/storage/dsa::panel-supply` | `dsa@formed` → `dsa@supplied` | 符合 NRAM/WRAM 容量的矩阵块 `(BM, BN=64, BK)`（`8×64×64` 至 `128×64×256`）及参与者核数，以及 `library` 档位下的 `library-cnnl` | 稠密、批处理、路由、窗口与量化收缩 |
| `levels/storage/dsa::ordered-supply` | `dsa@formed` → `dsa@supplied` | `panel-fold`（片上双遍 QK/PV WRAM 面板折叠）、`packed`（多 owner 打包流 `pack ∈ {2,4,8}`）、`general`（通用标量/向量流） | 注意力、线性注意力、Mamba/Gated Delta 等有序程序 |
| `exits/bangc::narrow-proven-addresses` | `bangc@callable` | 经 `proveNarrowAddress` 白名单证明不溢出的 32 位有符号坐标乘步长 vs 64 位回退 | GDRAM/NRAM 地址计算吞吐 |

### 外部库档位（CNNL `reuse="library"`）

`product.json` 将寒武纪 CNNL（`libcnnl.so` `>=1.15,<3`）登记为 `exits/bangc` 拥有的可选依赖 `cnnl`（探测钩子 `exits/bangc::cnnl-library` 校验动态库版本与全部 26 个 C API 符号，集成至 `doctor`）。`exits/bangc` 注册 5 项 `Reuse::Library` / `ImplementationSource::External` 类型化合同，默认 `reuse="vendor"` 自动以 `exceeds reuse=vendor` 排除；显式开启 `reuse="library"` 时，仅在作者数值许可与布局前提满足时与 `vendor` 候选同场按代价竞选（高档位允许低档位在小形状上胜出）：
- `exits/bangc::cnnl-matmul-f16`（`cnnlMatMul_v2`，FP32 累加，`ALLOW_TF32=0`，严格保留 FP16 次正规数）；
- `exits/bangc::cnnl-matmul-ftz`（BF16/FP32，要求函数级 `denormal-fp-math` 冲零许可）；
- `exits/bangc::cnnl-transpose`（`cnnlTranspose_v2`，8/16/32 位逐位精确二维转置）；
- `exits/bangc::cnnl-softmax`（`cnnlSoftmaxForward_v2`，要求 `fastmath` `afn,arcp,reassoc`）；
- `exits/bangc::cnnl-layer-norm`（`cnnlLayerNormForward`，要求 `fastmath` `afn,reassoc`）。

## 3. 代价项与局部估算器

- **节点级代价模型**：
  - `levels/storage/dsa::supply-cost`：结合每核工作集、工作集所处缓存层级（private / mid / shared 带宽）、DMA 启动固定延迟、双缓冲重叠收益及共享向量广播复用给分。
  - `levels/storage/dsa::matrix-supply-cost`：比较各 `(BM, BN, BK)` 面板的 GDRAM→NRAM→WRAM 搬运次数、核间负载均衡与尾块填充浪费。
  - `levels/storage/dsa::owner-supply-cost`：比较 `panel-fold` 的片上分数重放 + WRAM 吞吐、`packed` 流的连续向量化收益与 `general` 流的逐步访存开销。
- **出口局部估算器**：
  - `exits/bangc::dma`：按校准的 1 核 / 16 核读写带宽、DMA 同步壁垒与向量集体归约吞吐估算局部阶段耗时。
  - `exits/bangc::matrix-transfer`：估算 NRAM 转置、`NRAM2WRAM` 滤波装载、`__bang_conv` 矩阵发射及 BF16 指数窗守卫开销。

## 4. 机器参数与校准来源

校准文件：`exits/bangc/capabilities/mlu370.calibration.json`（配置名 `mlu370`，版本 `4`）。
所有参数均在 MLU370（`mtp_372`，`cncc v4.7.2`，16 block tasks，CNRT notifier）上通过独立合成微基准实测采集：
- 启动与同步固定开销：`launch`、DMA `sync_io` 壁垒；
- 分工作集（64 KiB / 1 MiB / 8 MiB / 256 MiB）与分并发（1 核 / 16 核）连续及跨步读写带宽：`read_{1,16}_{private,mid,shared}_bw`、`write_{1,16}_{private,mid,shared}_bw`；
- 向量超越函数、归约与 WRAM 卷积吞吐。

## 5. 验证程序与迭代闭环

- **校准与留出划分**：微基准独立拟合 `mlu370.calibration.json`；使用 `softmax`、`weighted_rms_norm`、`layer_norm`、`fused_add_rms_norm`、`transpose`、`gemm`、`batched_gemm`、`attention` 等跨类别程序检验候选排序与留出集 `model-accuracy`（所选/最佳比与成对排序准确率）。
- **单步派生与回滚纪律**：针对慢路径通过 `compilerinfra-dev derive` 派生候选或调整局部估算器，先在同一测量协议下核对 `check_protocol` 与第 1–6 组数值回归，再执行 `adopt`；若留出集退化或完整调用未获益，立即通过 `revert` 原子恢复至已验证安装指纹。
