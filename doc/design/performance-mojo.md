# Mojo 与 CPU 性能设计（x86_64 AVX-512 / AMX）

本文档按“原理 → 候选 → 代价项 → 机器参数 → 验证程序”记录 `levels/execution/cpu` 与 `exits/mojo` 在 x86_64 多核处理器上的性能模型、类型化合同与迭代路径。

## 1. 原理

x86 多核 CPU 上的算子吞吐由以下四项显式执行决定支配：

1. **线程切分与缓存驻留**：每个物理核具备私有 L1d（48 KiB）与 L2（2 MiB），跨核共享 LLC。行、矩阵块和有序 owner 的任务粒度必须平衡 OpenMP 调度启动延迟、多核工作集落入私有缓存的比例以及写行伪共享。
2. **遍历方向与跨步访存不对称性**：在 x86 写分配缓存层级上，跨步聚集读（gather）配连续向量写（`destination` 遍历）或片内 `8×8` SIMD 蝴蝶转置（`tile` 遍历）显著快于跨步散射写（`source` 遍历）。
3. **计算引擎与寄存器分块**：稠密、批处理、路由与量化矩阵收缩可在寄存器分块 SIMD FMA、AVX-512 VNNI `vpdpbusd` 点积与 oneDNN 3.13+ BRGeMM / AMX 微核之间按实际块形与打包摊销选择。
4. **严格 IEEE-754 合同与快速路径边界**：
   - 默认函数级 `denormal-fp-math=ieee,ieee` 下，BF16 矩阵收缩保留次正规数精度，走拓宽 FP32 SIMD 累加或带正规性守卫的路径；仅当函数显式授予次正规冲零许可（`allow_subnormal_ftz`）时启用 `onednn-brgemm-bf16`；
   - 有序注意力与状态递推通过 `proveValueReplay` 证明无别名写干扰后兑现 L1/L2 分数缓存或受守卫的 `score_window` 回放（超窗分支零回读自动重算），并在 `afn` 许可下将 `exp2(x)` 降为向量 `exp(x · ln 2)`。

## 2. 候选空间（Decision Families）

| 决策族 | 位置 | 枚举候选 | 影响范围 |
|---|---|---|---|
| `levels/execution/cpu::partition` | `cpu@formed` | 行粒度 `grain ∈ {1, 2, 4, 8}`、SIMD 宽度 `{64, 128, 256}`、循环展开 `unroll ∈ {1, 4}`、按键分片/私有栈直方图、单 owner 原子消除 | 逐点、行归约、Welford 统计、前缀扫描、分段与直方图 |
| `levels/execution/cpu::permutation` | `cpu@formed` | `source`（源行顺序）、`destination`（目标连续写）、`tile`（`32×32` 块内 `8×8` SIMD 蝴蝶转置） | 矩形转置与重排 |
| `levels/execution/cpu::matrix-partition` | `cpu@formed` | 寄存器块与 BRGeMM 面板 `(MR, NR, KC)`（`2×16×32` 至 `12×32×128`）、host 预打包与 `onednn-brgemm-{f32,f16,bf16,s8,u8s8}` / `simd-fma` / `vnni-dot4` / `q4k-{vnni,simd}` | 稠密、批处理、路由、FP8、INT8 与 GGUF `Q4_K` 矩阵收缩 |
| `levels/execution/cpu::ordered-owners` | `cpu@formed` | 独立粒度 `grain ∈ {1..256}`、`retained-scores` 全量分数缓存、`shared-interval-owners`（16-owner 连续批处理 + 1024-token `score_window`） | 注意力、分页解码、线性注意力、Gated Delta、三角求解与有序归约 |
| `exits/mojo::narrow-proven-addresses` | `mojo@realized` | 经 `proveNarrowAddress` 白名单证明不溢出的 32 位有符号坐标与偏移计算 vs 64 位回退 | Mojo 终端循环与内存寻址开销 |

## 3. 代价项与局部估算器

- **节点级代价模型**：
  - `levels/execution/cpu::task-time`：结合 OpenMP 启动/同步开销、活跃 worker 利用率、每核工作集对应的缓存层级读写带宽（`private` / `mid` / `shared` / `dram`）与向量算术吞吐。
  - `levels/execution/cpu::permutation-time`：利用实测 `panel_{gather,scatter}_*_bw` 与奇偶跨距 `span_{gather,scatter}_*_bw` 比较 `source`、`destination` 与 `tile` 遍历的成对搬运耗时和缓存行触碰数。
  - `levels/execution/cpu::matrix-work` 与 `ordered-tasks`：比较面板打包摊销、核间负载均衡、16-owner 共享区间访存复用及分数重算开销。
- **出口局部估算器**：
  - `exits/mojo::register-block`：区分 oneDNN BRGeMM/AMX 微核吞吐、AVX-512 VNNI `vpdpbusd`、寄存器分块 SIMD FMA 与 K 维拆分开销。
  - `exits/mojo::simd-row-primitive`：估算行归约、Hillis-Steele 扫描、Welford 归约、`8×8` 转置、Philox-4x32 微核及循环展开收益。

## 4. 机器参数与校准来源

校准文件：`exits/mojo/capabilities/x86.calibration.json`（8 核校准档，版本 `4`）与 `exits/mojo/capabilities/x86-1c.calibration.json`（单核校准档，版本 `4`），由 `runtime/cpu/binding/profile.py` 按运行期 CPU 亲和集与线程数自动匹配。
全部 71 项参数均在 `Intel(R) Xeon(R) Platinum 8470Q`（AVX-512 / VNNI / BF16 / FP16 / AMX，LLC 105 MiB）上由 `intent_bench.calibration` 独立微基准实测采集，涵盖分缓存层级连续/跨步带宽、面板/跨度 gather-scatter 带宽、线程启动同步延迟及向量计算吞吐。

## 5. 验证程序与迭代闭环

- **校准与留出划分**：合成微基准独立拟合 `x86` / `x86-1c` 校准参数；使用校准形状与未参与调参的留出形状（奇数跨距矩形转置、不同隐藏维行归约与注意力工作负载）通过 `compilerinfra-dev model-accuracy` 验证完整候选集与行域的所选/最佳耗时比（`selected_over_best`）与成对排序准确率（`pairwise_accuracy`），全部通过 `check_protocol` 可比性核验。
- **单步派生与回滚纪律**：使用 `derive` 派生结构配置或源码补丁，在同核集同协议下执行 `measure` 与第 1–6 组全量数值回归，确认受益后 `adopt`，并通过真实 `adopt → revert` 验证源码与安装库指纹的原子恢复。
