# Weft 与 CPU 性能设计（x86_64 与 RISC-V RVV 1.0）

本文档按“原理 → 候选 → 代价项 → 机器参数 → 验证程序”记录 `levels/execution/cpu` 与 `exits/weft` 在 x86_64 及四类 RISC-V Vector（SG2044、RVV V100、SpacemiT K3 X100、K3 A100）平台上的性能模型、类型化合同与迭代路径。

## 1. 原理

`exits/weft` 将 `cpu@tasked` 程序降低为显式的双区终端表示（`task.weftir` 内核区 + `host_c` 调度与打包区），其跨架构性能由四项物理因素决定：

1. **向量寄存器预算与动态展宽面板（VLEN / LMUL）**：x86 AVX-512（512-bit，32 寄存器）与 RISC-V RVV 1.0（SG2044 128-bit、V100/K3-X100 256-bit、K3-A100 1024-bit，`LMUL` 上限与 `zvfbfmin` 支持各异）具有截然不同的寄存器压力边界。`capabilities` 依实测 `vector_bits`、`vector_registers` 与软件 BF16 RNE 转换寄存器需求自适应界定 `wideningPanel` 与矩阵块候选，由 `exits/weft::native-panels` 对溢出风险计入惩罚。
2. **Host 显式预打包与路由散射**：所有 OpenMP 并行面板打包（A/B F32/BF16 展宽、VNNI S8 四元组重排）与 MoE 路由累加写回均在 `weft@realized` 显式物化为 `weft_host` 操作，经 `proveTailAccess` 消除可证明整除维度的标量尾部分支，经 `proveNarrowAddress` 收窄 32 位偏移。
3. **跨架构缓存与跨步访存特性**：x86 与 K3 X100/SG2044/V100 的跨步读（`vlse` / gather）与跨步写（`vsse` / scatter）带宽比差异显著；按实测每设备校准档由 `permutation-time` 自动在 `destination`、`tile` 与 `source` 间择优。
4. **严格 IEEE-754 合同与微核对齐**：
   - x86 上对满足布局与次正规合同的收缩绑定 `triton-weft >= 0.2.0` 的 `onednn-brgemm-{f32,f16,bf16,s8,u8s8}` 微核，默认 `ieee,ieee` 下 BF16 自动走拓宽 `weft-f32-panel-fma` 以保持次正规数精确，仅在显式 FTZ 许可下启用 `onednn-brgemm-bf16`；
   - RISC-V 上声明不支持 x86 专属 BRGeMM/VNNI，统一走原生 RVV `weft-simd-fma`、`weft-f32-panel-fma` 与声明式 `encoding_decl` `weft-q4k-dot`；
   - 行程序对齐 L1d 单行多次归约缓存、Welford 三元组归约、前缀扫描、区间票据预留与 10 轮 Philox-4x32 连续四通道复用。

## 2. 候选空间（Decision Families）

| 决策族 | 位置 | 枚举候选 | 影响范围 |
|---|---|---|---|
| `levels/execution/cpu::partition` | `cpu@formed` | 自适应行粒度 `{1, 2/4, 32}` × 列宽 `max(32, 4·V)`、按键分片/私有缓冲预留 | 行归约、归一化、扫描、直方图、路由对齐与 Philox RNG |
| `levels/execution/cpu::permutation` | `cpu@formed` | `source`、`destination`、`tile`（二维分块拷贝） | 跨架构矩阵转置与重排 |
| `levels/execution/cpu::matrix-partition` | `cpu@formed` | 按向量宽度 `V` 缩放的 `(2..8) × (1..4)V × {32,64,128}` 面板、`weft-simd-fma` / `weft-f32-panel-fma` / `weft-q4k-dot` / x86 `onednn-brgemm-*` | 稠密、批处理、路由、窗口、INT8/FP8 与 `Q4_K` 收缩 |
| `levels/execution/cpu::ordered-owners` | `cpu@formed` | 独立 owner 粒度、`retained-scores`、`shared-interval-owners`（二维 `outer_contract` QK/PV 面板与 `proveValueReplay` 分数窗口回放） | 注意力、分页解码、线性注意力、状态空间与三角求解 |
| `exits/weft::narrow-proven-addresses` | `weft@realized` | 基于 `HostParallelTilesOp` 整数值域接口与 `proveNarrowAddress` 的 32 位有符号地址收窄 | Host 面板指针与路由偏移计算 |

## 3. 代价项与局部估算器

- **节点级代价模型**：复用 `levels/execution/cpu::{task-time, permutation-time, matrix-work, ordered-tasks}`，输入来自当前激活设备的校准事实。
- **出口局部估算器**：`exits/weft::native-panels` 建模 Weft `outer_contract` 寄存器累加块占用、RVV `LMUL` 寄存器组溢出惩罚、host 预打包内存带宽摊销以及 x86 oneDNN `SEQ` BRGeMM 微核吞吐。

## 4. 机器参数与校准来源

`exits/weft/capabilities/` 随源码交付全部六类目标环境的独立实测校准档（版本 `4`，均含 71 项微基准事实与 LLC 容量溯源）：
- `x86.calibration.json` / `x86-1c.calibration.json`：`Intel(R) Xeon(R) Platinum 8470Q`（8 核与单核，AVX-512，LLC 105 MiB）；
- `sg2044.calibration.json`：算能 SG2044（`0x5b7/0x80000000090c0d00`，RVV 128-bit）；
- `rvv-v100.calibration.json`：进迭时空 V100（`0x710/0x8000000058000002`，RVV 256-bit，LLC 2 MiB）；
- `k3-x100.calibration.json`：SpacemiT K3 大核 `Spacemit(R) X100`（RVV 256-bit，LLC 4 MiB）；
- `k3-a100.calibration.json`：SpacemiT K3 能效核 `Spacemit(R) A100`（RVV 1024-bit，LLC 1 MiB）。

## 5. 验证程序与迭代闭环

- **校准与留出划分**：微基准独立生成各硬件校准档；在校准集与留出集上通过 `compilerinfra-dev model-accuracy` 验证 `weft` 出口完整候选与行域的 `selected_over_best` 及 `pairwise_accuracy`（经 `check_protocol` 校验运行回执一致性）。例如接入 `k3-x100.calibration.json` 后，`program/transpose` 自动由回退 `tile` 切换为实测最优 `destination` 遍历。
- **单步派生与回滚纪律**：通过 `derive → measure → adopt → revert` 闭环验证候选采纳与原子回滚，`inspect readiness` 核验当前安装指纹与 `last_revert.json` 一致。
