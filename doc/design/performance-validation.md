# 首轮结构模型验证划分

本轮在采集机器参数、运行结构候选和生成预测之前冻结以下清单。先前审计已经披露默认实现的性能和预期方向；本轮隔离候选排序与参数拟合。

| 集合 | 输入、设备与范围 | 用途 |
|---|---|---|
| 校准 | 仅合成独立微基准：连续读、连续写、跨步读、跨步写、空提交与同步；CPU 增加向量算术；MLU 增加本地到设备以及设备到本地的分尺寸 DMA。5090D、H100、x86、SG2044、RVV V100、MLU370 各自校准。 | 带宽、固定延迟及出口吞吐；记录实际步长、工作集、编译器、并发、波动。 |
| 留出 T | Bench 的 program/transpose，FP16 127×257；x86 Mojo/Weft、SG2044 Weft、RVV V100 Weft。 | 三种遍历完整调用、排序、P1。 |
| 留出 D | MLU370 的 softmax 64×1024 FP16 与 weighted_rms_norm 8192×4096 FP32，含 gain、inv_c 和 epsilon。 | 两种深度与工作划分；先存预测，再轮换顺序实测。 |
| 留出 R | program/relu，使用登记的作者输入；x86 Mojo/Weft。 | 同一访存分析在独立作者程序的适用性；相同物理访问的候选允许预测平局。 |

T、D、R 均不得用于拟合系数。Bench 按设备、输入、数值合同、线程集合分批，保留每一结构的三轮结果、配置 winner 和产物摘要。P1 单项比较同设备 V1 三轮范围。基于留出观测采用结构时，先报告采用前的预测表现；选后的成绩单独列出。

## 第二轮冻结

本轮仅以独立读、写、调度和 DMA 微基准校准参数。先保存各候选的预测和行域，再做三轮交错完整调用。上一轮的输入只用于旧回退的 P1 复测。

| 新留出项 | 固定合同 |
|---|---|
| T2a、T2b | FP16 紧凑矩阵转置，193×389 和 1024×1536，输出独立且连续；x86 Mojo、x86 Weft、SG2044 Weft、RVV V100 Weft |
| P2 | FP16 三维循环置换，输入 7×19×65，输出 19×65×7，映射 `out[j,k,i]=x[i,j,k]`，相同四个 CPU 执行组合 |
| R2 | indexed RoPE，FP16，B=1、S=97、Q/K 头数 32/8、宽度 128；Q/K 由最后一维步长为 2 的非重叠 view 提供，检查原地写回、奇数位置哨兵与数值。相同四个 CPU 执行组合 |

每项分别列出覆盖情况、合法结构数、成对排序、top-1 与选中耗时比；单结构只计数值与执行覆盖。缺失的数据保留缺项。旧用例和新用例不混算准确度。

### 0.3.2 补充留出

GPU 在 H100 和 5090D 的两个出口分别测 FP16 转置 353×701、1024×2048，以及 B=1、S=151 的连续 FP16 indexed RoPE（Q/K 32/8 头、宽 128）。MLU370 测 FP32 weighted RMSNorm 32×256、257×1024、2048×2048，枚举所有合法参与者数、单双缓冲和共享方式，行参数在同一结构内选择。先保存每个输入的默认预测，再测所有结构和参数行。

8034ea7 的 CPU 集合继续原样报告，其中在上次中断前读过的结果标为重复评估。既有 127×257 和登记的 RoPE 用于 P1 回归。校准仅使用合成搬运、启动、同步与向量微基准。

## 第 2 组首次留出 · 已通过数值检查的成员

本轮首先冻结 LayerNorm、训练 BatchNorm、Fused Add RMSNorm 和完整 dropout residual RMSNorm 的前向与反向成员。参考输入为现有 Bench case，未改 shape、dtype 或生成分布。Softmax 和 weighted RMSNorm 已在此前使用，作为回归对照。GroupNorm backward 和 LayerNorm backward 另待首次打通后独立冻结。

以下为产品 `4c9cfa6` 的逐成员首次结构预测。斜线分隔按预计耗时升序列出的结构，等分保持原 explain 顺序。最后一列为原 explain 文件 SHA-256 的十六位前缀。含逐行原始分项的编译记录位于独立 Bench 的 `output/experiments/v262-first/<设备>/<出口>/<workload>/<成员>/`。每次计分匹配同一编译的产物和自动调用回执，完整 program 另取三轮调用时间和有序成员身份。

| 设备 | 出口 | 成员 | 首次模型排序 | 选择 | explain 摘要 |
|---|---|---|---|---|---|
| 5090d | cutile | dropout_norm_backward | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | aed5768f8d0be217 |
| 5090d | cutile | dropout_norm_forward | resident-1 / stream-1 / resident-2 / stream-2 | resident-1 | a064b6832d511e30 |
| 5090d | cutile | fused_add_rms_norm | resident-1 / stream-1 / resident-2 / stream-2 | resident-1 | 153950ce69ac461d |
| 5090d | cutile | batch_norm_training | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 4a65d9ad4e76fc54 |
| 5090d | cutile | layer_norm | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | dfb16af190e8ea26 |
| 5090d | triton | dropout_norm_backward | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | c39c62ae3313c2fd |
| 5090d | triton | dropout_norm_forward | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | b8058546698cc24c |
| 5090d | triton | fused_add_rms_norm | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | f2be2340610ed3d1 |
| 5090d | triton | batch_norm_training | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 6c4a61fb97d194cd |
| 5090d | triton | layer_norm | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | afee25a5cb9f5c3b |
| h100 | cutile | dropout_norm_backward | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 1ca8014486860652 |
| h100 | cutile | dropout_norm_forward | resident-1 / stream-1 / resident-2 / stream-2 | resident-1 | 8861a98bfbc06029 |
| h100 | cutile | fused_add_rms_norm | resident-1 / stream-1 / resident-2 / stream-2 | resident-1 | 9c99c7754b5e09ff |
| h100 | cutile | batch_norm_training | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | ef79aa50f82d834c |
| h100 | cutile | layer_norm | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 3b04d750afb535c3 |
| h100 | triton | dropout_norm_backward | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 5528425def78cd4a |
| h100 | triton | dropout_norm_forward | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | c53481105a007ecc |
| h100 | triton | fused_add_rms_norm | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | a638522b037014fc |
| h100 | triton | batch_norm_training | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 518c0cdf91ceb92e |
| h100 | triton | layer_norm | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 7264ec27e2b1924c |
| mlu-native | bangc | dropout_norm_backward | double-buffer-shared-16 / double-buffer-16 / direct-shared-16 / direct-16 / double-buffer-shared-8 / double-buffer-8 / direct-shared-8 / direct-8 / double-buffer-shared-4 / double-buffer-4 / direct-shared-4 / direct-4 / double-buffer-shared-2 / double-buffer-2 / direct-shared-2 / direct-2 / double-buffer-shared-1 / double-buffer-1 / direct-shared-1 / direct-1 | double-buffer-shared-16 | 799285a97f342517 |
| mlu-native | bangc | dropout_norm_forward | double-buffer-shared-16 / double-buffer-16 / direct-shared-16 / direct-16 / double-buffer-shared-8 / double-buffer-8 / direct-shared-8 / direct-8 / double-buffer-shared-4 / double-buffer-4 / direct-shared-4 / direct-4 / double-buffer-shared-2 / double-buffer-2 / direct-shared-2 / direct-2 / double-buffer-shared-1 / double-buffer-1 / direct-shared-1 / direct-1 | double-buffer-shared-16 | 23bc917079028282 |
| mlu-native | bangc | fused_add_rms_norm | double-buffer-shared-16 / double-buffer-16 / direct-shared-16 / direct-16 / double-buffer-shared-8 / double-buffer-8 / direct-shared-8 / direct-8 / double-buffer-shared-4 / double-buffer-4 / direct-shared-4 / direct-4 / double-buffer-shared-2 / double-buffer-2 / direct-shared-2 / direct-2 / double-buffer-shared-1 / double-buffer-1 / direct-shared-1 / direct-1 | double-buffer-shared-16 | b5c378bc4aeebf70 |
| mlu-native | bangc | batch_norm_training | direct-16 / double-buffer-16 / double-buffer-8 / direct-8 / double-buffer-4 / direct-4 / double-buffer-2 / direct-2 / double-buffer-1 / direct-1 | direct-16 | 563735a53df62aec |
| mlu-native | bangc | layer_norm | double-buffer-shared-16 / double-buffer-16 / direct-shared-16 / direct-16 / double-buffer-shared-8 / double-buffer-8 / direct-shared-8 / direct-8 / double-buffer-shared-4 / double-buffer-4 / direct-shared-4 / direct-4 / double-buffer-shared-2 / double-buffer-2 / direct-shared-2 / direct-2 / double-buffer-shared-1 / double-buffer-1 / direct-shared-1 / direct-1 | double-buffer-shared-16 | 0f5182c33a6d6603 |
| rvv-native | weft | dropout_norm_backward | row-reduction | row-reduction | f073e93a1f5a82e5 |
| rvv-native | weft | dropout_norm_forward | row-reduction | row-reduction | 53fdcf4f97e05dc2 |
| rvv-native | weft | fused_add_rms_norm | row-reduction | row-reduction | e3c7fa0c7e0b23e3 |
| rvv-native | weft | batch_norm_training | tuple-reduction | tuple-reduction | b0c02f73903a7125 |
| rvv-native | weft | layer_norm | row-reduction | row-reduction | d84720859139e0a4 |
| rvv-v100 | weft | dropout_norm_backward | row-reduction | row-reduction | 19844608a6898db0 |
| rvv-v100 | weft | dropout_norm_forward | row-reduction | row-reduction | ad49d4a795b192d4 |
| rvv-v100 | weft | fused_add_rms_norm | row-reduction | row-reduction | 7a58c61d7802bf73 |
| rvv-v100 | weft | batch_norm_training | tuple-reduction | tuple-reduction | 1ae466b75e32479a |
| rvv-v100 | weft | layer_norm | row-reduction | row-reduction | 2bd314f1ec0a30e8 |
| x86 | mojo | dropout_norm_backward | row-reduction | row-reduction | f64e6b4449900ff5 |
| x86 | mojo | dropout_norm_forward | row-reduction | row-reduction | 3cffdaabcc283a4e |
| x86 | mojo | fused_add_rms_norm | row-reduction | row-reduction | bb13c5a550a8758b |
| x86 | mojo | batch_norm_training | tuple-reduction | tuple-reduction | 9d4293c31bbfa5dd |
| x86 | mojo | layer_norm | row-reduction | row-reduction | ff297c11d77b2fee |
| x86 | weft | dropout_norm_backward | row-reduction | row-reduction | 7209a2485ce3d691 |
| x86 | weft | dropout_norm_forward | row-reduction | row-reduction | 0f62691a94d62c82 |
| x86 | weft | fused_add_rms_norm | row-reduction | row-reduction | b3e02d516b145045 |
| x86 | weft | batch_norm_training | tuple-reduction | tuple-reduction | 1729648656e61753 |
| x86 | weft | layer_norm | row-reduction | row-reduction | c85713d5da484c05 |

## 第 2 组首次留出 · GroupNorm 反向

两个成员分别求输入梯度和仿射参数梯度，按现有 `program/group_norm_backward` 的 B=32、C=256、S=1024、32 组和原容差验证。完整 program 另外计时。以下是在任何该程序的 Bench 热调用采样前保存的顺序与选择。编译记录和产物仍在同一 Bench 首次留出目录。

| 设备 | 出口 | 成员 | 首次排序 | 默认 | explain 摘要 |
|---|---|---|---|---|---|
| 5090d | cutile | group_affine_gradient | stream-1 / stream-2 | stream-1 | 21b9f456670c0c27 |
| 5090d | cutile | group_input_gradient | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 04c27ef3367f151a |
| 5090d | triton | group_affine_gradient | stream-1 / stream-2 | stream-1 | 8ceb2c7402476929 |
| 5090d | triton | group_input_gradient | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 41067f160302d22e |
| h100 | cutile | group_affine_gradient | stream-1 / stream-2 | stream-1 | 3dcfb14f0651d3d9 |
| h100 | cutile | group_input_gradient | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 2a05af25f2dc4c38 |
| h100 | triton | group_affine_gradient | stream-1 / stream-2 | stream-1 | b413576b27cbb295 |
| h100 | triton | group_input_gradient | resident-1 / resident-2 / stream-1 / stream-2 | resident-1 | 66cf624f0863d15d |
| mlu-native | bangc | group_affine_gradient | double-buffer-16 / direct-16 / double-buffer-8 / direct-8 / double-buffer-4 / direct-4 / double-buffer-2 / direct-2 / double-buffer-1 / direct-1 | double-buffer-16 | 8bb17c5595b5e7f3 |
| mlu-native | bangc | group_input_gradient | double-buffer-16 / direct-16 / double-buffer-8 / direct-8 / double-buffer-4 / direct-4 / double-buffer-2 / direct-2 / double-buffer-1 / direct-1 | double-buffer-16 | e7231a948269228a |
| rvv-native | weft | group_affine_gradient | row-reduction | row-reduction | 719d67c424dc8ed5 |
| rvv-native | weft | group_input_gradient | row-reduction | row-reduction | a425325fad0b2377 |
| rvv-v100 | weft | group_affine_gradient | row-reduction | row-reduction | ba989cadebd1ac65 |
| rvv-v100 | weft | group_input_gradient | row-reduction | row-reduction | 2860f7c56aed98a9 |
| x86 | mojo | group_affine_gradient | row-reduction | row-reduction | 7dc89b544fca4601 |
| x86 | mojo | group_input_gradient | row-reduction | row-reduction | 51bc4f8440df955f |
| x86 | weft | group_affine_gradient | row-reduction | row-reduction | d1e4a5beed0959f5 |
| x86 | weft | group_input_gradient | row-reduction | row-reduction | e30cf82c7b4f69a5 |
