# cuTile 慢项的执行结构

以 RTX 5090 D、CUDA Tile 1.6 的完整调用复查。上版 CSV 中有 12 行计慢。按原数值口径和参照重跑，以本轮计时判断是否仍慢。[逐项前后对照](v0.6-semantics.md#本轮慢项复测)

| 机制 | 原计慢用例与设备代码 | 成熟实现与本轮发现 |
|---|---|---|
| 按键计数、散加和稳定排位 | `flaggems_histogram` 每个分区用块作用域的全局原子计数，再归并各列。`index_add` 先复制，再按列面板散加。`radix_pass` 用原子建桶，按 64 键分段，由段内相等键的前缀数确定稳定位置。 | CUB 的块私有桶和块内 radix rank 减少全局读改写。当前 cuTile 缺少片上直方图 primitive。实测按列独立计桶、加宽面板和改 CTA 占用率均未追上 CUB。散加现在按列数和目标每批可发出的元素数选面板。radix 改成 32 或 128 键、或改用位掩码计秩均更慢。[CUB AgentHistogram](https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_histogram.cuh) · [BlockRadixSort](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_radix_sort.cuh) |
| 密集与量化收缩 | `gemm_nt` 为转置 RHS 的 FP16 MMA。`awq_int4_gemm` 在 K 循环读压缩权重、组比例及零点。`q6k_gemv` 每个记录解码后求点积。 | cuBLASLt 按布局和分块选算法，CUTLASS 分组读取量化比例。缓存可容纳操作数时，模型按波次内共享面板、末波空闲和 MMA 工作量计费。对齐的单行元数据可形成矩形读。Q6_K 在改前复测已快于参照。[cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html) · [CUTLASS 混合类型 GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu) |
| 有依赖的矩阵更新 | `attention`、`flash_attention_backward` 分块推进 Q/K，反向分别求 dQ 与 dK/dV。`gated_delta`、`triangular_solve` 按时间或行的依赖顺序更新。 | Triton 的分块前反向、TileGym 的状态递推和批量三角求解保留不同的并行轴。四项改前复测已快于各自参照，沿用现有结构。[Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) · [TileGym](https://github.com/NVIDIA/TileGym) · [cuBLAS TRSM](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsmbatched) |
| 窗口及行归约 | `depthwise_conv2d` 按通道扫描空间窗。`group_norm_backward` 分别计算输入和参数梯度。 | ATen 的 depthwise 核复用每通道的卷积权重，GroupNorm 反向按输入和参数的归约域分工。深度卷积改前复测已快于 Inductor，归一化反向约在 Torch 的 5% 以内。[DepthwiseConv2d.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DepthwiseConv2d.cu) · [group_norm_kernel.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/group_norm_kernel.cu) |

五出口核查覆盖乘加分离、窄精度点积与除法许可。cuTile 的严格乘积在 FP16、BF16、FP32 原宽度保留舍入，三个反融合反例均得到零。MLU 库候选原先在函数上检查 FTZ，现由待实现的点积操作承载面板选择。MLU370 的后续双向反例支持 BF16 点积默认选原生矩阵核，显式 IEEE 或 FP32 乘积保留指数检查。Triton 的显式 FP32 乘积、Mojo/Weft 的窄点积许可和 RVV 的能力检查均保留。核查范围内未发现另一处无条件将乘积升到 FP64 的路径。
