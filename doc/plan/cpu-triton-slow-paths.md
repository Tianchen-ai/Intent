# Xeon 8470Q 和 RTX 5090 D 的剩余慢项

起点是 0.6 计分板中 Triton 的 7 行、Mojo 的 7 行和 Weft 的 9 行。下表按生成代码中的工作划分；同一用例可以在不同出口使用不同的核。[逐项时间与同机参照](v0.6-semantics.csv)

| 机制 | 出口、用例和现有执行结构 | 成熟实现的对照 |
|---|---|---|
| 计数、散加与稳定排位 | Triton 的 `flaggems_histogram` 先做分区直方图再按桶归并，`index_add` 先复制再散加，`radix_pass` 分段计秩并写出。Mojo、Weft 的小 `cpu/histogram` 都经两次调用；Weft 的 `index_add` 也保留复制阶段。 | CUB 的 DeviceHistogram 和 BlockRadixRank 分别在块内累计、排位；ATen 的 index_add 在输出上更新。参照中有重复键，不能假设写出无冲突。[CUB](https://github.com/NVIDIA/cccl/tree/main/cub/cub) · [ATen IndexKernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/IndexKernel.cpp) |
| 点积、布局和量化供数 | Triton 的 `awq_int4_gemm` 在 K 循环中解包再做 MMA。Mojo、Weft 的 `qkv_projection` 要供给三组转置权重。Mojo 的 `moe_expert_projection` 先将 8 个专家合计约 896 MiB 的权重排成 BF16 面板，再计算并散回路由行。Weft 的 `q6k_gemv` 逐块解码。 | oneDNN 的 transform 可按原矩阵步长生成 BRGEMM 请求的 B 布局；GGML 的 `ggml_vec_dot_q6_K_q8_K` 按编码块读 scale 和子块。[oneDNN transform](https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_ukernel_transform.html) · [GGML x86](https://github.com/ggml-org/ggml/tree/master/src/ggml-cpu/arch/x86) |
| 行归约与分块 attention | Triton 的 `qk_norm_rope` 融合行归一化和旋转，`jagged_mean` 按动态段长扫 K 并归约，`mamba_chunk_state` 每块形成带权 contraction。Weft 的 `softmax` 按行求峰值和行和。两 CPU 出口的 `flash_attention_backward` 分别求 dQ 和 dK/dV；`attention_unscaled_max` 有因果上界，Mojo 的 `alibi_attention` 需扫全 K，Weft 的 `paged_mla_decode_partials` 还需按页读 K/V。 | TritonBench 的 jagged 核联合分段长度与特征宽度分块。ATen FlashAttentionKernel 以 Q 行组并行，在 K 块间更新最大值、行和与输出。Weft unscaled 的前次测量为 50.9 ms，同轮 ATen 27.4 ms；库外周期和已试在线分块见专项记录。[Triton 教程](https://triton-lang.org/main/getting-started/tutorials/) · [ATen CPU flash](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp) · [分段记录](cpu-attention-0.6.md) |
| 窗口和布局 | 两 CPU 出口的 `depthwise_conv2d` 使用独立输出元素的九项 fold。Mojo 生成的连续八条 SIMD lane 实际沿 batch 轴，相邻写地址隔 401408 个 FP32；权重虽可广播，输入与输出都跨行。 | oneDNN 的直接卷积按访问布局组织内层空间循环，对边界处理填充。[oneDNN convolution](https://uxlfoundation.github.io/oneDNN/dev_guide_convolution.html) |

**已采用的降级。** 对 K 连续且 N 跨步的 VNNI 权重，两个 CPU 出口都按源缓存行转置打包，尾列和末个不完整字仍补零。专家投影的边界探针中，原打包占 10.1 s，聚集输入约 4 ms，BRGEMM 与散回约 0.09 s。完整调用从本轮复测的 6.21 s 降至约 0.12–0.15 s。Mojo 展开短窗口的 tap 循环。Weft 全展开在三轮中均慢于原实现，只保留在线 attention 的有效位复用和整块边界证明。

**GPU 的选择。** Triton 在计数、独占前缀与散写使用同一键时，将桶号和源 lane 合成整数键排序，再按桶起点计算局部秩。全相同键、跨块重复键和不足整块的尾部均逐项对上直方图、最终游标、偏移、键及原序号。分段归约按特征宽度单独估算有显式 warp 配置的 CTA。三轮中 32 列均为 16.4 µs，64 列为 18.4–20.4 µs，128 列为 20.5 µs。AWQ 的五种 M/N/K 分块为 75–109 µs，现有选择仍在最快一组。

**剩余差距。** Weft 反向 attention 的输出内积约 0.24 ms，dK/dV 与 dQ 分别为 415 和 343 ms，两段均未形成 BRGEMM。分页 MLA 的分段计算约 64.1 ms，合并约 0.07 ms。后续优先处理分段内按头的点积与面板访问。[完整调用与同轮参照](v0.6-semantics.md)
