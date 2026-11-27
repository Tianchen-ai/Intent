# MLU370 的 22 项慢调用

按改前计分板，每个输入计一行。下表据设备代码归类，分段时间单位为 µs。首批修重复键归约，覆盖四项及最大差距 330×。

| 机制（行数） | 用例与原设备代码 | 对应的成熟做法 |
|---|---|---|
| 重复键加法（4） | `index_add` 按列标量读改写，复制/累加 37/294211。三条 histogram（program、cpu large、flaggems）DMA 后逐键增桶。flaggems 计数/合并 43964/553。 | `cnnlIndexAdd`、`cnnlBincount`。冲突行用向量原子加，私有桶用 WRAM 每次匹配 64 键。 |
| 矩阵收缩（4） | `flash_attention_backward`、`paged_mla_decode_partials` 逐头向量归约，后者还按页寻址。`batched_gemm` 默认/FTZ 已用 WRAM 矩阵核，默认有 BF16 守卫。 | `cnnlGatherV2` 聚页，`cnnlStrideBatchMatMul` 合批。反传 P、dS 须在乘积前舍入，小批需计入打包。 |
| 选择和写出位置（4） | `topk_rows` 整行 DMA 后逐项维护八对值和索引。`radix_pass` 用稳定游标，`moe_alignment` 用原子旧票号，`nonzero` 按前缀散写。 | `cnnlTopKTensor_v3`、`cnnlCumsum_v2`、`cnnlScatterNd_v2`。并列次序、稳定性及返回票号分别制约选择。 |
| 稀疏读（1） | `csr_spmv` 逐行读 offsets、标量 gather x。 | `cnnlGatherV2` 加分段规约。 |
| 量化点积（2） | `q4_projection`、`q4_k_projection` 在子块内逐个解半字节、累加及修正。 | Torch 组合批量位提取和点积。BANG 向量位操作与矩阵核须保留块比例及最小值修正。 |
| 窗口（1） | `causal_convolution` 每个输出逐 tap 乘加再算 SiLU。 | `cnnlConvolutionForward` 分组卷积复用窗口。 |
| 行归约（4） | `layer_norm_welford` 合并行统计，`group_norm_backward` 分开求两类梯度，大 softmax 的 FP16/FP32 两项分片归约后做严格商。 | `cnnlLayerNormForward`、`cnnlGroupNormBackward`、`cnnlSoftmaxForward_v2`。参照需对齐方差和舍入点。 |
| 随机掩码（1） | `dropout` 生成 xorshift 位流及概率掩码。 | 同算法可用 `cnnlBitCompute_v2` 组合。 |
| 布局（1） | `transpose` 已调 CNNL，8 对 7。 | `cnnlTranspose_v2`，1 µs 差距未分解。 |

出处为设备 SDK 的 `cnnl.h` 上表符号及 BANG C《Vector Operation Functions》的 `__bang_atomic_add`、`__bang_histogram`、`__bang_band_scalar`、`__bang_conv`。[计分板](v0.6-semantics.md)
