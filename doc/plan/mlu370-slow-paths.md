# MLU370 的 22 项慢调用

按改前计分板，每个输入计一行。下表据设备代码归类，分段时间单位为 µs。首批修复重复键归约；后续收缩、选择、稀疏读与量化的完整调用列于[历次专项](v0.6-semantics.md)。余下十行的改前、改后和同轮参照见[本轮复测](v0.6-semantics.md#本轮慢项复测)。

| 机制（行数） | 用例与原设备代码 | 对应的成熟做法 |
|---|---|---|
| 重复键加法（4） | `index_add` 按列标量读改写，复制/累加 37/294211。三条 histogram（program、cpu large、flaggems）DMA 后逐键增桶。flaggems 计数/合并 43964/553。 | `cnnlIndexAdd`、`cnnlBincount`。冲突行用向量原子加，私有桶用 WRAM 每次匹配 64 键。 |
| 矩阵收缩（4） | `flash_attention_backward`、`paged_mla_decode_partials` 逐头向量归约，后者还按页寻址。`batched_gemm` 默认/FTZ 已用 WRAM 矩阵核，默认有 BF16 守卫。 | `cnnlGatherV2` 聚页，`cnnlStrideBatchMatMul` 合批。现按独立的行或头共用 WRAM 面板，分页读在页边界拆 DMA，P、dS 在作者位置舍入。MLU370 BF16 点积默认与 torch.bmm 的原生矩阵行为对齐，显式 IEEE 或 FP32 乘积保留指数守卫。[两组反例](v0.6-semantics.md#本轮慢项复测) |
| 选择和写出位置（4） | `topk_rows` 整行 DMA 后逐项维护八对值和索引。`radix_pass` 用稳定游标，`moe_alignment` 用原子旧票号，`nonzero` 按前缀散写。 | `cnnlTopKTensor_v3`、`cnnlCumsum_v2`、`cnnlScatterNd_v2`。已实现向量 argmax 与并列重放、私有行的整数前缀、分键原子预留及稳定过滤。 |
| 稀疏读（1） | `csr_spmv` 逐行读 offsets、标量 gather x。 | `cnnlGatherV2` 加分段归约。现以 NRAM 表和 `__bang_lut` 聚集，逐行按原顺序求和。 |
| 量化点积（2） | `q4_projection`、`q4_k_projection` 在子块内逐个解半字节、累加及修正。 | Torch 组合批量位提取和点积。现按记录布局批量解包，分组内整数点积精确，分别累计比例及最小值修正。 |
| 窗口（1） | `causal_convolution` 每个输出逐 tap 乘加再算 SiLU。 | `cnnlConvolutionForward` 分组卷积复用窗口。 |
| 行归约（4） | `layer_norm_welford` 合并行统计，`group_norm_backward` 分开求两类梯度，大 softmax 的 FP16/FP32 两项分片归约后做严格商。 | `cnnlLayerNormForward`、`cnnlGroupNormBackward`、`cnnlSoftmaxForward_v2`。参照需对齐方差和舍入点。 |
| 随机掩码（1） | `dropout` 生成 xorshift 位流及概率掩码。 | 同算法可用 `cnnlBitCompute_v2` 组合。 |
| 布局（1） | `transpose` 已调 CNNL，8 对 7。 | `cnnlTranspose_v2`，1 µs 差距未分解。 |

出处为设备 SDK 的 `cnnl.h` 上表符号及 BANG C《Vector Operation Functions》的 `__bang_atomic_add`、`__bang_histogram`、`__bang_band_scalar`、`__bang_conv`。MLU370 的 `__bang_lut` 支持 u32 索引及同宽数据，SDK 的 Discrete Move Functions 将 `__gather` 列为 mtp_592 起支持。[寒武纪 BANG C 算子开发指南](https://github.com/Cambricon/mlu-ops/blob/master/docs/BANG-C-OPS-Develop-Guide.md) · [计分板](v0.6-semantics.md)
