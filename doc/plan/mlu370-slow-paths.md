# MLU370 的 22 项慢调用

按 0.6 计分板本轮改动前的 71 行统计，22 行慢、19 行缺同设备参照。表中归类依据为实际 BANG C 设备函数的循环、传输、访存和库调用。倍率为完整调用对同口径参照，未单独计时的结构差异不能换算为时间占比。

| 机制（行数） | 用例与原实现 | MLU 上相应的实现 |
|---|---|---|
| 重复键的加法规约（4） | `index_add` 按列逐元素读改写，复制 37 µs，累加 294,211 µs。`program/histogram`、`cpu/histogram` large、`flaggems_histogram` 已经 DMA 读入 NRAM，随后逐键标量增桶。前者计数/合并 677/6 µs，后者 43,964/553 µs。 | CNNL `cnnlIndexAdd`、`cnnlBincount`。可重排的加法批量 DMA 源行并用 `__bang_atomic_add` 合并冲突。私有计数用 `__bang_histogram`，一次比较 64 个 WRAM 桶。 |
| 按头或批次的矩阵收缩（4） | `flash_attention_backward`、`paged_mla_decode_partials` 有向量运算和行 DMA，各头的内积仍逐行归约。页表还产生间接地址。`batched_gemm` 默认、显式 FTZ 两行已有 WRAM 卷积，默认还检查 BF16 指数范围。 | 页表可先用 `cnnlGatherV2` 形成连续面板，再用 `cnnlStrideBatchMatMul`。反传须保留 P、dS 进入下一次乘积前的 FP16 舍入。小批 GEMM 要把预打包和守卫计入比较。 |
| 选择、前缀和与写出位置（4） | `topk_rows` 已整行 DMA，依次维护八对值与索引。`radix_pass` 逐行计数并按游标稳定写出。`moe_alignment` 的路由阶段用原子加返回旧票号。`nonzero` 先算行前缀再散写。 | `cnnlTopKTensor_v3`、`cnnlCumsum_v2`、`cnnlScatterNd_v2`。分别保留并列次序、稳定排列、票号和压缩后的位置。 |
| 稀疏间接读（1） | `csr_spmv` 逐行读 offsets，再按列索引标量读 x 并累加。 | CNNL `cnnlGatherV2` 加分段归约。BANG 的离散搬运须核对目标架构、偏移位宽及跨度。 |
| 量化解码与点积（2） | `q4_projection`、`q4_k_projection` 搬入打包记录，子块内逐元素解半字节并做整数点积和修正。 | 同机 ATen 组合批量提取高低位并分组求和。BANG 的 `__bang_band_scalar` 和 `__bang_conv` 提供位操作和矩阵单元，使用前须兑现块比例和最小值修正。 |
| 窗口卷积（1） | `causal_convolution` 为每个输出读因果窗，逐 tap 乘加，再算 SiLU。 | 分组 `cnnlConvolutionForward` 复用相邻窗口，随后施加激活。 |
| 行归约与数值收尾（4） | `layer_norm_welford` 合并行统计量。`group_norm_backward` 分别计算输入和参数梯度。`softmax` 的 FP16、FP32 大输入分片求峰值和总和，再做严格商。 | `cnnlLayerNormForward`、`cnnlGroupNormBackward`、`cnnlSoftmaxForward_v2`。参照按作者方差、舍入点和商的许可筛选。 |
| 随机数与掩码（1） | `dropout` 生成 xorshift 位流、概率和掩码。 | 同算法可用 `cnnlBitCompute_v2` 组合位运算。 |
| 布局转换（1） | `transpose` 已调 CNNL，8 µs 对 7 µs。 | 双方均走转置，V2 绑定 `cnnlTranspose_v2`，1 µs 差距未分解。 |

出处为 MLU370 所装 CNNL 的 `cnnl.h` 对应符号及《Cambricon BANG C Developer Guide》的 Vector Operation Functions 和 Discrete Move Functions。首批选择重复键归约，覆盖四行及原最大差距 330×。[同语义计分板](v0.6-semantics.md)
