# 理想化 DSL 示例

这些文件展示作者应当表达的算法，不展示为迁就某一实现而加入的写法。它们刻意不包含 physical tile、`I.auto`、program id、warp、storage scope、target primitive 或 autotune config。

- [`pointwise.py`](pointwise.py)：完整 logical domain 上的纯张量定义；
- [`reduction.py`](reduction.py)：generic reduce 与 typed combine；
- [`gemm.py`](gemm.py)：完整域 contraction 与算法 constexpr 分支；
- [`online_softmax.py`](online_softmax.py)：用 typed record reduction 表达 online-softmax summary algebra，不使用 `state_stream`；
- [`flash_attention.py`](flash_attention.py)：用region fold显式表达chunk-local QK/max/sum/PV与summary merge，同时不暴露任何Q/K physical extent；
- [`causal_linear_attention.py`](causal_linear_attention.py)：用region scan表达不可观察分段上的slice summary、incoming state、causal slice output与final state；
- [`mamba_state_passing.py`](mamba_state_passing.py)：展示Mamba的chunk axis、per-chunk incoming states与final state已是算法ABI时，使用显式logical chunk domain与ordinary ordered carry，不使用region scan；
- [`ragged_grouped_gemm.py`](ragged_grouped_gemm.py)：用可机械展开的 ragged helper 表达 offsets/subregion/index relation，再进行 gather、contract 与 unique scatter；
- [`split_k_pipeline.py`](split_k_pipeline.py)：不用 `partition`，显式写 part domain、boundary arithmetic、source subregion、partial tensor 与两个 kernels 的 host orchestration。

示例是编程模型的规范性说明，不是测试 fixture；其中每种surface spelling都必须机械归一到对应的canonical semantics。
