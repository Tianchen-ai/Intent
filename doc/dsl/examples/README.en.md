# Idealized DSL examples

These files express author algorithms rather than accommodations to one implementation. They intentionally contain no physical tiles, I.auto, program IDs, warps, storage scopes, target primitives or autotune configurations.

- [Pointwise](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/pointwise.py): pure tensor definitions over complete logical domains.
- [Reduction](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/reduction.py): generic reduce with typed combines.
- [GEMM](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/gemm.py): whole-domain contractions and algorithm constexpr branches.
- [Online softmax](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/online_softmax.py): typed-record summary algebra, without state_stream.
- [FlashAttention](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/flash_attention.py): region fold explicitly expresses local QK/max/sum/PV and merges without exposing physical Q/K extents.
- [Causal linear attention](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/causal_linear_attention.py): region scan expresses slice summaries, incoming state, causal output and final state with unobservable segmentation.
- [Mamba state passing](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/mamba_state_passing.py): observable chunk axes/per-chunk incoming states/final state use explicit logical chunks and ordered carries, not region scan.
- [Ragged grouped GEMM](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/ragged_grouped_gemm.py): mechanical ragged helpers expand to offsets/subregions/index relations, followed by gather/contract/unique scatter.
- [Split-K pipeline](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/split_k_pipeline.py): explicit part domains, boundaries, source subregions, partial tensors and two-kernel orchestration without partition.

These are normative programming-model illustrations, not test fixtures. Each surface spelling must normalize mechanically to its canonical semantics.
