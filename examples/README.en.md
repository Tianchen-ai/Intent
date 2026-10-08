# Intent kernel examples

[English](README.en.md) | [简体中文](README.md)

`examples/kernels/` contains author-written, target-independent algorithms, grouped by responsibilities such as activation, contraction, normalization and streaming. The compiler preserves their types, shapes, numerical contracts and effects. Select a target when compiling a kernel.

See the [README](../README.md) for installation. The host examples reuse the definitions in `kernels/`.

## Run a complete program

[use.py](use.py) provides one entry point for 30 complete programs. Their inputs and host calls live in [programs/](programs/); each algorithm has one definition in `kernels/`. A program name selects a usage example. The compiler receives the original kernel definition and target, without using the example name as an optimization policy.

```bash
python examples/use.py --list
python examples/use.py layer_norm --target triton
python examples/use.py paged_decode --target cutile --prepared
python examples/use.py group_norm_backward --target mojo
python examples/use.py --all --target triton --stage source --jobs 4 --json
python examples/use.py attention --target mojo --stage native
python examples/use.py paged_decode --target triton --measure 10 --json
python examples/use.py --all --target weft --target-option vector_bits=256 --target-option workers=4 --stage source --jobs 4
python examples/use.py --all --target bangc --stage source --jobs 4
```

`--list` does not import Intent, PyTorch or a provider. Use `--compiler` to select a compiler build and `--device` to select a GPU. The stages are:

- `source`: generate provider source and bind the complete call's shapes and dtypes through the public interface.
- `native`: compile the bound native calls without launching them or reading their outputs.
- `run`: execute the complete program and retrieve its results.

`--all` uses the 30 programs below. `--jobs` prepares source or native calls concurrently; actual execution remains sequential. A batch exits unsuccessfully if any program fails and retains its actual failing stage, diagnostics and artifact paths. It does not retry with a different algorithm, dtype or target. `--jsonl` emits each completed record immediately. Other batch modes report progress on stderr. Records include the selected configuration and resources actually exposed by the SDK; compiling a candidate or estimating its resources does not establish execution.

Input preparation uses NumPy, with `ml_dtypes` for actual BF16/FP8 storage. Select `--examples` in `environment/install.py`, or install the package's `examples` extra. Triton, cuTile and Mojo bind actual Torch tensors at the call boundary; Weft and BANG C use their native buffers. A RISC-V host can prepare these inputs without Torch. BANG C's shapes and strides come from the bound arguments, without changing device information inside the algorithm.

`--seed` fixes the input stream and defaults to 0. A given program uses the same input bytes, dtype and algorithm across targets. `--measure N` records N complete prepared calls after first-use compilation, tuning and execution, returning their median and individual samples. Multi-kernel programs include all author calls and required workspace initialization. Restoring supplied `InOut` values and reading results back to the host happen outside the measured interval. GPU measurement uses CUDA events; CPU/MLU measurement uses the complete synchronous call's wall clock. `elapsed_seconds` includes preparation, generation, compilation and execution, so it is not kernel latency. Successful execution and timing are separate from numerical validation and performance assessment.

`--target-option NAME=JSON` forwards public target constructor arguments. Weft source generation requires explicit `vector_bits` and `workers`; native materialization additionally requires `profile`, a Weft `compiler`, and a nonempty `cc` command list. The profile uses [TargetProfile](../python/intent/runtime/weft/target.py)'s ISA, ABI, physical VLEN, execution CPUs and stack budget, checked against the actual machine when loading. `export_artifact` and `compile_artifact` support cross-host AOT; a target does not implicitly SSH or deploy. BANG C uses `device`, `neuware` and `compiler` to select an existing SDK and device.

The list includes implemented combinations and legal language combinations whose backend support is still incomplete. It is usage navigation, rather than a verified 30 × 5 support matrix. A compiler gap reports its actual stage; target capability limits and untested execution remain distinct.

| Program | Host example | Inputs and complete results |
|---|---|---|
| `relu` | [elementwise](programs/elementwise.py) | f16 matrix → allocated `Out` |
| `swiglu` | [elementwise](programs/elementwise.py) | Two bf16 matrices → bf16 output |
| `rope` | [elementwise](programs/elementwise.py) | f16 Q/K `InOut`, fixed head counts and f16 cosine/sine |
| `transpose` | [elementwise](programs/elementwise.py) | f16 M × N → N × M |
| `embedding` | [irregular](programs/irregular.py) | bf16 table and valid i64 row indices → lookup |
| `csr_spmv` | [irregular](programs/irregular.py) | Complete CSR offsets, indices and f32 values/vector |
| `jagged_mean` | [irregular](programs/irregular.py) | Nonempty unequal segments, offsets and f32 values → segmented mean |
| `softmax` | [collectives](programs/collectives.py) | f16 matrix → f16 probabilities |
| `layer_norm` | [collectives](programs/collectives.py) | f32 x/weight/bias, inverse dimension and epsilon |
| `batch_norm` | [collectives](programs/collectives.py) | Welford; output, saved statistics and updated running state |
| `group_norm_backward` | [collectives](programs/collectives.py) | Two kernels; dx/dweight/dbias |
| `cumsum` | [collectives](programs/collectives.py) | Ordered f32 row prefix |
| `causal_linear_attention` | [collectives](programs/collectives.py) | f32 Q/K/V → output and final matrix state |
| `nonzero` | [irregular](programs/irregular.py) | f32 values → indices and valid count per row |
| `moe_alignment` | [irregular](programs/irregular.py) | Four kernels; route IDs, expert blocks and valid padded count |
| `gemm` | [matrix](programs/matrix.py) | f16 A/B with explicit `Activation.NONE` specialization |
| `batched_gemm` | [matrix](programs/matrix.py) | Batched bf16 NN contraction |
| `ragged_gemm` | [matrix](programs/matrix.py) | Unequal group offsets and f16 grouped weights |
| `int8_gemm` | [matrix](programs/matrix.py) | i8 A/B and i32 bias → i32 output |
| `q4_projection` | [matrix](programs/matrix.py) | N × G × 144-byte Q4_K records and G × 256 f32 activations |
| `fp8_gemm` | [matrix](programs/matrix.py) | E4M3 fragments and u8 E8M0 scales → f32 accumulator |
| `attention` | [streaming](programs/streaming.py) | Grouped bf16 Q/K/V, causal specialization and scale |
| `paged_decode` | [streaming](programs/streaming.py) | Page/split tables; explicit partials → f32-to-f16 merge |
| `mamba_chunk_state` | [streaming](programs/streaming.py) | bf16 basis/x and f32 dt/decay → f32 chunk state |
| `gated_delta` | [streaming](programs/streaming.py) | bf16 inputs and ordered recurrence → output and f32 final state |
| `causal_convolution` | [structured](programs/structured.py) | f16 depthwise causal window with SiLU specialization |
| `cholesky` | [structured](programs/structured.py) | Positive definite f32 matrix `InOut` → lower factor |
| `dropout` | [elementwise](programs/elementwise.py) | Author's integer mixing, seed, drop and inverse-keep scalars |
| `adamw` | [elementwise](programs/elementwise.py) | f32 gradient, three mutable arrays and complete update scalars |
| `histogram` | [collectives](programs/collectives.py) | f32 samples, including masked out-of-range values → 256 i32 bins |

Dynamic-shape examples use inputs suitable for interactive use. Fixed-shape algorithms retain the constants in their definitions. These inputs demonstrate product calls and comparisons of the same workload; they do not represent large model workloads. The Q4_K example's zero records represent zero weights. A real model supplies its own packed records; the script does not convert its weight format.

## Compose multiple kernels

[composition.py](programs/composition.py) defines `GroupNormBackward`, `PagedDecode` and `MoEAlignment` around artifacts compiled by the caller. Input preparation and complete execution stay in this directory. The composition classes retain the author's explicit kernel order and workspace.

`run` allocates outputs, `into`/`run_into` use caller-supplied outputs and workspace, and `prepare` binds outputs and workspace before launch. The compiler compiles each original kernel definition; the host owns kernel order and intermediate tensors.

For example, compose paged partials and a merge compiled for the same target. Use these imports from a script in `examples/`, or set `PYTHONPATH=examples`:

```python
import intent

from programs.composition import PagedDecode
from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce

partials = intent.compile(paged_gqa_decode_partials, target=target,
                          constexprs={"PAGE_SIZE": page_size,
                                      "HEAD_GROUP": query_heads // kv_heads,
                                      "SPLITS": splits})
merge = intent.compile(splitk_attention_f32_to_f16_reduce, target=target)
program = PagedDecode(partials, merge)
call = program.prepare(q, key_cache, value_cache, page_offsets, page_indices,
                       lengths, split_offsets, scale)
call.launch()
output = call.result()
```

`prepare` allocates and binds outputs without reading intermediate results or executing partials. Launch always executes partials before merge. The MoE prepared call owns counts, cursors and padding workspace. Every `launch()` initializes these workspaces before invoking four kernels. Only the route prefix specified by `total_padded` and the corresponding block prefix are valid; padding uses the author's sentinel. This composition uses the author's fixed top-k of 2, capacity of 4096 tokens, and expert IDs in `[0, 64)`. Other top-k values and excess capacity are rejected. A different workload size requires changing the author's algorithm constants and host workspace.

## Integrate with a framework

| Entry point | Complete call demonstrated |
|---|---|
| `python examples/softmax.py --target triton` | Compile once, pass PyTorch tensors, allocate declared outputs and locate compiler artifacts |
| `python examples/softmax.py --target cutile` | Reuse the same algorithm in a separate cuTile environment |
| `python examples/softmax.py --target triton --prepared` | Supply outputs, prepare one call, then use `launch()` and `result()` separately |
| `python examples/softmax.py --target triton --inspect-native` | Inspect the current configuration, candidate status and SDK-provided native resources after execution |
| `python examples/softmax.py --target triton --torch-compile` | Complete JIT/tuning through an ordinary call, then enter `torch.compile(fullgraph=True)` through an opaque custom op |
| `python examples/softmax_forward_backward.py --target triton` | Register the existing backward, save forward outputs and obtain input gradients with `Tensor.backward(upstream)` |
| `python examples/softmax_forward_backward.py --target triton --torch-compile` | Register the same forward/backward with PyTorch graph compilation; both kernels complete first-use JIT/tuning before capture |
| `python examples/softmax_forward_backward.py --target mojo --torch-compile` | Integrate the same author-defined forward/backward with graph compilation and autograd on Mojo CPU |

`artifact.run(...)` omits declared `Out` arguments and returns newly allocated outputs. `artifact(...)` takes all runtime arguments in declaration order. `artifact.prepare(..., outputs=(...))` accepts the same inputs and supplied outputs without executing the kernel. The returned call owns its argument bindings: `launch()` executes, and `result()` returns its output container without execution or synchronization. Zero outputs return `None`, one returns the value, and multiple outputs return a tuple in declaration order. The caller always supplies `InOut` arguments.

`--inspect-native` reads `artifact.observation` and can be combined with `--prepared`. Resource observations carry their source, stage and units. A value unavailable from the SDK has an explicit reason. A snapshot is not a correctness or occupancy conclusion and does not cause an extra launch. Before the first actual call, the property is `None`.

The forward/backward example reuses the existing `4096 × 4097` f32 backward definition. The author compiles two kernels, registers PyTorch custom ops, and supplies a gradient formula through `register_autograd`. `setup_context` saves probabilities; the backward callback invokes the registered backward op. An ordinary Python wrapper defines the relationship between the kernels. The compiler does not infer gradients or add hidden calls. The example demonstrates first-order gradients; higher-order gradients need author-supplied formulas.

The PyTorch adapter supports `In`/scalar inputs, newly allocated `Out`, and mutation-declared `InOut` on GPU and Mojo CPU. `InOut` updates in place without an extra return; no outputs return `None`. Each `InOut` must not share Torch storage with another input, and returned aliases are unsupported. The runtime's `infer_outputs` supplies fake outputs using the ordinary call's shape, dtype and stride contract without invoking a provider or reading tensor data. `register_autograd` applies only to functional custom ops; PyTorch does not accept this registration for mutable custom ops. Weft and BANG C keep their native buffer interfaces.

Both scripts accept `--target triton`, `--target cutile` and `--target mojo`. Input tensors follow the artifact's actual CPU/CUDA device. Mojo uses the public `MojoTarget()` defaults; select its SDK with `INTENT_MOJO`. Switching targets preserves the example algorithm, shapes and dtypes. Script output is separate from benchmark results and numerical validation. Each script keeps host execution inside `main`, allowing it to be imported as a module.

## Compile and diagnose

To compile without launching a kernel:

```bash
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --json
```

JSON includes the original definition, actual compiler stage, source/IR/metadata and log paths. Inspect `diagnostic` and `files` after a failure. `intent doctor --target triton` checks prerequisites and target resolution. Complete algorithms stay in the ordinary examples; the manual MCP supplies general language rules.

Use these complete programs for compilation, real execution and pass optimization observations. Keep generated source, IR, caches and observations outside the repository. The [installation guide](../environment/README.md) describes each backend's prerequisites.

Add algorithms to `kernels/`; reuse the relevant `programs/` entry and public artifact API for complete host calls. Keep one algorithm definition and one author-owned composition, without copying references or adding another execution mechanism.
