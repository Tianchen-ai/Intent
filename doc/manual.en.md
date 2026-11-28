# IntentDSL Handbook

## 1. Installation and diagnostics (install)

Follow the source checkout README to build the CompilerInfra 0.6.0 static SDK, install the matching Python bindings and install IntentDSL 2.1.0. Source builds need Python ≥3.10, CMake, Ninja, C++17 and PIC static archives for LLVM/MLIR 20.1.8. The resulting wheel includes the native product library.

```sh
intentdsl setup --backend triton
intentdsl doctor --backend triton
```

Setup verifies the assets and frontend. Doctor reports the selected backend’s dependencies and device. Install GPU Triton and CPU triton-weft in separate environments.

| Backend (`backend`) | Execution Device | External Toolchains & Runtime Libraries |
|---|---|---|
| `triton` | NVIDIA GPU | PyTorch (CUDA), GPU `triton`, CUDA driver |
| `cutile` | NVIDIA GPU (`SM90`/`SM120`) | PyTorch (CUDA), `cuda-tile` (`cuda.tile`), `nvcc`, CUDA driver |
| `mojo` | x86_64 CPU (`AVX2`/`AVX-512`/`AMX`) | `mojo` compiler (or `INTENT_MOJO`), `gcc` + `libgomp.so`; optional `oneDNN >=3.13,<4` (`ONEDNN_ROOT` or `INTENT_ONEDNN_LIB`) |
| `weft` | x86_64 / riscv64 CPU | `triton-weft` + `weft_core`, C/C++ toolchain, homogeneous CPU affinity set (`taskset`) |
| `bangc` | Cambricon MLU (`MLU370`) | `cncc` (`mtp_372`), `libcnrt.so`, `torch_mlu`, Cambricon driver (`/dev/cambricon_dev*`); `libcnnl.so` used for library implementations |

## 2. Author Language and Numerical Contracts (`language`)

IntentDSL parses Python ASTs into typed `kir` IR; kernel bodies are never executed by the Python interpreter.

- **Directions and Symbolic Views**: `I.In[dtype, shape]`, `I.Out[dtype, shape]`, `I.InOut[dtype, shape]` with symbolic dimensions (`("M", "N")`) and typed scalars (`I.f32`, `I.i32`, `I.u64`).
- **Domains and Collectives**: `I.parallel(...)` declares parallel instances; `I.domain(0, extent)` declares reduction/streaming member axes; `I.static_range(K)` declares compile-time unrolled stripes; collectives include `I.reduce`, `I.sum`, `I.welford`, `I.cumsum`, `I.reduce_by_key`, `I.ordered_sum`, contractions, and ordered recurrences.
- **Numerical contracts**: Each reduction declares its combiner and accumulator type. `I.max_num` and `I.maximum` differ on NaNs. Explicit `denormal_fp_math="preserve-sign"` can license flushing subnormals when the operand types and device capability also permit it.

Author defaults follow the corresponding PyTorch or Inductor operation. Ordinary FP32 arithmetic preserves subnormals. For narrow matrix products, the exit registers its operand types, accumulator type and native arithmetic separately. On x86, AMX/VDPBF16PS can treat subnormal BF16 inputs as zero and flush subnormal outputs. On MLU370, the reference is `torch.bmm` through CNNL. Both the subnormal-input and underflow-result probes produced zero. CUDA MMA behavior depends on the instruction. SG2044 uses widened FP32 arithmetic for BF16.

Use `I.jit(kernel, backend="weft", denormal_fp_math="ieee")` to preserve subnormals. Use `float32_matmul_precision="highest"` to require FP32 products. For separate ordinary multiply/add rounding and source-order normalization, pass `fpmath_mode="strict"`. `I.ordered` keeps the declared iteration order. Inspect `ci.semantics` at `kir@canonical` for the effective choice and its source. [Per-target sources and counterexamples](https://github.com/Tianchen-ai/Intent/blob/main/doc/design/kir.md)

```python
import intentdsl as I

@I.kernel
def rms_norm(
    x: I.In[I.f32, ("M", "N")],
    gain: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
    inv_c: I.f32,
    eps: I.f32,
):
    for r in I.parallel(x.dim(0)):
        u = x[r, :]
        s = I.sum(u * u, axis=0, acc=I.f32)
        out[r, :] = (u * I.rsqrt(s * inv_c + eps)) * gain[:]
```

---

## 3. Model Integration and Runtime Execution (`model`)

- **JIT Compilation, Warm Execution, and Buffer Swapping**:
  `kernel = I.jit(rms_norm, backend="bangc")` (or `"triton"` / `"cutile"` / `"mojo"` / `"weft"`).
  - `y = kernel.run(x, gain, 1.0 / N, 1e-6)` allocates outputs, compiles on first call, and executes.
  - `call = kernel.prepare(x, gain, 1.0 / N, 1e-6, outputs=y)` separates preparation from warm replay (`call.execute()`, or `call.launch()` returning a `Completion` handle). Re-binding new buffers via `kernel.prepare(x2, gain, 1.0 / N, 1e-6, outputs=y2)` reuses the compiled native binary whenever shapes satisfy the artifact `guard`.
- **Static `@I.program` Sequences**: Captures `I.workspace`, `I.output`, `I.fill`, and multi-kernel topology without host Python overhead during replay (see 30 complete workloads in `intentdsl.examples`).

---

## 4. Component guide (components)

Inspect the registered units before changing a strategy or microkernel. Follow its owning node or exit from the decision and cost model to legalization and runtime.

Run the following commands from the Intent checkout with the agent package installed. The example reads the single-kernel author input in `doc/softmax.py`, derives a candidate, compiles to the KIR boundary and retrieves the trace by its returned compilation ID in the same session.

```sh
LIB=$(python -c 'import intentdsl; print(intentdsl.library_path())')
compilerinfra-dev --library "$LIB" --workspace . --output-dir ../candidate-output inspect
python examples/inspect_candidate.py
```

## 5. Formulas and examples (formulas)

From the checkout run `python examples/formulas.py` to compile the nine examples and compare each output to Torch. All functions are available from the installed `intentdsl.examples` module. Dimensions are symbolic; the row reductions below use the last axis.

The nine examples check formula outputs. The milestone scoreboard covers 355 full calls, with 244 faster, 85 tied, 7 slower and 19 without a matched reference. [Measurements and hardware](https://github.com/Tianchen-ai/Intent/blob/main/doc/plan/v0.6-semantics.md)

| Operator | Formula | Function and DSL primitives | Inputs → output |
|---|---|---|---|
| ReLU | max(x, 0) | `relu` · `I.maximum` | `f16[M,N] → f16[M,N]` |
| Softmax | exp(x−max(x)) / Σ exp(x−max(x)) | `softmax` · `I.reduce, I.exp, I.sum` | `f16[M,N] → f16[M,N]` |
| RMSNorm | x · rsqrt(Σx²/N + ε) · g | `rms_norm` · `I.sum, I.rsqrt` | `f32[M,N], g[N], 1/N, ε → f32[M,N]` |
| LayerNorm | (x−μ) · rsqrt(E[x²]−μ²+ε) · g + b | `layer_norm` · `I.sum, I.rsqrt` | `f32[R,C], g[C], b[C], 1/C, ε → f32[R,C]` |
| Transpose | Y[j,i] = X[i,j] | `transpose` · `I.parallel` | `f16[M,N] → f16[N,M]` |
| Prefix sum | Y[i,j] = Σₖ≤ⱼ X[i,k] | `cumsum` · `I.cumsum` | `f32[M,N] → f32[M,N]` |
| Embedding | Y[t,:] = W[ids[t],:] | `embedding` · `I.gather_row` | `bf16[V,D], i64[T] → bf16[T,D]` |
| GEMM | C[i,j] = Σₖ A[i,k]B[k,j] | `gemm` · `I.contract` | `f16[M,K], f16[K,N] → f16[M,N]` |
| Q6_K GEMV | y[i] = Σⱼ dequant(W[i,j]) · quant(x)[j] | `q6k_gemv` · `I.quantize_q8k, I.quant_contract` | `u8[N,G,210], f32[G,256] → f32[N]` |

Each 210-byte Q6_K record holds 256 weights: 192 bytes of bit planes, 16 signed subgroup scales and an FP16 scale. Activations are quantized in groups of 256 to Q8_K. The contraction accumulates the scaled products into FP32.
