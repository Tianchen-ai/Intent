# IntentDSL 工具书

## 1. 安装与诊断（install）

按源码仓 README，从同一工作目录构建 CompilerInfra 0.5.3 静态 SDK，安装同版本 Python 绑定，然后安装 IntentDSL 2.0.0。源码构建需要 Python ≥3.10、CMake、Ninja、C++17 和 LLVM/MLIR 20.1.8 的 PIC 静态库。安装生成的 wheel 后，运行机使用包内的编译库。

```sh
intentdsl setup --backend triton
intentdsl doctor --backend triton
```

setup 检查资产和前端；doctor 报告所选出口的依赖与设备状态。GPU Triton 和 CPU triton-weft 使用不同虚拟环境。

| 后端 (`backend`) | 执行设备 | 外部工具链与运行时依赖 |
|---|---|---|
| `triton` | NVIDIA GPU | PyTorch (CUDA)、GPU `triton`、CUDA 驱动 |
| `cutile` | NVIDIA GPU (`SM90`/`SM120`) | PyTorch (CUDA)、`cuda-tile` (`cuda.tile`)、`nvcc` 与 CUDA 驱动 |
| `mojo` | x86_64 CPU (`AVX2`/`AVX-512`/`AMX`) | `mojo` 编译器（或设 `INTENT_MOJO`）、`gcc` + `libgomp.so`；可选 `oneDNN >=3.13,<4`（设 `ONEDNN_ROOT` 或 `INTENT_ONEDNN_LIB`） |
| `weft` | x86_64 / riscv64 CPU | `triton-weft` + `weft_core`、C/C++ 工具链、同构 CPU 亲和集（`taskset`） |
| `bangc` | Cambricon MLU (`MLU370`) | `cncc`（`mtp_372`）、`libcnrt.so`、`torch_mlu`、寒武纪驱动（`/dev/cambricon_dev*`）；library 档使用 `libcnnl.so` |

## 2. 作者语言与数值合同 (`language`)

IntentDSL 只解析 Python AST 生成类型化 `kir` IR，不把内核体当作 Python 解释执行。

- **方向与标量/视图注解**：`I.In[dtype, shape]`、`I.Out[dtype, shape]`、`I.InOut[dtype, shape]`；支持符号维度（如 `("M", "N")`）与标量参数（`I.f32`、`I.i32`、`I.u64`）。
- **结构化域与集合原语**：`I.parallel(...)` 声明并行实例域；`I.domain(0, extent)` 声明归约/流式成员轴；`I.static_range(K)` 声明编译期展开条带；内置 `I.reduce`、`I.sum`、`I.welford`、`I.cumsum`、`I.reduce_by_key`、`I.ordered_sum` 及矩阵/递推原语。
- **数值合同**：每个归约声明其组合操作与累加类型。`I.max_num` 与 `I.maximum` 的 NaN 行为不同。可用 `denormal_fp_math="preserve-sign"` 显式授权相应次正规数冲零，出口还需检查类型与目标能力。

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

## 3. 模型接入与运行期调用 (`model`)

- **单内核 JIT、热调用与换 Buffer**：
  `kernel = I.jit(rms_norm, backend="bangc")`（或 `"triton"` / `"cutile"` / `"mojo"` / `"weft"`）。
  - `y = kernel.run(x, gain, 1.0 / N, 1e-6)`：一步完成分配、准备与执行。
  - `call = kernel.prepare(x, gain, 1.0 / N, 1e-6, outputs=y)`：分离准备与热执行（`call.execute()` 或返回异步句柄的 `call.launch()`）。更换输入/输出张量时再次调用 `kernel.prepare(x2, gain, 1.0 / N, 1e-6, outputs=y2)`，只要形状落在已编译产物的 `guard` 区间内即直接复用原生二进制，无需重编。
- **多阶段 `@I.program`**：静态捕获 `I.workspace`、`I.output`、`I.fill` 与子内核调用序列，在设备流或原生 CPU 任务区间内按序提交（内置 30 个完整示例见 `intentdsl.examples`）。

---

## 4. 组件开发（components）

组件登记和代价模型位于负责的节点或出口。先用 inspect 查实际注册位置，再从对应目录的 legalize、cost 或 runtime 源码追到发射点。

在源码仓中运行下列命令。脚本读取单内核作者输入 `doc/softmax.py`，在同一个开发会话中派生候选、编译到 KIR 边界，并用返回的编译句柄读取 trace。安装依赖是 README 中的可选 agent 包。

```sh
LIB=$(python -c 'import intentdsl; print(intentdsl.library_path())')
compilerinfra-dev --library "$LIB" --workspace . --output-dir ../candidate-output inspect
python examples/inspect_candidate.py
```

## 5. 公式、类型与示例（formulas）

在源码仓运行 `python examples/formulas.py`，逐项编译并用 Torch 验证下表九个公式。DSL 函数均可从已安装的 `intentdsl.examples` 导入。M、N、K 等是符号维，沿最后一轴归约。

| 算子 | 公式 | 函数与 DSL 原语 | 输入 → 输出 |
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

量化记录每 256 个值占 210 字节。前 192 字节为低、高位平面，后 16 字节为有符号子块尺度，末两字节为 FP16 尺度。激活按 256 元素分组量化为 Q8_K；乘积按子块尺度累加到 FP32。
