# IntentDSL

**Write an operator algorithm once. Compile it for different execution models.**

[English](https://github.com/Tianchen-ai/Intent/blob/main/README.md) · [简体中文](https://github.com/Tianchen-ai/Intent/blob/main/README.zh-CN.md)

[Documentation](https://tianchen-ai.github.io/Intent/en/) · [Installation](https://tianchen-ai.github.io/Intent/en/getting-started/installation/) · [Examples](https://github.com/Tianchen-ai/Intent/blob/main/examples/README.en.md) · [MCP](https://github.com/Tianchen-ai/Intent/blob/main/mcp/README.md) · [Contributing](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.md)

IntentDSL is a Python kernel language and an MLIR compiler for GPU, CPU, and accelerator programs. Authors describe tensor computations, logical domains, accesses, state, and explicit multi-kernel composition. Compiler passes form the physical program and lower it through Triton, cuTile, Mojo, Weft, or BANG C.

![Original IntentDSL paper overview: algorithm reuse and compiler workflow](https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-fig1-overview.png)

*Original paper overview ([PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-fig1-overview.pdf)). Labels describe the paper's implementation paths; current provider entry points are listed below.*

- **Programmable algorithms:** compose reductions, contractions, indexed access, control flow, and region computations inside a kernel.
- **Reusable optimization:** shared semantic analyses and execution-family passes organize computation, reuse, and storage; provider compilers perform their own layout and instruction lowering.
- **Ordinary Python calls:** compile once, pass tensors or native buffers, and inspect the generated source, IR, and selected configuration.

## Get started

The GPU setup uses **Linux, Python 3.10–3.12, and an NVIDIA GPU**. A prebuilt wheel installs the compiler without an LLVM/MLIR SDK. See the [installation guide](https://github.com/Tianchen-ai/Intent/blob/main/environment/README.md) for other backends and source-build prerequisites.

### Install with pip

Install the compiler from [PyPI](https://pypi.org/project/intentdsl/). The optional extras add the MCP servers and portable example inputs:

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install 'intentdsl[manual,examples]'
intent setup --target triton
intent doctor --target triton
```

The wheel targets **Linux x86-64 with glibc 2.35 or newer**. It includes the native compiler, optimizer, profiles, manual, and compiler runtime libraries, so installation needs no LLVM/MLIR SDK. Backend SDKs and drivers are selected separately. The base compiler installs with `pip install intentdsl`; its Python import is `intent`. Downloadable wheels are also available in the [GitHub preview](https://github.com/Tianchen-ai/Intent/releases/tag/preview).

### Install from source

```bash
git clone https://github.com/Tianchen-ai/Intent.git
cd Intent
python3 environment/install.py --backend triton --venv .venv-triton --examples
source .venv-triton/bin/activate
intent doctor --target triton
python -m examples.use softmax --target triton
```

Source builds need CMake, Ninja, a C++17 compiler, and the LLVM/MLIR 20 C++ SDK. The script builds and installs the compiler and Python package, then installs the chosen backend. Build artifacts stay outside the checkout. Use `--backend cutile --venv .venv-cutile` for cuTile in a separate environment. [Installation](https://github.com/Tianchen-ai/Intent/blob/main/environment/README.md) covers SDK packages, external CPU/MLU toolchains, and `environment/build.py` for building distributable wheels.

## Write and call a kernel

Save this as a Python file and run it in the installed Triton environment. It uses the same algorithm as the [ReLU example](https://github.com/Tianchen-ai/Intent/blob/main/examples/kernels/activation/pointwise.py).

```python
import torch
import intent
import intent.language as I


@intent.kernel
def relu(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        output[row, columns] = I.maximum(x[row, columns], I.cast(0.0, I.f16))


artifact = intent.compile(relu, target=intent.TritonTarget())
x = torch.randn((64, 1024), device="cuda", dtype=torch.float16)
y = artifact.run(x)
print(y.shape, y.dtype, y.device)
```

The kernel describes logical rows and columns. Compiler passes choose physical ownership, blocking, traversals, and storage. The target is selected on the host; the kernel does not branch on the backend.

`intent.compile` returns a callable artifact. Its first invocation can trigger native JIT compilation and autotuning; subsequent calls with the same specialization reuse that work. `artifact.run(...)` allocates declared `Out` tensors, while `artifact(...)` accepts explicit outputs. Keep the artifact in your application instead of recompiling on every call.

GPU and Mojo artifacts also provide `artifact.as_torch_op("your_project::name")` for opaque PyTorch custom operators and `torch.compile` integration. Authors register their own backward; the compiler preserves explicit kernel and host composition. See the [framework examples](https://github.com/Tianchen-ai/Intent/blob/main/examples/README.en.md#integrate-with-a-framework).

## Explore 30 complete programs

The product examples cover attention, recurrent state, normalization, matrix and quantized contractions, sparse access, scans, and mutable updates. Each program uses the unique algorithm in `examples/kernels/` and ordinary host composition in `examples/programs/`.

```bash
python -m examples.use --list
python -m examples.use attention --target triton
python -m examples.use paged_decode --target cutile
python -m examples.use --all --target triton --stage source --jobs 4
```

| Provider | Execution route |
|---|---|
| Triton / cuTile | NVIDIA GPU; PyTorch tensor and stream integration |
| Mojo | x86 CPU; external Mojo compiler |
| Weft | CPU lowering; native execution through an explicitly configured Weft deployment |
| BANG C | Cambricon MLU; external NeuWare SDK and compatible device |

Generation, native compilation, and execution are separate stages. Backend coverage depends on the program's operations, types, effects, and selected hardware. The [examples guide](https://github.com/Tianchen-ai/Intent/blob/main/examples/README.en.md) describes inputs, complete calls, and target options.

## How the compiler is organized

Passes consume typed semantics, def-use, coordinate relations, effects, and target capabilities. Execution decisions live in the current MLIR program. GPU, CPU, and DSA can reuse semantic knowledge while retaining their own execution and storage models.

![Original paper figure: one logical computation realized as a GPU program, CPU task hierarchy, and DSA local-storage program](https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-fig5-execution-models.png)

*Original execution-construction figure ([PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-fig5-execution-models.pdf)): each execution model assigns query ownership and carries state across source regions. CPU implementation requirements shape local blocks; DSA dependencies govern input-buffer reuse.*

Read the [language manual](https://tianchen-ai.github.io/Intent/en/), [compiler specification](https://github.com/Tianchen-ai/Intent/blob/main/doc/compiler/README.en.md), or [contribution guide](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.md) for the relevant entry point.

## Performance across execution models

The following original paper figures compare complete operators with target-written implementations. Speedup is **reference latency / generated latency**; higher is better. Median operator latency includes preparation and required layout materialization, and excludes compilation, JIT, and tuning. These are historical paper measurements, not a new evaluation of the current package or the 30 product programs.

![Paper GPU performance: generated Triton and cuTile against Triton references on RTX 5090D and H100](https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-eval-gpu-triton-reference.png)

*GPU ([PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-eval-gpu-triton-reference.pdf)): eight representative workloads and the geometric mean of all 44 common workloads. For each workload, the generated Triton and cuTile targets share the authored DSL algorithm, input scale, and external dtype. Representative shapes and precisions are labeled in the original figure; the gray bars are the paper's Triton references.*

| x86 CPU · 7 native comparisons | MLU370 · 6 library comparisons |
|---|---|
| <img src="https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-eval-cpu-native.png" alt="Paper x86 Mojo performance against five MAX-based and two independent Mojo implementations" width="360"> | <img src="https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-eval-mlu-libraries.png" alt="Paper MLU BANG C performance against CNNL, CNNL Extra and TMO implementations" width="360"> |
| [Original PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-eval-cpu-native.pdf) | [Original PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-eval-mlu-libraries.pdf) |

*CPU: generated Mojo programs against five Modular/MAX and two independent Mojo references, using eight physical x86 cores within one NUMA node. MLU: generated BANG C programs against CNNL/CNNL Extra/TMO, using MLU370 and CNCC/CNRT. Each comparison follows its paper workload's shapes, precision, outputs, and tolerance; these charts show per-workload ratios, not geometric means. [Figure sources](https://github.com/Tianchen-ai/Intent/blob/main/images/README.md) record the measurement scope.*

## Use with an agent

MCP has a dedicated [directory and setup guide](https://github.com/Tianchen-ai/Intent/blob/main/mcp/README.md). Both stdio servers ship with the `manual` package extra:

| Command | Responsibility |
|---|---|
| `intent-manual` | Read-only language contracts, API declarations, and minimal syntax fragments |
| `intent-compiler-mcp` | Compile a supplied program, inspect the environment, optimize supplied IR, and read artifacts |

Start with `read(id="doc/dsl/authoring.en.md")` and `api(name="intent.compile")` in the manual. Enable the compiler server when the agent should compile a supplied Python program. It uses the same compiler as the CLI; importing a program executes its ordinary top-level Python.

## Inspect compilation

```bash
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 \
  --stage kir --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 \
  --target triton --json
```

The CLI reports the actual compilation stage and artifact paths. In Python, use `artifact.source`, `artifact.mlir`, and `artifact.cache_directory`. Source generation and KIR inspection do not actively launch the selected kernel. See the [usage guide](https://tianchen-ai.github.io/Intent/en/getting-started/usage/) for installed tools and diagnostics.

## Contribute

Start with [CONTRIBUTING.md](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.md) for setup, compiler modules, pass contributions, and focused verification. Questions and reproducible problems belong in [GitHub issues](https://github.com/Tianchen-ai/Intent/issues).

IntentDSL is licensed under [Apache-2.0](https://github.com/Tianchen-ai/Intent/blob/main/LICENSE). Bundled third-party components retain their own notices and licenses.
