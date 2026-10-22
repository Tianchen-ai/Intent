# Compile and run

## Local JIT on the execution host

Install Intent and the chosen provider, then run on the machine that executes the kernel:

```python
import intent
from intent.examples.kernels.normalization.softmax import stable_softmax_f16

softmax = intent.jit(stable_softmax_f16, backend="triton")
y = softmax.run(x)
y2 = softmax.run(x2)
```

The required backend is one of `triton`, `cutile`, `mojo`, `weft`, or `bangc`.
Creating the object does not probe hardware. The first binding discovers the actual
local device and toolchain and caches generated artifacts. The first execution performs
necessary native compilation and tuning over the declared configurations. Subsequent
calls bind their own buffers and reuse applicable artifacts and configuration winners.
A failing backend never causes a switch to another provider.

The same code runs on H100 and 5090D using their locally discovered capabilities.
CPU discovery respects the calling process's allowed CPU set; `workers` is a thread budget.
Weft requires a native Weft and C toolchain on the RISC-V host. BANG C queries CNRT
resources and specializes required geometry from actual buffers. Ordinary calls do not
ask users to supply ISA, VLEN or on-chip storage capacity.

Inputs must belong to one compatible device. Calls without device-bearing inputs can
select a local device explicitly, for example `device="cuda:0"`. Conflicts fail without
implicit data movement. Use existing framework tensors or provider public buffers.
`backend_options` selects provider toolchain paths; `compiler` selects Intent's compiler.

## Preparation and execution

- `kernel.run(*inputs)` allocates declared Out buffers, executes and returns them.
  Callers still supply InOut.
- `kernel(*all_arguments)` takes every parameter, including Out, in declaration order,
  executes and returns None.
- `kernel.prepare(*inputs, outputs=(...))` binds inputs, explicit outputs and workspace.
  It may generate necessary IR/source, but never executes or tunes kernels.
  Omit outputs to allocate declared Out buffers.
- `call.compile()` compiles applicable native candidates without executing the author call.
- `call.launch()` executes the binding; `call.result()` retrieves output containers
  without executing or synchronizing.
- `kernel.close()` releases owned native artifacts. Keep argument storage alive until execution finishes.

Zero Out parameters return None; one returns that object; several return a tuple.
GPU calls retain the current stream's asynchronous submission semantics. Synchronize
through the framework when reading host results. Complete compilation, tuning and necessary
first execution before graph capture. Authors control multi-kernel ordering, intermediate
storage and gradients in ordinary Python; Intent does not rewrite the model graph.

## Installed author programs

The single source for algorithms and host orchestration is installed as
`intent.examples.kernels` and `intent.examples.programs`. Neither a checkout nor Bench is required:

```bash
python -m intent.examples --list
python -m intent.examples layer_norm --backend triton
python -m intent.examples paged_decode --backend cutile --prepared
python -m intent.examples group_norm_backward --backend mojo
python -m intent.examples attention --backend mojo --stage native
```

The source stage generates and checks call geometry; native compiles without executing;
run executes the complete program. Independent Bench owns numerical comparisons, repeated
measurements, suites and experimental protocols:

```bash
intent-bench check --suite mainline --backend triton --env local-gpu
intent-bench run softmax --env local-gpu --case 64x1024-float16 \
  --impl intent:triton,intent:cutile,triton --baseline triton
```

The [softmax example](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax.py)
demonstrates prepared calls and PyTorch integration. The
[forward/backward example](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax_forward_backward.py)
demonstrates author-registered gradients.

## Advanced IR and export interfaces

`intent.generate(..., target=...)`, `intent.compile(..., target=...)` and artifact export
remain available for explicit compilation and application integration. Offline compilation
requires complete target facts and does not establish successful local execution.

```bash
intent doctor --target triton --json
intent describe --target triton --json
intent compile intent.examples.kernels.normalization.softmax:stable_softmax_f16 --stage kir --json
intent compile intent.examples.kernels.normalization.softmax:stable_softmax_f16 --target triton --stage shared --json
intent optimize INPUT.mlir --pipeline 'builtin.module(canonicalize,cse)'
```

Standalone Intent uses `INTENT_CACHE_DIR`, defaulting to the XDG user cache.
Only Bench-initiated runs redirect this into Bench's output root. Generation, compilation,
execution and numerical checks are distinct outcomes. Compilation and tuning are not operator time.
