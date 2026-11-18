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

The explicit backend selects a component: `triton`, `cutile`, `mojo`, `weft`, `bangc`, or `triton_ascend`.
Creating the object does not probe hardware. The first binding discovers the actual
local device and toolchain and caches generated artifacts. The first execution performs
necessary native compilation and tuning over the declared configurations. Subsequent
calls bind their own buffers and reuse applicable artifacts and configuration winners.
A failing backend never causes a switch to another provider.

The same code runs on H100 and 5090D using their locally discovered capabilities.
CPU discovery respects the calling process's allowed CPU set; `workers` is a thread budget.
Weft uses full Triton–Weft, its matching native runtime and a C toolchain on the x86/RISC-V execution host.
Canonical `.weftir` enters `triton.compile`; tasks use the official native launcher. BANG C queries CNRT
resources and specializes required geometry from actual buffers. Ordinary calls do not
ask users to supply ISA, VLEN or on-chip storage capacity.
Triton–Ascend uses local NPU tensors, streams and the official Ascend toolchain. Follow
the [installation guide](installation.md) for distribution compatibility with CUDA Triton;
the product does not replace a process-global driver to combine incompatible installations.

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
GPU/NPU calls retain the current stream's asynchronous submission semantics. CPU native
calls return after all tasks and joins complete. Synchronize
through the framework when reading host results. Complete compilation, tuning and necessary
first execution before graph capture. Authors control multi-kernel ordering, intermediate
storage and gradients in ordinary Python; Intent does not rewrite the model graph.

## Ordered prepared calls

`intent.PreparedSequence(calls, outputs=..., owners=...)` retains an author's
already-bound ordered invocation. Members can be prepared calls, nested
sequences, and explicit workspace initializations. `kernel.empty_like(reference,
shape, dtype=...)` allocates author-owned workspace on the selected device;
`kernel.prepare_fill(buffer, value)` binds an initialization without writing it.
Place that initialization at its required position in the sequence.

`compile()` only compiles members, `launch()` executes the entire sequence, and
`result()` returns the declared output containers. BANG C submits members to one
queue and synchronizes at the end; GPU members retain current-stream semantics.
After first-use execution, CPU `native_calls()` exports ordered native bindings
and initializations. Native consumers retain their owners, call `validate()`
before execution and `complete()` afterward. This is an invocation interface,
not a timing protocol or a program-name-based orchestration mechanism.

The installed `AttentionBackward` program binds three kernels. It takes the saved forward output and natural-log LSE and returns dQ, dK and dV in that order. This example uses four query heads per KV head.

```python
from functools import partial
import intent
from intent.examples.programs.attention_backward import AttentionBackward

backward = AttentionBackward.from_jit(partial(intent.jit, backend="triton"),
                                      head_group=4, causal=True)
call = backward.prepare(q, k, v, output, grad_output, lse, scale)
call.compile()
call.launch()
dq, dk, dv = call.result()
```

`intent.examples.programs.structured.VarlenCausalConvolution` uses the same JIT binding for its two kernels. It returns the output for all tokens and the final state of each sequence.

Use `intent.HostBuffers(backend, device)` for explicit host transfers and ownership
across a complete call. `input(array, writable=False, preserve=False)` accepts a
NumPy array, CPU tensor or existing provider buffer, preserving shared allocations,
shape, strides and offsets. `preserve=True` saves initial state for `reset()`;
`finish()` copies writable staged storage back to its original host allocation.
Retain produced buffers with `retain_outputs()`. Complete device operations before
`close()` or leaving the context manager. This explicit transfer scope neither
changes ordinary JIT device checks nor infers an ABI or kernel sequence.

For Weft, a dense read-only input can use nonnegative element strides, including
a transposed or padded slice and a broadcast stride of zero. The JIT retains a
specialization for the set of strided input parameters. Calls with different
pitches reuse that specialization and pass their actual strides to the native
task. Packed records, writable outputs and InOut state require contiguous storage.

## Installed author programs

Pass `--tuning-config /path/to/config.json` to use the same complete configuration
table for all kernels of a program, both during source inspection and native JIT.
Its format is the same as the ordinary `intent.jit` option.

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
