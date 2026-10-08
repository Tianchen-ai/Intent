# Compile and run

## Choose a complete program

The repository's 30 complete programs reuse the single algorithm definitions in `examples/kernels/`. `examples/programs/` supplies inputs, outputs and explicit host orchestration. The compiler does not select templates from program names.

```bash
python examples/use.py --list
python examples/use.py layer_norm --target triton
python examples/use.py paged_decode --target cutile --prepared
python examples/use.py group_norm_backward --target mojo
```

`--stage source` generates source and binds complete-call shapes and dtypes. `native` compiles bound calls without execution. `run` executes the complete program and retrieves outputs. These outcomes establish generation, native compilation or execution respectively; none automatically establishes numerical correctness or performance quality.

```bash
python examples/use.py --all --target triton --stage source --jobs 4 --json
python examples/use.py attention --target mojo --stage native
python examples/use.py paged_decode --target triton --measure 10 --json
```

`--jobs` parallelizes source/native preparation only; device execution stays sequential. `--measure` times complete warm calls after initial compilation, tuning and execution. Multi-kernel programs include every author-defined call and necessary workspace initialization. Restoring supplied InOut values and retrieving host outputs occur outside timing. GPUs use CUDA events; CPU/MLU measurements use the wall clock of complete synchronous calls. Input preparation, compilation, tuning and `elapsed_seconds` are not kernel execution times.

## Python call interface

`intent.compile(kernel, target=..., constexprs=...)` returns a callable artifact. `intent.generate(...)` generates source/IR only. Hosts select targets, for example `intent.targets.TritonTarget()`; a target is not a value or branch condition inside a kernel.

- `artifact(...)` receives all runtime parameters in declaration order, including `Out`, executes and returns `None`.
- `artifact.run(...)` omits declared `Out` parameters, allocates and returns outputs. Callers still supply `InOut`.
- `artifact.prepare(..., outputs=(...))` binds inputs, outputs and resources without executing.
- A prepared call's `launch()` executes; `result()` retrieves output containers without execution or synchronization.

Zero `Out` parameters return `None`; one returns that object; several return a tuple in declaration order. Compile each artifact in a multi-kernel program separately and use ordinary Python to organize calls and intermediate tensor lifetimes.

The existing [softmax example](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax.py) demonstrates PyTorch tensors, prepared calls, native observations and `torch.compile`. The [forward/backward example](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax_forward_backward.py) demonstrates author-registered gradients. Intent neither derives backward algorithms nor adds hidden launches.

## Inspect generated structure

```bash
intent describe --target triton --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --stage kir --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --stage shared --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --json
intent read-artifact /path/returned/by/compile --offset 0 --limit 16000 --json
```

`--materialize` additionally creates a callable without launching. `--constexpr NAME=JSON` binds algorithm constexprs; `--target-option NAME=JSON` forwards public target constructor options. Run standard MLIR pipelines on existing IR with `intent optimize INPUT.mlir --pipeline 'builtin.module(canonicalize,cse)'`.

Source, IR and caches live in `$XDG_CACHE_HOME/intentdsl/` or `~/.cache/intentdsl/` by default; `INTENT_CACHE_DIR` overrides the root. Failures preserve the actual stage, diagnostics and artifact paths. Missing backend support is reported explicitly rather than changing the algorithm.
