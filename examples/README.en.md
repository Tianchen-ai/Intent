# Intent author programs and framework examples

[中文](README.md)

Algorithms have one source in [intent.examples.kernels](../python/intent/examples/kernels/).
Complete host orchestration lives in [intent.examples.programs](../python/intent/examples/programs/).
Both ship in the installed package; top-level scripts are usage entrypoints.

```bash
python -m intent.examples --list
python -m intent.examples softmax --backend triton
python -m intent.examples paged_decode --backend cutile --prepared
python -m intent.examples group_norm_backward --backend mojo
python -m intent.examples attention --backend weft --stage native
python -m intent.examples softmax --backend bangc
```

Select a backend explicitly. Programs run on the execution host, discovering capabilities
from their actual input device. Use `--device` for a local device, `--workers` for a CPU
thread budget and `--option NAME=JSON` for provider toolchain options.
The source/native/run stages mean generation, native compilation and execution respectively;
none implies a numerical comparison. Independent Bench owns batch checks and measurements.

## Compose multiple kernels

[composition.py](../python/intent/examples/programs/composition.py) contains GroupNormBackward,
PagedDecode and MoEAlignment, preserving author-defined order and workspace.
Create `intent.jit(definition, backend=...)` objects and pass them to these compositions.
The compiler does not select algorithms from program names.

## Integrate with a framework

```bash
python examples/softmax.py --backend triton --prepared
python examples/softmax.py --backend cutile --inspect-native
python examples/softmax.py --backend triton --torch-compile
python examples/softmax_forward_backward.py --backend mojo --torch-compile
```

GPU and Mojo can register a PyTorch custom op after preparing one actual device binding.
Authors define backward; complete compilation and tuning before graph capture.
`softmax.py --program /path/to/export` demonstrates explicit artifact loading.

See [compile and run](../doc/getting-started/usage.en.md) for interfaces and cache/execution boundaries.
