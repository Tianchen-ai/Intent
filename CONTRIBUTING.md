# Contributing to IntentDSL

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

Contributions can improve language usability, reusable compiler passes, provider lowering, runtime integration, or documentation. Start with the [handbook](https://tianchen-ai.github.io/Intent/en/) and the module relevant to your change. [Compiler development entry points](doc/development/compiler-reference.en.md) locate the public implementation and reusable components.

## Report a problem

Search [existing issues](https://github.com/Tianchen-ai/Intent/issues) first. For a new issue, include:

- A minimal source program or the affected existing example, its input shapes and dtypes, and the command used.
- The expected behavior and actual diagnostic, including the failed stage and relevant artifact paths.
- The selected backend, device, driver or SDK/compiler, Python environment, and whether source generation, native compilation, or execution failed.
- For a performance issue, the complete call's timing scope, input size, configuration, and whether JIT/tuning had already completed.

Use `intent describe --target BACKEND --json` to inspect the public interface and `intent doctor --target BACKEND --json` to inspect the selected environment. Compilation diagnostics and `artifact.cache_directory` locate the actual source, IR, and logs.

## Set up a development checkout

Fork and clone [the repository](https://github.com/Tianchen-ai/Intent), then follow the [installation guide](environment/README.md). Source builds use the LLVM/MLIR 20 SDK.

```bash
python3 environment/install.py --backend triton --venv .venv-triton --examples
source .venv-triton/bin/activate
python -m examples.use --list
```

For repeated C++ development, configure a separate build directory:

```bash
cmake -S . -B "$HOME/.cache/intentdsl/development" -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/.cache/intentdsl/development" \
  --target intent-compile intent-opt --parallel 4
```

Use `INTENT_COMPILER` to point the public Python tools at that build's `tools/intent-compile/intent-compile`. Reinstall the checkout when Python code changes, and check `intent.__file__` to confirm which installation you are using. Other providers require their declared external toolchains.

## Choose the right module

```text
Python definition → typed frontend → canonical KIR
                                      ├─ GPU IR and passes → Triton / cuTile
                                      ├─ CPU IR and passes → Mojo / Weft
                                      └─ DSA IR and passes → BANG C
                                    → provider source → native runtime
```

| Change | Start here |
|---|---|
| Author algorithms and complete calls | [examples/kernels/](examples/kernels/) and [examples/programs/](examples/programs/) |
| Language and frontend | [python/intent/language/](python/intent/language/), [python/intent/frontend/](python/intent/frontend/), [authoring contract](doc/dsl/authoring.en.md) |
| Canonical semantic analysis | [include/Intent/Analysis/](include/Intent/Analysis/) and [lib/Analysis/](lib/Analysis/) |
| KIR-to-execution-model construction | [lib/Conversion/](lib/Conversion/) |
| GPU, CPU, or DSA passes | [lib/Dialect/](lib/Dialect/) and matching [include/Intent/Dialect/](include/Intent/Dialect/) |
| Provider realization and serialization | [lib/Target/](lib/Target/) |
| Public calls and native integration | [python/intent/compiler/](python/intent/compiler/), [python/intent/targets/](python/intent/targets/), [python/intent/runtime/](python/intent/runtime/) |
| CLI and MCP | [python/intent/tools/](python/intent/tools/), [python/intent/mcp/](python/intent/mcp/), [MCP setup](mcp/README.md) |
| Installation and distribution | [environment/](environment/) and [cmake/](cmake/) |
| Bilingual documentation | [doc/](doc/); [documentation workflow](https://tianchen-ai.github.io/Intent/en/development/documentation/) |

`doc/` defines the stable language and compiler contracts. Read the relevant chapters before changing a semantic boundary. GPU and CPU physical programs are different execution models; reuse semantic analysis where it applies, and keep transformations in their own family.

## Contribute a reusable pass

A pass should express a reusable transformation of the current typed IR. Begin with the [pass and analysis contract](doc/compiler/passes-and-analyses.en.md) and [pass development guide](https://tianchen-ai.github.io/Intent/en/development/passes/).

1. State the facts consumed, legality conditions, actual IR rewrite, and preserved values, control, effects, numerics, and ABI.
2. Reuse existing analyses and IR carriers. Put shared semantic queries in the analysis layer; keep GPU, CPU, or DSA execution decisions in that family.
3. Declare the complete transformation and its options in the adjacent `Passes.td`, with implementation in the appropriate responsibility directory and public headers under `include/Intent/`.
4. Define analysis invalidation and verification at the completed transformation boundary. A serializer spells the decided program rather than selecting algorithms or reconstructing execution.
5. Check the mature target's corresponding mechanism. Delegate provider-native collectives, layouts, and instruction lowering to the provider when their contracts match.
6. Verify through the affected complete product programs, and explain why the rule applies beyond one spelling or operator.

Policies use current semantics, def-use, coordinates, effects, lifetime, and capabilities rather than kernel names. An author's algorithm, state order, and explicit multi-kernel calls remain authoritative. A legal program that fails to lower is a compiler gap.

## Verify and submit the change

Use the affected programs from the existing 30-program entry point:

```bash
python -m examples.use softmax --target triton --stage source
python -m examples.use softmax --target triton --stage native
python -m examples.use softmax --target triton
git diff --check
```

These commands illustrate the stages; choose the existing program and target affected by your change. Keep its algorithm, inputs, and numerical contract. Use its established tolerance for necessary numerical checks. Native execution requires the actual toolchain and device; source generation alone is not evidence of runtime correctness or performance.

Generated IR/source, build caches and performance observations stay outside the repository. Reuse the canonical algorithms in `examples/kernels/` and host composition in `examples/programs/` rather than duplicating them for a separate test path.

For distribution changes, run the existing `environment/build.py` recipe. For documentation changes, build the bilingual site as described in the [documentation guide](https://tianchen-ai.github.io/Intent/en/development/documentation/).

Package publication is a separate maintainer action in the Distribution workflow: `publish-preview` updates public GitHub assets and `publish-pypi` uploads the checked build using repository secret `PYPI_API_TOKEN`. Pull requests do not upload packages. See the [publication instructions](doc/development/documentation.en.md#configure-pypi-publication).

Keep a pull request coherent. Describe the concrete problem, resulting behavior, relevant compiler boundary, and verification actually performed. For optimization, separate compile/tuning costs from hot complete-call time and explain the structural reason for a gain. Follow the surrounding code's style and preserve third-party notices.

IntentDSL uses [Apache-2.0](LICENSE). Public implementation APIs and reuse locations are listed in the [compiler development guide](doc/development/compiler-reference.en.md).
