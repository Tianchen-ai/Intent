# Installation

Install on the machine that executes your kernels. Backend selection is explicit;
device resources are discovered there. Intent's Python package imports as `intent`.

## Existing wheel

```bash
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
python -m pip install 'intentdsl[manual,examples]'
intent setup --target triton
intent doctor --target triton
python -m intent.examples softmax --backend triton
```

Use `pip install /path/to/intentdsl.whl` for a locally built wheel. The public x86-64
wheel requires a compatible glibc system; a native wheel must match the execution
host's CPU architecture and system ABI. Wheels include intent-compile, intent-opt,
profiles, manual resources and required non-system compiler runtime libraries.
Installing a matching wheel needs no LLVM development SDK.

The manual extra installs MCP dependencies; examples installs portable array dtypes.
Provider dependencies and drivers remain separate. setup installs only the declared
Python dependencies from [the provider registry](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/targets/registry.py).
doctor resolves the installed compiler, provider toolchain and actual local target;
it does not execute a numerical test.

## Build on the execution host

Source builds need Linux, Python 3.10–3.12, CMake, Ninja, C++17 and the LLVM/MLIR 20 SDK
(headers, libraries, TableGen and CMake packages). Debian/Ubuntu packages include
llvm-20-dev, libmlir-20-dev and mlir-20-tools. Python MLIR bindings are unnecessary.

To include the Weft provider, build its canonical IR library against the same SDK
and pass both Weft flags below. Other providers can omit the Weft build and flags:

```bash
cmake -S /path/to/weft -B /path/to/weft-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR=/path/to/mlir/cmake -DLLVM_DIR=/path/to/llvm/cmake
cmake --build /path/to/weft-build --target WeftKernelDialect
python3 environment/install.py --backend triton --venv /path/to/venv --examples \
  --mlir-dir /path/to/mlir/cmake --llvm-dir /path/to/llvm/cmake \
  --weft-source-dir /path/to/weft --weft-binary-dir /path/to/weft-build \
  --build-dir /path/to/build/intent --jobs 8
```

The installer reuses the selected environment/build directory, installs the package,
runs setup and queries doctor. It does not install drivers, external compilers or services.
Build directories default to the user cache; choose explicit paths for managed development.
Use a complete SDK: a partial LLVM source build must first build its required libraries.

## Provider toolchains

| Backend | Execution host and dependencies |
|---|---|
| triton / cutile | CUDA host with compatible NVIDIA driver and selected Python provider |
| mojo | x86 host with Mojo, a C compiler (cc) and the GNU OpenMP runtime (libgomp.so.1); select Mojo through INTENT_MOJO or backend_options |
| weft | x86/RISC-V host with the full Triton–Weft distribution supporting Canonical `.weftir`, matching weft-core native SDK and CPU toolchain |
| bangc | MLU370 host with CNRT/CNCC from NeuWare |
| triton_ascend | Ascend NPU host with matching CANN, torch-npu and official Triton–Ascend |

For source installs, retain the common SDK flags, include the Weft canonical-library
flags when building that provider, and select:

```bash
# Additional install.py flags for the chosen backend:
# --backend mojo --target-option 'compiler="/path/to/mojo"'
# --backend weft
# --backend bangc --target-option 'neuware="/path/to/neuware"'
# --backend triton_ascend
```

Weft uses the complete Triton–Weft JIT: Canonical `.weftir` enters `triton.compile`, its cache
and native task export. Intent owns task scheduling; each provider task entry is single-threaded.
Users select affinity and worker budget, without supplying ISA/VLEN. Install a distribution
supporting this input format; the old `weft-compile` route is not a substitute. BANG C queries CNRT resources; currently
unsupported architectures are diagnosed instead of being treated as MLU370.
Ordinary BANG C JIT derives shape/stride bindings from actual buffers.

The Triton SDK selects its PTX assembler; `intent doctor --target triton`
reports the actual path and version. To select another local CUDA assembler,
set `TRITON_PTXAS_PATH=/path/to/ptxas` before starting Python; Blackwell uses
`TRITON_PTXAS_BLACKWELL_PATH`. Intent does not switch global SDK settings
during a call or retry a failed compilation/execution with another assembler.
Validate actual kernels again after changing the toolchain.

```python
run = intent.jit(definition, backend="mojo",
                 backend_options={"compiler": "/path/to/mojo"})
```

Doctor uses the same local target resolver. For example:
`intent doctor --target weft`. Explicit Weft toolchain options use
`backend_options={"options": {...}}` and the provider's own option parser.
Independent product calls use their own INTENT_CACHE_DIR/XDG settings; Bench redirects them
only for Bench-owned runs.

NVIDIA Triton, Triton–Weft and Triton–Ascend all provide the `triton` Python package.
Use separate virtual environments, installing Intent and the selected provider normally.
Do not overwrite distributions in place, change `sys.path`, or replace the global driver to switch providers.
Triton–Ascend consumes Block directly. Discovery reads the actual architecture and Cube/Vector core counts;
compatibility shared-memory placeholders are not physical capacity facts.

## Build a distributable package

```bash
python3 environment/build.py --platform native \
  --output-dir /path/to/dist --work-dir /path/to/package-work \
  --mlir-dir /path/to/mlir/cmake --llvm-dir /path/to/llvm/cmake \
  --weft-source-dir /path/to/weft --weft-binary-dir /path/to/weft-build
```

Native mode supports execution hosts such as x86-64 and RISC-V. The script builds from an
sdist, bundles runtime libraries/notices and checks the installed base compiler outside the
checkout. `--platform manylinux_2_35_x86_64` explicitly selects the Ubuntu 22.04 x86-64
auditwheel recipe. A native wheel is not automatically a manylinux wheel.

For custom SDK libraries, --runtime-notices supplies a JSON mapping from each shared library
to its license/notice files. Outputs remain outside the checkout. These commands neither
publish nor push. Compiler installation/generation, native compilation, execution and
numerical correctness are distinct outcomes.
