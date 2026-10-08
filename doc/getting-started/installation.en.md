# Installation

Install IntentDSL from [PyPI](https://pypi.org/project/intentdsl/), or use the source and GitHub wheel routes below. Public wheels support Linux x86-64, Python 3.10–3.12 and glibc ≥ 2.35. They include the independent compiler without requiring a local LLVM/MLIR SDK.

## Install from PyPI

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install 'intentdsl[manual,examples]'
intent setup --target triton
intent doctor --target triton
```

`pip install intentdsl` installs the compiler and Python API. The `manual` extra adds dependencies for both MCP servers; `examples` adds product-program input support. Complete algorithms and host programs live in the [source repository](https://github.com/Tianchen-ai/Intent/tree/main/examples); see [compile and run](usage.md).

Omit `intent setup` for compiler/KIR/MCP tools alone. GPU execution needs a compatible NVIDIA driver. Use `intent setup --target cutile` in a separate environment for cuTile; external CPU and MLU toolchains are described below. If no wheel matches your platform, pip's source build requires the LLVM/MLIR 20 SDK.

## Install from source

The compiler needs Linux, Python 3.10–3.12, CMake, Ninja, a C++17 compiler and the LLVM/MLIR 20 C++ SDK: headers, libraries, TableGen tools and CMake packages. Debian/Ubuntu users can install `llvm-20-dev`, `libmlir-20-dev` and `mlir-20-tools` from the [LLVM package repository](https://apt.llvm.org/). Python MLIR bindings are unnecessary.

Virtual environment creation also needs venv support for the selected Python. On Debian/Ubuntu with Python 3.10, install `python3.10-venv`. If creation reports `ensurepip is not available`, install that system package before running the bootstrap.

```bash
git clone https://github.com/Tianchen-ai/Intent.git
cd Intent
python3 environment/install.py --backend triton --venv .venv-triton --examples
source .venv-triton/bin/activate
intent doctor --target triton
python examples/softmax.py
```

The bootstrap creates a virtual environment, builds and installs the compiler and Python package, invokes the installed `intent setup` to install the selected backend's Python dependencies, and checks the environment. It does not install GPU drivers, external CPU compilers or NeuWare. Builds and caches stay in the user cache by default.

Use a separate environment for cuTile:

```bash
python3 environment/install.py --backend cutile --venv .venv-cutile --examples
source .venv-cutile/bin/activate
intent doctor --target cutile
python examples/softmax.py --target cutile
```

Select a non-system SDK with `--mlir-dir` and `--llvm-dir`, an external build directory with `--build-dir`, and build parallelism with `--jobs`. Backend dependency combinations are declared by the installation tools. `intent setup` installs the selected combination and runs `pip check`; conflicts are reported as errors.

## Install an existing wheel

Download a wheel from the [public development preview](https://github.com/Tianchen-ai/Intent/releases/tag/preview); no GitHub login is needed. Set `INTENT_WHEEL` to a real `.whl` path, keeping its complete filename:

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install "${INTENT_WHEEL}[manual,examples]"
intent setup --target triton
intent doctor --target triton
```

The wheel includes `intent-compile`, `intent-opt`, configuration tables, the language manual and the compiler's non-system dynamic libraries. Installation needs neither an LLVM/MLIR SDK nor a checkout. The standard `environment/build.py` distribution recipe targets Linux x86-64 with glibc ≥ 2.35, uses auditwheel to inspect real ELF dependencies and symbols, and repairs the artifact to `manylinux_2_35_x86_64`. Hosts still need a compatible architecture, system ABI and backend environment. A local wheel made directly with `pip wheel .` does not automatically satisfy that policy. Packaging does not establish device execution or numerical correctness.

For KIR tools and MCP alone, omit `intent setup`. `intent doctor --json` checks the base compiler; `intent describe --json` lists public interfaces.

## CPU and MLU toolchains

```bash
python3 environment/install.py --backend mojo --venv .venv-mojo \
  --provider-compiler /path/to/mojo --examples
python3 environment/install.py --backend weft --venv .venv-weft \
  --weft-source-dir /path/to/Weft --weft-binary-dir /path/to/weft-build --examples
python3 environment/install.py --backend bangc --venv .venv-bangc \
  --neuware /path/to/neuware --examples
```

These commands use existing toolchains. They do not install devices or external compilers. Supply Weft's actual vector width, worker count and native profile for the deployment. BANG C execution needs a compatible MLU device. A wheel offering Weft must include that provider at build time.

The repository [installation guide](https://github.com/Tianchen-ai/Intent/blob/main/environment/README.md) covers complete options, dependency combinations, distribution and third-party notices. `environment/build.py` builds the source distribution and wheel; output directories must be outside the checkout.
