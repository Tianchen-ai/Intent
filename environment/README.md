# Installation

[English](README.md) | [简体中文](README.zh-CN.md)

The public setup uses Linux and Python 3.10–3.12. Install an Intent wheel, then use the installed `intent setup` command to add one backend's Python dependencies. The source checkout also provides a virtual-environment bootstrap. Python MLIR bindings are not needed. GPU execution requires an appropriate NVIDIA driver; CPU and MLU toolchains are selected explicitly below. Installation, successful compilation, and numerical correctness are separate checks.

IntentDSL has a Python package and a wheel build route. It has not been published to PyPI, so `pip install intentdsl` is not an installation route yet. Download a wheel from the [public development preview](https://github.com/Tianchen-ai/Intent/releases/tag/preview), or build one below. A wheel installs the compiler independently of any provider runtime.

## Install an existing wheel

Set `INTENT_WHEEL` to the actual path of a built `.whl` file, preserving its filename. No checkout or LLVM/MLIR SDK is needed:

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install "${INTENT_WHEEL}[manual]"
intent setup --target triton
intent doctor --target triton
```

This route does not inspect SDK paths, build C++, or install MLIR bindings. The wheel includes `intent-compile`, `intent-opt`, their shared profiles, the language manual, and the tools' non-system ELF dependencies. The binaries find their libraries through relative `$ORIGIN` paths. Third-party copyright/license files and a library-to-notice listing are installed under `intent/_bin/third-party/`.

The distribution recipe produces `manylinux_2_35_x86_64` wheels for mainstream Linux x86-64 systems with glibc 2.35 or newer. It uses [auditwheel](https://github.com/pypa/auditwheel) to inspect the actual compiler executables, bundled libraries, external symbol versions and CPU ISA requirements, then repair the wheel for that policy. It does not make newer symbols work on an older system: an incompatible build fails rather than receiving a misleading tag. Backend/device requirements remain separate from the compiler wheel. For an environment whose backend dependencies are already installed, `python -m pip install "${INTENT_WHEEL}[manual]"` installs the package directly without a checkout.

For compiler/KIR tools and the MCP servers alone, omit `intent setup`: the wheel installation needs no provider SDK or tensor framework. Run `intent doctor --json` to check the installed compiler and its KIR output stage. Add a selected backend's dependencies when you need its runtime; a base compiler check does not establish device or kernel support.

`intent setup --target cutile` selects the public cuTile dependency route. Setup invokes pip in the Python environment containing that `intent` command, resolving PyTorch and provider dependencies together, then runs `pip check`. A dependency conflict is an installation error. It does not install drivers, external CPU/MLU compilers, or probe a device. Follow it with `intent doctor --target BACKEND` and the target options for your deployment. `--torch-index-url` selects another wheel index for the same declared PyTorch build; `--json` returns installation status while pip output goes to stderr. Use separate virtual environments for different GPU routes.

From a checkout, `python3 environment/install.py --backend triton --venv .venv-triton --wheel "$INTENT_WHEEL"` combines venv creation, wheel installation, the installed setup command and doctor. The checkout's bootstrap does not maintain a second dependency installation path. After installation, [the public softmax example](../examples/softmax.py) demonstrates a complete kernel invocation.

## Source build prerequisites

Install CMake, Ninja, a C++17 compiler, Python's venv support, and the LLVM/MLIR 20 C++ SDK: headers, libraries, TableGen tools, and CMake packages. With the [LLVM package repository](https://apt.llvm.org/) configured on Debian/Ubuntu, the relevant packages are `llvm-20-dev`, `libmlir-20-dev`, and `mlir-20-tools`. The SDK is a build dependency; the installed C++ compiler owns KIR parsing, normalization, and verification.

For Python 3.10 on Debian/Ubuntu, venv support is supplied by `python3.10-venv`. Install it if environment creation reports `ensurepip is not available`; the bootstrap does not install system packages.

## Choose a backend

Use a separate virtual environment for each public GPU route. The dependency declarations are in [backends.py](../python/intent/tools/backends.py); the installer and installed environment checker consume the same backend descriptions.

| Backend | Python runtime route | External requirements and current scope |
|---|---|---|
| `triton` | PyTorch 2.10.0+cu130, Triton 3.6.0, NumPy 1.26.4 | NVIDIA driver; callable GPU artifacts |
| `cutile` | PyTorch 2.10.0+cu128, cuTile 1.6.0, CUDA toolkit 13.3.1; see [dependency declarations](../python/intent/tools/backends.py) | NVIDIA driver compatible with the selected CUDA toolkit; callable GPU artifacts |
| `mojo` | PyTorch 2.10.0+cpu | Existing Mojo compiler, Linux x86-64 with AVX2/AVX512; callable CPU artifacts |
| `weft` | No additional tensor framework for source generation | Intent compiler built with Weft; public `WeftTarget` generates canonical Weft source, native execution requires its deployment/toolchain |
| `bangc` | Native buffer interface uses the Python standard library | Existing NeuWare SDK and MLU370 implementation; execution requires an actual compatible MLU device |

For Triton, from the Intent checkout:

```bash
python3 environment/install.py --backend triton \
  --venv .venv-triton
source .venv-triton/bin/activate
intent doctor --target triton
python examples/softmax.py
```

For cuTile, using the same source SDK:

```bash
python3 environment/install.py --backend cutile \
  --venv .venv-cutile
source .venv-cutile/bin/activate
intent doctor --target cutile
python examples/softmax.py --target cutile
```

The public cuTile route uses cuda-tile 1.6 and the matched CUDA 13.3 compiler components. Its PyTorch cu128 build uses the separate CUDA 12 runtime packages: PyTorch cu130 locks CUDA 13 runtime-library versions that conflict with this compiler toolkit's requirements. PyTorch supplies tensor storage and streams; cuTile owns TileIR compilation and driver launches. See the [PyTorch builds](https://pytorch.org/get-started/previous-versions/) and [cuTile compiler installation](https://docs.nvidia.com/cuda/cutile-python/quickstart.html). Backend/device support still depends on the selected SDK and the kernel's operations.

The common installer:

1. Creates or reuses the chosen virtual environment, without adding the caller's `PYTHONPATH` or user site packages.
2. Installs the selected wheel, or builds `intent-compile` and `intent-opt` and packages their runtime libraries and notices.
3. Calls the installed `intent setup` to install only the selected backend's Python dependencies.
4. Queries the installed compiler and checks the selected backend. The public CLI and both MCP entry points are included.

For another build SDK location, supply `--mlir-dir` and `--llvm-dir`, or set `INTENT_MLIR_DIR` and `INTENT_LLVM_DIR`. Add `--wheel "$INTENT_WHEEL"` to install an existing wheel instead; SDK/build flags are then unnecessary and rejected. `--torch-index-url` overrides the selected route's wheel index for the same PyTorch release; check the [PyTorch installation combinations](https://pytorch.org/get-started/previous-versions/).

Intent's C++ build defaults to the user cache, outside the checkout. `--build-dir` selects a different build directory and `--jobs` limits build parallelism. Re-running the installer reuses the environment and build directory. Failures stop installation and preserve the underlying command's diagnostics.

## Select an external CPU or MLU toolchain

These commands install Intent around existing toolchains; they do not install a compiler, driver, remote machine, or NeuWare license:

```bash
python3 environment/install.py --backend mojo --venv .venv-mojo \
  --provider-compiler /path/to/mojo

python3 environment/install.py --backend weft --venv .venv-weft \
  --weft-source-dir /path/to/Weft --weft-binary-dir /path/to/weft-build

python3 environment/install.py --backend bangc --venv .venv-bangc \
  --neuware /path/to/neuware
```

For a wheel, add `--wheel "$INTENT_WHEEL"`. Omit Weft source/build flags on that route: the wheel must already contain the Weft provider, which the installed compiler reports. Use selected external compiler/SDK paths in later target constructors or CLI options. For example:

```bash
.venv-mojo/bin/intent doctor --target mojo --target-option 'compiler="/path/to/mojo"'
.venv-weft/bin/intent doctor --target weft --target-option vector_bits=256 --target-option workers=8
.venv-bangc/bin/intent doctor --target bangc --target-option 'neuware="/path/to/neuware"'
```

Weft's vector width and worker count above are examples of explicit construction budgets; supply the values for your deployment. Its CMake integration requires matching LLVM/MLIR and generated Weft headers/libraries. Use the [complete examples](../examples/README.en.md) with your selected backend for compilation and execution.

## Build the package directly

The distribution recipe for Ubuntu 22.04, Linux x86-64, CPython 3.10–3.12 and the LLVM/MLIR 20 SDK is:

```bash
python3 environment/build.py \
  --output-dir /path/outside-checkout/dist \
  --work-dir /path/outside-checkout/build \
  --jobs 8
```

Use a new work directory and a separate output directory. The command builds the source distribution first, builds the wheel from that archive, applies auditwheel's `manylinux_2_35_x86_64` inspection and repair, and checks distribution metadata with `twine check --strict`. It installs the repaired wheel into an isolated environment outside the checkout, then runs the existing base doctor, API discovery, both MCP entry points, the original softmax definition's KIR compilation, and standard IR optimization. It needs no GPU or provider SDK. The output directory receives the source archive and repaired wheel after these steps complete; the work directory retains the installed environment, command output and compiler artifacts, including on failure.

This recipe fixes the native build route to GCC, Ninja and Release mode and uses the existing CMake runtime bundling rules. It accepts `--mlir-dir`, `--llvm-dir`, `--runtime-notices`, and the paired `--weft-source-dir` / `--weft-binary-dir` options. Its platform policy follows [PEP 600](https://peps.python.org/pep-0600/); a custom SDK must satisfy the same actual ABI and ISA checks. It does not publish the package or claim byte-identical builds. Other source environments can use the manual build below, which produces a local platform wheel unless separately repaired.

### Download a CI build

The [Distribution workflow](../.github/workflows/distribution.yml) invokes this same recipe for main-branch pushes and pull requests affecting the product, build inputs or documentation, and can also be started with **Run workflow** in GitHub Actions. It uses a GitHub-hosted Ubuntu 22.04 x86-64 runner, Python 3.10 and the LLVM/MLIR 20 packages. The base job installs no Torch, GPU SDK, external CPU compiler or device runner; optional Weft support is not built into this artifact.

Open the chosen workflow run and download `intentdsl-ubuntu-22.04-x86_64` from its **Artifacts** section. Extract the archive and install its repaired manylinux wheel using the existing wheel instructions above, then select `intent setup --target triton` or `--target cutile` in the corresponding separate environment. The archive also contains the source distribution. These are artifacts of that reviewed run, retained for 14 days, rather than a PyPI or GitHub release. They carry the same platform policy and third-party notices as the direct distribution build.

The `intentdsl-distribution-diagnostics` artifact retains the build transcript, ordered `commands/*.command`, `.log` and `.status` files, and the installed tools' actual JSON/IR/log outputs, including on failure. The build helper uses the same command logs locally. Hosted CI establishes packaging, installed compiler startup and the original softmax KIR/optimization path; device execution and performance use the thirty [complete examples](../examples/README.en.md). The fixed Ubuntu user space is part of the binary ABI baseline, so changing the runner to `ubuntu-latest` is not an equivalent build.

To install the compiler directly from the checkout once the SDK is installed:

```bash
python -m pip install '.[manual]' \
  --config-settings=cmake.define.MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  --config-settings=cmake.define.LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  --config-settings=build-dir=/path/to/intent-build
```

To build a wheel for later installation, use `python -m pip wheel . --no-deps --wheel-dir /path/to/wheelhouse` with the same CMake settings. PEP 517 installs build-only `patchelf` in its isolated build environment; provide it yourself if disabling build isolation. The package contains an independent compiler executable, so its Python ABI tag is `py3-none`; its platform tag remains Linux-specific. Package metadata comes from Git; the build does not create a release tag.

The installed package discovers its own compiler. Developers can override it with `compiler="/path/to/intent-compile"` or `INTENT_COMPILER`. Keep `profiles/`, private `lib/`, and the executable together. A normal CMake developer installation outside the Python wheel build continues to use the selected SDK's runtime libraries.

### Runtime libraries and notices

Wheel installation rules collect the actual compiler's ELF dependency closure. Unresolved dependencies, conflicting SONAMEs, or missing notices fail packaging. Libraries keep their SONAMEs; the staged copies receive relative RPATHs. The SDK's original files are not changed. glibc, its loader, libstdc++, and libgcc remain platform requirements.

For a Debian/Ubuntu SDK, notices come from each resolved library's actual package ownership and copyright file, including referenced `/usr/share/common-licenses/` texts. For custom SDKs or libraries without package ownership, provide a JSON object mapping exact library paths to their full notice/license files. Relative paths are resolved against the JSON file's directory:

```json
{
  "/path/to/sdk/lib/libMLIR.so.20.1": ["licenses/llvm-LICENSE.txt"],
  "/path/to/sdk/lib/libLLVM.so.20.1": ["licenses/llvm-LICENSE.txt"]
}
```

Use `--runtime-notices /path/to/notices.json` with the source installer, or `--config-settings=cmake.define.INTENT_RUNTIME_NOTICES=/path/to/notices.json` with `pip wheel`. The example covers only those two libraries; all other copied dependencies also need their corresponding notices. The map supplies attribution documents, not permission to redistribute third-party software or a license for Intent itself.

## Compile and diagnose

`intent describe --target BACKEND --json` reads the installed public API's signatures, compile option declarations, and target fields without probing an SDK or device. Omit `--target` to see all declared provider routes. `intent doctor --json` checks the base compiler independently of those routes.

`intent doctor --target BACKEND --json` launches the compiler's `--compiler-info` query and checks its actual built providers, then checks only the selected backend's dependencies and target facts. A built source provider is not a claim that a device or kernel is supported. `intent compile path/to/program.py:kernel --stage kir --json` captures, normalizes, and verifies KIR without a target/device. `--target BACKEND` instead selects full provider generation; `--stage shared --target BACKEND` stops after the execution-family program. These commands use the public pipeline and return artifact paths or the actual failure stage. An importable `package.module:kernel` is also accepted. `--constexpr NAME=JSON`, `--target-option NAME=JSON`, `--compiler`, and `--tuning-config` forward existing public settings. `--materialize` applies only to provider generation; the tool does not actively launch the selected kernel.

Use `intent read-artifact PATH --offset 0 --limit 16000 --json` on a returned source, IR, metadata, or log path. Offsets count Unicode characters; continue with `next_offset` until `eof`. Compilation failures retain their actual stage, exception causes and artifact paths; native compilation may report a different directory from the Intent compiler's output.

The `intent-manual` MCP stays read-only. Explicitly enable `intent-compiler-mcp` when an agent should compile an existing user-supplied `.py` file; its compilation tool shares the CLI implementation. Its `describe`, `environment`, and `read_artifact` tools provide discovery, prerequisite inspection, and paged artifact/log reading through that same implementation. Both servers use the `manual` dependency extra. Module loading executes ordinary top-level Python host code, while kernel execution and numerical validation remain separate.
