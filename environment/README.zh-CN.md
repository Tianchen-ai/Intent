# 安装与构建

[English](README.md) | [简体中文](README.zh-CN.md)

公开安装入口使用 Linux 和 Python 3.10–3.12。IntentDSL 的 Python 包包含独立编译器；选择后端后，再安装该后端的运行依赖。源码构建不需要 Python MLIR bindings，需要 LLVM/MLIR 20 的 C++ SDK。

目前尚未发布到 PyPI，因此还不能通过 `pip install intentdsl` 从公共包索引安装。可从 [公开开发预览](https://github.com/Tianchen-ai/Intent/releases/tag/preview) 下载 wheel，或使用下面的源码构建入口。项目使用 [Apache-2.0](../LICENSE) 许可，wheel 中附带的外部运行库保留各自的许可和声明。

## 安装已有 wheel

将 `INTENT_WHEEL` 设为实际 `.whl` 文件的完整路径，然后在独立环境安装：

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install "${INTENT_WHEEL}[manual]"
intent setup --target triton
intent doctor --target triton
```

wheel 包含 `intent-compile`、`intent-opt`、经验配置、语言手册和必要的非系统 ELF 运行库。安装 wheel 不需要源码仓库、LLVM/MLIR SDK 或 C++ 构建。编译器通过相对 `$ORIGIN` 路径找到自身运行库，第三方声明位于 `intent/_bin/third-party/`。

分发脚本产出 `manylinux_2_35_x86_64` wheel，适用于 glibc 2.35 或更新版本的主流 Linux x86-64 系统。[auditwheel](https://github.com/pypa/auditwheel) 检查实际编译器 executable、私有库、外部符号版本和 CPU ISA，再执行修复；不兼容的构建直接失败，不仅修改标签。后端与设备要求独立于基础 wheel。如果只使用编译器、KIR 工具或 MCP，省略 `intent setup` 即可，不需要 GPU、PyTorch 或 provider SDK。用 `intent doctor --json` 检查基础编译器。

两个 MCP 入口 `intent-manual` 与 `intent-compiler-mcp` 共用 `manual` extra。第一个只读语言手册，第二个允许编译用户已有的 Python 源文件；连接方法见 [MCP](../mcp/README.zh-CN.md)。

## 从源码安装一个后端

先准备 CMake、Ninja、C++17 编译器、Python venv 支持，以及 LLVM/MLIR 20 的头文件、库、TableGen 工具和 CMake 配置。Debian/Ubuntu 配置 [LLVM 官方软件源](https://apt.llvm.org/) 后，SDK 包包括 `llvm-20-dev`、`libmlir-20-dev` 和 `mlir-20-tools`。

Debian/Ubuntu 的 Python 3.10 需要 `python3.10-venv`。如果创建环境时出现 `ensurepip is not available`，先安装该系统包；安装脚本不自动安装系统依赖。

在仓库根目录执行：

```bash
python3 environment/install.py --backend triton --venv .venv-triton
source .venv-triton/bin/activate
python examples/softmax.py
```

cuTile 使用独立环境：

```bash
python3 environment/install.py --backend cutile --venv .venv-cutile
source .venv-cutile/bin/activate
python examples/softmax.py --target cutile
```

脚本创建或复用 venv，用统一 Python 打包路径构建 IntentDSL，然后调用已安装的 `intent setup` 和 `intent doctor`。不会安装驱动、外部 CPU 编译器或 NeuWare SDK。重跑会复用环境和构建目录，失败保留原始诊断。

SDK 位于其他目录时，传入 `--mlir-dir`、`--llvm-dir`，或设置 `INTENT_MLIR_DIR`、`INTENT_LLVM_DIR`。构建缓存默认位于用户缓存目录，`--build-dir` 和 `--jobs` 控制位置与并发。已有 wheel 时添加 `--wheel "$INTENT_WHEEL"`，不再传源码构建参数。`--examples` 安装产品程序需要的 NumPy/BF16/FP8 输入支持。

| 后端 | Python 依赖路线 | 外部条件与范围 |
|---|---|---|
| `triton` | PyTorch 2.10.0+cu130、Triton 3.6.0、NumPy 1.26.4 | NVIDIA 驱动，可调用 GPU 产物 |
| `cutile` | PyTorch 2.10.0+cu128、cuTile 1.6.0、CUDA 13.3.1 编译组件 | 兼容的 NVIDIA 驱动，可调用 GPU 产物 |
| `mojo` | PyTorch 2.10.0+cpu | 已安装的 Mojo，Linux x86-64 AVX2/AVX512 |
| `weft` | 生成源码无需额外 tensor framework | Intent 编译器需构建 Weft；原生执行需对应部署和工具链 |
| `bangc` | 原生 buffer 接口使用 Python 标准库 | 已安装的 NeuWare SDK 与兼容 MLU370 设备 |

依赖的唯一声明是 [backends.py](../python/intent/tools/backends.py)。cuTile 的 CUDA 13.3 编译组件与 PyTorch cu128 的 CUDA 12 运行库分开：PyTorch cu130 锁定的 CUDA 13 运行库版本会与当前编译组件冲突。PyTorch 提供 tensor 存储和 stream，cuTile 负责 TileIR 编译及 driver launch。不同 GPU 路线使用独立 venv。安装成功不代表设备可用或所有 kernel 已验证。

## 选择已有 CPU 或 MLU 工具链

```bash
python3 environment/install.py --backend mojo --venv .venv-mojo \
  --provider-compiler /path/to/mojo

python3 environment/install.py --backend weft --venv .venv-weft \
  --weft-source-dir /path/to/Weft --weft-binary-dir /path/to/weft-build

python3 environment/install.py --backend bangc --venv .venv-bangc \
  --neuware /path/to/neuware
```

这些路径必须指向真实存在的工具链。Weft 的源码、生成头文件和库必须匹配当前 LLVM/MLIR SDK。wheel 路线不传 Weft 源码参数，其自身必须已包含 Weft 支持。

后续调用仍需明确目标部署，例如：

```bash
.venv-mojo/bin/intent doctor --target mojo --target-option 'compiler="/path/to/mojo"'
.venv-weft/bin/intent doctor --target weft --target-option vector_bits=256 --target-option workers=8
.venv-bangc/bin/intent doctor --target bangc --target-option 'neuware="/path/to/neuware"'
```

Weft 的 `vector_bits` 和 `workers` 需替换为实际部署参数。脚本不推断私有设备配置，不安装远程部署，不把生成源码视为原生执行成功。

## 构建 sdist 和 wheel

Ubuntu 22.04、Linux x86-64、CPython 3.10–3.12 和 LLVM/MLIR 20 SDK 上可使用现有分发脚本：

```bash
python3 environment/build.py \
  --output-dir /path/outside-checkout/dist \
  --work-dir /path/outside-checkout/build \
  --jobs 8
```

输出目录和新建工作目录都必须位于仓库外，互不嵌套；省略 `--work-dir` 会自动使用用户缓存目录。脚本先构建 sdist，再从该归档构建 wheel，执行 auditwheel 的平台检查和 `manylinux_2_35_x86_64` 修复，使用 `twine check --strict` 检查分发元数据，在仓库外干净 venv 安装修复后的 wheel，然后调用基础 doctor、API discovery、两个 MCP、原有 softmax 的 KIR 编译和标准 IR 优化。它不会执行设备 kernel。

产物检查完成后，输出目录得到 `.tar.gz` 与 `.whl`；工作目录保留命令、日志、安装环境和编译产物。不会覆盖已有分发文件。`--mlir-dir`、`--llvm-dir`、`--runtime-notices` 和成对的 Weft 源码/构建目录参数用于选择现有 SDK。

其他源码构建环境可直接调用标准 Python 打包入口：

```bash
python -m pip install '.[manual]' \
  --config-settings=cmake.define.MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  --config-settings=cmake.define.LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  --config-settings=build-dir=/path/to/intent-build
```

将 `pip install` 换为 `pip wheel . --no-deps --wheel-dir /path/to/wheelhouse`，并保留 CMake 设置，可只构建本地平台 wheel；这一手动入口不会自动执行 auditwheel 修复。标准分发脚本采用 [PEP 600](https://peps.python.org/pep-0600/) 平台策略，自定义 SDK 也必须满足真实符号和 ISA 条件。构建隔离环境自动安装打包需要的 `patchelf`；关闭隔离时需自行提供。版本沿用现有 Git 元数据，脚本不创建发布标签，也不发布到 PyPI。

## GitHub 构建产物

[Distribution workflow](../.github/workflows/distribution.yml) 对相关 main push、pull request 和手动运行调用同一分发脚本。打开运行页面，在 Artifacts 中下载 `intentdsl-ubuntu-22.04-x86_64`，解压并安装实际 wheel，再选择所需后端。CI 基础 wheel 不构建可选 Weft，不安装 Torch 或设备 SDK。

这些是对应运行的 manylinux 构建产物，保留 14 天，不等于 PyPI 或 GitHub Release。`intentdsl-distribution-diagnostics` 保存诊断。CI 验证安装后的编译器和 KIR 工具；设备运行与性能通过既定 [30 个产品程序](../examples/README.md) 观察。

## 运行库声明和常用诊断

打包收集实际 ELF 依赖；无法解析依赖、冲突的 SONAME 或缺少第三方声明会直接失败。Debian/Ubuntu SDK 的声明来自实际归属包和其 copyright 文件。自定义 SDK 使用 `--runtime-notices` 提供库路径到完整许可文件列表的 JSON 映射，详见 [英文说明](README.md#runtime-libraries-and-notices)。只修改打包副本的 RPATH，不修改 SDK 原文件。

`intent describe --json` 列出公共 API，`intent doctor --target BACKEND --json` 检查所选环境；`intent compile path/to/program.py:kernel --stage kir --json` 无需目标设备即可构建 KIR。完整 provider 生成使用 `--target BACKEND`，`--materialize` 执行原生编译，不主动运行 kernel。编译返回源码、IR、日志和元数据路径；`intent read-artifact PATH --json` 可以分页读取。生成源码、原生编译、运行正确与性能结果分别判断。
