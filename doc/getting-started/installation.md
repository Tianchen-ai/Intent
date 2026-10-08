# 安装

IntentDSL 可以通过 pip 从源码构建，或安装已有 Linux wheel。当前没有已发布的 PyPI 包；不要把 `pip install intentdsl` 当作已经可用的公开下载命令。

## 从源码安装

编译器需要 Linux、Python 3.10–3.12、CMake、Ninja、C++17 编译器以及 LLVM/MLIR 20 C++ SDK（headers、libraries、TableGen 与 CMake packages）。Debian/Ubuntu 可以从 [LLVM 软件仓库](https://apt.llvm.org/)安装 `llvm-20-dev`、`libmlir-20-dev` 与 `mlir-20-tools`。不需要 Python MLIR bindings。

创建虚拟环境需要所选 Python 的 venv 支持。Debian/Ubuntu 的 Python 3.10 使用 `python3.10-venv`；如果出现 `ensurepip is not available`，先安装这个系统包，再运行安装脚本。

```bash
git clone https://github.com/Tianchen-ai/Intent.git
cd Intent
python3 environment/install.py --backend triton --venv .venv-triton --examples
source .venv-triton/bin/activate
intent doctor --target triton
python examples/softmax.py
```

安装脚本创建虚拟环境、构建并安装编译器与 Python 包、调用已安装的 `intent setup` 安装所选后端的 Python 依赖，然后运行环境检查。它不安装 GPU driver、外部 CPU 编译器或 NeuWare SDK。构建和缓存默认留在用户缓存中。

cuTile 使用独立环境：

```bash
python3 environment/install.py --backend cutile --venv .venv-cutile --examples
source .venv-cutile/bin/activate
intent doctor --target cutile
python examples/softmax.py --target cutile
```

用 `--mlir-dir`、`--llvm-dir` 指定非系统 SDK；`--build-dir` 选择仓库外构建目录，`--jobs` 限制构建并行度。现有后端依赖组合由安装工具声明，`intent setup` 安装这些组合并执行 `pip check`；依赖冲突直接报错。

## 安装现有 wheel

从[公开预览发行](https://github.com/Tianchen-ai/Intent/releases/tag/preview)下载 wheel，无需 GitHub 登录。设 `INTENT_WHEEL` 为真实 `.whl` 路径，保留完整文件名：

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install "${INTENT_WHEEL}[manual,examples]"
intent setup --target triton
intent doctor --target triton
```

Wheel 包含 `intent-compile`、`intent-opt`、配置表、语言手册和编译器需要的非系统动态库；安装不需要 LLVM/MLIR SDK 或源码 checkout。标准 `environment/build.py` 分发脚本以 Linux x86-64、glibc ≥ 2.35 为平台基线，使用 auditwheel 实际检查 ELF 依赖与符号并修复为 `manylinux_2_35_x86_64`。主机仍须具有兼容架构、系统 ABI 及所选后端环境；直接 `pip wheel .` 的本地 wheel 不自动满足该策略。构建成功也不代表设备执行或数值正确。

只使用 KIR 工具和 MCP 时，可省略 `intent setup`；`intent doctor --json` 检查基础编译器，`intent describe --json` 查看公开接口。

## CPU 与 MLU

```bash
python3 environment/install.py --backend mojo --venv .venv-mojo \
  --provider-compiler /path/to/mojo --examples
python3 environment/install.py --backend weft --venv .venv-weft \
  --weft-source-dir /path/to/Weft --weft-binary-dir /path/to/weft-build --examples
python3 environment/install.py --backend bangc --venv .venv-bangc \
  --neuware /path/to/neuware --examples
```

这些命令使用已有工具链，不安装设备或外部编译器。Weft 的实际 vector width、worker 数和 native profile 由部署明确提供；BANG C 执行需要兼容 MLU 设备。带 Weft 的 wheel 必须在构建时包含该 provider。

完整安装参数、依赖组合、wheel 分发和第三方 notices 见仓库的[安装指南](https://github.com/Tianchen-ai/Intent/blob/main/environment/README.md)。编译器构建脚本是 `environment/build.py`，从源码构建 sdist 和 wheel，输出目录必须在 checkout 外。
