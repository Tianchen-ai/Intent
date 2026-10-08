# IntentDSL

**写一次算子算法，编译到不同执行模型。**

[English](README.md) · [简体中文](README.zh-CN.md)

[工具书](https://tianchen-ai.github.io/Intent/) · [安装](https://tianchen-ai.github.io/Intent/getting-started/installation/) · [示例](examples/README.md) · [MCP](mcp/README.zh-CN.md) · [贡献指南](CONTRIBUTING.zh-CN.md)

IntentDSL 是面向 GPU、CPU 和加速器的 Python kernel 语言与 MLIR 编译器。作者描述张量计算、逻辑域、访问、状态和显式多 kernel 编排；编译器通过 passes 构造物理程序，发射到 Triton、cuTile、Mojo、Weft 或 BANG C。

![IntentDSL 算法、编译器与后端工作流](images/overview.zh-CN.svg)

- **算法可编程**：在 kernel 中组合归约、收缩、索引访问、控制流与 region 计算。
- **优化可沉淀**：共享语义分析与执行模型各自的 passes 组织计算、复用和存储；下层编译器继续完成布局与指令 lowering。
- **普通 Python 调用**：编译为 callable，传入 tensor 或 native buffer，查看生成源码、IR 和实际选中配置。

## 开始使用

GPU 安装路线使用 **Linux、Python 3.10–3.12 和 NVIDIA GPU**。使用预编译 wheel 安装编译器无需 LLVM/MLIR SDK；其他后端和源构建要求见[安装说明](environment/README.zh-CN.md)。

### 用 pip 安装 wheel

从[公开预览发行](https://github.com/Tianchen-ai/Intent/releases/tag/preview)下载 wheel，也可自行构建。将 `INTENT_WHEEL` 设置为实际下载的 `.whl` 文件路径：

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install "$INTENT_WHEEL[manual,examples]"
intent setup --target triton
intent doctor --target triton
```

分发脚本面向 **Linux x86-64、glibc 2.35 及以上**，使用 `auditwheel` 将 wheel 修复为 `manylinux_2_35_x86_64`。Wheel 包含原生编译器、优化器、profiles、手册与编译器运行库，安装时无需 LLVM/MLIR SDK；后端 SDK 与驱动另行选择。包名为 `intentdsl`，Python 导入名为 `intent`；**目前尚未公开发布到 PyPI**。

### 从源码安装

```bash
git clone https://github.com/Tianchen-ai/Intent.git
cd Intent
python3 environment/install.py --backend triton --venv .venv-triton --examples
source .venv-triton/bin/activate
intent doctor --target triton
python -m examples.use softmax --target triton
```

源构建需要 CMake、Ninja、C++17 编译器与 LLVM/MLIR 20 C++ SDK。脚本构建并安装编译器和 Python 包，再安装所选后端，构建产物保存在 checkout 外。cuTile 使用独立环境：`--backend cutile --venv .venv-cutile`。[安装说明](environment/README.zh-CN.md) 提供 SDK、外部 CPU/MLU 工具链及使用 `environment/build.py` 构建分发包的步骤。

## 编写并调用 kernel

将下面代码保存为 Python 文件，在已安装的 Triton 环境运行。算法与现有 [ReLU 示例](examples/kernels/activation/pointwise.py) 相同。

```python
import torch
import intent
import intent.language as I


@intent.kernel
def relu(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        output[row, columns] = I.maximum(x[row, columns], I.cast(0.0, I.f16))


artifact = intent.compile(relu, target=intent.TritonTarget())
x = torch.randn((64, 1024), device="cuda", dtype=torch.float16)
y = artifact.run(x)
print(y.shape, y.dtype, y.device)
```

Kernel 描述逻辑行、列和运算；编译器选择物理 ownership、blocking、遍历与存储。Target 在 host 调用中选择，kernel 无需判断后端。

`intent.compile` 返回可调用产物。首次调用可能触发原生 JIT 编译和 autotuning，相同 specialization 的后续调用复用这些工作。`artifact.run(...)` 分配声明的 `Out`；`artifact(...)` 接受显式输出。应用中应保留 artifact 并重复调用。

GPU 与 Mojo 产物还提供 `artifact.as_torch_op("your_project::name")`，可接入 PyTorch custom operator 和 `torch.compile`。Backward 由作者注册，kernel 与 host 关系保持显式。见[框架调用示例](examples/README.md#框架与产物调用示范)。

## 浏览 30 个完整程序

产品示例覆盖 attention、递归状态、归一化、矩阵与量化收缩、稀疏访问、scan 和可变状态更新。每个 program 复用 `examples/kernels/` 中的唯一算法，以及 `examples/programs/` 中的普通 host 编排。

```bash
python -m examples.use --list
python -m examples.use attention --target triton
python -m examples.use paged_decode --target cutile
python -m examples.use --all --target triton --stage source --jobs 4
```

| Provider | 执行路线 |
|---|---|
| Triton / cuTile | NVIDIA GPU；接入 PyTorch tensor 与 stream |
| Mojo | x86 CPU；使用外部 Mojo 编译器 |
| Weft | CPU lowering；原生执行使用显式配置的 Weft 部署 |
| BANG C | 寒武纪 MLU；使用外部 NeuWare SDK 和兼容设备 |

源码生成、原生编译与真实执行是不同阶段。实际支持范围取决于程序的操作、类型、effects 与所选硬件。[示例指南](examples/README.md) 介绍输入、完整调用与 target 参数。

## 编译器如何组织

Pass 根据 typed semantics、def-use、坐标关系、effects 和 target 能力决策，执行决定保存在当前 MLIR 程序中。GPU、CPU 和 DSA 可以复用语义知识，同时保留各自的执行与存储模型。

![同一 attention region 算法在 GPU program ownership、CPU task 层次与 DSA 局部存储中的物理组织](images/execution-models.zh-CN.svg)

*依据论文中的 region 计算图重新绘制，颜色区分数据、源 region、状态与计算。图片来源见[说明](images/README.md)。*

使用语言查阅[工具书](https://tianchen-ai.github.io/Intent/)，理解编译边界查阅[编译器规格](doc/compiler/README.md)，开发入口查阅[贡献指南](CONTRIBUTING.zh-CN.md)。

## 配合 agent 使用

MCP 有独立的[目录与配置说明](mcp/README.zh-CN.md)。两个 stdio 服务随 `manual` dependency extra 安装：

| 命令 | 职责 |
|---|---|
| `intent-manual` | 只读语言合同、API 声明与最小语法片段 |
| `intent-compiler-mcp` | 编译提供的程序、查看环境、优化提供的 IR、读取产物 |

手册可从 `read(id="doc/dsl/authoring.md")` 与 `api(name="intent.compile")` 开始。需要 agent 编译提供的 Python 程序时，启用 compiler 服务。它与 CLI 使用同一编译器；导入程序时会执行其普通顶层 Python。

## 查看编译过程

```bash
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 \
  --stage kir --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 \
  --target triton --json
```

CLI 返回真实编译阶段与产物路径。Python 中可读取 `artifact.source`、`artifact.mlir` 和 `artifact.cache_directory`。源码生成与 KIR 检查不主动 launch 所选 kernel。已安装工具与诊断用法见[使用指南](https://tianchen-ai.github.io/Intent/getting-started/usage/)。

## 参与贡献

[贡献指南](CONTRIBUTING.zh-CN.md) 提供环境设置、模块导航、pass 贡献方式和必要验证。问题与可复现错误请提交到 [GitHub issues](https://github.com/Tianchen-ai/Intent/issues)。

IntentDSL 使用 [Apache-2.0](LICENSE) 许可。随包分发的第三方组件保留各自的 notices 和许可。
