# IntentDSL

**写一次算子算法，编译到不同执行模型。**

[English](https://github.com/Tianchen-ai/Intent/blob/main/README.md) · [简体中文](https://github.com/Tianchen-ai/Intent/blob/main/README.zh-CN.md)

[工具书](https://tianchen-ai.github.io/Intent/) · [安装](https://tianchen-ai.github.io/Intent/getting-started/installation/) · [示例](https://github.com/Tianchen-ai/Intent/blob/main/examples/README.md) · [MCP](https://github.com/Tianchen-ai/Intent/blob/main/mcp/README.zh-CN.md) · [贡献指南](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.zh-CN.md)

IntentDSL 是面向 GPU、CPU 和加速器的 Python kernel 语言与 MLIR 编译器。作者描述张量计算、逻辑域、访问、状态和显式多 kernel 编排；编译器通过 passes 构造物理程序，发射到 Triton、cuTile、Mojo、Weft 或 BANG C。

![IntentDSL 论文原图：算法复用与编译器工作流](https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-fig1-overview.png)

*论文原总览图（[PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-fig1-overview.pdf)）。图中标签描述论文实现路径；当前产品的 provider 入口见下文。*

- **算法可编程**：在 kernel 中组合归约、收缩、索引访问、控制流与 region 计算。
- **优化可沉淀**：共享语义分析与执行模型各自的 passes 组织计算、复用和存储；下层编译器继续完成布局与指令 lowering。
- **普通 Python 调用**：编译为 callable，传入 tensor 或 native buffer，查看生成源码、IR 和实际选中配置。

## 开始使用

GPU 安装路线使用 **Linux、Python 3.10–3.12 和 NVIDIA GPU**。使用预编译 wheel 安装编译器无需 LLVM/MLIR SDK；其他后端和源构建要求见[安装说明](https://github.com/Tianchen-ai/Intent/blob/main/environment/README.zh-CN.md)。

### 用 pip 安装

从 [PyPI](https://pypi.org/project/intentdsl/) 安装编译器。可选 extras 同时提供 MCP 服务和示例输入依赖：

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install 'intentdsl[manual,examples]'
intent setup --target triton
intent doctor --target triton
```

Wheel 面向 **Linux x86-64、glibc 2.35 及以上**，包含原生编译器、优化器、profiles、手册与编译器运行库，安装时无需 LLVM/MLIR SDK；后端 SDK 与驱动另行选择。基础编译器直接使用 `pip install intentdsl` 安装，Python 导入名为 `intent`。[GitHub 预览发行](https://github.com/Tianchen-ai/Intent/releases/tag/preview)也提供可下载的 wheel。

### 从源码安装

```bash
git clone https://github.com/Tianchen-ai/Intent.git
cd Intent
python3 environment/install.py --backend triton --venv .venv-triton --examples
source .venv-triton/bin/activate
intent doctor --target triton
python -m examples.use softmax --target triton
```

源构建需要 CMake、Ninja、C++17 编译器与 LLVM/MLIR 20 C++ SDK。脚本构建并安装编译器和 Python 包，再安装所选后端，构建产物保存在 checkout 外。cuTile 使用独立环境：`--backend cutile --venv .venv-cutile`。[安装说明](https://github.com/Tianchen-ai/Intent/blob/main/environment/README.zh-CN.md) 提供 SDK、外部 CPU/MLU 工具链及使用 `environment/build.py` 构建分发包的步骤。

## 编写并调用 kernel

将下面代码保存为 Python 文件，在已安装的 Triton 环境运行。算法与现有 [ReLU 示例](https://github.com/Tianchen-ai/Intent/blob/main/examples/kernels/activation/pointwise.py) 相同。

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

GPU 与 Mojo 产物还提供 `artifact.as_torch_op("your_project::name")`，可接入 PyTorch custom operator 和 `torch.compile`。Backward 由作者注册，kernel 与 host 关系保持显式。见[框架调用示例](https://github.com/Tianchen-ai/Intent/blob/main/examples/README.md#框架与产物调用示范)。

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

源码生成、原生编译与真实执行是不同阶段。实际支持范围取决于程序的操作、类型、effects 与所选硬件。[示例指南](https://github.com/Tianchen-ai/Intent/blob/main/examples/README.md) 介绍输入、完整调用与 target 参数。

## 编译器如何组织

Pass 根据 typed semantics、def-use、坐标关系、effects 和 target 能力决策，执行决定保存在当前 MLIR 程序中。GPU、CPU 和 DSA 可以复用语义知识，同时保留各自的执行与存储模型。

![论文原图：同一逻辑计算形成 GPU program、CPU task 层次和 DSA 局部存储程序](https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-fig5-execution-models.png)

*论文原执行构造图（[PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-fig5-execution-models.pdf)）：各执行模型分配 query ownership，并在源 region 之间携带状态。CPU 局部块由实现要求塑形；DSA 的依赖关系决定输入缓冲复用。*

使用语言查阅[工具书](https://tianchen-ai.github.io/Intent/)，理解编译边界查阅[编译器规格](https://github.com/Tianchen-ai/Intent/blob/main/doc/compiler/README.md)，开发入口查阅[贡献指南](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.zh-CN.md)。

## 跨执行模型的整体性能

以下论文原图比较完整算子与相应目标上的手写实现。加速比为 **reference 耗时 / generated 耗时**，越高越好。计时使用完整算子耗时的中位数，包含准备和必要的布局转换，排除编译、JIT 与调优。这些是论文历史成绩，并非当前安装包或 30 个产品 program 的新测量。

![论文GPU整体性能：RTX 5090D和H100上的generated Triton/cuTile与Triton reference对比](https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-eval-gpu-triton-reference.png)

*GPU（[PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-eval-gpu-triton-reference.pdf)）：八个代表性 workload，以及全部 44 个共同 workload 的几何平均。每个 workload 的 generated Triton 与 cuTile 使用同一个作者 DSL 算法、输入规模和外部 dtype；原图标注代表性 shape 与精度，灰色柱是论文中的 Triton reference。*

| x86 CPU · 7 个原生实现对比 | MLU370 · 6 个库实现对比 |
|---|---|
| <img src="https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-eval-cpu-native.png" alt="论文x86 Mojo整体性能：与5个MAX和2个独立Mojo实现对比" width="360"> | <img src="https://raw.githubusercontent.com/Tianchen-ai/Intent/main/images/paper-eval-mlu-libraries.png" alt="论文MLU BANG C整体性能：与CNNL、CNNL Extra及TMO实现对比" width="360"> |
| [原 PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-eval-cpu-native.pdf) | [原 PDF](https://github.com/Tianchen-ai/Intent/blob/main/images/paper-eval-mlu-libraries.pdf) |

*CPU：generated Mojo 程序与 5 个 Modular/MAX、2 个独立 Mojo reference 对比，使用同一 NUMA 节点内的 8 个物理 x86 核。MLU：generated BANG C 程序与 CNNL/CNNL Extra/TMO 对比，使用 MLU370 与 CNCC/CNRT。各组采用论文对应 workload 的 shape、精度、输出与容差；两张小图是逐 workload 比值，不是几何平均。[图片来源](https://github.com/Tianchen-ai/Intent/blob/main/images/README.md) 记录测量范围。*

## 配合 agent 使用

MCP 有独立的[目录与配置说明](https://github.com/Tianchen-ai/Intent/blob/main/mcp/README.zh-CN.md)。两个 stdio 服务随 `manual` dependency extra 安装：

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

[贡献指南](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.zh-CN.md) 提供环境设置、模块导航、pass 贡献方式和必要验证。问题与可复现错误请提交到 [GitHub issues](https://github.com/Tianchen-ai/Intent/issues)。

IntentDSL 使用 [Apache-2.0](https://github.com/Tianchen-ai/Intent/blob/main/LICENSE) 许可。随包分发的第三方组件保留各自的 notices 和许可。
