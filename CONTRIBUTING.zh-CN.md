# 参与 IntentDSL 开发

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

贡献可以改善语言易用性、可复用 compiler passes、provider lowering、运行时集成与文档。先从[工具书](https://tianchen-ai.github.io/Intent/)和对应模块开始。[编译器开发入口](doc/development/compiler-reference.md) 定位公开实现与可复用组件。

## 反馈问题

先搜索[已有 issues](https://github.com/Tianchen-ai/Intent/issues)。提交新问题时，请提供：

- 最小源程序或受影响的既有 example、输入 shape/dtype 与实际命令。
- 预期行为和实际诊断，包括失败阶段及相关产物路径。
- 所选后端、设备、驱动或 SDK/compiler、Python 环境，以及失败发生在源码生成、原生编译还是执行阶段。
- 性能问题还需说明完整调用的计时范围、输入规模、配置，以及 JIT/调优是否已结束。

`intent describe --target BACKEND --json` 查看公开接口；`intent doctor --target BACKEND --json` 检查所选环境。编译诊断与 `artifact.cache_directory` 指向实际源码、IR 和日志。

## 配置开发环境

Fork 并克隆[仓库](https://github.com/Tianchen-ai/Intent)，按照[安装说明](environment/README.md) 准备环境。源构建使用 LLVM/MLIR 20 SDK。

```bash
python3 environment/install.py --backend triton --venv .venv-triton --examples
source .venv-triton/bin/activate
python -m examples.use --list
```

反复修改 C++ 时，可配置独立构建目录：

```bash
cmake -S . -B "$HOME/.cache/intentdsl/development" -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/.cache/intentdsl/development" \
  --target intent-compile intent-opt --parallel 4
```

通过 `INTENT_COMPILER` 指向该构建的 `tools/intent-compile/intent-compile`，让公开 Python 工具使用当前 C++ 编译器。修改 Python 后重新安装 checkout，并检查 `intent.__file__` 确认实际运行的安装。其他 provider 使用其声明的外部工具链。

## 找到应修改的模块

```text
Python 定义 → typed frontend → canonical KIR
                               ├─ GPU IR 与 passes → Triton / cuTile
                               ├─ CPU IR 与 passes → Mojo / Weft
                               └─ DSA IR 与 passes → BANG C
                             → provider source → native runtime
```

| 贡献类型 | 起点 |
|---|---|
| 作者算法与完整调用 | [examples/kernels/](examples/kernels/) 与 [examples/programs/](examples/programs/) |
| 语言与前端 | [python/intent/language/](python/intent/language/)、[python/intent/frontend/](python/intent/frontend/)、[作者合同](doc/dsl/authoring.md) |
| Canonical 语义分析 | [include/Intent/Analysis/](include/Intent/Analysis/) 与 [lib/Analysis/](lib/Analysis/) |
| KIR 到执行模型的构造 | [lib/Conversion/](lib/Conversion/) |
| GPU、CPU 或 DSA passes | [lib/Dialect/](lib/Dialect/) 与对应的 [include/Intent/Dialect/](include/Intent/Dialect/) |
| Provider realization 与序列化 | [lib/Target/](lib/Target/) |
| 公开调用与原生集成 | [python/intent/compiler/](python/intent/compiler/)、[python/intent/targets/](python/intent/targets/)、[python/intent/runtime/](python/intent/runtime/) |
| CLI 与 MCP | [python/intent/tools/](python/intent/tools/)、[python/intent/mcp/](python/intent/mcp/)、[MCP 配置](mcp/README.zh-CN.md) |
| 安装与分发 | [environment/](environment/) 与 [cmake/](cmake/) |
| 双语文档 | [doc/](doc/)；[文档工作流](https://tianchen-ai.github.io/Intent/development/documentation/) |

`doc/` 定义稳定的语言与 compiler 合同，改变语义边界前完整阅读相关章节。GPU 与 CPU 的 physical program 属于不同执行模型；共享适用的语义分析，物理变换保留在各自 family。

## 贡献可复用的 pass

Pass 应表达对当前 typed IR 的可复用变换。先读 [pass 与 analysis 合同](doc/compiler/passes-and-analyses.md) 和 [pass 开发指南](https://tianchen-ai.github.io/Intent/development/passes/)。

1. 说明读取的 facts、合法性条件、实际 IR 改写，以及保持的值、control、effects、数值与 ABI。
2. 复用现有 analysis 与 IR carrier。共享语义查询放在分析层；GPU、CPU 或 DSA 的执行决定放在相应 family。
3. 在相邻 `Passes.td` 声明完整 transformation 与 options，实现进入对应职责子目录，公共头文件放在 `include/Intent/`。
4. 说明 analysis 失效与重算，在完整变换边界验证。Serializer 拼写已经决定的程序，不选择算法或重建执行结构。
5. 对照成熟目标的对应机制。合同一致时，provider 原生 collective、layout 与指令 lowering 交给 provider。
6. 使用受影响的完整产品 program 验证，并解释规则为何能复用于其他写法或算法。

Policy 根据当前语义、def-use、坐标、effects、lifetime 和能力决策，不按 kernel 名称选择模板。作者算法、状态顺序和显式多 kernel 调用是权威；合法程序 lowering 失败属于 compiler 缺口。

## 验证并提交改动

使用既定 30 个完整 program 中受影响的入口：

```bash
python -m examples.use softmax --target triton --stage source
python -m examples.use softmax --target triton --stage native
python -m examples.use softmax --target triton
git diff --check
```

这些命令展示不同阶段；实际选择本次改动影响的既有 program 和 target，保持算法、输入与数值合同。必要的数值比较沿用既定容差。原生执行需要真实工具链与设备；源码生成成功不等于运行正确或性能达标。

生成 IR/源码、构建缓存与性能观察保存在仓库外。复用 `examples/kernels/` 的唯一算法与 `examples/programs/` 的 host 编排，不为另一条测试路径复制它们。

分发修改使用现有 `environment/build.py`；文档修改按[文档指南](https://tianchen-ai.github.io/Intent/development/documentation/) 构建双语站点。

一个 PR 应形成连贯改动。描述具体问题、修改后的行为、涉及的编译边界与实际验证。优化改动区分编译/调优成本和完整热调用时间，解释收益对应的物理结构。遵循相邻代码风格，保留第三方 notices。

IntentDSL 使用 [Apache-2.0](LICENSE)。公开实现 API 与复用位置见[编译器开发指南](doc/development/compiler-reference.md)。
