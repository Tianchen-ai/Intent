# 扩展 Pass

Pass 是优化知识的入口。它读取当前 typed semantics 与执行关系，改写真实 IR，再把完整程序交给后续 consumer。新增优化前，先阅读[Pass 与分析](../compiler/passes-and-analyses.md)、对应 [GPU](../compiler/gpu-program-ir.md)/[CPU](../compiler/cpu-program-ir.md) 程序合同和[目标 lowering](../compiler/target-lowering.md)。

## 找到负责层

| 改动 | 负责位置 |
|---|---|
| domain、dtype、effects 与作者算法的语义 | DSL / canonical KIR；设计变化需要单独讨论 |
| def-use、coordinate、alias、dependence、working-set 查询 | 对应 dialect 的 `Analysis/` |
| ownership、遍历、blocking、reuse、materialization | GPU/CPU/DSA 的 `Transforms/` 职责组 |
| 目标能力、特定实现、local forms 与 legality | `lib/Target/<Provider>/` |
| 源码 API 拼写 | 目标 serialization |
| ABI 参数绑定、工具链启动、调优调用 | `python/intent/runtime/<provider>/` |

GPU 与 CPU 可以复用分析思想，但不要求共享一套 physical topology。一个 pass 不必在所有后端运行；共享事实、独立决策、各自目标 consumer 是有效复用。

## 一个完整优化组件

1. 从当前 typed operands、regions、def-use、access relation、effects、lifetime 与 capability 定义合法性，不能以 kernel 名称、操作数量或字符串标签触发。
2. 用 analysis 提供可重算的事实。Unknown 保持 unknown，不能解释为 replay、alias 或初始化许可。
3. 实际改写 operations、types、regions、coordinates、carry 或 resource lifetime；不另造共同解释执行的 schema/plan。
4. 保持 logical members、dtype、数值许可、ordered control、effects、ABI 和作者 kernel 数量。
5. 声明失效的 analysis，在完整 transformation group 后检查后置条件。Verifier 不修程序，serializer 不补 loops、workspace 或参数。

Pass 声明、注册与 pipeline 使用 MLIR 的既有机制。先查看 `include/Intent/Dialect/<Family>/Transforms/Passes.td`、`lib/Dialect/<Family>/Transforms/` 和对应的 `Passes.cpp`；target-local transformations 从 `lib/Target/<Provider>/Transforms/` 进入。实际可调用 pass 及选项可通过 `intent-opt --help` 查阅。

例如 GPU 归约优化放在 `lib/Dialect/GPU/Transforms/Reduction/`，声明在 GPU `Passes.td`，完整 transformation 的入口和顺序在 GPU `Transforms/Passes.cpp`。CPU 的 task 分配和 storage 复用分别位于 CPU `Transforms/Task/` 与 `Transforms/Storage/`，CPU pipeline 由其 `Transforms/Passes.cpp` 组装；不把 GPU ownership 变换挪过去共享。Provider 的经验配置表位于相应 `Transforms/` 或 `Transforms/Configuration/`。

新增 pass 时在相邻 CMake 列表加入实现；从已有 transformation wrapper 复用注册、依赖 dialect、错误传播及完成边界验证。把它插入满足输入条件的位置，而不是只为某个 program 加一条旁路。

只改完整经验配置时不创建新 pass。Provider 已负责的 collective tree、线程通信、distributed layout 和 machine pipeline，直接提供合法的输入，不在 Intent 重建。

## 使用已有程序观察

选择受影响的现有完整 program，检查前后 IR 结构、生成源码和实际调用结果，再观察同算法同输入的热调用；不要复制算法或扩建测试矩阵。若另一同类 program 受益，可以说明知识的复用范围。生成、native 编译、执行、数值检查与性能改善分别陈述。

可对已经生成、满足该 pass 前置条件的 IR 使用标准 MLIR pipeline：

```bash
intent optimize /path/to/current.mlir --pipeline 'builtin.module(canonicalize,cse)' --json
```

将 pipeline 换成当前 `intent-opt --help` 所列、适用于该阶段的变换。共享 GPU pass 不能应用到 KIR 或已经加入 provider 原生形式的模块；完整编译中的 pass 输入可以通过 native compiler 的 MLIR IR 打印选项定位。源码生成后的优化观察仍需重新运行对应完整程序，单独 IR 优化不证明 runtime 结果。

仓库的[贡献指南](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.zh-CN.md)说明开发环境；[编译器开发入口](compiler-reference.md)定位公开 API、各层实现与阶段诊断。
