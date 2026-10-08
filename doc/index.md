# IntentDSL 工具书

IntentDSL 是用于编写结构化 kernel 算法的 Python DSL 和 MLIR 编译器。作者表达逻辑成员、计算、控制、状态与显式多 kernel 编排；编译器为所选执行模型建立物理程序，再交给 Triton、cuTile、Mojo、Weft 或 BANG C 等目标工具链。

## 从这里开始

- [安装](getting-started/installation.md)：从源码或现有 wheel 安装，选择一个后端环境。
- [编译与运行](getting-started/usage.md)：运行现有完整程序，查看 KIR、物理 IR 与生成源码。
- [配置与调优](getting-started/configuration.md)：完整经验配置、首次调优与热调用的边界。
- [Agent 与 MCP](getting-started/mcp.md)：连接只读语言手册和显式编译工具。

## 查阅语言与编译器

写 kernel 时先看[作者速查](dsl/authoring.md)，需要确认合同再查[语言构造](dsl/core.md)和[类型、数值与 Effects](dsl/types-numerics-and-effects.md)。开发优化时从[编译器概述](compiler/README.md)、[Pass 与分析](compiler/passes-and-analyses.md)以及[扩展 Pass](development/passes.md)进入。

本目录保存使用说明与稳定设计，不保存实现进度、测试结果、性能数字或历史决策过程。规格分为三层：

- [编程模型](programming-model/README.md)：定义作者、kernel、host 与编译器之间的语义边界；
- [语言参考](dsl/README.md)：定义作者可使用的语言表面、canonical semantics、surface shorthand 及其理想化示例；
- [编译器](compiler/README.md)：定义 canonical KIR 之后的 executable physical programs、passes、target extensions 与下层编译器边界。

## 语义权威

target 由编译调用在 DSL 之外选择，不是 source value、constexpr、类型或控制条件。同一份 source 与 canonical Kernel IR 可以被编译到不同 target；target capability 只决定 lowering 是否成立，不改变这里定义的逻辑结果、控制、effects 和接口语义。

编译器 IR、pass、target lowering 和运行时语义建立在编程模型与 DSL 之上。现有实现、历史报告和任一 target API 都不能反向修改这里的语言语义。

页面顶部可以切换中文与 English。中文规格保留原 Markdown 路径，英文页使用同名 `.en.md`；[文档构建与发布](development/documentation.md)说明本地预览与 GitHub Pages 部署。
