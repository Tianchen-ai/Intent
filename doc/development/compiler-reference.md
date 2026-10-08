# 编译器开发入口

本页按一次编译调用定位公开实现。语言与 IR 合同见[编译器规格](../compiler/README.md)；新增变换见[扩展 Pass](passes.md)。源码链接指向当前 `main`，具体 API、可用 provider 和选项可由已安装工具查询。

## 从定义到运行

```text
@intent.kernel / @intent.fn
  → typed Python frontend
  → canonical KIR
  → GPU / CPU / DSA physical IR
  → family passes
  → provider legalization
  → provider source + interface metadata
  → native materialization
  → explicit runtime call
```

`intent.generate` 返回可保存的 `GeneratedProgram`，不启动 kernel；`intent.compile` 组合生成与 runtime 绑定。首次实际调用可能才触发外部 JIT 或调优。作者在 host 中显式调用多个 kernel；这条编译路径保持各 kernel 的算法与接口。

| 阶段 | 当前实现入口 | 修改范围 |
|---|---|---|
| Kernel 与 helper 声明 | [python/intent/api/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/api/) | 装饰器、定义和作者接口 |
| Typed frontend | [python/intent/frontend/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/frontend/) | 类型化捕获、作用域、控制与 intrinsic 构造 |
| 公开编译 API | [compiler/pipeline.py](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/compiler/pipeline.py) | `compile`、`generate`、`compile_ir`、`generate_from_ir` 与阶段诊断 |
| Native driver | [lib/Compiler/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Compiler/) | 请求、目标选择与 family/provider pipeline；CLI 解析及文件读写在 [tools/intent-compile/](https://github.com/Tianchen-ai/Intent/tree/main/tools/intent-compile/) |
| KIR 规范化 | [lib/Transforms/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Transforms/) | Canonical KIR 的跨目标规范化 |
| 执行模型构造 | [lib/Conversion/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Conversion/) | KIR 到 GPU、CPU 或 DSA 物理程序 |
| Family 分析与变换 | [lib/Dialect/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Dialect/) | 各执行模型的 typed IR、分析、变换与 verifier |
| Provider 实现 | [lib/Target/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Target/) | 目标合法化、原生构造及序列化 |
| 产物保存与恢复 | [compiler/artifact.py](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/compiler/artifact.py) | Source、IR、metadata、`save/load/materialize` |
| 目标与真实调用 | [python/intent/targets/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/targets/)、[python/intent/runtime/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/runtime/) | 目标能力、环境绑定、ABI、JIT、调优与 launch |
| CLI 与 MCP | [python/intent/tools/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/tools/)、[python/intent/mcp/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/mcp/) | 同一公开编译 API 的命令行与 MCP 接入 |

## 复用组件的位置

前端在 `frontend/mlir/` 建立 typed values、operations、blocks 与 regions，最终一次序列化。Helper、控制合流和 structured intrinsic 复用 `frontend/lowering/` 的作用域、product、shape 与 region 构造；不同 intrinsic 族集中在 `frontend/lowering/intrinsics/`。新增规则应消费已有上下文，不维护第二套类型或 SSA 可见性。

语义分析位于 `include/Intent/Analysis/` 与 `lib/Analysis/`；family 的物理关系位于对应 dialect 的 `Analysis/`。GPU、CPU 和 DSA 有独立执行模型，复用适用的语义证明，物理变换仍在各自 `Transforms/`。目录中的 `Reduction/`、`Storage/`、`Task/` 等职责组用于放置同层、强相关的实现。

Provider 的 `Transforms/` 选择并合法化目标原生构造，`Serialization/` 拼写已经形成的程序。契合合同的 provider 原生归约、scan、布局和指令 lowering 由 provider 完成。公开头文件位于对应 `include/Intent/`，局部实现 helper 可留在相邻实现目录。

## 查询与定位问题

先在安装环境中查询当前声明和依赖：

```bash
intent describe --target triton --json
intent doctor --target triton --json
```

`describe` 查询接口；`doctor` 检查实际工具链与目标环境。它们不编译或执行 kernel。

以下命令使用现有完整算法，停在不同阶段：

```bash
python -m examples.use softmax --target triton --stage source
python -m examples.use softmax --target triton --stage native
python -m examples.use softmax --target triton
```

完整程序入口准备原有输入并调用原有 host 编排。KIR 阶段不需要 target 或设备；shared 阶段需要明确编译目标；native 与运行阶段需要对应工具链和设备。选择单个定义并查看 KIR 的命令见[编译与运行](../getting-started/usage.md)。

失败时从返回的 `stage`、诊断与 `cache_directory` 找到实际输入、IR、源码和日志。先确认错误属于 frontend、物理构造、family pass、provider 合法化、外部编译还是 runtime；按该边界修改，不在后续层重新猜测遗漏语义。

## 配置、数值与性能

经验配置表位于各 provider 的 `Transforms/` 或 `Transforms/Configuration/`；它们声明完整关联行，runtime 调优这些行。配置格式、覆盖和首次调用行为见[配置与调优](../getting-started/configuration.md)。

`CompileOptions` 与目标参数是不同接口：前者选择已定义的数值许可和可选优化，后者提供目标能力与编译事实。启用优化不自动授权额外数值变化，具体合同见[类型、数值与 Effects](../dsl/types-numerics-and-effects.md)。

性能贡献说明变换读取的事实、合法性、实际 IR 改写和预期结构收益。验证受影响的完整 program，并将生成成功、原生编译、执行、数值比较和热调用时间分别记录。JIT、调优与输入准备不混入 kernel 热调用结论。详见[贡献指南](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.zh-CN.md)。
