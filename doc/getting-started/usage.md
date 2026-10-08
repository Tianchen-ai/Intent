# 编译与运行

## 选择现有完整程序

仓库的 30 个完整 program 复用 `examples/kernels/` 中唯一的算法定义。`examples/programs/` 负责输入、输出和显式多 kernel host 编排，编译器不读取 program 名称选择模板。

```bash
python examples/use.py --list
python examples/use.py layer_norm --target triton
python examples/use.py paged_decode --target cutile --prepared
python examples/use.py group_norm_backward --target mojo
```

`--stage source` 生成源码并绑定完整调用的 shape/dtype；`native` 编译已绑定调用但不执行；`run` 执行完整程序并读回输出。阶段成功分别说明生成、原生编译或执行；它们不自动证明数值正确或性能达标。

```bash
python examples/use.py --all --target triton --stage source --jobs 4 --json
python examples/use.py attention --target mojo --stage native
python examples/use.py paged_decode --target triton --measure 10 --json
```

`--jobs` 只并发 source/native 准备，设备执行按顺序进行。`--measure` 在首次编译、调优和执行后计时完整热调用；多 kernel 程序包括作者全部调用及必要 workspace 初始化。传入 InOut 的恢复与 host 读回在计时之外。GPU 使用 CUDA events，CPU/MLU 使用完整同步调用的 wall clock。输入准备、编译、调优与 `elapsed_seconds` 不应当作 kernel 时间。

## Python 调用接口

`intent.compile(kernel, target=..., constexprs=...)` 返回 callable artifact；`intent.generate(...)` 只生成 source/IR。Target 由 host 选择，例如 `intent.targets.TritonTarget()`，不是 kernel 内的 value 或分支条件。

- `artifact(...)` 按声明顺序传入全部 runtime 参数，包括 `Out`，执行并返回 `None`。
- `artifact.run(...)` 省略声明的 `Out`，由 runtime 分配并返回输出；`InOut` 仍由调用方传入。
- `artifact.prepare(..., outputs=(...))` 提前绑定输入、输出和资源，不执行 kernel。
- prepared call 的 `launch()` 执行，`result()` 取得输出容器，不执行或同步。

没有 `Out` 时返回 `None`，一个 `Out` 返回该对象，多个 `Out` 按声明顺序返回 tuple。多 kernel 程序分别编译 artifacts，再由普通 Python host 组织调用与中间 tensor lifetime。

现有[softmax 示例](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax.py)还演示 PyTorch tensor、prepared call、native observation 和 `torch.compile`；[forward/backward 示例](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax_forward_backward.py)演示作者显式注册梯度。Intent 不自动推导 backward 或添加隐藏 launch。

## 检查生成结构

```bash
intent describe --target triton --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --stage kir --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --stage shared --json
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --json
intent read-artifact /path/returned/by/compile --offset 0 --limit 16000 --json
```

`--materialize` 额外创建 callable，但不 launch。`--constexpr NAME=JSON` 绑定算法 constexpr；`--target-option NAME=JSON` 传入公开 target 构造器。已有 IR 可用 `intent optimize INPUT.mlir --pipeline 'builtin.module(canonicalize,cse)'` 运行标准 MLIR pipeline。

源码、IR 与缓存默认在 `$XDG_CACHE_HOME/intentdsl/` 或 `~/.cache/intentdsl/`，`INTENT_CACHE_DIR` 可覆盖。失败保留真实阶段、诊断和产物路径；后端未实现会明确报告，不改换算法。
