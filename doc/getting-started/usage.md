# 编译与运行

## 在执行机器上使用 JIT

安装 Intent 及所选 provider 后，在实际执行主机上运行：

```python
import intent
from intent.examples.kernels.normalization.softmax import stable_softmax_f16

softmax = intent.jit(stable_softmax_f16, backend="triton")
y = softmax.run(x)
y2 = softmax.run(x2)
```

`backend` 必填，可选 Triton、cuTile、Mojo、Weft 或 BANG C 对应的
`triton`、`cutile`、`mojo`、`weft`、`bangc`。对象创建不执行设备探测。
首次绑定根据实际输入设备解析本机硬件及工具链，生成并缓存编译结果；
首次执行完成必要的原生编译与已声明候选调优。后续调用重新绑定实际 buffer，
复用适用的编译结果及已选配置。后端失败时不会切换其他后端。

H100 与 5090D 使用同样的调用代码，设备能力由当前执行主机探测。
CPU 根据当前进程允许的 CPU 集合查询有效目标；`workers` 是线程预算。
Weft 需要在 RISC-V 执行主机安装原生 Weft 与 C 工具链。BANG C 查询本机
CNRT 资源，并从实际 buffer 取得 lowering 必需的 shape/stride。
用户不为普通调用填写 ISA、VLEN 或片上存储容量。

输入必须属于同一个兼容设备；无设备输入时用 `device="cuda:0"` 等显式选择。
显式 device 与输入冲突会报错，不隐式搬运。模型可传入已有的 PyTorch tensor，
或使用相应 provider 的公开 buffer。工具链路径放在 `backend_options`；
`compiler` 指定 Intent 编译器。

## 准备与执行

- `kernel.run(*inputs)` 分配声明的 Out，执行并返回输出。InOut 仍由调用方提供。
- `kernel(*all_arguments)` 按作者声明顺序包含全部 Out，执行并返回 None。
- `kernel.prepare(*inputs, outputs=(...))` 绑定输入、显式输出与 workspace；
  可以生成必要 IR/source，但不执行 kernel、不调优。省略 outputs 时分配 Out。
- `call.compile()` 编译适用原生候选，不执行作者调用。
- `call.launch()` 执行已绑定调用；`call.result()` 取得输出容器，不执行或同步。
- `kernel.close()` 释放该对象持有的原生产物；调用方保持参数存储存活到执行完成。

零个 Out 返回 None，一个返回该对象，多个按声明顺序返回 tuple。
GPU 保持当前 stream 的异步提交语义；读取 CPU 结果时由调用者按框架语义同步。
图捕获前先完成编译、调优与必要首次执行。作者通过普通 Python 控制多 kernel
次序、中间存储及梯度定义，Intent 不自动改写模型图。

## 安装包中的完整程序

唯一的作者算法和 host 编排位于 `intent.examples.kernels` 与
`intent.examples.programs`，无需源码 checkout 或 Bench：

```bash
python -m intent.examples --list
python -m intent.examples layer_norm --backend triton
python -m intent.examples paged_decode --backend cutile --prepared
python -m intent.examples group_norm_backward --backend mojo
python -m intent.examples attention --backend mojo --stage native
```

`source` 生成并检查调用几何，`native` 编译而不执行，
`run` 执行完整程序。数值对照、重复计时、批量 suite 和实验协议由独立 Bench 提供：

```bash
intent-bench check --suite mainline --backend triton --env local-gpu
intent-bench run softmax --env local-gpu --case 64x1024-float16 \
  --impl intent:triton,intent:cutile,triton --baseline triton
```

[Softmax 示例](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax.py)
演示 prepared call 与 PyTorch 接入；
[forward/backward 示例](https://github.com/Tianchen-ai/Intent/blob/main/examples/softmax_forward_backward.py)
演示作者显式注册梯度。

## 高级 IR 与导出接口

`intent.generate(..., target=...)`、`intent.compile(..., target=...)` 和
artifact 导出继续用于显式编译与应用集成。离线编译必须提供完整目标事实；
它不代表本机执行成功。

```bash
intent doctor --target triton --json
intent describe --target triton --json
intent compile intent.examples.kernels.normalization.softmax:stable_softmax_f16 --stage kir --json
intent compile intent.examples.kernels.normalization.softmax:stable_softmax_f16 --target triton --stage shared --json
intent optimize INPUT.mlir --pipeline 'builtin.module(canonicalize,cse)'
```

独立产品缓存使用 `INTENT_CACHE_DIR`，默认在 XDG 用户缓存目录。
Bench 启动产品时才重定向到它的 output 根目录。生成、编译、执行和数值通过
分别报告；编译或调优耗时不属于算子执行时间。
