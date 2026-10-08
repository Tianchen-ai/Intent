# Intent kernel 示例

[English](README.en.md) | [简体中文](README.md)

`examples/kernels/` 只保存作者编写的目标无关算法，按 activation、contraction、normalization、streaming 等算法职责分类。编译器保持这些程序的类型、shape、数值和 effects；target 在编译调用中选择。

安装说明见 [中文 README](../README.zh-CN.md)。普通 host 示例直接复用 `kernels/` 中的定义：

首批 30 个完整场景由 [use.py](use.py) 导航，host 输入与调用放在
[programs/](programs/)，算法仍唯一保存在 `kernels/`。选择名称只决定使用示范，
编译器接收原 kernel definition 和 target，不读取场景名称。

```bash
python examples/use.py --list
python examples/use.py layer_norm --target triton
python examples/use.py paged_decode --target cutile --prepared
python examples/use.py group_norm_backward --target mojo
python examples/use.py --all --target triton --stage source --jobs 4 --json
python examples/use.py attention --target mojo --stage native
python examples/use.py paged_decode --target triton --measure 10 --json
python examples/use.py --all --target weft --target-option vector_bits=256 --target-option workers=4 --stage source --jobs 4
python examples/use.py --all --target bangc --stage source --jobs 4
```

`--list` 不导入 Intent、PyTorch 或 provider。运行时可以用 `--compiler` 选择当前
构建的编译器；GPU 可用 `--device` 选择设备。`--stage source` 生成后端源码并按
公开接口绑定完整调用的 shape/dtype；`native` 编译已绑定的原生调用，不执行或
读取其输出；`run` 执行完整程序并读回结果。`--all` 使用这里既定的 30 项，
`--jobs` 只并发准备 source/native，真实执行按顺序进行。批量中任一项失败，命令
返回非零，并保留实际失败阶段、诊断及产物路径；不会更换算法、dtype 或 target。
`--jsonl` 逐项输出完成记录；长批次的普通/JSON 模式也在 stderr 显示完成状态，
无需等最后一个原生编译结束。运行记录包含实际选中配置和 SDK 提供的资源观察，
不会把候选可编译或资源估算当作已经执行。

输入准备只需要 NumPy；BF16/FP8 使用 `ml_dtypes` 的实际存储格式。安装时选择
`environment/install.py --examples`，或安装本项目的 `examples` optional dependency。
Triton、cuTile、Mojo 在调用边界接到实际 Torch tensor，Weft/BANG C 接到已有的
native buffer。没有 Torch 的 RISC-V host 也可以准备同样的输入。BANG C 的实际
shape/stride 从这些参数绑定，不需要在 kernel 内改写设备信息。

`--seed` 固定输入流，默认 0；同一场景在各 target 使用相同输入字节、dtype 和
算法。`--measure N` 在首次编译/调优与执行完成后，计时 N 次完整 prepared call，
输出中位数和原始样本。多 kernel 场景包含作者全部调用及必要 workspace 初始化；
传入 InOut 的恢复和 host 读回位于计时之外。GPU 使用 CUDA events，CPU/MLU 使用
完整同步调用的 wall clock。`elapsed_seconds` 则包含输入准备、生成、编译和运行，
不能当作 kernel 时间。执行状态与计时不代表数值验证或性能达标。

`--target-option NAME=JSON` 直接传入公开 target 构造器。Weft source 要显式提供
`vector_bits`、`workers`；原生 materialization 另外提供 `profile`、Weft `compiler`
与非空 `cc` 命令列表。`profile` 沿用
[TargetProfile](../python/intent/runtime/weft/target.py) 的 march、ABI、物理 VLEN、
execution CPUs 与 stack budget，runtime 在加载时检查实际机器。已有的
`export_artifact`/`compile_artifact` 仍用于跨主机 AOT；target 不隐式 SSH 或部署。
BANG C 使用 `device`、`neuware` 和 `compiler` 选择现成 SDK 与设备。

场景列表是使用导航，包含已实现及尚待闭合的合法语言组合，并非 30×5 的已验证
支持承诺。真实 compiler 缺口会显示在对应阶段；目标能力限制与尚未执行分别说明。

| 场景名称 | Host 用法 | 输入及完整结果 |
|---|---|---|
| `relu` | [elementwise](programs/elementwise.py) | f16 matrix → 分配 `Out` |
| `swiglu` | [elementwise](programs/elementwise.py) | 两份 bf16 matrix → bf16 输出 |
| `rope` | [elementwise](programs/elementwise.py) | f16 Q/K `InOut`，固定 head 数与 f16 cosine/sine |
| `transpose` | [elementwise](programs/elementwise.py) | f16 M×N → N×M |
| `embedding` | [irregular](programs/irregular.py) | bf16 表与合法 i64 行索引 → lookup |
| `csr_spmv` | [irregular](programs/irregular.py) | 完整 CSR offsets、indices、f32 values/vector |
| `jagged_mean` | [irregular](programs/irregular.py) | 非空不等长 offsets、f32 values → 分段均值 |
| `softmax` | [collectives](programs/collectives.py) | f16 matrix → f16 概率 |
| `layer_norm` | [collectives](programs/collectives.py) | f32 x/weight/bias、逆维度和 epsilon |
| `batch_norm` | [collectives](programs/collectives.py) | Welford；返回输出、保存统计及更新后的 running state |
| `group_norm_backward` | [collectives](programs/collectives.py) | 两份 kernel；返回 dx/dweight/dbias |
| `cumsum` | [collectives](programs/collectives.py) | f32 有序行 prefix |
| `causal_linear_attention` | [collectives](programs/collectives.py) | f32 Q/K/V → 输出与最终矩阵状态 |
| `nonzero` | [irregular](programs/irregular.py) | f32 values → 索引与每行有效数量 |
| `moe_alignment` | [irregular](programs/irregular.py) | 四份 kernel；route IDs、expert blocks、有效 padded 数量 |
| `gemm` | [matrix](programs/matrix.py) | f16 A/B，显式 `Activation.NONE` specialization |
| `batched_gemm` | [matrix](programs/matrix.py) | bf16 batched NN contraction |
| `ragged_gemm` | [matrix](programs/matrix.py) | 不等长 group offsets 与 f16 分组权重 |
| `int8_gemm` | [matrix](programs/matrix.py) | i8 A/B、i32 bias → i32 输出 |
| `q4_projection` | [matrix](programs/matrix.py) | N×G×144 字节 Q4_K records、G×256 f32 activation |
| `fp8_gemm` | [matrix](programs/matrix.py) | E4M3 fragments、u8 E8M0 scales → f32 accumulator |
| `attention` | [streaming](programs/streaming.py) | bf16 grouped Q/K/V，causal specialization 与 scale |
| `paged_decode` | [streaming](programs/streaming.py) | page/split tables；显式 partials → f32-to-f16 merge |
| `mamba_chunk_state` | [streaming](programs/streaming.py) | bf16 basis/x、f32 dt/decay → f32 chunk state |
| `gated_delta` | [streaming](programs/streaming.py) | bf16 输入、有序 recurrence → 输出与 f32 final state |
| `causal_convolution` | [structured](programs/structured.py) | f16 depthwise causal window 与 SiLU specialization |
| `cholesky` | [structured](programs/structured.py) | 正定 f32 matrix `InOut` → lower factor |
| `dropout` | [elementwise](programs/elementwise.py) | 作者整数混合、seed、drop 与 inverse-keep scalars |
| `adamw` | [elementwise](programs/elementwise.py) | f32 gradient 与三个 mutable arrays、完整更新 scalars |
| `histogram` | [collectives](programs/collectives.py) | f32 samples，包括 masked 越界值 → 256 i32 bins |

动态 shape 的用法使用适合交互的输入大小；固定 shape 的算法沿用定义中的常量。
这些输入用于产品调用与必要的同规模自比较，不代替大型模型工作量。
Q4_K 示例的全零 record 表示零权重，实际模型使用自身已有的 packed
records；脚本不替模型转换权重格式。

### 多 kernel 的唯一 host 编排

[composition.py](programs/composition.py) 中的 `GroupNormBackward`、`PagedDecode`、
`MoEAlignment` 组合调用者已编译的 artifacts。输入准备与完整运行入口均在本目录，
这些组合类保留作者显式声明的 kernel 次序和工作区。
`run` 分配输出，`into`/`run_into` 使用调用者的输出和工作区，`prepare` 提前绑定实际输出与跨 kernel
workspace。编译器仍逐个编译原定义，host 负责 kernel 次序和中间 tensor。

例如，把同一 target 编译的 paged partials 与 merge 组成普通调用：
在 `examples/` 下的脚本中使用这些 imports，或运行时设置 `PYTHONPATH=examples`。

```python
import intent

from programs.composition import PagedDecode
from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce

partials = intent.compile(paged_gqa_decode_partials, target=target,
                          constexprs={"PAGE_SIZE": page_size,
                                      "HEAD_GROUP": query_heads // kv_heads,
                                      "SPLITS": splits})
merge = intent.compile(splitk_attention_f32_to_f16_reduce, target=target)
program = PagedDecode(partials, merge)
call = program.prepare(q, key_cache, value_cache, page_offsets, page_indices,
                       lengths, split_offsets, scale)
call.launch()
output = call.result()
```

`prepare` 只分配/绑定输出 handles，不读取中间结果、不执行 partials。真实 launch
始终先 partials 再 merge。MoE 的 prepared call 持有 counts、cursors 与 padding
workspace；每次 `launch()` 都先初始化这些 workspace，再运行四份 kernel。输出只有
`total_padded` 指定的 route 前缀和对应 block 前缀有效，padding sentinel 沿用作者定义。
该 host 组合使用作者固定的 top-k 2 和最多 4096 tokens 的 workspace；expert IDs
必须位于 `[0, 64)`。其它 top-k 或超容量输入明确拒绝，改变规模需要作者调整相应
算法常量和 host workspace。

### 框架与产物调用示范

| 入口 | 展示的完整调用 |
|---|---|
| `python examples/softmax.py --target triton` | 编译一次、传入 PyTorch tensor、分配声明的输出、定位编译产物 |
| `python examples/softmax.py --target cutile` | 在独立 cuTile 环境中复用同一算法定义 |
| `python examples/softmax.py --target triton --prepared` | 显式提供输出，准备一次调用，再分别执行 `launch()` 与 `result()` |
| `python examples/softmax.py --target triton --inspect-native` | 执行后查看当前配置、候选状态与 SDK 实际提供的原生资源 |
| `python examples/softmax.py --target triton --torch-compile` | 先普通调用同一算子完成 JIT/调优，再通过 opaque custom op 进入 `torch.compile(fullgraph=True)` |
| `python examples/softmax_forward_backward.py --target triton` | 作者注册已有 backward，保存 forward 输出，用 `Tensor.backward(upstream)` 取得输入梯度 |
| `python examples/softmax_forward_backward.py --target triton --torch-compile` | 同一 forward/backward 注册进入 PyTorch 图编译；两份 kernel 在捕获前完成首次 JIT/调优 |
| `python examples/softmax_forward_backward.py --target mojo --torch-compile` | 同一作者 forward/backward 在 Mojo CPU runtime 上接入 PyTorch 图编译与 autograd |

`artifact.run(...)` 省略声明的 `Out`，返回新分配的输出；显式调用 `artifact(...)` 保留全部 runtime 参数的声明顺序。`artifact.prepare(..., outputs=(...))` 接受相同输入和指定的 `Out`，准备过程不执行 kernel。返回的调用对象拥有这次参数绑定：`launch()` 执行，`result()` 取得输出容器而不执行或同步。零输出返回 `None`，单输出返回该值，多输出按声明顺序返回 tuple；`InOut` 始终由调用方传入。

`--inspect-native` 读取 `artifact.observation`，也可与 `--prepared` 一起使用。资源带有来源、阶段和单位；SDK 未提供的值明确显示原因。快照不是 correctness 或 occupancy 结论，也不会为了观察额外执行 kernel。首次真实调用前该属性为 `None`。

Forward/backward 示例复用既有的 `4096 × 4097`、f32 backward 定义。作者分别编译两个 kernels，把它们注册为 PyTorch custom ops，再通过 `register_autograd` 提供梯度公式：`setup_context` 保存 probabilities，backward 回调调用已注册的 backward op。普通 Python wrapper 决定这两个 kernels 的关系，compiler 不推导梯度或隐藏增加调用。这里只演示一阶梯度；更高阶梯度需要作者另行提供相应公式。

PyTorch adapter 支持 GPU 和 Mojo CPU runtime 的 `In`/scalar 输入、新分配的 `Out`，以及声明为 mutation 的 `InOut`。`InOut` 原位更新，不额外返回；没有 `Out` 时返回 `None`。每个 `InOut` 不得与其他输入共享 Torch storage，返回 alias 仍不支持。Runtime 通过 `infer_outputs` 提供 fake 输出，复用普通调用的公共 shape、dtype 和 stride 合同，不调用 provider、不读取 tensor 数据。`register_autograd` 仅用于 functional custom op，PyTorch 不接受 mutable custom op 的该注册；Weft 和 BANG C 的原生 buffer 接口不冒充 Torch tensor 接口。

两个脚本都接受 `--target triton`、`--target cutile` 和 `--target mojo`，输入 tensor 跟随 artifact 的实际 CPU/CUDA 设备。Mojo 使用公开 `MojoTarget()` 的默认配置；SDK 可通过 `INTENT_MOJO` 指定。换 target 不改变示例算法、shape 或 dtype。示例输出不是 benchmark 或数值验证结论。每个脚本把 host 执行放在 `main` 下，因此也能作为模块加载。

只编译而不运行可用：

```bash
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --json
```

JSON 包含原定义位置、真实编译阶段、source/IR/metadata 和日志路径。编译失败时从 `diagnostic` 与 `files` 继续查看；`intent doctor --target triton` 检查依赖和目标解析。完整算法保留在普通示例中，manual MCP 只提供通用语言规则。

编译、真实执行及 pass 优化观察使用这里的完整程序入口。生成源码、IR、缓存及
运行观察保存在仓库外；所选后端的安装条件见 [安装指南](../environment/README.zh-CN.md)。

新增算法放在 `kernels/`，完整 host 调用复用 `programs/` 的相应入口和公开 artifact
API。保留一份算法、一份作者编排；不要复制 reference 或新增并行执行机制。
