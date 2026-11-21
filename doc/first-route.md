# 构建并运行第一条 GPU 路线

构建机需要 CompilerInfra 0.3.0 SDK、与该 SDK 匹配的 MLIR 和 C++ 工具链。执行机使用同一 ABI 的产品编译库、CompilerInfra Python 绑定、PyTorch 和适合本机 GPU 的 Triton。以运行产品的虚拟环境作为安装前缀时，可以直接导入公开包。

```sh
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH="$COMPILERINFRA_PREFIX;$LLVM_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
intentdsl-compile --layout
```

```python
import torch
import intentdsl as intent
from intentdsl.examples import softmax

x = torch.randn(64, 1024, dtype=torch.float16, device="cuda")
kernel = intent.jit(softmax, backend="triton")
y = kernel.run(x)
call = kernel.prepare(x, outputs=y)
call.compile()
call.launch()
torch.testing.assert_close(call.result(), torch.softmax(x, dim=-1))
```

实施范围为二维行并行、整行读写、权重向量广播、f16/f32 算术、max_num 和 sum。诊断或比较时可给 `jit` 显式传入安装好的产品库路径。安装在独立前缀时，将其 Python site-packages 加入执行环境；编译库随同该包定位。

行归约实现位于 `levels/execution/block/decisions/strategies/RowReduction.cpp`。同一目录的 `<Pattern>.cpp` 通过函数 `register<Pattern>(Unit&)` 登记策略，链接由相邻 CMake 组装。默认顺序由 `cmake/default.json` 中的联合结构家族声明。出口经验行与硬件约束位于 `exits/triton/capabilities/`。

开发工具的 `measure` 动作使用独立安装的 `intent_bench.compilerinfra_adapter`。在执行主机激活有 CUDA 依赖的 Python 环境，使 `python3`、CompilerInfra worker、Bench 和产品公共包均可用。负载选择例如：

```json
{"name":"softmax","cases":["64x1024-float16"],"environment":"local-gpu","implementations":["triton","intent:triton","intent-v2:triton"],"baseline":"triton"}
```
