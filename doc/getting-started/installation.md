# 安装

在实际执行 kernel 的机器上安装。后端显式选择，硬件资源由本机探测。
Python 包名是 intentdsl，导入名为 intent。

## 安装已有 wheel

```bash
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
python -m pip install 'intentdsl[manual,examples]'
intent setup --target triton
intent doctor --target triton
python -m intent.examples softmax --backend triton
```

本地构建的 wheel 用 `pip install /path/to/intentdsl.whl` 安装。
公开 x86-64 wheel 需要兼容 glibc；本机 wheel 必须匹配执行主机架构与系统 ABI。
wheel 包含 intent-compile、intent-opt、配置表、手册及所需的非系统运行库，
安装匹配的二进制包不需要 LLVM 开发 SDK。

manual extra 安装 MCP 依赖，examples extra 安装数组与 dtype 支持。
setup 只安装[provider 注册表](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/targets/registry.py)声明的 Python 依赖，
不安装驱动。doctor 检查实际编译器、provider 工具链与本机目标，不代表数值验证。

## 在执行机上从源码构建

需要 Linux、Python 3.10–3.12、CMake、Ninja、C++17 与 LLVM/MLIR 20 SDK，
包括 headers、libraries、TableGen 与 CMake packages。Debian/Ubuntu 对应包包含
llvm-20-dev、libmlir-20-dev、mlir-20-tools；不需要 Python MLIR bindings。

构建 Weft provider 时需链接当前 Triton–Weft 的 Canonical IR 库，先用同一套 SDK 构建并传入下面的
两个 Weft 参数。仅构建其他 provider 时可省略 Weft 构建与这两个参数：

```bash
cmake -S /path/to/triton-weft/third_party/weft -B /path/to/weft-build -G Ninja \
  -DWEFT_CANONICAL_ONLY=ON -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/path/to/mlir/cmake -DLLVM_DIR=/path/to/llvm/cmake
cmake --build /path/to/weft-build --target WeftKernelDialect
python3 environment/install.py --backend triton --venv /path/to/venv --examples \
  --mlir-dir /path/to/mlir/cmake --llvm-dir /path/to/llvm/cmake \
  --weft-source-dir /path/to/triton-weft/third_party/weft --weft-binary-dir /path/to/weft-build \
  --build-dir /path/to/build/intent --jobs 8
```

Canonical-only 构建只提供 Intent 链接的类型、操作和 verifier；执行机仍需安装完整
Triton–Weft JIT 与匹配的原生 SDK。旧 standalone Weft dialect 缺少原生 scan 等操作，
不能代替当前接口。Canonical 库与 Intent 使用同一套 MLIR，原生 provider 使用其自己的 SDK。

安装器复用环境与构建目录，安装包、执行 setup，再调用 doctor。
它不安装驱动、外部编译器或服务。默认构建目录在用户缓存中；
开发和 Bench 使用时显式选择受控路径。SDK 必须完整，部分 LLVM 源码构建需先补齐依赖库。

## 各 provider 工具链

| 后端 | 执行主机与依赖 |
|---|---|
| triton / cutile | CUDA 主机、兼容 NVIDIA 驱动及所选 Python provider |
| mojo | x86 主机、Mojo、C 编译器（cc）与 GNU OpenMP 运行库（libgomp.so.1）；通过 INTENT_MOJO 或 backend_options 选择 Mojo 编译器 |
| weft | x86/RISC-V 主机、支持 Canonical `.weftir` 的完整 Triton–Weft 发行包、匹配的 weft-core 原生 SDK 及 CPU 工具链 |
| bangc | MLU370 主机与 NeuWare 的 CNRT/CNCC |
| triton_ascend | 昇腾 NPU 主机、匹配的 CANN、torch-npu 与正式 Triton–Ascend 发行包 |

源码安装保留前面的 SDK 参数；构建 Weft 时加入 canonical library 参数，再选择后端附加参数：

```bash
# install.py 的对应附加参数：
# --backend mojo --target-option 'compiler="/path/to/mojo"'
# --backend weft
# --backend bangc --target-option 'neuware="/path/to/neuware"'
# --backend triton_ascend
```

Weft 使用完整 Triton–Weft JIT：Intent 生成 Canonical `.weftir`，交给
`triton.compile` 编译、缓存和导出原生任务入口，再按作者任务阶段执行。用户选择
CPU 集合和线程预算，不手填 ISA/VLEN；Intent 拥有任务调度，provider 的单任务入口不再派发另一层线程池。
执行机必须安装支持该输入格式的发行包，旧 `weft-compile` 不替代这条路径。
BANG C 查询 CNRT 的实际资源；不支持的架构会准确拒绝，不伪装成 MLU370。
普通 JIT 的 shape/stride specialization 来自实际输入 buffer。

Triton 的 PTX 汇编器由 Triton SDK 选择，`intent doctor --target triton`
显示实际路径与版本。需要选择本机另一套 CUDA 汇编器时，在启动 Python 前设置
`TRITON_PTXAS_PATH=/path/to/ptxas`；Blackwell 使用对应的
`TRITON_PTXAS_BLACKWELL_PATH`。Intent 不在调用中切换全局 SDK 配置，也不在
原生编译或执行失败后更换汇编器。工具链更新后应重新验证实际 kernel。

```python
run = intent.jit(definition, backend="mojo",
                 backend_options={"compiler": "/path/to/mojo"})
```

doctor 使用相同的本机目标解析，例如
`intent doctor --target weft`。Weft 的显式工具链选项通过
`backend_options={"options": {...}}` 传给 Triton–Weft 的正式选项解析器。
独立产品使用自己的 INTENT_CACHE_DIR/XDG 缓存；仅在 Bench 启动时由 Bench 重定向。

NVIDIA Triton、Triton–Weft 与 Triton–Ascend 都提供 `triton` Python 包。
为这些发行包使用独立虚拟环境；不要在一个环境反复覆盖安装，或修改 `sys.path`、全局 driver 来切换。
每个环境正常安装 Intent 与相应 provider，选择实际设备后由 `intent doctor` 检查工具链。
Triton–Ascend 直接消费 Block 程序，硬件查询读取实际架构和 Cube/Vector 核数；不把兼容 API 的共享内存占位字段当成物理容量。

## 构建分发包

```bash
python3 environment/build.py --platform native \
  --output-dir /path/to/dist --work-dir /path/to/package-work \
  --mlir-dir /path/to/mlir/cmake --llvm-dir /path/to/llvm/cmake \
  --weft-source-dir /path/to/triton-weft/third_party/weft --weft-binary-dir /path/to/weft-build
```

native 模式适用于 x86-64、RISC-V 等执行主机。脚本从 sdist 构建、打包运行库与
第三方 notices，再在 checkout 外检查安装后的基础编译器。
显式使用 `--platform manylinux_2_35_x86_64` 才选择 Ubuntu 22.04 x86-64 的
auditwheel 路线；native wheel 不自动成为 manylinux wheel。

自定义 SDK 用 --runtime-notices 提供共享库到 license/notice 文件的 JSON 映射。
所有产物留在 checkout 外，不自动发布或 push。
安装/生成、原生编译、执行和数值正确分别判断。
