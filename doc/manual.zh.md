# IntentDSL 用户与组件开发手册（中文）

## 1. 安装与环境准备 (`install`)

### 从 Wheel 安装（无需 LLVM 开发 SDK）

发行版 `intentdsl` 将所需的 MLIR/LLVM 组件静态链接进 `libintentdsl.so`，运行期无需安装 LLVM/MLIR 开发包：

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install compilerinfra==0.4.1 intentdsl==2.0.0
# 如需 CLI / MCP 编译器开发动作，额外安装：
python -m pip install compilerinfra-agent==0.4.1
```

支持的原生二进制基线：
- **x86_64**：`manylinux_2_31_x86_64`（`glibc >= 2.31`，覆盖 Ubuntu 20.04 MLU 主机及更高版本 x86_64 Linux）与原生 `linux_x86_64`。
- **riscv64**：`linux_riscv64`（`glibc >= 2.38`，原生支持 SG2044、RVV V100 等 RVV 1.0 平台）。

安装后验证绑定库、工具链与可见设备：

```sh
intentdsl doctor
intentdsl setup
intentdsl manual --lang zh
```

### 各后端外部依赖与同进程组合说明

第三方硬件编译器与运行库作为外部依赖按需探测，不随 wheel 重分发：

| 后端 (`backend`) | 执行设备 | 外部工具链与运行时依赖 |
|---|---|---|
| `triton` | NVIDIA GPU | PyTorch (CUDA)、GPU `triton`、CUDA 驱动 |
| `cutile` | NVIDIA GPU (`SM90`/`SM120`) | PyTorch (CUDA)、`cuda-tile` (`cuda.tile`)、`nvcc` 与 CUDA 驱动 |
| `mojo` | x86_64 CPU (`AVX2`/`AVX-512`/`AMX`) | `mojo` 编译器（或设 `INTENT_MOJO`）、`gcc` + `libgomp.so`；可选 `oneDNN >=3.13,<4`（设 `ONEDNN_ROOT` 或 `INTENT_ONEDNN_LIB`） |
| `weft` | x86_64 / riscv64 CPU | `triton-weft` + `weft_core`、C/C++ 工具链、同构 CPU 亲和集（`taskset`） |
| `bangc` | Cambricon MLU (`MLU370`) | `cncc`（`mtp_372`）、`libcnrt.so`、`torch_mlu`、寒武纪驱动（`/dev/cambricon_dev*`）；`libcnnl.so` 记录为外部对照库依赖 |

**Triton 发行组合说明**：GPU 版 `triton` 与 CPU 版 `triton-weft` 均占用顶层 `triton` Python 包名，因此同一进程内不能同时加载 GPU `triton` 与 CPU `weft`，请分置独立虚拟环境；`mojo` 可与 GPU `triton`/`cutile` 或 CPU `weft` 同进程共存，`bangc` 在 MLU 主机独立运行。

### 从源码发布包 (`sdist`) 干净重建

构建机需安装：
- `CompilerInfra == 0.4.1` 原生 SDK 与 Python 包
- `LLVM / MLIR == 20.1.8`（静态库需带 `-fPIC`）
- `CMake >= 3.22`、`Ninja`、支持 C++17 的 `GCC >= 11` 或 `Clang >= 16`、`Python >= 3.10`

```sh
unset PYTHONPATH INTENTDSL_LIBRARY
python -m venv /tmp/intentdsl-build-env
/tmp/intentdsl-build-env/bin/pip install --upgrade pip build
/tmp/intentdsl-build-env/bin/pip install compilerinfra-0.4.1-*.whl
CMAKE_PREFIX_PATH="/path/to/compilerinfra-0.4.1:/path/to/llvm-20" \
  /tmp/intentdsl-build-env/bin/pip wheel --no-build-isolation --no-deps \
  intentdsl-2.0.0.tar.gz -w /tmp/dist
```

---

## 2. 作者语言与数值合同 (`language`)

IntentDSL 只解析 Python AST 生成类型化 `kir` IR，不把内核体当作 Python 解释执行。

- **方向与标量/视图注解**：`I.In[dtype, shape]`、`I.Out[dtype, shape]`、`I.InOut[dtype, shape]`；支持符号维度（如 `("M", "N")`）与标量参数（`I.f32`、`I.i32`、`I.u64`）。
- **结构化域与集合原语**：`I.parallel(...)` 声明并行实例域；`I.domain(0, extent)` 声明归约/流式成员轴；`I.static_range(K)` 声明编译期展开条带；内置 `I.reduce`、`I.sum`、`I.welford`、`I.cumsum`、`I.reduce_by_key`、`I.ordered_sum` 及矩阵/递推原语。
- **严格数值合同**：默认保留 IEEE-754 `f32` 累加、NaN 传播与最近偶数舍入（如 MLU `div`/`sqrt` 0-ULP 认证微核与精确 BF16 舍入）。仅当作者显式声明数值许可（如 `batched_gemm_subnormal_ftz` 的 `allow_subnormal_ftz=True`）时，代价模型才允许启用冲零次正规数的硬件快路（如 AMX BF16）。

```python
import intentdsl as I

@I.kernel
def rms_norm(
    x: I.In[I.f32, ("M", "N")],
    gain: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
    inv_c: I.f32,
    eps: I.f32,
):
    for r in I.parallel(x.dim(0)):
        u = x[r, :]
        s = I.sum(u * u, axis=0, acc=I.f32)
        out[r, :] = (u * I.rsqrt(s * inv_c + eps)) * gain[:]
```

---

## 3. 模型接入与运行期调用 (`model`)

- **单内核 JIT、热调用与换 Buffer**：
  `kernel = I.jit(rms_norm, backend="bangc")`（或 `"triton"` / `"cutile"` / `"mojo"` / `"weft"`）。
  - `y = kernel.run(x, gain, 1.0 / N, 1e-6)`：一步完成分配、准备与执行。
  - `call = kernel.prepare(x, gain, 1.0 / N, 1e-6, outputs=y)`：分离准备与热执行（`call.execute()` 或返回异步句柄的 `call.launch()`）。更换输入/输出张量时再次调用 `kernel.prepare(x2, gain, 1.0 / N, 1e-6, outputs=y2)`，只要形状落在已编译产物的 `guard` 区间内即直接复用原生二进制，无需重编。
- **多阶段 `@I.program`**：静态捕获 `I.workspace`、`I.output`、`I.fill` 与子内核调用序列，在设备流或原生 CPU 任务区间内按序提交（内置 30 个完整示例见 `intentdsl.examples`）。

---

## 4. 组件开发、候选对比与 Agent 接入 (`components`)

### 在既有节点或出口新增策略 / 微核

1. **查清真实落点**：运行 `layout` / `inspect` 查看 `kir -> {block -> {triton, cutile, dsa -> bangc}, cpu -> {mojo, weft}}` 编译树、各边界契约与组件源码行号。
2. **节点策略**：例如在 `levels/storage/dsa/decisions/Supply.cpp` 或 `levels/execution/block/decisions/strategies/` 扩展候选枚举与 `verify`，在同单元 `cost/` 更新代价项，并在 `cmake/default.json` 登记策略名。
3. **出口微核**：在 `runtime/<exit>/kernels/` 添加微核源码，在 `exits/<exit>/ir/` 注册 `ImplementationContract`（写明操作数、内存空间、累加类型与数值界），在 `legalize/` 绑定终端 IR op，由 `TerminalPrinter` 逐 op 拼写。

### 可复制的 CLI 与 MCP 开发闭环命令

设产品工作树为 `$REPO`，已安装库为 `$LIB`（可由 `python -c "import intentdsl; print(intentdsl.library_path())"` 取得），开发输出目录置于源码树外 `/tmp/ci-dev`：

```sh
# 1. 查看已安装编译树、前端绑定与随包 skills
compilerinfra-dev --library "$LIB" --workspace "$REPO" --output-dir /tmp/ci-dev inspect
compilerinfra-dev --library "$LIB" --workspace "$REPO" --output-dir /tmp/ci-dev \
  read-skill '{"name":"agent/skills/intentdsl-development.md"}'

# 2. 在同一持久开发会话中派生候选（derive）、编译观测（compile / explain）并交给独立 Bench 测量（measure）
compilerinfra-dev --library "$LIB" --workspace "$REPO" --output-dir /tmp/ci-dev << 'EOF'
{"action":"derive","arguments":{"configuration":{"exits":{"bangc":{"options":{"isa":372,"nram_bytes":786432,"wram_bytes":524288,"clusters":4,"cores":4,"calibration":"mlu370"}}}}}}
{"action":"doctor","arguments":{"compiler":"base"}}
EOF

# 在交互会话中基于返回句柄 <cid> 执行编译解释与独立 Bench 端到端测量：
# {"action":"compile","arguments":{"compiler":"<cid>","input":"softmax.py","exit":"triton","author_input":true}}
# {"action":"explain","arguments":{"compilation_id":"<compilation_id>"}}
# {"action":"measure","arguments":{"compiler":"<cid>","output_dir":"/tmp/bench-out","workload":{"name":"program/softmax","cases":[],"implementations":["intent-v2:triton"]}}}

# 3. 生成 Codex 与 Claude 的 stdio MCP 配置（共享同一动作目录与 skills）
compilerinfra-dev host-config -- \
  '{"host":"codex","library":"'"$LIB"'","workspace":"'"$REPO"'","output_dir":"/tmp/ci-dev"}'
compilerinfra-dev host-config -- \
  '{"host":"claude","library":"'"$LIB"'","workspace":"'"$REPO"'","output_dir":"/tmp/ci-dev"}'
```
