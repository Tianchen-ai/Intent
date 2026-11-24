# V2-6.R：发行包与独立安装计划

本计划承接 [V2-6 第 4 节](v2-6.md) 的发行要求，目标是让用户在脱离源码树与 LLVM 开发 SDK 的干净环境中安装并调用 IntentDSL 五条本机 JIT 路线，同时保持打包流程对并发提交可重复执行。

## 1. 发行包拓扑与许可证边界

| 包名 | 版本 | 构建输入 | 交付内容 |
|---|---|---|---|
| `compilerinfra` | `0.4.1` | `compilerinfra/python` | 进程内 C ABI 绑定、受控模块加载器、发布事务与测量解包 |
| `compilerinfra-agent` | `0.4.1` | `compilerinfra/agent` | `compilerinfra-dev` CLI、`compilerinfra-agent` stdio MCP 及框架开发 skills |
| `intentdsl` | `2.0.0` | `projects/intentdsl` + 已安装 `CompilerInfra 0.4.1` SDK | `libintentdsl.so`、`intentdsl-compile`、`intentdsl.assets/`（前端、8 个 runtime、微核源码、产品 skills）、ABI 头、30 个作者 `program` 及中英文离线手册 |

- **无需运行时 LLVM SDK**：构建发行 wheel 时开启静态链接 `COMPILERINFRA_LINK_MLIR_DYLIB=OFF`（带 `-Wl,--exclude-libs,ALL`），将所需的 MLIR/LLVM 静态库编入 `libintentdsl.so`，用户机器无需安装 `libMLIR.so` / `libLLVM.so` 或 LLVM 开发头。
- **外部依赖不随包重分发**：CUDA / Triton / cuTile、Mojo / GCC OpenMP、oneDNN、Triton–Weft 以及寒武纪 `CNCC`、`CNRT`、`CNNL` 和设备驱动均作为用户环境外部依赖按需探测，清理源码中所有本地开发机绝对路径。

## 2. 可重复打包与 sdist 重建

其他工作分支仍在持续合入 `main`，因此打包不在源码树内遗留 `build/` 或 `dist/`（避免污染 `tree_build.py` 的产品根目录审计）：

1. `projects/intentdsl/pyproject.toml` 通过自定义 `setuptools` 构建后端在外部临时目录调用 CMake，从 `CMAKE_PREFIX_PATH` / `CompilerInfra_DIR` 定位 SDK `0.4.1`，将安装树写入 wheel 的 `intentdsl/_bundled/`，由 `intentdsl` 默认定位并兼容 `INTENTDSL_LIBRARY` 覆盖。
2. `sdist` 完整收录手写 C++、TableGen（`.td`）、`CMakeLists.txt`、`product.json`、`cmake/`、`frontend/`、`runtime/`、`agent/`、`doc/` 与 `README.md`。
3. 提供幂等脚本 `cmake/package_release.py`：可在任意时刻基于最新 `main` 一键产出 `compilerinfra`、`compilerinfra-agent`、`intentdsl` 的 `sdist` 与原生 `wheel`，并在清除 `PYTHONPATH` 与 `INTENTDSL_LIBRARY` 的全新虚拟环境中验证“从 `sdist` 重编 wheel + 安装核验”。

## 3. 架构与 manylinux / glibc 基线

| 平台与包类型 | 目标标签 | 构建与验证环境 |
|---|---|---|
| x86_64 兼容包 | `manylinux_2_31_x86_64` | 在 `glibc 2.31`（Ubuntu 20.04，MLU370 主机）构建并核验 `objdump -T` 动态符号上限 `<= GLIBC_2.31`；分别在 `glibc 2.31`（MLU370）和 `glibc 2.35`（x86_64 GPU/CPU 主机）实装验证 |
| x86_64 原生包 | `linux_x86_64` | 在本地 x86_64（`glibc 2.35`）构建并验证 `mojo`、`weft`、`triton`、`cutile` |
| riscv64 原生包 | `linux_riscv64`（`glibc 2.38`） | 在原生 RISC-V 主机（SG2044 / RVV V100）从源码/sdist 构建并验证 `weft`、ISA/VLEN 与同构 CPU 亲和集 |

- 同进程 Triton 组合说明：x86_64 上 GPU `triton` 与 `triton-weft` 共享顶层 `triton` Python 包命名空间，需分置独立环境（或在同一 Triton–Weft 环境中按已安装后端启用），在 `doctor` 与手册中如实说明。

## 4. 外部依赖、`setup` 与 `doctor`

- **BANG C / MLU 外部依赖登记**：在 `exits/bangc` 与前端诊断中探测 `cncc`（编译器与目标架构 `mtp_372`）、`libcnrt.so`（运行时与本地 MLU 设备计数/ISA）、`libcnnl.so`（成熟算子对照库）及寒武纪驱动（`/dev/cambricon_dev*` / `cnmon`），记录为外部依赖；CNNL 按既有裁决不注册为 `product.json` 的 `library` 档收缩实现。
- **`intentdsl setup` 与 `intentdsl doctor`**：提供 Python API（`intentdsl.doctor()`、`intentdsl.setup()`）与 CLI（`intentdsl doctor`、`intentdsl setup`），输出已绑定 `libintentdsl.so` 位置、`ci_doctor()` 原生可选依赖报告，以及五个显式后端（`triton`、`cutile`、`mojo`、`weft`、`bangc`）的工具链、运行时库、可见设备与缺失项修复指引；同时校验错误设备（如在 CPU 张量上请求 `bangc`/`triton`）和缺工具链（如屏蔽 `PATH` 中的 `cncc` / `mojo`）时的明确报错。

## 5. 双语手册、离线查询与 Agent 接入

- `doc/` 新增精炼双语手册（`doc/manual.zh.md`、`doc/manual.en.md`），覆盖：安装与从 `sdist` 构建、作者语言与数值许可、代价模型与在既有节点/出口新增策略或微核、父子候选 `derive → compare → explain → measure` 闭环及独立 Bench 交接、Codex/Claude 的 `compilerinfra-dev host-config` 可复制命令。
- 根 `README.md` 只保留安装、首个真实调用、离线手册查询和开发贡献入口。
- 手册与产品 skills 随 `intentdsl` 包安装；`intentdsl manual [topic] [--lang zh|en]` 与 `intentdsl.manual(...)` 直接读取包内资产，断网可用。

## 6. DSH 适配去留裁决

- **决定**：删除 `compilerinfra/hosts/dsh`，并移除 `compilerinfra/doc/plugins.md` 与 `compilerinfra/README.md` 中的旧引用。
- **理由**：早期参考 DSH 仅借鉴其“作用域化工具挂载与回滚”语义（该机制已在 `compilerinfra` 的 `derive / adopt / revert` 与 `installation.publish` 中原生实现），并非要维护 DSH 宿主适配；`hosts/dsh` 停留在 v0.2.0 原型且依赖未公开发布的 `@deepseek-ai/dsh-mcp-client@0.0.1-rc.1`，未跟进 v0.3/v0.4 合同，当前 Codex 与 Claude 统一通过 `compilerinfra-agent`（MCP）和 `compilerinfra-dev`（CLI）消费同一动作目录与 skills。
