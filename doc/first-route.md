# 从源码到第一次测量

从 [README](https://github.com/Tianchen-ai/Intent#build-from-source) 的三个并排检出开始。在同一终端激活虚拟环境，安装 SDK、Python 绑定、产品及 Bench。GPU 首测需要 CUDA 版 Torch 和 Triton。运行 `examples/first_kernel.py` 时，前端读取函数的源文件，按符号形状绑定输入，把每一行的最大值与和降为设备归约，再由 Triton 编译和启动。末尾的 Torch 比较覆盖整个输出。

在 Linux 空目录运行以下命令。需预装 Git、Python ≥3.10、Ninja、CMake ≥3.22、C++17 编译器，以及带 PIC 静态库和 TableGen 的 LLVM/MLIR 20.1.8 开发包。

```sh
git clone https://github.com/Tianchen-ai/compilerinfra.git
git clone https://github.com/Tianchen-ai/Intent.git
git clone https://github.com/Tianchen-ai/Intent-Bench.git
python3 -m venv .venv
. .venv/bin/activate
LLVM_PREFIX=$(llvm-config-20 --prefix)
cmake -S compilerinfra -B build/sdk -G Ninja \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILERINFRA_LINK_MLIR_DYLIB=OFF -DCMAKE_INSTALL_PREFIX="$PWD/sdk"
cmake --build build/sdk --parallel 8
cmake --install build/sdk
export CMAKE_PREFIX_PATH="$PWD/sdk:$LLVM_PREFIX"
python -m pip install ./compilerinfra/python ./compilerinfra/agent
python -m pip install ./Intent
python -m pip install -e ./Intent-Bench
python -m pip install torch triton
```

1. `intentdsl setup --backend triton` 检查打包的库、资产和前端。查看报告中的具体探针。
2. `intentdsl doctor --backend triton` 报告所选出口的 `ready` 和 `missing`。补齐列出的驱动、编译器或 Python 包后重跑。
3. 在三仓的父目录执行 `python Intent/examples/first_kernel.py`。看到 `softmax passed` 表示设备已编译、执行且数值通过。
4. 按 README 进入 `Intent-Bench`，先运行带 `--check-only` 的命令，再运行计时命令。`intent-v2:triton` 是本产品，`triton` 是独立参考实现；数值裁判另行执行。用 `--output-root` 指定的目录查看 `latest` 下本次运行的 `results.json`，检查各行的 `status`、`correctness`、`median_ms` 与 `samples_ms`。

[中文工具书](index.md) · [English handbook](index.en.md)
