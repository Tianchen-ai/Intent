# IntentDSL

IntentDSL compiles typed Python author kernels (`@intentdsl.kernel`) and static multi-kernel sequences (`@intentdsl.program`) to five native JIT backends: **Triton** and **cuTile** (GPU), **Mojo** and **Weft** (x86_64 and RISC-V CPU), and **BANG C** (Cambricon MLU).

## Install / 安装

```sh
python -m pip install compilerinfra==0.5.2 intentdsl==2.0.0
intentdsl doctor
```

Wheel installations bundle `libintentdsl.so` with static MLIR/LLVM linkage (no LLVM dev SDK required at runtime). Device compilers/runtimes (`triton`, `cuda.tile`/`nvcc`, `mojo`/`gcc`, `triton-weft`, `cncc`/`cnrt`/`cnnl`) are probed as external dependencies via `intentdsl doctor` and `intentdsl setup`.

## First Call / 首个真实调用

```python
import torch
import intentdsl as I
from intentdsl.examples import softmax

x = torch.randn(64, 1024, dtype=torch.float16, device="cuda")  # or "cpu" / "mlu"
kernel = I.jit(softmax, backend="triton")                      # triton | cutile | mojo | weft | bangc
y = kernel.run(x)
torch.testing.assert_close(y, torch.softmax(x.float(), dim=-1).half())
```

## Offline Bilingual Manual / 离线双语手册

Query the packaged manual without network access:

```sh
intentdsl manual --lang zh
intentdsl manual --lang en
intentdsl manual components --lang zh
```

- 中文手册：[doc/manual.zh.md](doc/manual.zh.md)
- English Manual: [doc/manual.en.md](doc/manual.en.md)

## Contributing / 开发与贡献入口

- Architecture & Compilation Tree: [doc/architecture.md](doc/architecture.md)
- Agent & Component Development Rules: [AGENTS.md](AGENTS.md) and `agent/skills/intentdsl-development.md`
- Install `compilerinfra-agent==0.5.2` and run `compilerinfra-dev --library "$(python -c 'import intentdsl; print(intentdsl.library_path())')" --workspace . inspect` to inspect units, derive candidates, and connect to Bench.
