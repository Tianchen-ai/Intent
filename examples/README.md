# Intent 作者程序与框架示例

[English](README.en.md)

算法唯一保存于 [intent.examples.kernels](../python/intent/examples/kernels/)，
完整 host 编排位于 [intent.examples.programs](../python/intent/examples/programs/)。
两者随安装包提供，源码目录中的脚本只是使用入口。

```bash
python -m intent.examples --list
python -m intent.examples softmax --backend triton
python -m intent.examples paged_decode --backend cutile --prepared
python -m intent.examples group_norm_backward --backend mojo
python -m intent.examples attention --backend weft --stage native
python -m intent.examples softmax --backend bangc
```

后端必须显式选择；程序在执行主机使用实际输入设备，自动探测硬件能力。
`--device` 选择本机设备，`--workers` 选择 CPU 线程预算，
`--option NAME=JSON` 选择 provider 工具链选项。
source/native/run 分别表示生成、原生编译和执行，不混同数值通过。
批量检查与计时使用独立 Bench 的 `check` 和 `run`。

## 多 kernel 编排

[composition.py](../python/intent/examples/programs/composition.py)
中的 GroupNormBackward、PagedDecode、MoEAlignment 等组合保留作者的
kernel 次序和 workspace。创建 `intent.jit(definition, backend=...)`，
再将这些调用对象交给组合；compiler 不根据 program 名称选择算法。

## 框架与产物调用示范

```bash
python examples/softmax.py --backend triton --prepared
python examples/softmax.py --backend cutile --inspect-native
python examples/softmax.py --backend triton --torch-compile
python examples/softmax_forward_backward.py --backend mojo --torch-compile
```

GPU 与 Mojo 可在准备一个实际设备调用后注册 PyTorch custom op。
作者负责 backward，图捕获前完成必要编译与调优。
`softmax.py --program /path/to/export` 展示显式产物加载。

完整接口及缓存/执行边界见[编译与运行](../doc/getting-started/usage.md)。
