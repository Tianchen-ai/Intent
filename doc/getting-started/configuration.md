# 配置与调优

算法 source 不包含 tile、warp、stage、设备型号或 autotune winner。它们由物理程序、完整经验配置和目标工具链负责。

## 使用默认配置

正常的 `intent.compile(..., target=...)` 使用随编译器分发的所选 provider 配置。编译器依据当前 typed program 的计算与遍历结构选择一个配置 family，投影完整经验行，删除可证明非法的行并归并等价行。它不按 kernel 名称选择配置，也不把多个参数列做笛卡尔积。

Triton 消费实际 `Config` 集合；cuTile 与 BANG C runtime 对同一已声明候选集合择优。CPU 配置还绑定明确的实现集合。第一次原生调用可能包含 JIT 与调优，后续调用复用已选配置。调优试跑保持初始 InOut 内容与实际 alias 关系；winner 对调用方参数只执行一次。

## 覆盖完整经验行

`intent.compile(..., tuning_config="/path/to/config.json")` 和 CLI 的 `--tuning-config` 接受有限 JSON 覆盖。给出的 family 整组替换默认行，未给出的 family 继续用默认数据：

```json
{
  "triton": {
    "contraction_narrow": [
      [128, 256, 64, 1, 128, 1, 8, 8, 3, 1, 0],
      [64, 128, 32, 1, 128, 1, 8, 4, 4, 1, 0]
    ]
  }
}
```

前七列是 `ownership_m, ownership_n, reduction, reduction_outer, scan, traversal_workers, traversal_group`；Triton 后四列是 `NUM_WARPS, NUM_STAGES, NUM_CTAS, USE_TENSOR_DESCRIPTOR`。这是有相关性的完整配置，不是各轴的候选集合。投影、约束与过滤最多从一条经验行形成一条候选，不能补候选、拼接其它行的列或把值 `1` 当禁用标志。

配置文件只在 kernel 编译时读取；修改配置需要重新编译 artifact，launch 与调优不再读取它。未知字段、错误列数或类型、重复行、空表、非法值直接诊断；无合法候选不会切换算法或回退默认表。

cuTile、CPU、DSA 列与绑定责任，以及缓存/调优状态见[物理参数规格](../compiler/physical-parameters.md)和[CPU 程序 IR](../compiler/cpu-program-ir.md)。

## 查看实际结果

现有 `examples/softmax.py --inspect-native` 读取 `artifact.observation`。它包含实际选中配置、候选状态，以及 SDK 提供的原生资源。未提供的资源显示缺失原因；估算工作集不等于机器寄存器或 shared memory 分配。观察与调优本身不证明数值正确。
