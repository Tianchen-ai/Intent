# Target Extensions 与 Lowering

## 1. 两个正交选择

Compilation context分别选择：

- source provider：Triton、cuTile或未来的直接backend；
- hardware target：NVIDIA/AMD及具体SM/gfx，或其它GPU architecture。

Provider与hardware不是同一分类。Triton source可以继续由外部Triton编译到NVIDIA或AMD；Intent不得为每个provider预建一份vendor IR，也不得为每个SM/gfx版本建立dialect。

## 2. Target capabilities

Physical module可以读取external target-capability object。它以typed features表达：

- grid/program limits；
- warp/wave size与resource limits；
- supported scalar/fragment dtypes；
- structured primitive与atomic能力；
- provider source forms；
- architecture-specific extension legality。

Passes查询features而不是匹配设备名称。SM90、SM100、SM120或gfx family通过同一IR上的feature predicates、pipeline selection与lowering patterns表达差异。

## 3. Common IR 与 local extension

共同GPU IR始终是唯一完整executable authority。Target-local extension只能扩展当前program，不能复制program mapping、value/access graph或structured semantics。

一项local extension必须同时满足：

1. 共同IR无法无损表达且不应公共化；
2. 不是单纯API名称、参数顺序或字符串差异；
3. 有后续pass/consumer需要读取或改写；
4. 需要独立legality/verifier；
5. serialization前必须成为current program的一部分。

否则使用thin legalization或直接serialization。

设计local form前先核对source与provider的operation合同。合同相同且provider已有原生primitive时，优先直接映射；不能因为Intent内部helper的组织方式，重新分类或实现provider已负责的reduce/scan tree、thread communication或layout strategy。额外更强的顺序或数值要求须有明确作者语义依据，不能从现有lowering或旧文档的偶然限制反推必要性。

显式近似数学与 FTZ 属于已有 unary/binary operation 的共同数值语义，不为 provider API 名称另建 dialect。Provider legality 检查 dtype 与硬件能力；serialization 机械发出满足该数值属性的原语或原语包装，不得把逐操作 opt-in 扩大为影响整份 kernel 的 fast-math 选项。Cloning、bufferization 和 scalarization 必须保留这两个属性；普通运算的既有 lowering 不因相邻操作 opt-in 而改变。

## 4. Triton

共同GPU program可以机械映射：

- program coordinates → `tl.program_id`与grid；
- physical ranges/fragments → `tl.arange`、scalar/blocked tensor values；
- explicit access/validity/fill → pointer expressions、mask/other与`tl.load/store`；
- predicate-proven physical effective ranges → 已选loop bounds、all-valid unmasked body与mixed-range guard；
- gather/scatter/atomic →对应Triton operations；
- physical reduce → `tl.reduce`，保留已确定的component dtype、identity与NaN/tie规则；
- physical scan → `tl.associative_scan`，保留prefix、direction与inclusive/exclusive语义；
- physical contract/scaled contract → `tl.dot/dot_scaled`或满足原合同的合法展开；
- physical region fold/scan → 已选segment loop、summarizer body、summary combine、scan apply/emit与其中显式structured operations；
- structured control → Python/Triton structured control。

普通reduce已经声明结合交换合同，provider无需从combine body重新证明该合同，也无需额外生成保序树或Scan+terminal路径。Triton的默认promotion、NaN规则或dot精度不一定等于Intent合同：先用已确定的类型转换、callback与原生精度选项闭合差异，不将“直接映射”理解为盲用默认参数，也不复制外部compiler的内部instruction dtype。

普通浮点乘加按[数值合同](../dsl/types-numerics-and-effects.md#55-普通乘加的fma融合)使用Triton的`enable_fp_fusion=True`，由其LLVM/PTX lowering形成合法FMA；不另建乘加识别路径或调优参数。该开关独立于任意重结合、dot输入精度和FTZ；libdevice的FTZ reflection保持关闭。显式数值cast和operation规定的独立舍入边界必须由相应lowering保留，不能因开启普通FMA而消失。

Triton-local form可以包括真正影响source program的descriptor value/access、某些provider compile-time branches与Config binding。若pointer/descriptor只是终端spelling差异，可直接serialize；若descriptor选择改变operands、static block constraints并被多个passes消费，则用local extension op表达。

Tensor-descriptor extension必须在current provider program中显式保存base、shape、strides、block shape、padding、alignment与accesses，并声明descriptor runtime allocation的size/alignment、allocator ABI和launch lifetime。Provider launcher负责绑定已声明的allocator requirement；若selected Triton/runtime无法提供，provider legality在serialization前拒绝。Serializer不得临时创建未声明descriptor workspace或allocator路径。

Intent不复制外部Triton的：

- TTGIR distributed encodings与`convert_layout`；
- coalescing、thread locality与MMA/shared/TMEM representation；
- TMA lowering、software pipeline与warp specialization；
- fence/barrier/register与LLVM/PTX lowering；
- autotune winner。

## 5. cuTile

cuTile与Triton共享block-program核心：program/block identity、compile-time fragment extents、pointwise values、load/store/gather/scatter、reduction/prefix operations、MMA与control。

Thin mapping包括：

- program coordinates → `ct.bid`及所需grid values；
- fragment shape与coordinate relation → cuTile tile-space indices；
- common accesses → `ct.load/store/gather/scatter`；
- structured operations → cuTile支持的`ct.sum`/`ct.max`/`ct.cumsum`等reduction/prefix forms，以及`ct.mma/mma_scaled`；region fold/scan按共同program中已经形成的segment loop、summary与state/output flow投影。

cuTile-local legality/forms包括最多三维block identity、tile-space index multiplication、check-bounds/padding、advanced indexing、MMA-scaled layout与provider tuning constraints。它们只有在不只是机械index conversion时才形成local extension。

若selected cuTile surface不提供共同program某项协作行为所需的copy/barrier/sync形式，provider必须明确legalize到其已有等价能力或拒绝，不能生成串行慢路径冒充支持。

## 6. Vendor 与 architecture extensions

共同IR加局部extensions采用与TritonGPU相同的组织原则：

```text
common GPU program
    + optional provider/vendor extension ops
    → target-specific legalization
    → source or lower IR
```

Extension op可以只在某些features上合法；同一pass也可以按features选择不同rewrite pattern。Architecture差异不得进入KIR，也不通过kernel-name branch表达。

## 7. Provider verifier

Provider legalization完成后检查：

- grid rank、program coordinate与compile-time fragment requirements；
- every common operation具有唯一surface lowering；
- local extension operands/results/regions完整；
- unsupported dtype/primitive/access/sync组合已在serialization前拒绝；
- physical parameters完整绑定到provider constexpr/config；
- no provider pass重新读取KIR去推导shared ownership、axis、range、coordinate provenance、access或validity。

## 8. Terminal serialization

Serializer只能：

- 发出imports、signature与decorators；
- 按current IR顺序打印values、control与operations；
- 机械转换types、attributes与provider API spelling；
- 发出已声明的Config/candidate集合与launch wrapper。

Serializer不得：

- 从logical result shape猜fragment width；
- 从KIR relation现场重建pointer/mask；
- 根据role或op名称选择ownership/primitive；
- 创建buffer、workspace、persistent loop或pipeline；
- 增加未声明physical parameters；
- 捕获异常并切换fallback。

## 9. Unsupported 的性质

Unsupported必须在最早拥有足够信息的层声明：

- shared GPU legality：该physical program无法满足共同GPU执行约束；
- provider surface：目标语言没有等价operation/form；
- hardware capability：selected architecture不支持必要dtype/resource/primitive；
- lower compilation cost：source合法，但外部compiler在给定成本界限内无法完成。

这四类不能混成一个compile failure，也不能通过改变算法、缩小scope或发射数量级更慢的替代路径隐藏。
