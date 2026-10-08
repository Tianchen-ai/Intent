# Shared Executable GPU Program IR

## 1. 角色

Shared GPU IR是一份provider-neutral、launchable、可独立验证的block program。它位于canonical KIR之后、Triton/cuTile source legalization之前。

它表达GPU source providers共同需要的program-level事实，但不表达下层distributed layout或机器指令。Provider serializer不得把缺失的执行结构留到字符串阶段补齐。

## 2. 顶层结构

一个physical module包含：

- external target-capability object；
- 一个或多个由不同KIR specializations产生的physical kernels；
- 每个kernel的public ABI和compiler-private resource parameters；
- compile-time physical parameter declarations；
- kernel的program space、body与origin map。

一个physical kernel对应一个target artifact和一次launch。Compiler-private workspace可以成为额外kernel argument并由runtime分配，但不能引入隐藏的第二次launch或跨kernel同步。

## 3. Program space

Program space定义有限physical program instances。IR显式表示：

- program-space extents；
- current program coordinates；
- execution-group segment lengths与prefix offsets；
- coordinates到execution group与logical workset coordinates的双射或guarded mapping；
- tail/empty program validity；
- 可选grouping、swizzle或grid-stride traversal。

Program space可以是任意logical rank；provider legalization负责flatten或映射到目标surface允许的grid rank。Mapping必须是一等IR关系，不能由serializer根据axis role或result shape重建。Runtime extents只能依赖launch-visible ABI scalars与shape relations；需要读取device data才能确定长度的ragged/member domain必须留在program body内，除非KIR本身已提供可直接用于launch的mapping。

每个workset由`execution_group` region持有实际执行体。其operands保存当前linear program coordinate与runtime extents，typed attributes保存对应launch extents、coordinate roles和segment identity/bounds；region arguments是按这些extents作row-major解码的index coordinates。外部dispatch guard仍是显式structured control，group本身不增加launch或barrier，也不改变body中的effects。修改mapping必须同时重绑region arguments与其执行体。

Provider入口在形成native grid后统一展开execution groups：region arguments替换为纯`delinearize`的结果，body原位展开。`delinearize`只表示`linear`和`extents`的数学解码，不保存launch、segment或ownership事实；普通CSE/DCE可合并或删除它，不需要无use保活规则。

Program coordinates只存在于physical IR，不回流成DSL/KIR values。

### 3.1 LaunchExpr

Program-space extents、execution-group segment offsets与provider grid使用受限typed `LaunchExpr`。其leaves只能是：

- integer/bool constants与已绑定physical parameters；
- public ABI中按值传入的scalar parameters；
- external view ABI中host可见的shape/stride metadata。

允许的operations是无effect的integer/bool arithmetic、comparison、`select`、`min/max`、floor/ceil division与tuple indexing；canonical equality/bounds facts只用于证明表达式合法，不能替代runtime value。`LaunchExpr`不能load device memory、调用kernel helper或依赖kernel-body SSA。

Provider wrapper在launch前求值`LaunchExpr`以形成grid；kernel body从同一typed expression重建dispatch/validity。若provider launch ABI不能在两侧直接共享某个metadata value，lowering把wrapper求得的结果声明为compiler-private scalar argument；该argument及其producer必须在physical module中显式存在，不能由serializer临时添加。

## 4. Values 与 types

共同GPU IR至少区分：

### 4.1 Scalar

Canonical scalar、logical index与predicate保持精确定义。Scalar不会为了统一代码路径被伪装成一元素fragment。

### 4.2 Fragment

Fragment是一个program instance拥有的静态或compile-time-parameterized shaped SSA value：

```text
fragment<element_type, [physical_extent_exprs]>
```

Fragment保存：

- element dtype；
- physical shape expressions；
- 各fragment axes到logical coordinates的compositional mapping及其source-axis provenance；
- active member/index set与validity的shape relation；
- owning program instance/workset。

Fragment不保存register/lane/warp/CTA distribution、shared/TMEM layout、MMA encoding或provider memory space。这些由外部provider compiler或被真实差异逼出的local extensions决定。

### 4.3 View

External view保留public ABI中的allocation identity、element offset、logical shape、element strides、access mode、bounds与alias semantics。Physical access使用显式coordinates消费view，不把pointer arithmetic留给emitter。

### 4.4 Buffer

Physical buffer是kernel内mutable resource，显式保存：

- element type与shape；
- allocation scope：program-private、iteration-private或invocation workspace；
- allocation-instance identity与ownership domain；
- initialization mode与可选full initial value，或uninitialized-first-write obligation；
- lifetime与uses；
- sharing/visibility obligation；
- 是否需要compiler-private workspace ABI。

具有full initial value的buffer在每个logical allocation instance上恰好初始化一次。未初始化buffer不产生默认fill，保留作者的先写后读义务；verifier不因静态初始化证明不足而拒绝程序。依赖初始化事实的优化仍须取得所需证明，不能把Unknown当成已初始化。Invocation workspace由runtime在同一次launch前分配；每个slice必须有唯一owner或明确的atomic/scatter-reduction semantics，不能依赖隐藏init kernel或kernel-global barrier。

共同IR不指定register、local、shared或TMEM。它先决定SSA value、program-private resource还是invocation workspace；provider/storage passes再从lifetime、sharing、resource limits与capability选择目标memory form。

## 5. Control

共同IR复用结构化SSA control：

- scalar-condition `if`；
- ordered `for/while`及loop-carried values；
- execution-group dispatch；
- compiler生成的physical traversal/blocking loops。

KIR中的unordered `parallel`在construction阶段被program mapping或明确的program-internal independent workset兑现，不继续作为一个等待provider解释的语义标记。

Physical loops可以使用runtime bounds或compile-time physical parameters。作者不可观察的blocking loops只存在于physical IR；作者定义的ordered control通过origin与semantic-preservation rules保持。

## 6. Coordinate 与 access relation

每个physical access显式包含：

- resource/view/buffer；
- result axes；
- 每个source axis的typed coordinate expression；
- coordinate对program ids、loops、fragments、runtime indices与source subregions的SSA dependencies；
- 从logical predicate及source bounds组合出的active member/index set与validity；
- load fill或write collision/effect semantics。

共同access operations是：

- load；
- store/unique scatter；
- gather；
- scatter-reduce；
- atomic load/store/RMW/CAS；
- buffer load/store。

Invalid load不产生memory access并返回显式fill；invalid write不产生effect。Pointer tensor、cuTile tile index或descriptor是provider representation，不是共同access identity。

Atomic operation另外显式保存memory order、logical sharing domain、RMW kind、old-value/CAS result schema。Provider scope由该sharing domain与current program mapping推得，而不是由serializer选择默认值。

## 7. Value operations

共同IR包含对scalar和fragment都定义明确的：

- arithmetic、comparison、select、cast与bitcast；
- broadcast、reshape、transpose、join、tuple与record operations；
- explicit scalar↔fragment broadcast/extract；
- immutable value reuse与rematerialization；
- typed pure helper/combiner regions。

Unary/binary operations 保留 KIR 的逐操作 `approximate` 与 `flush_to_zero` 数值属性，默认均为 false；合法 operator/dtype 组合与精度范围由 DSL 数值规格定义。属性不改变 fragment shape、ownership 或 ABI，不是 provider 参数；相同 operands/operator kind 但数值属性不同的操作不是同一纯值。

Broadcast、reshape、transpose、slice、tuple/record extraction与pure helper call同时组合value def-use与coordinate provenance。Physical pass若选择重算pure producer，必须真实改变def-use并保持等价coordinate map；若选择materialize，必须产生buffer/value及对应uses。不能只写`replay=true`、provenance ID或value ID列表。

## 8. Structured compute

Physical structured operations消费当前scalar/fragment SSA：

- reduce：physical axes、逐component identity、声明可结合与交换的typed pure combine region、dynamic extent、result relation与accumulator flow；
- scan：physical axis、逐component identity、可结合的typed pure combine region、direction、inclusive/exclusive、prefix result relation与carry；它保留prefix成员顺序，不继承reduce的重排许可；
- region fold：physical segment loop、source-slice operands、typed summarizer region、summary identity/combine、captures与result flow；
- region scan：physical segment loop、summarizer与transition combine、incoming-state application、slice emitter、source-aligned output assembly与final-state flow；
- contract：physical lhs/rhs/accumulator fragments、paired reduction与batch axis maps、free/result-axis order、zero-reduction result rule、result relation与accumulator dtype；
- scaled/sparse contract：保留KIR的logical format/schema，同时具有physical operand/access mapping；
- histogram：physical input fragment、validity与count result；
-其它被canonical KIR正式定义的local structured operation。

Arg-reduce等复合result还保存tie、NaN与index semantics；ordinary ordered loop则显式保存runtime condition/bounds、loop-carried values、effects与terminators。Region summarizer内原本显式存在的contract/reduce继续是独立physical structured ops；compiler不需要从summary combine猜回它们。上述ops不携带provider primitive名称、MMA version、input precision hint、K-pack、warp policy或pipeline stage。Provider可以直接映射到native primitive、合法展开或明确拒绝，但不得改变KIR semantic schema。

普通reduce的结合交换合同来自operation语义，不需要附加一份combine代数证明或重排计划。Shared passes形成blocking、access与accumulator flow；provider已有等价collective时，其内部归约树、lane/warp通信与同步由provider compiler完成。Scan与ordered region operations继续按各自的顺序合同lower，不为统一实现而把ordinary reduce变成scan再取末项。

Structured op可以另外保存一个由typed predicate analysis得到的physical effective source range。该range只能收窄physical traversal，不修改KIR logical source；被排除的members必须已被证明对每个free lane产生identity/no effect。All-true区间可删除predicate，mixed区间保留原validity，all-false区间可从physical loop/access graph中删除。Effective range与改写后的loop/access SSA属于current program；proof是由current coordinate/predicate relations重算的analysis result，可缓存但不参与执行解释。

## 9. Dependency、storage 与 synchronization

共同IR使用SSA、memory effects、resource identities和explicit dependence edges表达producer-consumer、visibility与lifetime obligations。它不预先选择TMA/cp.async、mbarrier、named barrier或software pipeline。

共同IR不定义跨physical program instances的barrier；跨instances的冲突只能由KIR已有的atomic/scatter-reduction semantics闭合，否则该mapping非法。只有shared physical transformation在一个program instance内部真实创建了异步producer-consumer行为时，才可引入具有跨provider execution meaning的dependency token/operation。仅为拼出某个provider的copy API或为NVIDIA拼出mbarrier，不得污染共同IR。

Provider-local pass可以把共同dependency/lifetime展开成explicit allocation、copy、wait与barrier，也可以把它们委托给下层compiler。

## 10. 完整性不变量

任意shared GPU pass前后，program均满足：

1. exactly one executable physical kernel body per KIR kernel specialization；
2. program space与logical effect/result coverage完整；
3. every SSA value具有合法scalar/fragment/resource type；
4. fragment extents是常量或已声明physical parameter expressions；
5. every access具有resource、compositional coordinates/provenance、active member set、validity及fill/effect；
6. every control region具有完整arguments、yields与dominance；
7. every buffer具有initialization或first-write obligation、lifetime与ownership；
8. every structured op具有完整physical operands/results与semantic schema；
9. non-atomic conflicting effects非法；
10. runtime grid extents不依赖launch后才能读取的device data；
11. every physical range narrowing具有原logical relation的subset/identity proof，并已真实改写loop、access与validity；
12. execution不依赖KIR clone、axis/role字符串或side decision records。

## 11. 明确排除

共同GPU IR不包含：

- provider API spelling；
- kernel-name或whole-op template identity；
- warp/lane/register分布布局；
- TTGIR-style blocked/MMA/shared encodings与`convert_layout`；
- CUDA shared/TMEM address spaces；
- target copy instructions、TMA、WGMMA、TCGEN05、MFMA；
- software-pipeline stages、warp specialization与barrier protocol；
- autotune winner。

上述内容由provider-local extensions或外部provider compiler拥有。

## 12. GEMM 形态示例

完整域contract经过physicalization后，shared GPU IR应具有下列结构，而不是一个contract record加KIR shape：

```text
kernel @gemm<BM, BN, BK>(A, B, C, M, N, K) {
  grid = (ceil_div(M, BM), ceil_div(N, BN))
  (pm, pn) = program_coord(grid)
  m = pm * BM + range(0, BM)
  n = pn * BN + range(0, BN)
  acc = full<fragment<f32, [BM, BN]>>(0)

  for k0 = 0 to K step BK iter_args(acc):
    k = k0 + range(0, BK)
    lhs = load A[m, k] valid (m < M && k < K) fill 0
    rhs = load B[k, n] valid (k < K && n < N) fill 0
    acc = contract lhs, rhs, acc

  store C[m, n] = acc valid (m < M && n < N)
}
```

BM/BN/BK、grid、loops、fragments、coordinates、validity、loads、accumulator与store都属于当前program。Triton serializer只把它们机械写成`tl.program_id`、`tl.arange`、`tl.load`、`tl.dot`和`tl.store`；cuTile映射到`ct.bid`、tile indices、`ct.load/mma/store`。
