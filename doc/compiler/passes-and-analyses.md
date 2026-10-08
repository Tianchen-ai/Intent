# Analyses、Passes 与 Verification

## 1. 基本纪律

Pass的成立依据是它改变了当前program，并保持可验证不变量；不是文件名、pass数量或旁表字段。

每个transformation必须声明：

- 读取的KIR semantic facts与current physical facts；
- legality/preconditions；
- 改写的types、operations、regions或def-use；
- 保持的logical values、control、effects与ABI；
- invalidated/recomputed analyses；
- pass后verifier。

不允许kernel-name、op-count、whole-region template或provider字符串驱动shared policy。

编译调用的有效许可和可选控制由 module 的 `intent.compile_options` 保存：`numerics` 为 `source` 或 `relaxed_normalization`，`onlineReduction` 决定是否考虑普通 normalized summary 的 online 改写，`optimizationRemarks` 决定是否输出采用/拒绝原因。默认绑定为 `source/true/false`。该属性随 construction、分支克隆和 provider 容器传递；物理 IR 续编译读取已有属性，不能无声替换或补入另一套默认许可。Provider metadata 的 `compile_options` 与它一致，编译缓存消费实际选项。

Online transformation 先检查当前 summary/轴关系、数值许可、identity、dtype 和 replay/effect 条件，再创建执行结构；开关不跳过验证或其它必需 lowering。当前额外模式的实现域为 f32 score、maximum、mass/moment accumulation，以及同 dtype 的 f16/bf16/f32 weight/value operands；不匹配时保留普通分块路径。这是优化实现的资格，不收紧 DSL 允许的 dtype。有限值前置条件与允许的具体舍入变化由[数值规格 §5.6](../dsl/types-numerics-and-effects.md#56-编译调用的数值许可)定义。

可选 remark 说明已采用、关闭、缺数值许可或不满足当前实现条件，并定位到当前操作；它不是执行计划。额外许可在没有对应优化的 execution family 上可以保持未使用，不能据此改变该 family 的运算。

## 2. Analysis 分层

### 2.1 Canonical analyses

只在immutable KIR上计算：

- shape/domain/subregion identity；
- coordinate provenance、axis maps、active index sets、index relations与alias；
- operation effects与collision semantics；
- control dependence、loop carry与ordered state；
- reduce/scan/region-fold/region-scan/contract schemas；
- logical-buffer lifetime；
- def-use、producer-consumer与logical reuse。

这些analysis以stable KIR IDs为key，不在physical program上重新运行一个假装canonical的KernelModel。

### 2.2 Construction facts

KIR→GPU conversion使用canonical analyses产生：

- logical worksets与execution groups；
- initial program-space mapping；
- KIR origin map；
- initial physical scalar/fragment/value/access/control graph。

Construction完成后，这些结果已经进入GPU IR；后续passes不再把construction facts当作另一份program authority。

### 2.3 Physical analyses

只从current GPU IR计算：

- program coverage与mapping；
- physical dominance、def-use与loop dependence；
- fragment axes、compositional coordinate maps、predicate ranges、access footprints与validity；
- reuse、rematerialization、buffer lifetime与sharing；
- structured-op operand/accumulator flow；
- target-independent resource estimates；
- provider-form eligibility。

Physical analyses可以通过origin map查询immutable KIR semantics，但不能用KIR adjacency或shape替当前IR补出缺失的execution structure。

Analysis默认在相关IR mutation后失效；只有证明保留的analysis才可缓存。

Transformation group是完整的执行边界：group负责实际rewrite及其依赖的value/access/aggregate/accumulator relation维护；中间repair helper不是可单独对外运行的pass。每个group完成后验证同一current program，失败诊断指明rewrite或postcondition失败及group名称，不在verifier内修程序。

## 3. Canonical GPU pipeline

### 3.1 Construct executable program

从KIR、canonical analyses与selected GPU capabilities建立[`kir-to-gpu.md`](kir-to-gpu.md)定义的完整initial program。Conversion同时完成types、regions、values、accesses与origin mapping，不产生等待emitter解释的holes。

Construction仅关闭初始program及indexed-access composition。结构化source规范化、ownership/blocking、online summary与region/reduction/contraction realization属于随后明确的shared transformation groups，不隐藏在construction完成函数中。依赖顺序保证source relation在ownership变换前正规化，physical ranges在structured realization前形成，新产生的access在provider legalization前重新组合；mapping refinement与候选物化消费完成后的program。

### 3.2 Refine program mapping

根据worksets、access、reuse、effects与device resources改写：

- program-space rank与linearization；
- execution-group placement；
- 每个physical program instance的logical ownership extent；
- grouping/swizzle；
- grid-stride或persistent traversal eligibility/structure。

若某种persistent form已由provider lower compiler从普通program structure可靠产生，shared pass不复制其loop；它只保留下层需要的legal input。若Intent负责该结构，pass必须创建真实physical loop。

### 3.3 Form blocking 与 fragments

把scalar或较小work ownership重写为parameterized fragments与nested physical loops：

- 扩大一个physical program instance拥有的logical instances；
- 分离free、ordered、reduction与scan axes；
- 创建scalar↔fragment conversions；
- 更新所有users、access coordinates与validity；
- 保持dynamic subregion的source coordinates。

Blocking pass不得修改KIR logical subregions或作者可观察的page/window/chunk boundaries。

### 3.4 Co-realize values、accesses 与 structured operations

这三类不能由三个互不知情的旁表pass完成：fragment shape改变access footprint，access form影响materialization，contract/reduce skeleton又决定accumulator与reuse。

实现可以拆成多个passes并迭代canonicalization，但每一步必须保持完整program：

- replay/rematerialization改变physical def-use；
- bufferization产生真实allocation/lifetime/read/write；
- access realization产生coordinate/validity/fill/effect operands；
- contract/reduce/scan/region-fold/region-scan realization产生真实fragments、loops、carry与accumulator graph；region summarizer中的contract保持显式，不能依赖attention-shaped algebraic recognition恢复；
- ordinary contraction与其唯一同dtype加法consumer可按DSL的局部融合语义合流：从当前def-use、零初值、结果坐标对应关系与dominance证明合法性，把另一add operand直接接入physical accumulator；不能跨数值cast或在serializer中隐式补做；
- sole-use浮点乘法与同dtype加法归约可按DSL的局部融合合同，在结构化source规范化时形成contract；从当前broadcast、归约轴和结果坐标关系确定paired/free/batch axes，复用既有contraction分块与lowering，不跨数值cast或改变作者的ordered control；
- boundary neutralization删除或简化实际validity/fill，不只删除padding record。

### 3.5 Predicate range narrowing 与 summary emptiness

Canonical coordinate-provenance analysis从`I.indices`、domain/subregion/index relation出发，组合helper calls、slice、broadcast、reshape/transpose、integer arithmetic与comparison。Blocking形成current physical fragment后，predicate-range analysis才把logical predicate投影到该fragment的source ranges：

- 输入必须是typed coordinate expressions、current physical ownership/ranges与structured-op identity/effect facts；
- 只有单调性、bounds与set inclusion被精确证明时，才可得到all-true、mixed与all-false连续区间；否则保留原遍历与predicate；
- all-true区间可去掉冗余validity，mixed区间必须保留原predicate，all-false区间只在summarizer/reduction对每个free lane产生identity且没有effect时才可跳过；
- rewrite必须真实改写physical loop bounds、access coordinates、active sets与validity SSA，但不改变KIR logical source relation。

All-false到identity的证明由局部typed value propagation完成：将predicate代入`select/mask`，再使用builtin reduce identity、zero contraction、pointwise constant folding和tuple/record逐component equality。它不识别summarizer名称或attention形状；任一component无法证明等于identity时，整个区间不得删除。

例如current query fragment是`q in [q_begin,q_end)`，predicate是`q >= k`时，`k in [0,q_begin)`是all-true，`[q_begin,q_end)`是mixed，`[q_end,K)`是all-false。这是coordinate/index-set rule，不是attention或causal kernel matcher。

Range narrowing之后，summary-emptiness analysis可证明某个physical traversal对每个free lane已先产生non-empty summary，或所选concrete combine graph不会让两个empty summaries相互combine。在保持NaN、identity、source order与empty-input语义的前提下，pass可从physical carry中删除optional/valid component及其selects。Canonical KIR中的total identity不因此被改写；证明失效或某路径可使empty summaries两两相合时保留validity。

### 3.6 Target-capability legalization

共同GPU verifier先检查provider-neutral legality。随后selected provider/hardware capability运行local checks与被真实差异逼出的extensions：

- grid rank与static fragment requirements；
- native structured-op/input dtype支持；
- descriptor/tile-index/storage/copy/sync forms；
- compile-time parameter与resource constraints。

不能表达的组合在最早拥有足够信息的层明确拒绝，不生成慢一个数量级的伪支持或等待JIT超时。

### 3.7 Deterministic serialization

Serializer只遍历已legalized current program并发出provider source。它不得调用canonical KernelModel、按result shape反推fragment、解析role字符串、创造workspace/loop/mask/grid或追加search parameters。

## 4. Semantic-preserving rewrites

Physical IR可以与KIR op graph不同，但变化必须属于KIR semantics与编译调用明确数值许可允许的实现。例如：

- 合并pure producer与consumer；
- rematerialize pure value；
- ordinary reduce可按其结合交换合同重结合与重排；scan和ordered region operations只按各自合同保留成员顺序地重结合；contract采用其独立数值合同允许的reassociation；
- 按DSL局部contraction-add规则合并零初值contraction与唯一同dtype加法consumer；
- 按DSL局部乘法归约规则将符合条件的浮点乘积sum规范化为contract；
- flatten/permutation paired contract axes；
- 创建blocking loops与fragment accumulators；
- 在明确的 normalized summary 数值许可及适用条件下形成 online 分段重标定；
- 从coordinate predicate证明all-true/all-false/mixed ranges，删除identity-only physical traversal或冗余summary validity；
- 把exact read→write关系映射成bulk transfer；
- 将program instances group、swizzle或grid-stride遍历。

不合法的变化包括：

- 把ordered recurrence改成reduce；
- 改变KIR combine、dtype、NaN/tie或atomic order；
- 修改logical subregion members；
- 改变kernel数量或引入隐藏跨kernel workspace；
- 把一个算法替换为另一个whole-operator template。

## 5. Verifiers

### 5.1 KIR verifier

验证programming-model与DSL定义的semantic legality；不检查GPU profitability或provider capability。

### 5.2 Construction verifier

验证KIR coverage、origin mapping与第一份完整physical program，规则见[`kir-to-gpu.md`](kir-to-gpu.md)。

### 5.3 Shared GPU verifier

每个shared pass后至少检查：

- program space、execution groups与ownership coverage；
- scalar/fragment shape与parameter expressions；
- structured control、dominance、loop carries与terminators；
- accesses、validity、fill、effects与alias/conflict；
- coordinate provenance、axis-map composition、active index-set subset与predicate/range proof；
- buffers、initialization/first-write obligation、lifetime与visibility；
- structured-op operands/results/accumulators；
- physical parameters均已声明并有合法domains；
- 无KIR operations、unresolved decision records或unknown executable fields。

### 5.4 Provider verifier

只检查共同program到selected surface的legality及local extension完整性，不重新决定shared structure。

### 5.5 Serializer verifier

Serialization前必须证明所有ops具有唯一provider spelling或已明确unsupported；不允许serializer fallback。

## 6. Side information 的合法范围

下面内容可以独立于executable SSA存在：

- immutable diagnostic origin/provenance；
- analysis cache；
- diagnostics；
- target capabilities；
- physical parameter declarations与candidate domains；
- benchmark/tuning artifacts。

Provenance/range analysis cache可以是side information；但影响执行的coordinate map、active member set、validity、narrowed loop range与identity elimination必须同时改写进current executable IR。

下面内容不得只存在于side records：

- program ownership与grid mapping；
- loop/range nesting；
- physical value shape/residency-required uses；
- pointer/index/validity；
- accumulator/replay/materialization；
- buffer allocation/lifetime；
- persistent/pipeline control structure；
- provider primitive operands/results。

## 7. Pass quality 的证据

一项compiler capability必须能展示：

1. pass前后真实IR差异；
2. legality与semantic-preservation条件；
3. 换一个kernel/shape后是否产生不同决定；
4. 不依赖触发kernel名称；
5. 受影响程序的数值与性能repro。

只改变attribute字符串或候选名字、却不改变当前program的改动，不构成新的physical compilation capability。
