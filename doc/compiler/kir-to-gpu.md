# KIR 到 GPU Program

## 1. Immutable semantic input

KIR 在进入 physical construction 后冻结。它唯一规定：

- kernel ABI、logical shapes、views、alias与effects；
- domains、source-derived subregions、indexed relations、logical identities、coordinate expressions与predicate/member sets；
- ordered control、unordered parallel、loop carry与停止条件；
- reduce/scan/region-fold/region-scan/contract family、histogram、atomic、RNG及其数值语义；
- 一个算法使用几个kernels以及每个kernel的边界。

GPU conversion可以分析这些事实并产生等价physical program，不能改变logical members、operation semantics、effect order或kernel数量。

## 2. Logical workset

GPU program instance的来源不是某个固定语法节点。Compiler首先构造logical worksets。

一个logical workset由有限instance domain和一组connected computations/effects组成，并满足：

1. 每个instance执行一次该组程序slice；
2. instances之间没有ordered data dependence；
3. 重叠effects只允许使用明确的scatter-reduction或atomic semantics；
4. 该组产生的values/effects具有唯一ownership；
5. 多个outputs与共享producers可以属于同一个workset。

Workset facts可以来自：

- 显式unordered `parallel` iteration；
- unique pointwise/tensor writes；
- reduce/scan/contract的batch与free axes；
- indexed write、atomic或scatter operation的iteration domain；
- def-use、alias、effect与dependence analysis证明的其它independent dimensions。

Ordered loop axes、reduction axes、scan prefix axes、strict recurrence与跨iteration mutable state留在program instance内部，不成为相互独立的program instances。

## 3. Execution group

不能把每个output element机械变成一个program：多输出可能共享producer，整tensor structured operation可能没有显式point loop，scan与recurrence也不能沿dependent axis拆开。

Construction把互相连接、必须共同拥有或存在依赖的worksets形成execution groups：

- 共享producer或共同buffer state的outputs保持在同一group；
- 没有依赖的groups可以拥有不同instance domains；
- 无法证明independence的程序slice退化为单instance group，并在内部保留ordered loops；
- cross-instance logical-buffer dependence使相关instances合并，除非KIR已经使用atomic/reduction semantics定义并发更新。

Execution group只是同一个physical kernel body中的内部dispatch region，不是kernel、artifact或launch。一个`@intent.kernel` specialization仍只产生一次launch；多个execution groups通过同一个physical program space执行，不形成隐藏kernels。多kernel算法由作者定义多个kernels并由host wrapper编排，这是一项Intent语义边界，不是对Triton/cuTile一般能力的描述。

## 4. Initial physical program

第一份GPU program使用保守但完整的统一映射：

1. 每个execution group只把launch前可得的有限instance domain按logical row-major order双射flatten为一个一维segment；
2. 每个segment的runtime长度`L_g`是由ABI scalar parameters、constexpr与已验证shape relations组成的typed launch expression；prefix offset `O_g = sum_{h<g} L_h`，kernel program-space长度是`sum_g L_g`，provider wrapper与kernel body从同一表达式绑定grid和dispatch；
3. `L_g == 0`产生空segment，不启动属于该group的program；若provider grid需要round-up，超出总长度的program由显式program-validity guard禁止执行effects；
4. 一个physical program id依据`O_g <= id < O_g + L_g`选择segment，再以`id - O_g`解码为该group的logical coordinates；
5. 需要先读取device tensor才能知道成员数的data-dependent/ragged domain不得决定launch cardinality。Initial conversion只把其launch-visible outer domain映射成program instances，data-derived subregion/members留在instance内部按序或按已有structured operation遍历；若outer extent本身也不可在launch前得到，该group先使用长度1的segment并在唯一instance内遍历整个domain。没有作者写下的额外kernel，也不允许compiler先生成prefix/count launch；
6. group body中所有dependent axes、runtime subregions和strict control保留为ordered structured loops；
7. scalar computation保持scalar，tensor computation先采用最小合法fragment或scalar loop；
8. 每次load/store/gather/scatter/atomic都立即产生显式access relation、typed coordinate SSA、source provenance dependencies、active member set与validity；
9. logical buffers立即产生allocation scope、initialization mode或first-write obligation、read/write、ownership与lifetime，不把这些事实留给emitter；
10. reduce/scan/region-fold/region-scan/contract立即成为可执行physical structured ops；未分块的版本可以慢，但不能只保留一个等待materializer解释的record。

若kernel只有一个无法拆分的ordered group，该group的segment长度为1，整个算法在一个program instance中按序执行。这是完整性结果，不是所有kernel的默认GPU策略。后续passes可以扩大一个physical program instance的logical ownership extent、重新group/swizzle program space或形成persistent traversal，但每一步都必须保持workset覆盖与effects。只有当新mapping能由launch-visible values或已有index relation直接计算时，pass才能把data-dependent member traversal提升为program space；不得为获得mapping引入隐藏launch。

## 5. Buffer 与 effect scope

KIR中的每次logical buffer allocation都有lexical allocation instance。Construction必须把它归入以下physical scope之一：

- **program-private**：每个拥有它的physical program instance各有一份；
- **iteration-private**：在声明所在ordered loop/region的每次动态执行中产生一份；
- **invocation workspace**：同一次kernel invocation中的多个program instances按显式ownership slices访问一份compiler-private resource；它作为hidden ABI argument由runtime分配，但不能引入额外launch；
- **external view**：由public ABI传入，allocation、initial contents与跨kernel lifetime由host程序拥有。

若KIR提供完整initial value，该initialization在每个logical allocation instance上恰好执行一次。若KIR创建未初始化buffer，construction不得合成默认初始化，须保留作者的先写后读义务及已有读写顺序；静态初始化证明不足不阻断construction或保序的storage lowering。依赖初始化事实的变换仍须证明其所需条件。若一个mutable buffer在program instances间形成非atomic dependence，保守construction必须把这些computations收回同一program instance；不能依赖kernel内不存在的global barrier。Invocation workspace只有在每个slice有唯一owner，或KIR已有scatter-reduction/atomic semantics时才能并行共享。

Atomic op保存KIR memory order、logical allocation/address relation、返回值schema及其logical sharing domain。Physical pass根据program mapping选择provider所需的scope；不得把scope写死在KIR，也不得丢失old-value/CAS success语义。

## 6. Multi-output 与不同 worksets

同一execution group可以从一个program instance写多个output relations。例如两个outputs共享一次normalization summary时，它们必须共享fragment SSA和program ownership。

互不依赖且shape不同的groups使用disjoint program-space segments。每个program只执行命中的group dispatch分支；该dispatch对单个program instance是uniform control。Group body内部由runtime data产生的`if`、validity与predication仍可diverge，且必须保持原有effects。后续profitability pass可以把兼容groups合并到同一program mapping，也可以保留初始union。

同一group中的每个output都保存独立的result relation与per-instance slice。共享producer只共享value/ownership，不要求outputs具有相同shape。某个output slice为空时，其write relation产生零effects，不会取消同一instance对其它outputs的计算。

这种表示保存一个kernel一次launch，同时不要求所有outputs伪装成同一logical shape。

## 7. Structured operations 的初始 mapping

- pointwise/unique write：result/free axes进入workset，初始每instance处理最小合法value slice；
- reduce：非reduction axes进入workset，reduction axes留在instance内部；
- scan：非scan axes进入workset，scan axis由physical scan或ordered carry处理；
- region fold：summary free axes进入workset；source axis留在instance内部，physical pass选择连续非空segments并在每段执行显式summarizer，再按summary combine合并；
- region scan：output/free axes进入workset；source axis留在instance内部，physical program显式保存segment summarizer、transition combine、incoming-state application、slice emitter与final-state flow；
- contract：batch与free axes进入workset，paired reduction axes留在instance内部；
- ordered recurrence：可证明independent的outer axes进入workset，state-carry axes留在instance内部；
- dynamic subregion：begin/end/source provenance成为runtime SSA和access validity，不改变program instance identity；
- helper-produced coordinate、slice、broadcast、reshape、transpose与predicate必须组合进同一physical coordinate/relation graph；某个pass若重算它们，必须真实改写SSA def-use，不能只留provenance ID或range record；
- scatter/atomic：source iteration可以成为workset，collision与ordering由operation semantics保存。

Ordinary `for/while`在physical IR中保存runtime condition/bounds、region arguments、loop-carried SSA、memory effects、`break/continue` edges与yields；compiler不需要证明终止，只需保持KIR控制语义。Reduce/scan保存physical axes、逐component identity、typed combine region与accumulator schema；scan另外保存inclusive/exclusive、direction与result relation。Region fold/scan还保存source slicing relation、summarizer region及显式captures；region scan保存transition combine、apply、emit、output assembly与final-state flow。Arg-reduce的tie/NaN规则、dynamic extent与所有stop conditions同样是显式operands/attributes/regions，不能由provider从op名称猜出。

## 8. Origin 与 semantic preservation

Construction为kernel、operation、value、region和block argument建立stable origin references。Origin只服务：

- conversion legality与semantic-preservation verification；
- diagnostics和generated-source attribution；
- 将immutable KIR facts投影到physical analysis的索引。

Physical program不得依赖origin去补执行结构或coordinate provenance。Coordinate expressions、axis maps、active member sets与validity在conversion时已成为current program的SSA/operation payload；origin只能检查这些mapping是否保持KIR语义。Combine regions、ordered control和其它执行所需semantic payload同样在conversion时lower成physical regions/operations；provider无需读取KIR才能执行或serialize。

Origin mapping至少区分KernelID、OpID、ValueID、RegionID和region argument position。字符串op name只能用于诊断，不能作为identity。

## 9. Construction verifier

Initial conversion完成后必须验证：

- 每个KIR observable effect恰好由一个physical execution slice覆盖；
- 每个logical result/member的ownership完整且无非语义定义的重叠；
- ordered dependencies没有跨独立program instances断开；
- unordered conflicts由unique、reduction或atomic semantics闭合；
- every access的coordinates、validity与fill/effect完整；
- every buffer保留initial value或作者的first-write obligation，读写的ordered dependencies完整；
- every structured op具有完整operands、results、regions与accumulator；
- every region fold/scan的segment source relation、summarizer、identity、combine以及scan apply/emit/result assembly完整，且没有segment identity/extent泄漏为KIR observable value；
- all physical values具有合法scalar/fragment types；
- every runtime program-space extent只依赖launch-visible values，data-derived member domains仍有完整的program-internal traversal；
- every buffer的allocation scope、instance identity、initialization mode/coverage、ownership与visibility完整；
- every dynamic/indexed access由explicit guard、已验证relation或canonical `assume_in_bounds`证明在目标resource范围内；
- every physical coordinate expression具有source identity/rank与完整SSA provenance，slice/broadcast/reshape/transpose/helper-call composition与KIR relation等价；
- every active-member/range narrowing都是原relation的显式subset，并具有typed predicate/range proof；不能从shape、op name或origin side record重建；
- every atomic op保存order、logical sharing domain与result semantics，并能从physical mapping得到合法provider scope；
- physical program不需要KIR或side records才能解释执行。
