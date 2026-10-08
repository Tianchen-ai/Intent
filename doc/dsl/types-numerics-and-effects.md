# Types、Numerics 与 Effects

## 1. Canonical value types

普通logical values使用：

- `bool`；
- signed 64-bit logical `index`；
- `i8/i16/i32/i64`；
- `u8/u16/u32/u64`；
- `f16/bf16/f32/f64`；
- 明确定义bit encoding的`f8e4m3fn/f8e5m2`；
- typed tuples/records；
- ranked tensor values；
- external views与kernel-local logical buffers。

physical lane shape、padding、layout、address width和target encoding不进入canonical value type。

Tuple和record是immutable structural product types，不是ranked tensor或memory layout。Tuple type由固定component序列定义；record type由有序、唯一的field names及逐字段value types定义。它们可包含不同shape/dtype的tensor components，也可嵌套；不提供统一`.shape`或`.dtype`。它们可作为SSA、helper result、loop carry和structured accumulator，但不直接进入logical buffer、public kernel ABI或external-view element type。Logical buffer只保存ranked tensor中的单一scalar element dtype。

`i4/u4/fp4`不作为普通可寻址tensor element type。普通packed data使用`u8/u16/u32` carrier，加显式index、shift、mask、sign extension与scale arithmetic。FP4等microscaling formats由scaled-contract schema定义。

闭合的 Q4_K/Q8_K [量化计算](quantized-operations.md)同样使用普通 u8 carrier，
其 record 格式与数值由独立 quantize/quantized-dot schema 定义，不是新 scalar dtype。

## 2. Logical index 与 shape values

logical `index`使用signed 64-bit arithmetic。physical address width由compiler根据shape、stride与bounds证明后选择，不能改变logical index结果。

dynamic extent是有identity的runtime shape value。不同dynamic extents只有在来自同一value或operation产生明确equality relation时才兼容。

tensor rank与logical extents来自domains/subregions和shape transforms。broadcast、reshape与join的规则见[`logical-program.md`](../programming-model/logical-program.md)。

Ranked tensor与external view的`.shape`是logical extent tuple；dynamic members保留shape identity。`I.full(shape, fill, dtype)`要求每个extent非负，把scalar `fill`按本文的literal/cast规则实例化为`dtype`并广播到所有elements；zero extent产生empty tensor。`value.shape`可直接作为`I.full`的shape，但不转化成physical fragment shape。

`I.Enum`是constexpr-only closed named type。成员及其唯一值只参与specialization identity、同类型比较与硬件无关constexpr control；不同enum不隐式比较，enumeration不进入runtime scalar/tensor/buffer/view ABI，不允许算术、cast或bitcast。Python surface可使用`IntEnum`-like declaration convenience，但其underlying Python integer representation不是canonical ABI事实。

## 3. Literals 与 promotion

Python literal是untyped source literal，可以按直接使用位置的expected dtype实例化，前提是其值可表示。首次形成runtime value而没有expected dtype时，Python `bool/int/float`分别实例化为`bool/i64/f64`；后续使用不反向改变已形成value的dtype。

Pointwise表达式在frontend确定结果dtype，再以显式typed operations进入KIR；不让provider默认promotion重新决定逻辑结果。下面的混合runtime dtype显式转换规则是本surface的数值选择，不是Triton或typed IR的必然要求。跨provider一致性来自统一的语言规则及其lowering，不要求所有语言都禁止implicit promotion：

- dtype相同可直接运算；
- dtype不同必须显式`I.cast`；
- comparison返回`bool`；
- structured operation的accumulator/result dtype由operation显式给出或使用本文定义的固定builtin规则。

`reduce.sum`与`cumsum`在未显式给出accumulator dtype时使用：

- `i8/i16 -> i32`，`u8/u16 -> u32`；
- `i32/i64/u32/u64`保持输入dtype；
- `f8e4m3fn/f8e5m2/f16/bf16 -> f32`；
- `f32/f64`保持输入dtype。

其它builtin reduce与cummax默认保持输入dtype，除非surface明确要求另一result schema。Dot/matvec/vecmat/matmul保持与contract相同的显式accumulator/result dtype，不从provider推导输入精度；outer使用普通同dtype乘法。

External view的storage dtype、进入运算的operand dtype、accumulator/result dtype与provider内部instruction dtype是不同层次。Builtin `acc_dtype`及本节固定widening由frontend插入转换，不要求作者先转换整个external tensor；generic reduce/scan则要求进入combine的各component与对应identity/result dtype一致。后端可以用不同的内部计算表示实现同一数值合同，不要求每条机器指令的dtype与logical value逐一相同。

## 4. Integer arithmetic

fixed-width integers使用二进制补码与modulo arithmetic：

- add/sub/mul按`2^width` wrap；
- signed `//`与`%`使用Python floor-division relation：`a == q*b + r`，`q=floor(a/b)`，非零`b`时`r`与`b`同号或为零；
- unsigned `//`与`%`使用Euclidean quotient/remainder：`a == q*b + r`且`0 <= r < b`；
- integer operands不重载`/`；需要floating division时必须先显式cast；
- signed right shift是arithmetic shift，unsigned right shift是logical shift；
- left shift保留低`width` bits；
- bitwise operations作用于固定宽度bit pattern；
- shift amount必须满足`0 <= amount < width`；
- division by zero与`signed_min // -1`是非法输入。

`%`只用于integer remainder。floating remainder必须使用具有独立、明确定义语义的named pointwise operation，不能继承Python、C或provider对`%`的偶然解释。

这些规则由每个target机械兑现，不继承C/CUDA/Triton的偶然差异。

## 5. Floating point、cast 与 bitcast

普通floating operations遵循对应格式的IEEE-754值与round-to-nearest-even。除本节规定的乘加融合、作者使用structured operation或显式approximate math外，compiler保持source expression的数据依赖与求值关系，不启用会改变结果集合的隐式fast-math。

`I.lgamma(x)` 的数学定义为 `log|Gamma(x)|`。在 1、2 处返回 `+0`；
在非正整数极点（包括正负零）和正负无穷处返回 `+inf`；NaN 输入返回 NaN。
结果使用原浮点 dtype，仍受该格式的表示范围与普通转换规则约束。

`I.asin(x)` 计算反正弦主值，结果位于 `[-pi/2, pi/2]`，保留 signed zero；`abs(x) > 1`（包括无穷）或 NaN 输入返回 NaN。它保持输入的浮点 dtype 与 shape，采用数学库精度，不承诺 correctly-rounded 或内部近似式的逐步舍入结果。低于 f32 精度的输入按 f32 求值后遵循普通 cast 规则返回原 dtype；f32/f64 使用对应精度求值。特殊值规则对齐 [libdevice asin](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinf.html)。

`I.log1p(x)`、`I.erfc(x)`、`I.i0(x)` 是 pure 逐元素数学库函数，保持输入的浮点 dtype 与 shape：

- `log1p` 计算 `log(1+x)`，不按先舍入 `1+x` 再取对数的普通运算组合定义；保留 signed zero，`-1` 返回 `-inf`，小于 `-1` 返回 NaN，`+inf` 返回 `+inf`。
- `erfc` 计算互补误差函数，不按先舍入 `erf(x)` 再从 1 相减的普通运算组合定义；正负零返回 1，`+inf` 返回 `+0`，`-inf` 返回 2。
- `i0` 计算第一类零阶修正贝塞尔函数，是偶函数；正负零返回 1，正负无穷返回 `+inf`，有限大值受结果格式的溢出规则约束。

三者均传播 NaN。它们采用数学库精度，不承诺 correctly-rounded 或内部近似式的逐步舍入结果。`f32/f64` 相对对应格式 RN-even 结果的最大 ULP 误差分别为：`log1p` 1/1、`erfc` 4/5、`i0` 6/6；更低精度浮点输入按 f32 求值后遵循普通 cast 规则返回原 dtype。这些界限参照 [CUDA 数学库精度说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#mathematical-functions-appendix)，不改变相邻普通算术、显式近似或 FTZ 的合同。

### 5.1 显式近似数学

`fdiv`、`exp2` 和 `tanh` 的 `approximate` 是操作本身的语义，不是优化 hint；默认 `False` 保持普通运算。`fdiv/exp2` 另外接受 `flush_to_zero`，仅允许在 `approximate=True` 时启用。两个参数必须是 constexpr bool，非默认模式的 operands/result 都是 `f32`，不做隐式 dtype 转换。这是一个闭合的逐操作能力，不额外改变相邻普通运算、contraction或reduction的数值合同。

近似模式采用以下跨 target 的数值契约，不承诺 correctly-rounded 或不同 target bitwise 相同：

- `exp2`：相对正确舍入结果的最大误差为 2 ULP；`-inf`、`+inf`、NaN 分别产生 `+0`、`+inf`、NaN，任一符号的零产生 1。
- `tanh`：有限非零结果的最大相对误差为 `2^-11`；保留 signed zero，正/负无穷产生正/负 1，NaN 产生 NaN，subnormal 输入保持其近零值。
- `fdiv`：使用近似倒数与乘法的除法语义。对 `2^-126 <= abs(rhs) <= 2^126`，最大误差为 2 ULP；对 `2^126 < abs(rhs) < 2^128`，有限 lhs 产生按商符号的零，无穷 lhs 产生 NaN。其余特殊值遵循对应除法分类，不把 NaN 当成有限近似结果。

`flush_to_zero=True` 在选中操作的输入和输出边界把 subnormal 转为保留符号的零；不修改相邻操作或 memory 中的数据。不支持非默认模式的 provider/hardware 必须明确拒绝。Canonical KIR 与 shared GPU unary/binary operations 保存这两个 typed bool attributes；CSE、克隆、重算和所有 lowering 必须保留其区别，不以 plain operator kind 代替完整数值语义。

以上近似界限采用公开 [PTX 浮点指令语义](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions) 的闭合精度范围作为语言契约；其它硬件实现也必须满足它，不能用各 provider 的默认行为重新定义。

### 5.2 Cast 与 bitcast

`I.cast`定义：

- integer→integer先把source解释为数学整数，再对`2^destination_width`取模，最后按destination signedness解释该bit pattern；同signedness widening因此分别等价于sign/zero extension，narrowing等价于保留低bits；
- integer→float使用round-to-nearest-even；
- float→integer向零截断，NaN、infinity或越界是非法输入；
- float narrowing到`bf16/f16/f32`使用round-to-nearest-even，有限溢出产生对应infinity；到8-bit float使用下述format-specific规则；
- saturating conversion使用单独、明确的saturating operation，不是普通cast的隐藏模式。

`I.bitcast`要求source/result总bitwidth相同，只保存bit pattern，不进行数值转换或storage repacking。

`I.maximum/I.minimum`传播NaN；忽略单侧NaN的`maximum_num/minimum_num`是不同pointwise operations，不能依赖target默认行为。普通float cast与IEEE arithmetic保留signed zero；NaN result只保证仍是NaN，不把payload、signaling状态或NaN sign定义为可观察语义。

两种8-bit float的logical encoding固定为：

- `f8e4m3fn`：1 sign bit、4 exponent bits、3 fraction bits、bias 7、支持subnormal、无infinity；`S.1111.111`是NaN，其余编码按finite E4M3FN解释，最大finite magnitude为448。普通cast使用round-to-nearest-even，有限输入超出可表示范围是非法输入；需要clamp时使用显式saturating conversion；
- `f8e5m2`：1 sign bit、5 exponent bits、2 fraction bits、bias 15、支持subnormal；exponent全1且fraction为0表示infinity，fraction非零表示NaN。普通cast使用round-to-nearest-even，finite overflow产生对应infinity。

### 5.3 局部 contraction-add 融合

普通`contract`的零初值结果只有一个加法consumer，且另一operand `C`与contraction accumulator/result具有相同dtype和逐元素对应的结果关系时，`contract(A, B) + C`或`C + contract(A, B)`允许实现为以`C`为初值的contraction累加。该局部组合采用融合累加的舍入与特殊值语义，不承诺先独立舍入完整contraction再做add；空reduction的融合结果为`C`。Compiler必须保持lhs/rhs输入精度、paired axes、accumulator dtype与所有外部effects，不引入TF32、FTZ或其它近似选择。

该许可不依赖target或kernel identity，不需要新的作者hint或tuning参数。它不覆盖跨数值cast、多个consumer、非零初值contraction的再次重结合，也不授权任意重结合或未声明的近似运算。纯shape projection只有在逐元素对应关系保持不变时才能随融合改写。

### 5.4 局部乘法归约融合

浮点乘法结果仅供一个以加法零为identity、以同dtype普通加法为combine的reduce使用，且乘法到归约之间没有数值cast、乘积与accumulator/result的dtype相同时，允许将该组合实现为普通contract。纯shape projection必须保持逐元素关系；paired reduction、batch与free axes必须由原broadcast和归约关系确定，不改变参与归约的成员、结果坐标或effects。默认widening引入的数值cast不属于此许可，ordered loop与scan也不属于此规则。

融合保持lhs/rhs输入精度与accumulator dtype，不启用TF32、FTZ或其它近似。该局部组合采用contract的融合乘加数值语义：允许FMA改变舍入，也允许极端溢出或抵消时的Inf/NaN结果不同于先舍入乘积再归约；空归约仍返回原dtype的加法零。该规则不额外扩大其它普通运算的数值许可，不需要新增作者hint或tuning参数。

### 5.5 普通乘加的FMA融合

普通同dtype浮点乘加表达式，如`a*b+c`、`c+a*b`、`a*b-c`和`c-a*b`，允许使用原生FMA完成一次round-to-nearest-even舍入，不要求先独立舍入乘积再执行加减。融合可以改变末位、signed-zero以及极端溢出或抵消时的Inf/NaN结果；未融合的使用处仍遵循对应普通运算的规则。该许可不要求乘法结果只有一个consumer，但不能因一个使用处的融合而改变其它使用处所需的独立乘法结果。

Compiler保持operand/result dtype、输入精度、数值cast、循环迭代顺序与external effects。该许可只覆盖乘加融合，不授权普通加法或乘法的任意重结合、输入降精度、TF32、FTZ或未声明的近似数学。是否融合由provider选择，不增加作者hint、调优参数或另一套算法；不同合法实现不保证逐操作bitwise一致。

Operation显式规定的独立舍入边界优先于本节，例如[量化计算](quantized-operations.md)中每条record内的`R32`序列；不得跨过这些边界形成FMA。

### 5.6 编译调用的数值许可

编译调用的 `CompileOptions.numerics` 默认为 `source`：保留本文已有的普通浮点、FMA、局部融合及各 structured operation 的合同。它不要求逐操作 bitwise 相同，也不自动增加其它重结合权限。

`relaxed_normalization` 是显式选择的额外许可，仅针对同一成员域上由 validity、maximum、指数权重之和及加权 contraction 构成的 normalized summary。它允许沿该域连续分段，以局部 maximum 计算权重，再用最大值差的指数缩放并合并部分 mass/moment。该许可包括低精度 weight cast 的归一化参考从全域 maximum 改为分段 maximum，以及相应部分累加和重缩放带来的有限精度差异。

选择此模式的调用方必须保证：valid member 的 score 均为有限值；所有实际参与加权 contraction 的 value 元素均为有限值，包括权重为零仍参与乘法的元素。真正未发生的 masked memory access 不新增该要求。编译器不额外读取输入或插入运行时 finite 检查；违反前置条件的输入不在该模式的数值保证内。`source` 模式没有这一额外前置条件。

在此前提下，允许上述具体重组产生的舍入、下溢及中间溢出差异；不承诺与全域归一化后的低精度权重逐元素相同。空成员域和全 inactive 域仍产生原 summary identity，member 集合、结果坐标、外部 effects、ABI 和声明的 accumulator/result dtype 必须保持。该模式不授权其它普通表达式的任意重结合，不引入 TF32、FTZ 或未声明的近似数学，也不跨越量化格式独立规定的舍入边界。

数值许可与优化选择分开：`online_reduction=False` 只禁用这项可选改写，不改变 source 操作的定义；启用也不保证一定采用。作者显式提供 `region_fold/region_scan` 的 summarize/combine 仍按第 6 节合同实现，不依赖此额外模式。

## 6. Reduce 与 scan 数值语义

Generic reduce是并行归约。作者选择该operation，即声明combine具有结合、交换及identity中立的算法合同；compiler可以选择parenthesization与element permutation，不保证logical source order或ordinary left fold。对于floating-point，这项许可接受并行归约树与重排带来的finite-precision差异，不要求combine逐bit满足实数代数等式；它不改变输入成员、声明的accumulator/result dtype、NaN/tie规则或effects，也不授权未声明的TF32、FTZ或其它近似。

Scan定义每个logical prefix。它要求combine可结合，但不要求可交换；允许保持prefix内source order的parenthesization，不允许重排prefix成员或改变各prefix的成员集合。严格left fold、不可重结合的recurrence及ordered effects使用ordinary loop。

这些代数性质是作者选择operation时承担的前置条件；compiler验证typed schema、axes、purity、captures与effects，不为每个自定义combine重新证明结合律或交换律。Source component可以含被归约axes；删除这些axes后的component shape是accumulator、identity、combine参数与result shape。

Generic reduce/scan的identity逐component显式给出；builtin reduce/prefix由操作定义产生同一canonical identity，不要求作者重复传入：

- empty reduce返回identity；
- empty scan返回empty tensor；
- inclusive/exclusive与forward/reverse由scan参数决定。

NaN与tie behavior来自明确combine。`reduce.max`与`cummax`使用propagating maximum；其identity为dtype最小值：有infinity的float取负无穷，f8e4m3fn取-448，signed integer/index取最小整数，unsigned integer取0。Sum/cumsum的identity为result dtype的加法零，bool any/all分别为false/true。`arg_reduce.max`在values相等时选择lowest logical index，并传播NaN。其它策略必须通过不同typed combine明确写出。

Region fold/scan沿compiler-selected连续source slices允许logical-order-preserving reassociation，不继承ordinary reduce的permutation许可。它们要求region summarizer与summary combine满足：

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

Compiler不证明这些代数定律；作者选择operation即声明它们成立。Verifier必须检查source-axis一致、summary/identity/combine schema、captures、purity、output relation及禁止的effects。

`identity`必须在本节定义的float/NaN语义下真正中立。把`maximum=-inf`、其它components为零的record无条件送进包含`exp(maximum - merged_maximum)`的combine，会在`-inf - -inf`处产生NaN，因此不是合法identity。此类summary必须携带显式validity，并在执行指数运算前把invalid分支规范化为有限差值；或者使用另一种能够证明双侧中立的typed表示。Empty source的region fold返回identity；empty region scan返回empty output与initial state。

Region summarizer不能观察compiler-selected segment identity或extent。它若需要位置，必须消费由source axis产生的absolute logical coordinates；这些coordinates切片后不重新编号。Region scan除summary homomorphism外，还要求`apply(identity,state) == state`，以及整段emit等于按相邻slices分段emit，后一段消费前一段summary作用后的incoming state。Floating-point region fold/scan接受由合法segmentation与parenthesization造成的舍入差异，但不允许改变source order、NaN policy、accumulator dtype或approximation contract。

## 7. Scaled、sparse 与 histogram numerics

scaled-contract schema定义logical element format、carrier packing、scale encoding、group relation、rounding、special values与accumulator dtype。普通target不能以native primitive限制反向缩窄该语义。

本规范的canonical microscaling formats定义如下：

- `e2m1` logical element占4 bits，sign为bit 3；正值编码`0..7`依次表示`0, 0.5, 1, 1.5, 2, 3, 4, 6`，sign bit取反数值符号；无NaN或infinity。两个elements按logical reduction coordinate递增顺序打包到一个`u8`，较小coordinate使用低nibble；
- `e4m3`采用上文`f8e4m3fn`的8-bit encoding；
- `e8m0` scale是8-bit无sign exponent，code `0..254`表示`2^(code-127)`，code `255`表示NaN scale。

group size是正logical index。Scaled contract的closed positional schema为lhs carrier/scale `[M,G,C_lhs]`/`[M,G]`与rhs carrier/scale `[G,C_rhs,N]`/`[N,G]`；两侧使用同一个`G`与相同group size，reduction固定配对`G`和carrier-inner axes。沿flatten后的logical K，coordinate `k`读取group `k // group_size`的scale；`C_lhs/C_rhs`分别等于`group_size / elements_per_carrier(format)`。Decoded logical element先乘对应scale，再进入paired-axis multiply-add。其它rank或axis order必须由作者用普通logical transforms显式归一，shared/provider不得从shape猜另一group relation。

sparse-contract schema定义compressed ordering、metadata到logical reduction positions的解释、invalid metadata与accumulator。canonical metadata是logical positions，不直接使用某个instruction要求的packed metadata layout：

- `one_of_two`要求logical compression axis extent可被2整除。每组2个logical positions保存1个compressed value；metadata是每组一个logical `index`，合法值为`0`或`1`，指明该value位于组内哪个position；
- `two_of_four`要求logical compression axis extent可被4整除。每组4个logical positions保存2个compressed values，按position递增顺序存放；metadata是每组一个typed record `{first: index, second: index}`，要求`0 <= first < second < 4`。

其它logical positions为加法零。若external input使用packed metadata carrier，作者以普通bit/index arithmetic把它解释成上述logical positions；这不会物化dense operand，也不会丢失structured sparsity。非法metadata或不满足group整除关系是调用方错误；provider为native sparse instruction重新编码metadata属于physical representation，不能改变上述logical mapping。

histogram要求`values`是integer tensor，`bins`是正logical index，`valid`是bool scalar或可broadcast到`values.shape`的bool tensor。每个active value必须位于`[0,bins)`并贡献一次；inactive lane既不贡献，也不要求其value在range内。result shape为`(bins,)`。count dtype显式给出，overflow使用该integer dtype的wrap语义，empty input返回全零。

## 8. External views

external view定义：

- element dtype；
- logical shape；
- base allocation identity与element offset；
- runtime strides，单位是elements而不是bytes；
- `In/Out/InOut`访问方向；
- bounds与必要alias relation。

对logical coordinate tuple `c`，element address是`base_allocation[offset + sum(c_i * stride_i)]`。negative stride合法。zero stride对只读broadcast view合法；可写view只有在对应write/collision semantics允许地址重复时才合法。每个active logical coordinate都必须映射到有效allocation；inactive access不形成地址，语言也不隐式clamp。

不同view参数默认可以alias。`noalias`若出现，是调用方必须满足的语义前置条件，不是performance hint。alignment、contiguity、vector width、descriptor/TMA eligibility由runtime/compiler/provider处理，除非某个外部数据格式的正确解释本身要求特定alignment。

## 9. Logical buffer

调用形式为 `I.buffer(shape, dtype, init=None)`。`shape` 是逻辑维度 tuple，
`dtype` 是 Intent dtype；两者必填。省略 `init` 或传入 `None` 创建未初始化
buffer；`I.buffer(values.shape, I.f32, init=values)` 显式提供初值。

logical buffer是kernel-local mutable state：

- shape与dtype由作者定义；
- 新buffer不alias external views或其它新buffer；
- 可以用完整initial value创建，也可以未初始化创建；
- 作者必须保证每次active read之前，对应element已在同一allocation instance中写入；
- lifetime是lexical，buffer不能逃逸到kernel外；
- storage、materialization与placement由compiler决定。

跨kernel存活的tensor由host显式拥有，不是logical buffer。

静态初始化证明不足不作为编译拒绝条件，也不代表element已经初始化。
依赖初始化事实的优化仍须证明其所需条件；普通lowering保持作者的读写顺序和effects。
读取尚未写入的element违反语言约定，编译成功不保证程序满足先写后读义务。

## 10. Indexed access 与 collision semantics

所有access使用typed index relation与active validity。relation保存source identity/rank、result logical axes、每个source axis的typed coordinate expression，以及这些expressions对domain/subregion/data-derived indices的SSA provenance。relation composition必须组合这些expressions，不能只保存result shape。invalid read不产生memory access并返回同dtype、可broadcast到result shape的显式fill；invalid write不产生effect。

canonical effects区分：

- external read/write；
- logical-buffer read/write；
- arbitrary-index unique store；
- `scatter_reduce`；
- atomic load/store/RMW/CAS。

unique store要求active destination mapping可证明injective。`scatter_reduce`使用typed pure combine定义重复地址的合并，不承诺每次更新的linearization或old value。ordinary ordered control中的effects保持program order；unordered parallel iterations之间若可能访问同一地址，必须由unique、scatter-reduction或atomic semantics消解，否则程序非法。

plain logical copy使用read→immutable SSA value→write表达，不建立独立canonical op。non-atomic read/write/gather/scatter不带memory order或physical scope。

## 11. Atomic memory semantics

canonical atomic operations是：

- `atomic_load`；
- `atomic_store`；
- `atomic_rmw(kind=exchange/add/max/min/and/or/xor)`；
- `atomic_compare_exchange`。

每个logical atomic address具有单一modification order。RMW不可分割并返回old value；CAS返回typed record `{old_value, success}`。

memory order定义：

- `relaxed`：只保证该atomic object的atomicity与modification order；
- `acquire`：读取matching release/release sequence后建立happens-before；
- `release`：发布该operation之前的ordinary/atomic effects；
- `acq_rel`：同时具有acquire与release。

load只允许`relaxed/acquire`；store只允许`relaxed/release`；RMW与CAS允许全部四种。CAS failure order由success order机械派生：`relaxed -> relaxed`、`acquire -> acquire`、`release -> relaxed`、`acq_rel -> acquire`。

atomic operation没有public/canonical `scope`。同一kernel invocation中，通过同一logical allocation/address relation访问该atomic object的executions是逻辑参与者；compiler在target physical program中选择正确scope。

普通conflicting non-atomic accesses是非法data race。语言不保证parallel iterations同时驻留，也不保证依赖spin-wait的程序取得forward progress。

## 12. Deterministic RNG

canonical RNG是pure stateless Philox4x32-10 random-bits operation。`seed`是`u64`，`logical_counter`是非负`u64`或可无损转换为`u64`的logical index；`I.random.bits`返回一个`u32`：

```text
block = logical_counter // 4
word  = logical_counter % 4
(c0, c1, c2, c3) = (low32(block), high32(block), 0, 0)
(k0, k1) = (low32(seed), high32(seed))

repeat 10 rounds:
    (hi0, lo0) = mul_wide_u32(0xD2511F53, c0)
    (hi1, lo1) = mul_wide_u32(0xCD9E8D57, c2)
    (c0, c1, c2, c3) = (hi1 ^ c1 ^ k0, lo1,
                         hi0 ^ c3 ^ k1, lo0)
    k0 = k0 + 0x9E3779B9  (mod 2^32)
    k1 = k1 + 0xBB67AE85  (mod 2^32)

result = (c0, c1, c2, c3)[word]
```

它不读取target、program、thread、lane或调用序号。`uniform(..., dtype=f32)`固定为`f32(bits >> 8) * 2^-24`，结果位于`[0,1)`；normal等复合distribution使用普通DSL helper构造。provider只有在证明native RNG产生同一bit stream时才能替换该integer computation。

## 13. Closed typed input constraints

当前语言只提供两类closed typed输入约束：

- `I.assume_in_bounds(index, view_or_buffer, axis=...)`声明给定typed index relation在指定resource axis内；
- external view annotation中的`alias/noalias`约束base allocation relation。

`I.assume_in_bounds`是无返回值的约束语句，不产生新的index或tensor。调用后继续使用原`index`，不要把调用结果赋给索引变量。

语言不提供接受任意bool expression的`assume`，也不把alignment、contiguity、sorted/unique、shape equality或encoded-format validity作为优化hint。将来只有真实kernel无法由现有类型、relation和operations表达时才增加新的约束；新增项必须具有closed typed schema、明确违反语义以及唯一canonical KIR表示，不能成为开放式布尔断言或授权优化的hint。
