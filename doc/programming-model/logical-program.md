# Logical Program

## 1. Domain、subregion 与 identity

domain 表示有限、有序的 logical coordinate set。基本 domain 是半开整数序列：

```text
[begin, end), step > 0
```

空 domain 合法；begin/end/step 可以来自 runtime scalars。logical `index` 的数值行为跨 target 唯一，不继承 provider 的整数差异。

连续 subregion 从 source domain 取得：

```python
axis = I.domain(0, N)
prefix = axis[0:valid_length]
window = axis[begin:end]
```

它定义 source 中的半开区间 `[begin,end)`，必须满足 `source.begin <= begin <= end <= source.end`。`begin == end` 得到空 subregion；越界不会隐式 clamp。subregion 保留 source identity、bounds 与 provenance：

- 迭代得到 source logical coordinate，不重新编号为局部 ordinal；
- `I.indices(subregion)` 返回这些 source coordinates；
- 局部 ordinal 由 `enumerate` 或显式 ordinal domain表达；
- physical padding 不改变 logical members。

非连续、重复或重排的成员使用 indexed relation，不冒充 subregion。只有作者边界、输入 relation 或算法 metadata 能在 Kernel IR 中产生 subregion；compiler blocking 只存在于 target physical program。

### 1.1 Coordinate provenance

`I.indices(axis_or_subregion)`产生logical `index` tensor。每个result element的canonical relation同时保存：

- source domain identity、source axis/rank与logical coordinate dtype；
- 从result logical axes到source coordinate的typed expression；
- 该expression对domain、subregion、indexed values与predicate的SSA dependencies；
- active member set与已知bounds。

Provenance是canonical value/relation的一部分，不是字符串axis name或只供调试的origin。`@intent.fn`调用必须逐component传递它；helper inline与非inlined call representation得到同一结果。Slice组合source bounds但不重新编号coordinate。Tuple/record construction只分组components，不丢失各component provenance。

Broadcast保存显式axis map；transpose/permute组合axis permutation；reshape通过logical row-major linear coordinate组合old/new axis maps。对coordinate values的integer arithmetic、comparison与select保留可表示的typed expression及其dependencies。某个operation无法使用canonical expression精确表示时，numerical value仍正确，但coordinate/range analysis必须返回unknown，不得从shape、名称或附近结构猜测。

## 2. Shape 与 tensor values

tensor value 的每个 dynamic extent 是有 identity 的 runtime shape value。两个未知 extent 不会因为“都是 dynamic”而自动相等；相等只能来自同一 value或operation明确产生的shape relation。

pointwise surface 允许 scalar 与 size-one broadcasting，frontend 将其归一为显式 canonical broadcast relation：

- tensor axes 从尾部对齐；
- 对齐维度必须相等，或其中一个为 1；
- `0` 与 `1` broadcast 为 `0`，`0` 与其它正 extent 不兼容；
- runtime equality/size-one condition 必须保留，不能按静态 shape 猜测。

`reshape` 保持 logical row-major element order与元素总数；最多一个 inferred extent，并且只有结果唯一时合法。它不改变 dtype、bit representation 或 physical storage。`transpose/permute` 明确给出 logical axis permutation。这些operations的result relation必须组合上节的coordinate maps，不能只保存result shape。

`join(lhs,rhs)` 是唯一的 two-input value construction：两个输入 dtype 与 logical shape相同，结果增加一个 trailing logical axis：

```text
result[..., 0] = lhs[...]
result[..., 1] = rhs[...]
shape(result) = shape(lhs) + [2]
```

它不是 record、concatenate 或一般 interleave。随后 reshape 所产生的 interleave 由上述 element mapping 与 logical reshape order共同决定。

## 3. Unordered parallel iteration

```python
for point in I.parallel(domain):
    ...
```

定义 unordered forall：iteration space 中每个 logical point 执行一次，但 points 之间没有可观察顺序。

- 不允许跨 iteration loop-carried values；
- 非 atomic/reduction 的冲突 effects 使程序非法；
- `break` 非法，`continue` 只跳过当前 logical iteration；
- 不指定 program count、thread、lane、vector width、tile 或真实同时执行。

compiler 可以 serial、flatten、thread 或 vectorize该语义。纯整域 tensor expression已经具有唯一写入时，compiler也可以从数据流推导等价 data parallelism，作者不必额外包一层 `parallel`。

## 4. Ordered control 与 carry

普通 `if`、`for` 和 `while` 保留程序顺序：

- scalar bool 控制 `if`，分支值通过 SSA merge合并；
- 普通 domain iteration 按 logical order执行；
- 循环更新的 values 是 loop carry；
- `break`、`continue` 和 early return按结构化控制语义归一。

普通有序程序不需要 `ordered` 标记。compiler只有在 dependence/effect analysis证明结果相同后才能改变它的执行组织。

## 5. Recurrence、homomorphic region operations 与 ordered control

Intent 不提供允许 arbitrary body 随 compiler-selected extent 重新分段的 `state_stream`。这类构造没有说明改变分段后为什么仍是同一个程序：body 可以观察调用次数、tail、局部 shape 与 effects，因而一般不具备唯一语义。

语言区分五种结构：

1. 每个 logical element 已经是 summary，combine可结合、可交换，只需要最终可重结合与重排的结果：`reduce`；
2. 每个 logical element 已经是 summary，combine可结合，需要保持成员顺序的每个 logical prefix：`scan`；
3. 作者定义“任意连续 source slice 怎样产生 summary”，只需要最终 summary：`region_fold`；
4. 作者同时定义 slice summary、summary composition、incoming state application 与 slice output，需要每个 logical position 的结果：`region_scan`；
5. 更新依赖严格顺序、动态停止、非结合 state或 ordered effects：普通 `for/while` 与 loop carry。

`region_fold`作用于一个有序 source axis。对任意保持顺序、完整覆盖该axis的连续分段`R0 ... Rn`，作者提供：

```text
summarize(Ri) -> Summary
combine(Summary, Summary) -> Summary
identity : Summary
```

并要求：

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

其中`A`、`B`是相邻且保持source顺序的连续slices。`combine`允许保持logical order的任意parenthesization，不允许permutation。作者选择该operation，即声明这些等式属于算法定义；compiler不证明数学结合律，但验证types、schema、purity、effects与source-axis关系。

`summarize`接收沿同一source axis切出的tensor components以及显式captures。它可以包含pure tensor operations，包括reduce、scan与contract；不得包含external/logical-buffer write、scatter、atomic、RNG或其它可观察effect。它不能读取segment ordinal、segment count、chosen extent或chunk-relative coordinate。需要坐标时，作者把`I.indices(source_axis)`作为source component传入；切片后仍是absolute source coordinates。

`region_scan`使用相同的summary algebra，但还显式提供：

```text
apply(prefix_summary, initial_state) -> incoming_state
emit(source_slice, incoming_state, captures) -> output_slice
```

Transition identity与composition必须对state构成合法action：

```text
apply(identity, state) == state
apply(combine(a, b), state) == apply(b, apply(a, state))
```

对相邻slices `A`、`B`，还必须有：

```text
emit(A ++ B, state)
  == concat(emit(A, state),
            emit(B, apply(summarize(A), state)))
```

`summarize`/`combine`/`apply`/`emit`都是typed pure helpers。`emit`产生与该source slice同一logical成员关系的output；operation把各slice outputs重新组成原source axis上的结果，并返回`apply(summarize(full_source), initial_state)`作为final state。Compiler选择的segment数量、边界和内部prefix states不可由作者观察，也不能成为result shape或ABI。若算法本身输出per-chunk states或chunk数量出现在ABI中，chunk是logical data，作者应使用显式chunk domain、source subregions与ordinary scan，而不是`region_scan`。

`reduce/scan`处理element summaries；`region_fold/region_scan`只在作者确实写下region-level summarizer或emitter时使用。Compiler依据这些作者定义形成合法的分段遍历、段内计算、摘要合并与输出结构，保留各operation的source顺序和effects合同；内部operations继续按各自语义lowering。

若 page、window、group 或 chunk boundary本身影响读取集合、输出shape或ABI，作者显式计算边界并构造source-derived subregion。若boundary只服务physical blocking，作者不写其extent；只有上述homomorphism使compiler-selected segmentation具有唯一语义。

## 6. Structured tensor operations

`reduce`、`scan`、`contract`、`scaled_contract`、`sparse_contract` 和 `histogram` 是 first-class logical tensor operations，不是 target primitive请求。

- reduce 保存 accumulator schema、axes、identity 与声明可结合、可交换的typed pure combine；
- scan 保存可结合的typed pure combine，并定义保留成员顺序的direction与inclusive/exclusive prefixes；
- contract 保存二元 multiply-add contraction 的 paired batch/reduction axes、free axes 与 accumulator；
- scaled contract 用closed `[M,G,C]/[M,G] × [G,C,N]/[N,G]` positional schema保存microscaling formats、scale relation与逻辑contraction；
- sparse contract 保存 compressed values、typed format schema、metadata interpretation与逻辑 contraction；
- histogram 保存 values 到 count tensor 的 binning semantics。

这些 operations 允许各自明确规定的 reassociation，但不指定 reduction tree、MMA、layout、storage 或 pipeline。target没有等价 native primitive时可以合法展开或明确拒绝；不能修改 operation 的逻辑定义。

## 7. Logical parts 不使用 `partition`

若 part 数量、identity、边界公式或中间 tensor interface 可观察，作者用普通 domain、source-derived subregion、index arithmetic 与 control 写出这些逻辑关系。`parallel` 表达逻辑 points 无序独立；source slicing 定义实际成员集合。空 part、tail 与中间 tensor shape 都属于作者定义的算法，不表示 physical tile、线程块或额外 launch。

Intent 不提供 `partition(auto/count/extent)` operation。可观察 partitioning 已由普通语义完整表达；不可观察 blocking 属于 physical program。

## 8. Ragged 与 indexed relations

ragged data 由普通关系组成：

- outer domain；
- member source domain；
- offsets；
- 可选 member index mapping。

对 outer identity `g`，连续 members 是 `member_source[offsets[g]:offsets[g+1]]`；非连续 mapping 再形成 indexed relation。subregion/relation 的 SSA identity 与 provenance区分多个同时存在的关系。

canonical Kernel IR 不建立独立 RaggedOp 或 MembersOp。`ragged(...)`、`members(...)` 可以是普通 surface helper，但必须机械展开到上述唯一关系路径。

## 9. Indexed memory 与 effects

所有 indexed access 共享 typed index relation 与 active validity。relation保存source identity/rank、result logical axes、每个source axis的typed coordinate expression，以及这些expressions对domain、subregion或data-derived indices的SSA provenance；composition必须组合关系本身，不能退化成只保存result shape。

在该共同表示上：

- immutable tensor selection 是 pure gather；
- external view load是 external read；
- logical-buffer load是 mutable-resource read；
- read 的 invalid lane不访问 source，并返回显式 fill；
- invalid write lane不产生 effect。

ordinary assignment 与 unique scatter使用 arbitrary-index unique store；目的 relation必须可证明 injective。collision reduction使用独立 `scatter_reduce` 与 typed combine。atomic load/store/RMW/CAS保存不可分割性、modification order、旧值和 memory order。

Intent 不提供 canonical copy op。一次 immutable SSA read 加 indexed write 已完整定义 snapshot、mapping、validity、cast、alias 与 effect order；bulk transfer、async copy、DMA/TMA和同步协议由 physical program形成。

## 10. Atomic 与 RNG

atomic operation不带作者可见 physical scope。它对本次 kernel invocation 中通过同一 logical allocation/address relation访问同一 atomic object的 executions定义 modification order。`relaxed/acquire/release/acq_rel` 改变 happens-before，因此属于 operation语义；provider根据 logical sharing与target execution mapping选择物理 scope。

非 atomic conflicting accesses是非法 data race。语言不承诺不同 parallel iterations 同时驻留或 spin-wait 必然取得进展。

RNG 是纯 stateless counter operation。canonical random bits固定为 Philox4x32-10：seed、logical counter、uint32 wrap、round constants、输出 words与 uniform conversion均跨 target一致。它不读取 program/thread/lane identity，也不依赖调用顺序或 mutable provider RNG state。
