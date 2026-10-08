# Physical parameters and tuning

## 1. Parameters are not names in a side table

A physical parameter is a typed symbol bound at target compile time that participates directly in the current GPU program. For example, BM/BN/BK constrain fragment types, program-space extents, loops, access coordinates and validity together; they must not appear only in a search declaration or generated variable name.

Each parameter declaration defines at least:

- A stable symbol.
- A semantic role, such as logical ownership extent, reduction chunk, scan/region segment, group size or provider option.
- An integer, boolean or enum kind.
- A finite candidate domain or verifiable constraints.
- The types, operations and launch fields it directly affects.
- Target/provider legality requirements.

Physical parameters do not enter KIR or become author DSL parameters.

## Compile-time candidate data

`intent.compile(..., tuning_config=path)` and `compile_shared_gpu(..., tuning_config=path)` accept a finite JSON configuration override; the CLI option is `--tuning-config <path>`. Without an override, the compiler reads the selected provider's complete configuration table distributed with the compiler. The table lives alongside provider configuration responsibilities and is read once before configuration materialization. Editing the table does not require rebuilding C++, but does require recompiling the kernel artifact. Launch and autotuning do not read the file again.

Each row specifies shared granularity preferences and provider options together, following Triton's complete `Config` list organization. An override uses the selected provider namespace, `triton` or `cutile`, and maps existing structural families to arrays of complete rows. For example:

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

An explicitly supplied family replaces that family's default rows as a group; omitted families retain their default data. The compiler selects one family from the current typed program's computation and traversal structure. All parameters consume the same row; independent parameter families must not be joined into a row. Configuration is not selected by kernel name, source or registry, and JSON cannot specify conditions or algorithms. An unreadable file, unknown field/namespace/family, incorrect column count or type, value outside its column type, duplicate row or empty candidate table is diagnosed directly.

The first seven columns are `ownership_m, ownership_n, reduction, reduction_outer, scan, traversal_workers, traversal_group`. These are correlated role-based granularity preferences, not parameter bindings imposed on arbitrary shapes. Projection selects the largest value in the typed domain that does not exceed the requested value, or the domain minimum if no such value exists; dimensions already fixed by the program remain fixed. Axis relations and resource constraints may deterministically adjust bindings within the same row or reject the row, but cannot append candidates. Default and override tables use the same rules. The final tuple records actual values explicitly rather than presenting profile requests as final bindings.

When current reduction axis relations prove that pointwise ownership represents retained row axes, it consumes the same row's `ownership_m` row budget; pointwise traversal and other column axes retain their original role preferences. A complete row can therefore express a small number of retained rows alongside larger traversal/reduction chunks without adding candidates. This mapping does not reinterpret a contraction's M/N, change logical axes or change source order.

Triton rows then contain `NUM_WARPS, NUM_STAGES, NUM_CTAS, USE_TENSOR_DESCRIPTOR`; cuTile rows contain `CUTILE_CTAS, CUTILE_WORKER_WARPS, CUTILE_OCCUPANCY, CUTILE_ACCESS_FORM, CUTILE_LOAD_POLICY`. Boolean columns use 0/1; other columns are positive integers. Provider options are declared through formal `ParameterAttr` values, with shared/provider bindings from the same row kept correlated through the existing `ConfigurationSetAttr`. The shared stage may retain options not yet consumed by provider structure. Providers filter provably illegal combinations by complete row, remove options with no actual consumer and deduplicate. They do not multiply tiles by launch, access and pipeline options again, add axis endpoints or infer combinations that were not supplied. Stage differences that do not affect the program may be merged; stage choices for actual pipelines retain the original row. cuTile residency is derived from the same row's CTA/occupancy bindings, never exchanged between rows. If the current program has no legal candidate, compilation fails without changing the algorithm or falling back to the default table. External provider compilers retain responsibility for their own machine resource constraints.

## 2. Parameter expressions

Fragment shapes and compile-time loop steps can use typed integer expressions composed of constants and physical parameters. Runtime logical extents remain SSA values; the two must not be mixed through strings.

Typical relations include:

```text
grid_m = ceil_div(M, BM)
m_coord = program_m * BM + range(0, BM)
m_valid = m_coord < M
fragment_type = fragment<f16, [BM, BK]>
```

Here M is a runtime logical extent and BM/BK are compile-time physical parameters. Grid and validity explicitly connect the two kinds of values.

## 3. Structural decisions and parameter decisions

Shared passes first determine one physical program structure, including program mapping, loop nesting, the value/access graph and structured-operation skeleton. Parameters bind that program's granularity and provider compile options.

Intent does not construct an arbitrary Cartesian product of physical programs or a graph-level structural autotuner. A choice that changes program structure and cannot be formed by the lower compiler through parameters is made by a typed compiler pass and recorded in IR. If the lower compiler already exposes a choice as compile-time configuration and can measure its winner, Intent supplies only a legal parameter domain.

Although `num_warps`, `num_stages` and `num_ctas` can drive structural changes in Triton TTGIR, they remain Triton provider parameters. Intent supplies empirical configurations and provable constraints compatible with the current program; the provider tuner selects the winner. A parameter domain describes the values of supplied configuration columns, not permission to generate their Cartesian product. Each complete empirical configuration materializes at most one candidate; filtering and deduplication may reduce the count.

## 4. Candidates and instantiation

A candidate is a concrete binding of all required physical/provider parameters. Candidate instantiation produces a complete, legal provider program. A candidate cannot merely change strings while leaving the emitter to interpret the program.

Legality has two levels:

1. Intent verifies provable constraints: positive/static extents, shape relations, operation capabilities, grid limits, known resource bounds and provider surface requirements.
2. The provider compiler verifies the layout, register, shared-memory, instruction and pipeline constraints it owns.

Intent removes definitely illegal candidates without duplicating the lower compiler's complete resource allocator. A lower compiler failure caused by machine resources that could not be predicted may invalidate that candidate; it cannot change KIR or silently substitute another algorithm.

## 5. Provider autotuners

The provider autotuner compiles and benchmarks the remaining candidates and selects a winner. The winner is a runtime/tuning artifact: it is not written into canonical KIR or shared GPU IR and does not become a device-model branch.

Tuning trials are not an author-observable invocation. The runtime maintains trial state according to external-view read/write directions and actual allocation alias relationships. `InOut` contents needed by the program, read/write aliases and atomic state have the same initial contents before every trial, not merely between candidates. Trial arguments preserve shape, dtype, strides, offsets and aliases; views must not be cloned independently into non-aliasing inputs. The winner executes exactly once on caller arguments. Tuning failure must not leave trial effects in caller state.

The Triton path generates a kernel with explicit parameter roles and a real `Config` set, allowing Triton to tune BM/BN/BK, num_warps, num_stages, num_ctas and permitted provider forms. cuTile uses its actually supported tuning entry. Where no lower tuner exists, a provider runtime may measure and select from the same declared candidate set.

## 6. Provider form candidates

Local forms such as pointer/descriptor access or cuTile gather spellings enter the candidate surface only when:

- Both forms implement the same shared GPU access/operation.
- Each form can be legalized independently before serialization.
- The lower compiler or provider runtime can actually compile and measure them.
- Selection does not change the KIR algorithm, kernel count or timing scope.

If a form changes operands, types or control, it first becomes a provider-local extension or compile-time branch in the current program. The serializer cannot switch it through string conditions alone.

## 7. Cache and specialization identity

A compiled candidate's identity includes at least:

- Canonical kernel specialization and the public ABI.
- The runtime shape/dtype specialization key.
- The selected provider and hardware target.
- Concrete physical/provider parameter bindings.
- Compiler/provider code identity and relevant options.

The autotune winner cache additionally indexes candidate timings by the runtime tuning key. Compiled artifact and winner caches are separate layers; recording a winner in IR cannot substitute for a compilation decision.

Intent compilation artifacts default to `$XDG_CACHE_HOME/intentdsl/`, or `~/.cache/intentdsl/` if the XDG path is unset; `INTENT_CACHE_DIR` can override the root. Each entry stores input KIR, target IR, provider source and interface metadata. Results from `intent.generate` and `intent.compile` expose that location through `cache_directory`. Identical compilation inputs reuse generated results. Changes to compiler file identity, target/capability options, default profiles or explicit tuning configuration contents prevent reuse of an old entry. Concurrent requests serialize only on the same cache entry. Successful artifacts are published only when complete; failed requests retain inputs and diagnostics but are not reused as successful results.

This layer serves Intent source/IR compilation calls. Triton, cuTile and other lower compilers retain their own native JIT caches for runtime specialization, concrete candidates and native binary identity/reuse. Experimental directories do not store generated source or intermediate IR separately.

## 8. Performance comparison boundaries

Comparing generated and handwritten provider-source performance uses the algorithm, input shape, external dtype and explicit invocation/timing scope as the basis. Both sides may tune independently; measurement does not require matching complete candidate sets or exhausting all candidates first. Small differences in rounding locations, approximate math or intermediate precision are documented rather than categorically blocking same-algorithm performance comparisons. This does not relax the compiler's obligation to preserve Intent semantics.

Attributing a performance gap specifically to compiler program quality requires controlling relevant parameter roles, candidate budgets and timing scopes, or explaining their effects. Selecting the same tuning winner is not the comparison goal, and candidate tuning time is not operator execution time. Differences in ABI representation, auxiliary outputs or layout conversion must be explicitly included in or excluded from the measurement scope without hiding extra work.

Tuning selects parameters and declared local forms; it cannot hide missing program mapping, access graphs or structured realization. If generated code requires a larger search space to compensate for missing structure, that remains an IR/pass problem.

## 9. Complete DSA configurations

BANG C follows the same complete empirical-list principle using parameters for its independent execution model, not GPU parameter roles. Its default table lives in the provider configuration module; `bangc.block` in `tuning_config` replaces the list as a group. Each row is `tile, tile_m, tile_n, tile_k, region_tile, tasks, local_bytes`, verified by the existing `dsa.ConfigurationAttr`. Configuration is neither hardware-target identity nor an author DSL parameter. `local_bytes` is the configuration's storage budget and does not change the selected architecture's actual capacity.

Each row independently forms a complete, bound DSA physical program from the same canonical KIR. Standard nested ModuleOp values store these candidates, and each candidate passes through the original shared/provider passes and verifiers. The public ABI, numerical permissions and architecture remain identical. The serializer mechanically publishes source ranges, entries and candidate configurations; the runtime does not reconstruct the program from configuration.

A single legal candidate is called directly. Multiple candidates are compiled natively, measured, selected and cached. Trials preserve initial contents and alias relationships by allocation owner, retain view dtype, offset and strides, and restore state before each measurement. The winner executes once on the original caller arguments. Compilation failures are recorded for their candidates; if none can compile, the failure is diagnosed. An execution failure preserves the original error and terminates without trying other candidates to conceal it. Complete configuration lists cannot hide missing DSA implementation or supply structure.
