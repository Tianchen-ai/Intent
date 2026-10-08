# Configuration and tuning

Algorithm source does not contain tiles, warps, stages, device names or autotune winners. Physical programs, complete empirical configurations and target toolchains own these choices.

## Use default configurations

Ordinary `intent.compile(..., target=...)` uses the selected provider's distributed configurations. The compiler selects one family from the current typed computation and traversal structure, projects complete empirical rows, removes provably illegal rows and merges equivalent rows. It neither chooses configurations by kernel name nor constructs Cartesian products of parameter columns.

Triton receives actual `Config` objects. cuTile and BANG C runtimes tune the same declared candidate set. CPU configurations also bind explicit implementation sets. The first native call may include JIT and tuning; later calls reuse the selected configuration. Tuning trials preserve initial InOut contents and real alias relationships. The winner executes on caller arguments once.

## Override complete empirical rows

`intent.compile(..., tuning_config="/path/to/config.json")` and CLI `--tuning-config` accept finite JSON overrides. A supplied family replaces its default rows as a group; omitted families retain defaults:

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

The first seven columns are `ownership_m, ownership_n, reduction, reduction_outer, scan, traversal_workers, traversal_group`. Triton's final four are `NUM_WARPS, NUM_STAGES, NUM_CTAS, USE_TENSOR_DESCRIPTOR`. A row is a correlated complete configuration, not independent axis candidates. Projection, constraints and filtering produce at most one candidate per empirical row. They do not append candidates, borrow columns from other rows or interpret `1` as a disabling marker.

The compiler reads the configuration file only when compiling a kernel. Changing it requires recompiling the artifact; launch and tuning do not reread it. Unknown fields, wrong column counts/types, duplicates, empty tables and illegal values produce diagnostics. Having no legal candidate does not switch algorithms or fall back to defaults.

See [physical parameters](../compiler/physical-parameters.md) and [CPU program IR](../compiler/cpu-program-ir.md) for cuTile, CPU and DSA columns, binding responsibilities, caches and tuning state.

## Inspect the actual outcome

The existing `examples/softmax.py --inspect-native` reads `artifact.observation`: the actual selected configuration, candidate status and SDK-provided native resources. Unavailable resources retain their reason. Estimated working sets are not machine register or shared-memory allocation. Observation and tuning do not establish numerical correctness.
