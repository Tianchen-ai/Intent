# Developing IntentDSL Components

IntentDSL compiles typed Python author kernels (`@intentdsl.kernel`) and static host sequences (`@intentdsl.program`) across five native JIT exits from one compilation tree (`product.json`):

- `levels/algorithm/kir` (`imported -> canonical`): author semantics, view projections, side effects, and explicit numerical permissions (such as `allow_subnormal_ftz`).
- `levels/execution/block` (`formed -> decided -> realized -> normalized`): instance grids, resident/streaming row traversals, tile contractions, and ordered owner tiles. Feeds `exits/triton` and `exits/cutile` at `block@normalized`, and `levels/storage/dsa` at `block@realized`.
- `levels/storage/dsa` (`formed -> supplied`): explicit NRAM/WRAM staging, double-buffered slot lifetimes, async DMA issue/wait/drain schedules, and owner streams. Feeds `exits/bangc` at `dsa@supplied`.
- `levels/execution/cpu` (`formed -> tasked -> buffers`): isolated OpenMP/Weft row tasks, packed right-hand panels, and thread-local scratch. Feeds `exits/weft` at `cpu@tasked` and `exits/mojo` at `cpu@buffers`.

## Adding a Strategy or Microkernel

1. **Inspect layout and ownership first**: run `compilerinfra-dev --library <libintentdsl.so> --workspace <repo> inspect` to locate exact unit files, decision families, and active strategies in `cmake/default.json`.
2. **Keep representation boundaries pure (v0.4)**:
   - Structural choices, loop schedules, buffer capacities, and numerical guards must be formed in node/exit IR before terminal printing.
   - Each exit's `TerminalPrinter` only spells operations already present in its terminal dialect; every runtime microkernel under `runtime/<exit>/kernels/` must have a matching `ImplementationContract` in `exits/<exit>/ir/` verified at `@callable`.
3. **Derive, compare, explain, and measure**:
   - Derive a configuration or source patch candidate with `derive` rather than editing baseline install state directly.
   - Compare parent (`base`) and child (`<candidate>`) IR at the affected boundary with `compare`, verify strategy selection and cost terms with `explain` and `impact`, then run full-call numerical and performance checks through `measure` (`intent_bench.compilerinfra_adapter`).
   - Adopt (`adopt`) only when numerical contracts hold and end-to-end call latency improves; otherwise `discard`.
