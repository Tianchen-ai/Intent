# Changelog

## 2.1.0

- Uses the CompilerInfra 0.6.0 SDK. Inspection exposes reference defaults, per-operation numerical permissions and target-native dot contracts. Explicit IEEE and FP32 product requirements constrain narrowing.
- CPU exits share a cost model for narrow matrix products and weighted row folds. Mojo can use contiguous masked loads for guarded windows. CUDA Tile adds stable bucket scatter through sorted runs. BANG C uses certified vector division and the native BF16 dot default observed in CNNL.
- The five-exit scoreboard covers 355 complete calls and records three rounds of quartiles for each measured row. [Measurements](doc/plan/v0.6-semantics.md)
