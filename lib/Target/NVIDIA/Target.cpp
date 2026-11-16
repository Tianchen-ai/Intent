#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/TargetOptions.h"

namespace intent::compiler {
mlir::LogicalResult configureNVIDIATarget(mlir::ModuleOp module, Request &request) {
  auto options = readTargetOptions(module, request,
      {"compute_units", "shared_memory_per_unit", "max_dynamic_shared_memory_per_block",
       "registers_per_unit", "max_threads_per_block", "compute_capability_major",
       "compute_capability_minor", "single_to_double_precision_perf_ratio",
       "matrix_units", "dynamic_vector_width"});
  if (mlir::failed(options)) return mlir::failure();
  auto &target = request.gpu;
  for (auto [name, value] : {
      std::pair<llvm::StringRef, int64_t *>{"compute_units", &target.computeUnits},
      {"shared_memory_per_unit", &target.sharedMemoryPerUnit},
      {"max_dynamic_shared_memory_per_block", &target.maxDynamicSharedMemoryPerBlock},
      {"registers_per_unit", &target.registersPerUnit},
      {"max_threads_per_block", &target.maxThreadsPerBlock},
      {"compute_capability_major", &target.computeCapabilityMajor},
      {"compute_capability_minor", &target.computeCapabilityMinor},
      {"single_to_double_precision_perf_ratio", &target.singleToDoublePrecisionPerfRatio}})
    if (mlir::failed(readIntegerOption(module, *options, name, *value))) return mlir::failure();
  if (mlir::failed(readBooleanOption(module, *options, "matrix_units", target.matrixUnits)) ||
      mlir::failed(readBooleanOption(module, *options, "dynamic_vector_width", target.dynamicVectorWidth)))
    return mlir::failure();
  return mlir::success();
}
}
