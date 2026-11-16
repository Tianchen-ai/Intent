#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/TargetOptions.h"
#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"

namespace intent::compiler {
mlir::LogicalResult configureNVIDIATarget(mlir::ModuleOp module, Request &request,
    bool tupleReductions, bool constantIdentity, bool fragmentGather) {
  auto options = readTargetOptions(module, request,
      {"compute_units", "shared_memory_per_unit", "max_dynamic_shared_memory_per_block",
       "registers_per_unit", "max_threads_per_block", "compute_capability_major",
       "compute_capability_minor", "single_to_double_precision_perf_ratio",
       "matrix_units", "dynamic_vector_width"});
  if (mlir::failed(options)) return mlir::failure();
  const llvm::StringRef names[] = {"compute_units", "shared_memory_per_unit",
      "max_dynamic_shared_memory_per_block", "registers_per_unit", "max_threads_per_block",
      "compute_capability_major", "compute_capability_minor", "single_to_double_precision_perf_ratio"};
  int64_t values[8];
  for (unsigned i = 0; i < 8; ++i)
    if (mlir::failed(readIntegerOption(module, *options, names[i], values[i]))) return mlir::failure();
  bool matrixUnits, dynamicVectorWidth;
  if (mlir::failed(readBooleanOption(module, *options, "matrix_units", matrixUnits)) ||
      mlir::failed(readBooleanOption(module, *options, "dynamic_vector_width", dynamicVectorWidth)))
    return mlir::failure();
  module.getContext()->getOrLoadDialect<nvidia::IntentNVIDIADialect>();
  module.getContext()->getOrLoadDialect<block::IntentBlockDialect>();
  auto device = nvidia::DeviceAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), values[0], values[1], values[2], values[3], values[4],
      values[5], values[6], values[7], matrixUnits, dynamicVectorWidth);
  if (!device) return mlir::failure();
  request.target = block::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), mlir::Attribute(device), matrixUnits, dynamicVectorWidth,
      tupleReductions, constantIdentity, fragmentGather);
  if (!request.target) return mlir::failure();
  return mlir::success();
}
}
