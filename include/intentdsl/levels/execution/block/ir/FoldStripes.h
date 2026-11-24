#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intentdsl::block {
uint64_t foldableStripeCount(mlir::Operation *parallelOp);
mlir::LogicalResult foldStripes(mlir::func::FuncOp fn,
                                const compilerinfra::TargetProfile &profile);
} // namespace intentdsl::block
