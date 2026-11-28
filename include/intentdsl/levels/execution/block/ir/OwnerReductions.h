#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::block {
void fuseOwnerReductions(mlir::func::FuncOp fn,
                         const compilerinfra::TargetProfile &profile,
                         bool partitionOutputs);
}
