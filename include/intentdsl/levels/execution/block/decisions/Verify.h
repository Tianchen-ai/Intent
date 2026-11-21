#pragma once
#include "compilerinfra/Assembly.h"
namespace intentdsl {
llvm::Error verifyRows(mlir::Operation *anchor, mlir::ArrayAttr rows, mlir::StringAttr topology,
                       mlir::DenseI64ArrayAttr guard,const compilerinfra::TargetProfile &);
mlir::LogicalResult verifyMapping(mlir::ModuleOp,const compilerinfra::TargetProfile &,llvm::StringRef dialect);
}
