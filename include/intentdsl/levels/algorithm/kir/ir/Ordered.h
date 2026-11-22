#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::kir {
bool orderedOwners(mlir::func::FuncOp);
mlir::LogicalResult verifyOwners(mlir::func::FuncOp, llvm::StringRef dialect);
mlir::LogicalResult verifyOwnerBody(mlir::func::FuncOp, mlir::Operation &,
                                    llvm::StringRef);
int64_t ownerCount(mlir::Operation *, const CallFacts &);
mlir::LogicalResult verifyProjectedOwners(mlir::Operation &, const CallFacts &);
// The extent identity survives dtype conversions and scf loop carries.
std::optional<std::pair<mlir::BlockArgument, unsigned>> memberAxis(mlir::Value);
int64_t memberExtent(mlir::func::FuncOp, mlir::Value,
                     llvm::StringRef shapeAttr);
} // namespace intentdsl::kir
