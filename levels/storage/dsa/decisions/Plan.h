#pragma once
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
namespace intentdsl::dsa {
llvm::Expected<mlir::DictionaryAttr> plan(mlir::func::FuncOp,
                                          const LocalSupply &, int64_t depth,
                                          int64_t participants,
                                          bool persistent = false,
                                          int64_t batch = 1);
bool reusableInput(mlir::func::FuncOp);
bool batchableInstances(mlir::func::FuncOp);
llvm::Error stageTransfers(mlir::Operation *, mlir::Attribute,
                           const compilerinfra::TargetProfile &,
                           mlir::RewriterBase &);
} // namespace intentdsl::dsa
