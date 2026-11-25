#pragma once
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include <optional>
namespace intentdsl::dsa {
llvm::Expected<mlir::DictionaryAttr> plan(mlir::func::FuncOp,
                                          const LocalSupply &, int64_t depth,
                                          int64_t participants,
                                          bool persistent = false,
                                          int64_t batch = 1);
std::optional<mlir::DictionaryAttr>
planLibrary(mlir::func::FuncOp, const compilerinfra::TargetProfile &);
bool reusableInput(mlir::func::FuncOp);
bool batchableInstances(mlir::func::FuncOp);
bool canAsyncWriteBack(mlir::func::FuncOp, int64_t depth);
llvm::Error stageTransfers(mlir::Operation *, mlir::Attribute,
                           const compilerinfra::TargetProfile &,
                           mlir::RewriterBase &);
} // namespace intentdsl::dsa
