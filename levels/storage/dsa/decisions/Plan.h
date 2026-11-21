#pragma once
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
namespace intentdsl::dsa {
llvm::Expected<mlir::DictionaryAttr> plan(mlir::func::FuncOp,
                                          const LocalSupply &, int64_t depth);
llvm::Error stageTransfers(mlir::Operation *, mlir::Attribute,
                           const compilerinfra::TargetProfile &,
                           mlir::RewriterBase &);
} // namespace intentdsl::dsa
