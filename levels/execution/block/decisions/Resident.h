#pragma once
#include "compilerinfra/Assembly.h"
namespace CI_UNIT_NAMESPACE {
using RowMatch = bool(*)(mlir::Operation *);
inline bool contains(mlir::Operation *op, llvm::ArrayRef<llvm::StringRef> names){
  return op->walk([&](mlir::Operation *v){return llvm::is_contained(names,v->getName().stripDialect())?mlir::WalkResult::interrupt():mlir::WalkResult::advance();}).wasInterrupted();
}
void registerResident(compilerinfra::Unit &,llvm::StringRef,RowMatch,bool singleOwner=false);
}
