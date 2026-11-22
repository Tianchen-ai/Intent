#pragma once
#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
namespace intentdsl::dsa {
struct MatrixStorage {
  int64_t (*privateBytes)(int64_t rows, int64_t columns, int64_t depth);
};
mlir::LogicalResult verifyMatrix(mlir::func::FuncOp,
                                 const compilerinfra::TargetProfile &,
                                 bool supplied);
}
