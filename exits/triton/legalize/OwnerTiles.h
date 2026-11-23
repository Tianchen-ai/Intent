#pragma once
#include "TileBuilder.h"
#include "intentdsl/levels/execution/block/ir/FoldIssue.h"

namespace CI_UNIT_NAMESPACE {
// An owner is one parallel execution of the author's ordered region. Outside
// panel folds scalars become [owner] and member vectors become [owner,member].
struct OwnerTiles : Tiles {
  Operation &space;
  int64_t rows, logical;
  Value active;
  using Values = llvm::DenseMap<Value, Value>;
  Values outer;
  OwnerTiles(func::FuncOp fn, Operation &scope);
  Type lifted(Value source);
  Value widen(Value v);
  Value castElement(Value v, Type element);
  Value broadcast(Value v, Type result, int axis = 0);
  Value apply(Operation &old, ValueRange args, Type result);
  Value constant(arith::ConstantOp c);
  Value select(Value condition, Value yes, Value no);
  Value slice(Operation &old, ArrayRef<Value> coords, Value mask,
              Value stored = {});
  Value at(Value old, const Values &env);
  SmallVector<Value> lowerBody(Block &body, Values env, Value mask);
  SmallVector<Value> serialLoop(scf::ForOp loop, const Values &env, Value mask);
  SmallVector<Value> panelLoop(scf::ForOp loop, const Values &env, Value mask);
  LogicalResult build();
};
} // namespace CI_UNIT_NAMESPACE
