#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#pragma once
#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"

namespace intentdsl::block {
// An owner is one parallel execution of the author's ordered region. Outside
// panel folds scalars become [owner] and member vectors become [owner,member].
struct OwnerTiles : Tiles {
  Operation &space;
  int64_t rows, logical;
  Value active;
  OwnerAccess access;
  using Values = llvm::DenseMap<Value, Value>;
  Values outer;
  OwnerTiles(func::FuncOp fn, Operation &scope, StringRef unit,
             OwnerAccess access);
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
  struct OnlinePanelFusion {
    scf::ForOp maxLoop;
    Value scoreVal;
    Value condVal;
    Operation *subOp = nullptr;
    Operation *expOp = nullptr;
    Operation *guardOp = nullptr;
    Value weightVal;
    bool isExp2 = false;
    arith::FastMathFlags maxFlags = arith::FastMathFlags::none;
    arith::FastMathFlags subFlags = arith::FastMathFlags::none;
    arith::FastMathFlags expFlags = arith::FastMathFlags::none;
    Value causalUpper;
    bool causalInclusive = false;
  };
  std::optional<OnlinePanelFusion> matchOnlinePanelPair(scf::ForOp maxLoop,
                                                        scf::ForOp sumLoop);
  SmallVector<Value> lowerBody(Block &body, Values env, Value mask);
  SmallVector<Value> serialLoop(scf::ForOp loop, const Values &env, Value mask);
  SmallVector<Value> panelLoop(scf::ForOp loop, const Values &env, Value mask,
                               const OnlinePanelFusion *online = nullptr);
  LogicalResult build();
};
} // namespace intentdsl::block
