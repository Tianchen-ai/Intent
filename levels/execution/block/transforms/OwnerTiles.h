#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#pragma once
#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
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
    Operation *mulOp = nullptr;
    Value scaleVal;
    Operation *expOp = nullptr;
    Operation *guardOp = nullptr;
    Value weightVal;
    bool isExp2 = false;
    arith::FastMathFlags maxFlags = arith::FastMathFlags::none;
    arith::FastMathFlags subFlags = arith::FastMathFlags::none;
    arith::FastMathFlags scaleFlags = arith::FastMathFlags::none;
    arith::FastMathFlags expFlags = arith::FastMathFlags::none;
    Value causalUpper;
    bool causalInclusive = false;
  };
  bool canVectorizeIndependentLoop(scf::ForOp loop);
  void vectorizedLoop(scf::ForOp loop, const Values &env, Value mask);
  void triangularLoop(scf::ForOp loop, const TriangularRecurrenceInfo &tri,
                      const Values &env, Value mask);
  SmallVector<Value> exclusiveScanLoop(scf::ForOp loop,
                                       const ExclusiveScanLoopInfo &scan,
                                       const Values &env, Value mask);
  void boundedHistogramLoop(scf::ForOp loop,
                            const BoundedHistogramLoopInfo &hist,
                            const Values &env, Value mask);
  void boundedScatterLoop(scf::ForOp loop, const BoundedScatterLoopInfo &scat,
                          const Values &env, Value mask);
  void associativeScatterLoop(scf::ForOp loop,
                              const AssociativeScatterLoopInfo &info,
                              const Values &env, Value mask);
  SmallVector<Value> selectionLoop(scf::ForOp loop,
                                   const OrderedSelectionInfo &sel,
                                   const Values &env, Value mask);
  std::optional<OnlinePanelFusion> matchOnlinePanelPair(scf::ForOp maxLoop,
                                                        scf::ForOp sumLoop);
  SmallVector<Value> lowerBody(Block &body, Values env, Value mask);
  SmallVector<Value> serialLoop(scf::ForOp loop, const Values &env, Value mask);
  SmallVector<Value> panelLoop(scf::ForOp loop, const Values &env, Value mask,
                               const OnlinePanelFusion *online = nullptr);
  LogicalResult build();
};
} // namespace intentdsl::block
