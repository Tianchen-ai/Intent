#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Target/CuTile/Transforms/Compute/MMALoops.h"

#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::cutile {
namespace {

bool isSpecializationExpression(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return false;
  if (isa<arith::ConstantOp, gpu::ParameterOp, gpu::PhysicalExprOp, gpu::DimOp>(
          operation))
    return true;
  if (!isa<gpu::BinaryOp, gpu::CompareOp, gpu::CastOp>(operation))
    return false;
  return llvm::all_of(operation->getOperands(), isSpecializationExpression);
}

bool positiveExtent(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    return integer && integer.getInt() > 0;
  }
  auto parameter = value.getDefiningOp<gpu::ParameterOp>();
  return parameter && !parameter.getDeclaration().getCandidates().empty() &&
         llvm::all_of(parameter.getDeclaration().getCandidates().asArrayRef(),
                      [](int64_t candidate) { return candidate > 0; });
}

bool safeScalarGuardOperation(Operation *operation) {
  if (isa<arith::ConstantOp, gpu::CompareOp, gpu::DimOp,
          gpu::ParameterOp>(operation))
    return true;
  if (auto cast = dyn_cast<gpu::CastOp>(operation))
    return cast.getValue().getType().isIntOrIndex() &&
           cast.getResult().getType().isIntOrIndex();
  auto binary = dyn_cast<gpu::BinaryOp>(operation);
  if (!binary || !binary.getResult().getType().isIntOrIndex())
    return false;
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::Subtract:
  case BinaryOperator::Multiply:
  case BinaryOperator::Minimum:
  case BinaryOperator::Maximum:
  case BinaryOperator::MinimumNum:
  case BinaryOperator::MaximumNum:
  case BinaryOperator::LogicalAnd:
  case BinaryOperator::LogicalOr:
  case BinaryOperator::BitwiseAnd:
  case BinaryOperator::BitwiseOr:
    return true;
  case BinaryOperator::FloorDivide:
  case BinaryOperator::Remainder:
    return positiveExtent(binary.getRhs());
  default:
    return false;
  }
}

bool collectInvariantGuard(Value value, scf::ForOp loop,
                           llvm::SmallPtrSetImpl<Operation *> &visited,
                           SmallVectorImpl<Operation *> &operations) {
  Operation *definition = value.getDefiningOp();
  Operation *owner = definition
                         ? definition
                         : cast<BlockArgument>(value).getOwner()->getParentOp();
  if (owner != loop.getOperation() && !loop->isAncestor(owner))
    return true;
  // Existing external SSA reads can be reused, but never move a read across
  // iterations or speculate a potentially invalid scalar operation.
  if (!definition || !safeScalarGuardOperation(definition))
    return false;
  if (!visited.insert(definition).second)
    return true;
  for (Value operand : definition->getOperands())
    if (!collectInvariantGuard(operand, loop, visited, operations))
      return false;
  operations.push_back(definition);
  return true;
}

void unswitchNativeAccessGuard(scf::ForOp loop) {
  scf::IfOp selected;
  SmallVector<Operation *> guardOperations;
  loop.walk([&](scf::IfOp conditional) {
    if (selected || conditional->getParentOfType<scf::ForOp>() != loop ||
        conditional.getNumResults() == 0 ||
        isSpecializationExpression(conditional.getCondition()))
      return;
    bool tileLoad = false;
    bool gatherLoad = false;
    conditional.getThenRegion().walk([&](TileLoadOp) { tileLoad = true; });
    conditional.getElseRegion().walk([&](GatherLoadOp) { gatherLoad = true; });
    if (!tileLoad || !gatherLoad)
      return;
    llvm::SmallPtrSet<Operation *, 16> visited;
    SmallVector<Operation *> operations;
    if (!collectInvariantGuard(conditional.getCondition(), loop, visited,
                               operations))
      return;
    selected = conditional;
    guardOperations = std::move(operations);
  });
  if (!selected)
    return;
  SmallVector<scf::IfOp> equivalentChoices;
  loop.walk<WalkOrder::PostOrder>([&](scf::IfOp conditional) {
    if (conditional->getParentOfType<scf::ForOp>() == loop &&
        gpu::samePhysicalScalarExpression(conditional.getCondition(),
                                          selected.getCondition()))
      equivalentChoices.push_back(conditional);
  });

  // One program-uniform access decision, hence at most two loop versions.
  // Specialization-only choices stay for the provider compiler to fold.
  OpBuilder builder(loop);
  IRMapping guardMapping;
  for (Operation *operation : guardOperations)
    builder.clone(*operation, guardMapping);
  Value condition = guardMapping.lookupOrDefault(selected.getCondition());
  auto version = builder.create<scf::IfOp>(
      loop.getLoc(), loop.getResultTypes(), condition,
      /*withElseRegion=*/true);
  for (bool takeThen : {true, false}) {
    Region &region =
        takeThen ? version.getThenRegion() : version.getElseRegion();
    Block &block = region.front();
    if (!block.empty() && isa<scf::YieldOp>(block.back()))
      block.back().erase();
    OpBuilder nested(&block, block.end());
    IRMapping mapping;
    auto copied = cast<scf::ForOp>(nested.clone(*loop.getOperation(), mapping));
    // K/V or other cooperating accesses can carry separate, equivalent guard
    // expressions. Refine every occurrence, including nested ones, so a native
    // loop version does not retain another copy of the same runtime branch.
    for (scf::IfOp originalChoice : equivalentChoices) {
      auto choice = cast<scf::IfOp>(
          mapping.lookup(&originalChoice.getThenRegion().front())->getParentOp());
      Region &chosen = takeThen ? choice.getThenRegion() : choice.getElseRegion();
      if (!chosen.empty()) {
        Block &body = chosen.front();
        auto yield = cast<scf::YieldOp>(body.getTerminator());
        for (Operation &operation :
             llvm::make_early_inc_range(body.without_terminator()))
          operation.moveBefore(choice);
        choice.getResults().replaceAllUsesWith(yield.getResults());
      }
      choice.erase();
    }
    nested.create<scf::YieldOp>(loop.getLoc(), copied.getResults());
  }
  loop.getResults().replaceAllUsesWith(version.getResults());
  loop.erase();
}

void boundNativeReductionWidth(MMAOp mma) {
  constexpr int64_t nativeReductionLimit = 32;
  constexpr int64_t reductionChunk = 16;
  auto lhs = mma.getLhs().getType();
  auto rhs = mma.getRhs().getType();
  if (mma.getReductionChunk() || !lhs.getElementType().isF32() ||
      !rhs.getElementType().isF32() ||
      !mma.getResult().getType().getElementType().isF32())
    return;
  auto reduction = cast<gpu::PhysicalExprAttr>(lhs.getShape().getValue().back());
  if (reduction.getKind() ==
          gpu::PhysicalExprKind::Constant &&
      (reduction.getValue() <= nativeReductionLimit ||
       reduction.getValue() % reductionChunk != 0))
    return;
  OpBuilder builder(mma);
  auto extent = [&](gpu::FragmentType type, unsigned axis) -> Value {
    return builder.create<gpu::PhysicalExprOp>(
        mma.getLoc(), builder.getIndexType(),
        cast<gpu::PhysicalExprAttr>(type.getShape()[axis]));
  };
  unsigned matrixAxis = lhs.getShape().size() - 2;
  Value m = extent(lhs, matrixAxis);
  Value n = extent(rhs, matrixAxis + 1);
  Value k = extent(lhs, matrixAxis + 1);
  Value one = builder.create<arith::ConstantIndexOp>(mma.getLoc(), 1);
  Value chunk = builder.create<arith::ConstantIndexOp>(mma.getLoc(), reductionChunk);
  Value nativeWidth = builder.create<arith::ConstantIndexOp>(mma.getLoc(), nativeReductionLimit);
  Value zero = builder.create<arith::ConstantIndexOp>(mma.getLoc(), 0);
  Value minimum = builder.create<arith::ConstantIndexOp>(mma.getLoc(), 256);
  auto compare = [&](Value left, Value right, ComparePredicate predicate) -> Value {
    return builder.create<gpu::CompareOp>(mma.getLoc(), builder.getI1Type(),
                                         left, right, predicate);
  };
  Value split = compare(k, nativeWidth, ComparePredicate::Gt);
  auto require = [&](Value condition) {
    split = builder.create<gpu::BinaryOp>(mma.getLoc(), builder.getI1Type(),
        split, condition, BinaryOperator::LogicalAnd);
  };
  require(compare(m, one, ComparePredicate::Gt));
  require(compare(n, one, ComparePredicate::Gt));
  Value elements = builder.create<gpu::BinaryOp>(mma.getLoc(),
      builder.getIndexType(), m, n, BinaryOperator::Multiply);
  require(compare(elements, minimum, ComparePredicate::Ge));
  Value remainder = builder.create<gpu::BinaryOp>(mma.getLoc(),
      builder.getIndexType(), k, chunk, BinaryOperator::Remainder);
  require(compare(remainder, zero, ComparePredicate::Eq));

  // Keep the coalesced load tile while bounding live f32 MMA intermediates.
  // Unit free axes and short K fragments retain the native reduction.
  auto choice = builder.create<scf::IfOp>(
      mma.getLoc(), TypeRange{mma.getResult().getType()}, split, true);
  if (Attribute origin = mma->getAttr(gpu::originAttr))
    choice->setAttr(gpu::originAttr, origin);
  for (bool useChunks : {true, false}) {
    Region &region = useChunks ? choice.getThenRegion() : choice.getElseRegion();
    builder.setInsertionPointToStart(&region.front());
    auto copy = cast<MMAOp>(builder.clone(*mma));
    if (useChunks)
      copy.setReductionChunkAttr(builder.getI64IntegerAttr(reductionChunk));
    builder.create<scf::YieldOp>(mma.getLoc(), copy.getResult());
  }
  mma.getResult().replaceAllUsesWith(choice.getResult(0));
  mma.erase();
}

} // namespace

LogicalResult refineMMALoops(ModuleOp module) {
  auto physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  SmallVector<scf::ForOp> loops;
  physicalKernel->walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [index, argument] : llvm::enumerate(loop.getRegionIterArgs())) {
      auto original = dyn_cast<gpu::FragmentType>(argument.getType());
      if (!original || original.getShape().size() != 3 || !argument.hasOneUse())
        continue;
      auto unit = dyn_cast<gpu::PhysicalExprAttr>(original.getShape()[0]);
      if (!unit ||
          unit.getKind() !=
              gpu::PhysicalExprKind::Constant ||
          unit.getValue() != 1)
        continue;
      auto incoming = dyn_cast<gpu::ReshapeOp>(*argument.getUsers().begin());
      auto outgoing = yield.getOperand(index).getDefiningOp<gpu::ReshapeOp>();
      if (!incoming || !outgoing || incoming->getBlock() != loop.getBody() ||
          outgoing->getBlock() != loop.getBody() ||
          !outgoing.getResult().hasOneUse())
        continue;
      auto matrix = dyn_cast<gpu::FragmentType>(incoming.getResult().getType());
      auto mma = outgoing.getValue().getDefiningOp<MMAOp>();
      if (!matrix || matrix.getShape().size() != 2 || !mma ||
          mma.getAccumulator() != incoming.getResult() ||
          mma.getResult().getType() != matrix ||
          matrix.getShape().getValue() !=
              original.getShape().getValue().drop_front())
        continue;
      // Project only at the loop boundaries; the native MMA carries a matrix.
      OpBuilder before(loop);
      auto init = before.create<gpu::ReshapeOp>(
          loop.getLoc(), matrix, loop.getInitArgs()[index],
          incoming.getReassociation());
      loop.getInitArgsMutable()[index].assign(init.getResult());
      argument.setType(matrix);
      incoming.getResult().replaceAllUsesWith(argument);
      incoming.erase();
      yield->setOperand(index, outgoing.getValue());
      auto restoreRelation = outgoing.getReassociation();
      outgoing.erase();
      Value result = loop.getResult(index);
      result.setType(matrix);
      OpBuilder after(loop);
      after.setInsertionPointAfter(loop);
      auto restored = after.create<gpu::ReshapeOp>(
          loop.getLoc(), original, result, restoreRelation);
      result.replaceAllUsesExcept(restored.getResult(), restored.getOperation());
    }
    bool hasMMA = false;
    loop.getBody()->walk([&](MMAOp mma) {
      hasMMA |= mma->getParentOfType<scf::ForOp>() == loop;
    });
    if (hasMMA)
      unswitchNativeAccessGuard(loop);
  }
  SmallVector<MMAOp> mmas;
  physicalKernel->walk([&](MMAOp mma) { mmas.push_back(mma); });
  for (MMAOp mma : mmas)
    boundNativeReductionWidth(mma);
  gpu::eraseDeadPhysicalValues(*physicalKernel);
  return success();
}

} // namespace intent::cutile
