#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/IterationDependencies.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool canVectorizeIterations(Block &block, scf::ForOp loop,
                           llvm::DenseMap<Value, bool> &varying) {
  auto data = [](Type type) {
    return isa<IntegerType, IndexType, FloatType, FragmentType>(type);
  };
  for (Operation &operation : block.without_terminator()) {
    if (!llvm::all_of(operation.getResultTypes(), data) ||
        llvm::any_of(operation.getOperandTypes(), [](Type type) {
          return isa<RecordType>(type);
        }))
      return false;
    // A range still describes one shared column domain. A lane-dependent
    // extent needs a different traversal, not a tensor-valued range bound.
    if (isa<MakeRangeOp>(operation) &&
        llvm::any_of(operation.getOperands(), [&](Value operand) {
          return variesWithIteration(operand, loop, varying);
        }))
      return false;
    // Gathering from a lane-expanded source would also need a new source-axis
    // coordinate. Keep this transform to reads from an unchanged SSA source.
    if (auto gather = dyn_cast<GatherOp>(operation))
      if (!loop.isDefinedOutsideOfLoop(gather.getSource()))
        return false;
    if (auto nested = dyn_cast<scf::ForOp>(operation)) {
      if (variesWithIteration(nested.getLowerBound(), loop, varying) ||
          variesWithIteration(nested.getUpperBound(), loop, varying) ||
          variesWithIteration(nested.getStep(), loop, varying) ||
          !canVectorizeIterations(*nested.getBody(), loop, varying))
        return false;
    } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
      if (!canVectorizeIterations(*branch.thenBlock(), loop, varying) ||
          (!branch.getElseRegion().empty() &&
           !canVectorizeIterations(*branch.elseBlock(), loop, varying)))
        return false;
    } else if (auto store = dyn_cast<StoreOp>(operation)) {
      // Compiler-created retained storage may be reused sequentially by the
      // original points. Its allocation has not been widened to lane slices.
      if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
          buffer && buffer.isInvocationWorkspace())
        return false;
      if (!data(store.getValue().getType()) ||
          !llvm::all_of(store.getCoordinates(), [&](Value value) {
            return data(value.getType());
          }))
        return false;
    } else if (isa<ReshapeOp, TransposeOp>(operation)) {
      continue;
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      if (!llvm::all_of(reduce.getCombine().front().without_terminator(),
                        [](Operation &nested) {
                          return canPredicateValueOperation(&nested) &&
                                 !isa<LoadOp, GatherOp>(nested);
                        }))
        return false;
    } else if (operation.getNumRegions() ||
               !canPredicateValueOperation(&operation)) {
      return false;
    }
  }
  return true;
}

LogicalResult vectorizeIterations(func::FuncOp kernel,
                                 uint64_t &source, int64_t &dimension) {
  SmallVector<scf::ForOp> loops;
  // Lift the inner independent axis first. Its fragment becomes the suffix
  // when an enclosing independent loop is subsequently widened.
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    IndexRelations relations;
    Type inductionType = loop.getInductionVar().getType();
    auto integer = dyn_cast<IntegerType>(inductionType);
    bool signedI32 = integer && !integer.isUnsigned() && integer.getWidth() == 32;
    llvm::DenseMap<Value, bool> varying;
    if (loop.getNumResults() || relations.constant(loop.getStep()) != 1 ||
        (!inductionType.isIndex() && !signedI32) ||
        // i32 endpoints and their difference fit the new i64 traversal. For
        // index endpoints, retain a domain where span and span-1 cannot wrap.
        (!signedI32 && (!relations.nonnegative(loop.getLowerBound()) ||
                       !relations.nonnegative(loop.getUpperBound()))) ||
        !canVectorizeIterations(*loop.getBody(), loop, varying))
      continue;
    IndependentIterationAccesses accesses;
    if (!loop->hasAttr(independentIterationAttr)) {
      auto independent = queryIndependentIterationAccesses(loop);
      if (failed(independent)) continue;
      accesses = std::move(*independent);
    }
    auto &guardedViews = accesses.guardedViews;
    auto &disjointViews = accesses.disjointViews;
    uint32_t elementBitWidth = 0;
    loop.walk([&](StoreOp store) {
      Type type = store.getValue().getType();
      if (auto fragment = dyn_cast<FragmentType>(type))
        type = fragment.getElementType();
      elementBitWidth = std::max(
          elementBitWidth, type.isIndex() ? 64u : type.getIntOrFloatBitWidth());
    });
    if (!elementBitWidth)
      continue;
    if (!guardedViews.empty()) {
      SmallVector<Value> injective;
      for (Value view : guardedViews) {
        auto condition = materializeNonOverlappingView(kernel, view);
        if (failed(condition))
          break;
        injective.push_back(*condition);
      }
      if (injective.size() != guardedViews.size())
        continue;
      OpBuilder builder(loop);
      Value safe = injective.front();
      auto require = [&](Value condition) {
        safe = builder.create<BinaryOp>(loop.getLoc(), builder.getI1Type(),
                                        safe, condition, BinaryOperator::LogicalAnd);
      };
      for (Value condition : ArrayRef<Value>(injective).drop_front())
        require(condition);
      for (auto [lhs, rhs] : disjointViews) {
        Value overlap;
        for (ViewOverlapOp fact : kernel.getOps<ViewOverlapOp>())
          if ((fact.getLhs() == lhs && fact.getRhs() == rhs) ||
              (fact.getLhs() == rhs && fact.getRhs() == lhs)) {
            overlap = fact.getResult();
            break;
          }
        if (!overlap) {
          OpBuilder entry(&kernel.front(), kernel.front().begin());
          overlap = entry.create<ViewOverlapOp>(loop.getLoc(), entry.getI1Type(),
                                               lhs, rhs);
        }
        Value zero = builder.create<arith::ConstantIntOp>(loop.getLoc(), 0, 1);
        require(builder.create<CompareOp>(loop.getLoc(), builder.getI1Type(),
                                          overlap, zero, ComparePredicate::Eq));
      }
      auto branch = builder.create<scf::IfOp>(loop.getLoc(), safe, true);
      builder.setInsertionPointToStart(branch.thenBlock());
      auto selected = cast<scf::ForOp>(builder.clone(*loop));
      loop->moveBefore(branch.elseBlock()->getTerminator());
      loop = selected;
    }
    auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
    int64_t maximumWidth = capabilities.getMaxThreadsPerBlock();
    auto lower = relations.constant(loop.getLowerBound());
    auto upper = relations.constant(loop.getUpperBound());
    if (lower && upper) {
      __int128 length = static_cast<__int128>(*upper) - *lower;
      int64_t covered = 1;
      while (covered < length && covered <= maximumWidth / 2)
        covered *= 2;
      maximumWidth = covered;
    }
    SmallVector<int64_t> candidates;
    for (int64_t width = 1; width <= maximumWidth;
         width *= 2)
      candidates.push_back(width);
    bool completeChunks = lower && upper && *upper > *lower &&
        llvm::all_of(candidates, [&](int64_t width) {
          return (static_cast<__int128>(*upper) - *lower) % width == 0;
        });
    auto name = ("ITERATION_" + Twine(source)).str();
    auto reference = getOrCreatePhysicalParameter(
        kernel, name, ParameterRole::OwnershipN, ParameterCategory::Pointwise,
        elementBitWidth, candidates);
    if (failed(reference))
      return failure();
    OpBuilder builder(loop);
    // This is a fresh one-dimensional iteration domain. Load/store sourceAxes
    // continue to map its coordinates to the original resource axes.
    auto ordinal = AxisMapAttr::get(kernel.getContext(), source++, 0,
                                   dimension++, 0, false);
    auto widthSchema = lookupParameter(kernel, *reference);
    if (failed(updateParameter(kernel, widthSchema.withBinding(widthSchema.getBinding()
            .withSource(PhysicalSourceAttr::get(kernel.getContext(), ordinal.getSourceId(), 0, false))
            .withPointwiseChunk(true))))) return failure();
    auto width = materializeParameter(builder, loop.getLoc(), *reference);
    auto extent = PhysicalExprAttr::get(
        kernel.getContext(), PhysicalExprKind::Parameter,
        0, *reference, builder.getArrayAttr({}));
    auto shape = FragmentType::get(
        kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({extent}),
        builder.getArrayAttr({ordinal}), 1, /*owner=*/1);
    auto boolean = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), shape.getShape(),
        shape.getAxisMaps(), shape.getValidity(), shape.getOwner());
    Location location = loop.getLoc();
    Type index = builder.getIndexType();
    Value begin = loop.getLowerBound(), end = loop.getUpperBound();
    if (signedI32) {
      begin = builder.create<CastOp>(location, index, begin);
      end = builder.create<CastOp>(location, index, end);
    }
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value nonempty = builder.create<CompareOp>(
        location, builder.getI1Type(), begin, end,
        ComparePredicate::Lt);
    Value span = builder.create<BinaryOp>(
        location, index, end, begin,
        BinaryOperator::Subtract);
    Value last = builder.create<BinaryOp>(location, index, span, one,
                                         BinaryOperator::Subtract);
    last = builder.create<SelectOp>(location, index, nonempty, last, zero);
    Value chunks = builder.create<BinaryOp>(
        location, index, last, width.getResult(), BinaryOperator::FloorDivide);
    Value present = builder.create<CastOp>(location, index, nonempty);
    chunks = builder.create<BinaryOp>(location, index, chunks, present,
                                     BinaryOperator::Add);
    auto blocked = builder.create<scf::ForOp>(location, zero, chunks, one);
    blocked->setAttrs(loop->getAttrs());
    builder.setInsertionPoint(blocked.getBody()->getTerminator());
    Value offset = builder.create<BinaryOp>(
        location, index, blocked.getInductionVar(), width.getResult(),
        BinaryOperator::Multiply);
    Value start = builder.create<BinaryOp>(location, index, begin,
                                          offset, BinaryOperator::Add);
    Value members = builder.create<MakeRangeOp>(
        loop.getLoc(), shape, start, width.getResult(), one, begin, end,
        ordinal.getSourceId(), 0, false);
    // Count chunks and guard lane ordinals so a padded final lane cannot wrap
    // past a large logical upper bound and accidentally become active again.
    Value active;
    if (completeChunks) {
      active = builder.create<arith::ConstantIntOp>(location, 1, 1);
    } else {
      Value remaining = builder.create<BinaryOp>(location, index, span, offset,
                                                BinaryOperator::Subtract);
      Value base = builder.create<BroadcastOp>(location, shape, start);
      Value lanes = builder.create<BinaryOp>(location, shape, members, base,
                                            BinaryOperator::Subtract);
      Value end = builder.create<BroadcastOp>(location, shape, remaining);
      active = builder.create<CompareOp>(loop.getLoc(), boolean, lanes, end,
                                        ComparePredicate::Lt);
    }
    IRMapping values;
    Value induction = members;
    if (signedI32) {
      auto bodyShape = FragmentType::get(kernel.getContext(), inductionType,
          shape.getShape(), shape.getAxisMaps(), shape.getValidity(), shape.getOwner());
      induction = builder.create<CastOp>(location, bodyShape, members);
    }
    values.map(loop.getInductionVar(), induction);
    for (Operation &operation : loop.getBody()->without_terminator())
      if (failed(clonePredicatedScalarOperation(
              builder, &operation, values, active, shape,
              /*nonemptyIterations=*/true)))
        return failure();
    loop.erase();
  }
  return success();
}

LogicalResult assignIterationRoles(func::FuncOp kernel) {
  llvm::DenseMap<ParameterAttr, unsigned> roles;
  kernel.walk([&](StoreOp store) {
    auto value = dyn_cast<FragmentType>(store.getValue().getType());
    if (!value || value.getShape().size() != 2)
      return;
    SmallVector<std::pair<int64_t, ParameterAttr>, 2> axes;
    for (auto [coordinate, resourceAxis] :
         llvm::zip(store.getCoordinates(), store.getSourceAxes())) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      if (type.getShape().size() != 1)
        return;
      auto extent = cast<PhysicalExprAttr>(type.getShape()[0]);
      if (extent.getKind() != PhysicalExprKind::Parameter)
        return;
      auto parameter = queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      if (failed(parameter) || !parameter->getBinding().getPointwiseChunk() ||
          parameter->getCategory() !=
              ParameterCategory::Pointwise ||
          parameter->getCandidates().size() < 2)
        return;
      auto source = sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[0]));
      auto binding = queryParameterBinding(*parameter);
      auto projection = queryFragmentAxis(value, source);
      if (!binding.isExact() || !binding.source || !(*binding.source == source) ||
          !projection.isExact() ||
          value.getShape()[projection.fragmentAxis] != extent)
        return;
      axes.emplace_back(resourceAxis, *parameter);
    }
    if (axes.size() != 2 || axes[0].second == axes[1].second ||
        axes[0].first == axes[1].first)
      return;
    if (axes[0].first > axes[1].first)
      std::swap(axes[0], axes[1]);
    roles[axes[0].second] |= 1;
    roles[axes[1].second] |= 2;
  });
  // Use the resource axis order, not the fragment prefix order introduced by
  // nested-loop lifting. Independent one-dimensional stages keep their roles.
  for (auto [schema, mask] : roles) {
    if (mask == 3)
      continue;
    auto role = mask == 1 ? ParameterRole::OwnershipM : ParameterRole::OwnershipN;
    if (failed(updateParameter(kernel, ParameterAttr::get(
            kernel.getContext(), schema.getName(), schema.getValueType(), role,
            schema.getCategory(), schema.getElementBitWidth(), schema.getCandidates(),
            schema.getPhase(), schema.getBinding())))) return failure();
  }
  return success();
}

} // namespace

static LogicalResult vectorizeBufferLoopsImpl(ModuleOp module) {
  auto physical = getPhysicalKernel(module);
  if (failed(physical))
    return failure();
  func::FuncOp kernel = *physical;
  auto [source, dimension] = nextPhysicalAxisIdentities(kernel);
  if (failed(vectorizeIterations(kernel, source, dimension)))
    return failure();
  if (failed(assignIterationRoles(kernel))) return failure();
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult vectorizeBufferLoops(ModuleOp module) {
  if (failed(vectorizeBufferLoopsImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
