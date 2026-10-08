#include "Intent/Dialect/CPU/Transforms/Collective/Collectives.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include <limits>

using namespace mlir;

namespace intent::cpu {
namespace {

bool scalarType(Type type) {
  return isa<FloatType, IntegerType, IndexType>(type);
}

// These are completed ordinary reductions, before implementation selection.
// Flatten their logical members, not a provider's vector or reduction tree.
bool collapseReductionChain(linalg::GenericOp outer, StorageAnalysis &storage) {
  if (outer.getNumResults() || outer.getInputs().size() != 1 ||
      outer.getOutputs().size() != 1 || !outer.getNumLoops() ||
      outer.getNumReductionLoops() != outer.getNumLoops() ||
      outer->hasAttr("intent_cpu.implementation")) return false;
  auto order = outer->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (!order || !order.getAdjacentReassociation() || !order.getElementPermutation())
    return false;
  Value partial = outer.getInputs().front(), destination = outer.getOutputs().front();
  auto partialType = dyn_cast<MemRefType>(partial.getType());
  auto resultType = dyn_cast<MemRefType>(destination.getType());
  auto allocation = partial.getDefiningOp<memref::AllocOp>();
  if (!partialType || !resultType || resultType.getRank() ||
      partialType.getElementType() != resultType.getElementType() ||
      !allocation || allocation->getBlock() != outer->getBlock() ||
      !outer.getIndexingMapsArray().front().isIdentity() ||
      outer.getIndexingMapsArray().back().getNumResults()) return false;
  Block &outerBody = outer.getRegion().front();
  for (Operation &instruction : outerBody.without_terminator())
    if (instruction.getNumRegions() || !isMemoryEffectFree(&instruction) ||
        !isSpeculatable(&instruction)) return false;
  Operation *combine = outerBody.getTerminator()->getOperand(0).getDefiningOp();
  if (!combine || combine->getNumOperands() != 2 || combine->getNumResults() != 1 ||
      combine->getNumRegions() || !isMemoryEffectFree(combine) ||
      !((combine->getOperand(0) == outerBody.getArgument(0) &&
         combine->getOperand(1) == outerBody.getArgument(1)) ||
        (combine->getOperand(1) == outerBody.getArgument(0) &&
         combine->getOperand(0) == outerBody.getArgument(1)))) return false;
  auto neutral = arith::getNeutralElement(combine);
  if (!neutral) return false;
  auto initializedBy = [&](Value buffer, Operation *before) -> std::optional<UniformBufferValue> {
    auto initial = queryUniformBufferValue(buffer, before, storage);
    Attribute value;
    if (!initial || initial->definition.operation->getBlock() != before->getBlock() ||
        !matchPattern(initial->value, m_Constant(&value)) || value != *neutral)
      return std::nullopt;
    return initial;
  };
  if (!initializedBy(destination, outer)) return false;
  auto current = findCurrentBufferWrite(partial, outer, storage);
  auto inner = current ? dyn_cast<linalg::GenericOp>(current->operation) : linalg::GenericOp{};
  if (!inner || inner.getNumResults() || inner.getOutputs().size() != 1 ||
      inner->getBlock() != outer->getBlock() || inner.getOutputs().front() != partial ||
      !inner.getNumReductionLoops() || !inner.getNumParallelLoops() ||
      inner->hasAttr("intent_cpu.implementation") ||
      inner->getAttr("intent_cpu.reduction_order") != order) return false;
  auto initial = initializedBy(partial, inner);
  if (!initial || !isa<linalg::FillOp>(initial->definition.operation)) return false;
  Operation *initialization = initial->definition.operation;
  // Removing the seed write must not change an observation before the partial
  // reduction, even though such a read would preserve the buffer's contents.
  if (initial->definition.observed) return false;
  Block &innerBody = inner.getRegion().front();
  BlockArgument accumulator = innerBody.getArguments().back();
  Operation *update = innerBody.getTerminator()->getOperand(0).getDefiningOp();
  if (!update || update->getName() != combine->getName() ||
      update->getNumOperands() != 2 || update->getNumResults() != 1 ||
      !accumulator.hasOneUse() ||
      !llvm::is_contained(update->getOperands(), accumulator) ||
      update->getResult(0).getType() != combine->getResult(0).getType()) return false;
  NamedAttrList innerAttrs(update->getAttrs()), outerAttrs(combine->getAttrs());
  // `contract` permits a neighboring multiply-add; it does not change this
  // binary reduction's identity. Keep the original member/update operations.
  auto stripContract = [](NamedAttrList &attributes, Operation *operation) {
    if (auto arithmetic = dyn_cast<arith::ArithFastMathInterface>(operation)) {
      auto flags = arithmetic.getFastMathFlagsAttr().getValue() &
                   ~arith::FastMathFlags::contract;
      attributes.set("fastmath", arith::FastMathFlagsAttr::get(operation->getContext(), flags));
    }
  };
  stripContract(innerAttrs, update);
  stripContract(outerAttrs, combine);
  if (innerAttrs != outerAttrs) return false;
  for (Operation &instruction : innerBody.without_terminator())
    if (instruction.getNumRegions() || isa<linalg::IndexOp>(instruction) ||
        !isMemoryEffectFree(&instruction)) return false;

  auto maps = inner.getIndexingMapsArray();
  Value shaped;
  SmallVector<ReassociationIndices> reassociation(1);
  for (unsigned axis = 0; axis < inner.getNumLoops(); ++axis)
    reassociation.front().push_back(axis);
  for (auto [input, map] : llvm::zip(inner.getInputs(), maps)) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type) {
      if (!scalarType(input.getType()) || map.getNumResults()) return false;
      continue;
    }
    if (!map.isIdentity() || type.getRank() != inner.getNumLoops() ||
        !memref::CollapseShapeOp::isGuaranteedCollapsible(type, reassociation) ||
        !storage.disjoint(input, partial) || !storage.readStable(input, inner, outer))
      return false;
    if (!shaped) shaped = input;
    for (unsigned axis = 0; axis < inner.getNumLoops(); ++axis)
      if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(shaped, axis),
                            ValueBoundsConstraintSet::Variable(input, axis))) return false;
  }
  if (!shaped || partialType.getRank() != inner.getNumParallelLoops()) return false;
  llvm::SmallBitVector freeAxes(inner.getNumLoops());
  for (auto [axis, expression] : llvm::enumerate(maps.back().getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || freeAxes.test(dimension.getPosition()) ||
        inner.getIteratorTypesArray()[dimension.getPosition()] != utils::IteratorType::parallel ||
        !haveEqualExtents(ValueBoundsConstraintSet::Variable(shaped, dimension.getPosition()),
                          ValueBoundsConstraintSet::Variable(partial, axis))) return false;
    freeAxes.set(dimension.getPosition());
  }
  int64_t elements = 1;
  for (unsigned axis = 0; axis < inner.getNumLoops(); ++axis) {
    auto bound = constantExtentUpperBound(ValueBoundsConstraintSet::Variable(shaped, axis));
    if (!bound || *bound < 0 || (*bound && elements > std::numeric_limits<int64_t>::max() / *bound))
      return false;
    elements *= *bound;
  }
  auto version = queryCompletedBufferVersion(partial, inner, storage);
  if (failed(version) || version->uses.size() != 1 ||
      version->uses.front().operation != outer || version->uses.front().input != 0)
    return false;

  OpBuilder builder(outer);
  SmallVector<Value> inputs;
  SmallVector<AffineMap> flatMaps;
  for (Value input : inner.getInputs()) {
    bool memory = isa<MemRefType>(input.getType());
    inputs.push_back(memory ? builder.create<memref::CollapseShapeOp>(outer.getLoc(), input,
                                                                     reassociation).getResult()
                            : input);
    flatMaps.push_back(memory ? builder.getMultiDimIdentityMap(1)
                             : AffineMap::get(1, 0, {}, builder.getContext()));
  }
  flatMaps.push_back(AffineMap::get(1, 0, {}, builder.getContext()));
  auto merged = builder.create<linalg::GenericOp>(outer.getLoc(), inputs,
      outer.getOutputs(), flatMaps,
      ArrayRef<utils::IteratorType>{utils::IteratorType::reduction},
      [&](OpBuilder &nested, Location location, ValueRange arguments) {
        IRMapping mapping;
        mapping.map(innerBody.getArguments(), arguments);
        for (Operation &instruction : innerBody.without_terminator())
          nested.clone(instruction, mapping);
        nested.create<linalg::YieldOp>(location,
            mapping.lookupOrDefault(innerBody.getTerminator()->getOperand(0)));
      });
  merged->setDiscardableAttrs(llvm::to_vector(inner->getDiscardableAttrs()));
  outer.erase();
  inner.erase();
  initialization->erase();
  return true;
}

bool collapseOneReductionChain(func::FuncOp function) {
  SmallVector<linalg::GenericOp> operations;
  function.walk([&](linalg::GenericOp operation) { operations.push_back(operation); });
  StorageAnalysis storage(function);
  for (auto operation : operations)
    if (collapseReductionChain(operation, storage)) return true;
  return false;
}

void normalize(linalg::GenericOp operation) {
  if (operation.getNumResults() || operation.getOutputs().size() != 1 ||
      operation.getNumLoops() != 1 || operation.getNumReductionLoops() != 1 ||
      operation->hasAttr("intent_cpu.implementation"))
    return;
  auto order = operation->getAttrOfType<ReductionOrderAttr>(
      "intent_cpu.reduction_order");
  Value destination = operation.getOutputs().front();
  auto allocation = destination.getDefiningOp<memref::AllocOp>();
  auto type = dyn_cast<MemRefType>(destination.getType());
  if (!order || !allocation || !type || type.getRank() ||
      !scalarType(type.getElementType()) ||
      allocation->getBlock() != operation->getBlock() ||
      llvm::is_contained(operation.getInputs(), destination))
    return;
  StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
  for (Value input : operation.getInputs())
    if (isa<MemRefType>(input.getType()) && !storage.disjoint(input, destination)) return;
  auto initialValue = queryUniformBufferValue(destination, operation, storage);
  if (!initialValue || initialValue->definition.operation->getBlock() != operation->getBlock()) return;

  auto maps = operation.getIndexingMapsArray();
  if (maps.back().getNumResults())
    return;
  Value extentSource;
  int64_t extentAxis = 0;
  for (auto [input, map] : llvm::zip(operation.getInputs(), maps)) {
    auto memory = dyn_cast<MemRefType>(input.getType());
    if (!scalarType(memory ? memory.getElementType() : input.getType()) ||
        map.getNumDims() != 1 || map.getNumSymbols())
      return;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
        if (dimension.getPosition() != 0 || !memory)
          return;
        if (!extentSource) {
          extentSource = input;
          extentAxis = axis;
        }
      } else if (auto constant = dyn_cast<AffineConstantExpr>(expression);
                 !constant || constant.getValue() != 0) {
        return;
      }
    }
  }
  if (!extentSource)
    return;
  Block &body = operation.getRegion().front();
  for (Operation &instruction : body.without_terminator())
    if (instruction.getNumRegions() || isa<linalg::IndexOp>(instruction) ||
        !isMemoryEffectFree(&instruction) ||
        !llvm::all_of(instruction.getOperandTypes(), scalarType) ||
        !llvm::all_of(instruction.getResultTypes(), scalarType))
      return;

  // Replace just the completed scalar version. Earlier or later versions may
  // share this private slot without observing this reduction's result.
  Operation *initialization = initialValue->definition.operation;
  Value initial = initialValue->value;
  auto observations = queryBufferVersionObservations(destination, operation, storage);
  if (failed(observations)) return;
  SmallVector<memref::LoadOp> reads;
  for (Operation *user : observations->reads) {
    if (user->getBlock() != operation->getBlock())
      return;
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      if (load.getMemref() != destination || !load.getIndices().empty() ||
          !operation->isBeforeInBlock(load))
        return;
      reads.push_back(load);
      continue;
    }
    return;
  }
  if (!initialization || initial.getType() != type.getElementType())
    return;

  OpBuilder builder(operation);
  Location location = operation.getLoc();
  Value extent = builder.create<memref::DimOp>(location, extentSource, extentAxis);
  auto reduction = builder.create<ReduceOp>(location, initial.getType(), extent,
      initial, operation.getInputs(),
      builder.getAffineMapArrayAttr(ArrayRef(maps).drop_back()), order);
  NamedAttrList attributes(llvm::to_vector(operation->getDiscardableAttrs()));
  attributes.erase("intent_cpu.reduction_order");
  reduction->setDiscardableAttrs(attributes.getAttrs());
  Block &target = reduction.getCombine().emplaceBlock();
  target.addArgument(initial.getType(), location);
  for (BlockArgument argument : body.getArguments().drop_back())
    target.addArgument(argument.getType(), location);
  builder.setInsertionPointToStart(&target);
  IRMapping mapping;
  mapping.map(body.getArguments().back(), target.getArgument(0));
  mapping.map(body.getArguments().drop_back(), target.getArguments().drop_front());
  for (Operation &instruction : body.without_terminator())
    builder.clone(instruction, mapping);
  builder.create<YieldOp>(location,
      mapping.lookupOrDefault(body.getTerminator()->getOperand(0)));
  for (memref::LoadOp read : reads) {
    read.getResult().replaceAllUsesWith(reduction.getResult());
    read.erase();
  }
  operation.erase();
  if (!initialValue->definition.observed) initialization->erase();
  SmallVector<Operation *> remaining(destination.getUsers());
  if (llvm::all_of(remaining, [](Operation *user) { return isa<memref::DeallocOp>(user); })) {
    for (Operation *user : remaining) user->erase();
    allocation.erase();
  }
}

} // namespace

LogicalResult normalizeScalarReductions(func::FuncOp function) {
  while (collapseOneReductionChain(function)) {}
  SmallVector<linalg::GenericOp> reductions;
  function.walk([&](linalg::GenericOp operation) {
    if (operation.getNumReductionLoops())
      reductions.push_back(operation);
  });
  for (linalg::GenericOp operation : reductions)
    normalize(operation);
  return success();
}

} // namespace intent::cpu
