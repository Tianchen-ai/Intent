#include "ProducerVectorization.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool depends(Value value, Value input, Operation *scope,
             llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (value == input) return true;
  Operation *operation = value.getDefiningOp();
  if (!operation || !scope->isAncestor(operation) ||
      !visited.insert(operation).second)
    return false;
  // Region captures and yielded values contribute even when the operation's
  // immediate operands (in particular an if condition) are lane invariant.
  return operation->walk([&](Operation *nested) -> WalkResult {
    for (Value operand : nested->getOperands())
      if (depends(operand, input, scope, visited)) return WalkResult::interrupt();
    return WalkResult::advance();
  }).wasInterrupted();
}

bool availableAt(Value value, OpBuilder &builder, DominanceInfo &dominance) {
  Block *block = builder.getInsertionBlock();
  if (auto argument = dyn_cast<BlockArgument>(value))
    return dominance.dominates(argument.getOwner(), block);
  Operation *definition = value.getDefiningOp();
  return definition && dominance.properlyDominates(
      definition->getBlock(), definition->getIterator(), block,
      builder.getInsertionPoint(), /*enclosingOk=*/false);
}

// This is a dependency-restricted view of an existing proof, not another
// effect/legality analysis. A branch may expose its own immediate operations as
// roots only while rebuilding inside that branch's corresponding control scope.
struct ScalarReplay {
  ScalarReplay(const ProducerReplay &payload, IRMapping &known,
               OpBuilder &builder)
      : payload(payload), known(known), builder(builder) {
    selected.scope = payload.scope;
    selected.nodes = payload.nodes;
  }

  bool collect(Value value) {
    if (!visited.insert(value).second) return true;
    if (known.contains(value) && availableAt(known.lookup(value), builder, dominance)) {
      mapping.map(value, known.lookup(value));
      if (llvm::is_contained(payload.frontier, value))
        selected.frontier.push_back(value);
      return true;
    }
    if (llvm::is_contained(payload.frontier, value)) {
      selected.frontier.push_back(value);
      return false;
    }
    if (llvm::is_contained(payload.externalValues, value)) {
      selected.externalValues.push_back(value);
      return true;
    }
    Operation *operation = value.getDefiningOp();
    if (!operation || !llvm::is_contained(payload.operations, operation))
      return false;
    auto status = operation->walk([&](Operation *nested) -> WalkResult {
      for (Value operand : nested->getOperands()) {
        Operation *owner = operand.getDefiningOp();
        if (!owner) owner = cast<BlockArgument>(operand).getOwner()->getParentOp();
        if (!operation->isAncestor(owner) && !collect(operand))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (status.wasInterrupted()) return false;
    if (!llvm::is_contained(selected.operations, operation))
      selected.operations.push_back(operation);
    return true;
  }

  FailureOr<Value> materialize(Value value) {
    if (!collect(value)) return failure();
    auto result = materializeProducerValue(selected, value, builder, mapping);
    if (failed(result)) return failure();
    // Native region cloning also records nested definitions. They must remain
    // local to those branches, rather than becoming a scalar cache above them.
    for (auto [original, replacement] : mapping.getValueMap())
      if (original == value || llvm::is_contained(selected.frontier, original) ||
          llvm::is_contained(selected.externalValues, original) ||
          llvm::is_contained(selected.operations, original.getDefiningOp()))
        known.map(original, replacement);
    return result;
  }

  const ProducerReplay &payload;
  IRMapping &known;
  OpBuilder &builder;
  DominanceInfo dominance;
  ProducerReplay selected;
  IRMapping mapping;
  llvm::SmallDenseSet<Value> visited;
};

bool supportedRegions(Operation *operation) {
  return !operation->walk([](Operation *nested) -> WalkResult {
    if (nested->getNumRegions() && !isa<scf::IfOp>(nested))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }).wasInterrupted();
}

class Widening {
public:
  Widening(const ProducerReplay &payload, Value coordinate, int64_t width,
           OpBuilder &builder, IRMapping &scalars, IRMapping &vectors,
           OpBuilder *invariants, ValueRange varyingFrontier = {})
      : payload(payload), coordinate(coordinate), adapter(payload, coordinate, varyingFrontier),
        width(width), builder(builder), scalars(scalars), vectors(vectors),
        invariants(invariants), varyingFrontier(varyingFrontier.begin(), varyingFrontier.end()) {}

  FailureOr<Value> scalar(Value value) {
    return adapter.materializeScalar(value, builder, scalars, invariants);
  }

  FailureOr<Value> value(Value original) {
    if (vectors.contains(original)) return vectors.lookup(original);
    if (!ProducerVectorization::isElementType(original.getType())) return failure();
    auto type = VectorType::get({width}, original.getType());
    Location loc = original.getLoc();
    Value result;
    if (original == coordinate) {
      auto base = scalar(original);
      if (failed(base)) return failure();
      Value start = builder.create<vector::BroadcastOp>(loc, type, *base);
      Value lanes = builder.create<vector::StepOp>(loc, type);
      result = builder.create<arith::AddIOp>(loc, start, lanes);
    } else if (adapter.isUniform(original)) {
      auto uniform = scalar(original);
      if (failed(uniform)) return failure();
      result = builder.create<vector::BroadcastOp>(loc, type, *uniform);
    } else if (auto load = original.getDefiningOp<memref::LoadOp>()) {
      if (llvm::all_of(load.getIndices(), [&](Value index) {
            return adapter.coefficient(index) == 0;
          })) {
        auto uniform = scalar(original);
        if (failed(uniform)) return failure();
        result = builder.create<vector::BroadcastOp>(loc, type, *uniform);
        vectors.map(original, result);
        return result;
      }
      auto memory = scalar(load.getMemref());
      if (failed(memory)) return failure();
      SmallVector<Value> indices;
      for (Value index : load.getIndices()) {
        auto mapped = scalar(index);
        if (failed(mapped)) return failure();
        indices.push_back(*mapped);
      }
      auto widened = builder.create<vector::LoadOp>(loc, type, *memory, indices,
                                                   load.getNontemporal());
      widened->setAttrs(load->getAttrs());
      result = widened;
    } else if (auto branch = original.getDefiningOp<scf::IfOp>()) {
      if (failed(conditional(branch))) return failure();
      return vectors.lookup(original);
    } else {
      Operation *operation = original.getDefiningOp();
      if (!operation || operation->getNumRegions() || operation->getNumResults() != 1 ||
          !operation->hasTrait<OpTrait::Elementwise>())
        return failure();
      IRMapping mapping;
      for (Value operand : operation->getOperands()) {
        auto widened = value(operand);
        if (failed(widened)) return failure();
        mapping.map(operand, *widened);
      }
      Operation *cloned = builder.clone(*operation, mapping);
      cloned->getResult(0).setType(type);
      result = cloned->getResult(0);
    }
    vectors.map(original, result);
    return result;
  }

private:
  LogicalResult conditional(scf::IfOp original) {
    if (!adapter.isUniform(original.getCondition())) return failure();
    auto condition = scalar(original.getCondition());
    if (failed(condition)) return failure();
    SmallVector<Type> types;
    for (Type type : original.getResultTypes()) {
      if (!ProducerVectorization::isElementType(type)) return failure();
      types.push_back(VectorType::get({width}, type));
    }
    auto replacement = builder.create<scf::IfOp>(original.getLoc(), types, *condition, true);
    replacement->setAttrs(original->getAttrs());
    for (auto [source, target] : llvm::zip(original->getRegions(), replacement->getRegions())) {
      OpBuilder branchBuilder = OpBuilder::atBlockBegin(&target.front());
      ProducerReplay branchPayload = payload;
      for (Operation &operation : source.front().without_terminator())
        if (!llvm::is_contained(branchPayload.operations, &operation))
          branchPayload.operations.push_back(&operation);
      IRMapping branchScalars = scalars, branchVectors = vectors;
      // Never use the outer invariant insertion point inside a conditional:
      // scalar loads and non-speculatable arithmetic belong to this branch.
      Widening nested(branchPayload, coordinate, width, branchBuilder,
                      branchScalars, branchVectors, nullptr, varyingFrontier);
      SmallVector<Value> yielded;
      for (Value input : source.front().getTerminator()->getOperands()) {
        auto widened = nested.value(input);
        if (failed(widened)) {
          replacement.erase();
          return failure();
        }
        yielded.push_back(*widened);
      }
      branchBuilder.create<scf::YieldOp>(original.getLoc(), yielded);
    }
    for (auto [before, after] : llvm::zip(original.getResults(), replacement.getResults()))
      vectors.map(before, after);
    return success();
  }

  const ProducerReplay &payload;
  Value coordinate;
  ProducerVectorization adapter;
  int64_t width;
  OpBuilder &builder;
  IRMapping &scalars;
  IRMapping &vectors;
  OpBuilder *invariants;
  SmallVector<Value> varyingFrontier;
};

} // namespace

bool ProducerVectorization::dependsOn(Value value, Value input) const {
  llvm::SmallPtrSet<Operation *, 16> visited;
  return depends(value, input, payload.scope, visited);
}

bool ProducerVectorization::isUniform(Value value) const {
  return (!coordinate || !dependsOn(value, coordinate)) &&
      llvm::none_of(varyingFrontier, [&](Value input) { return dependsOn(value, input); });
}

std::optional<int64_t> ProducerVectorization::coefficient(Value value) const {
  if (value == coordinate) return 1;
  // A bound varying element is not necessarily an affine loop coordinate.
  // In particular a zero-operand linalg.index must not fall through to the
  // vacuous all-operands-uniform rule below.
  if (llvm::is_contained(varyingFrontier, value)) return std::nullopt;
  if (isUniform(value)) return 0;
  // Intent's logical index coordinates have the signed 64-bit contract. This
  // is the same affine coefficient query previously owned by VectorizeLoops.
  auto folded = foldIntegerDifference(describeScalarValue(value),
      [&](Value input) { return coefficient(input); },
      [](Value input) { return getConstantIntValue(input); }, /*indexBitWidth=*/64);
  if (folded) return folded;
  Operation *operation = value.getDefiningOp();
  if (operation && !operation->getNumRegions() &&
      llvm::all_of(operation->getOperands(), [&](Value input) {
        return coefficient(input) == 0;
      })) return 0;
  return std::nullopt;
}

bool ProducerVectorization::isElementType(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
      isa<Float8E4M3FNType, Float8E5M2Type>(type) || type.isIndex() ||
      type.isSignlessInteger(1) || type.isSignlessInteger(8) ||
      type.isSignlessInteger(16) || type.isSignlessInteger(32) ||
      type.isSignlessInteger(64);
}

bool ProducerVectorization::canWiden(
    Value value, SmallVectorImpl<Value> *guardedMemories) const {
  if (!isElementType(value.getType())) return false;
  if (value == coordinate || llvm::is_contained(varyingFrontier, value)) return true;
  Operation *operation = value.getDefiningOp();
  if (operation && llvm::is_contained(payload.nodes, operation) &&
      !supportedRegions(operation)) return false;
  if (isUniform(value)) return true;
  if (!operation || !llvm::is_contained(payload.nodes, operation)) return false;
  if (auto load = dyn_cast<memref::LoadOp>(operation)) {
    auto type = load.getMemRefType();
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!type.getRank() || type.getElementType().isInteger(1) ||
        !isUniform(load.getMemref()) ||
        failed(type.getStridesAndOffset(strides, offset))) return false;
    if (llvm::all_of(load.getIndices(), [&](Value index) {
          return coefficient(index) == 0;
        })) return true;
    for (auto [axis, index] : llvm::enumerate(load.getIndices())) {
      auto factor = coefficient(index);
      if (!factor || *factor != (axis + 1 == type.getRank() ? 1 : 0)) return false;
    }
    if (strides.back() == 1) return true;
    Operation *owner = load.getMemref().getDefiningOp();
    if (!guardedMemories || !ShapedType::isDynamic(strides.back()) ||
        (owner && payload.scope->isAncestor(owner))) return false;
    if (!llvm::is_contained(*guardedMemories, load.getMemref()))
      guardedMemories->push_back(load.getMemref());
    return true;
  }
  if (auto branch = dyn_cast<scf::IfOp>(operation)) {
    if (!isUniform(branch.getCondition()) || branch.getElseRegion().empty() ||
        !llvm::all_of(branch.getResultTypes(), isElementType)) return false;
    for (Region &region : branch->getRegions())
      for (Value yielded : region.front().getTerminator()->getOperands())
        if (!canWiden(yielded, guardedMemories)) return false;
    return true;
  }
  return !operation->getNumRegions() && operation->getNumResults() == 1 &&
      operation->hasTrait<OpTrait::Elementwise>() &&
      llvm::all_of(operation->getOperands(), [&](Value input) {
        return canWiden(input, guardedMemories);
      });
}

FailureOr<Value> ProducerVectorization::materializeScalar(
    Value value, OpBuilder &builder, IRMapping &scalars, OpBuilder *invariants) const {
  // An already materialized scalar remains usable here even when it cannot be
  // hoisted. Other cached values are filtered against the chosen insertion point.
  DominanceInfo dominance;
  if (scalars.contains(value) && availableAt(scalars.lookup(value), builder, dominance))
    return scalars.lookup(value);
  bool invariant = invariants && llvm::none_of(payload.frontier, [&](Value input) {
    return dependsOn(value, input);
  });
  ScalarReplay replay(payload, scalars, invariant ? *invariants : builder);
  return replay.materialize(value);
}

FailureOr<Value> ProducerVectorization::materialize(
    Value value, int64_t width, OpBuilder &builder, IRMapping &scalars,
    IRMapping &vectors, OpBuilder *invariants) const {
  if (width <= 0) return failure();
  Widening widening(payload, coordinate, width, builder, scalars, vectors, invariants,
                    varyingFrontier);
  return widening.value(value);
}

} // namespace intent::cpu
