#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "../Access/AccessComposition.h"
#include "../Value/ScopePlacement.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool sameBound(Value lhs, Value rhs, ArrayAttr tuples) {
  if (samePhysicalScalarExpression(lhs, rhs))
    return true;
  PhysicalExprAttr left = queryLaunchExpression(lhs);
  PhysicalExprAttr right = queryLaunchExpression(rhs);
  if (left && left == right)
    return true;
  auto first = lhs.getDefiningOp<ParameterOp>();
  auto second = rhs.getDefiningOp<ParameterOp>();
  if (!first || !second || !tuples || tuples.empty())
    return false;
  return llvm::all_of(tuples, [&](Attribute attribute) {
    auto tuple = cast<DictionaryAttr>(attribute);
    auto a = tuple.getAs<IntegerAttr>(first.getDeclaration().getName().getValue());
    auto b = tuple.getAs<IntegerAttr>(second.getDeclaration().getName().getValue());
    return a && b && a == b;
  });
}

// Equivalence under the one induction-variable substitution that sibling
// fusion actually performs. In particular, equal initial values do not make
// two loop-carried arguments the same evolving value.
class SharedTraversalValues {
public:
  SharedTraversalValues(scf::ForOp first, scf::ForOp second)
      : first(first), second(second) {}

  bool hasReusableWork() {
    bool usefulSharing = false;
    firstSize = std::distance(first.getBody()->begin(),
                              first.getBody()->getTerminator()->getIterator());
    for (auto [rightIndex, rhs] :
         llvm::enumerate(second.getBody()->without_terminator())) {
      if (rhs.use_empty())
        continue;
      for (auto [leftIndex, lhs] :
           llvm::enumerate(first.getBody()->without_terminator())) {
        if (!equivalent(&lhs, &rhs))
          continue;
        shared.emplace_back(leftIndex, rightIndex);
        usefulSharing |= useful(&rhs);
        break;
      }
    }
    return usefulSharing;
  }

  void mergeClonedValues(scf::ForOp fused) {
    SmallVector<Operation *> body;
    for (Operation &operation : fused.getBody()->without_terminator())
      body.push_back(&operation);
    // The native sibling utility clones the complete first body followed by
    // the complete second body. All pairs refer to this one fresh body; they
    // are never reused after a subsequent rewrite.
    for (auto [left, right] : shared)
      body[firstSize + right]->replaceAllUsesWith(body[left]->getResults());
    for (auto [left, right] : llvm::reverse(shared))
      body[firstSize + right]->erase();
  }

private:
  bool useful(Operation *operation) {
    if (operation->use_empty())
      return false;
    if (isa<LoadOp>(operation))
      return true;
    return placement::isMovableValueOperation(operation) &&
           operation->hasTrait<OpTrait::Elementwise>() &&
           llvm::any_of(operation->getResultTypes(),
                        [](Type type) { return isa<FragmentType>(type); });
  }

  bool equivalent(Value lhs, Value rhs) {
    if (lhs == rhs)
      return true;
    if (lhs.getType() != rhs.getType())
      return false;
    if (lhs == first.getInductionVar() && rhs == second.getInductionVar())
      return true;
    auto left = dyn_cast<OpResult>(lhs);
    auto right = dyn_cast<OpResult>(rhs);
    if (!left || !right || left.getResultNumber() != right.getResultNumber())
      return false;
    auto key = std::make_pair(lhs, rhs);
    if (auto found = values.find(key); found != values.end())
      return found->second;
    bool same = equivalent(left.getOwner(), right.getOwner());
    values.try_emplace(key, same);
    return same;
  }

  bool equivalent(Operation *lhs, Operation *rhs) {
    if (lhs->getNumRegions() || rhs->getNumRegions() ||
        lhs->getName() != rhs->getName() ||
        !llvm::equal(lhs->getResultTypes(), rhs->getResultTypes()) ||
        lhs->getNumOperands() != rhs->getNumOperands())
      return false;
    if (isa<LoadOp>(lhs)) {
      // Only the reads in the two traversals have a joint independence proof.
      // Distinct captured reads may have observed different memory contents.
      if (lhs->getBlock() != first.getBody() ||
          rhs->getBlock() != second.getBody())
        return false;
    } else if (!placement::isMovableValueOperation(lhs) ||
               !placement::isMovableValueOperation(rhs)) {
      return false;
    }
    NamedAttrList leftAttrs(lhs->getAttrs()), rightAttrs(rhs->getAttrs());
    leftAttrs.erase(originAttr);
    rightAttrs.erase(originAttr);
    if (leftAttrs != rightAttrs ||
        lhs->getPropertiesAsAttribute() != rhs->getPropertiesAsAttribute())
      return false;
    return llvm::all_of(llvm::zip(lhs->getOperands(), rhs->getOperands()),
                        [&](auto pair) {
      return equivalent(std::get<0>(pair), std::get<1>(pair));
    });
  }

  scf::ForOp first, second;
  llvm::DenseMap<std::pair<Value, Value>, bool> values;
  SmallVector<std::pair<unsigned, unsigned>> shared;
  unsigned firstSize = 0;
};

PhysicalExprAttr carryFootprint(Type type, Builder &builder) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragmentRegisterFootprint(fragment);
  auto constant = [&](int64_t words) {
    return PhysicalExprAttr::get(builder.getContext(), PhysicalExprKind::Constant,
                                 words, builder.getStringAttr(""),
                                 builder.getArrayAttr({}));
  };
  if (auto record = dyn_cast<RecordType>(type)) {
    PhysicalExprAttr total = constant(0);
    for (Attribute field : record.getFieldTypes()) {
      auto words = carryFootprint(cast<TypeAttr>(field).getValue(), builder);
      if (!words)
        return {};
      total = PhysicalExprAttr::get(builder.getContext(), PhysicalExprKind::Add,
                                    0, builder.getStringAttr(""),
                                    builder.getArrayAttr({total, words}));
    }
    return total;
  }
  if (!type.isIntOrIndexOrFloat())
    return {};
  unsigned bits = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return constant(std::max(1u, (bits + 31) / 32));
}

bool boundedCombinedCarries(scf::ForOp first, scf::ForOp second,
                            func::FuncOp kernel) {
  if (!first.getNumResults() || !second.getNumResults())
    return true;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return false;
  Builder builder(kernel.getContext());
  PhysicalExprAttr total;
  for (scf::ForOp loop : {first, second}) {
    for (Type type : loop.getResultTypes()) {
      auto words = carryFootprint(type, builder);
      if (!words)
        return false;
      total = !total ? words : PhysicalExprAttr::get(
          kernel.getContext(), PhysicalExprKind::Add, 0,
          builder.getStringAttr(""), builder.getArrayAttr({total, words}));
    }
  }
  auto limit = PhysicalExprAttr::get(
      kernel.getContext(), PhysicalExprKind::Constant,
      capabilities.getRegistersPerUnit(), builder.getStringAttr(""),
      builder.getArrayAttr({}));
  // This limits the newly simultaneous carried payload. It is not a register
  // allocation or occupancy proof, and does not change the candidate set.
  return configurationExpressionAtMost(kernel, total, limit);
}

bool compatibleLoopAttributes(scf::ForOp first, scf::ForOp second) {
  NamedAttrList lhs(first->getAttrs()), rhs(second->getAttrs());
  for (StringRef name : {originAttr, reductionSourcesAttr,
                         independentIterationAttr}) {
    lhs.erase(name);
    rhs.erase(name);
  }
  return lhs == rhs;
}

bool tryFuse(scf::ForOp first, scf::ForOp second, func::FuncOp kernel,
             ArrayAttr tuples) {
  if (first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second) ||
      !sameBound(first.getLowerBound(), second.getLowerBound(), tuples) ||
      !sameBound(first.getUpperBound(), second.getUpperBound(), tuples) ||
      !sameBound(first.getStep(), second.getStep(), tuples))
    return false;
  SharedTraversalValues shared(first, second);
  if (!compatibleLoopAttributes(first, second) ||
      !boundedCombinedCarries(first, second, kernel) ||
      !shared.hasReusableWork())
    return false;
  ResourceAliasAnalysis aliases;
  if (!placement::independentMemoryEffects(first, second, aliases))
    return false;
  for (Operation *operation = first->getNextNode(); operation != second;
       operation = operation->getNextNode())
    if (!placement::independentMemoryEffects(operation, second, aliases))
      return false;

  if (!placement::moveInputsBefore(first, second, kernel))
    return false;

  // The complete candidate set proves equal granularity even when the two
  // traversals retain distinct parameter roles in their fragment schemas.
  second.setLowerBound(first.getLowerBound());
  second.setUpperBound(first.getUpperBound());
  second.setStep(first.getStep());
  auto attributes = first->getAttrDictionary();
  SmallVector<Attribute> sources;
  for (scf::ForOp loop : {first, second})
    if (auto axes = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr))
      for (Attribute axis : axes)
        if (!llvm::is_contained(sources, axis)) sources.push_back(axis);
  bool independent = first->hasAttr(independentIterationAttr) &&
                     second->hasAttr(independentIterationAttr);
  IRRewriter rewriter(kernel.getContext());
  Location location = rewriter.getFusedLoc({first.getLoc(), second.getLoc()});
  second->moveBefore(first);
  scf::ForOp fused = mlir::fuseIndependentSiblingForLoops(first, second, rewriter);
  shared.mergeClonedValues(fused);
  fused->setLoc(location);
  fused->setAttrs(attributes);
  if (!sources.empty())
    fused->setAttr(reductionSourcesAttr, rewriter.getArrayAttr(sources));
  if (!independent || !fused.getInitArgs().empty())
    fused->removeAttr(independentIterationAttr);
  return true;
}

ReduceOp tryFuse(ReduceOp first, ReduceOp second, func::FuncOp kernel) {
  if (first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second) || first.getAxes() != second.getAxes())
    return {};
  auto shape = dyn_cast<FragmentType>(first.getSources().front().getType());
  if (!shape)
    return {};
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  UniformValueAnalysis constants(describeUniformValue);
  for (ReduceOp reduce : {first, second}) {
    // Joint custom primitives can have a narrower identity domain than a
    // provider's single-component builtins. Keep those builtins available when
    // the new tuple would not have a legal native identity.
    if (capabilities.getNativeTupleReductionRequiresConstantIdentity() &&
        llvm::any_of(reduce.getIdentities(), [&](Value identity) {
          return !constants.evaluate(identity);
        }))
      return {};
    for (NamedAttribute attribute : reduce->getDiscardableAttrs())
      if (attribute.getName() != originAttr)
        return {};
    for (Value source : reduce.getSources()) {
      auto type = dyn_cast<FragmentType>(source.getType());
      if (!type || type.getShape() != shape.getShape() ||
          type.getAxisMaps() != shape.getAxisMaps() ||
          type.getValidity() != shape.getValidity() ||
          type.getOwner() != shape.getOwner())
        return {};
    }
  }
  if (!placement::moveInputsBefore(first, second, kernel))
    return {};

  SmallVector<Value> sources, identities, captures;
  for (ReduceOp reduce : {first, second}) {
    llvm::append_range(sources,
                       reduce.getSources());
    llvm::append_range(identities, reduce.getIdentities());
    llvm::append_range(captures, reduce.getCaptures());
  }
  OpBuilder builder(first);
  Location location = builder.getFusedLoc({first.getLoc(), second.getLoc()});
  auto fused = builder.create<ReduceOp>(location, sources, identities,
                                        captures, first.getAxes());
  if (Attribute origin = first->getAttr(originAttr))
    fused->setAttr(originAttr, origin);
  auto target = cast<StructuredOpInterface>(fused.getOperation());
  auto arguments = target.getCombineArgumentTypes();
  builder.createBlock(&fused.getCombine(), {}, arguments,
                      SmallVector<Location>(arguments.size(), location));
  SmallVector<Value> yields;
  unsigned componentOffset = 0, captureOffset = 0;
  for (ReduceOp reduce : {first, second}) {
    Block &combine = reduce.getCombine().front();
    auto source = cast<StructuredOpInterface>(reduce.getOperation());
    unsigned size = reduce.getSources().size();
    IRMapping mapping;
    for (unsigned index = 0; index < size; ++index) {
      mapping.map(source.getCombineLhs()[index], target.getCombineLhs()[componentOffset + index]);
      mapping.map(source.getCombineRhs()[index], target.getCombineRhs()[componentOffset + index]);
    }
    for (unsigned index = 0; index < reduce.getCaptures().size(); ++index)
      mapping.map(source.getCombineCaptures()[index], target.getCombineCaptures()[captureOffset + index]);
    for (Operation &operation : combine.without_terminator())
      builder.clone(operation, mapping);
    for (Value value : combine.getTerminator()->getOperands())
      yields.push_back(mapping.lookupOrDefault(value));
    componentOffset += size;
    captureOffset += reduce.getCaptures().size();
  }
  builder.create<YieldOp>(location, yields);
  first->replaceAllUsesWith(fused.getResults().take_front(first.getNumResults()));
  second->replaceAllUsesWith(fused.getResults().take_back(second.getNumResults()));
  first.erase();
  second.erase();
  return fused;
}

} // namespace

LogicalResult fuseIndependentReductions(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (!kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr)
           .getNativeTupleReductions())
    return success();
  kernel.walk<WalkOrder::PostOrder>([&](Block *block) {
    SmallVector<ReduceOp> reductions(block->getOps<ReduceOp>());
    for (unsigned first = 0; first < reductions.size(); ++first) {
      if (!reductions[first])
        continue;
      for (unsigned second = first + 1; second < reductions.size(); ++second) {
        if (!reductions[second])
          continue;
        if (ReduceOp fused =
                tryFuse(reductions[first], reductions[second], kernel)) {
          reductions[first] = fused;
          reductions[second] = {};
        }
      }
    }
  });
  return success();
}

LogicalResult fuseIndependentTraversals(ModuleOp module,
                                       bool hoistLoopInvariants) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  auto configurations =
      kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
  auto tuples = configurations ? configurations.getRows() : ArrayAttr();
  bool changed;
  do {
    changed = false;
    SmallVector<scf::ForOp> loops;
    kernel.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp first : loops) {
      for (scf::ForOp second : loops)
        if (first != second && tryFuse(first, second, kernel, tuples)) {
          changed = true;
          break;
        }
      if (changed)
        break;
    }
  } while (changed);
  auto simplify = [&]() {
    return hoistLoopInvariants ? hoistLoopInvariantValues(module)
                               : eliminateCommonValues(module);
  };
  if (failed(simplify()))
    return failure();
  if (access::reuseStableLoads(kernel) && failed(simplify()))
    return failure();
  access::sinkStableLoadChains(kernel);
  return success();
}

} // namespace intent::gpu
