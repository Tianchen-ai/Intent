#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {

PhysicalExprAttr fragmentElementCount(FragmentType fragment) {
  MLIRContext *context = fragment.getContext();
  auto footprint = PhysicalExprAttr::get(
      context, PhysicalExprKind::Constant,
      1, StringAttr::get(context, ""),
      ArrayAttr::get(context, {}));
  for (Attribute extent : fragment.getShape())
    footprint = PhysicalExprAttr::get(
        context, PhysicalExprKind::Multiply, 0,
        StringAttr::get(context, ""), ArrayAttr::get(context, {footprint, extent}));
  return footprint;
}

PhysicalExprAttr fragmentRegisterFootprint(FragmentType fragment) {
  Type element = fragment.getElementType();
  unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
  auto elements = fragmentElementCount(fragment);
  int64_t words = std::max(1u, (bits + 31) / 32);
  if (words == 1) return elements;
  MLIRContext *context = fragment.getContext();
  auto width = PhysicalExprAttr::get(context, PhysicalExprKind::Constant,
      words, StringAttr::get(context, ""), ArrayAttr::get(context, {}));
  return PhysicalExprAttr::get(context, PhysicalExprKind::Multiply, 0,
      StringAttr::get(context, ""), ArrayAttr::get(context, {width, elements}));
}

std::optional<int64_t> minimumFragmentRegisterFootprint(
    func::FuncOp kernel, Value value, int64_t limit,
    FragmentFootprintScope scope) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || !fragment.getElementType().isIntOrIndexOrFloat() ||
      limit < 0 || limit == std::numeric_limits<int64_t>::max())
    return std::nullopt;
  std::optional<PhysicalProgramAnalysis> analysis;
  if (scope == FragmentFootprintScope::FullScalarSeedCapacity)
    analysis.emplace(kernel);
  Type element = fragment.getElementType();
  unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
  int64_t words = std::max(1u, (bits + 31) / 32);
  const __int128 saturation = static_cast<__int128>(limit) + 1;
  for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    int64_t minimum;
    if (scope == FragmentFootprintScope::FullScalarSeedCapacity &&
        analysis->axisRealization(value, axis).constructionScalarSeed) {
      auto range = queryExactLogicalRange(analysis->axisRanges(value, axis));
      auto capacity = succeeded(range) ? queryLogicalRangeCapacity(*range)
                                       : PhysicalExprAttr();
      auto count = capacity ? constantPhysicalExpression(capacity) : std::nullopt;
      if (!count || *count <= 0) return std::nullopt;
      auto rounded = static_cast<__int128>(
          llvm::PowerOf2Ceil(static_cast<uint64_t>(*count)));
      minimum = static_cast<int64_t>(std::min(saturation, rounded));
    } else if (extent.getKind() == PhysicalExprKind::Constant) {
      minimum = extent.getValue();
    } else if (extent.getKind() == PhysicalExprKind::Parameter) {
      auto parameter = queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      if (failed(parameter)) return std::nullopt;
      minimum = *llvm::min_element(parameter->getCandidates().asArrayRef());
    } else {
      return std::nullopt;
    }
    if (minimum <= 0) return std::nullopt;
    words = static_cast<int64_t>(std::min(saturation, static_cast<__int128>(words) * minimum));
  }
  return words;
}

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf,
    llvm::function_ref<std::optional<int64_t>(ParameterRefAttr)>
        resolveActivation) {
  using Status = RequirementStatus;
  using Predicate = ConfigurationRequirementPredicate;
  if (auto reference = requirement.getActivation()) {
    auto activation =
        resolveActivation ? resolveActivation(reference) : std::nullopt;
    if (!activation)
      return {Status::Unknown, std::nullopt, std::nullopt};
    if (*activation == 0)
      return {Status::Inactive, std::nullopt, std::nullopt};
    if (*activation != 1)
      return {Status::Invalid, std::nullopt, std::nullopt};
  }
  auto usage = evaluatePhysicalExpression(requirement.getUsage(), resolveLeaf);
  auto limit = requirement.getLimit()
                   ? evaluatePhysicalExpression(requirement.getLimit(),
                                                resolveLeaf)
                   : std::nullopt;
  auto predicate = requirement.getPredicate();
  if ((usage && *usage <= 0) || (limit && *limit < 0) ||
      ((predicate == Predicate::MultipleOf || predicate == Predicate::Equal) &&
       limit && *limit == 0))
    return {Status::Invalid, usage, limit};
  if (predicate == Predicate::Positive)
    return {usage ? Status::Satisfied : Status::Unknown, usage, limit};
  if (predicate == Predicate::PowerOfTwo)
    return {!usage ? Status::Unknown
                   : llvm::isPowerOf2_64(*usage) ? Status::Satisfied
                                               : Status::Violated,
            usage, limit};
  if (predicate == Predicate::MultipleOf)
    return {!usage || !limit ? Status::Unknown
                            : *usage % *limit == 0 ? Status::Satisfied
                                                  : Status::Violated,
            usage, limit};
  if (predicate == Predicate::Equal)
    return {!usage || !limit ? Status::Unknown
                            : *usage == *limit ? Status::Satisfied
                                              : Status::Violated,
            usage, limit};
  if (!usage && limit) {
    // Preserve a finite upper-budget decision for positive products whose exact
    // count exceeds i64. Missing leaves and unproved arithmetic remain unknown.
    const __int128 cap = static_cast<__int128>(*limit) + 1;
    std::function<std::optional<__int128>(PhysicalExprAttr)> bounded =
        [&](PhysicalExprAttr expression) -> std::optional<__int128> {
      if (auto exact = evaluatePhysicalExpression(expression, resolveLeaf))
        return *exact < 0
                   ? std::nullopt
                   : std::optional<__int128>(
                         std::min(cap, static_cast<__int128>(*exact)));
      auto kind = expression.getKind();
      if (kind != PhysicalExprKind::Add && kind != PhysicalExprKind::Multiply)
        return std::nullopt;
      auto operands = expression.getOperands();
      auto left = bounded(cast<PhysicalExprAttr>(operands[0]));
      auto right = bounded(cast<PhysicalExprAttr>(operands[1]));
      if (!left || !right)
        return std::nullopt;
      return std::min(cap, kind == PhysicalExprKind::Add ? *left + *right
                                                       : *left * *right);
    };
    if (auto count = bounded(requirement.getUsage()); count && *count > *limit)
      return {Status::Violated, usage, limit};
  }
  if (!usage || !limit)
    return {Status::Unknown, usage, limit};
  return {*usage <= *limit ? Status::Satisfied : Status::Violated,
          usage, limit};
}

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, DictionaryAttr bindings) {
  return evaluateConfigurationRequirement(
      requirement,
      [&](PhysicalExprAttr leaf) -> std::optional<int64_t> {
        if (leaf.getKind() != PhysicalExprKind::Parameter || !bindings)
          return std::nullopt;
        auto value =
            bindings.getAs<IntegerAttr>(leaf.getParameterReference().getName());
        return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
      },
      [&](ParameterRefAttr reference) -> std::optional<int64_t> {
        auto value = bindings ? bindings.getAs<IntegerAttr>(reference.getName())
                              : IntegerAttr();
        // Configuration rows store every declaration, including i1 choices,
        // as i64 bindings. Do not sign-extend an SSA i1 true into -1 here.
        return value && value.getType().isSignlessInteger(64)
                   ? std::optional<int64_t>(value.getInt())
                   : std::nullopt;
      });
}

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, DictionaryAttr bindings,
    func::FuncOp kernel) {
  auto exact = evaluateConfigurationRequirement(requirement, bindings);
  if (exact.status != RequirementStatus::Unknown || !exact.limit ||
      requirement.getActivation() ||
      requirement.getKind() != ConfigurationRequirementKind::NominalBudget ||
      requirement.getMetric() != ConfigurationRequirementMetric::FragmentRegisterWords ||
      requirement.getPredicate() != ConfigurationRequirementPredicate::LessEqual)
    return exact;
  std::function<bool(PhysicalExprAttr)> monotone = [&](PhysicalExprAttr value) {
    auto kind = value.getKind();
    if (kind == PhysicalExprKind::Constant) return value.getValue() >= 0;
    if (kind == PhysicalExprKind::Parameter) return true;
    return (kind == PhysicalExprKind::Add || kind == PhysicalExprKind::Multiply) &&
        llvm::all_of(value.getOperands(), [&](Attribute operand) {
          return monotone(cast<PhysicalExprAttr>(operand));
        });
  };
  if (!monotone(requirement.getUsage())) return exact;
  auto lower = evaluateConfigurationRequirement(requirement,
      [&](PhysicalExprAttr leaf) -> std::optional<int64_t> {
        if (leaf.getKind() != PhysicalExprKind::Parameter) return std::nullopt;
        auto name = leaf.getParameterReference().getName();
        if (auto value = bindings ? bindings.getAs<IntegerAttr>(name) : IntegerAttr())
          return value.getInt() > 0 ? std::optional<int64_t>(value.getInt()) : std::nullopt;
        auto parameter = lookupParameter(kernel, leaf.getParameterReference());
        if (!parameter || !parameter.isExtent() || !parameter.isDeferred())
          return std::nullopt;
        auto candidates = parameter.getCandidates().asArrayRef();
        if (candidates.empty()) return std::nullopt;
        int64_t minimum = *llvm::min_element(candidates);
        return minimum > 0 ? std::optional<int64_t>(minimum) : std::nullopt;
      });
  return lower.status == RequirementStatus::Violated ? lower : exact;
}

namespace {

// Shape views keep their underlying payload alive; they do not allocate a
// second full fragment. Numeric casts and computations remain distinct values.
void collectPayloads(Value value, llvm::DenseSet<Value> &payloads,
                     bool completeTypes = false) {
  if (auto view = value.getDefiningOp<BroadcastOp>())
    return collectPayloads(view.getValue(), payloads, completeTypes);
  if (auto view = value.getDefiningOp<ReshapeOp>())
    return collectPayloads(view.getValue(), payloads, completeTypes);
  if (auto view = value.getDefiningOp<SplatOp>())
    return collectPayloads(view.getValue(), payloads, completeTypes);
  if (completeTypes) {
    if (auto view = value.getDefiningOp<TransposeOp>())
      return collectPayloads(view.getValue(), payloads, completeTypes);
    if (auto join = value.getDefiningOp<JoinOp>()) {
      for (Value operand : join->getOperands()) collectPayloads(operand, payloads, completeTypes);
      return;
    }
  }
  if (auto record = value.getDefiningOp<MakeRecordOp>()) {
    for (Value field : record.getFields()) collectPayloads(field, payloads, completeTypes);
    return;
  }
  if (auto field = value.getDefiningOp<ExtractOp>()) {
    if (auto record = field.getRecord().getDefiningOp<MakeRecordOp>())
      return collectPayloads(record.getFields()[field.getField()], payloads, completeTypes);
    // An opaque carried record owns its fields together. Count it once even
    // when several field views and the record itself span the collective.
    return collectPayloads(field.getRecord(), payloads, completeTypes);
  }
  if (isa<FragmentType, RecordType>(value.getType()) ||
      (completeTypes && !isa<ViewType, BufferType>(value.getType()))) payloads.insert(value);
}

bool collectLivePayloads(Operation *point, const Liveness &liveness,
                         llvm::DenseSet<Value> &payloads,
                         bool completeTypes = false) {
  auto *block = liveness.getLiveness(point->getBlock());
  if (!block) return false;
  for (Value value : block->currentlyLiveValues(point))
    collectPayloads(value, payloads, completeTypes);
  // A nested stage also spans values retained by its enclosing operation for
  // later consumers. Exclude the parent's not-yet-produced results.
  for (Operation *parent = point->getParentOp();
       parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp()) {
    auto *parentBlock = liveness.getLiveness(parent->getBlock());
    if (!parentBlock) {
      if (completeTypes) return false;
      continue;
    }
    for (Value value : parentBlock->currentlyLiveValues(parent)) {
      if (value.getDefiningOp() == parent || liveness.isDeadAfter(value, parent))
        continue;
      collectPayloads(value, payloads, completeTypes);
    }
  }
  return true;
}

SmallVector<Value> orderedKernelValues(func::FuncOp kernel) {
  SmallVector<Value> values;
  kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    llvm::append_range(values, operation->getResults());
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        llvm::append_range(values, block.getArguments());
  });
  return values;
}

PhysicalExprAttr wordConstant(MLIRContext *context, int64_t words) {
  return PhysicalExprAttr::get(context, PhysicalExprKind::Constant, words,
      StringAttr::get(context, ""), ArrayAttr::get(context, {}));
}

PhysicalExprAttr combineWords(PhysicalExprAttr lhs, PhysicalExprAttr rhs,
                              PhysicalExprKind kind) {
  if (!lhs || !rhs) return {};
  if (kind == PhysicalExprKind::Maximum && lhs == rhs) return lhs;
  auto left = constantPhysicalExpression(lhs), right = constantPhysicalExpression(rhs);
  if (left && right) {
    if (kind == PhysicalExprKind::Maximum)
      return wordConstant(lhs.getContext(), std::max(*left, *right));
    if (kind == PhysicalExprKind::Add && *left >= 0 && *right >= 0 &&
        *left <= std::numeric_limits<int64_t>::max() - *right)
      return wordConstant(lhs.getContext(), *left + *right);
  }
  auto constant = [](PhysicalExprAttr value, int64_t integer) {
    return value.getKind() == PhysicalExprKind::Constant && value.getValue() == integer;
  };
  if (kind == PhysicalExprKind::Add) {
    if (constant(lhs, 0)) return rhs;
    if (constant(rhs, 0)) return lhs;
  }
  return PhysicalExprAttr::get(lhs.getContext(), kind, 0,
      StringAttr::get(lhs.getContext(), ""), ArrayAttr::get(lhs.getContext(), {lhs, rhs}));
}

struct NominalStage {
  explicit NominalStage(PhysicalExprAttr value) : expression(value) {}
  PhysicalExprAttr expression;
  SmallVector<std::pair<unsigned, unsigned>> terms;
  int64_t constant = 0;
  bool nonnegative = false;
};

class NominalMaximum {
public:
  explicit NominalMaximum(func::FuncOp kernel) : kernel(kernel) {}

  bool append(PhysicalExprAttr expression) {
    if (!expression) return false;
    if (expression.getKind() == PhysicalExprKind::Maximum) {
      if (expression.getOperands().size() != 2) return false;
      if (!visitedMaxima.insert(expression).second) return true;
      for (Attribute operand : expression.getOperands())
        if (!append(cast<PhysicalExprAttr>(operand))) return false;
      return true;
    }
    NominalStage stage = normalize(expression);
    for (const NominalStage &previous : stages)
      if (previous.expression == stage.expression || dominates(previous, stage))
        return true;
    llvm::erase_if(stages, [&](const NominalStage &previous) {
      return dominates(stage, previous);
    });
    stages.push_back(std::move(stage));
    return true;
  }

  PhysicalExprAttr maximum() const {
    PhysicalExprAttr result;
    for (const NominalStage &stage : stages)
      result = result ? combineWords(result, stage.expression, PhysicalExprKind::Maximum)
                      : stage.expression;
    return result;
  }

private:
  NominalStage normalize(PhysicalExprAttr expression) {
    NominalStage stage{expression};
    SmallVector<PhysicalExprAttr> leaves;
    std::function<void(PhysicalExprAttr)> collect = [&](PhysicalExprAttr current) {
      if (current.getKind() == PhysicalExprKind::Add && current.getOperands().size() == 2) {
        for (Attribute operand : current.getOperands()) collect(cast<PhysicalExprAttr>(operand));
      } else {
        leaves.push_back(current);
      }
    };
    collect(expression);
    llvm::DenseMap<unsigned, unsigned> counts;
    for (PhysicalExprAttr leaf : leaves) {
      if (auto constant = constantPhysicalExpression(leaf)) {
        // Nominal payloads are nonnegative. Keep an unproved stage intact,
        // including its original checked arithmetic evaluation order.
        if (*constant < 0 || stage.constant > std::numeric_limits<int64_t>::max() - *constant)
          return NominalStage{expression};
        stage.constant += *constant;
        continue;
      }
      auto found = nonnegative.find(leaf);
      if (found == nonnegative.end()) {
        auto range = queryPhysicalExpressionRange(leaf, kernel);
        found = nonnegative.try_emplace(leaf, range && !range->smin().isNegative()).first;
      }
      if (!found->second) return NominalStage{expression};
      auto [ordinal, inserted] = ordinals.try_emplace(leaf, terms.size());
      if (inserted) terms.push_back(leaf);
      unsigned &count = counts[ordinal->second];
      if (count == std::numeric_limits<unsigned>::max()) return NominalStage{expression};
      ++count;
    }
    for (auto [term, count] : counts) stage.terms.emplace_back(term, count);
    llvm::sort(stage.terms, [](auto lhs, auto rhs) { return lhs.first < rhs.first; });
    stage.expression = wordConstant(kernel.getContext(), 0);
    for (auto [term, count] : stage.terms) {
      PhysicalExprAttr value = terms[term];
      if (count != 1)
        value = combineWords(wordConstant(kernel.getContext(), count), value,
                             PhysicalExprKind::Multiply);
      stage.expression = combineWords(stage.expression, value, PhysicalExprKind::Add);
    }
    stage.expression = combineWords(stage.expression,
        wordConstant(kernel.getContext(), stage.constant), PhysicalExprKind::Add);
    stage.nonnegative = true;
    return stage;
  }

  static bool dominates(const NominalStage &larger, const NominalStage &smaller) {
    if (!larger.nonnegative || !smaller.nonnegative ||
        larger.constant < smaller.constant) return false;
    unsigned next = 0;
    for (auto [term, count] : smaller.terms) {
      while (next < larger.terms.size() && larger.terms[next].first < term) ++next;
      if (next == larger.terms.size() || larger.terms[next].first != term ||
          larger.terms[next].second < count) return false;
    }
    return true;
  }

  func::FuncOp kernel;
  llvm::DenseSet<PhysicalExprAttr> visitedMaxima;
  llvm::DenseMap<PhysicalExprAttr, bool> nonnegative;
  llvm::DenseMap<PhysicalExprAttr, unsigned> ordinals;
  SmallVector<PhysicalExprAttr> terms;
  SmallVector<NominalStage> stages;
};

PhysicalExprAttr typePayloadWords(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    if (!fragment.getElementType().isIntOrIndexOrFloat()) return {};
    return fragmentRegisterFootprint(fragment);
  }
  if (auto record = dyn_cast<RecordType>(type)) {
    auto words = wordConstant(type.getContext(), 0);
    for (Attribute field : record.getFieldTypes()) {
      words = combineWords(words, typePayloadWords(cast<TypeAttr>(field).getValue()),
                           PhysicalExprKind::Add);
      if (!words) return {};
    }
    return words;
  }
  if (isa<ViewType, BufferType>(type)) return wordConstant(type.getContext(), 0);
  if (!type.isIntOrIndexOrFloat()) return {};
  unsigned bits = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return wordConstant(type.getContext(), std::max(1u, (bits + 31) / 32));
}

PhysicalExprAttr sumPayloads(MLIRContext *context,
                             const llvm::DenseSet<Value> &payloads,
                             ArrayRef<Value> orderedValues,
                             const llvm::DenseMap<Value, Type> &replacements,
                             ArrayRef<Operation *> ignored = {}) {
  auto words = wordConstant(context, 0);
  unsigned accounted = 0;
  for (Value value : orderedValues) {
    if (!payloads.contains(value)) continue;
    ++accounted;
    Operation *definition = value.getDefiningOp();
    if (definition && llvm::any_of(ignored, [&](Operation *operation) {
          return operation == definition || operation->isProperAncestor(definition);
        })) continue;
    auto replacement = replacements.find(value);
    Type type = replacement == replacements.end() ? value.getType() : replacement->second;
    words = combineWords(words, typePayloadWords(type), PhysicalExprKind::Add);
    if (!words) return {};
  }
  return accounted == payloads.size() ? words : PhysicalExprAttr();
}

PhysicalExprAttr reductionRegisterFootprint(
    Operation *reduction, const Liveness &liveness, ArrayRef<Value> orderedValues) {
  llvm::DenseSet<Value> payloads;
  if (!collectLivePayloads(reduction, liveness, payloads)) return {};
  PhysicalExprAttr registers;
  bool tunable = false;
  auto kernel = reduction->getParentOfType<func::FuncOp>();
  auto append = [&](FragmentType fragment) {
    AttrTypeWalker parameters;
    parameters.addWalk([&](PhysicalExprAttr expression) {
      if (expression.getKind() != PhysicalExprKind::Parameter) return;
      auto parameter = lookupParameter(kernel, expression.getParameterReference());
      if (!parameter) return;
      auto role = parameter.getRole();
      tunable |= role == ParameterRole::OwnershipM ||
                 role == ParameterRole::OwnershipN ||
                 role == ParameterRole::Reduction ||
                 role == ParameterRole::ReductionOuter ||
                 role == ParameterRole::ReductionInner;
    });
    parameters.walk(fragment.getShape());
    auto footprint = fragmentRegisterFootprint(fragment);
    registers = !registers ? footprint : PhysicalExprAttr::get(
        reduction->getContext(), PhysicalExprKind::Add, 0,
        StringAttr::get(reduction->getContext(), ""),
        ArrayAttr::get(reduction->getContext(), {registers, footprint}));
  };
  std::function<void(Type)> appendType = [&](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type)) append(fragment);
    else if (auto record = dyn_cast<RecordType>(type))
      for (Attribute field : record.getFieldTypes())
        appendType(cast<TypeAttr>(field).getValue());
  };
  // Liveness sets are unordered. Preserve lexical value order in the published
  // expression, including distinct values that happen to have the same type.
  for (Value value : orderedValues) {
    if (payloads.contains(value)) appendType(value.getType());
  }
  // This projects existing traversal candidates. It does not reject a fixed
  // provider program on the basis of a nominal register-allocation estimate.
  return tunable ? registers : PhysicalExprAttr();
}

} // namespace

PhysicalExprAttr nominalPayloadWords(TypeRange types) {
  if (types.empty()) return {};
  auto words = wordConstant(types.front().getContext(), 0);
  for (Type type : types) {
    words = combineWords(words, typePayloadWords(type), PhysicalExprKind::Add);
    if (!words) return {};
  }
  return words;
}

PhysicalExprAttr maximumNominalWorkingSet(func::FuncOp kernel,
                                         ArrayRef<PhysicalExprAttr> stages) {
  if (!kernel || stages.empty()) return {};
  NominalMaximum maximum(kernel);
  for (PhysicalExprAttr stage : stages)
    if (!maximum.append(stage)) return {};
  return maximum.maximum();
}

PhysicalExprAttr nominalLivePayloadWords(Operation *point, ValueRange additional) {
  auto kernel = point ? point->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!kernel || !point->getBlock()) return {};
  DominanceInfo dominance(kernel);
  if (llvm::any_of(additional, [&](Value value) {
        return !value || (value.getDefiningOp() != point && !dominance.dominates(value, point));
      })) return {};
  Liveness liveness(kernel);
  llvm::DenseSet<Value> payloads;
  if (!collectLivePayloads(point, liveness, payloads, /*completeTypes=*/true)) return {};
  for (Value value : additional) collectPayloads(value, payloads, /*completeTypes=*/true);
  llvm::DenseMap<Value, Type> replacements;
  return sumPayloads(point->getContext(), payloads, orderedKernelValues(kernel), replacements);
}

PhysicalExprAttr nominalLoopWorkingSet(scf::ForOp loop, TypeRange replacementCarries,
                                       ArrayRef<Operation *> ignored,
                                       TypeRange additionalPayloads,
                                       ValueRange retainedUntilYield) {
  auto kernel = loop ? loop->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!kernel || (!replacementCarries.empty() &&
      replacementCarries.size() != loop.getNumRegionIterArgs()) ||
      llvm::any_of(ignored, [&](Operation *operation) {
        return !operation || !loop->isProperAncestor(operation);
      })) return {};
  DominanceInfo dominance(kernel);
  if (llvm::any_of(retainedUntilYield, [&](Value value) {
        return !value || (value.getParentBlock() != loop.getBody() &&
                         !dominance.dominates(value, loop.getOperation()));
      })) return {};
  llvm::DenseMap<Value, Type> replacements;
  for (auto [argument, type] : llvm::zip(loop.getRegionIterArgs(), replacementCarries))
    replacements[argument] = type;
  auto additional = additionalPayloads.empty() ? wordConstant(loop.getContext(), 0)
                                               : nominalPayloadWords(additionalPayloads);
  if (!additional) return {};
  Liveness liveness(kernel);
  auto values = orderedKernelValues(kernel);
  SmallVector<PhysicalExprAttr> stages;
  WalkResult result = loop.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (operation == loop.getOperation()) return WalkResult::advance();
    llvm::DenseSet<Value> payloads;
    if (!collectLivePayloads(operation, liveness, payloads, /*completeTypes=*/true))
      return WalkResult::interrupt();
    if (!replacements.empty())
      for (Value argument : loop.getRegionIterArgs())
        collectPayloads(argument, payloads, /*completeTypes=*/true);
    for (Value value : retainedUntilYield)
      if (value.getDefiningOp() == operation || dominance.dominates(value, operation))
        collectPayloads(value, payloads, /*completeTypes=*/true);
    auto stage = sumPayloads(loop.getContext(), payloads, values, replacements, ignored);
    stage = combineWords(stage, additional, PhysicalExprKind::Add);
    if (!stage) return WalkResult::interrupt();
    stages.push_back(stage);
    // Helper/collective regions are atomic at this level. Only executable
    // structured control contributes nested physical-program stages.
    return llvm::is_contained(ignored, operation) ||
                   !isa<scf::ForOp, scf::IfOp, scf::WhileOp, ExecutionGroupOp>(operation)
               ? WalkResult::skip() : WalkResult::advance();
  });
  return result.wasInterrupted() ? PhysicalExprAttr() : maximumNominalWorkingSet(kernel, stages);
}

FootprintBound checkFragmentFootprint(
    FragmentType fragment, int64_t limit, int64_t wordsPerElement,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf) {
  if (limit < 0 || wordsPerElement <= 0)
    return FootprintBound::Invalid;
  bool unknown = false;
  // Saturating the product after each dimension preserves a proof that the
  // bound was exceeded without overflowing even for high-rank fragments.
  __int128 count = wordsPerElement;
  const __int128 saturation = static_cast<__int128>(limit) + 1;
  for (Attribute attribute : fragment.getShape()) {
    auto extent = evaluatePhysicalExpression(cast<PhysicalExprAttr>(attribute),
                                             resolveLeaf);
    if (!extent) {
      unknown = true;
      continue;
    }
    if (*extent <= 0)
      return FootprintBound::Invalid;
    count = std::min(saturation, count * *extent);
  }
  if (unknown)
    return FootprintBound::Unknown;
  return count > limit ? FootprintBound::Exceeds : FootprintBound::Within;
}

FragmentResourceAnalysis::FragmentResourceAnalysis(func::FuncOp kernel) {
  llvm::DenseSet<FragmentType> seenValues, seenPayloads;
  AttrTypeWalker valueTypes;
  valueTypes.addWalk([&](FragmentType fragment) {
    if (seenValues.insert(fragment).second)
      values.push_back(fragment);
  });
  kernel.walk([&](Operation *operation) {
    for (Type type : operation->getOperandTypes())
      valueTypes.walk(type);
    for (Type type : operation->getResultTypes())
      valueTypes.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          valueTypes.walk(argument.getType());
    if (isa<BroadcastOp, SplatOp, ReshapeOp>(operation))
      return;
    for (Type type : operation->getResultTypes()) {
      auto fragment = dyn_cast<FragmentType>(type);
      if (!fragment || !seenPayloads.insert(fragment).second)
        continue;
      llvm::DenseSet<StringAttr> symbols;
      AttrTypeWalker parameters;
      parameters.addWalk([&](PhysicalExprAttr expression) {
        if (expression.getKind() ==
            PhysicalExprKind::Parameter)
          symbols.insert(expression.getParameterReference().getName());
      });
      parameters.walk(fragment.getShape());
      for (StringAttr name : symbols)
        payloads[name].push_back(fragment);
    }
  });
  SmallVector<Operation *> reductions;
  kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  collectives = collectReductionRequirements(
      kernel, reductions, ReductionRequirementScope::AllCandidates);
}

ArrayRef<FragmentType>
FragmentResourceAnalysis::materializedTypesUsing(StringAttr name) const {
  auto found = payloads.find(name);
  return found == payloads.end() ? ArrayRef<FragmentType>()
                                : ArrayRef<FragmentType>(found->second);
}

namespace {

bool dependsOnInvocation(ConfigurationRequirementAttr requirement,
                         func::FuncOp kernel) {
  // Sampling the nondeferred domains only asks whether all leaves are known;
  // it does not establish a bound for the other candidates.
  auto known = evaluateConfigurationRequirement(requirement,
      [&](PhysicalExprAttr expression) -> std::optional<int64_t> {
        if (expression.getKind() != PhysicalExprKind::Parameter) return std::nullopt;
        auto parameter = lookupParameter(kernel, expression.getParameterReference());
        if (!parameter || parameter.isDeferred() ||
            parameter.getCategory() == ParameterCategory::Coverage)
          return std::nullopt;
        return parameter.getCandidates().asArrayRef().front();
      });
  return !known.usage;
}

} // namespace

bool isPointwiseTraversalParameter(ParameterAttr parameter) {
  return parameter.getBinding().getPointwiseChunk() ||
      (parameter.getCategory() == ParameterCategory::Pointwise &&
       (parameter.getRole() == ParameterRole::OwnershipM ||
        parameter.getRole() == ParameterRole::OwnershipN));
}

SmallVector<ConfigurationRequirementAttr> collectPointwiseRequirements(
    func::FuncOp kernel, const FragmentResourceAnalysis &resources) {
  SmallVector<ConfigurationRequirementAttr> requirements;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0) return requirements;
  Builder builder(kernel.getContext());
  auto limit = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Constant,
      capabilities.getRegistersPerUnit() - 1, builder.getStringAttr(""), builder.getArrayAttr({}));
  llvm::DenseSet<FragmentType> seen;
  for (Attribute attribute : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(attribute);
    if (!isPointwiseTraversalParameter(parameter)) continue;
    for (FragmentType fragment : resources.materializedTypesUsing(parameter.getName())) {
      if (!seen.insert(fragment).second) continue;
      auto requirement = ConfigurationRequirementAttr::get(kernel.getContext(),
          ConfigurationRequirementKind::NominalBudget,
          ConfigurationRequirementMetric::FragmentRegisterWords,
          ConfigurationRequirementPredicate::LessEqual,
          fragmentRegisterFootprint(fragment), limit, ParameterRefAttr(),
          builder.getStringAttr("pointwise payload exceeds the candidate register budget"));
      if (dependsOnInvocation(requirement, kernel) &&
          !llvm::is_contained(requirements, requirement))
        requirements.push_back(requirement);
    }
  }
  return requirements;
}

SmallVector<ConfigurationRequirementAttr> collectReductionRequirements(
    func::FuncOp kernel, ArrayRef<Operation *> reductions,
    ReductionRequirementScope scope) {
  SmallVector<ConfigurationRequirementAttr> requirements;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return requirements;
  if (reductions.empty()) return requirements;
  Builder builder(kernel.getContext());
  auto limit = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Constant,
      capabilities.getRegistersPerUnit(), builder.getStringAttr(""), builder.getArrayAttr({}));
  Liveness liveness(kernel);
  auto values = orderedKernelValues(kernel);
  for (Operation *reduction : reductions) {
    auto footprint = reductionRegisterFootprint(reduction, liveness, values);
    if (!footprint) continue;
    auto requirement = ConfigurationRequirementAttr::get(kernel.getContext(),
        ConfigurationRequirementKind::NominalBudget,
        ConfigurationRequirementMetric::FragmentRegisterWords,
        ConfigurationRequirementPredicate::LessEqual, footprint, limit,
        ParameterRefAttr(),
        builder.getStringAttr("collective live payloads exceed the candidate register budget"));
    if (scope == ReductionRequirementScope::InvocationDependent &&
        !dependsOnInvocation(requirement, kernel)) continue;
    if (!llvm::is_contained(requirements, requirement)) requirements.push_back(requirement);
  }
  return requirements;
}

LogicalResult verifyConfigurationRequirements(
    func::FuncOp kernel, ArrayRef<ConfigurationRequirementAttr> expected) {
  auto space = ParameterSpace::read(kernel);
  if (failed(space) || failed(space->verifyRequirements(expected)))
    return failure();
  auto rows = space->configurations(ConfigurationStage::Complete);
  auto requirements = space->requirements();
  if (failed(rows) || failed(requirements))
    return failure();
  llvm::DenseSet<Attribute> remaining;
  for (ConfigurationRequirementAttr requirement : expected)
    if (!remaining.insert(requirement).second)
      return kernel.emitError(
          "current program produced a duplicate candidate requirement");
  for (ConfigurationRequirementAttr requirement : *requirements)
    if (!remaining.erase(requirement))
      return kernel.emitError(
                 "candidate requirement is not justified by the current program: ")
             << requirement;
  if (!remaining.empty())
    return kernel.emitError(
               "candidate requirements omit a current program condition: ")
           << *remaining.begin();
  for (DictionaryAttr row : *rows)
    for (ConfigurationRequirementAttr requirement : *requirements) {
      auto evaluation = evaluateConfigurationRequirement(requirement, row, kernel);
      if (evaluation.status != RequirementStatus::Violated &&
          evaluation.status != RequirementStatus::Invalid)
        continue;
      auto diagnostic = kernel.emitError(
          "candidate violates a current program requirement: ");
      diagnostic << requirement.getMessage().getValue()
                 << "; predicate="
                 << stringifyConfigurationRequirementPredicate(
                        requirement.getPredicate())
                 << "; bindings=" << row;
      if (evaluation.usage)
        diagnostic << "; usage=" << *evaluation.usage;
      if (evaluation.limit)
        diagnostic << "; limit=" << *evaluation.limit;
      return failure();
    }
  return success();
}

} // namespace intent::gpu
