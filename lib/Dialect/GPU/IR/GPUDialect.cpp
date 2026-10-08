#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent::gpu;

#include "Intent/Dialect/GPU/IR/GPUAttrsEnums.cpp.inc"

#define GET_DIALECT_DEF
#include "Intent/Dialect/GPU/IR/GPUDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/GPU/IR/GPUAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Intent/Dialect/GPU/IR/GPUTypes.cpp.inc"

namespace intent::gpu {

ParameterRefAttr PhysicalExprAttr::getParameterReference() const {
  return mlir::dyn_cast<ParameterRefAttr>(getSymbol());
}

ArgumentRefAttr PhysicalExprAttr::getArgumentReference() const {
  return mlir::dyn_cast<ArgumentRefAttr>(getSymbol());
}

ParameterRefAttr ParameterAttr::getReference() const {
  return ParameterRefAttr::get(getContext(), getName());
}
bool ParameterAttr::isExtent() const { return getValueType().isIndex(); }
bool ParameterAttr::isDeferred() const {
  return getPhase() == ConfigurationBindingPhase::Deferred;
}
ParameterAttr ParameterAttr::withName(StringAttr value) const {
  return get(getContext(), value, getValueType(), getRole(), getCategory(),
             getElementBitWidth(), getCandidates(), getPhase(), getBinding());
}
ParameterAttr ParameterAttr::withCandidates(DenseI64ArrayAttr value) const {
  return get(getContext(), getName(), getValueType(), getRole(), getCategory(),
             getElementBitWidth(), value, getPhase(), getBinding());
}
ParameterAttr ParameterAttr::withPhase(ConfigurationBindingPhase value) const {
  return get(getContext(), getName(), getValueType(), getRole(), getCategory(),
             getElementBitWidth(), getCandidates(), value, getBinding());
}
ParameterAttr ParameterAttr::withBinding(ParameterBindingAttr value) const {
  return get(getContext(), getName(), getValueType(), getRole(), getCategory(),
             getElementBitWidth(), getCandidates(), getPhase(), value);
}

ParameterBindingAttr ParameterBindingAttr::withDimension(IntegerAttr value) const {
  return get(getContext(), value, getSource(), getCoverageBound(), getGroup(),
             getPointwiseChunk(), getPointwiseLocal());
}
ParameterBindingAttr ParameterBindingAttr::withSource(PhysicalSourceAttr value) const {
  return get(getContext(), getDimension(), value, getCoverageBound(), getGroup(),
             getPointwiseChunk(), getPointwiseLocal());
}
ParameterBindingAttr ParameterBindingAttr::withCoverageBound(PhysicalExprAttr value) const {
  return get(getContext(), getDimension(), getSource(), value, getGroup(),
             getPointwiseChunk(), getPointwiseLocal());
}
ParameterBindingAttr ParameterBindingAttr::withGroup(ArrayAttr value) const {
  return get(getContext(), getDimension(), getSource(), getCoverageBound(), value,
             getPointwiseChunk(), getPointwiseLocal());
}
ParameterBindingAttr ParameterBindingAttr::withPointwiseChunk(bool value) const {
  return get(getContext(), getDimension(), getSource(), getCoverageBound(), getGroup(),
             value, getPointwiseLocal());
}
ParameterBindingAttr ParameterBindingAttr::withPointwiseLocal(bool value) const {
  return get(getContext(), getDimension(), getSource(), getCoverageBound(), getGroup(),
             getPointwiseChunk(), value);
}

ParameterAttr lookupParameterDeclaration(Operation *anchor,
                                         ParameterRefAttr reference) {
  if (!anchor || !reference) return {};
  auto kernel = mlir::dyn_cast<func::FuncOp>(anchor);
  if (!kernel) kernel = anchor->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->getAttrOfType<UnitAttr>(kernelAttr)) return {};
  auto declarations = kernel->getAttrOfType<ArrayAttr>(parametersAttr);
  if (!declarations) return {};
  ParameterAttr found;
  for (Attribute attribute : declarations) {
    auto declaration = mlir::dyn_cast<ParameterAttr>(attribute);
    if (!declaration || declaration.getName() != reference.getName()) continue;
    if (found) return {};
    found = declaration;
  }
  return found;
}

LogicalResult verifyParameterDeclarations(Operation *kernel) {
  if (!mlir::isa<func::FuncOp>(kernel) || !kernel->getAttrOfType<UnitAttr>(kernelAttr))
    return kernel->emitOpError("physical GPU kernel ownership requires func.func with a unit kernel marker");
  auto declarations = kernel->getAttrOfType<ArrayAttr>(parametersAttr);
  if (!declarations)
    return kernel->emitOpError("requires a kernel-owned parameter declaration table");
  llvm::DenseMap<StringAttr, ParameterAttr> names;
  for (Attribute attribute : declarations) {
    auto declaration = mlir::dyn_cast<ParameterAttr>(attribute);
    if (!declaration || !names.try_emplace(declaration.getName(), declaration).second)
      return kernel->emitOpError("parameter declarations must be typed and have unique names");
    auto diagnostic = [&] {
      auto result = kernel->emitOpError("invalid parameter declaration ");
      result << declaration.getName() << ": ";
      return result;
    };
    auto binding = declaration.getBinding();
    if (!binding || failed(ParameterBindingAttr::verify(
            diagnostic, binding.getDimension(), binding.getSource(), binding.getCoverageBound(),
            binding.getGroup(), binding.getPointwiseChunk(), binding.getPointwiseLocal())) ||
        failed(ParameterAttr::verify(diagnostic, declaration.getName(), declaration.getValueType(),
            declaration.getRole(), declaration.getCategory(), declaration.getElementBitWidth(),
            declaration.getCandidates(), declaration.getPhase(), binding)))
      return failure();
  }
  bool valid = true;
  AttrTypeWalker walker;
  walker.addWalk([&](ParameterRefAttr reference) {
    if (valid && !names.contains(reference.getName())) {
      kernel->emitOpError("references an undeclared compile-time parameter ") << reference;
      valid = false;
    }
  });
  walker.addWalk([&](PhysicalExprAttr expression) {
    if (!valid || expression.getKind() != PhysicalExprKind::Parameter) return;
    auto reference = expression.getParameterReference();
    auto found = reference ? names.find(reference.getName()) : names.end();
    if (found == names.end() || !found->second.isExtent()) {
      kernel->emitOpError("physical extent expression requires a declared positive index parameter");
      valid = false;
    }
  });
  kernel->walk([&](Operation *operation) {
    walker.walk(operation->getAttrDictionary());
    for (Type type : operation->getResultTypes()) walker.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) walker.walk(argument.getType());
  });
  return success(valid);
}

LogicalResult IntentGPUDialect::verifyOperationAttribute(Operation *operation,
                                                        NamedAttribute attribute) {
  if (attribute.getName() == kernelAttr) {
    if (failed(verifyParameterDeclarations(operation))) return failure();
    return verifyProgramInterface(cast<func::FuncOp>(operation));
  }
  if (attribute.getName() == parametersAttr &&
      (!mlir::isa<func::FuncOp>(operation) || !operation->getAttrOfType<UnitAttr>(kernelAttr)))
    return operation->emitOpError("parameter declarations require a physical GPU kernel owner");
  return success();
}

bool isCompileTimePhysicalExpr(PhysicalExprAttr expression) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Constant || kind == PhysicalExprKind::Parameter)
    return true;
  if (kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimePhysicalExpr(cast<PhysicalExprAttr>(operand));
  });
}

std::optional<SmallVector<int64_t>>
queryAxisPermutation(FragmentType source, FragmentType target) {
  if (source.getOwner() != target.getOwner() ||
      source.getValidity() != target.getValidity() ||
      source.getShape().size() != target.getShape().size())
    return std::nullopt;
  SmallVector<int64_t> permutation;
  for (Attribute attribute : target.getAxisMaps()) {
    auto targetAxis = cast<AxisMapAttr>(attribute);
    std::optional<int64_t> matched;
    for (auto [axis, mapping] : llvm::enumerate(source.getAxisMaps())) {
      auto sourceAxis = cast<AxisMapAttr>(mapping);
      if (sourceAxis.getSourceId() != targetAxis.getSourceId() ||
          sourceAxis.getSourceAxis() != targetAxis.getSourceAxis() ||
          sourceAxis.getDerived() != targetAxis.getDerived() ||
          sourceAxis.getDimensionId() != targetAxis.getDimensionId())
        continue;
      if (matched)
        return std::nullopt;
      matched = axis;
    }
    if (!matched || llvm::is_contained(permutation, *matched))
      return std::nullopt;
    permutation.push_back(*matched);
  }
  return permutation;
}

BroadcastProjection queryAxisProjection(FragmentType source,
                                        FragmentType target) {
  BroadcastProjection result;
  if (source.getOwner() != target.getOwner() ||
      source.getShape().size() > target.getShape().size())
    return result;
  result.targetToSource.resize(target.getShape().size());
  SmallVector<bool> sourceUsed(source.getShape().size(), false);
  const unsigned offset = target.getShape().size() - source.getShape().size();

  auto bindUnique = [&](unsigned sourceIndex,
                        function_ref<bool(AxisMapAttr)> selects) {
    std::optional<unsigned> selected;
    for (auto [targetIndex, mapping] : llvm::enumerate(target.getAxisMaps())) {
      if (result.targetToSource[targetIndex] ||
          !selects(cast<AxisMapAttr>(mapping)))
        continue;
      if (!selected || targetIndex == offset + sourceIndex)
        selected = targetIndex;
    }
    if (!selected)
      return true;
    unsigned matches = llvm::count_if(
        llvm::enumerate(target.getAxisMaps()), [&](auto item) {
          return !result.targetToSource[item.index()] &&
                 selects(cast<AxisMapAttr>(item.value()));
        });
    if (matches > 1 && *selected != offset + sourceIndex) {
      result.state = BroadcastProjectionState::Ambiguous;
      return false;
    }
    result.targetToSource[*selected] = sourceIndex;
    sourceUsed[sourceIndex] = true;
    return true;
  };

  // Rank-expanding physical broadcasts also embed coordinate vectors. Respect
  // a unique source-axis occurrence already present in the result relation;
  // a row coordinate must not become a column merely because both extents are N.
  if (source.getShape().size() < target.getShape().size())
    for (auto [sourceIndex, attribute] : llvm::enumerate(source.getAxisMaps())) {
      auto axis = cast<AxisMapAttr>(attribute);
      auto sameSource = [&](Attribute candidate) {
        auto other = cast<AxisMapAttr>(candidate);
        return axis.getSourceId() == other.getSourceId() &&
               axis.getSourceAxis() == other.getSourceAxis() &&
               axis.getDerived() == other.getDerived();
      };
      auto sameOccurrence = [&](Attribute candidate) {
        return sameSource(candidate) && axis.getDimensionId() > 0 &&
               axis.getDimensionId() ==
                   cast<AxisMapAttr>(candidate).getDimensionId();
      };
      if (llvm::count_if(source.getAxisMaps(), sameOccurrence) == 1 &&
          llvm::count_if(target.getAxisMaps(), sameOccurrence) == 1) {
        if (!bindUnique(sourceIndex, [&](AxisMapAttr other) {
              return sameOccurrence(other);
            }))
          return result;
        continue;
      }
      if (llvm::count_if(source.getAxisMaps(), sameSource) == 1 &&
          llvm::count_if(target.getAxisMaps(), sameSource) == 1)
        if (!bindUnique(sourceIndex, [&](AxisMapAttr other) { return sameSource(other); }))
          return result;
    }

  // Same-rank broadcasts preserve the explicit positional relation. Use
  // that explicit occurrence relation before source-identity matching: one
  // logical source axis may legitimately occur more than once in a Cartesian
  // result, and identity-first matching would let the wrong occurrence consume
  // the positional target.  A positional pair must still carry either the same
  // immutable source or the same logical dimension: a physically singleton
  // ownership axis is not an anonymous broadcast axis.
  for (auto [sourceIndex, mapping] : llvm::enumerate(source.getAxisMaps())) {
    unsigned targetIndex = offset + sourceIndex;
    if (sourceUsed[sourceIndex] || result.targetToSource[targetIndex])
      continue;
    auto sourceAxis = cast<AxisMapAttr>(mapping);
    auto targetAxis = cast<AxisMapAttr>(target.getAxisMaps()[targetIndex]);
    bool sameSource =
        sourceAxis.getSourceId() == targetAxis.getSourceId() &&
        sourceAxis.getSourceAxis() == targetAxis.getSourceAxis() &&
        sourceAxis.getDerived() == targetAxis.getDerived();
    if (!sameSource &&
        sourceAxis.getDimensionId() != targetAxis.getDimensionId())
      continue;
    result.targetToSource[targetIndex] = sourceIndex;
    sourceUsed[sourceIndex] = true;
  }

  for (auto [sourceIndex, mapping] : llvm::enumerate(source.getAxisMaps())) {
    if (sourceUsed[sourceIndex])
      continue;
    auto sourceAxis = cast<AxisMapAttr>(mapping);
    if (!bindUnique(sourceIndex, [&](AxisMapAttr targetAxis) {
          return sourceAxis.getSourceId() == targetAxis.getSourceId() &&
                 sourceAxis.getSourceAxis() == targetAxis.getSourceAxis() &&
                 sourceAxis.getDerived() == targetAxis.getDerived();
        }))
      return result;
  }
  for (auto [sourceIndex, mapping] : llvm::enumerate(source.getAxisMaps())) {
    if (sourceUsed[sourceIndex])
      continue;
    auto sourceAxis = cast<AxisMapAttr>(mapping);
    auto sourceExtent = cast<PhysicalExprAttr>(source.getShape()[sourceIndex]);
    bool singleton =
        sourceExtent.getKind() ==
            PhysicalExprKind::Constant &&
        sourceExtent.getValue() == 1;
    if (singleton)
      continue;
    unsigned sourceOccurrences = llvm::count_if(
        llvm::enumerate(source.getAxisMaps()), [&](auto item) {
          if (sourceUsed[item.index()])
            return false;
          return cast<AxisMapAttr>(item.value()).getDimensionId() ==
                 sourceAxis.getDimensionId();
        });
    unsigned targetOccurrences = llvm::count_if(
        llvm::enumerate(target.getAxisMaps()), [&](auto item) {
          if (result.targetToSource[item.index()])
            return false;
          return cast<AxisMapAttr>(item.value()).getDimensionId() ==
                 sourceAxis.getDimensionId();
        });
    if (targetOccurrences == 0)
      continue;
    if (sourceOccurrences > 1 || targetOccurrences > 1) {
      result.state = BroadcastProjectionState::Ambiguous;
      return result;
    }
    if (!bindUnique(sourceIndex, [&](AxisMapAttr targetAxis) {
          return sourceAxis.getDimensionId() == targetAxis.getDimensionId();
        }))
      return result;
  }
  // Any remaining axes use the broadcast operation's explicit trailing-axis
  // relation.  Typed positional pairs were consumed first so repeated source
  // occurrences cannot steal one another's target; this final step also keeps
  // physical rematerializations whose producer and consumer carry different
  // derived source identities.
  for (unsigned sourceIndex = 0; sourceIndex < source.getShape().size();
       ++sourceIndex) {
    if (sourceUsed[sourceIndex])
      continue;
    unsigned targetIndex = offset + sourceIndex;
    if (result.targetToSource[targetIndex])
      return result;
    result.targetToSource[targetIndex] = sourceIndex;
    sourceUsed[sourceIndex] = true;
  }

  unsigned previous = 0;
  bool sawSource = false;
  for (std::optional<unsigned> sourceIndex : result.targetToSource) {
    if (!sourceIndex)
      continue;
    if (sawSource && *sourceIndex <= previous)
      return result;
    previous = *sourceIndex;
    sawSource = true;
  }
  result.state = BroadcastProjectionState::Exact;
  return result;
}

BroadcastProjection queryBroadcastProjection(FragmentType source,
                                              FragmentType target) {
  BroadcastProjection result = queryAxisProjection(source, target);
  if (!result.isExact())
    return result;
  for (auto [targetAxis, sourceAxis] :
       llvm::enumerate(result.targetToSource)) {
    if (!sourceAxis)
      continue;
    auto sourceExtent = cast<PhysicalExprAttr>(source.getShape()[*sourceAxis]);
    bool singleton =
        sourceExtent.getKind() ==
            PhysicalExprKind::Constant &&
        sourceExtent.getValue() == 1;
    if (!singleton && source.getShape()[*sourceAxis] != target.getShape()[targetAxis]) {
      result.state = BroadcastProjectionState::Unknown;
      return result;
    }
  }
  return result;
}

void IntentGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Dialect/GPU/IR/GPUAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Intent/Dialect/GPU/IR/GPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/GPU/IR/GPUOps.cpp.inc"
      >();
}

LogicalResult ParameterRefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name) {
  return name && !name.empty() ? success()
                             : emitError() << "parameter reference requires a nonempty name";
}

LogicalResult ParameterBindingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, IntegerAttr dimension,
    PhysicalSourceAttr source, PhysicalExprAttr, ArrayAttr, bool, bool) {
  if (dimension && (!dimension.getType().isSignlessInteger(64) || dimension.getInt() <= 0))
    return emitError() << "parameter dimension must be a positive i64 identity";
  if (source && source.getSourceId() == 0)
    return emitError() << "parameter source requires a nonzero source identity";
  return success();
}

LogicalResult ArgumentRefAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                      uint64_t id) {
  return id ? success() : emitError() << "physical argument identity must be nonzero";
}

LogicalResult ArgumentBindingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArgumentRefAttr reference,
    ArgumentKind kind, IntegerAttr publicOrdinal, ArgumentRefAttr source,
    IntegerAttr axis, IntegerAttr dimension) {
  if (!reference || reference.getId() == 0 || kind > ArgumentKind::Workspace)
    return emitError() << "physical argument binding requires a valid identity and kind";
  auto nonnegative = [](IntegerAttr value) {
    return value && value.getType().isSignlessInteger(64) && value.getInt() >= 0;
  };
  if (kind == ArgumentKind::Public)
    return nonnegative(publicOrdinal) && !source && !axis && !dimension
      ? success() : emitError() << "public binding requires only its public ordinal";
  if (kind == ArgumentKind::Workspace)
    return !publicOrdinal && !source && !axis && !dimension
      ? success() : emitError() << "workspace binding cannot carry public or metadata fields";
  if (publicOrdinal || !source || !nonnegative(axis))
    return emitError() << "metadata binding requires only a source argument and source axis";
  if (kind == ArgumentKind::Dimension)
    return nonnegative(dimension) && dimension.getInt() > 0
      ? success() : emitError() << "dimension binding requires a positive logical identity";
  return !dimension ? success() : emitError() << "stride binding cannot carry a dimension identity";
}

LogicalResult PhysicalExprAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, PhysicalExprKind kind,
    int64_t value, Attribute symbol, ArrayAttr operands) {
  if (kind > PhysicalExprKind::NextPowerOfTwo || !symbol || !operands)
    return emitError() << "physical expression has an invalid closed schema";
  for (Attribute operand : operands)
    if (!mlir::isa<PhysicalExprAttr>(operand))
      return emitError() << "physical expression operands must be #intent_gpu.expr";
  auto name = mlir::dyn_cast<StringAttr>(symbol);
  if (kind == PhysicalExprKind::Constant)
    return name && name.empty() && operands.empty()
               ? success()
               : emitError() << "constant physical expression cannot carry a symbol or operands";
  if (kind == PhysicalExprKind::Parameter)
    return mlir::isa<ParameterRefAttr>(symbol) && value == 0 && operands.empty()
               ? success()
               : emitError() << "parameter expression requires a typed declaration reference";
  if (kind == PhysicalExprKind::Dimension || kind == PhysicalExprKind::ScalarABI)
    return mlir::isa<ArgumentRefAttr>(symbol) && operands.empty() &&
                   (kind == PhysicalExprKind::Dimension ? value > 0 : value == 0)
               ? success()
               : emitError() << "runtime physical expression requires a typed argument reference";
  if (!name || !name.empty() || value != 0)
    return emitError() << "composed physical expression cannot carry a symbol or literal payload";
  unsigned expected = kind == PhysicalExprKind::Select ? 3
                    : kind == PhysicalExprKind::NextPowerOfTwo ? 1 : 2;
  if (operands.size() != expected)
    return emitError() << "composed physical expression has the wrong arity";
  if (kind == PhysicalExprKind::CeilDiv ||
      kind == PhysicalExprKind::FloorDiv) {
    auto divisor = mlir::cast<PhysicalExprAttr>(operands[1]);
    if (divisor.getKind() ==
            PhysicalExprKind::Constant &&
        divisor.getValue() == 0)
      return emitError() << "physical division requires a nonzero divisor";
  }
  return success();
}

LogicalResult TuningProfileTableAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr columns,
    DictionaryAttr families) {
  if (!columns || columns.empty() || !families || families.empty())
    return emitError() << "tuning profile table requires columns and families";
  llvm::DenseSet<Attribute> columnNames;
  for (Attribute attribute : columns) {
    auto name = mlir::dyn_cast<StringAttr>(attribute);
    if (!name || name.empty() || !columnNames.insert(name).second)
      return emitError() << "tuning profile columns must be distinct nonempty names";
  }
  for (NamedAttribute family : families) {
    auto rows = mlir::dyn_cast<ArrayAttr>(family.getValue());
    if (family.getName().empty() || !rows || rows.empty())
      return emitError() << "tuning family requires a name and nonempty rows";
    llvm::DenseSet<Attribute> unique;
    for (Attribute attribute : rows) {
      auto row = mlir::dyn_cast<DenseI64ArrayAttr>(attribute);
      if (!row || row.size() != columns.size() ||
          llvm::any_of(row.asArrayRef(), [](int64_t value) { return value < 0; }))
        return emitError() << "tuning family '" << family.getName()
                           << "' requires " << columns.size()
                           << " nonnegative integers per row";
      if (!unique.insert(row).second)
        return emitError() << "tuning family contains duplicate rows";
    }
  }
  return success();
}

LogicalResult TuningProfilesAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DictionaryAttr spaces) {
  if (!spaces || spaces.empty())
    return emitError() << "tuning profiles require resolved namespace tables";
  for (NamedAttribute space : spaces)
    if (space.getName().empty() ||
        !mlir::isa<TuningProfileTableAttr>(space.getValue()))
      return emitError() << "tuning namespace requires a name and typed profile table";
  return success();
}

LogicalResult ConfigurationRequirementAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ConfigurationRequirementKind kind, ConfigurationRequirementMetric metric,
    ConfigurationRequirementPredicate predicate, PhysicalExprAttr usage,
    PhysicalExprAttr limit, ParameterRefAttr activation, StringAttr message) {
  if ((kind != ConfigurationRequirementKind::Legality &&
       kind != ConfigurationRequirementKind::NominalBudget) ||
      (metric != ConfigurationRequirementMetric::FragmentElements &&
       metric != ConfigurationRequirementMetric::FragmentRegisterWords &&
       metric != ConfigurationRequirementMetric::FragmentBytes &&
       metric != ConfigurationRequirementMetric::ResidentWorkers) ||
      !symbolizeConfigurationRequirementPredicate(
          static_cast<uint32_t>(predicate)) ||
      !usage || !message || message.empty())
    return emitError()
           << "configuration requirement needs a kind, metric, predicate, "
              "quantity and diagnostic message";
  const bool binary = predicate == ConfigurationRequirementPredicate::LessEqual ||
                      predicate == ConfigurationRequirementPredicate::MultipleOf ||
                      predicate == ConfigurationRequirementPredicate::Equal;
  if (binary != static_cast<bool>(limit))
    return emitError()
           << "only less-equal, multiple-of and equal requirements take a limit";
  if (kind == ConfigurationRequirementKind::NominalBudget &&
      predicate != ConfigurationRequirementPredicate::LessEqual)
    return emitError() << "nominal budgets require a less-equal quantity test";
  if (activation &&
      failed(ParameterRefAttr::verify(emitError, activation.getName())))
    return failure();
  bool valid = true;
  AttrTypeWalker walker;
  walker.addWalk([&](PhysicalExprAttr expression) {
    if (!valid) return;
    if (failed(PhysicalExprAttr::verify(emitError, expression.getKind(),
          expression.getValue(), expression.getSymbol(), expression.getOperands()))) {
      valid = false;
    } else if (expression.getKind() == PhysicalExprKind::ScalarABI) {
      emitError() << "configuration requirement cannot depend on a runtime scalar argument";
      valid = false;
    }
  });
  walker.walk(usage);
  if (limit) walker.walk(limit);
  return success(valid);
}

LogicalResult ConfigurationSetAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ConfigurationStage stage,
    ArrayAttr rows, ArrayAttr requirements) {
  if ((stage != ConfigurationStage::Shared &&
       stage != ConfigurationStage::Complete) || !rows || !requirements)
    return emitError() << "configuration set requires a binding stage, rows and explicit requirements";
  llvm::DenseSet<Attribute> unique;
  for (Attribute attribute : rows) {
    auto row = mlir::dyn_cast<DictionaryAttr>(attribute);
    if (!row || !unique.insert(row).second)
      return emitError() << "configuration rows must be distinct binding dictionaries";
    for (NamedAttribute binding : row) {
      auto value = mlir::dyn_cast<IntegerAttr>(binding.getValue());
      if (binding.getName().empty() || !value ||
          !value.getType().isSignlessInteger(64))
        return emitError() << "configuration bindings require named i64 values";
    }
  }
  unique.clear();
  for (Attribute attribute : requirements) {
    auto requirement = mlir::dyn_cast<ConfigurationRequirementAttr>(attribute);
    if (!requirement || !unique.insert(attribute).second)
      return emitError() << "configuration requirements must be distinct typed constraints";
    if (failed(ConfigurationRequirementAttr::verify(
            emitError, requirement.getKind(), requirement.getMetric(),
            requirement.getPredicate(), requirement.getUsage(),
            requirement.getLimit(), requirement.getActivation(),
            requirement.getMessage())))
      return failure();
  }
  return success();
}

LogicalResult ParameterAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name, Type valueType,
    ParameterRole role, ParameterCategory category, uint32_t elementBitWidth,
    DenseI64ArrayAttr candidates, ConfigurationBindingPhase phase,
    ParameterBindingAttr binding) {
  if (!name || name.empty() ||
      !symbolizeParameterRole(static_cast<uint32_t>(role)) ||
      category >
          ParameterCategory::Histogram ||
      !candidates || candidates.empty() || !valueType ||
      (!valueType.isIndex() && !valueType.isSignlessInteger(1)) || !binding)
    return emitError()
           << "physical parameter requires a name, role, category and candidates";
  auto typedCategory = category;
  auto typedRole = role;
  const bool providerRole =
      typedRole == ParameterRole::ProviderWarps ||
      typedRole == ParameterRole::ProviderStages ||
      typedRole == ParameterRole::ProviderCTAs ||
      typedRole == ParameterRole::ProviderAccessForm ||
      typedRole == ParameterRole::ProviderOccupancy ||
      typedRole == ParameterRole::ProviderLoadPolicy;
  if (providerRole != (typedCategory == ParameterCategory::Provider))
    return emitError()
           << "provider parameter roles require exactly the provider category";
  if ((phase != ConfigurationBindingPhase::Shared &&
       phase != ConfigurationBindingPhase::Provider &&
       phase != ConfigurationBindingPhase::Deferred) ||
      providerRole != (phase == ConfigurationBindingPhase::Provider) ||
      (typedCategory == ParameterCategory::Coverage && phase != ConfigurationBindingPhase::Deferred))
    return emitError() << "parameter binding phase disagrees with its category";
  if (binding.getCoverageBound() && phase != ConfigurationBindingPhase::Deferred)
    return emitError() << "coverage bounds require a deferred parameter declaration";
  if (valueType.isSignlessInteger(1) && phase != ConfigurationBindingPhase::Provider)
    return emitError() << "boolean configuration choices must be provider declarations";
  if ((typedRole == ParameterRole::ReductionOuter ||
       typedRole == ParameterRole::ReductionInner) &&
      typedCategory != ParameterCategory::Reduction)
    return emitError()
           << "multi-axis reduction parameter roles require the reduction category";
  const bool carriesDataGranularity =
      typedCategory == ParameterCategory::Pointwise ||
      typedCategory == ParameterCategory::Reduction ||
      typedCategory == ParameterCategory::Scan ||
      typedCategory == ParameterCategory::Contraction ||
      typedCategory == ParameterCategory::RegionReduction ||
      typedCategory == ParameterCategory::RegionContraction ||
      typedCategory == ParameterCategory::PersistentContraction ||
      typedCategory == ParameterCategory::Histogram;
  if (carriesDataGranularity != (elementBitWidth > 0))
    return emitError()
           << "data-granularity parameter categories require an element bit width and non-data categories forbid one";
  llvm::DenseSet<int64_t> unique;
  for (int64_t candidate : candidates.asArrayRef())
    if ((valueType.isIndex() ? candidate <= 0 : candidate < 0 || candidate > 1) ||
        !unique.insert(candidate).second)
      return emitError() << "parameter candidates must be unique and in the declared value-type domain";
  return success();
}

LogicalResult AxisMapAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint64_t sourceId,
    uint32_t sourceAxis, int64_t dimensionId, uint32_t fragmentAxis,
    bool derived) {
  (void)sourceAxis;
  (void)fragmentAxis;
  (void)derived;
  return sourceId != 0 && dimensionId > 0
             ? success()
             : emitError()
                   << "axis mapping requires source and logical-dimension identities";
}

LogicalResult ReshapeGroupAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr sourceAxes, DenseI64ArrayAttr resultAxes) {
  if (!sourceAxes || !resultAxes ||
      (sourceAxes.empty() && resultAxes.empty()))
    return emitError()
           << "reshape group must cover at least one source or result axis";
  auto consecutive = [](ArrayRef<int64_t> axes) {
    if (axes.size() < 2)
      return true;
    return llvm::all_of(llvm::seq<size_t>(1, axes.size()), [&](size_t index) {
      return axes[index] == axes[index - 1] + 1;
    });
  };
  if ((!sourceAxes.empty() && sourceAxes[0] < 0) ||
      (!resultAxes.empty() && resultAxes[0] < 0) ||
      !consecutive(sourceAxes.asArrayRef()) ||
      !consecutive(resultAxes.asArrayRef()))
    return emitError()
           << "reshape group axes must be non-negative and consecutive";
  return success();
}

LogicalResult ViewLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayAttr extents, DenseI64ArrayAttr dimensionIds,
    ArrayAttr strides) {
  if (!extents || !dimensionIds || !strides ||
      static_cast<int64_t>(extents.size()) != dimensionIds.size() ||
      strides.size() != extents.size())
    return emitError() << "view layout requires dimensions and strides";
  for (Attribute extent : extents)
    if (!mlir::isa<PhysicalExprAttr>(extent))
      return emitError() << "view extents must be typed physical expressions";
  for (int64_t dimension : dimensionIds.asArrayRef())
    if (dimension < 0)
      return emitError() << "view dimension identity cannot be negative";
  for (Attribute stride : strides)
    if (!mlir::isa<PhysicalExprAttr>(stride))
      return emitError() << "view stride must be a typed physical expression";
  return success();
}

LogicalResult CapabilitiesAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int64_t computeUnits,
    int64_t sharedMemoryPerUnit, int64_t maxDynamicSharedMemoryPerBlock,
    int64_t registersPerUnit, int64_t maxThreadsPerBlock,
    int64_t computeCapabilityMajor, int64_t computeCapabilityMinor,
    int64_t singleToDoublePrecisionPerfRatio,
    bool matrixUnits, bool dynamicVectorWidth, bool nativeTupleReductions,
    bool nativeTupleReductionRequiresConstantIdentity,
    bool nativeFragmentGather) {
  (void)matrixUnits;
  (void)dynamicVectorWidth;
  (void)nativeTupleReductions;
  (void)nativeTupleReductionRequiresConstantIdentity;
  (void)nativeFragmentGather;
  if (computeUnits <= 0 || sharedMemoryPerUnit <= 0 ||
      maxDynamicSharedMemoryPerBlock <= 0 || registersPerUnit <= 0 ||
      maxThreadsPerBlock <= 0 || computeCapabilityMajor <= 0 ||
      computeCapabilityMinor < 0 || singleToDoublePrecisionPerfRatio <= 0)
    return emitError() << "selected GPU capabilities require positive resource limits";
  return success();
}

LogicalResult ViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType,
                               uint32_t access,
                               uint64_t sourceId,
                               ViewLayoutAttr layout) {
  if (!elementType || access > 2 || sourceId == 0 || !layout)
    return emitError() << "physical view schema is incomplete";
  return success();
}

LogicalResult FragmentType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type elementType,
    ArrayAttr shape, ArrayAttr axisMaps, uint32_t validity, uint64_t owner) {
  if (!elementType || !shape || !axisMaps || validity > 2 || owner == 0 ||
      shape.empty())
    return emitError() << "physical fragment requires type, shape, mappings and owner";
  if (axisMaps.size() != shape.size())
    return emitError() << "fragment axis mappings must match physical rank";
  llvm::DenseSet<uint32_t> fragmentAxes;
  for (auto [axis, extent, mapping] : llvm::enumerate(shape, axisMaps)) {
    auto physical = mlir::dyn_cast<PhysicalExprAttr>(extent);
    if (!physical)
      return emitError() << "fragment extents must be typed physical expressions";
    if (!isCompileTimePhysicalExpr(physical))
      return emitError()
             << "fragment extents must be constant or physical-parameter expressions";
    auto typed = mlir::dyn_cast<AxisMapAttr>(mapping);
    if (!typed || typed.getFragmentAxis() != axis ||
        !fragmentAxes.insert(typed.getFragmentAxis()).second)
      return emitError() << "fragment axis mapping is incomplete or duplicated";
  }
  return success();
}

LogicalResult RangeType::verify(function_ref<InFlightDiagnostic()> emitError,
                                uint64_t sourceId, uint32_t sourceAxis,
                                int64_t dimensionId, bool derived) {
  (void)sourceAxis;
  (void)derived;
  return sourceId != 0 && dimensionId > 0
             ? success()
             : emitError()
                   << "physical range requires source and dimension provenance";
}

LogicalResult BufferType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type elementType,
    ArrayAttr shape, BufferScopeAttr scope, uint64_t instance, uint64_t owner,
    BufferInitializationAttr initialization, uint32_t visibility) {
  if (!elementType || !shape || shape.empty() || !scope || instance == 0 ||
      owner == 0 || !initialization || visibility > 2)
    return emitError() << "physical buffer schema is incomplete";
  if (scope.getValue() != BufferScope::ProgramPrivate &&
      scope.getValue() != BufferScope::IterationPrivate &&
      scope.getValue() != BufferScope::InvocationWorkspace)
    return emitError() << "physical buffer has an unknown allocation scope";
  for (Attribute extent : shape)
    if (!mlir::isa<PhysicalExprAttr>(extent))
      return emitError() << "buffer extents must be typed physical expressions";
  return success();
}

LogicalResult RecordType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr fieldNames,
    ArrayAttr fieldTypes, uint64_t owner) {
  if (!fieldNames || !fieldTypes || fieldNames.empty() ||
      fieldNames.size() != fieldTypes.size() || owner == 0)
    return emitError() << "physical record requires named typed components and owner";
  llvm::StringSet<> names;
  for (auto [name, type] : llvm::zip(fieldNames, fieldTypes)) {
    auto fieldName = mlir::dyn_cast<StringAttr>(name);
    auto fieldType = mlir::dyn_cast<TypeAttr>(type);
    if (!fieldName || fieldName.empty() || !fieldType ||
        !names.insert(fieldName.getValue()).second)
      return emitError() << "physical record fields must have unique names and types";
  }
  return success();
}

} // namespace intent::gpu
