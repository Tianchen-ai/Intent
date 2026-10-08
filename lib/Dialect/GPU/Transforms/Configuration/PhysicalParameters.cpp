#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <string>

using namespace mlir;

namespace intent::gpu {
namespace {

void invalidateConfigurationRows(func::FuncOp kernel) {
  auto set = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
  if (!set) return;
  kernel->setAttr(configurationsAttr, ConfigurationSetAttr::get(
      kernel.getContext(), ConfigurationStage::Shared,
      ArrayAttr::get(kernel.getContext(), {}), set.getRequirements()));
}

void invalidateConfigurations(func::FuncOp kernel, ParameterAttr before,
                              ParameterAttr after) {
  auto set = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
  if (!set) return;
  bool providerOnly = (!before || before.getPhase() == ConfigurationBindingPhase::Provider) &&
                      (!after || after.getPhase() == ConfigurationBindingPhase::Provider);
  if (!providerOnly || set.getStage() == ConfigurationStage::Complete)
    invalidateConfigurationRows(kernel);
}

void deduplicateRequirements(func::FuncOp kernel) {
  auto set = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
  if (!set) return;
  llvm::DenseSet<Attribute> unique;
  SmallVector<Attribute> retained;
  for (Attribute requirement : set.getRequirements())
    if (unique.insert(requirement).second) retained.push_back(requirement);
  if (retained.size() != set.getRequirements().size())
    kernel->setAttr(configurationsAttr, ConfigurationSetAttr::get(
        kernel.getContext(), set.getStage(), set.getRows(),
        ArrayAttr::get(kernel.getContext(), retained)));
}

LogicalResult verifyDeclaration(func::FuncOp kernel, ParameterAttr declaration) {
  if (!declaration) return kernel.emitError("cannot publish a null parameter declaration");
  auto emit = [&] { return kernel.emitError("invalid compile-time parameter declaration"); };
  auto binding = declaration.getBinding();
  if (!binding) return emit() << "; missing typed binding";
  if (failed(ParameterBindingAttr::verify(emit, binding.getDimension(), binding.getSource(),
          binding.getCoverageBound(), binding.getGroup(), binding.getPointwiseChunk(),
          binding.getPointwiseLocal())))
    return failure();
  return ParameterAttr::verify(emit, declaration.getName(), declaration.getValueType(),
      declaration.getRole(), declaration.getCategory(), declaration.getElementBitWidth(),
      declaration.getCandidates(), declaration.getPhase(), binding);
}

llvm::DenseSet<ParameterRefAttr> referencedParameters(func::FuncOp kernel) {
  llvm::DenseSet<ParameterRefAttr> references;
  AttrTypeWalker walker;
  walker.addWalk([&](ParameterRefAttr reference) { references.insert(reference); });
  kernel.walk([&](Operation *operation) {
    if (operation == kernel.getOperation()) {
      // Rows do not keep a declaration alive. Requirements do: their extents
      // must remain available even after all SSA parameter reads are folded.
      // A retained declaration's dependencies are reached below.
      for (NamedAttribute attribute : operation->getAttrs()) {
        if (attribute.getName() == configurationsAttr)
          walker.walk(cast<ConfigurationSetAttr>(attribute.getValue()).getRequirements());
        else if (attribute.getName() != parametersAttr)
          walker.walk(attribute.getValue());
      }
    } else {
      walker.walk(operation->getAttrDictionary());
    }
    for (Type type : operation->getResultTypes()) walker.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) walker.walk(argument.getType());
  });
  auto declarations = getParameterDeclarations(kernel);
  if (!declarations) return references;
  // Provider options are part of the native compile interface even when they
  // are not read by the kernel body (for example Triton's warp/CTA counts).
  for (Attribute attribute : declarations) {
    auto declaration = cast<ParameterAttr>(attribute);
    if (declaration.getPhase() == ConfigurationBindingPhase::Provider)
      references.insert(declaration.getReference());
  }
  llvm::DenseSet<ParameterRefAttr> visited;
  bool changed;
  do {
    changed = false;
    for (Attribute attribute : declarations) {
      auto declaration = cast<ParameterAttr>(attribute);
      auto reference = declaration.getReference();
      if (!references.contains(reference) || !visited.insert(reference).second) continue;
      walker.walk(declaration.getBinding());
      changed = true;
    }
  } while (changed);
  return references;
}

} // namespace

LogicalResult writeConfigurations(func::FuncOp kernel,
                                  ArrayRef<DictionaryAttr> rows,
                                  ConfigurationStage stage) {
  auto space = ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto requirements = space->requirements();
  if (failed(requirements)) return failure();
  return writeConfigurations(kernel, rows, stage, *requirements);
}

LogicalResult writeConfigurations(
    func::FuncOp kernel, ArrayRef<DictionaryAttr> rows, ConfigurationStage stage,
    ArrayRef<ConfigurationRequirementAttr> requirements) {
  auto space = ParameterSpace::read(kernel);
  if (failed(space) || failed(space->verifyRequirements(requirements)))
    return failure();
  if (rows.empty())
    return kernel.emitError("cannot publish an empty candidate set");
  for (DictionaryAttr row : rows)
    if (failed(space->verifyBindings(row, stage))) return failure();
  SmallVector<Attribute> encoded(rows.begin(), rows.end());
  SmallVector<Attribute> encodedRequirements(requirements.begin(), requirements.end());
  auto set = ConfigurationSetAttr::getChecked(
      [&] { return kernel.emitError(); }, kernel.getContext(), stage,
      ArrayAttr::get(kernel.getContext(), encoded),
      ArrayAttr::get(kernel.getContext(), encodedRequirements));
  if (!set) return failure();
  kernel->setAttr(configurationsAttr, set);
  return success();
}

FailureOr<ParameterRefAttr> declareParameter(func::FuncOp kernel,
                                             ParameterAttr declaration) {
  if (failed(verifyDeclaration(kernel, declaration))) return failure();
  auto declarations = getParameterDeclarations(kernel);
  if (!declarations)
    return kernel.emitError("cannot declare a parameter without its kernel-owned table"), failure();
  if (auto existing = lookupParameter(kernel, declaration.getName())) {
    if (existing == declaration) return existing.getReference();
    return kernel.emitError("compile-time parameter already has a different declaration: ")
           << declaration.getName(), failure();
  }
  SmallVector<Attribute> updated{declaration};
  llvm::append_range(updated, declarations);
  kernel->setAttr(parametersAttr, ArrayAttr::get(kernel.getContext(), updated));
  invalidateConfigurations(kernel, {}, declaration);
  return declaration.getReference();
}

FailureOr<ParameterRefAttr> getOrCreatePhysicalParameter(
    func::FuncOp kernel, StringRef name, ParameterRole role,
    ParameterCategory category, uint32_t elementBitWidth,
    ArrayRef<int64_t> candidates, ParameterBindingAttr binding) {
  Builder builder(kernel.getContext());
  const bool explicitBinding = bool(binding);
  if (!binding) binding = ParameterBindingAttr::get(kernel.getContext(), {}, {}, {}, {}, false, false);
  auto phase = category == ParameterCategory::Provider ? ConfigurationBindingPhase::Provider
             : category == ParameterCategory::Coverage ? ConfigurationBindingPhase::Deferred
                                                       : ConfigurationBindingPhase::Shared;
  auto declaration = ParameterAttr::get(kernel.getContext(), builder.getStringAttr(name),
      builder.getIndexType(), role, category,
      elementBitWidth, builder.getDenseI64ArrayAttr(candidates), phase, binding);
  if (auto existing = lookupParameter(kernel, declaration.getName())) {
    if (!existing.isExtent() || existing.getRole() != declaration.getRole() ||
        existing.getCategory() != declaration.getCategory() ||
        existing.getCandidates() != declaration.getCandidates() ||
        (explicitBinding && existing.getBinding() != binding))
      return kernel.emitError("physical parameter is reused with an incompatible domain: ")
             << name, failure();
    uint32_t width = std::max(existing.getElementBitWidth(), elementBitWidth);
    if (width != existing.getElementBitWidth()) {
      auto widened = ParameterAttr::get(kernel.getContext(), existing.getName(), existing.getValueType(),
          existing.getRole(), existing.getCategory(), width, existing.getCandidates(),
          existing.getPhase(), existing.getBinding());
      if (failed(updateParameter(kernel, widened))) return failure();
    }
    return existing.getReference();
  }
  return declareParameter(kernel, declaration);
}

ParameterOp materializeParameter(OpBuilder &builder, Location location,
                                 ParameterRefAttr reference) {
  return builder.create<ParameterOp>(location, builder.getIndexType(), reference);
}

LogicalResult updateParameter(func::FuncOp kernel, ParameterAttr declaration) {
  if (failed(verifyDeclaration(kernel, declaration))) return failure();
  auto previous = lookupParameter(kernel, declaration.getName());
  if (!previous) return kernel.emitError("cannot update an undeclared parameter: ") << declaration.getName();
  if (previous == declaration) return success();
  if (previous.getValueType() != declaration.getValueType())
    return kernel.emitError("parameter update cannot change the type of existing references");
  SmallVector<Attribute> updated;
  for (Attribute attribute : getParameterDeclarations(kernel))
    updated.push_back(attribute == previous ? declaration : attribute);
  kernel->setAttr(parametersAttr, ArrayAttr::get(kernel.getContext(), updated));
  invalidateConfigurations(kernel, previous, declaration);
  return success();
}

LogicalResult replaceParameter(func::FuncOp kernel, ParameterRefAttr previous,
                                ParameterRefAttr replacement) {
  if (previous == replacement) return success();
  auto before = lookupParameter(kernel, previous);
  auto after = lookupParameter(kernel, replacement);
  if (!before || !after || before.getValueType() != after.getValueType())
    return kernel.emitError("parameter replacement requires two same-type declarations");
  invalidateConfigurations(kernel, before, after);
  SmallVector<Attribute> updated;
  for (Attribute attribute : getParameterDeclarations(kernel))
    if (attribute != before) updated.push_back(attribute);
  kernel->setAttr(parametersAttr, ArrayAttr::get(kernel.getContext(), updated));
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParameterRefAttr reference) -> std::optional<Attribute> {
    return reference == previous ? std::optional<Attribute>(replacement) : std::nullopt;
  });
  replacer.recursivelyReplaceElementsIn(kernel, true, false, true);
  // Merging declarations can make two formerly distinct requirements equal.
  deduplicateRequirements(kernel);
  return success();
}

LogicalResult renameParameters(func::FuncOp kernel,
                               function_ref<StringAttr(StringAttr)> rename) {
  auto space = ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  llvm::DenseMap<StringAttr, StringAttr> names;
  llvm::DenseSet<StringAttr> unique;
  for (ParameterAttr declaration : space->declarations()) {
    auto name = rename(declaration.getName());
    if (!name || name.empty() || !unique.insert(name).second)
      return kernel.emitError("parameter rename must produce unique nonempty names");
    names.try_emplace(declaration.getName(), name);
  }
  invalidateConfigurationRows(kernel);
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParameterAttr declaration) -> std::optional<Attribute> {
    auto found = names.find(declaration.getName());
    return found == names.end() ? std::nullopt
                               : std::optional<Attribute>(declaration.withName(found->second));
  });
  replacer.addReplacement([&](ParameterRefAttr reference) -> std::optional<Attribute> {
    auto found = names.find(reference.getName());
    return found == names.end() ? std::nullopt
        : std::optional<Attribute>(ParameterRefAttr::get(kernel.getContext(), found->second));
  });
  replacer.recursivelyReplaceElementsIn(kernel, true, false, true);
  return success();
}

void eraseUnusedParameters(func::FuncOp kernel) {
  SmallVector<ParameterOp> deadReads;
  kernel.walk([&](ParameterOp read) {
    if (read.getResult().use_empty()) deadReads.push_back(read);
  });
  for (ParameterOp read : deadReads) read.erase();
  auto declarations = getParameterDeclarations(kernel);
  if (!declarations) return;
  auto referenced = referencedParameters(kernel);
  SmallVector<Attribute> retained;
  for (Attribute attribute : declarations) {
    auto declaration = cast<ParameterAttr>(attribute);
    if (referenced.contains(declaration.getReference())) retained.push_back(declaration);
    else invalidateConfigurations(kernel, declaration, {});
  }
  if (retained.size() != declarations.size())
    kernel->setAttr(parametersAttr, ArrayAttr::get(kernel.getContext(), retained));
}

LogicalResult projectBoundedParameterUses(func::FuncOp kernel) {
  SmallVector<ParameterAttr> parameters;
  for (Attribute attribute : getParameterDeclarations(kernel))
    parameters.push_back(cast<ParameterAttr>(attribute));
  auto walkProgram = [&](AttrTypeWalker &walker) {
    kernel.walk([&](Operation *operation) {
      for (NamedAttribute attribute : operation->getAttrs())
        if (operation != kernel.getOperation() || attribute.getName() != parametersAttr)
          walker.walk<WalkOrder::PreOrder>(attribute.getValue());
      for (Type type : operation->getResultTypes())
        walker.walk<WalkOrder::PreOrder>(type);
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            walker.walk<WalkOrder::PreOrder>(argument.getType());
    });
  };
  for (ParameterAttr parameter : parameters) {
    if (!parameter.isExtent() || parameter.getPhase() != ConfigurationBindingPhase::Shared)
      continue;
    auto reference = parameter.getReference();
    ParameterAttr capacity;
    PhysicalExprAttr bounded;
    bool ambiguous = false;
    auto match = [&](PhysicalExprAttr expression) -> ParameterAttr {
      if (expression.getKind() != PhysicalExprKind::Minimum || expression.getOperands().size() != 2)
        return {};
      auto first = cast<PhysicalExprAttr>(expression.getOperands()[0]);
      auto second = cast<PhysicalExprAttr>(expression.getOperands()[1]);
      if (first.getKind() != PhysicalExprKind::Parameter ||
          second.getKind() != PhysicalExprKind::Parameter)
        return {};
      if (second.getParameterReference() == reference) std::swap(first, second);
      if (first.getParameterReference() != reference) return {};
      auto other = lookupParameter(kernel, second.getParameterReference());
      return other && other.isDeferred() && other.getBinding().getCoverageBound()
                 ? other : ParameterAttr{};
    };
    AttrTypeWalker discovery;
    discovery.addWalk([&](PhysicalExprAttr expression) {
      auto found = match(expression);
      if (!found) return;
      if (capacity && capacity != found) ambiguous = true;
      else { capacity = found; bounded = expression; }
    });
    walkProgram(discovery);
    if (!bounded || ambiguous) continue;

    bool early = false;
    AttrTypeWalker earlyUses;
    earlyUses.addWalk([&](ParameterRefAttr use) { early |= use == reference; });
    for (Attribute attribute : getParameterDeclarations(kernel))
      earlyUses.walk(cast<ParameterAttr>(attribute).getBinding());
    for (BlockArgument argument : kernel.getArguments())
      if (auto binding = getArgumentBinding(argument);
          binding && binding.getKind() == ArgumentKind::Workspace)
        earlyUses.walk(argument.getType());
    if (early) continue;

    bool raw = false;
    AttrTypeWalker uses;
    uses.addWalk([&](PhysicalExprAttr expression) {
      return match(expression) == capacity ? WalkResult::skip() : WalkResult::advance();
    });
    uses.addWalk([&](ParameterRefAttr use) { raw |= use == reference; });
    walkProgram(uses);
    if (raw) continue;

    Builder builder(kernel.getContext());
    std::string stem = parameter.getName().getValue().str() + "_EFFECTIVE";
    std::string name = stem;
    for (unsigned suffix = 1; lookupParameter(kernel, builder.getStringAttr(name)); ++suffix)
      name = stem + "_" + std::to_string(suffix);
    auto projectedReference = ParameterRefAttr::get(kernel.getContext(), builder.getStringAttr(name));
    auto projectedExpression = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Parameter,
        0, projectedReference, builder.getArrayAttr({}));
    SmallVector<int64_t> candidates(parameter.getCandidates().asArrayRef());
    llvm::append_range(candidates, capacity.getCandidates().asArrayRef());
    llvm::sort(candidates);
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    auto projected = ParameterAttr::get(kernel.getContext(), builder.getStringAttr(name),
        parameter.getValueType(), parameter.getRole(), parameter.getCategory(),
        parameter.getElementBitWidth(), builder.getDenseI64ArrayAttr(candidates),
        ConfigurationBindingPhase::Deferred, parameter.getBinding().withCoverageBound(bounded));
    // Rewrite the current program before adding the new declaration, whose
    // binding deliberately retains the original expression and candidate row.
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](PhysicalExprAttr expression) -> std::optional<Attribute> {
      return match(expression) == capacity ? std::optional<Attribute>(projectedExpression) : std::nullopt;
    });
    replacer.recursivelyReplaceElementsIn(kernel, true, false, true);
    auto configurations = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
    if (failed(declareParameter(kernel, projected))) return failure();
    if (configurations) {
      SmallVector<DictionaryAttr> rows;
      SmallVector<ConfigurationRequirementAttr> requirements;
      for (Attribute row : configurations.getRows()) rows.push_back(cast<DictionaryAttr>(row));
      for (Attribute requirement : configurations.getRequirements())
        requirements.push_back(cast<ConfigurationRequirementAttr>(requirement));
      if (failed(writeConfigurations(kernel, rows, configurations.getStage(), requirements)))
        return failure();
    }
  }
  return success();
}

} // namespace intent::gpu
