#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/Analysis/Contractions.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/Resources.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/IR/PhysicalExpressions.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/IR/ProgramInterface.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Target/Triton/IR/Target.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;

namespace intent::triton {

std::optional<int64_t> evaluateCompileTimeExpression(
    block::PhysicalExprAttr expression, DictionaryAttr bindings) {
  return block::evaluatePhysicalExpression(expression, [&](block::PhysicalExprAttr leaf)
      -> std::optional<int64_t> {
    if (leaf.getKind() != block::PhysicalExprKind::Parameter)
      return std::nullopt;
    auto value = bindings ? bindings.getAs<IntegerAttr>(leaf.getParameterReference().getName()) : IntegerAttr();
    return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
  });
}

bool isTritonFragmentExtent(Attribute attribute) {
  auto expression = dyn_cast<block::PhysicalExprAttr>(attribute);
  if (!expression)
    return false;
  auto kind = expression.getKind();
  return kind != block::PhysicalExprKind::ScalarABI &&
         isTritonExpression(expression) &&
         llvm::all_of(expression.getOperands(), isTritonFragmentExtent);
}

bool isTritonExpression(block::PhysicalExprAttr expression) {
  auto kind = expression.getKind();
  switch (kind) {
  case block::PhysicalExprKind::Constant:
  case block::PhysicalExprKind::Parameter:
  case block::PhysicalExprKind::Dimension:
  case block::PhysicalExprKind::ScalarABI:
    return expression.getOperands().empty();
  case block::PhysicalExprKind::Add:
  case block::PhysicalExprKind::Multiply:
  case block::PhysicalExprKind::CeilDiv:
  case block::PhysicalExprKind::Minimum:
  case block::PhysicalExprKind::Subtract:
  case block::PhysicalExprKind::FloorDiv:
  case block::PhysicalExprKind::Maximum:
    if (expression.getOperands().size() != 2)
      return false;
    break;
  case block::PhysicalExprKind::Select:
    if (expression.getOperands().size() != 3)
      return false;
    break;
  case block::PhysicalExprKind::NextPowerOfTwo:
    if (expression.getOperands().size() != 1)
      return false;
    break;
  default:
    return false;
  }
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isTritonExpression(cast<block::PhysicalExprAttr>(operand));
  });
}

std::optional<int64_t> descriptorElementBytes(Type type) {
  unsigned bitWidth = type.getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return std::nullopt;
  return bitWidth / 8;
}

namespace {

SmallVector<block::FragmentType> collectiveFragments(func::FuncOp kernel) {
  SmallVector<block::FragmentType> fragments;
  kernel.walk([&](Operation *operation) {
    auto collect = [&](Value value) {
      if (auto fragment = dyn_cast<block::FragmentType>(value.getType());
          fragment && !llvm::is_contained(fragments, fragment))
        fragments.push_back(fragment);
    };
    if (isa<block::ContractOp, block::ScaledContractOp, block::SparseContractOp>(operation)) {
      for (Value result : operation->getResults())
        collect(result);
      return;
    }
    auto store = dyn_cast<block::StoreOp>(operation);
    bool workspaceStore = store && isa<block::BufferType>(store.getResource().getType());
    if (store)
      if (auto argument = dyn_cast<BlockArgument>(store.getResource());
          argument && argument.getOwner() == &kernel.front()) {
        auto binding = block::getArgumentBinding(argument);
        workspaceStore |= binding && binding.getKind() == block::ArgumentKind::Workspace;
      }
    if (!isa<block::ReduceOp, block::ScanOp, ReduceOp, ScanOp>(operation) &&
        !workspaceStore)
      return;
    if (workspaceStore)
      collect(store.getValue());
    else
      for (Value operand : operation->getOperands())
        collect(operand);
  });
  return fragments;
}

} // namespace

FailureOr<SmallVector<block::ConfigurationRequirementAttr>>
collectConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = block::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  Builder builder(kernel.getContext());
  const block::FragmentResourceAnalysis resources(kernel);
  SmallVector<Operation *> reductions;
  TensorDescriptorChoiceOp descriptorChoice;
  SmallVector<TensorDescriptorOp> descriptors;
  SmallVector<block::MakeRangeOp> ranges;
  SmallVector<block::ContractOp> contracts;
  kernel.walk([&](Operation *operation) {
    if (auto reduce = dyn_cast<block::ReduceOp>(operation))
      reductions.push_back(reduce);
    else if (auto reduce = dyn_cast<ReduceOp>(operation))
      reductions.push_back(reduce);
    else if (auto choice = dyn_cast<TensorDescriptorChoiceOp>(operation))
      descriptorChoice = choice;
    else if (auto descriptor = dyn_cast<TensorDescriptorOp>(operation))
      descriptors.push_back(descriptor);
    else if (auto range = dyn_cast<block::MakeRangeOp>(operation))
      ranges.push_back(range);
    else if (auto contract = dyn_cast<block::ContractOp>(operation))
      contracts.push_back(contract);
  });
  auto requirements = block::collectReductionRequirements(
      kernel, reductions, block::ReductionRequirementScope::AllCandidates);
  llvm::append_range(requirements, block::collectPointwiseRequirements(kernel, resources));
  auto expression = [&](block::PhysicalExprKind kind, int64_t value,
                        ArrayRef<Attribute> operands) {
    return block::PhysicalExprAttr::get(kernel.getContext(), kind, value,
        builder.getStringAttr(""), builder.getArrayAttr(operands));
  };
  auto constant = [&](int64_t value) {
    return expression(block::PhysicalExprKind::Constant, value, {});
  };
  auto parameter = [&](block::ParameterAttr declaration) {
    return block::PhysicalExprAttr::get(kernel.getContext(),
        block::PhysicalExprKind::Parameter, 0, declaration.getReference(),
        builder.getArrayAttr({}));
  };
  auto append = [&](block::ConfigurationRequirementKind kind,
                    block::ConfigurationRequirementMetric metric,
                    block::ConfigurationRequirementPredicate predicate,
                    block::PhysicalExprAttr usage, block::PhysicalExprAttr limit,
                    block::ParameterRefAttr activation, StringRef message) {
    auto requirement = block::ConfigurationRequirementAttr::get(kernel.getContext(),
        kind, metric, predicate, usage, limit, activation,
        builder.getStringAttr(message), true);
    if (!llvm::is_contained(requirements, requirement))
      requirements.push_back(requirement);
  };
  using Kind = block::ConfigurationRequirementKind;
  using Metric = block::ConfigurationRequirementMetric;
  using Predicate = block::ConfigurationRequirementPredicate;
  for (block::ContractOp contract : contracts) {
    if (failed(verifyProgramAttributes(contract)))
      return failure();
    auto requirement = contractionExpansionRequirement(contract);
    if (!requirement)
      return contract.emitOpError("Triton contraction expansion requires a reduction axis"), failure();
    if (!llvm::is_contained(requirements, requirement))
      requirements.push_back(requirement);
  }
  for (block::FragmentType fragment : resources.valueTypes()) {
    auto elements = block::fragmentElementCount(fragment);
    if (!isTritonFragmentExtent(elements))
      return kernel.emitError("Triton tensor bounds require constexpr fragment extents"),
             failure();
    for (Attribute extent : fragment.getShape())
      append(Kind::Legality, Metric::FragmentElements, Predicate::Positive,
          cast<block::PhysicalExprAttr>(extent), {}, {},
          "Triton fragment extents must be positive");
    append(Kind::Legality, Metric::FragmentElements, Predicate::LessEqual,
        elements, constant(maxTritonTensorElements), {},
        "Triton block tensor exceeds the maximum element count");
  }
  for (block::MakeRangeOp range : ranges) {
    auto fragment = dyn_cast<block::FragmentType>(range.getResult().getType());
    if (!fragment || fragment.getShape().size() != 1)
      return range.emitOpError("Triton ranges require one-dimensional fragments"),
             failure();
    append(Kind::Legality, Metric::FragmentElements, Predicate::PowerOfTwo,
        cast<block::PhysicalExprAttr>(fragment.getShape()[0]), {}, {},
        "Triton range extent must be a positive power of two");
  }

  if (!descriptors.empty()) {
    auto stages = parameters->lookup("NUM_STAGES");
    if (!descriptorChoice || !stages)
      return kernel.emitError("Triton descriptors require their access-form and stage declarations"),
             failure();
    auto activation = descriptorChoice.getConfigParameter();
    auto multipleStages = expression(block::PhysicalExprKind::Subtract, 0,
                                    {parameter(stages), constant(1)});
    for (TensorDescriptorOp descriptor : descriptors) {
      auto view = cast<block::ViewType>(descriptor.getBase().getType());
      auto bytes = descriptorElementBytes(view.getElementType());
      if (!bytes)
        return descriptor.emitOpError("descriptor element type must occupy whole bytes"),
               failure();
      block::PhysicalExprAttr elements = constant(1), lastExtent;
      for (Value value : descriptor.getBlockShape()) {
        auto physical = value.getDefiningOp<block::PhysicalExprOp>();
        if (!physical || !isTritonFragmentExtent(physical.getExpression()))
          return descriptor.emitOpError("descriptor block shape requires compile-time physical expressions"),
                 failure();
        lastExtent = physical.getExpression();
        append(Kind::Legality, Metric::FragmentElements, Predicate::PowerOfTwo,
            lastExtent, {}, activation,
            "Triton descriptor block extent must be a positive power of two");
        elements = expression(block::PhysicalExprKind::Multiply, 0,
                              {elements, lastExtent});
      }
      if (!lastExtent)
        return descriptor.emitOpError("descriptor block shape must not be empty"),
               failure();
      append(Kind::Legality, Metric::FragmentElements, Predicate::LessEqual,
          elements, constant(descriptor.getMaximumBlockElements()), activation,
          "Triton descriptor block exceeds its maximum element count");
      auto contiguousBytes = expression(block::PhysicalExprKind::Multiply, 0,
                                       {lastExtent, constant(*bytes)});
      append(Kind::Legality, Metric::FragmentBytes, Predicate::LessEqual,
          constant(descriptor.getMinimumContiguousBytes()), contiguousBytes,
          activation, "Triton descriptor contiguous block is too small");
      auto blockBytes = expression(block::PhysicalExprKind::Multiply, 0,
                                  {elements, constant(*bytes)});
      auto alignment = constant(descriptor.getPipelineBlockAlignment());
      auto stageBytes = expression(block::PhysicalExprKind::Select, 0,
                                  {multipleStages, blockBytes, alignment});
      append(Kind::Legality, Metric::FragmentBytes, Predicate::MultipleOf,
          stageBytes, alignment, activation,
          "Triton descriptor pipeline stage must preserve block alignment");
    }
  }

  auto capabilities = kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
  auto warps = parameters->lookup("NUM_WARPS");
  auto target = capabilities ? capabilities.getTarget() : Attribute();
  auto interface = targetInterface(target);
  int64_t wordsPerWarp = interface ? interface->valueWordsPerWarp(target) : 0;
  if (wordsPerWarp > 0 && capabilities.getValueBudgetWords() > 0 && warps &&
      warps.getCandidates().size() > 1) {
    auto belowMaximum = expression(block::PhysicalExprKind::Subtract, 0,
        {parameter(warps), constant(*llvm::max_element(warps.getCandidates().asArrayRef()))});
    auto nominalBudget = expression(block::PhysicalExprKind::Multiply, 0,
                                   {parameter(warps), constant(wordsPerWarp)});
    // Preserve the nominal per-fragment policy: the widest option may spill.
    auto budget = expression(block::PhysicalExprKind::Select, 0,
        {belowMaximum, nominalBudget, constant(std::numeric_limits<int64_t>::max())});
    for (block::FragmentType fragment : collectiveFragments(kernel))
      append(Kind::NominalBudget, Metric::FragmentRegisterWords,
          Predicate::LessEqual, block::fragmentRegisterFootprint(fragment), budget,
          {}, "Triton collective fragment exceeds the nominal per-thread register budget");
  }
  if (descriptorChoice && !contracts.empty()) {
    // The descriptor variant can move live matrix tiles to shared memory.
    // Its actual register use is determined after native layout selection.
    // Pointer variants retain the ordinary compilation budget.
    SmallVector<block::ConfigurationRequirementAttr> byForm;
    auto activation = descriptorChoice.getConfigParameter();
    for (auto rule : requirements) {
      if (rule.getKind() != Kind::NominalBudget ||
          rule.getMetric() != Metric::FragmentRegisterWords || rule.getActivation()) {
        byForm.push_back(rule);
        continue;
      }
      for (bool descriptor : {false, true})
        byForm.push_back(block::ConfigurationRequirementAttr::get(kernel.getContext(),
            descriptor ? Kind::WorkingSetEstimate : Kind::NominalBudget,
            rule.getMetric(), rule.getPredicate(), rule.getUsage(), rule.getLimit(),
            activation, rule.getMessage(), descriptor));
    }
    return byForm;
  }
  return requirements;
}

LogicalResult verifyConfigurationRequirements(func::FuncOp kernel) {
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements))
    return failure();
  return block::verifyConfigurationRequirements(kernel, *requirements);
}

} // namespace intent::triton
