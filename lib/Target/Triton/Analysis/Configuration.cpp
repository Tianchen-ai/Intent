#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/Analysis/Contractions.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;

namespace intent::triton {

std::optional<int64_t> evaluateCompileTimeExpression(
    gpu::PhysicalExprAttr expression, DictionaryAttr bindings) {
  return gpu::evaluatePhysicalExpression(expression, [&](gpu::PhysicalExprAttr leaf)
      -> std::optional<int64_t> {
    if (leaf.getKind() != gpu::PhysicalExprKind::Parameter)
      return std::nullopt;
    auto value = bindings ? bindings.getAs<IntegerAttr>(leaf.getParameterReference().getName()) : IntegerAttr();
    return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
  });
}

bool isTritonFragmentExtent(Attribute attribute) {
  auto expression = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!expression)
    return false;
  auto kind = expression.getKind();
  return kind != gpu::PhysicalExprKind::ScalarABI &&
         isTritonExpression(expression) &&
         llvm::all_of(expression.getOperands(), isTritonFragmentExtent);
}

bool isTritonExpression(gpu::PhysicalExprAttr expression) {
  auto kind = expression.getKind();
  switch (kind) {
  case gpu::PhysicalExprKind::Constant:
  case gpu::PhysicalExprKind::Parameter:
  case gpu::PhysicalExprKind::Dimension:
  case gpu::PhysicalExprKind::ScalarABI:
    return expression.getOperands().empty();
  case gpu::PhysicalExprKind::Add:
  case gpu::PhysicalExprKind::Multiply:
  case gpu::PhysicalExprKind::CeilDiv:
  case gpu::PhysicalExprKind::Minimum:
  case gpu::PhysicalExprKind::Subtract:
  case gpu::PhysicalExprKind::FloorDiv:
  case gpu::PhysicalExprKind::Maximum:
    if (expression.getOperands().size() != 2)
      return false;
    break;
  case gpu::PhysicalExprKind::Select:
    if (expression.getOperands().size() != 3)
      return false;
    break;
  case gpu::PhysicalExprKind::NextPowerOfTwo:
    if (expression.getOperands().size() != 1)
      return false;
    break;
  default:
    return false;
  }
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isTritonExpression(cast<gpu::PhysicalExprAttr>(operand));
  });
}

std::optional<int64_t> descriptorElementBytes(Type type) {
  unsigned bitWidth = type.getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return std::nullopt;
  return bitWidth / 8;
}

namespace {

SmallVector<gpu::FragmentType> collectiveFragments(func::FuncOp kernel) {
  SmallVector<gpu::FragmentType> fragments;
  kernel.walk([&](Operation *operation) {
    auto collect = [&](Value value) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
          fragment && !llvm::is_contained(fragments, fragment))
        fragments.push_back(fragment);
    };
    if (isa<gpu::ContractOp, gpu::ScaledContractOp, gpu::SparseContractOp>(operation)) {
      for (Value result : operation->getResults())
        collect(result);
      return;
    }
    auto store = dyn_cast<gpu::StoreOp>(operation);
    bool workspaceStore = store && isa<gpu::BufferType>(store.getResource().getType());
    if (store)
      if (auto argument = dyn_cast<BlockArgument>(store.getResource());
          argument && argument.getOwner() == &kernel.front()) {
        auto binding = gpu::getArgumentBinding(argument);
        workspaceStore |= binding && binding.getKind() == gpu::ArgumentKind::Workspace;
      }
    if (!isa<gpu::ReduceOp, gpu::ScanOp, ReduceOp, ScanOp>(operation) &&
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

FailureOr<SmallVector<gpu::ConfigurationRequirementAttr>>
collectConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = gpu::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  Builder builder(kernel.getContext());
  const gpu::FragmentResourceAnalysis resources(kernel);
  SmallVector<Operation *> reductions;
  TensorDescriptorChoiceOp descriptorChoice;
  SmallVector<TensorDescriptorOp> descriptors;
  SmallVector<gpu::MakeRangeOp> ranges;
  SmallVector<gpu::ContractOp> contracts;
  kernel.walk([&](Operation *operation) {
    if (auto reduce = dyn_cast<gpu::ReduceOp>(operation))
      reductions.push_back(reduce);
    else if (auto reduce = dyn_cast<ReduceOp>(operation))
      reductions.push_back(reduce);
    else if (auto choice = dyn_cast<TensorDescriptorChoiceOp>(operation))
      descriptorChoice = choice;
    else if (auto descriptor = dyn_cast<TensorDescriptorOp>(operation))
      descriptors.push_back(descriptor);
    else if (auto range = dyn_cast<gpu::MakeRangeOp>(operation))
      ranges.push_back(range);
    else if (auto contract = dyn_cast<gpu::ContractOp>(operation))
      contracts.push_back(contract);
  });
  auto requirements = gpu::collectReductionRequirements(
      kernel, reductions, gpu::ReductionRequirementScope::AllCandidates);
  llvm::append_range(requirements, gpu::collectPointwiseRequirements(kernel, resources));
  auto expression = [&](gpu::PhysicalExprKind kind, int64_t value,
                        ArrayRef<Attribute> operands) {
    return gpu::PhysicalExprAttr::get(kernel.getContext(), kind, value,
        builder.getStringAttr(""), builder.getArrayAttr(operands));
  };
  auto constant = [&](int64_t value) {
    return expression(gpu::PhysicalExprKind::Constant, value, {});
  };
  auto parameter = [&](gpu::ParameterAttr declaration) {
    return gpu::PhysicalExprAttr::get(kernel.getContext(),
        gpu::PhysicalExprKind::Parameter, 0, declaration.getReference(),
        builder.getArrayAttr({}));
  };
  auto append = [&](gpu::ConfigurationRequirementKind kind,
                    gpu::ConfigurationRequirementMetric metric,
                    gpu::ConfigurationRequirementPredicate predicate,
                    gpu::PhysicalExprAttr usage, gpu::PhysicalExprAttr limit,
                    gpu::ParameterRefAttr activation, StringRef message) {
    auto requirement = gpu::ConfigurationRequirementAttr::get(kernel.getContext(),
        kind, metric, predicate, usage, limit, activation,
        builder.getStringAttr(message));
    if (!llvm::is_contained(requirements, requirement))
      requirements.push_back(requirement);
  };
  using Kind = gpu::ConfigurationRequirementKind;
  using Metric = gpu::ConfigurationRequirementMetric;
  using Predicate = gpu::ConfigurationRequirementPredicate;
  for (gpu::ContractOp contract : contracts) {
    if (failed(verifyProgramAttributes(contract)))
      return failure();
    auto requirement = contractionExpansionRequirement(contract);
    if (!requirement)
      return contract.emitOpError("Triton contraction expansion requires a reduction axis"), failure();
    if (!llvm::is_contained(requirements, requirement))
      requirements.push_back(requirement);
  }
  for (gpu::FragmentType fragment : resources.valueTypes()) {
    auto elements = gpu::fragmentElementCount(fragment);
    if (!isTritonFragmentExtent(elements))
      return kernel.emitError("Triton tensor bounds require constexpr fragment extents"),
             failure();
    for (Attribute extent : fragment.getShape())
      append(Kind::Legality, Metric::FragmentElements, Predicate::Positive,
          cast<gpu::PhysicalExprAttr>(extent), {}, {},
          "Triton fragment extents must be positive");
    append(Kind::Legality, Metric::FragmentElements, Predicate::LessEqual,
        elements, constant(maxTritonTensorElements), {},
        "Triton block tensor exceeds the maximum element count");
  }
  for (gpu::MakeRangeOp range : ranges) {
    auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
    if (!fragment || fragment.getShape().size() != 1)
      return range.emitOpError("Triton ranges require one-dimensional fragments"),
             failure();
    append(Kind::Legality, Metric::FragmentElements, Predicate::PowerOfTwo,
        cast<gpu::PhysicalExprAttr>(fragment.getShape()[0]), {}, {},
        "Triton range extent must be a positive power of two");
  }

  if (!descriptors.empty()) {
    auto stages = parameters->find(gpu::ParameterRole::ProviderStages);
    if (!descriptorChoice || !stages)
      return kernel.emitError("Triton descriptors require their access-form and stage declarations"),
             failure();
    auto activation = descriptorChoice.getConfigParameter();
    auto multipleStages = expression(gpu::PhysicalExprKind::Subtract, 0,
                                    {parameter(stages), constant(1)});
    for (TensorDescriptorOp descriptor : descriptors) {
      auto view = cast<gpu::ViewType>(descriptor.getBase().getType());
      auto bytes = descriptorElementBytes(view.getElementType());
      if (!bytes)
        return descriptor.emitOpError("descriptor element type must occupy whole bytes"),
               failure();
      gpu::PhysicalExprAttr elements = constant(1), lastExtent;
      for (Value value : descriptor.getBlockShape()) {
        auto physical = value.getDefiningOp<gpu::PhysicalExprOp>();
        if (!physical || !isTritonFragmentExtent(physical.getExpression()))
          return descriptor.emitOpError("descriptor block shape requires compile-time physical expressions"),
                 failure();
        lastExtent = physical.getExpression();
        append(Kind::Legality, Metric::FragmentElements, Predicate::PowerOfTwo,
            lastExtent, {}, activation,
            "Triton descriptor block extent must be a positive power of two");
        elements = expression(gpu::PhysicalExprKind::Multiply, 0,
                              {elements, lastExtent});
      }
      if (!lastExtent)
        return descriptor.emitOpError("descriptor block shape must not be empty"),
               failure();
      append(Kind::Legality, Metric::FragmentElements, Predicate::LessEqual,
          elements, constant(descriptor.getMaximumBlockElements()), activation,
          "Triton descriptor block exceeds its maximum element count");
      auto contiguousBytes = expression(gpu::PhysicalExprKind::Multiply, 0,
                                       {lastExtent, constant(*bytes)});
      append(Kind::Legality, Metric::FragmentBytes, Predicate::LessEqual,
          constant(descriptor.getMinimumContiguousBytes()), contiguousBytes,
          activation, "Triton descriptor contiguous block is too small");
      auto blockBytes = expression(gpu::PhysicalExprKind::Multiply, 0,
                                  {elements, constant(*bytes)});
      auto alignment = constant(descriptor.getPipelineBlockAlignment());
      auto stageBytes = expression(gpu::PhysicalExprKind::Select, 0,
                                  {multipleStages, blockBytes, alignment});
      append(Kind::Legality, Metric::FragmentBytes, Predicate::MultipleOf,
          stageBytes, alignment, activation,
          "Triton descriptor pipeline stage must preserve block alignment");
    }
  }

  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  auto warps = parameters->find(gpu::ParameterRole::ProviderWarps);
  if (capabilities && capabilities.getRegistersPerUnit() > 0 && warps &&
      warps.getCandidates().size() > 1) {
    auto belowMaximum = expression(gpu::PhysicalExprKind::Subtract, 0,
        {parameter(warps), constant(*llvm::max_element(warps.getCandidates().asArrayRef()))});
    auto nominalBudget = expression(gpu::PhysicalExprKind::Multiply, 0,
                                   {parameter(warps), constant(32 * 255)});
    // Preserve the nominal per-fragment policy: the widest option may spill.
    auto budget = expression(gpu::PhysicalExprKind::Select, 0,
        {belowMaximum, nominalBudget, constant(std::numeric_limits<int64_t>::max())});
    for (gpu::FragmentType fragment : collectiveFragments(kernel))
      append(Kind::NominalBudget, Metric::FragmentRegisterWords,
          Predicate::LessEqual, gpu::fragmentRegisterFootprint(fragment), budget,
          {}, "Triton collective fragment exceeds the nominal per-thread register budget");
  }
  return requirements;
}

LogicalResult verifyConfigurationRequirements(func::FuncOp kernel) {
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements))
    return failure();
  return gpu::verifyConfigurationRequirements(kernel, *requirements);
}

} // namespace intent::triton
