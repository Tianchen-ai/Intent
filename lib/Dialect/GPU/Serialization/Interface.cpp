#include "Intent/Dialect/GPU/Serialization/Interface.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

llvm::json::Array expressions(ArrayAttr attributes) {
  llvm::json::Array result;
  for (Attribute attribute : attributes)
    result.push_back(serializeExpression(cast<PhysicalExprAttr>(attribute)));
  return result;
}

llvm::json::Array integers(ArrayRef<int64_t> values) {
  llvm::json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

FailureOr<llvm::json::Array> configurations(const ParameterSpace &space) {
  auto rows = space.configurations(ConfigurationStage::Complete);
  if (failed(rows))
    return failure();
  llvm::json::Array result;
  for (DictionaryAttr bindings : *rows) {
    llvm::json::Object encoded;
    for (NamedAttribute binding : bindings)
      encoded[binding.getName().getValue()] =
          cast<IntegerAttr>(binding.getValue()).getInt();
    result.push_back(std::move(encoded));
  }
  return result;
}

} // namespace

llvm::json::Value serializeExpression(PhysicalExprAttr expression) {
  StringRef kind;
  switch (expression.getKind()) {
  case PhysicalExprKind::Constant:
    return llvm::json::Object{{"kind", "constant"},
                              {"value", expression.getValue()}};
  case PhysicalExprKind::Parameter: kind = "parameter"; break;
  case PhysicalExprKind::Dimension:
  case PhysicalExprKind::ScalarABI:
    return llvm::json::Object{{"kind", "argument"},
        {"id", expression.getArgumentReference().getId()}};
  case PhysicalExprKind::Add: kind = "add"; break;
  case PhysicalExprKind::Subtract: kind = "subtract"; break;
  case PhysicalExprKind::Multiply: kind = "multiply"; break;
  case PhysicalExprKind::CeilDiv: kind = "ceil_div"; break;
  case PhysicalExprKind::FloorDiv: kind = "floor_div"; break;
  case PhysicalExprKind::Minimum: kind = "min"; break;
  case PhysicalExprKind::Maximum: kind = "max"; break;
  case PhysicalExprKind::Select: kind = "select"; break;
  case PhysicalExprKind::NextPowerOfTwo: kind = "next_power_of_two"; break;
  default: llvm_unreachable("unverified physical expression kind");
  }
  if (kind == "parameter")
    return llvm::json::Object{{"kind", kind},
                              {"symbol", expression.getParameterReference().getName().getValue()}};
  return llvm::json::Object{{"kind", kind},
                            {"operands", expressions(expression.getOperands())}};
}

FailureOr<llvm::json::Object> serializeInterface(
    func::FuncOp kernel, StringRef provider,
    llvm::function_ref<std::string(Value)> kernelName) {
  auto options = readCompileOptions(kernel);
  if (failed(options)) return failure();
  auto facts = ProgramInterface::read(kernel);
  if (failed(facts))
    return failure();
  auto configurationSpace = ParameterSpace::read(kernel);
  if (failed(configurationSpace))
    return failure();
  auto publicInterface = serializePublicInterface(kernel, facts->getPublicInterface());
  if (failed(publicInterface)) return failure();
  llvm::json::Array arguments, parameters;
  for (const ProgramArgument &argument : facts->arguments()) {
    auto binding = argument.binding;
    llvm::json::Object entry{
        {"abi", argument.value.getArgNumber()},
        {"id", binding.getReference().getId()},
        {"kernel_name", kernelName(argument.value)}};
    switch (binding.getKind()) {
    case ArgumentKind::Public:
      entry["kind"] = "public";
      entry["parameter"] = binding.getPublicOrdinal().getInt();
      break;
    case ArgumentKind::Dimension:
    case ArgumentKind::Stride:
      entry["kind"] = binding.getKind() == ArgumentKind::Dimension ? "dimension" : "stride";
      entry["source"] = binding.getSource().getId();
      entry["axis"] = binding.getAxis().getInt();
      if (binding.getDimension()) entry["dimension"] = binding.getDimension().getInt();
      break;
    case ArgumentKind::Workspace: {
      auto view = cast<ViewType>(argument.value.getType());
      auto layout = view.getLayout();
      entry["kind"] = "workspace";
      entry["dtype"] = scalarABIName(view.getElementType());
      entry["shape"] = expressions(layout.getExtents());
      entry["strides"] = expressions(layout.getStrides());
      entry["dimensions"] = integers(layout.getDimensionIds().asArrayRef());
      break;
    }
    }
    arguments.push_back(std::move(entry));
  }

  for (ParameterAttr declaration : configurationSpace->declarations()) {
    PhysicalParameterBinding binding = queryParameterBinding(declaration);
    llvm::json::Object entry{
        {"name", declaration.getName().getValue()},
        {"value_type", scalarABIName(declaration.getValueType())},
        {"role", static_cast<uint32_t>(declaration.getRole())},
        {"category", static_cast<uint32_t>(declaration.getCategory())},
        {"element_bits", declaration.getElementBitWidth()},
        {"candidates", integers(declaration.getCandidates().asArrayRef())}};
    if (declaration.isDeferred()) {
      auto bound = declaration.getBinding().getCoverageBound();
      if (!bound) {
        kernel.emitError("full-coverage declaration has no typed bound expression");
        return failure();
      }
      entry["coverage"] = serializeExpression(bound);
    }
    if (binding.dimension) {
      entry["dimension"] = *binding.dimension;
      if (auto dimension = facts->dimension(*binding.dimension)) {
        auto metadata = getArgumentBinding(dimension);
        auto source = facts->resolve(metadata.getSource());
        auto owner = getArgumentBinding(source);
        if (owner.getKind() == ArgumentKind::Public)
          entry["argument_axis"] = llvm::json::Array{
              owner.getPublicOrdinal().getInt(), metadata.getAxis().getInt()};
      }
    }
    if (binding.source)
      entry["source"] = llvm::json::Array{binding.source->sourceId,
                                         binding.source->sourceAxis,
                                         binding.source->derived};
    parameters.push_back(std::move(entry));
  }

  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space) {
    kernel.emitError("final GPU program has no launch grid");
    return failure();
  }
  llvm::json::Array overlaps;
  bool valid = true;
  kernel.walk([&](ViewOverlapOp overlap) {
    auto lhs = dyn_cast<BlockArgument>(overlap.getLhs());
    auto rhs = dyn_cast<BlockArgument>(overlap.getRhs());
    if (!lhs || !rhs || lhs.getOwner() != &kernel.front() ||
        rhs.getOwner() != &kernel.front()) {
      overlap.emitError("launch overlap must refer to physical ABI arguments");
      valid = false;
      return;
    }
    overlaps.push_back(llvm::json::Object{
        {"name", kernelName(overlap.getResult())},
        {"lhs", getArgumentReference(lhs).getId()},
        {"rhs", getArgumentReference(rhs).getId()}});
  });
  if (!valid)
    return failure();

  auto declaredRequirements = configurationSpace->requirements();
  if (failed(declaredRequirements)) return failure();
  llvm::json::Array requirements;
  for (ConfigurationRequirementAttr requirement : *declaredRequirements)
    requirements.push_back(llvm::json::Object{
        {"kind", stringifyConfigurationRequirementKind(requirement.getKind())},
        {"metric", stringifyConfigurationRequirementMetric(requirement.getMetric())},
        {"predicate", stringifyConfigurationRequirementPredicate(requirement.getPredicate())},
        {"usage", serializeExpression(requirement.getUsage())},
        {"limit", requirement.getLimit()
                      ? serializeExpression(requirement.getLimit())
                      : llvm::json::Value(nullptr)},
        {"activation", requirement.getActivation()
                           ? llvm::json::Value(requirement.getActivation().getName().getValue())
                           : llvm::json::Value(nullptr)},
        {"message", requirement.getMessage().getValue()}});
  auto configs = configurations(*configurationSpace);
  if (failed(configs))
    return failure();
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities) {
    kernel.emitError("generated GPU program has no compilation target");
    return failure();
  }
  llvm::json::Object target{
      {"family", "gpu"},
      {"capabilities", llvm::json::Object{
          {"compute_units", capabilities.getComputeUnits()},
          {"shared_memory_per_unit", capabilities.getSharedMemoryPerUnit()},
          {"max_dynamic_shared_memory_per_block", capabilities.getMaxDynamicSharedMemoryPerBlock()},
          {"registers_per_unit", capabilities.getRegistersPerUnit()},
          {"max_threads_per_block", capabilities.getMaxThreadsPerBlock()},
          {"compute_capability_major", capabilities.getComputeCapabilityMajor()},
          {"compute_capability_minor", capabilities.getComputeCapabilityMinor()},
          {"single_to_double_precision_perf_ratio", capabilities.getSingleToDoublePrecisionPerfRatio()},
          {"matrix_units", capabilities.getMatrixUnits()},
          {"dynamic_vector_width", capabilities.getDynamicVectorWidth()}}}};
  llvm::json::Object gpu{
      {"arguments", std::move(arguments)},
      {"parameters", std::move(parameters)}, {"grid", expressions(space)},
      {"overlaps", std::move(overlaps)},
      {"configurations", std::move(*configs)},
      {"requirements", std::move(requirements)}};
  return llvm::json::Object{{"provider", provider},
                            {"compile_options", serializeCompileOptions(*options)},
                            {"entry_name", kernel.getName()},
                            {"target", std::move(target)},
                            {"interface", std::move(*publicInterface)},
                            {"gpu", std::move(gpu)}};
}

} // namespace intent::gpu
