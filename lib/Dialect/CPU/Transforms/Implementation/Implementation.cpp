#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Dialect/CPU/IR/ImplementationProvider.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::cpu {

ImplementationParameter ImplementationParameter::local(StringRef name, ImplementationParameterDomain domain) {
  return {name, Local{name}, std::move(domain)};
}
ImplementationParameter ImplementationParameter::alias(StringRef name, StringRef source) {
  return {name, Local{source}, {}};
}
ImplementationParameter ImplementationParameter::constant(StringRef name, int64_t value) {
  return {name, Constant{value}, {}};
}
ImplementationParameter ImplementationParameter::minimum(StringRef name, int64_t limit, ArrayRef<Axis> axes) {
  return {name, Minimum{limit, llvm::to_vector(axes)}, {}};
}

namespace {
int64_t minimumParameter(const ImplementationParameter::Minimum &source, ConfigurationAttr config) {
  int64_t value = source.limit;
  for (auto axis : source.axes)
    value = std::min(value, axis == ImplementationParameter::Axis::TileM
                               ? config.getTileM() : config.getTileN());
  return value;
}

std::optional<std::string> checkParameters(const Implementation &implementation,
    DictionaryAttr values, CapabilitiesAttr capabilities, ConfigurationAttr configuration) {
  if (values.size() != implementation.parameters.size())
    return "parameter keys do not match the implementation schema";
  for (const auto &parameter : implementation.parameters) {
    auto integer = values.getAs<IntegerAttr>(parameter.name);
    if (!integer || !integer.getValue().isSignedIntN(64) || integer.getInt() <= 0)
      return "parameter '" + parameter.name.str() + "' requires a positive signed-64-bit representable integer";
    int64_t value = integer.getInt();
    const auto &domain = parameter.domain;
    if ((domain.powerOfTwo && !llvm::isPowerOf2_64(value)) ||
        (domain.maximum && value > *domain.maximum) ||
        (domain.laneBits && value > capabilities.getVectorBits() / domain.laneBits) ||
        (!domain.values.empty() && !llvm::is_contained(domain.values, value)))
      return "parameter '" + parameter.name.str() + "' is outside its declared implementation domain";
    if (auto local = std::get_if<ImplementationParameter::Local>(&parameter.source)) {
      if (local->name != parameter.name) {
        auto source = values.getAs<IntegerAttr>(local->name);
        if (!source || !source.getValue().isSignedIntN(64) || source.getInt() != value)
          return "parameter '" + parameter.name.str() + "' must equal '" + local->name.str() + "'";
      }
    } else if (auto fixed = std::get_if<ImplementationParameter::Constant>(&parameter.source)) {
      if (value != fixed->value)
        return "parameter '" + parameter.name.str() + "' must equal " + std::to_string(fixed->value);
    } else if (value != minimumParameter(std::get<ImplementationParameter::Minimum>(parameter.source), configuration)) {
      return "parameter '" + parameter.name.str() + "' disagrees with its declared shared-extent derivation";
    }
  }
  return implementation.parameterRelations ? implementation.parameterRelations(values, capabilities) : std::nullopt;
}

std::optional<std::string> bindParameters(const Implementation &implementation,
    Builder &builder, CapabilitiesAttr capabilities, const Configuration &configuration,
    ConfigurationAttr shared, DictionaryAttr &bound) {
  NamedAttrList values;
  for (const auto &parameter : implementation.parameters) {
    Attribute value;
    if (auto local = std::get_if<ImplementationParameter::Local>(&parameter.source)) {
      value = configuration.local.get(local->name);
      if (!value) return "missing local parameter '" + local->name.str() + "'";
    } else if (auto fixed = std::get_if<ImplementationParameter::Constant>(&parameter.source)) {
      value = builder.getI64IntegerAttr(fixed->value);
    } else {
      value = builder.getI64IntegerAttr(minimumParameter(
          std::get<ImplementationParameter::Minimum>(parameter.source), shared));
    }
    values.append(parameter.name, value);
  }
  bound = values.getDictionary(builder.getContext());
  return checkParameters(implementation, bound, capabilities, shared);
}
} // namespace

FailureOr<const ImplementationRegistry *>
lookupImplementationProvider(Operation *operation, StringRef provider) {
  if (provider.empty())
    return operation->emitError("CPU transformation requires an explicit provider option"), failure();
  auto dialect = operation->getContext()->getOrLoadDialect<IntentCPUDialect>();
  auto interface = dialect->getRegisteredInterface<ImplementationProviderInterface>();
  if (!interface)
    return operation->emitError("CPU implementation providers were not installed in this compiler context"), failure();
  auto registry = interface->lookup(provider);
  if (failed(registry))
    return operation->emitError("CPU implementation provider is unavailable: ") << provider, failure();
  return *registry;
}

std::optional<std::string> checkInputRequirement(
    Value source, Value element, const InputRequirement &requirement) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || requirement.panelSize <= 0 || requirement.alignment <= 0 ||
      requirement.windowAlignment <= 0 || requirement.panelSize % requirement.windowAlignment != 0 ||
      !llvm::isPowerOf2_64(requirement.alignment))
    return "implementation has an invalid input representation requirement";
  if (!requirement.elementType || !requirement.elementType.isIntOrIndexOrFloat())
    return "implementation input requires an explicit scalar representation type";
  if (requirement.storageScope == InputStorageScope::Invocation &&
      requirement.reuse != InputReuse::Consumers)
    return "invocation input storage requires consumer reuse";
  if (requirement.elementType != type.getElementType()) {
    if (element.use_empty() || !llvm::all_of(element.getUsers(), [&](Operation *user) {
          auto widen = dyn_cast<arith::ExtFOp>(user);
          return widen && widen.getType() == requirement.elementType;
        }))
      return "input representation must preserve the consumer's explicit floating extension";
  }
  int64_t bits = requirement.elementType.isIndex() ? 64 : requirement.elementType.getIntOrFloatBitWidth();
  if (requirement.panelAxis >= static_cast<unsigned>(type.getRank()) || requirement.alignment < (bits + 7) / 8)
    return "implementation input panel does not match its typed source";
  return std::nullopt;
}

std::optional<std::string> checkInputRequirements(
    linalg::GenericOp operation, ArrayRef<InputRequirement> requirements) {
  SmallVector<unsigned> operands;
  for (const auto &requirement : requirements) {
    if (requirement.operand >= operation.getInputs().size() ||
        llvm::is_contained(operands, requirement.operand))
      return "implementation has an invalid or repeated input representation operand";
    operands.push_back(requirement.operand);
    if (auto reason = checkInputRequirement(operation.getInputs()[requirement.operand],
            operation.getRegion().front().getArgument(requirement.operand), requirement))
      return "operand " + std::to_string(requirement.operand) + ": " + *reason;
  }
  return std::nullopt;
}

SmallVector<InputRequirement> Implementation::inputRequirements(
    linalg::GenericOp operation, ConfigurationAttr configuration, ImplementationAttr binding) const {
  return inputs ? inputs(operation, configuration, binding) : SmallVector<InputRequirement>{};
}

bool ContractionRequirements::acceptsInputLayout(unsigned operand, MemRefType type) const {
  if (operand >= unitInnerStride.size()) return false;
  if (!unitInnerStride[operand]) return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.getRank() && succeeded(type.getStridesAndOffset(strides, offset)) && strides.back() == 1;
}

namespace {
std::optional<std::string> checkInputLayouts(Operation *operation, const ContractionRequirements &requirements) {
  if (llvm::none_of(requirements.unitInnerStride, [](bool required) { return required; })) return std::nullopt;
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  if (!generic || generic.getInputs().size() != requirements.unitInnerStride.size())
    return "contraction input layout requirements need two shaped inputs";
  for (auto [operand, input] : llvm::enumerate(generic.getInputs())) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || !requirements.acceptsInputLayout(operand, type))
      return "operand " + std::to_string(operand) + " requires a proven unit inner stride";
  }
  return std::nullopt;
}

ConfigurationAttr sharedConfiguration(MLIRContext *context, const Configuration &configuration) {
  return ConfigurationAttr::get(context, configuration.taskGrain, configuration.tileM,
      configuration.tileN, configuration.tileK, configuration.regionSize);
}

std::optional<std::string> matchImplementation(
    const Implementation &implementation, Operation *operation,
    CapabilitiesAttr capabilities, const Configuration &configuration,
    ImplementationAttr &binding) {
  if (!implementation.matches(operation))
    return "implementation does not match this computation";
  if (auto reason = checkInputLayouts(operation, implementation.contraction)) return reason;
  Builder builder(operation->getContext());
  DictionaryAttr parameters;
  if (auto reason = bindParameters(implementation, builder, capabilities, configuration,
          sharedConfiguration(builder.getContext(), configuration), parameters)) return reason;
  if (auto reason = implementation.check(operation, capabilities, configuration)) return reason;
  auto selected = ImplementationAttr::get(builder.getContext(), builder.getStringAttr(implementation.name),
      parameters);
  if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    auto requirements = implementation.inputRequirements(
        generic, sharedConfiguration(builder.getContext(), configuration), selected);
    if (auto reason = checkInputRequirements(generic, requirements)) return reason;
  } else if (implementation.inputs) {
    return "input representation requirements need a structured linalg computation";
  }
  binding = selected;
  return std::nullopt;
}
} // namespace

void ImplementationRegistry::addProfile(StringRef name,
                                         ArrayRef<StringRef> localParameters) {
  profiles.push_back({name, llvm::to_vector(localParameters)});
}

std::optional<ArrayRef<StringRef>>
ImplementationRegistry::profileParameters(StringRef name) const {
  const ProfileSchema *selected = nullptr;
  for (const ProfileSchema &schema : profiles) {
    if (schema.name != name)
      continue;
    if (selected)
      return std::nullopt;
    selected = &schema;
  }
  return selected ? std::optional<ArrayRef<StringRef>>(selected->localParameters)
                  : std::nullopt;
}

bool ImplementationRegistry::hasImplementation(StringRef name) const {
  return llvm::count_if(implementations, [&](const Implementation &implementation) {
    return implementation.name == name;
  }) == 1;
}

FailureOr<SmallVector<StringRef>>
ImplementationRegistry::localParameters(ArrayAttr selection) const {
  if (!selection || selection.empty()) return failure();
  SmallVector<StringRef> names, parameters;
  for (Attribute attribute : selection) {
    auto name = dyn_cast<StringAttr>(attribute);
    if (!name || !hasImplementation(name.getValue()) ||
        llvm::is_contained(names, name.getValue())) return failure();
    names.push_back(name.getValue());
    const auto &implementation = *llvm::find_if(implementations,
        [&](const Implementation &candidate) {
          return candidate.name == name.getValue();
        });
    for (const auto &parameter : implementation.parameters)
      if (auto local = std::get_if<ImplementationParameter::Local>(&parameter.source))
        if (!llvm::is_contained(parameters, local->name)) parameters.push_back(local->name);
  }
  return parameters;
}

bool ImplementationRegistry::needsImplementation(Operation *operation) const {
  if (auto function = dyn_cast<func::FuncOp>(operation); function && function.isExternal())
    return false;
  return llvm::any_of(implementations, [&](const Implementation &implementation) {
    return implementation.operationKind(operation);
  });
}

FailureOr<const Implementation *> ImplementationRegistry::lookup(Operation *operation) const {
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function) function = operation->getParentOfType<func::FuncOp>();
  auto module = operation->getParentOfType<ModuleOp>();
  auto implementation = verifyBinding(binding,
      module ? module->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities") : CapabilitiesAttr(),
      function ? function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration") : ConfigurationAttr(), operation);
  if (failed(implementation)) return failure();
  if ((*implementation)->matches(operation) &&
      !checkInputLayouts(operation, (*implementation)->contraction))
    return *implementation;
  operation->emitError("CPU computation has lost its selected implementation binding");
  return failure();
}

LogicalResult ImplementationRegistry::materialize(Operation *operation) const {
  auto implementation = lookup(operation);
  if (failed(implementation)) return failure();
  if (!(*implementation)->materialize)
    return operation->emitError(
        "selected CPU implementation has no structured materialization");
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  return (*implementation)->materialize(operation, binding);
}

FailureOr<const Implementation *> ImplementationRegistry::verifyBinding(
    ImplementationAttr binding, CapabilitiesAttr capabilities,
    ConfigurationAttr configuration, Operation *diagnostic) const {
  if (!binding || !capabilities || !configuration)
    return diagnostic->emitError("CPU implementation validation requires a typed binding, capabilities and configuration"), failure();
  for (const auto &implementation : implementations) {
    if (implementation.name != binding.getName().getValue()) continue;
    if (auto reason = checkParameters(implementation, binding.getParameters(), capabilities, configuration))
      return diagnostic->emitError("invalid CPU implementation '") << implementation.name << "': " << *reason, failure();
    if (implementation.requiresMatrixI8I32 && !capabilities.getMatrixI8I32())
      return diagnostic->emitError("CPU implementation requires the matrix_i8_i32 capability"), failure();
    return &implementation;
  }
  return diagnostic->emitError("CPU binding names an implementation outside the selected provider: ") << binding.getName(), failure();
}

LogicalResult ImplementationRegistry::verifyBindings(ModuleOp module) const {
  auto capabilities = module->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isExternal() && function->hasAttr("cpu.external_runtime")) continue;
    auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
    auto summary = function->getAttrOfType<ArrayAttr>("intent_cpu.implementations");
    auto requiresMatrix = function->getAttrOfType<BoolAttr>("intent_cpu.requires_matrix_i8_i32");
    if (!capabilities || !configuration || !summary || summary.empty() || !requiresMatrix)
      return function.emitError("CPU provider requires complete retained implementation bindings");
    bool matrix = false;
    for (Attribute attribute : summary) {
      auto implementation = verifyBinding(dyn_cast<ImplementationAttr>(attribute), capabilities, configuration, function);
      if (failed(implementation)) return failure();
      matrix |= (*implementation)->requiresMatrixI8I32;
    }
    if (matrix != requiresMatrix.getValue())
      return function.emitError("CPU matrix requirement disagrees with retained implementation bindings");
    auto status = function.walk([&](Operation *operation) -> WalkResult {
      if (!operation->hasAttr("intent_cpu.implementation")) return WalkResult::advance();
      auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
      return succeeded(verifyBinding(binding, capabilities, configuration, operation))
          ? WalkResult::advance() : WalkResult::interrupt();
    });
    if (status.wasInterrupted()) return failure();
  }
  return success();
}

LogicalResult verifyImplementationBindings(ModuleOp module, StringRef provider) {
  auto registry = lookupImplementationProvider(module, provider);
  return failed(registry) ? failure() : (**registry).verifyBindings(module);
}

FailureOr<SmallVector<ImplementationAttr>> ImplementationRegistry::select(
    func::FuncOp function, CapabilitiesAttr capabilities,
    const Configuration &configuration,
    llvm::function_ref<void(Operation *, StringRef, StringRef)> rejected) const {
  if (!configuration.implementations || configuration.implementations.empty()) {
    rejected(function, {}, "configuration has no explicit implementation selection");
    return failure();
  }
  SmallVector<const Implementation *> selected;
  for (Attribute attribute : configuration.implementations) {
    auto name = dyn_cast<StringAttr>(attribute);
    auto found = name ? llvm::find_if(implementations, [&](const Implementation &implementation) {
      return implementation.name == name.getValue();
    }) : implementations.end();
    if (found == implementations.end() || llvm::is_contained(selected, &*found)) {
      rejected(function, {}, "configuration has an unknown or repeated implementation selection");
      return failure();
    }
    selected.push_back(&*found);
  }
  auto declaredParameters = localParameters(configuration.implementations);
  if (failed(declaredParameters)) {
    rejected(function, {}, "configuration has an invalid implementation selection");
    return failure();
  }
  for (NamedAttribute parameter : configuration.local)
    if (!llvm::is_contained(*declaredParameters, parameter.getName().getValue())) {
      rejected(function, {},
          "local parameter '" + parameter.getName().getValue().str() +
          "' is not declared by a selected implementation");
      return failure();
    }
  SmallVector<ImplementationAttr> bindings;
  bool invalid = false;
  function.walk([&](Operation *operation) {
    if (!needsImplementation(operation)) return;
    // Applicability resolves the declared computation, before legality checks.
    // A rejected selection must not silently choose another implementation.
    const Implementation *implementation = nullptr;
    for (const Implementation *choice : selected) {
      if (!choice->matches(operation)) continue;
      if (implementation) {
        rejected(operation, choice->name,
                 "configuration selects multiple implementations for this computation");
        invalid = true;
        return;
      }
      implementation = choice;
    }
    if (!implementation) {
      rejected(operation, {}, "configuration selects no implementation for this computation");
      invalid = true;
      return;
    }
    ImplementationAttr binding;
    if (auto reason = matchImplementation(*implementation, operation, capabilities, configuration, binding)) {
      rejected(operation, implementation->name, *reason);
      invalid = true;
      return;
    }
    bindings.push_back(binding);
  });
  if (invalid) return failure();
  // The executable bindings contain only actual typed consumers. Parameters of
  // absent selected members do not create variants or enter the current IR.
  return bindings;
}

LogicalResult ImplementationRegistry::bind(func::FuncOp function, CapabilitiesAttr capabilities,
                                            const Configuration &configuration,
                                            ArrayRef<ImplementationAttr> bindings) const {
  Builder builder(function.getContext());
  unsigned ordinal = 0;
  bool matrix = false;
  SmallVector<std::pair<Operation *, ImplementationAttr>> selected;
  SmallVector<Attribute> distinctBindings;
  auto status = function.walk([&](Operation *operation) -> WalkResult {
    if (!needsImplementation(operation)) return WalkResult::advance();
    if (ordinal == bindings.size()) {
      operation->emitError("CPU candidate is missing an implementation binding");
      return WalkResult::interrupt();
    }
    auto binding = bindings[ordinal++];
    auto implementation = llvm::find_if(implementations, [&](const Implementation &candidate) {
      return candidate.name == binding.getName().getValue();
    });
    if (implementation == implementations.end()) {
      operation->emitError("CPU candidate names an unregistered implementation: ") << binding.getName();
      return WalkResult::interrupt();
    }
    ImplementationAttr selectedBinding;
    if (auto reason = matchImplementation(*implementation, operation, capabilities, configuration, selectedBinding)) {
      operation->emitError("CPU candidate implementation '") << implementation->name << "' rejected: " << *reason;
      return WalkResult::interrupt();
    }
    if (selectedBinding != binding) {
      operation->emitError("CPU candidate parameters do not match the selected implementation");
      return WalkResult::interrupt();
    }
    selected.emplace_back(operation, binding);
    if (!llvm::is_contained(distinctBindings, Attribute(binding)))
      distinctBindings.push_back(binding);
    matrix |= implementation->requiresMatrixI8I32;
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) return failure();
  if (ordinal != bindings.size())
    return function.emitError("CPU candidate implementation bindings do not match its current computations");
  for (auto [operation, binding] : selected)
    operation->setAttr("intent_cpu.implementation", binding);
  function->setAttr("intent_cpu.configuration", sharedConfiguration(function.getContext(), configuration));
  // This summary records selected implementations for artifact inspection.
  // Execution consumers use bindings on the current operation or lexical owner.
  function->setAttr("intent_cpu.implementations", builder.getArrayAttr(distinctBindings));
  function->setAttr("intent_cpu.requires_matrix_i8_i32", builder.getBoolAttr(matrix));
  return success();
}

int64_t implementationParameter(ImplementationAttr binding, StringRef name) {
  return cast<IntegerAttr>(binding.getParameters().get(name)).getInt();
}

}
