#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

using namespace mlir;

namespace intent::gpu {

ArgumentBindingAttr getArgumentBinding(BlockArgument argument) {
  if (!argument) return {};
  auto function = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || function.isExternal() || argument.getOwner() != &function.front())
    return {};
  return function.getArgAttrOfType<ArgumentBindingAttr>(argument.getArgNumber(), argumentBindingAttr);
}

ArgumentRefAttr getArgumentReference(Value value) {
  auto argument = dyn_cast_or_null<BlockArgument>(value);
  auto binding = argument ? getArgumentBinding(argument) : ArgumentBindingAttr{};
  return binding ? binding.getReference() : ArgumentRefAttr{};
}

BlockArgument resolveArgument(func::FuncOp kernel, ArgumentRefAttr reference) {
  if (!kernel || kernel.isExternal() || !reference) return {};
  for (BlockArgument argument : kernel.getArguments())
    if (auto binding = getArgumentBinding(argument);
        binding && binding.getReference() == reference)
      return argument;
  return {};
}

PhysicalExprAttr queryArgumentExpression(BlockArgument argument) {
  auto binding = getArgumentBinding(argument);
  if (!binding || !isa<IndexType, IntegerType>(argument.getType())) return {};
  bool dimension = binding.getKind() == ArgumentKind::Dimension;
  if (!dimension && binding.getKind() != ArgumentKind::Public &&
      binding.getKind() != ArgumentKind::Stride) return {};
  return PhysicalExprAttr::get(argument.getContext(),
      dimension ? PhysicalExprKind::Dimension : PhysicalExprKind::ScalarABI,
      dimension ? binding.getDimension().getInt() : 0, binding.getReference(),
      ArrayAttr::get(argument.getContext(), {}));
}

BlockArgument resolveDimension(func::FuncOp kernel, int64_t identity) {
  for (BlockArgument argument : kernel.getArguments())
    if (auto binding = getArgumentBinding(argument);
        binding && binding.getKind() == ArgumentKind::Dimension &&
        binding.getDimension().getInt() == identity)
      return argument;
  return {};
}

intent::PublicParameterAttr publicParameter(Value value) {
  auto argument = dyn_cast_or_null<BlockArgument>(value);
  auto binding = argument ? getArgumentBinding(argument) : ArgumentBindingAttr{};
  if (!binding || binding.getKind() != ArgumentKind::Public) return {};
  auto kernel = cast<func::FuncOp>(argument.getOwner()->getParentOp());
  auto interface = intent::getPublicInterface(kernel);
  int64_t ordinal = binding.getPublicOrdinal().getInt();
  if (!interface || ordinal < 0 || ordinal >= static_cast<int64_t>(interface.getArguments().size()))
    return {};
  return dyn_cast<intent::PublicParameterAttr>(interface.getArguments()[ordinal]);
}

intent::ViewType getPublicView(Value value) {
  auto parameter = publicParameter(value);
  return parameter ? dyn_cast<intent::ViewType>(parameter.getType()) : intent::ViewType{};
}

bool isInvocationWorkspace(Value value) {
  if (value)
    if (auto buffer = dyn_cast<BufferType>(value.getType()))
      return buffer.isInvocationWorkspace();
  auto argument = dyn_cast_or_null<BlockArgument>(value);
  auto binding = argument ? getArgumentBinding(argument) : ArgumentBindingAttr{};
  return binding && binding.getKind() == ArgumentKind::Workspace;
}

namespace {

std::optional<int64_t> workspaceStride(func::FuncOp kernel, ViewType view,
                                        unsigned axis) {
  auto resolve = [&](PhysicalExprAttr expression) -> std::optional<int64_t> {
    if (expression.getKind() != PhysicalExprKind::Dimension) return std::nullopt;
    auto argument = resolveArgument(kernel, expression.getArgumentReference());
    auto binding = argument ? getArgumentBinding(argument) : ArgumentBindingAttr{};
    if (!binding || binding.getKind() != ArgumentKind::Dimension ||
        binding.getDimension().getInt() != expression.getValue()) return std::nullopt;
    auto source = resolveArgument(kernel, binding.getSource());
    auto logical = getPublicView(source);
    if (!logical) return std::nullopt;
    auto tensor = intent::publicViewTensor(logical);
    unsigned sourceAxis = binding.getAxis().getInt();
    return tensor.isDynamicDim(sourceAxis)
        ? std::nullopt : std::optional<int64_t>(tensor.getDimSize(sourceAxis));
  };
  auto context = kernel.getContext();
  auto constant = [&](int64_t value) {
    return PhysicalExprAttr::get(context, PhysicalExprKind::Constant, value,
        StringAttr::get(context), ArrayAttr::get(context, {}));
  };
  PhysicalExprAttr stride = constant(1);
  // torch.empty uses a contiguous allocation: zero-sized trailing dimensions
  // contribute one to the stride, as do unit dimensions.
  for (unsigned following = axis + 1; following < view.getRank(); ++following) {
    auto extent = evaluatePhysicalExpression(
        cast<PhysicalExprAttr>(view.getLayout().getExtents()[following]), resolve);
    if (!extent || *extent < 0) return std::nullopt;
    stride = PhysicalExprAttr::get(context, PhysicalExprKind::Multiply, 0,
        StringAttr::get(context), ArrayAttr::get(context,
            {stride, constant(std::max<int64_t>(*extent, 1))}));
  }
  return constantPhysicalExpression(stride);
}

class HostDependencies {
public:
  HostDependencies(func::FuncOp kernel,
                   const llvm::DenseMap<ArgumentRefAttr, BlockArgument> &arguments)
      : kernel(kernel), arguments(arguments) {}

  LogicalResult verify() {
    // Public inputs and allocated outputs are roots. All remaining bindings are
    // dependencies of host allocation or launch, independently of ABI slot order.
    for (BlockArgument argument : kernel.getArguments())
      if (failed(visit(getArgumentReference(argument)))) return failure();
    auto declarations = kernel->getAttrOfType<ArrayAttr>(parametersAttr);
    if (!declarations) return kernel.emitError("host bindings require parameter declarations");
    for (Attribute attribute : declarations) {
      auto parameter = dyn_cast<ParameterAttr>(attribute);
      if (!parameter || !parameter.getBinding())
        return kernel.emitError("host binding references an invalid parameter declaration");
      // Shared construction may still be preparing a deferred declaration. A
      // host consumer cannot use it until its coverage bound is present.
      if (parameter.isDeferred() && parameter.getBinding().getCoverageBound() &&
          failed(visitDeferred(parameter.getReference()))) return failure();
    }
    return success();
  }

private:
  LogicalResult visitDeferred(ParameterRefAttr reference) {
    auto found = deferredState.find(reference);
    if (found != deferredState.end()) {
      if (found->second == Complete) return success();
      return kernel.emitError("cycle in deferred parameter bindings: ") << reference;
    }
    auto parameter = lookupParameterDeclaration(kernel, reference);
    if (!parameter || !parameter.isExtent())
      return kernel.emitError("deferred extent references an unavailable extent parameter: ") << reference;
    if (!parameter.isDeferred()) return success();
    auto bound = parameter.getBinding().getCoverageBound();
    if (!bound) return kernel.emitError("deferred extent has no bound expression: ") << reference;
    deferredState[reference] = Visiting;
    std::function<LogicalResult(PhysicalExprAttr)> check = [&](PhysicalExprAttr value) {
      if (value.getKind() == PhysicalExprKind::Parameter)
        return visitDeferred(value.getParameterReference());
      if (value.getKind() == PhysicalExprKind::Dimension ||
          value.getKind() == PhysicalExprKind::ScalarABI)
        return visit(value.getArgumentReference());
      for (Attribute operand : value.getOperands())
        if (failed(check(cast<PhysicalExprAttr>(operand)))) return failure();
      return success();
    };
    if (failed(check(bound))) return failure();
    deferredState[reference] = Complete;
    return success();
  }

  LogicalResult expression(PhysicalExprAttr value, bool allowDeferred) {
    switch (value.getKind()) {
    case PhysicalExprKind::Constant: return success();
    case PhysicalExprKind::Dimension:
    case PhysicalExprKind::ScalarABI:
      return visit(value.getArgumentReference());
    case PhysicalExprKind::Parameter:
      if (!allowDeferred)
        return kernel.emitError("coverage bound cannot depend on configuration parameter ")
            << value.getParameterReference();
      return visit(value.getParameterReference());
    default:
      for (Attribute operand : value.getOperands())
        if (failed(expression(cast<PhysicalExprAttr>(operand), allowDeferred)))
          return failure();
      return success();
    }
  }

  LogicalResult visit(Attribute node) {
    auto found = state.find(node);
    if (found != state.end()) {
      if (found->second == Complete) return success();
      auto error = kernel.emitError("cycle in GPU host binding dependencies: ");
      for (Attribute ancestor : stack) error << ancestor << " -> ";
      error << node;
      return failure();
    }
    state[node] = Visiting;
    stack.push_back(node);
    LogicalResult result = dependencies(node);
    stack.pop_back();
    if (failed(result)) return failure();
    state[node] = Complete;
    return success();
  }

  LogicalResult dependencies(Attribute node) {
    if (auto reference = dyn_cast<ArgumentRefAttr>(node)) {
      BlockArgument argument = arguments.lookup(reference);
      if (!argument)
        return kernel.emitError("host expression references an unavailable argument ") << reference;
      auto binding = getArgumentBinding(argument);
      switch (binding.getKind()) {
      case ArgumentKind::Public: return success();
      case ArgumentKind::Dimension:
      case ArgumentKind::Stride: return visit(binding.getSource());
      case ArgumentKind::Workspace: {
        auto view = cast<ViewType>(argument.getType());
        ArrayAttr shape = view.getLayout().getExtents();
        for (Attribute extent : shape)
          if (failed(expression(cast<PhysicalExprAttr>(extent), true))) return failure();
        return success();
      }
      }
    }
    auto reference = dyn_cast<ParameterRefAttr>(node);
    auto parameter = reference ? lookupParameterDeclaration(kernel, reference) : ParameterAttr{};
    if (!parameter)
      return kernel.emitError("host expression references an unavailable configuration parameter ") << node;
    if (!parameter.isDeferred())
      return kernel.emitError("workspace allocation cannot depend on a candidate-row parameter before candidate selection: ")
          << reference;
    auto bound = parameter.getBinding().getCoverageBound();
    if (!bound)
      return kernel.emitError("host expression requires a deferred parameter with a coverage bound: ") << reference;
    return expression(bound, false);
  }

  enum State { Visiting, Complete };
  func::FuncOp kernel;
  const llvm::DenseMap<ArgumentRefAttr, BlockArgument> &arguments;
  llvm::DenseMap<Attribute, State> state;
  llvm::DenseMap<ParameterRefAttr, State> deferredState;
  SmallVector<Attribute> stack;
};

} // namespace

LogicalResult verifyProgramInterface(func::FuncOp kernel) {
  auto interface = intent::getPublicInterface(kernel);
  if (failed(intent::verifyPublicInterface(kernel, interface))) return failure();
  if (kernel.isExternal()) return kernel.emitError("physical GPU program requires an entry block");
  llvm::DenseMap<ArgumentRefAttr, BlockArgument> references;
  llvm::DenseSet<int64_t> publicOrdinals, dimensions;
  llvm::DenseSet<std::pair<Attribute, int64_t>> strides;
  for (BlockArgument argument : kernel.getArguments()) {
    auto binding = getArgumentBinding(argument);
    if (!binding) return kernel.emitError("every physical argument requires a typed binding");
    if (failed(ArgumentBindingAttr::verify([&] { return kernel.emitError(); },
        binding.getReference(), binding.getKind(), binding.getPublicOrdinal(),
        binding.getSource(), binding.getAxis(), binding.getDimension()))) return failure();
    if (!references.try_emplace(binding.getReference(), argument).second)
      return kernel.emitError("physical argument identities must be unique");
  }
  for (BlockArgument argument : kernel.getArguments()) {
    auto binding = getArgumentBinding(argument);
    Type type = argument.getType();
    if (binding.getKind() == ArgumentKind::Public) {
      int64_t ordinal = binding.getPublicOrdinal().getInt();
      if (ordinal >= static_cast<int64_t>(interface.getArguments().size()) ||
          !publicOrdinals.insert(ordinal).second)
        return kernel.emitError("public argument binding must uniquely reference the public interface");
      auto parameter = cast<intent::PublicParameterAttr>(interface.getArguments()[ordinal]);
      auto logical = dyn_cast<intent::ViewType>(parameter.getType());
      if (!logical) {
        if (type != parameter.getType()) return kernel.emitError("physical scalar disagrees with its public type");
        continue;
      }
      auto physical = dyn_cast<ViewType>(type);
      auto tensor = intent::publicViewTensor(logical);
      if (!physical || physical.getElementType() != tensor.getElementType() ||
          physical.getRank() != tensor.getRank() || physical.getAccess() != logical.getAccess() ||
          physical.getLayout().getDimensionIds() != intent::publicViewDimensions(logical))
        return kernel.emitError("physical view disagrees with its public type and dimensions");
      for (auto [axis, attribute] : llvm::enumerate(physical.getLayout().getExtents())) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        if (tensor.isDynamicDim(axis)) {
          if (extent.getKind() != PhysicalExprKind::Dimension ||
              extent.getValue() != intent::publicViewDimensions(logical)[axis])
            return kernel.emitError("dynamic public extent requires its declared dimension binding");
        } else if (extent.getKind() != PhysicalExprKind::Constant || extent.getValue() != tensor.getDimSize(axis))
          return kernel.emitError("static public extent disagrees with its physical view");
      }
    } else if (binding.getKind() == ArgumentKind::Workspace) {
      auto view = dyn_cast<ViewType>(type);
      if (!view || view.getAccess() != 2)
        return kernel.emitError("workspace ABI binding requires a lowered physical view");
      auto shape = view.getLayout().getExtents();
      for (Attribute attribute : shape)
        if (auto extent = constantPhysicalExpression(cast<PhysicalExprAttr>(attribute));
            extent && *extent < 0)
          return kernel.emitError("workspace allocation cannot have a negative extent");
    } else {
      if (!type.isIndex()) return kernel.emitError("dimension and stride arguments must have index type");
      BlockArgument source = references.lookup(binding.getSource());
      auto view = source ? dyn_cast<ViewType>(source.getType()) : ViewType{};
      int64_t axis = binding.getAxis().getInt();
      if (!view || axis >= view.getRank())
        return kernel.emitError("metadata source must reference an axis of a current physical view");
      if (binding.getKind() == ArgumentKind::Dimension) {
        int64_t dimension = binding.getDimension().getInt();
        if (getArgumentBinding(source).getKind() != ArgumentKind::Public)
          return kernel.emitError("logical dimension metadata must be owned by a public view");
        if (view.getLayout().getDimensionIds()[axis] != dimension || !dimensions.insert(dimension).second)
          return kernel.emitError("dimension binding must uniquely identify the declared source dimension");
      } else if (!strides.insert({binding.getSource(), axis}).second)
        return kernel.emitError("stride binding must uniquely identify one resource axis");
    }
  }
  if (publicOrdinals.size() != interface.getArguments().size())
    return kernel.emitError("physical program does not bind every public parameter");

  for (BlockArgument argument : kernel.getArguments()) {
    auto view = dyn_cast<ViewType>(argument.getType());
    if (!view) continue;
    for (auto [axis, attribute] : llvm::enumerate(view.getLayout().getStrides())) {
      auto expression = cast<PhysicalExprAttr>(attribute);
      if (expression.getKind() == PhysicalExprKind::Constant) {
        std::optional<int64_t> required;
        if (auto logical = getPublicView(argument)) {
          auto constraints = logical.getConstraints();
          if (constraints.getHasStrides())
            if (auto fixed = dyn_cast<IntegerAttr>(constraints.getStrides()[axis]))
              required = fixed.getInt();
        } else if (isInvocationWorkspace(argument)) {
          required = workspaceStride(kernel, view, axis);
        }
        if (!required || *required != expression.getValue())
          return kernel.emitError("constant physical stride is not established by the public contract or workspace allocation at argument ")
              << getArgumentReference(argument) << " axis " << axis;
        continue;
      }
      auto value = references.lookup(expression.getArgumentReference());
      auto binding = value ? getArgumentBinding(value) : ArgumentBindingAttr{};
      if (expression.getKind() != PhysicalExprKind::ScalarABI || !binding ||
          binding.getKind() != ArgumentKind::Stride ||
          binding.getSource() != getArgumentReference(argument) ||
          binding.getAxis().getInt() != static_cast<int64_t>(axis))
        return kernel.emitError("physical stride expression must reference metadata for the same resource axis");
    }
  }

  bool valid = true;
  AttrTypeWalker walker;
  walker.addWalk([&](PhysicalExprAttr expression) {
    if (!valid) return;
    auto kind = expression.getKind();
    if (kind != PhysicalExprKind::Dimension && kind != PhysicalExprKind::ScalarABI) return;
    auto argument = references.lookup(expression.getArgumentReference());
    auto expected = argument ? queryArgumentExpression(argument) : PhysicalExprAttr{};
    if (!expected || expression != expected) {
      kernel.emitError("physical expression does not match its current argument binding: ") << expression;
      valid = false;
    }
  });
  kernel.walk([&](Operation *operation) {
    walker.walk(operation->getAttrDictionary());
    for (Type type : operation->getResultTypes()) walker.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) walker.walk(argument.getType());
  });
  if (!valid) return failure();
  return HostDependencies(kernel, references).verify();
}

} // namespace intent::gpu
