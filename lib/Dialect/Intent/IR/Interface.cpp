#include "Intent/Dialect/Intent/IR/Interface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent {
std::string scalarABIName(Type type) {
  if (type.isIndex()) return "index";
  if (auto integer = mlir::dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1) return "bool";
    return (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
  }
  if (type.isF16()) return "f16";
  if (type.isBF16()) return "bf16";
  if (type.isF32()) return "f32";
  if (type.isF64()) return "f64";
  if (isa<Float8E4M3FNType>(type)) return "f8e4m3fn";
  assert(isa<Float8E5M2Type>(type) && "verified public scalar type");
  return "f8e5m2";
}

RankedTensorType publicViewTensor(ViewType view) {
  return cast<RankedTensorType>(view.getTensor());
}

DenseI64ArrayAttr publicViewDimensions(ViewType view) {
  return cast<TensorShapeAttr>(publicViewTensor(view).getEncoding()).getDimensions();
}

LogicalResult PublicParameterAttr::verify(function_ref<InFlightDiagnostic()> error,
                                          StringAttr name, Type type) {
  if (!name || name.empty() || !type)
    return error() << "public parameter requires a name and canonical type";
  auto view = mlir::dyn_cast<ViewType>(type);
  if (!view)
    return isCanonicalScalarType(type) ? success() : error() << "public scalar has no canonical ABI type";
  auto tensor = mlir::dyn_cast<RankedTensorType>(view.getTensor());
  if (!tensor || !isCanonicalScalarType(tensor.getElementType()))
    return error() << "public view requires an addressable canonical storage element type";
  auto shape = mlir::dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  if (!shape || shape.getDimensions().size() != tensor.getRank())
    return error() << "public view requires complete logical dimension identities";
  if (failed(verifyCanonicalType(error, type))) return failure();
  auto constraints = view.getConstraints();
  if (!constraints || (constraints.getHasStrides() &&
      constraints.getStrides().size() != static_cast<size_t>(tensor.getRank())))
    return error() << "public view stride constraints must cover its rank";
  return success();
}

LogicalResult InterfaceAttr::verify(function_ref<InFlightDiagnostic()> error,
                                    ArrayAttr arguments) {
  if (!arguments) return error() << "public interface requires a parameter sequence";
  llvm::DenseSet<StringAttr> names;
  llvm::DenseMap<int64_t, int64_t> staticDimensions;
  for (Attribute attribute : arguments) {
    auto parameter = mlir::dyn_cast<PublicParameterAttr>(attribute);
    if (!parameter) return error() << "public interface contains an untyped parameter";
    if (failed(PublicParameterAttr::verify(error, parameter.getName(), parameter.getType())))
      return failure();
    if (!names.insert(parameter.getName()).second)
      return error() << "public parameter names must be unique";
    if (auto view = mlir::dyn_cast<ViewType>(parameter.getType())) {
      auto tensor = publicViewTensor(view);
      for (auto [extent, identity] : llvm::zip(tensor.getShape(), publicViewDimensions(view).asArrayRef())) {
        if (identity == 0 || ShapedType::isDynamic(extent)) continue;
        auto [entry, inserted] = staticDimensions.try_emplace(identity, extent);
        if (!inserted && entry->second != extent)
          return error() << "one logical dimension has conflicting static public extents";
      }
    }
  }
  return success();
}

InterfaceAttr getPublicInterface(func::FuncOp function) {
  return function->getAttrOfType<InterfaceAttr>(interfaceAttr);
}

ParameterAttr getSourceParameter(BlockArgument argument) {
  if (!argument) return {};
  auto function = mlir::dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.front()) return {};
  return function.getArgAttrOfType<ParameterAttr>(argument.getArgNumber(), sourceParameterAttr);
}

LogicalResult verifySourceInterface(func::FuncOp function) {
  auto kind = function->getAttrOfType<FunctionKindAttr>("intent.kind");
  if (!kind || function.isExternal() || !llvm::hasSingleElement(function.getBody()))
    return function.emitOpError("source interface requires an Intent function with one entry block");
  if (kind.getKind() == 0 && function.getNumResults())
    return function.emitOpError("kernel cannot return device values to the host");
  llvm::DenseSet<StringAttr> names;
  llvm::DenseSet<int64_t> origins;
  for (BlockArgument argument : function.getArguments()) {
    auto parameter = getSourceParameter(argument);
    if (!parameter)
      return function.emitOpError("argument requires a typed source parameter declaration");
    if (failed(ParameterAttr::verify([&] { return function.emitOpError(); },
                                    parameter.getName(), parameter.getOriginId()))) return failure();
    if (!names.insert(parameter.getName()).second || !origins.insert(parameter.getOriginId()).second)
      return function.emitOpError("source parameter names and provenance IDs must be unique");
    Type type = argument.getType();
    if (failed(verifyCanonicalType(function, type))) return failure();
    if (auto view = mlir::dyn_cast<ViewType>(type)) {
      auto constraints = view.getConstraints();
      if (constraints.getHasStrides() && constraints.getStrides().size() !=
          static_cast<size_t>(publicViewTensor(view).getRank()))
        return function.emitOpError("view stride-constraint rank must match its logical rank");
    }
  }
  for (Type type : function.getResultTypes())
    if (failed(verifyCanonicalType(function, type))) return failure();
  return success();
}

ViewType getPublicView(InterfaceAttr interface, unsigned publicOrdinal) {
  auto parameter = cast<PublicParameterAttr>(interface.getArguments()[publicOrdinal]);
  return mlir::dyn_cast<ViewType>(parameter.getType());
}

LogicalResult verifyPublicInterface(Operation *owner, InterfaceAttr interface) {
  if (!interface) return owner->emitOpError("requires a typed public interface");
  return InterfaceAttr::verify([&] { return owner->emitOpError(); }, interface.getArguments());
}

FailureOr<InterfaceAttr> buildPublicInterface(func::FuncOp kernel) {
  SmallVector<Attribute> runtime;
  for (BlockArgument argument : kernel.getArguments()) {
    auto parameter = getSourceParameter(argument);
    if (!parameter) return kernel.emitError("canonical argument requires its source parameter declaration"), failure();
    if (isa<ConstexprType>(argument.getType())) {
      if (!argument.use_empty())
        return kernel.emitError("public ABI requires fully specialized constexpr parameters"), failure();
      continue;
    }
    auto publicParameter = PublicParameterAttr::getChecked(
        [&] { return kernel.emitError(); }, kernel.getContext(), parameter.getName(), argument.getType());
    if (!publicParameter) return failure();
    runtime.push_back(publicParameter);
  }
  auto interface = InterfaceAttr::getChecked([&] { return kernel.emitError(); },
      kernel.getContext(), ArrayAttr::get(kernel.getContext(), runtime));
  return interface ? FailureOr<InterfaceAttr>(interface) : FailureOr<InterfaceAttr>(failure());
}

FailureOr<llvm::json::Object> serializePublicInterface(Operation *owner,
                                                      InterfaceAttr interface) {
  if (failed(verifyPublicInterface(owner, interface))) return failure();
  llvm::json::Array parameters;
  for (Attribute attribute : interface.getArguments()) {
    auto parameter = cast<PublicParameterAttr>(attribute);
    auto view = mlir::dyn_cast<ViewType>(parameter.getType());
    if (!view) {
      parameters.push_back(llvm::json::Object{{"kind", "scalar"},
          {"name", parameter.getName().getValue()}, {"dtype", scalarABIName(parameter.getType())}});
      continue;
    }
    auto tensor = publicViewTensor(view);
    auto constraints = view.getConstraints();
    llvm::json::Array shape, dimensions, strides;
    for (int64_t extent : tensor.getShape())
      shape.push_back(ShapedType::isDynamic(extent) ? int64_t{-1} : extent);
    for (int64_t identity : publicViewDimensions(view).asArrayRef()) dimensions.push_back(identity);
    for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
      Attribute stride = constraints.getHasStrides() ? constraints.getStrides()[axis] : Attribute();
      if (auto fixed = mlir::dyn_cast_or_null<IntegerAttr>(stride)) strides.push_back(fixed.getInt());
      else if (auto symbol = mlir::dyn_cast_or_null<StringAttr>(stride))
        strides.push_back(llvm::json::Object{{"kind", "stride_symbol"}, {"symbol", symbol.getValue()}});
      else if (!stride || isa<UnitAttr>(stride)) strides.push_back(nullptr);
      else return owner->emitOpError("public view contains an unsupported stride constraint"), failure();
    }
    parameters.push_back(llvm::json::Object{{"kind", "view"}, {"name", parameter.getName().getValue()},
        {"dtype", scalarABIName(tensor.getElementType())}, {"shape", std::move(shape)},
        {"dimensions", std::move(dimensions)}, {"strides", std::move(strides)},
        {"access", view.getAccess()}, {"alias", constraints.getAlias().getValue()},
        {"noalias", constraints.getNoalias()}});
  }
  return llvm::json::Object{{"parameters", std::move(parameters)}};
}

} // namespace intent
