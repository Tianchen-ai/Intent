#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent;

#include "Intent/Dialect/Intent/IR/IntentAttrsEnums.cpp.inc"

#define GET_DIALECT_DEF
#include "Intent/Dialect/Intent/IR/IntentDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Intent/Dialect/Intent/IR/IntentTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/Intent/IR/IntentAttrs.cpp.inc"

LogicalResult IntentDialect::verifyOperationAttribute(Operation *operation, NamedAttribute attribute) {
  if (attribute.getName() == compileOptionsAttr) {
    if (!isa<ModuleOp>(operation) || !isa<CompileOptionsAttr>(attribute.getValue()))
      return operation->emitOpError("intent.compile_options requires a typed module policy");
    if (auto parent = operation->getParentOfType<ModuleOp>())
      if (Attribute inherited = parent->getAttr(compileOptionsAttr);
          inherited && inherited != attribute.getValue())
        return operation->emitOpError("nested program modules must preserve their parent's compile options");
    return success();
  }
  if (attribute.getName() != "intent.kind") return success();
  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function || !isa<FunctionKindAttr>(attribute.getValue()))
    return operation->emitOpError("intent.kind requires a typed source function");
  return verifySourceInterface(function);
}

LogicalResult IntentDialect::verifyRegionArgAttribute(
    Operation *operation, unsigned regionIndex, unsigned argumentIndex, NamedAttribute attribute) {
  if (attribute.getName() != sourceParameterAttr) return success();
  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function || regionIndex != 0 || argumentIndex >= function.getNumArguments() ||
      !function->getAttrOfType<FunctionKindAttr>("intent.kind") ||
      !isa<ParameterAttr>(attribute.getValue()))
    return operation->emitOpError("intent.parameter must declare a source function argument");
  return success();
}

void IntentDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Dialect/Intent/IR/IntentAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Intent/Dialect/Intent/IR/IntentTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/Intent/IR/IntentOps.cpp.inc"
      >();
}

LogicalResult LogicalIndexType::verify(
    function_ref<InFlightDiagnostic()> emitError, uint64_t sourceId,
    uint32_t axis) {
  (void)sourceId;
  (void)axis;
  return success();
}

LogicalResult DomainType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 uint32_t flavor, uint32_t rank,
                                 uint64_t originId) {
  (void)originId;
  if (flavor > 3)
    return emitError() << "domain flavor is outside the canonical enumeration";
  if (rank == 0)
    return emitError() << "domain rank must be positive";
  return success();
}

LogicalResult RegionType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 uint64_t sourceId, uint32_t rank,
                                 uint64_t originId) {
  (void)sourceId;
  (void)originId;
  if (rank == 0)
    return emitError() << "subregion rank must be positive";
  return success();
}

LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type tensor, uint64_t originId) {
  (void)originId;
  if (!llvm::isa<RankedTensorType>(tensor))
    return emitError() << "logical buffer storage schema must be a ranked tensor";
  return success();
}

LogicalResult intent::TupleType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr componentTypes) {
  if (!componentTypes)
    return emitError() << "tuple requires a component-type array";
  for (Attribute componentAttribute : componentTypes) {
    auto component = mlir::dyn_cast<TypeAttr>(componentAttribute);
    if (!component)
      return emitError() << "tuple components must be TypeAttr entries";
    if (mlir::isa<ViewType, BufferType, ConstexprType, EnumType, DomainType,
                  RegionType>(
            component.getValue()))
      return emitError()
             << "tuple components must be scalar/tensor/tuple/record SSA values";
  }
  return success();
}

LogicalResult RecordType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayAttr fieldNames,
                                 ArrayAttr fieldTypes) {
  if (!fieldNames || !fieldTypes || fieldNames.empty() ||
      fieldNames.size() != fieldTypes.size())
    return emitError()
           << "record requires equally sized, non-empty field-name/type arrays";
  llvm::StringSet<> names;
  for (auto [nameAttribute, typeAttribute] :
       llvm::zip(fieldNames, fieldTypes)) {
    auto name = llvm::dyn_cast<StringAttr>(nameAttribute);
    auto type = llvm::dyn_cast<TypeAttr>(typeAttribute);
    if (!name || name.getValue().empty() || !type)
      return emitError()
             << "record fields require non-empty names and TypeAttr entries";
    if (!names.insert(name.getValue()).second)
      return emitError() << "record field names must be unique";
    if (llvm::isa<ViewType, BufferType, ConstexprType, EnumType, DomainType,
                  RegionType>(type.getValue()))
      return emitError()
             << "record fields must be scalar/tensor/tuple/record SSA values";
  }
  return success();
}

LogicalResult ConstexprType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type valueType) {
  if (!llvm::isa<IntegerType, IndexType, FloatType, EnumType>(valueType))
    return emitError()
           << "constexpr payload must be a scalar or closed Intent enum";
  return success();
}

LogicalResult EnumType::verify(function_ref<InFlightDiagnostic()> emitError,
                               StringAttr name, ArrayAttr memberNames,
                               DenseI64ArrayAttr memberValues) {
  if (!name || name.getValue().empty() || !memberNames ||
      memberNames.empty() || !memberValues ||
      memberNames.size() != static_cast<size_t>(memberValues.size()))
    return emitError()
           << "enum requires a name and equally sized non-empty members";
  llvm::StringSet<> names;
  llvm::DenseSet<int64_t> values;
  for (auto [memberName, memberValue] :
       llvm::zip(memberNames, memberValues.asArrayRef())) {
    auto string = llvm::dyn_cast<StringAttr>(memberName);
    if (!string || string.getValue().empty() ||
        !names.insert(string.getValue()).second ||
        !values.insert(memberValue).second)
      return emitError() << "enum member names and values must be unique";
  }
  return success();
}

LogicalResult ViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type tensor, uint32_t access,
                               ViewConstraintsAttr constraints) {
  if (!llvm::isa<RankedTensorType>(tensor))
    return emitError() << "external view schema must be a ranked tensor";
  if (access > 2)
    return emitError() << "external view access must be in, out, or inout";
  if (!constraints)
    return emitError() << "external view requires typed stride/alias constraints";
  return success();
}

LogicalResult TensorShapeAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr dimensions) {
  if (!dimensions)
    return emitError() << "tensor shape encoding requires dimension identities";
  for (int64_t identity : dimensions.asArrayRef())
    if (identity <= 0)
      return emitError() << "tensor dimension identity must be positive";
  return success();
}

LogicalResult ShapeExprAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t kind,
    int64_t dimension, int64_t payload) {
  if (kind > 2)
    return emitError() << "shape expression kind is outside the canonical enum";
  if (kind == 0)
    return dimension > 0 && payload >= 0
               ? success()
               : emitError() << "static shape expression requires a positive identity and non-negative extent";
  if (dimension <= 0)
    return emitError() << "dynamic shape expression requires a positive identity";
  if ((kind == 1 && payload < 0) || (kind == 2 && payload != -1))
    return emitError() << "shape expression payload is invalid for its kind";
  return success();
}

LogicalResult ShapeRelationAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr axes) {
  if (!axes)
    return emitError() << "shape relation requires an axis array";
  for (Attribute axis : axes)
    if (!mlir::isa<ShapeExprAttr>(axis))
      return emitError() << "shape relation axes must be #intent.shape_expr values";
  return success();
}

LogicalResult IndexTermAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t kind,
    DenseI64ArrayAttr operandPositions, DenseI64ArrayAttr staticValues) {
  if (kind > 5 || !operandPositions || !staticValues)
    return emitError() << "index term has an invalid canonical schema";
  for (int64_t position : operandPositions.asArrayRef())
    if (position < -1)
      return emitError() << "index term operand slots must be absent or non-negative";
  return success();
}

LogicalResult IndexRelationAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t sourceRank,
    uint32_t resultRank, DenseI64ArrayAttr resultDimensions, ArrayAttr terms) {
  if (!resultDimensions || resultDimensions.size() != resultRank)
    return emitError()
           << "index relation result dimensions must align with result rank";
  for (int64_t identity : resultDimensions.asArrayRef())
    if (identity < 0)
      return emitError() << "index relation dimension identity must be non-negative";
  if (!terms || (terms.empty() && (sourceRank || resultRank)))
    return emitError() << "index relation requires typed terms";
  unsigned consumed = 0;
  for (Attribute term : terms) {
    auto typed = mlir::dyn_cast<IndexTermAttr>(term);
    if (!typed)
      return emitError() << "index relation terms must be #intent.index_term values";
    if (typed.getKind() != 1)
      ++consumed;
  }
  if (consumed != sourceRank)
    return emitError() << "index relation must consume every source axis exactly once";
  return success();
}

LogicalResult ViewConstraintsAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, bool hasStrides,
    ArrayAttr strides, StringAttr alias, bool noalias) {
  if (!strides || !alias)
    return emitError() << "view constraints require strides and alias fields";
  if (!hasStrides && !strides.empty())
    return emitError() << "absent stride constraints must use an empty array";
  if (!alias.getValue().empty() && noalias)
    return emitError() << "alias and noalias constraints are mutually exclusive";
  for (Attribute stride : strides)
    if (!mlir::isa<UnitAttr, IntegerAttr, StringAttr>(stride))
      return emitError() << "view stride constraint has an invalid typed entry";
  return success();
}

LogicalResult ParameterAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    int64_t originId) {
  if (!name || name.getValue().empty() || originId < 0)
    return emitError() << "parameter requires a name and non-negative source origin";
  return success();
}

LogicalResult FunctionKindAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t kind) {
  return kind <= 1
             ? success()
             : emitError() << "function kind is outside kernel/helper enum";
}

LogicalResult SparseFormatAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t kind,
    uint32_t compressionAxis) {
  (void)compressionAxis;
  return kind <= 1
             ? success()
             : emitError() << "sparse format kind is outside the closed schema";
}
