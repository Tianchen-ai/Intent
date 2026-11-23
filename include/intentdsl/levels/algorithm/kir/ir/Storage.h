#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "intentdsl/levels/algorithm/kir/Storage.h.inc"

namespace intentdsl::kir {
inline mlir::Type storageFormat(mlir::Operation &op) {
  auto code=op.getName().stripDialect();
  if(code=="q4k_dot")return Q4KType::get(op.getContext());
  if(code=="quantize_q8k")return Q8KType::get(op.getContext());
  if(code=="scaled_contract")return MX8Type::get(op.getContext());
  return {};
}
inline mlir::LogicalResult verifyStorageFormat(mlir::Operation &op, bool required) {
  auto expected=storageFormat(op);
  auto attr=op.getAttr("kir.storage");
  if ((!required && !attr) || attr==(expected ? mlir::TypeAttr::get(expected) : mlir::Attribute{}))
    return mlir::success();
  return op.emitError("storage type disagrees with encoded operands");
}
} // namespace intentdsl::kir
