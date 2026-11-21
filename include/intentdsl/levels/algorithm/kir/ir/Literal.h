#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallString.h"

namespace intentdsl::kir {
inline std::string literal(mlir::TypedAttr attr) {
  if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return std::to_string(i.getInt());
  auto value = mlir::cast<mlir::FloatAttr>(attr);
  llvm::SmallString<32> text;
  value.getValue().toString(text);
  if (!text.str().contains('.') && !text.str().contains_insensitive("e"))
    text += ".0";
  return std::string(text);
}
} // namespace intentdsl::kir
