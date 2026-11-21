#pragma once
#include "mlir/IR/Builders.h"
#include "llvm/Support/JSON.h"
namespace intentdsl::block {
inline mlir::ArrayAttr grid(mlir::Builder &b, bool columns) {
  auto rows = b.getStrArrayAttr({"ceildiv","M","rows"});
  if (!columns) return rows;
  return b.getArrayAttr({b.getStringAttr("mul"),rows,b.getStrArrayAttr({"ceildiv","N","width"})});
}
inline llvm::json::Value gridExpression(mlir::Attribute a) {
  if (auto s=mlir::dyn_cast<mlir::StringAttr>(a)) return s.str();
  llvm::json::Array expr;
  for(auto member:mlir::cast<mlir::ArrayAttr>(a)) expr.push_back(gridExpression(member));
  return expr;
}
}
