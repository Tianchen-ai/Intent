#ifndef INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_H
#define INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

#define GET_DIALECT_DECL
#include "Intent/Target/CuTile/IR/CuTileDialect.h.inc"

namespace intent::cutile {
inline constexpr llvm::StringLiteral arrayIndexTileBoundsAttr =
    "intent_cutile.array_index_tile_bounds";
} // namespace intent::cutile

#endif
