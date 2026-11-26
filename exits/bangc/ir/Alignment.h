#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include <numeric>

namespace intentdsl::bangc {
// Local storage declarations promise 128-byte alignment. Determine how much
// of that guarantee survives the generated C pointer arithmetic.
inline constexpr unsigned localAllocationAlignment = 128;

inline unsigned integerAlignment(mlir::Value v, unsigned depth = 0) {
  using namespace mlir;
  if (depth > 32)
    return 1;
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return 1u << std::min<unsigned>(7, c.countr_zero());
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (loop && v == loop.getInductionVar())
      return std::min(integerAlignment(loop.getLowerBound(), depth + 1),
                      integerAlignment(loop.getStep(), depth + 1));
    return 1;
  }
  auto *op = v.getDefiningOp();
  if (!op)
    return 1;
  if (isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(op)) {
    unsigned a = integerAlignment(op->getOperand(0), depth + 1);
    unsigned b = integerAlignment(op->getOperand(1), depth + 1);
    return isa<arith::MulIOp>(op) ? std::min(a * b, localAllocationAlignment)
                                  : std::min(a, b);
  }
  if (isa<arith::SelectOp>(op))
    return std::min(integerAlignment(op->getOperand(1), depth + 1),
                    integerAlignment(op->getOperand(2), depth + 1));
  if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return integerAlignment(op->getOperand(0), depth + 1);
  if (op->getName().getStringRef() == "bangc_src.symbol") {
    auto name = op->getAttrOfType<StringAttr>("name").getValue();
    if (name != "P" && name != "BYTES")
      return 1;
    auto fn = op->getParentOfType<func::FuncOp>();
    auto layout = fn ? fn->getAttrOfType<DictionaryAttr>("dsa.layout") : DictionaryAttr();
    auto rows = layout ? layout.getAs<ArrayAttr>("rows") : ArrayAttr();
    if (rows && !rows.empty()) {
      unsigned alignment = localAllocationAlignment;
      for (auto row : rows) {
        auto pitch = cast<DictionaryAttr>(row).getAs<IntegerAttr>(name == "P" ? "pitch" : "bytes");
        if (!pitch || pitch.getInt() <= 0)
          return 1;
        alignment = std::gcd<uint64_t>(alignment, pitch.getInt());
      }
      return alignment;
    }
  }
  return 1;
}

inline uint64_t storageAlignment(mlir::Value value) {
  using namespace mlir;
  Operation *op = value.getDefiningOp();
  if (!op)
    return 1;
  auto name = op->getName().getStringRef();
  if (name == "bangc_src.nram_alloc" || name == "bangc_src.nram_buffer" ||
      name == "bangc_src.sram_buffer" || name == "bangc_src.wram_buffer" ||
      name == "bangc_src.local_var")
    return localAllocationAlignment;
  if (name != "bangc_src.address" && name != "bangc_src.nram_ptr")
    return 1;
  Value base = op->getOperand(0);
  unsigned alignment = storageAlignment(base);
  unsigned bytes = cast<MemRefType>(base.getType()).getElementTypeBitWidth() / 8;
  if (!bytes)
    return 1;
  if (name == "bangc_src.nram_ptr")
    return std::gcd(alignment, bytes * integerAlignment(op->getOperand(1)));
  unsigned rank = (op->getNumOperands() - 1) / 2;
  for (unsigned i = 0; i < rank; ++i)
    alignment = std::gcd(alignment, bytes * integerAlignment(op->getOperand(i + 1)) *
                                            integerAlignment(op->getOperand(i + 1 + rank)));
  return alignment;
}
} // namespace intentdsl::bangc
