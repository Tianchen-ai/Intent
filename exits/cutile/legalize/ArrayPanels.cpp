#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using intentdsl::block::Tiles;
namespace {
Value reshape(Tiles &t, Value value, ArrayRef<int64_t> shape) {
  return t.op("reshape", value, t.shaped(shape, t.elem(value.getType())))
      ->getResult(0);
}
Value scalarIndex(Tiles &t, Value value, StringRef op = "min") {
  return isa<RankedTensorType>(value.getType()) ? t.reduce(value, op, 0)
                                                : value;
}
Value blockIndex(Tiles &t, Value index, int64_t width) {
  return t.b.create<arith::DivUIOp>(t.loc, index, t.integer(width));
}
Value load(Tiles &t, Value base, ValueRange at, ArrayRef<int64_t> shape,
           bool tma) {
  SmallVector<Value> args{base};
  llvm::append_range(args, at);
  return t
      .op("rect_read", args,
          t.shaped(shape, cast<MemRefType>(base.getType()).getElementType()),
          {t.b.getNamedAttr("allow_tma", t.b.getBoolAttr(tma))})
      ->getResult(0);
}
bool rectangle(Tiles &t, Operation &op, ArrayRef<Value> coords, int64_t rows,
               Value stored, Value &result) {
  if (!op.hasAttr("ct_src.owner_rectangle"))
    return false;
  auto base = op.getOperand(stored ? 1 : 0);
  int64_t width = t.shape(base).asArrayRef().back();
  SmallVector<Value> at;
  for (Value outer : coords.drop_back())
    at.push_back(scalarIndex(t, outer));
  at.push_back(blockIndex(t, scalarIndex(t, coords.back()), rows));
  at.push_back(t.integer(0));
  SmallVector<int64_t> dims(at.size(), 1);
  dims[dims.size() - 2] = rows;
  dims.back() = width;
  if (stored) {
    SmallVector<Value> args{base, reshape(t, stored, dims)};
    llvm::append_range(args, at);
    t.op("rect_write", args);
  } else
    result = reshape(t, load(t, base, at, dims, false), {rows, width});
  return true;
}
Value panel(Tiles &t, Operation &op, ArrayRef<Value> coords, Value start,
            int64_t span, int64_t width) {
  if (!llvm::isPowerOf2_64(width))
    return {};
  auto page = op.getAttrOfType<IntegerAttr>("block.page_extent");
  if (!page && !op.hasAttr("block.rectangular_tokens"))
    return {};
  SmallVector<Value> at;
  int axis = page ? 1 : coords.size() - 1;
  for (auto [i, v] : llvm::enumerate(coords)) {
    if (int(i) == axis) {
      auto offset = page ? Value(t.b.create<arith::RemUIOp>(
                               t.loc, start, t.integer(page.getInt())))
                         : start;
      at.push_back(blockIndex(t, offset, span));
    } else
      at.push_back(scalarIndex(t, v, i == 0 && page ? "max" : "min"));
  }
  at.push_back(t.integer(0));
  SmallVector<int64_t> dims(at.size(), 1);
  dims[axis] = span;
  dims.back() = width;
  bool tma = op.hasAttr(page ? "ct_src.page_tma" : "ct_src.panel_tma");
  return reshape(t, load(t, op.getOperand(0), at, dims, tma), {span, width});
}
} // namespace
intentdsl::block::OwnerAccess nativeOwnerViews() { return {rectangle, panel}; }
} // namespace CI_UNIT_NAMESPACE
