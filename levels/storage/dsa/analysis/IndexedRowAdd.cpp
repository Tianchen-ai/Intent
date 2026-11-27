#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intentdsl::dsa {
std::optional<IndexedRowAdd>
indexedRowAdd(func::FuncOp fn, const compilerinfra::TargetProfile &p) {
  auto caps = p.services.get<LocalSupply>();
  if (!caps || !p.features.count("row_atomic_add"))
    return {};
  auto &space = fn.front().front();
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  if (!axes || axes.asArrayRef() != ArrayRef<int64_t>{1})
    return {};
  auto &body = space.getRegion(0).front();
  if (body.getNumArguments() != 1)
    return {};
  auto col = body.getArgument(0);
  scf::ForOp fold;
  for (auto &v : body.without_terminator()) {
    if (auto loop = dyn_cast<scf::ForOp>(v); loop && !fold)
      fold = loop;
    else if (!isa<arith::ConstantOp, memref::DimOp>(v))
      return {};
  }
  if (!fold || !fold->hasAttr("kir.reassociate") || fold.getNumResults() ||
      !matchPattern(fold.getLowerBound(), m_Zero()) ||
      !matchPattern(fold.getStep(), m_One()))
    return {};
  memref::StoreOp store;
  for (auto &v : fold.getBody()->without_terminator()) {
    if (auto s = dyn_cast<memref::StoreOp>(v); s && !store)
      store = s;
    else if (!isa<memref::LoadOp, arith::AddFOp, arith::AddIOp,
                  arith::IndexCastOp>(v))
      return {};
  }
  if (!store || store.getIndices().size() != 2 ||
      store.getIndices()[1] != col || store.getMemref() != space.getOperand(0))
    return {};
  auto add = store.getValue().getDefiningOp();
  if (!add || !isa<arith::AddFOp, arith::AddIOp>(add) ||
      (isa<arith::AddFOp>(add) &&
       !compilerinfra::hasFastMath(add, arith::FastMathFlags::reassoc)))
    return {};
  auto old = add->getOperand(0).getDefiningOp<memref::LoadOp>();
  auto src = add->getOperand(1).getDefiningOp<memref::LoadOp>();
  if (!old || !src)
    return {};
  if (src.getMemref() == store.getMemref())
    std::swap(src, old);
  if (old.getMemref() != store.getMemref() ||
      old.getIndices() != store.getIndices() || src.getIndices().size() != 2 ||
      src.getIndices()[0] != fold.getInductionVar() ||
      src.getIndices()[1] != col)
    return {};
  auto indexCast = old.getIndices()[0].getDefiningOp<arith::IndexCastOp>();
  auto key = indexCast ? indexCast.getIn().getDefiningOp<memref::LoadOp>()
                       : memref::LoadOp{};
  if (!key || !key.getType().isInteger(32) || key.getIndices().size() != 1 ||
      key.getIndices()[0] != fold.getInductionVar())
    return {};
  auto dst = store.getMemref();
  auto source = src.getMemref();
  auto keys = key.getMemref();
  if (!llvm::all_of(ArrayRef<Value>{keys, source, dst}, [&](Value v) {
        auto arg = dyn_cast<BlockArgument>(v);
        return arg && arg.getOwner() == &fn.front();
      }))
    return {};
  auto attr = [&](Value v, StringRef name) {
    return fn.getArgAttr(mlir::cast<BlockArgument>(v).getArgNumber(), name);
  };
  auto shape = [&](Value v) {
    return mlir::cast<DenseI64ArrayAttr>(attr(v, "block.shape")).asArrayRef();
  };
  auto strides = [&](Value v) {
    return mlir::cast<DenseI64ArrayAttr>(attr(v, "block.strides")).asArrayRef();
  };
  auto group = attr(dst, "block.alias_group");
  auto knownGroup = [&](Value v) {
    auto value = dyn_cast_or_null<IntegerAttr>(attr(v, "block.alias_group"));
    return value && value.getInt() >= 0;
  };
  if (!knownGroup(dst) || !knownGroup(keys) || !knownGroup(source))
    return {};
  if (group == attr(keys, "block.alias_group") ||
      group == attr(source, "block.alias_group"))
    return {};
  auto width = shape(dst)[1], rows = shape(source)[0];
  if (rows <= 0 || shape(keys)[0] != rows || shape(source)[1] != width ||
      width <= 0 || width > caps->budget / 4 ||
      strides(keys) != ArrayRef<int64_t>{1} ||
      strides(source) != ArrayRef<int64_t>{width, 1} ||
      strides(dst) != ArrayRef<int64_t>{width, 1} ||
      (!old.getType().isF32() && !old.getType().isInteger(32)))
    return {};
  auto ub = fold.getUpperBound().getDefiningOp<memref::DimOp>();
  llvm::APInt boundAP;
  if (!(ub && (ub.getSource() == source || ub.getSource() == keys) &&
        ub.getConstantIndex() == 0) &&
      !(matchPattern(fold.getUpperBound(), m_ConstantInt(&boundAP)) &&
        boundAP.getSExtValue() == rows))
    return {};
  // One NRAM result row receives the old atomic values. Keep independent
  // batches on all IPUs and amortize DMA over adjacent source rows.
  int64_t bytes = std::min(caps->capacity, caps->budget);
  int64_t batch = std::min<int64_t>(
      llvm::divideCeil(rows, caps->consumers),
      (bytes - int64_t(llvm::alignTo(width * 4, caps->alignment)) -
       2 * caps->alignment) /
          (width * 4 + 4));
  if (batch < 1)
    return {};
  return IndexedRowAdd{keys, source, dst, rows, width, batch};
}
} // namespace intentdsl::dsa
