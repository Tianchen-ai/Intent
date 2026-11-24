#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace intentdsl::block {
using namespace mlir;
using namespace compilerinfra;
static llvm::ArrayRef<int64_t> shape(Type t) {
  if (auto s = dyn_cast<RankedTensorType>(t))
    return s.getShape();
  return {};
}
LogicalResult
verifyTileProgram(func::FuncOp fn, const TargetProfile &profile,
                  llvm::function_ref<bool(Operation &)> canNarrowAddress) {
  if (auto num = fn->getAttr("kir.numerics")) {
    auto s = dyn_cast<StringAttr>(num);
    if (!s ||
        (s.getValue() != "source" && s.getValue() != "relaxed_normalization"))
      return fn.emitError("invalid kernel numerical permission");
  }
  if (!llvm::hasSingleElement(fn.getBody()) ||
      fn.front().getOperations().size() != 2 ||
      !isa<func::ReturnOp>(fn.front().back()))
    return fn.emitError("expected single tile_program followed by return");
  auto &sp = fn.front().front();
  if (sp.getName().stripDialect() != "tile_program")
    return sp.emitError("expected tile_program entry");
  auto facts = profile.services.get<intentdsl::CallFacts>();
  if (!facts || !fn->getAttrOfType<StringAttr>("kir.abi") ||
      !fn->hasAttr("kir.domain_arg"))
    return failure();
  auto mode = sp.getAttrOfType<StringAttr>("calling_convention");
  auto rows = sp.getAttrOfType<IntegerAttr>("block.owner_tile"),
       owners = sp.getAttrOfType<IntegerAttr>("block.owners");
  if (mode == "owners") {
    if (!rows || rows.getInt() < 1 || !owners ||
        owners.getInt() != intentdsl::kir::ownerCount(&sp, *facts))
      return sp.emitError("invalid owner grid");
  } else if (mode == "matrix") {
    auto dims = sp.getAttrOfType<DenseI64ArrayAttr>("block.extents");
    auto tile = sp.getAttrOfType<DenseI64ArrayAttr>("block.output_tile");
    auto native = sp.getAttrOfType<StringAttr>("block.native_row");
    auto caps = profile.services.get<intentdsl::block::MatrixTarget>();
    int64_t dotElemBytes = 0;
    sp.walk([&](Operation *inst) {
      auto k = inst->getName().stripDialect();
      if (k != "tile_dot" && k != "mma")
        return;
      for (unsigned i = 0; i < std::min(2u, inst->getNumOperands()); ++i) {
        Type elem = getElementTypeOrSelf(inst->getOperand(i).getType());
        if (elem.isIntOrFloat())
          dotElemBytes = std::max<int64_t>(
              dotElemBytes, llvm::divideCeil(elem.getIntOrFloatBitWidth(), 8));
      }
    });
    if (!dims || dims.size() != 4 ||
        llvm::any_of(dims.asArrayRef(), [](auto i) { return i < 0; }) ||
        !tile || tile.size() != 3 || !native || !caps ||
        !llvm::any_of(caps->tiles, [&](auto &t) {
          return tile.asArrayRef() == ArrayRef<int64_t>{t.m, t.n, t.k} &&
                 native == t.native &&
                 (!t.maxOperandBytes || dotElemBytes <= t.maxOperandBytes);
        }))
      return sp.emitError(
          "invalid installed contraction grid or native parameter row");
  } else
    return sp.emitError("unknown tile calling convention");
  auto check = fn.walk([&](Operation *op) {
    auto name = op->getName().stripDialect();
    if (name == "drop_unit_owner") {
      auto input = op->getOperand(0).getType(),
           out = op->getResult(0).getType();
      if (shape(input).size() != 2 || shape(input)[0] != 1 ||
          shape(input).drop_front() != shape(out) ||
          getElementTypeOrSelf(input) != getElementTypeOrSelf(out))
        return op->emitError("fold capture must have exactly one owner"),
               WalkResult::interrupt();
    }
    if (name == "tile_prefix") {
      auto input = op->getOperand(0).getType(),
           out = op->getResult(0).getType();
      auto a = shape(input), y = shape(out);
      if (a.size() != 1 || y.size() != 1 || y[0] < 1 || y[0] > a[0] ||
          !llvm::isPowerOf2_64(y[0]) || y[0] > INT32_MAX ||
          getElementTypeOrSelf(input) != getElementTypeOrSelf(out))
        return op->emitError("invalid logical fragment prefix"),
               WalkResult::interrupt();
    }
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto dense = dyn_cast<DenseElementsAttr>(c.getValue());
          dense && !dense.isSplat())
        return op->emitError("tile constants require scalar or splat values"),
               WalkResult::interrupt();
    }
    if (auto cmp = dyn_cast<arith::CmpFOp>(op);
        cmp && cmp.getPredicate() != arith::CmpFPredicate::OEQ &&
        cmp.getPredicate() != arith::CmpFPredicate::UNE)
      return op->emitError("unsupported tile float comparison"),
             WalkResult::interrupt();
    if (name == "e8m0" || name == "from_le_half") {
      auto result = op->getResult(0).getType();
      if (!getElementTypeOrSelf(result).isF32() ||
          llvm::any_of(op->getOperands(), [&](Value v) {
            return !getElementTypeOrSelf(v.getType()).isInteger(32) ||
                   shape(v.getType()) != shape(result);
          }))
        return op->emitError(
                   "encoded field requires i32 fields and matching f32 lanes"),
               WalkResult::interrupt();
    }
    if (name == "tile_address") {
      auto bits = op->getAttrOfType<IntegerAttr>("index_bits").getInt();
      if (bits != 64 && (bits != 32 || !canNarrowAddress(*op)))
        return op->emitError(
                   "address narrowing lacks a static signed range proof"),
               WalkResult::interrupt();
      auto view = dyn_cast<BlockArgument>(op->getOperand(0));
      auto axes = op->getAttrOfType<DenseI64ArrayAttr>("projection"),
           strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
      auto rank = cast<MemRefType>(op->getOperand(0).getType()).getRank();
      if (!view || view.getOwner() != &fn.front() || axes.size() != rank ||
          strides.size() != rank || op->getNumOperands() != rank + 1 ||
          !getElementTypeOrSelf(op->getResult(0).getType()).isInteger(64))
        return op->emitError("invalid address rank or index width"),
               WalkResult::interrupt();
      auto native =
          facts->arguments[view.getArgNumber()].getAsObject()->getArray(
              "strides");
      auto output = shape(op->getResult(0).getType());
      if (!native || native->size() != rank || output.size() > 2)
        return WalkResult::interrupt();
      for (int i = 0; i < rank; i++) {
        auto type = op->getOperand(i + 1).getType();
        auto coord = shape(type);
        auto axis = axes[i];
        if ((*native)[i].getAsInteger() != strides[i] ||
            !getElementTypeOrSelf(type).isInteger(64) ||
            (coord.empty()
                 ? axis != -1
                 : (coord.size() != 1 || axis < 0 ||
                    size_t(axis) >= output.size() || coord[0] != output[axis])))
          return op->emitError("coordinate projection or ABI stride mismatch"),
                 WalkResult::interrupt();
      }
    }
    if (name == "tile_pid") {
      auto dim = op->getAttrOfType<IntegerAttr>("axis").getInt();
      if (dim < 0 || dim >= 3)
        return op->emitError("invalid launch dimension"),
               WalkResult::interrupt();
    }
    if (name == "tile_axis") {
      auto result = op->getResult(0).getType();
      if (shape(result).size() != 1 || shape(result)[0] <= 0 ||
          !llvm::isPowerOf2_64(shape(result)[0]) ||
          !getElementTypeOrSelf(result).isInteger(64))
        return op->emitError("invalid i64 tile axis"), WalkResult::interrupt();
    }
    if (name == "tile_broadcast") {
      auto src = op->getOperand(0).getType(), dst = op->getResult(0).getType();
      auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
      if (getElementTypeOrSelf(src) != getElementTypeOrSelf(dst) ||
          (shape(src).empty()
               ? axis != -1
               : (shape(src).size() != 1 || shape(dst).size() != 2 ||
                  axis < 0 || axis > 1 || shape(src)[0] != shape(dst)[axis])))
        return op->emitError("invalid tile broadcast projection"),
               WalkResult::interrupt();
    }
    if (name == "tile_read" || name == "tile_write") {
      auto offset = op->getOperand(1);
      auto *addr = offset.getDefiningOp();
      bool load = name == "tile_read";
      auto val = load ? op->getResult(0) : op->getOperand(2);
      if (op->getNumOperands() == (load ? 3 : 4)) {
        auto mask = op->getOperands().back().getType();
        if (shape(mask) != shape(offset.getType()) ||
            !getElementTypeOrSelf(mask).isInteger(1))
          return op->emitError("tile mask must cover its projected address"),
                 WalkResult::interrupt();
      }
      if (!addr ||
          addr->getName().getDialectNamespace() !=
              sp.getName().getDialectNamespace() ||
          addr->getName().stripDialect() != "tile_address" ||
          addr->getOperand(0) != op->getOperand(0) ||
          shape(offset.getType()) != shape(val.getType()) ||
          getElementTypeOrSelf(val.getType()) !=
              cast<MemRefType>(op->getOperand(0).getType()).getElementType())
        return op->emitError("tile access does not match its verified view"),
               WalkResult::interrupt();
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op);
        cmp && cmp.getPredicate() != arith::CmpIPredicate::eq &&
        cmp.getPredicate() != arith::CmpIPredicate::ne &&
        cmp.getPredicate() != arith::CmpIPredicate::sge &&
        cmp.getPredicate() != arith::CmpIPredicate::sle &&
        cmp.getPredicate() != arith::CmpIPredicate::sgt &&
        cmp.getPredicate() != arith::CmpIPredicate::slt)
      return op->emitError("unsupported tile comparison"),
             WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (check.wasInterrupted())
    return failure();
  return success();
}
} // namespace intentdsl::block
