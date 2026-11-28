#include "../capabilities/Index.h"
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
LogicalResult verifyArrayGeometry(func::FuncOp fn,
                                  const compilerinfra::TargetProfile &target) {
  auto cap = target.services.get<intentdsl::cutile::ArrayIndices>();
  if (!cap || fn->getAttr("ct_src.index_bits") !=
                  Builder(fn.getContext()).getI32IntegerAttr(cap->bits))
    return fn.emitError("array index ABI differs from target");
  llvm::json::Object views;
  if (auto attr = fn->getAttrOfType<StringAttr>("ct_src.native_views")) {
    auto metadata = llvm::json::parse(attr.getValue());
    if (!metadata) {
      llvm::consumeError(metadata.takeError());
      return fn.emitError("invalid packed array shapes");
    }
    if (!metadata->getAsObject())
      return fn.emitError("expected packed array shapes");
    views = std::move(*metadata->getAsObject());
    for (auto &item : views) {
      unsigned slot = 0;
      if (llvm::StringRef(item.first).getAsInteger(10, slot) ||
          slot >= fn.getNumArguments())
        return fn.emitError("invalid packed array slot");
      auto s = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "block.shape"),
           strides =
               fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "block.strides");
      auto *shape = item.second.getAsArray();
      if (!s || s.size() != 3 || !strides || strides.size() != 3 || !shape ||
          shape->size() != 2)
        return fn.emitError("packed matrix view requires rank-three source and "
                            "rank-two target");
      bool match = false;
      for (int axis : {0, 1}) {
        auto adjacent =
            llvm::checkedMul<int64_t>(s[axis + 1], strides[axis + 1]);
        auto product = llvm::checkedMul<int64_t>(s[axis], s[axis + 1]);
        if (adjacent && *adjacent == strides[axis] && product &&
            (*shape)[axis].getAsInteger() == *product &&
            (*shape)[1 - axis].getAsInteger() == s[axis ? 0 : 2])
          match = true;
      }
      if (!match)
        return fn.emitError(
            "packed matrix view does not preserve the original strides");
    }
  }
  return success(
      !fn.walk([&](Operation *op) {
           auto n = op->getName().getStringRef();
           auto reject = [&](StringRef message) {
             op->emitError(message);
             return WalkResult::interrupt();
           };
           if (n == "ct_src.rect_read" || n == "ct_src.rect_write") {
             bool write = n == "ct_src.rect_write";
             auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
             if (!arg || arg.getOwner() != &fn.front())
               return reject("rectangular access must bind a function view");
             auto view = cast<MemRefType>(arg.getType());
             auto tile =
                 cast<RankedTensorType>(write ? op->getOperand(1).getType()
                                              : op->getResult(0).getType());
             int rank = views.get(std::to_string(arg.getArgNumber()))
                            ? 2
                            : view.getRank();
             if (tile.getRank() != rank ||
                 tile.getElementType() != view.getElementType() ||
                 op->getNumOperands() != (write ? 2 : 1) + rank ||
                 !llvm::all_of(tile.getShape(), [](int64_t n) {
                   return n > 0 && llvm::isPowerOf2_64(n);
                 }))
               return reject("array view, tile shape and indices disagree");
             if (auto latency = op->getAttrOfType<IntegerAttr>("latency");
                 latency &&
                 (write || latency.getInt() < 1 || latency.getInt() > 10))
               return reject(
                   "array load latency must be within the SDK range 1..10");
             if (!write &&
                 op->getAttrOfType<BoolAttr>("allow_tma").getValue() &&
                 !cap->permitTma)
               return reject("unsupported tiled transfer");
           }
           if (n == "ct_src.reshape" || n == "ct_src.transpose" ||
               n == "ct_src.expand_scale") {
             auto from = cast<RankedTensorType>(op->getOperand(0).getType()),
                  to = cast<RankedTensorType>(op->getResult(0).getType());
             if (!from.hasStaticShape() || !to.hasStaticShape() ||
                 from.getElementType() != to.getElementType())
               return reject("incompatible tile element or extents");
             if (n == "ct_src.reshape" &&
                 from.getNumElements() != to.getNumElements())
               return reject("reshape changes tile size");
             if (n == "ct_src.transpose" &&
                 (from.getRank() != 2 || to.getRank() != 2 ||
                  from.getDimSize(0) != to.getDimSize(1) ||
                  from.getDimSize(1) != to.getDimSize(0)))
               return reject("invalid matrix transpose");
             if (n == "ct_src.expand_scale") {
               auto a = op->getAttrOfType<IntegerAttr>("axis").getInt();
               SmallVector<int64_t> shape(to.getShape());
               if (a < 0 || a >= to.getRank())
                 return reject("invalid scale replication axis");
               shape.erase(shape.begin() + a);
               if (ArrayRef<int64_t>(shape) != from.getShape())
                 return reject("scale does not cover the grouped axes");
             }
           }
           return WalkResult::advance();
         }).wasInterrupted());
}
} // namespace CI_UNIT_NAMESPACE
