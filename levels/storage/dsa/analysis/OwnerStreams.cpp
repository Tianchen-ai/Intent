#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
namespace intentdsl::dsa {
OwnerStreams ownerStreams(func::FuncOp fn,
                          const compilerinfra::TargetProfile &p) {
  OwnerStreams result;
  auto &space = fn.front().front();
  auto abi = llvm::cantFail(
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi")));
  // Delayed writeback is legal for independent destinations. An InOut may
  // have intervening reads in this loop and uses the general lowering.
  for (auto &arg : *abi.getAsArray())
    if (arg.getAsObject()->getString("direction") == "InOut")
      return {};
  for (auto loop : space.getRegion(0).front().getOps<scf::ForOp>()) {
    auto bound = loop.getUpperBound().getDefiningOp<memref::DimOp>();
    if (!bound || !bound.getConstantIndex() ||
        !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One()))
      continue;
    auto layout = [&](Value v, StringRef k) {
      return fn.getArgAttrOfType<DenseI64ArrayAttr>(
          cast<BlockArgument>(v).getArgNumber(), k);
    };
    auto steps =
        layout(bound.getSource(), "block.shape")[*bound.getConstantIndex()];
    if (!steps || steps > INT_MAX)
      continue;
    for (auto &op : *loop.getBody()) {
      SmallVector<Value> coords;
      Value view;
      int member = -1;
      if (auto l = dyn_cast<memref::LoadOp>(op)) {
        view = l.getMemref();
        llvm::append_range(coords, l.getIndices());
      } else if (auto s = dyn_cast<memref::StoreOp>(op)) {
        view = s.getMemref();
        llvm::append_range(coords, s.getIndices());
      } else if (op.getName().stripDialect() == "slice_read") {
        view = op.getOperand(0);
        llvm::append_range(coords, op.getOperands().drop_front());
        member = op.getAttrOfType<IntegerAttr>("axis").getInt();
        coords.insert(coords.begin() + member, Value{});
      } else
        continue;
      auto axis = llvm::find(coords, loop.getInductionVar()) - coords.begin();
      if (axis == int64_t(coords.size()) ||
          llvm::count(coords, loop.getInductionVar()) != 1)
        continue;
      if (llvm::any_of(coords, [&](Value v) {
            return v && v != loop.getInductionVar() &&
                   !loop.isDefinedOutsideOfLoop(v);
          }))
        continue;
      auto sh = layout(view, "block.shape"),
           stride = layout(view, "block.strides");
      if (steps > sh[axis] || (member >= 0 && stride[member] != 1))
        continue;
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        // A second write in the same loop could overlap after projection.
        if (llvm::any_of(view.getUsers(), [&](Operation *other) {
              return other != &op && isa<memref::StoreOp>(other);
            }))
          continue;
      }
      int64_t width = member >= 0 ? sh[member] : 1, bytes = 0;
      result.bytes = llvm::alignTo(result.bytes, 128);
      if (width <= 0 || __builtin_mul_overflow(width, steps, &bytes) ||
          __builtin_mul_overflow(bytes,
                                 (int64_t)cast<MemRefType>(view.getType())
                                         .getElementTypeBitWidth() /
                                     8,
                                 &bytes))
        return {};
      result.transfers.push_back({&op, result.bytes, axis, steps, width});
      if (__builtin_add_overflow(bytes, result.bytes, &result.bytes))
        return {};
    }
  }
  auto caps = p.services.get<LocalSupply>();
  return caps && result.bytes <= std::min(caps->budget, caps->capacity)
             ? result
             : OwnerStreams{};
}
} // namespace intentdsl::dsa
