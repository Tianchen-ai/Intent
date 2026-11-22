#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/MapVector.h"
using namespace mlir;
namespace intentdsl::block {
// Expand a bounded private slab into SSA. The dry run and the rewrite use
// the same interpreter for induction values, memory addresses and work limit.
namespace {
struct Forward {
  Operation &space;
  ArrayRef<int64_t> dims, strides;
  OpBuilder *build;
  size_t work = 0;
  llvm::SmallDenseMap<Value, int64_t> integers;
  IRMapping ssa;
  llvm::MapVector<int64_t, std::pair<SmallVector<Value>, Value>> writes;
  llvm::DenseMap<int64_t, Value> memory;
  llvm::DenseMap<int64_t, Value> constants;
  void index(Value v, int64_t i) {
    integers[v] = i;
    if (build) {
      auto &value = constants[i];
      if (!value)
        value = build->create<arith::ConstantIndexOp>(space.getLoc(), i);
      ssa.map(v, value);
    }
  }
  bool body(Block &block) {
    for (auto &op : block.without_terminator()) {
      if (++work > 24576)
        return false;
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        auto lo = integers.find(loop.getLowerBound()),
             hi = integers.find(loop.getUpperBound()),
             step = integers.find(loop.getStep());
        if (lo == integers.end() || hi == integers.end() ||
            step == integers.end() || step->second != 1)
          return false;
        auto lower = lo->second, upper = hi->second;
        if (upper > lower && upper - lower > 64)
          return false;
        SmallVector<Value> carry;
        if (build)
          for (auto v : loop.getInitArgs())
            carry.push_back(ssa.lookupOrDefault(v));
        for (int64_t i = lower; i < upper; ++i) {
          index(loop.getInductionVar(), i);
          if (build)
            ssa.map(loop.getRegionIterArgs(), carry);
          if (!body(*loop.getBody()))
            return false;
          if (build) {
            carry.clear();
            for (auto v : loop.getBody()->getTerminator()->getOperands())
              carry.push_back(ssa.lookupOrDefault(v));
          }
        }
        if (build)
          ssa.map(loop.getResults(), carry);
        continue;
      }
      if (auto c = dyn_cast<arith::ConstantIndexOp>(op)) {
        index(c, c.value());
        continue;
      }
      if (auto d = dyn_cast<memref::DimOp>(op)) {
        if (!d.getConstantIndex() || d.getSource() != space.getOperand(0))
          return false;
        index(d, dims[*d.getConstantIndex()]);
        continue;
      }
      if (auto add = dyn_cast<arith::AddIOp>(op)) {
        auto a = integers.find(add.getLhs()), b = integers.find(add.getRhs());
        int64_t n;
        if (a == integers.end() || b == integers.end() ||
            __builtin_add_overflow(a->second, b->second, &n))
          return false;
        index(add.getResult(), n);
        continue;
      }
      auto load = dyn_cast<memref::LoadOp>(op);
      auto store = dyn_cast<memref::StoreOp>(op);
      if (load || store) {
        auto coords = load ? load.getIndices() : store.getIndices();
        if ((load ? load.getMemref() : store.getMemref()) !=
                space.getOperand(0) ||
            coords.size() != dims.size() ||
            coords[0] != space.getRegion(0).front().getArgument(0))
          return false;
        int64_t address = 0;
        for (unsigned i = 1; i < coords.size(); ++i) {
          auto n = integers.find(coords[i]);
          if (n == integers.end() || n->second < 0 || n->second >= dims[i])
            return false;
          int64_t term;
          if (__builtin_mul_overflow(n->second, strides[i], &term) ||
              __builtin_add_overflow(address, term, &address))
            return false;
        }
        if (build) {
          auto &value = memory[address];
          if (load) {
            if (!value)
              value = build->clone(op, ssa)->getResult(0);
            ssa.map(load.getResult(), value);
          } else {
            value = ssa.lookupOrDefault(store.getValue());
            SmallVector<Value> mapped;
            for (auto v : coords)
              mapped.push_back(ssa.lookupOrDefault(v));
            writes[address] = {mapped, value};
          }
        }
        continue;
      }
      if (op.getNumRegions() || !isMemoryEffectFree(&op) ||
          op.getNumResults() != 1)
        return false;
      if (build)
        build->clone(op, ssa);
    }
    return true;
  }
};
} // namespace
bool canForwardOwners(Operation *space,
                      const compilerinfra::TargetProfile &target) {
  auto caps = target.services.get<OwnerMachine>();
  auto fn = space->getParentOfType<func::FuncOp>();
  if (!caps || !caps->residentElements || fn.getNumArguments() != 1 ||
      space->getNumOperands() != 1 ||
      space->getAttrOfType<DenseI64ArrayAttr>("axes")[0] != 0)
    return false;
  auto f = target.services.get<CallFacts>();
  auto *arg = f->arguments[0].getAsObject();
  SmallVector<int64_t> dims, strides;
  for (auto &v : *arg->getArray("shape"))
    dims.push_back(*v.getAsInteger());
  for (auto &v : *arg->getArray("strides"))
    strides.push_back(*v.getAsInteger());
  if (dims.size() < 2 || dims.size() != strides.size())
    return false;
  int64_t size = 1;
  for (auto n : ArrayRef<int64_t>(dims).drop_front())
    if (n <= 0 || __builtin_mul_overflow(n, size, &size) ||
        size > caps->residentElements)
      return false;
  Forward plan{*space, dims, strides, nullptr};
  return plan.body(space->getRegion(0).front());
}
LogicalResult forwardOwners(func::FuncOp fn) {
  auto &space = fn.front().front();
  if (!space.hasAttr("block.forward_slab"))
    return success();
  auto &bb = space.getRegion(0).front();
  auto dims = fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.shape"),
       strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.strides");
  Block next;
  OpBuilder b(fn.getContext());
  b.setInsertionPointToEnd(&next);
  Forward plan{space, dims.asArrayRef(), strides.asArrayRef(), &b};
  if (!plan.body(bb))
    return space.emitError("invalid static owner working set");
  for (auto &[key, write] : plan.writes)
    b.create<memref::StoreOp>(space.getLoc(), write.second, space.getOperand(0),
                              write.first);
  while (auto *last = bb.back().getPrevNode())
    last->erase();
  bb.getOperations().splice(bb.begin(), next.getOperations());
  return success();
}
} // namespace intentdsl::block
