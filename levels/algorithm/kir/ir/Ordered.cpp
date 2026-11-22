#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
using namespace mlir;
namespace intentdsl::kir {
std::optional<std::pair<BlockArgument, unsigned>> memberAxis(Value v) {
  if (!isa<RankedTensorType>(v.getType()))
    return std::nullopt;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop || !arg.getArgNumber())
      return std::nullopt;
    return memberAxis(loop.getInitArgs()[arg.getArgNumber() - 1]);
  }
  auto *op = v.getDefiningOp();
  auto kind = op->getName().stripDialect();
  if (kind == "slice_read" || kind == "lane_splat") {
    auto view =
        dyn_cast<BlockArgument>(op->getOperand(kind == "slice_read" ? 0 : 1));
    auto ax = op->getAttrOfType<IntegerAttr>("axis");
    if (!view || !ax || ax.getInt() < 0 ||
        ax.getInt() >= cast<MemRefType>(view.getType()).getRank())
      return std::nullopt;
    return std::make_pair(view, unsigned(ax.getInt()));
  }
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return memberAxis(loop.getInitArgs()[cast<OpResult>(v).getResultNumber()]);
  if (auto branch = dyn_cast<scf::IfOp>(op))
    return memberAxis(
        branch.thenYield().getOperand(cast<OpResult>(v).getResultNumber()));
  if (op->getNumOperands())
    return memberAxis(op->getOperand(0));
  return std::nullopt;
}
int64_t memberExtent(func::FuncOp fn, Value v, StringRef key) {
  auto [a, axis] = *memberAxis(v);
  return fn.getArgAttrOfType<DenseI64ArrayAttr>(a.getArgNumber(), key)[axis];
}
int64_t ownerCount(mlir::Operation *op, const CallFacts &facts) {
  auto axes = op->getAttrOfType<mlir::DenseI64ArrayAttr>("axes");
  int64_t result = 1;
  for (auto [i, view] : llvm::enumerate(op->getOperands())) {
    auto *shape = facts.shapeOf(view);
    if (!shape || axes[i] < 0 || size_t(axes[i]) >= shape->size())
      return -1;
    auto n = (*shape)[axes[i]].getAsInteger();
    if (!n || *n < 0 || __builtin_mul_overflow(result, *n, &result))
      return -1;
  }
  return result;
}
bool orderedOwners(func::FuncOp fn) {
  return fn && !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().stripDialect() == "owner_space";
}
LogicalResult verifyProjectedOwners(Operation &space, const CallFacts &facts) {
  auto &body = space.getRegion(0).front();
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  auto check = [&](Value view, ValueRange indices, int member, Operation *op) {
    for (auto [i, v] : llvm::enumerate(indices)) {
      auto div = v.getDefiningOp<arith::FloorDivSIOp>();
      if (!div)
        continue;
      auto origin = dyn_cast<BlockArgument>(div.getLhs());
      auto group = div.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      if (!origin || origin.getOwner() != &body || !group)
        continue;
      auto d = group.value();
      auto axis = origin.getArgNumber();
      auto *shape = facts.shapeOf(space.getOperand(axis));
      auto *target = facts.shapeOf(view);
      auto dstAxis = i + (member >= 0 && i >= size_t(member));
      if (!shape || !target || d <= 0 || axes[axis] < 0 ||
          axes[axis] >= int64_t(shape->size()) || dstAxis >= target->size())
        return op->emitError("invalid grouped owner projection"), failure();
      auto n = (*shape)[axes[axis]].getAsInteger(),
           m = (*target)[dstAxis].getAsInteger();
      if (!n || !m || *n < 0 || *m < 0 || (*n && (*n - 1) / d >= *m))
        return op->emitError("grouped owner exceeds the view extent"),
               failure();
    }
    return success();
  };
  auto result = body.walk([&](Operation *o) {
    LogicalResult ok = success();
    if (auto l = dyn_cast<memref::LoadOp>(o))
      ok = check(l.getMemref(), l.getIndices(), -1, o);
    else if (o->getName().stripDialect() == "slice_read")
      ok = check(o->getOperand(0), o->getOperands().drop_front(),
                 o->getAttrOfType<IntegerAttr>("axis").getInt(), o);
    return failed(ok) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}
LogicalResult verifyOwners(func::FuncOp fn, StringRef dialect) {
  auto &op = fn.front().front();
  if (!llvm::hasSingleElement(fn.getBody()) ||
      fn.front().getOperations().size() != 2)
    return fn.emitError("expected exactly one independent owner space");
  return verifyOwnerBody(fn, op, dialect);
}
LogicalResult verifyOwnerBody(func::FuncOp fn, Operation &op,
                              StringRef dialect) {
  if (op.getName().stripDialect() != "owner_space" ||
      op.getName().getDialectNamespace() != dialect ||
      !llvm::hasSingleElement(op.getRegion(0)))
    return fn.emitError("expected one independent owner region");
  auto &body = op.getRegion(0).front();
  auto axes = op.getAttrOfType<DenseI64ArrayAttr>("axes");
  if (!axes || !op.getNumOperands() || axes.size() != op.getNumOperands() ||
      body.getNumArguments() != op.getNumOperands() || body.empty() ||
      body.back().getName().stripDialect() != "end")
    return op.emitError("owner axes and region signature disagree");
  auto attr = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!attr)
    return fn.emitError("missing ordered ABI");
  auto doc = llvm::json::parse(attr.getValue());
  if (!doc) {
    llvm::consumeError(doc.takeError());
    return fn.emitError("malformed ordered ABI");
  }
  auto *args = doc->getAsArray();
  if (!args || args->size() != fn.getNumArguments())
    return fn.emitError("ordered argument count mismatch");
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    auto *entry = (*args)[i].getAsObject();
    auto view = dyn_cast<MemRefType>(v.getType());
    auto rank = view ? view.getRank() : 0;
    if (!entry || !entry->getArray("dims") ||
        entry->getArray("dims")->size() != size_t(rank))
      return fn.emitError("ordered view rank disagrees with ABI");
    auto t = view ? view.getElementType() : v.getType();
    auto d = entry->getString("dtype");
    auto dir = entry->getString("direction");
    if ((rank ? dir != "In" && dir != "Out" && dir != "InOut"
              : dir != "Scalar") ||
        !((t.isF16() && d == "f16") || (t.isF32() && d == "f32") ||
          (t.isBF16() && d == "bf16") || (t.isInteger(32) && d == "i32")))
      return fn.emitError("unsupported ordered argument type or direction");
  }
  auto entry = [&](BlockArgument a) {
    return (*args)[a.getArgNumber()].getAsObject();
  };
  SmallVector<llvm::json::Value> bounds;
  for (auto [i, arg] : llvm::enumerate(op.getOperands())) {
    auto v = dyn_cast<BlockArgument>(arg);
    if (!v || v.getOwner() != &fn.front() ||
        !body.getArgument(i).getType().isIndex() || axes[i] < 0 ||
        axes[i] >= cast<MemRefType>(v.getType()).getRank())
      return op.emitError("invalid captured owner dimension");
    bounds.push_back((*entry(v)->getArray("dims"))[axes[i]]);
  }
  llvm::DenseSet<unsigned> written;
  llvm::DenseMap<Value, SmallVector<unsigned>> partitions;
  auto partition = [&](BlockArgument view, ValueRange coordinates) {
    SmallVector<unsigned> projection;
    for (auto [i, iv] : llvm::enumerate(body.getArguments())) {
      auto pos = llvm::find(coordinates, iv);
      if (pos == coordinates.end())
        return false;
      auto axis = pos - coordinates.begin();
      if ((*entry(view)->getArray("dims"))[axis] != bounds[i])
        return false;
      projection.push_back(axis);
    }
    auto [it, newView] = partitions.try_emplace(view, projection);
    return newView || it->second == projection;
  };
  auto walk = op.getRegion(0).walk([&](Operation *child) {
    StringRef ns = child->getName().getDialectNamespace();
    auto kind = child->getName().stripDialect();
    auto sameDomain = [&](Value a, Value b) {
      if (!isa<RankedTensorType>(a.getType()))
        return true;
      auto sa = memberAxis(a), sb = memberAxis(b);
      return sa && sb && sa->first.getOwner() == &fn.front() &&
             sb->first.getOwner() == &fn.front() &&
             (*entry(sa->first)->getArray("dims"))[sa->second] ==
                 (*entry(sb->first)->getArray("dims"))[sb->second];
    };
    if (ns == "arith" && child->getNumOperands() == 2 &&
        !sameDomain(child->getOperand(0), child->getOperand(1))) {
      child->emitError("member arithmetic requires identical logical axes");
      return WalkResult::interrupt();
    }
    if (auto loop = dyn_cast<scf::ForOp>(child))
      for (auto [initial, updated] :
           llvm::zip(loop.getInitArgs(),
                     loop.getBody()->getTerminator()->getOperands()))
        if (!sameDomain(initial, updated)) {
          child->emitError("ordered carry changes its member extent");
          return WalkResult::interrupt();
        }
    if (ns == dialect && (kind == "lane_splat" || kind == "slice_read" ||
                          kind == "slice_write" || kind == "lane_sum")) {
      auto v = kind == "slice_write" || kind == "lane_sum"
                   ? child->getOperand(0)
                   : child->getResult(0);
      auto source = memberAxis(v);
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getRank() != 1 || !source ||
          source->first.getOwner() != &fn.front()) {
        child->emitError("missing member vector extent");
        return WalkResult::interrupt();
      }
      if (kind == "lane_sum") {
        if (t.getElementType().isF32())
          return WalkResult::advance();
        child->emitError("member sum requires f32 elements");
        return WalkResult::interrupt();
      }
      auto base = dyn_cast<BlockArgument>(
          child->getOperand(kind == "slice_read" ? 0 : 1));
      auto axis = child->getAttrOfType<IntegerAttr>("axis").getInt();
      if (!base || base.getOwner() != &fn.front() || axis < 0 ||
          axis >= cast<MemRefType>(base.getType()).getRank() ||
          (*entry(base)->getArray("dims"))[axis] !=
              (*entry(source->first)->getArray("dims"))[source->second])
        return WalkResult::interrupt();
      if (kind == "lane_splat")
        return child->getOperand(0).getType() == t.getElementType()
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      auto coords =
          child->getOperands().drop_front(kind == "slice_write" ? 2 : 1);
      if (coords.size() + 1 != cast<MemRefType>(base.getType()).getRank() ||
          t.getElementType() !=
              cast<MemRefType>(base.getType()).getElementType())
        return WalkResult::interrupt();
      if (kind == "slice_write") {
        if (entry(base)->getString("direction") != "Out")
          return WalkResult::interrupt();
        SmallVector<Value> at(coords);
        at.insert(at.begin() + axis, Value{});
        if (!partition(base, at))
          return WalkResult::interrupt();
        written.insert(base.getArgNumber());
      } else if (entry(base)->getString("direction") != "In")
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<memref::StoreOp>(child)) {
      auto v = dyn_cast<BlockArgument>(store.getMemref());
      if (!v || v.getOwner() != &fn.front() ||
          entry(v)->getString("direction") == "In") {
        child->emitError("store requires a writable capture");
        return WalkResult::interrupt();
      }
      if (!partition(v, store.getIndices())) {
        child->emitError("inconsistent output ownership");
        return WalkResult::interrupt();
      }
      written.insert(v.getArgNumber());
    } else if (auto load = dyn_cast<memref::LoadOp>(child)) {
      auto v = dyn_cast<BlockArgument>(load.getMemref());
      if (!v || v.getOwner() != &fn.front() ||
          entry(v)->getString("direction") == "Out") {
        child->emitError("load requires a readable capture");
        return WalkResult::interrupt();
      }
      if (entry(v)->getString("direction") == "InOut" &&
          !partition(v, load.getIndices())) {
        child->emitError("InOut read crosses output owner");
        return WalkResult::interrupt();
      }
    } else if (ns != "arith" && ns != "math" &&
               child->getName().getStringRef() != "memref.dim" &&
               child->getName().getStringRef() != "scf.for" &&
               child->getName().getStringRef() != "scf.yield" &&
               child->getName().getStringRef() != "scf.if" &&
               child != &body.back()) {
      child->emitError("unexpected operation in ordered region");
      return WalkResult::interrupt();
    }
    if (auto loop = dyn_cast<scf::ForOp>(child)) {
      auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
      if (!step || step.value() <= 0) {
        child->emitError("ordered-for requires a positive fixed step");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();
  for (auto a : fn.getArguments()) {
    auto dir = entry(a)->getString("direction");
    if ((dir == "Out" || dir == "InOut") && !written.contains(a.getArgNumber()))
      return fn.emitError("unpublished ordered output");
  }
  return success();
}
} // namespace intentdsl::kir
