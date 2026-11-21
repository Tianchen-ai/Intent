#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "../decisions/Plan.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
using namespace mlir;
namespace intentdsl::dsa {
namespace {
// Each slot has one producer and one consuming interval per iteration.
// A lookahead copy targets the other slot only after the current slot is ready.
bool legalSchedule(Block &schedule, int64_t depth, StringRef source,bool persistent) {
  auto it = schedule.begin();
  auto take = [](Block &block, Block::iterator &pos,
                 StringRef code) -> Operation * {
    if (pos == block.end() || pos->getName().getStringRef() != code)
      return nullptr;
    return &*pos++;
  };
  auto *r = take(schedule, it, "dsa.range"),
       *z = take(schedule, it, "dsa.slot_zero");
  if (!r || !z || r->getAttrOfType<StringAttr>("source") != source)
    return false;
  auto initial = z->getResult(0), limit = r->getResult(1),
       step = r->getResult(2);
  auto accepts = [](Operation *op, ValueRange values) {
    return op && llvm::equal(op->getOperands(), values);
  };
  if(persistent && !accepts(take(schedule,it,"dsa.load_shared"),{r->getResult(0),limit}))return false;
  if (depth == 2 && !accepts(take(schedule, it, "dsa.issue_if_in_bounds"),
                             {r->getResult(0), limit, initial}))
    return false;
  auto *loop = take(schedule, it, "dsa.iterate");
  if (!accepts(loop, {r->getResult(0), limit, step, initial}) ||
      !take(schedule, it, "dsa.end") || it != schedule.end() ||
      !llvm::hasSingleElement(loop->getRegion(0)))
    return false;
  auto &body = loop->getRegion(0).front();
  if (body.getNumArguments() != 2)
    return false;
  auto pos = body.getArgument(0), slot = body.getArgument(1);
  Value nextSlot = slot;
  auto cursor = body.begin();
  if (depth == 2) {
    if (!accepts(take(body, cursor, "dsa.wait_slot"), slot))
      return false;
    auto *next = take(body, cursor, "dsa.next"),
         *flip = take(body, cursor, "dsa.other_slot");
    if (!accepts(next, {pos, step}) || !accepts(flip, slot))
      return false;
    nextSlot = flip->getResult(0);
    if (!accepts(take(body, cursor, "dsa.issue_if_in_bounds"),
                 {next->getResult(0), limit, nextSlot}))
      return false;
  } else if (!accepts(take(body, cursor, "dsa.issue_if_in_bounds"),
                      {pos, limit, slot}))
    return false;
  return accepts(take(body, cursor, "dsa.compute_members"), {pos, slot}) &&
         accepts(take(body, cursor, "dsa.advance"), nextSlot) &&
         cursor == body.end();
}
bool access(Operation *op, unsigned mask, Operation &window, func::FuncOp fn,
            const llvm::json::Array &abi) {
  if (op->getOperand(mask) != window.getResult(2))
    return false;
  auto coords = op->getOperands().drop_front(mask + 1);
  auto view = dyn_cast<BlockArgument>(op->getOperand(0));
  auto domain = dyn_cast<BlockArgument>(window.getOperand(0));
  if (!view || !domain || view.getOwner() != &fn.front() ||
      domain.getOwner() != &fn.front())
    return false;
  auto dims = [&](BlockArgument a) -> const llvm::json::Array * {
    auto *s = abi[a.getArgNumber()].getAsObject();
    return s ? s->getArray("dims") : nullptr;
  };
  auto *actual = dims(view), *logical = dims(domain);
  if (!actual || !logical || (logical->empty() || logical->size()>4) ||
      actual->size() != coords.size())
    return false;
  llvm::SmallDenseSet<unsigned> seen;
  for (auto [i, v] : llvm::enumerate(coords)) {
    auto axis = dyn_cast<OpResult>(v);
    if(!axis)return false;
    int64_t dim=-1;
    auto owner = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
    if(owner && v==window.getResult(0)) dim=owner.getInt();
    else if(owner && axis.getOwner()->getName().getStringRef()=="dsa.slab_axis" &&
            axis.getOwner()->getOperand(0)==window.getResult(1) && axis.getOwner()->getOperand(1)==domain){
      dim=axis.getOwner()->getAttrOfType<IntegerAttr>("axis").getInt();
      if(dim==owner.getInt())return false;
    }
    else if(owner)return false;
    else if(v==window.getResult(1))dim=logical->size()-1;
    else if(logical->size()==2 && v==window.getResult(0))dim=0;
    else if (axis.getOwner()->getName().getStringRef() == "dsa.member_axis" &&
             axis.getOwner()->getOperand(0) == window.getResult(0) &&
             axis.getOwner()->getOperand(1) == domain) {
      dim=axis.getOwner()->getAttrOfType<IntegerAttr>("axis").getInt();
      if (dim < 0 || size_t(dim) + 1 >= logical->size())
        return false;
    }
    if(dim<0 || size_t(dim)>=logical->size() || !seen.insert(dim).second || (*actual)[i]!=(*logical)[dim])return false;
  }
  // A read may broadcast an axis. A write must cover a permutation of
  // the entire logical domain, including dynamic axes of unknown extent.
  return mask != 2 || seen.size() == logical->size();
}
} // namespace
bool projectsConsecutiveRows(Operation *op, unsigned maskIndex) {
  auto rank = cast<MemRefType>(op->getOperand(0).getType()).getRank();
  if (rank < 3 || op->getNumOperands() != maskIndex + 1 + rank)
    return false;
  auto mask = dyn_cast<OpResult>(op->getOperand(maskIndex));
  if (!mask || mask.getOwner()->getName().getStringRef() != "dsa.window")
    return false;
  auto *window = mask.getOwner();
  auto domain = window->getOperand(0);
  auto logicalRank = cast<MemRefType>(domain.getType()).getRank();
  auto coordinates = op->getOperands().drop_front(maskIndex + 1);
  auto *axis = coordinates[rank - 2].getDefiningOp();
  return axis && axis->getName().getStringRef() == "dsa.member_axis" &&
         axis->getOperand(0) == window->getResult(0) && axis->getOperand(1) == domain &&
         axis->getAttrOfType<IntegerAttr>("axis").getInt() == logicalRank - 2 &&
         coordinates.back() == window->getResult(1);
}
LogicalResult verifyProgram(ModuleOp m, const compilerinfra::TargetProfile &p,
                            bool supplied) {
  auto fs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(fs))
    return m.emitError("requires one tile function");
  auto fn = *fs.begin();
  if(failed(intentdsl::verifyDomain(fn)))return failure();
  auto d = fn->getAttrOfType<DictionaryAttr>(supplied ? "dsa.layout"
                                                      : "dsa.pending");
  auto caps = p.services.get<LocalSupply>();
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()))
    return fn.emitError("expected one defined tile region");
  if (auto err = intentdsl::verifyRows(fn,
          fn->getAttrOfType<ArrayAttr>("block.configs"),
          fn->getAttrOfType<StringAttr>("block.topology"),
          fn->getAttrOfType<DenseI64ArrayAttr>("block.guard"), p))
    return fn.emitError() << llvm::toString(std::move(err));
  Builder builder(m.getContext());
  if (fn->getAttr("block.grid") !=
      builder.getStrArrayAttr({"ceildiv", "M", "rows"}))
    return fn.emitError("row ownership requires the parent's instance grid");
  if (!supplied && failed(intentdsl::verifyMapping(m, p, "block")))
    return failure();
  if (!d || !caps || !d.getAs<ArrayAttr>("rows") ||
      d.getAs<ArrayAttr>("rows").empty() || !d.getAs<IntegerAttr>("depth"))
    return fn.emitError("missing supply rows or depth");
  auto verifyLayout = [&]() -> LogicalResult {
    auto expected = plan(fn, *caps, d.getAs<IntegerAttr>("depth").getInt(), d.getAs<IntegerAttr>("participants")?d.getAs<IntegerAttr>("participants").getInt():0,d.getAs<BoolAttr>("persistent") && d.getAs<BoolAttr>("persistent").getValue());
    if (!expected) {
      return fn.emitError() << llvm::toString(expected.takeError());
    }
    if (*expected != d)
      return fn.emitError("parameter rows or peak private allocation disagree "
                          "with live stages");
    return success();
  };
  auto sig = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!sig)
    return fn.emitError("missing access contract");
  auto parsed = llvm::json::parse(sig.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  auto *abi = parsed->getAsArray();
  if (!abi || abi->size() != fn.getNumArguments())
    return failure();
  llvm::SmallDenseSet<unsigned> written;
  auto coverage = fn.walk([&](Operation *op) {
    bool owner=op->getName().getStringRef()==(supplied?"dsa.owner_write":"block.owner_store");
    if (!owner && op->getName().getStringRef() !=
        (supplied ? "dsa.write_back" : "block.store"))
      return WalkResult::advance();
    auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
    auto view = dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto value = dyn_cast<RankedTensorType>(
        op->getOperand(owner?2:supplied ? 1 : 4).getType());
    if (!arg || arg.getOwner() != &fn.front() || !view || !value ||
        value.getElementType() != view.getElementType()) {
      op->emitError("write value and destination must have the same element type");
      return WalkResult::interrupt();
    }
    auto *slot = (*abi)[arg.getArgNumber()].getAsObject();
    auto direction = slot ? slot->getString("direction") : std::nullopt;
    if ((direction != "Out" && direction != "InOut") ||
        !written.insert(arg.getArgNumber()).second) {
      op->emitError("destination requires one full-domain writer");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (coverage.wasInterrupted())
    return failure();
  for (auto [i, entry] : llvm::enumerate(*abi)) {
    auto *slot = entry.getAsObject();
    if (!slot || (slot->getString("direction") == "Out" && !written.contains(i)))
      return fn.emitError("declared output lacks a full-domain writer");
  }
  if (!supplied)
    return verifyLayout();
  auto idx = fn->getAttrOfType<IntegerAttr>("kir.domain_arg");
  if (!idx || idx.getInt() < 0 || idx.getInt() >= fn.getNumArguments())
    return fn.emitError("missing member domain");
  bool resident = fn->getAttrOfType<StringAttr>("block.topology") == "resident";
  if (!resident) {
    auto &start = fn.front().front();
    auto *groups = start.getNextNode();
    if (fn.front().getOperations().size() != 3 ||
        start.getName().getStringRef() != "dsa.range" ||
        start.getAttrOfType<StringAttr>("source") != "participant_stride" ||
        !groups || groups->getName().getStringRef() != "dsa.task_groups" ||
        groups->getOperands() != start.getResults())
      return fn.emitError("stream group loop must consume start, bound and "
                          "stride of the participant range");
  }
  bool persistent=d.getAs<BoolAttr>("persistent") && d.getAs<BoolAttr>("persistent").getValue();
  unsigned count = 0;
  auto walk = fn.walk([&](Operation *o) {
    if (o->getName().getStringRef() != "dsa.phase")
      return WalkResult::advance();
    ++count;
    Builder b(m.getContext());
    auto depth = d.getAs<IntegerAttr>("depth").getInt();
    if (o->getAttr("slots") != b.getI32IntegerAttr(depth) ||
        o->getAttr("lookahead") != b.getI32IntegerAttr(depth - 1) ||
        o->getAttrOfType<StringAttr>("visibility") != "participant_private")
      return WalkResult::interrupt();
    for (auto &r : o->getRegions())
      if (!llvm::hasSingleElement(r) || r.front().empty())
        return WalkResult::interrupt();
    auto &fetch = o->getRegion(0).front(), &body = o->getRegion(1).front();
    auto axis = o->getAttrOfType<StringAttr>("axis");
    if (axis != (fn->getAttrOfType<StringAttr>("block.topology") == "resident"
                     ? "instances"
                     : "columns"))
      return WalkResult::interrupt();
    if (!legalSchedule(o->getRegion(2).front(), depth,
                       resident ? "participant_stride" : "tile_extent",persistent))
      return WalkResult::interrupt();
    for (auto *window : {&fetch.front(), &body.front()})
      if (window->getName().getStringRef() != "dsa.window" ||
          window->getOperand(0) != fn.getArgument(idx.getInt()))
        return WalkResult::interrupt();
    for (auto &op : body)
      if (op.getName().getStringRef() == "block.linear_index" &&
          (op.getOperand(0) != body.front().getOperand(0) ||
           op.getOperand(1) != body.front().getResult(0) ||
           op.getOperand(2) != body.front().getResult(1)))
        return op.emitError("ordinal detached from phase coordinates"),
               WalkResult::interrupt();
    auto windowType =
        cast<RankedTensorType>(body.front().getResult(2).getType());
    if (windowType.getRank() != 2 ||
        body.front().getResultTypes() != fetch.front().getResultTypes())
      return WalkResult::interrupt();
    SmallVector<Value> writes;
    for (auto &item : body) {
      auto name = item.getName().getStringRef();
      if(name=="dsa.owner_write" || name=="dsa.owner_read"){
        auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
        auto slot=dyn_cast<BlockArgument>(item.getOperand(0));
        if(!owner || !slot || slot.getOwner()!=&fn.front() ||
           item.getOperand(1)!=body.front().getResult(0))return WalkResult::interrupt();
        auto *desc=(*abi)[slot.getArgNumber()].getAsObject(),*dom=(*abi)[idx.getInt()].getAsObject();
        if(!desc || !dom || !desc->getArray("dims") || !dom->getArray("dims") ||
           desc->getArray("dims")->size()!=1 || size_t(owner.getInt())>=dom->getArray("dims")->size() ||
           (*desc->getArray("dims"))[0]!=(*dom->getArray("dims"))[owner.getInt()])
          return WalkResult::interrupt();
        if(name=="dsa.owner_write")writes.push_back(item.getResult(0));
      }
      if (name == "dsa.write_back") {
        if (!access(&item, 2, body.front(), fn, *abi) ||
            cast<RankedTensorType>(item.getOperand(1).getType()).getShape() != windowType.getShape()) {
          item.emitError("write must cover every active logical axis exactly once");
          return WalkResult::interrupt();
        }
        writes.push_back(item.getResult(0));
      }
      if ((name == "dsa.reduce" || name=="block.summary_reduce") &&
          (item.getOperand(1) != body.front().getResult(2) ||
           !isa<RankedTensorType>(item.getOperand(0).getType()) ||
           cast<RankedTensorType>(item.getOperand(0).getType()).getShape() !=
               windowType.getShape()))
        return WalkResult::interrupt();
    }
    auto &issued = fetch.back();
    if (issued.getName().getStringRef() != "dsa.issued" ||
        body.back().getName().getStringRef() != "dsa.complete" ||
        !body.back().getPrevNode() ||
        body.back().getPrevNode()->getName().getStringRef() != "dsa.drain" ||
        !llvm::equal(body.back().getPrevNode()->getOperands(), writes) ||
        body.back().getOperand(0) != body.back().getPrevNode()->getResult(0) ||
        body.getNumArguments() != issued.getNumOperands() + o->getNumResults())
      return WalkResult::interrupt();
    if (
        o->getNumOperands() != o->getNumResults() ||
        !llvm::equal(o->getResultTypes(),
                     body.back().getOperands().drop_front().getTypes()))
      return WalkResult::interrupt();
    auto transfer = std::next(fetch.begin());
    for (auto [i, arg] : llvm::enumerate(issued.getOperands())) {
      auto *def = arg.getDefiningOp();
      auto in = body.getArgument(i);
      while(transfer!=fetch.end() && (transfer->getName().getStringRef()=="dsa.member_axis" || transfer->getName().getStringRef()=="dsa.slab_axis"))++transfer;
      if (!def || transfer==fetch.end() || def != &*transfer++ || def->getBlock() != &fetch ||
          def->getName().getStringRef() != "dsa.copy_in" ||
          arg.getType() != in.getType() || !in.hasOneUse() ||
          in.use_begin()->getOwner()->getName().getStringRef() != "dsa.await")
        return WalkResult::interrupt();
      if (def->hasAttr("dsa.shared")!=(persistent && cast<MemRefType>(def->getOperand(0).getType()).getRank()==1))return WalkResult::interrupt();
      if (!access(def, 1, fetch.front(), fn, *abi))
        return WalkResult::interrupt();
    }
    if(transfer==fetch.end() || &*transfer!=&issued)return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (count == 0 || walk.wasInterrupted())
    return fn.emitError(
        "phase completion and staged fragment lifetimes differ");
  return verifyLayout();
}
} // namespace intentdsl::dsa
