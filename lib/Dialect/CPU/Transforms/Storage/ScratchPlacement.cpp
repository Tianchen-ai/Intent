#include "ScratchStorage.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::cpu::detail {
namespace {

bool invariantCapacity(const ScratchAllocation &scratch, scf::ForOp loop,
                       DominanceInfo &dominance) {
  using Bounds = ValueBoundsConstraintSet;
  auto zero = IntegerAttr::get(IndexType::get(loop.getContext()), 0);
  for (Value operand : scratch.operation->getOperands()) {
    if (!dominance.properlyDominates(operand, loop) ||
        !Bounds::compare(Bounds::Variable(operand), Bounds::GE,
                         Bounds::Variable(zero)))
      return false;
  }
  return true;
}

bool fitsLoopScratch(ScratchSnapshot &snapshot, scf::ForOp loop,
                     int64_t additionalBytes, int64_t byteLimit) {
  if (additionalBytes > byteLimit) return false;
  int64_t bytes = additionalBytes;
  for (auto [number, candidate] : llvm::enumerate(snapshot.candidates())) {
    if (!isa<memref::AllocOp>(candidate.operation)) continue;
    auto *scratch = snapshot.get(number);
    if (!scratch) continue;
    Operation *anchor = scratch->operation->getBlock()->findAncestorOpInBlock(*loop.getOperation());
    if (!anchor || !scratch->operation->isBeforeInBlock(anchor) ||
        !anchor->isBeforeInBlock(scratch->release.getOperation())) continue;
    auto exclusive = snapshot.hasLoopOnlyDataUses(*scratch, loop);
    if (!exclusive) return false;
    if (!*exclusive) continue;
    // This pool contains current loop-exclusive backings, not owner state or
    // prepared inputs initialized outside the loop. Unknown pool capacity must
    // not become zero when a later fixed-point step extends another lifetime.
    auto capacity = scratchCapacity(*scratch, std::numeric_limits<int64_t>::max());
    if (!capacity) return false;
    auto reserved = scratchBytes(*scratch, *capacity);
    if (!reserved || llvm::AddOverflow(bytes, *reserved, bytes) || bytes > byteLimit)
      return false;
  }
  return true;
}

bool placeConditionalScratch(ScratchSnapshot &snapshot, int64_t byteLimit) {
  if (byteLimit <= 0) return false;
  struct Group {
    scf::ForOp loop;
    SmallVector<std::pair<ScratchAllocation *, int64_t>> allocations;
    int64_t bytes = 0;
    bool fits = true;
  };
  SmallVector<Group, 0> groups;
  for (auto [number, candidate] : llvm::enumerate(snapshot.candidates())) {
    if (!isa<memref::AllocOp>(candidate.operation)) continue;
    Operation *owner = candidate.operation->getParentOp();
    if (!isa<scf::IfOp>(owner)) continue;
    do {
      owner = owner->getParentOp();
    } while (isa_and_nonnull<scf::IfOp>(owner));
    auto loop = dyn_cast_or_null<scf::ForOp>(owner);
    if (!loop) continue;
    if (auto trips = constantTripCount(loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
        trips && *trips <= 1) continue;
    auto *scratch = snapshot.get(number);
    if (!scratch) continue;
    auto capacity = scratchCapacity(*scratch, std::numeric_limits<int64_t>::max());
    if (!capacity) continue;
    auto bytes = scratchBytes(*scratch, *capacity);
    if (!bytes) continue;
    auto group = llvm::find_if(groups, [&](const Group &group) { return group.loop == loop; });
    if (group == groups.end()) {
      groups.push_back({loop, {}, 0, true});
      group = std::prev(groups.end());
    }
    group->allocations.emplace_back(scratch, *capacity);
    int64_t total;
    if (!group->fits || llvm::AddOverflow(group->bytes, *bytes, total) || total > byteLimit)
      group->fits = false;
    else
      group->bytes = total;
  }
  for (auto &group : groups) {
    if (!group.fits || !fitsLoopScratch(snapshot, group.loop, group.bytes, byteLimit)) continue;
    // Crossing only the If chain would not reduce allocation frequency. Place
    // all proved backings at the serial-loop boundary, counting their full
    // overlapping capacities even when their original branches were exclusive.
    for (auto [scratch, capacity] : group.allocations) {
      OpBuilder before(group.loop);
      Value backing = createScratchBacking(before, *scratch, capacity,
                                            scratchAlignment(*scratch));
      OpBuilder at(scratch->operation);
      Value descriptor = scratchDescriptor(at, *scratch, backing);
      scratch->memory.replaceAllUsesWith(descriptor);
      OpBuilder after(group.loop);
      after.setInsertionPointAfter(group.loop);
      after.create<memref::DeallocOp>(scratch->operation->getLoc(), backing);
      scratch->release.erase();
      scratch->operation->erase();
    }
    return true;
  }
  return false;
}

} // namespace

bool placeScratchAllocations(ScratchSnapshot &snapshot, int64_t byteLimit,
                             ScratchRepresentation representation) {
  if (representation == ScratchRepresentation::LinearCapacity &&
      placeConditionalScratch(snapshot, byteLimit)) return true;
  auto candidates = snapshot.candidates();
  for (size_t number = candidates.size(); number > 0;) {
    Operation *operation = candidates[--number].operation;
    // The remaining path crosses only a direct serial-loop boundary, without
    // moving allocations through any other enclosing operation.
    auto loop = dyn_cast<scf::ForOp>(operation->getParentOp());
    if (!loop) continue;
    auto *scratch = snapshot.get(number);
    if (!scratch) continue;
    if (representation == ScratchRepresentation::LinearCapacity && !scratch->stack) {
      auto capacity = scratchCapacity(*scratch, std::numeric_limits<int64_t>::max());
      if (capacity) {
        auto bytes = scratchBytes(*scratch, *capacity);
        if (!bytes || !fitsLoopScratch(snapshot, loop, *bytes, byteLimit)) continue;
      }
    }
    if (invariantCapacity(*scratch, loop, snapshot.getDominance())) {
      operation->moveBefore(loop);
      if (scratch->release) scratch->release->moveAfter(loop);
      return true;
    }
    if (representation != ScratchRepresentation::LinearCapacity) continue;
    auto capacity = scratchCapacity(*scratch, byteLimit);
    if (!capacity) continue;
    OpBuilder before(loop);
    Value backing = createScratchBacking(before, *scratch, *capacity,
                                          scratchAlignment(*scratch));
    OpBuilder at(operation);
    Value descriptor = scratchDescriptor(at, *scratch, backing);
    scratch->memory.replaceAllUsesWith(descriptor);
    if (scratch->release) {
      OpBuilder after(loop);
      after.setInsertionPointAfter(loop);
      after.create<memref::DeallocOp>(operation->getLoc(), backing);
      scratch->release.erase();
    }
    operation->erase();
    return true;
  }
  return false;
}

} // namespace intent::cpu::detail
