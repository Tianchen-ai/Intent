#include "Intent/Interfaces/StructuredOpInterface.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#include "Intent/Interfaces/StructuredOpInterface.cpp.inc"

namespace intent {
namespace {

Block::BlockArgListType arguments(Region *region) {
  return region ? region->front().getArguments() : Block::BlockArgListType{};
}

ValueRange yields(Region *region) {
  return region ? ValueRange(region->front().getTerminator()->getOperands())
                : ValueRange{};
}

LogicalResult verifyRegion(Operation *owner, Region *region, StringRef role,
                           unsigned argumentCount, unsigned yieldCount) {
  if (!region || !llvm::hasSingleElement(*region))
    return owner->emitOpError() << role << " requires one block";
  Block &block = region->front();
  if (block.getNumArguments() != argumentCount)
    return owner->emitOpError() << role << " requires " << argumentCount
                               << " arguments, got " << block.getNumArguments();
  if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>() ||
      block.back().getNumOperands() != yieldCount)
    return owner->emitOpError() << role << " requires a terminator yielding "
                               << yieldCount << " values";
  return success();
}

} // namespace

SmallVector<Type> StructuredOpInterface::getCombineArgumentTypes() {
  SmallVector<Type> types;
  llvm::append_range(types, getIdentities().getTypes());
  llvm::append_range(types, getIdentities().getTypes());
  if (getStructuredKind() == StructuredOpKind::Reduce ||
      getStructuredKind() == StructuredOpKind::Scan)
    llvm::append_range(types, getCaptures().getTypes());
  return types;
}

SmallVector<Type> StructuredOpInterface::getSummarizeArgumentTypes(TypeRange sliceTypes) {
  assert(getSummarizeRegion() && sliceTypes.size() == getSources().size());
  SmallVector<Type> types(sliceTypes.begin(), sliceTypes.end());
  llvm::append_range(types, getCaptures().getTypes());
  return types;
}

SmallVector<Type> StructuredOpInterface::getApplyArgumentTypes() {
  assert(getApplyRegion());
  SmallVector<Type> types;
  llvm::append_range(types, getIdentities().getTypes());
  llvm::append_range(types, getInitialStates().getTypes());
  return types;
}

SmallVector<Type> StructuredOpInterface::getEmitArgumentTypes(TypeRange sliceTypes) {
  assert(getEmitRegion() && sliceTypes.size() == getSources().size());
  SmallVector<Type> types(sliceTypes.begin(), sliceTypes.end());
  llvm::append_range(types, getInitialStates().getTypes());
  llvm::append_range(types, getCaptures().getTypes());
  return types;
}

Block::BlockArgListType StructuredOpInterface::getCombineLhs() {
  return arguments(&getCombine()).take_front(getIdentities().size());
}
Block::BlockArgListType StructuredOpInterface::getCombineRhs() {
  return arguments(&getCombine()).slice(getIdentities().size(), getIdentities().size());
}
Block::BlockArgListType StructuredOpInterface::getCombineCaptures() {
  return arguments(&getCombine()).drop_front(2 * getIdentities().size());
}
Block::BlockArgListType StructuredOpInterface::getSummarizeSources() {
  auto *region = getSummarizeRegion();
  return region ? arguments(region).take_front(getSources().size()) : Block::BlockArgListType{};
}
Block::BlockArgListType StructuredOpInterface::getSummarizeCaptures() {
  auto *region = getSummarizeRegion();
  return region ? arguments(region).drop_front(getSources().size()) : Block::BlockArgListType{};
}
Block::BlockArgListType StructuredOpInterface::getApplySummaries() {
  auto *region = getApplyRegion();
  return region ? arguments(region).take_front(getIdentities().size()) : Block::BlockArgListType{};
}
Block::BlockArgListType StructuredOpInterface::getApplyStates() {
  auto *region = getApplyRegion();
  return region ? arguments(region).drop_front(getIdentities().size()) : Block::BlockArgListType{};
}
Block::BlockArgListType StructuredOpInterface::getEmitSources() {
  auto *region = getEmitRegion();
  return region ? arguments(region).take_front(getSources().size()) : Block::BlockArgListType{};
}
Block::BlockArgListType StructuredOpInterface::getEmitStates() {
  auto *region = getEmitRegion();
  return region ? arguments(region).slice(getSources().size(), getInitialStates().size())
                : Block::BlockArgListType{};
}
Block::BlockArgListType StructuredOpInterface::getEmitCaptures() {
  auto *region = getEmitRegion();
  return region ? arguments(region).drop_front(getSources().size() + getInitialStates().size())
                : Block::BlockArgListType{};
}
ValueRange StructuredOpInterface::getCombineYields() { return yields(&getCombine()); }
ValueRange StructuredOpInterface::getSummarizeYields() { return yields(getSummarizeRegion()); }
ValueRange StructuredOpInterface::getApplyYields() { return yields(getApplyRegion()); }
ValueRange StructuredOpInterface::getEmitYields() { return yields(getEmitRegion()); }

SmallVector<StructuredValueRelation>
StructuredOpInterface::getRegionArgumentRelations(Region &region) {
  SmallVector<StructuredValueRelation> relations;
  using K = StructuredRelationKind;
  auto connect = [&](ValueRange from, ValueRange to, K kind,
                     ArrayRef<int64_t> axes = {}) {
    for (auto [source, target] : llvm::zip_equal(from, to))
      relations.push_back({source, target, kind, {axes.begin(), axes.end()}});
  };
  auto axes = getIterationAxes();
  auto kind = getStructuredKind();
  auto identities = getIdentities();
  if (&region == &getCombine()) {
    connect(identities, getCombineLhs(), K::SameSchema);
    connect(identities, getCombineRhs(), K::SameSchema);
    if (kind == StructuredOpKind::Reduce || kind == StructuredOpKind::Scan)
      connect(getCaptures(), getCombineCaptures(), K::Capture);
  } else if (&region == getSummarizeRegion()) {
    connect(getSources(), getSummarizeSources(), K::SourceSlice, axes);
    connect(getCaptures(), getSummarizeCaptures(), K::Capture);
  } else if (&region == getApplyRegion()) {
    connect(identities, getApplySummaries(), K::SameSchema);
    connect(getInitialStates(), getApplyStates(), K::SameSchema);
  } else if (&region == getEmitRegion()) {
    connect(getSources(), getEmitSources(), K::SourceSlice, axes);
    connect(getInitialStates(), getEmitStates(), K::SameSchema);
    connect(getCaptures(), getEmitCaptures(), K::Capture);
  }
  return relations;
}

SmallVector<StructuredValueRelation> StructuredOpInterface::getValueRelations() {
  SmallVector<StructuredValueRelation> relations;
  for (Region &region : getOperation()->getRegions())
    llvm::append_range(relations, getRegionArgumentRelations(region));
  using K = StructuredRelationKind;
  auto connect = [&](ValueRange from, ValueRange to, K kind,
                     ArrayRef<int64_t> axes = {}) {
    for (auto [source, target] : llvm::zip_equal(from, to))
      relations.push_back({source, target, kind, {axes.begin(), axes.end()}});
  };
  auto axes = getIterationAxes();
  auto kind = getStructuredKind();
  auto identities = getIdentities();
  connect(getCombineYields(), identities, K::SameSchema);
  if (kind == StructuredOpKind::Reduce || kind == StructuredOpKind::Scan) {
    connect(getSources(), getOperation()->getResults(),
            kind == StructuredOpKind::Reduce ? K::Reduction : K::Prefix, axes);
    // KIR identities may be scalar/slice schemas, while GPU identities have
    // already been lifted to the full physical accumulator fragment.
    connect(identities, getOperation()->getResults(), K::Accumulator);
    return relations;
  }
  connect(getSummarizeYields(), identities, K::SameSchema);
  if (kind == StructuredOpKind::RegionFold) {
    connect(identities, getOperation()->getResults(), K::SameSchema);
    return relations;
  }
  connect(getApplyYields(), getFinalStates(), K::SameSchema);
  connect(getInitialStates(), getFinalStates(), K::SameSchema);
  connect(getEmitYields(), getEmittedResults(), K::Emission);
  return relations;
}

LogicalResult verifyStructuredArity(StructuredOpInterface operation) {
  Operation *owner = operation.getOperation();
  unsigned sources = operation.getSources().size();
  unsigned identities = operation.getIdentities().size();
  unsigned captures = operation.getCaptures().size();
  if (!sources || !identities)
    return owner->emitOpError("requires nonempty source and identity operand groups");
  auto kind = operation.getStructuredKind();
  if (kind == StructuredOpKind::Reduce || kind == StructuredOpKind::Scan) {
    if (sources != identities || owner->getNumResults() != identities)
      return owner->emitOpError("requires one identity and result per source component");
    return verifyRegion(owner, &operation.getCombine(), "combine",
                        2 * identities + captures, identities);
  }
  if (failed(verifyRegion(owner, operation.getSummarizeRegion(), "summarize",
                          sources + captures, identities)) ||
      failed(verifyRegion(owner, &operation.getCombine(), "combine",
                          2 * identities, identities)))
    return failure();
  if (kind == StructuredOpKind::RegionFold) {
    if (owner->getNumResults() != identities)
      return owner->emitOpError("requires one summary result per identity");
    return success();
  }
  unsigned states = operation.getInitialStates().size();
  unsigned outputs = operation.getEmittedResults().size();
  if (!states || !outputs || operation.getFinalStates().size() != states)
    return owner->emitOpError("requires nonempty emitted outputs and paired initial/final states");
  if (failed(verifyRegion(owner, operation.getApplyRegion(), "apply",
                          identities + states, states)))
    return failure();
  return verifyRegion(owner, operation.getEmitRegion(), "emit",
                      sources + states + captures, outputs);
}

} // namespace intent
