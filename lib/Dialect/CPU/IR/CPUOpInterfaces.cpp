#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Visitors.h"

#include "Intent/Dialect/CPU/IR/CPUOpInterfaces.cpp.inc"

using namespace mlir;
using namespace intent::cpu;

FailureOr<SmallVector<Value>> RegionOpInterface::bindRegionArguments(
    Region &region, llvm::function_ref<ValueRange(RegionArgumentKind)> values) {
  auto schema = getRegionSchema(region);
  if (failed(schema)) return failure();
  SmallVector<Value> arguments;
  for (unsigned begin = 0; begin < schema->size();) {
    unsigned end = begin + 1;
    while (end < schema->size() && (*schema)[end].kind == (*schema)[begin].kind) ++end;
    ValueRange actual = values((*schema)[begin].kind);
    if (actual.size() != end - begin)
      return emitOpError("region argument binding does not match its declared group"), failure();
    llvm::append_range(arguments, actual);
    begin = end;
  }
  return arguments;
}

FailureOr<SmallVector<RegionArgumentSchema>>
intent::cpu::detail::regionArgumentSchema(Operation *operation, Region &region) {
  auto program = cast<RegionOpInterface>(operation);
  if (region.getParentOp() != operation || !llvm::hasSingleElement(region))
    return operation->emitOpError("region helper must have one owned block"), failure();
  struct Entry {
    Value prototype;
    RegionArgumentKind kind;
    std::optional<int64_t> sliceAxis;
  };
  SmallVector<Entry> entries;
  auto append = [&](ValueRange values, RegionArgumentKind kind,
                    std::optional<int64_t> axis = std::nullopt) {
    for (Value value : values) entries.push_back({value, kind, axis});
  };
  using K = RegionArgumentKind;
  bool valueForm = operation->getNumResults() != 0;
  if (auto scan = dyn_cast<RegionScanOp>(operation); scan && valueForm)
    if (scan.getEmittedResults().size() != scan.getEmittedOutputs().size() ||
        scan.getFinalStates().size() != scan.getFinalStateOutputs().size())
      return operation->emitOpError("region scan result groups must match their destination groups"), failure();
  if (&region == &program.getSummarize()) {
    append(program.getSources(), K::Sources, program.getAxis());
    append(program.getCaptures(), K::Captures);
    if (!valueForm) append(program.getIdentities(), K::Destinations);
  } else if (&region == &program.getCombine()) {
    append(program.getIdentities(), K::LeftSummary);
    append(program.getIdentities(), K::RightSummary);
    if (!valueForm) append(program.getIdentities(), K::Destinations);
  } else if (&region == program.getApplyRegion()) {
    append(program.getIdentities(), K::Summary);
    append(program.getInitialStates(), K::State);
    if (!valueForm) append(program.getInitialStates(), K::Destinations);
  } else if (&region == program.getEmitRegion()) {
    append(program.getSources(), K::Sources, program.getAxis());
    append(program.getInitialStates(), K::State);
    append(program.getCaptures(), K::Captures);
    auto axes = operation->getAttrOfType<DenseI64ArrayAttr>("output_axes");
    if (!axes || static_cast<size_t>(axes.size()) != program.getEmittedOutputs().size())
      return operation->emitOpError("each emitted output requires its source member axis"), failure();
    if (!valueForm)
      for (auto [output, axis] : llvm::zip(program.getEmittedOutputs(), axes.asArrayRef()))
        entries.push_back({output, K::Destinations, axis});
  } else {
    return operation->emitOpError("unrecognized region helper"), failure();
  }
  Block &body = region.front();
  if (body.getNumArguments() != entries.size())
    return operation->emitOpError("region helper argument groups are incomplete"), failure();
  SmallVector<RegionArgumentSchema> result;
  for (auto [argument, entry] : llvm::zip(body.getArguments(), entries))
    result.push_back({argument, entry.prototype, entry.kind, entry.sliceAxis});
  return result;
}

Block::BlockArgListType intent::cpu::detail::regionArguments(
    Operation *operation, Region &region, RegionArgumentKind kind) {
  auto schema = regionArgumentSchema(operation, region);
  assert(succeeded(schema) && "helper arguments require verified region schema");
  unsigned begin = 0;
  while (begin < schema->size() && (*schema)[begin].kind != kind) ++begin;
  unsigned end = begin;
  while (end < schema->size() && (*schema)[end].kind == kind) ++end;
  return region.front().getArguments().slice(begin, end - begin);
}

namespace {

Value borrowedRegionInput(RegionOpInterface program, BlockArgument argument) {
  auto schema = program.getRegionSchema(*argument.getOwner()->getParent());
  assert(succeeded(schema) && "buffer flow requires a verified helper schema");
  const auto &relation = (*schema)[argument.getArgNumber()];
  if (relation.kind == RegionArgumentKind::Sources ||
      relation.kind == RegionArgumentKind::Captures)
    return relation.prototype;
  // A state/summary prototype declares shape and dtype, not storage identity.
  // Its actual binding is formed by region realization, as are destinations.
  return {};
}

void populateRegionDependencies(
    RegionOpInterface program,
    bufferization::RegisterDependenciesFn registerDependencies) {
  for (Region &region : program->getRegions())
    for (BlockArgument argument : region.getArguments())
      if (isa<BaseMemRefType>(argument.getType()))
        if (Value source = borrowedRegionInput(program, argument))
          registerDependencies(source, argument);
}

void regionEffects(RegionOpInterface program,
                   SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Helper destination formals are separate scratch/state bindings, not aliases
  // of their schema prototypes. Effects at this boundary name actual operands.
  unsigned outputBegin =
      program->getNumOperands() - program.getDestinations().size();
  for (OpOperand &operand : program->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType()))
      continue;
    if (operand.getOperandNumber() >= outputBegin)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

Type slotType(Type type) {
  if (auto memory = dyn_cast<ShapedType>(type))
    return MemRefType::get(memory.getShape(), memory.getElementType());
  return MemRefType::get({}, type);
}

LogicalResult verifyRegionProgram(Operation *operation) {
  auto program = cast<RegionOpInterface>(operation);
  if (failed(verifyCollectiveOutputs(operation, program.getDestinations())))
    return failure();
  bool valueForm = operation->getNumResults() != 0;
  if (program.getSources().empty() || program.getIdentities().empty() ||
      (program.isScan() && (program.getInitialStates().empty() || program.getEmittedOutputs().empty())) ||
      program.getInitialValues().size() != program.getFinalDestinations().size())
    return operation->emitOpError("region source, identity, state and destination groups are incomplete");
  if (auto size = operation->getAttrOfType<IntegerAttr>("segment_size"))
    if (size.getInt() <= 0) return operation->emitOpError("region segment size must be positive");
  for (Value source : program.getSources()) {
    auto type = cast<ShapedType>(source.getType());
    if (isa<RankedTensorType>(source.getType()) != valueForm)
      return operation->emitOpError("region sources and destinations require the same value/buffer form");
    if (program.getAxis() >= static_cast<uint64_t>(type.getRank()))
      return operation->emitOpError("region source must contain its explicit member axis");
  }
  auto fields = operation->getAttrOfType<ArrayAttr>("summary_fields");
  if (!fields || fields.size() != program.getIdentities().size())
    return operation->emitOpError("region product fields must name every typed summary component");
  if (program.isScan()) {
    auto fields = operation->getAttrOfType<ArrayAttr>("state_fields");
    if (!fields || fields.size() != program.getInitialStates().size())
      return operation->emitOpError("region product fields must name every typed state component");
  }
  for (auto [value, output] : llvm::zip(program.getInitialValues(), program.getFinalDestinations()))
    if (slotType(value.getType()) !=
        (valueForm ? slotType(output.getType()) : output.getType()))
      return operation->emitOpError("identity/state and destination types must agree");
  if (valueForm) {
    auto finalResults = operation->getResults().take_back(program.getInitialValues().size());
    if (!llvm::equal(finalResults.getTypes(), program.getInitialValues().getTypes()))
      return operation->emitOpError("final values must preserve the declared summary/state component types");
  }
  for (Region &region : operation->getRegions()) {
    auto schema = program.getRegionSchema(region);
    if (failed(schema)) return failure();
    if (region.front().empty() || !isa<RegionYieldOp>(region.front().getTerminator()))
      return operation->emitOpError("region helpers require region_yield");
    for (const auto &relation : *schema) {
      Type expected = relation.prototype.getType();
      if (relation.sliceAxis) {
        auto memory = dyn_cast<ShapedType>(expected);
        int64_t axis = *relation.sliceAxis;
        if (!memory || axis < 0 || axis >= memory.getRank())
          return operation->emitOpError("helper slice axis is outside its source rank");
        SmallVector<int64_t> shape(memory.getShape());
        shape[axis] = ShapedType::kDynamic;
        if (valueForm)
          expected = RankedTensorType::get(shape, memory.getElementType());
        else
          expected = MemRefType::get(shape, memory.getElementType(),
              StridedLayoutAttr::get(operation->getContext(), ShapedType::kDynamic,
                  SmallVector<int64_t>(shape.size(), ShapedType::kDynamic)));
      } else if (!valueForm && relation.kind != RegionArgumentKind::Captures) {
        expected = slotType(expected);
      }
      if (relation.argument.getType() != expected)
        return operation->emitOpError("region helper argument type disagrees with its source/state/output schema");
    }
    unsigned destinations = llvm::count_if(*schema, [](const auto &argument) {
      return argument.kind == RegionArgumentKind::Destinations;
    });
    auto yield = cast<RegionYieldOp>(region.front().getTerminator());
    SmallVector<Type> yielded;
    if (valueForm) {
      if (&region == &program.getSummarize() || &region == &program.getCombine())
        llvm::append_range(yielded, program.getIdentities().getTypes());
      else if (&region == program.getApplyRegion())
        llvm::append_range(yielded, program.getInitialStates().getTypes());
      else {
        auto axes = operation->getAttrOfType<DenseI64ArrayAttr>("output_axes");
        for (auto [output, axis] : llvm::zip(program.getEmittedOutputs(), axes.asArrayRef())) {
          auto tensor = cast<RankedTensorType>(output.getType());
          if (axis < 0 || axis >= tensor.getRank())
            return operation->emitOpError("emission member axis is outside its destination rank");
          SmallVector<int64_t> shape(tensor.getShape());
          shape[axis] = ShapedType::kDynamic;
          yielded.push_back(RankedTensorType::get(shape, tensor.getElementType()));
        }
      }
    }
    if (!llvm::equal(yield.getOperandTypes(), yielded))
      return operation->emitOpError("region helper yields must match its value results or buffer destinations");
    if (failed(verifyCollectiveHelperEffects(operation, region, destinations)))
      return failure();
  }
  return success();
}
}

LogicalResult RegionFoldOp::verify() { return verifyRegionProgram(*this); }
LogicalResult RegionScanOp::verify() { return verifyRegionProgram(*this); }

void RegionFoldOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  regionEffects(cast<RegionOpInterface>(getOperation()), effects);
}

void RegionScanOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  regionEffects(cast<RegionOpInterface>(getOperation()), effects);
}

void RegionFoldOp::populateDependencies(
    bufferization::RegisterDependenciesFn registerDependencies) {
  populateRegionDependencies(cast<RegionOpInterface>(getOperation()),
                             registerDependencies);
}

bool RegionFoldOp::mayBeTerminalBuffer(Value value) {
  return !borrowedRegionInput(cast<RegionOpInterface>(getOperation()),
                             cast<BlockArgument>(value));
}

void RegionScanOp::populateDependencies(
    bufferization::RegisterDependenciesFn registerDependencies) {
  populateRegionDependencies(cast<RegionOpInterface>(getOperation()),
                             registerDependencies);
}

bool RegionScanOp::mayBeTerminalBuffer(Value value) {
  return !borrowedRegionInput(cast<RegionOpInterface>(getOperation()),
                             cast<BlockArgument>(value));
}
