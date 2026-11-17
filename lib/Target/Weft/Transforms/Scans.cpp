#include "Intent/Target/Weft/Transforms/Scans.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::weft_provider {

bool hasNativeScanCombine(cpu::ScanOp operation) {
  if (operation.isDestinationPassing() || !operation.getCaptures().empty())
    return false;
  return llvm::all_of(operation.getCombine().front().without_terminator(),
      [](Operation &nested) {
        return nested.getName().getDialectNamespace() == "arith" &&
            (isa<arith::ConstantOp>(nested) ||
             nested.hasTrait<OpTrait::Elementwise>()) &&
            nested.getNumResults() == 1;
      });
}

bool isNativeScanTile(cpu::ScanOp operation) {
  auto type = dyn_cast<MemRefType>(operation.getSources()[0].getType());
  auto binding = operation->getAttrOfType<cpu::ImplementationAttr>(
      "intent_cpu.implementation");
  return binding && type && type.getRank() == 1 && type.hasStaticShape() &&
      type.getDimSize(0) > 0 &&
      type.getDimSize(0) <= cpu::implementationParameter(binding, "panel") &&
      hasNativeScanCombine(operation);
}

LogicalResult tileScans(func::FuncOp function) {
  SmallVector<cpu::ScanOp> scans;
  function.walk([&](cpu::ScanOp scan) { scans.push_back(scan); });
  for (cpu::ScanOp scan : scans) {
    if (!hasNativeScanCombine(scan) || isNativeScanTile(scan)) continue;
    auto binding = scan->getAttrOfType<cpu::ImplementationAttr>(
        "intent_cpu.implementation");
    if (!binding) return scan.emitError("scan tiling requires a selected implementation");
    int64_t panel = cpu::implementationParameter(binding, "panel");
    if (panel <= 1) continue;
    OpBuilder b(scan);
    Location loc = scan.getLoc();
    auto type = cast<MemRefType>(scan.getSources()[0].getType());
    int64_t scanAxis = scan.getAxis();
    auto index = [&](int64_t number) -> Value {
      return b.create<arith::ConstantIndexOp>(loc, number);
    };
    Value zero = index(0), one = index(1), step = index(panel);
    SmallVector<Value> dimensions, position(type.getRank(), zero);
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      dimensions.push_back(b.createOrFold<memref::DimOp>(loc, scan.getSources()[0], axis));
    auto combine = [&](ValueRange left, ValueRange right) {
      IRMapping mapping;
      Block &body = scan.getCombine().front();
      for (auto [old, value] : llvm::zip(body.getArguments().take_front(left.size()), left))
        mapping.map(old, value);
      for (auto [old, value] : llvm::zip(body.getArguments().drop_front(left.size()), right))
        mapping.map(old, value);
      for (Operation &nested : body.without_terminator()) b.clone(nested, mapping);
      SmallVector<Value> result;
      for (Value value : body.getTerminator()->getOperands()) result.push_back(mapping.lookup(value));
      return result;
    };
    auto slice = [&](Value memory) -> Value {
      SmallVector<OpFoldResult> offsets, sizes(type.getRank(), b.getIndexAttr(1));
      SmallVector<OpFoldResult> strides(type.getRank(), b.getIndexAttr(1));
      for (Value coordinate : position) offsets.push_back(coordinate);
      sizes[scanAxis] = b.getIndexAttr(panel);
      auto sliced = memref::SubViewOp::inferRankReducedResultType({panel},
          cast<MemRefType>(memory.getType()), offsets, sizes, strides);
      return b.create<memref::SubViewOp>(loc, cast<MemRefType>(sliced), memory,
                                        offsets, sizes, strides);
    };
    std::function<void(int64_t)> visit = [&](int64_t axis) {
      if (axis < type.getRank()) {
        if (axis == scanAxis) { visit(axis + 1); return; }
        auto outer = b.create<scf::ForOp>(loc, zero, dimensions[axis], one);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(outer.getBody());
        position[axis] = outer.getInductionVar();
        visit(axis + 1);
        return;
      }
      Value extent = dimensions[scanAxis];
      Value end = b.create<arith::SubIOp>(loc, extent,
          b.create<arith::RemUIOp>(loc, extent, step));
      auto chunks = b.create<scf::ForOp>(loc, zero, end, step, scan.getInitials());
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(chunks.getBody());
        Value start = chunks.getInductionVar();
        if (scan.getReverse()) start = b.create<arith::SubIOp>(loc,
            b.create<arith::SubIOp>(loc, extent, step), start);
        position[scanAxis] = start;
        SmallVector<Value> sources, outputs;
        for (Value source : scan.getSources()) sources.push_back(slice(source));
        for (Value output : scan.getOutputs()) outputs.push_back(slice(output));
        Value last = index(scan.getReverse() ? 0 : panel - 1);
        SmallVector<Value> lastInputs;
        if (!scan.getInclusive())
          for (Value source : sources)
            lastInputs.push_back(b.create<memref::LoadOp>(loc, source, last));
        auto tile = b.create<cpu::ScanOp>(loc, TypeRange{}, sources,
            chunks.getRegionIterArgs(), ValueRange{}, outputs, 0,
            scan.getInclusive(), scan.getReverse());
        tile->setAttr("intent_cpu.implementation", binding);
        IRMapping mapping;
        scan.getCombine().cloneInto(&tile.getCombine(), mapping);
        SmallVector<Value> next;
        for (Value output : outputs)
          next.push_back(b.create<memref::LoadOp>(loc, output, last));
        if (!scan.getInclusive()) next = combine(next, lastInputs);
        b.create<scf::YieldOp>(loc, next);
      }
      auto tail = b.create<scf::ForOp>(loc, end, extent, one, chunks.getResults());
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(tail.getBody());
        Value ordinal = tail.getInductionVar();
        if (scan.getReverse()) ordinal = b.create<arith::SubIOp>(loc,
            b.create<arith::SubIOp>(loc, extent, one), ordinal);
        position[scanAxis] = ordinal;
        SmallVector<Value> inputs;
        for (Value source : scan.getSources())
          inputs.push_back(b.create<memref::LoadOp>(loc, source, position));
        auto next = combine(tail.getRegionIterArgs(), inputs);
        ValueRange result = scan.getInclusive() ? ValueRange(next) : tail.getRegionIterArgs();
        for (auto [output, value] : llvm::zip(scan.getOutputs(), result))
          b.create<memref::StoreOp>(loc, value, output, position);
        b.create<scf::YieldOp>(loc, next);
      }
    };
    visit(0);
    scan.erase();
  }
  return success();
}

} // namespace intent::weft_provider
