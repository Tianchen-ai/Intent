#include "Intent/Dialect/DSA/Analysis/UniformValues.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::dsa {
namespace {
bool isPaddingZero(Value value) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant))) return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant)) return integer.getValue().isZero();
  auto floating = dyn_cast<FloatAttr>(constant);
  return floating && floating.getValue().isZero() &&
         !floating.getValue().isNegative();
}
} // namespace

UniformMemoryAnalysis::UniformMemoryAnalysis(func::FuncOp function,
                                             StorageAnalysis &storage)
    : function(function), storage(storage), dominance(function) {}

bool UniformMemoryAnalysis::completeCounts(Value rows, Value columns,
                                           MemRefType shape) {
  auto rowCount = integerInterval(rows, function);
  auto columnCount = integerInterval(columns, function);
  return shape.getRank() == 2 && rowCount && columnCount &&
      rowCount->first == shape.getDimSize(0) && rowCount->second == rowCount->first &&
      columnCount->first == shape.getDimSize(1) && columnCount->second == columnCount->first;
}

Value UniformMemoryAnalysis::read(Value memory, Operation *reader) {
  Value scalar = readSnapshot(memory, reader);
  return scalar && dominance.properlyDominates(scalar, reader) ? scalar : Value{};
}

Value UniformMemoryAnalysis::readSnapshot(Value memory, Operation *reader) {
  auto type = dyn_cast<MemRefType>(memory.getType());
  if (!type || !isCompleteLocalStorageView(memory)) return {};
  Value origin = storage.uniqueOrigin(memory);
  if (!origin ||
      !origin.getDefiningOp<memref::AllocaOp>() ||
      !isCompleteStorageViewOf(memory, origin) || !storage.aliases(origin).complete ||
      !visiting.insert({memory, reader}).second) return {};
  Operation *writer = storage.lastWriterBefore(memory, reader);
  auto covers = [&](Value output) {
    return storage.uniqueOrigin(output) == origin &&
           isCompleteStorageViewOf(output, origin);
  };
  Value result;
  if (auto fill = dyn_cast_or_null<FillOp>(writer)) {
    if (covers(fill.getOutput())) result = fill.getValue();
  } else if (auto copy = dyn_cast_or_null<memref::CopyOp>(writer)) {
    if (covers(copy.getTarget()) && storage.disjoint(copy.getSource(), copy.getTarget()))
      result = readSnapshot(copy.getSource(), copy);
  } else if (auto load = dyn_cast_or_null<LoadTileOp>(writer)) {
    if (!load.getAsynchronous() && covers(load.getOutput()) &&
        storage.disjoint(load.getSource(), load.getOutput())) {
      Value scalar = readSnapshot(load.getSource(), load);
      if (scalar && (isPaddingZero(scalar) ||
          completeCounts(load.getRows(), load.getColumns(),
                         cast<MemRefType>(load.getOutput().getType())))) result = scalar;
    }
  } else if (auto transpose = dyn_cast_or_null<TransposeOp>(writer)) {
    if (covers(transpose.getOutput()) &&
        storage.disjoint(transpose.getInput(), transpose.getOutput())) {
      Value scalar = readSnapshot(transpose.getInput(), transpose);
      if (scalar && (isPaddingZero(scalar) ||
          completeCounts(transpose.getRows(), transpose.getColumns(),
                         cast<MemRefType>(transpose.getInput().getType())))) result = scalar;
    }
  }
  visiting.erase({memory, reader});
  return result && result.getType() == type.getElementType() ? result : Value{};
}

} // namespace intent::dsa
