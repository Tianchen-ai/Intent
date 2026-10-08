#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "Intent/Dialect/DSA/IR/ExecutionRelations.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
using namespace mlir;
using namespace intent::dsa;
#define GET_OP_CLASSES
#include "Intent/Dialect/DSA/IR/DSAOps.cpp.inc"
namespace {
bool tile(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 2 && type.hasStaticShape() &&
      type.getNumElements() > 0 && isStorageElementType(type.getElementType()) &&
      type.getMemorySpaceAsInt() == nramSpace;
}
bool same(Value a, Value b) { return a.getType() == b.getType() && tile(a); }
intent::BufferStorageAnalysis storageFor(Operation *operation) {
  return intent::BufferStorageAnalysis(
      operation->getParentOfType<func::FuncOp>(), storagePolicy());
}
}
LogicalResult TaskIdOp::verify() { return success(); }
LogicalResult SynchronizeOp::verify() { return success(); }
LogicalResult TaskCountOp::verify() { return success(); }
LogicalResult GroupIdOp::verify() { return success(); }
LogicalResult GroupCountOp::verify() { return success(); }
LogicalResult LocalIdOp::verify() { return success(); }
LogicalResult IsMemoryCoreOp::verify() { return success(); }
LogicalResult GroupSynchronizeOp::verify() { return success(); }
LogicalResult StageTileOp::verify() {
  auto source = cast<MemRefType>(getSource().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  if (source.getMemorySpaceAsInt() != 0 || output.getMemorySpaceAsInt() != sharedSpace ||
      output.getRank() != 2 || !output.hasStaticShape() || output.getNumElements() <= 0 ||
      !output.getLayout().isIdentity() || source.getElementType() != output.getElementType())
    return emitOpError("staging requires external input and a dense bounded shared tile of the same dtype");
  return success();
}
LogicalResult StrideOp::verify() {
  auto type = cast<MemRefType>(getSource().getType());
  return getAxis() < static_cast<uint64_t>(type.getRank()) ? success() : emitOpError("invalid resource axis");
}
LogicalResult LoadScalarOp::verify() {
  return cast<MemRefType>(getSource().getType()).getElementType() == getResult().getType()
      ? success() : emitOpError("scalar load changes the storage dtype");
}
LogicalResult StoreScalarOp::verify() {
  return cast<MemRefType>(getDestination().getType()).getElementType() == getValue().getType()
      ? success() : emitOpError("scalar store changes the storage dtype");
}
LogicalResult AtomicAddOp::verify() {
  auto source = cast<MemRefType>(getSource().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  Type element = source.getElementType();
  if (source.getMemorySpaceAsInt() != 0 || !element.isInteger(32) ||
      getValue().getType() != element || output.getElementType() != element ||
      !tile(getOutput()) || output.getNumElements() != 1)
    return emitOpError("atomic add requires an external i32 object and a one-element local old-value slot");
  return success();
}
LogicalResult LoadTileOp::verify() {
  auto sourceSpace = cast<MemRefType>(getSource().getType()).getMemorySpaceAsInt();
  if (getAsynchronous() && sourceSpace != 0 && sourceSpace != sharedSpace)
    return emitOpError("asynchronous tile loads require external or shared source storage");
  return tile(getOutput()) && cast<MemRefType>(getSource().getType()).getElementType() ==
      cast<MemRefType>(getOutput().getType()).getElementType()
      ? success() : emitOpError("load must preserve dtype into a static local tile");
}
LogicalResult GatherPlanOp::verify() {
  if (!tile(getRowOffsets()) || !tile(getLaneIndices()) || !tile(getOutput()))
    return emitOpError("run preparation requires local offsets, indices and workspace");
  auto offsets = cast<MemRefType>(getRowOffsets().getType());
  auto indices = cast<MemRefType>(getLaneIndices().getType());
  auto plan = cast<MemRefType>(getOutput().getType());
  int64_t rows = offsets.getDimSize(1);
  if (!offsets.getLayout().isIdentity() || !indices.getLayout().isIdentity() || !plan.getLayout().isIdentity() ||
      !offsets.getElementType().isInteger(64) || offsets.getDimSize(0) != 1 ||
      !indices.getElementType().isF32() || indices.getShape() != ArrayRef<int64_t>({1, rows}) ||
      !plan.getElementType().isInteger(64) || plan.getShape() != ArrayRef<int64_t>({3, rows}) ||
      rows < 64 || rows % 64 || rows > 65536)
    return emitOpError("run preparation needs i64[1,R], f32[1,R] and i64[3,R], R a multiple of 64 up to 65536");
  auto storage = storageFor(getOperation());
  if (!storage.disjoint(getRowOffsets(), getOutput()) ||
      !storage.disjoint(getLaneIndices(), getOutput()))
    return emitOpError("run workspace must be independent of its inputs");
  return success();
}
LogicalResult GatherRowsOp::verify() {
  if (!tile(getOutput()) || !tile(getRowOffsets()))
    return emitOpError("gather requires local output and row-offset tiles");
  auto source = cast<MemRefType>(getSource().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  auto offsets = cast<MemRefType>(getRowOffsets().getType());
  auto storage = storageFor(getOperation());
  if (getAsynchronous() && !getPlan()) return emitOpError("asynchronous gathers require an explicit run plan");
  if (getPlan()) {
    if (!tile(getPlan())) return emitOpError("run descriptor must use local storage");
    auto workspace = cast<MemRefType>(getPlan().getType());
    int64_t rows = output.getDimSize(0);
    if (!workspace.getLayout().isIdentity() || !workspace.getElementType().isInteger(64) ||
        workspace.getShape() != ArrayRef<int64_t>({3, rows}) || rows < 64 || rows % 64 || rows > 65536)
      return emitOpError("run descriptor needs i64[3,R], R a multiple of 64 up to 65536");
    SmallVector<Value> owners;
    for (Value value : {getRowOffsets(), getOutput(), getPlan()}) {
      if (llvm::any_of(owners, [&](Value other) {
            return !storage.disjoint(value, other);
          }))
        return emitOpError("run detection buffers must have independent owners");
      owners.push_back(value);
    }
  }
  return source.getMemorySpaceAsInt() == 0 && source.getElementType() == output.getElementType() &&
      output.getLayout().isIdentity() && offsets.getLayout().isIdentity() &&
      offsets.getElementType().isInteger(64) && offsets.getDimSize(0) == 1 &&
      offsets.getDimSize(1) == output.getDimSize(0) && storage.disjoint(getRowOffsets(), getOutput())
      ? success() : emitOpError("gather needs external storage and independent dense i64 offsets, one per local row");
}
LogicalResult StoreTileOp::verify() {
  return tile(getInput()) && cast<MemRefType>(getDestination().getType()).getElementType() ==
      cast<MemRefType>(getInput().getType()).getElementType()
      ? success() : emitOpError("store must preserve dtype from a static local tile");
}
LogicalResult GroupGatherRowsOp::verify() {
  if (!tile(getOutput()) || !tile(getRowOffsets()) || !tile(getPlan()))
    return emitOpError("group gather requires owned local output, offsets and run descriptor");
  auto source = cast<MemRefType>(getSource().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  auto offsets = cast<MemRefType>(getRowOffsets().getType());
  auto plan = cast<MemRefType>(getPlan().getType());
  auto data = cast<MemRefType>(getSharedData().getType());
  auto metadata = cast<MemRefType>(getSharedMetadata().getType());
  int64_t rows = output.getDimSize(0), columns = output.getDimSize(1);
  if (source.getMemorySpaceAsInt() != 0 || source.getElementType() != output.getElementType() ||
      !output.getLayout().isIdentity() || !offsets.getLayout().isIdentity() || !plan.getLayout().isIdentity() ||
      !offsets.getElementType().isInteger(64) || offsets.getShape() != ArrayRef<int64_t>({1, rows}) ||
      !plan.getElementType().isInteger(64) || plan.getShape() != ArrayRef<int64_t>({3, rows}) ||
      rows < 64 || rows % 64 || rows > 65536 ||
      data.getMemorySpaceAsInt() != sharedSpace || data.getElementType() != output.getElementType() ||
      !data.getLayout().isIdentity() || data.getShape() != ArrayRef<int64_t>({rows, 4 * columns}) ||
      metadata.getMemorySpaceAsInt() != sharedSpace || !metadata.getElementType().isInteger(64) ||
      !metadata.getLayout().isIdentity() || metadata.getShape() != ArrayRef<int64_t>({4, rows}))
    return emitOpError("group gather requires dense local and four-owner shared tiles with a complete descriptor");
  SmallVector<Value> owners;
  auto storage = storageFor(getOperation());
  for (Value value : {getOutput(), getRowOffsets(), getPlan(), getSharedData(), getSharedMetadata()}) {
    Value origin = storage.uniqueOrigin(value);
    if (!origin || !origin.getDefiningOp<memref::AllocaOp>() ||
        llvm::any_of(owners, [&](Value other) { return !storage.disjoint(value, other); }))
      return emitOpError("group gather storage must have independent allocation owners");
    owners.push_back(value);
  }
  return success();
}
LogicalResult IndexLayoutOp::verify() {
  if (!tile(getOutput())) return emitOpError("index layout requires a dense local output");
  auto output = cast<MemRefType>(getOutput().getType());
  if (!output.getLayout().isIdentity()) return emitOpError("index layout output must be contiguous");
  if (getInput().getType().isInteger(64))
    return output.getElementType().isInteger(32) && output.getDimSize(0) == 2 ? success() :
        emitOpError("scalar index splat requires two i32 planes");
  if (!tile(getInput())) return emitOpError("index layout input must be an i64 scalar or local tile");
  auto input = cast<MemRefType>(getInput().getType());
  bool split = input.getElementType().isInteger(64) && input.getDimSize(0) == 1 &&
      output.getElementType().isInteger(32) && output.getDimSize(0) == 2;
  bool join = output.getElementType().isInteger(64) && output.getDimSize(0) == 1 &&
      input.getElementType().isInteger(32) && input.getDimSize(0) == 2;
  return input.getLayout().isIdentity() && input.getDimSize(1) == output.getDimSize(1) && (split || join) ?
      success() : emitOpError("index layout converts i64[1,N] and i32[2,N]");
}
LogicalResult BroadcastRowsOp::verify() {
  if (!tile(getInput()) || !tile(getOutput()) || !tile(getScratch()))
    return emitOpError("row broadcasting requires local input, output and staging storage");
  auto input = cast<MemRefType>(getInput().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  auto scratch = cast<MemRefType>(getScratch().getType());
  auto storage = storageFor(getOperation());
  return input.getLayout().isIdentity() && output.getLayout().isIdentity() && scratch.getLayout().isIdentity() &&
      input.getElementType() == output.getElementType() && scratch.getElementType() == output.getElementType() &&
      scratch.getDimSize(0) == output.getDimSize(1) && scratch.getDimSize(1) == output.getDimSize(0) &&
      storage.disjoint(getScratch(), getInput()) && storage.disjoint(getScratch(), getOutput()) ? success() :
      emitOpError("row broadcasting needs matching dtypes and independent transposed staging storage");
}
LogicalResult IndexBinaryOp::verify() {
  if (!same(getLhs(), getOutput())) return emitOpError("index arithmetic needs equally shaped local word planes");
  auto type = cast<MemRefType>(getOutput().getType());
  bool scalar = getRhs().getType().isInteger(64);
  if (!type.getLayout().isIdentity() || !type.getElementType().isInteger(32) || type.getDimSize(0) != 2 ||
      (!scalar && !same(getRhs(), getLhs()))) return emitOpError("index arithmetic requires i32[2,N] and a matching tile or i64 scalar");
  auto storage = storageFor(getOperation());
  if (!storage.disjoint(getOutput(), getLhs()) || (!scalar && !storage.disjoint(getOutput(), getRhs())))
    return emitOpError("index arithmetic output must have independent storage");
  if (getKind() == intent::BinaryOperator::LeftShift || getKind() == intent::BinaryOperator::RightShift) {
    auto shift = getRhs().getDefiningOp<arith::ConstantIntOp>();
    return scalar && shift && shift.value() >= 0 && shift.value() < 64 ? success() :
        emitOpError("index shifts require a constant shift in [0,64)");
  }
  return getKind() == intent::BinaryOperator::Add || getKind() == intent::BinaryOperator::Subtract ||
      getKind() == intent::BinaryOperator::Multiply ? success() : emitOpError("unsupported exact index arithmetic");
}
LogicalResult CompareRangeOp::verify() {
  auto rows = cast<MemRefType>(getRowCoordinates().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  return tile(getRowCoordinates()) && tile(getOutput()) && rows.getLayout().isIdentity() &&
      output.getLayout().isIdentity() && rows.getElementType().isInteger(64) &&
      rows.getDimSize(1) == 1 && output.getDimSize(0) == rows.getDimSize(0) &&
      output.getElementType().isInteger(1) && output.getDimSize(1) > 0 &&
      getBaseAttr().getInt() <= INT64_MAX - (output.getDimSize(1) - 1)
      ? success() : emitOpError("range comparison requires i64 row coordinates and a nonoverflowing unit-step column range");
}
LogicalResult IotaOp::verify() {
  auto type = cast<MemRefType>(getOutput().getType());
  return tile(getOutput()) && type.getLayout().isIdentity() && type.getDimSize(0) == 1 &&
      type.getDimSize(1) > 0 && type.getDimSize(1) <= (1 << 24) &&
      (type.getElementType().isF32() || type.getElementType().isInteger(64))
      ? success() : emitOpError("lane indices require a bounded dense f32 or i64 row");
}
LogicalResult CompareRampOp::verify() {
  if (!tile(getOutput())) return emitOpError("ramp mask requires local output");
  auto output = cast<MemRefType>(getOutput().getType());
  int64_t columns = output.getDimSize(1), count = output.getNumElements();
  if (!output.getLayout().isIdentity() || columns < 64 || (columns & (columns - 1)) ||
      count > 65536 || getBaseAttr().getInt() > INT64_MAX - (columns - 1))
    return emitOpError("ramp mask requires bounded power-of-two columns");
  if (output.getElementType().isInteger(32))
    return !getScratch() ? success() : emitOpError("full-width ramp masks need no conversion workspace");
  if (!getScratch() || !tile(getScratch())) return emitOpError("i1 ramp mask requires local conversion workspace");
  auto scratch = cast<MemRefType>(getScratch().getType());
  auto storage = storageFor(getOperation());
  return output.getLayout().isIdentity() && scratch.getLayout().isIdentity() &&
      output.getElementType().isInteger(1) && scratch.getElementType().isF32() &&
      columns >= 64 && (columns & (columns - 1)) == 0 && count <= 65536 &&
      scratch.getShape() == ArrayRef<int64_t>({1, count}) && storage.disjoint(getScratch(), getOutput()) &&
      getBaseAttr().getInt() <= INT64_MAX - (columns - 1)
      ? success() : emitOpError("ramp mask requires bounded power-of-two columns and a full f32 workspace row");
}
LogicalResult FillOp::verify() {
  return tile(getOutput()) && cast<MemRefType>(getOutput().getType()).getElementType() == getValue().getType()
      ? success() : emitOpError("fill value and local tile must have the same dtype");
}
LogicalResult TransposeOp::verify() {
  if (!tile(getInput()) || !tile(getOutput())) return emitOpError("transpose requires local tiles");
  auto input = cast<MemRefType>(getInput().getType()), output = cast<MemRefType>(getOutput().getType());
  auto storage = storageFor(getOperation());
  return input.getElementType() == output.getElementType() && input.getLayout().isIdentity() &&
      output.getLayout().isIdentity() && input.getDimSize(0) == output.getDimSize(1) &&
      input.getDimSize(1) == output.getDimSize(0) && storage.disjoint(getInput(), getOutput())
      ? success() : emitOpError("transpose requires disjoint dense local tiles with reversed extents");
}
LogicalResult SelectOp::verify() {
  auto predicate = dyn_cast<MemRefType>(getCondition().getType());
  bool condition = getCondition().getType().isInteger(1) || (tile(getCondition()) &&
      predicate.getElementType().isInteger(1) && predicate.getShape() == cast<MemRefType>(getOutput().getType()).getShape());
  if (Value scratch = getScratch()) {
    auto data = cast<MemRefType>(getOutput().getType());
    if (!predicate || !data.getElementType().isF32() || !tile(scratch))
      return emitOpError("selection workspace requires an f32 tile and tensor predicate");
    auto type = cast<MemRefType>(scratch.getType());
    int64_t rows = isa<MemRefType>(getFalseValue().getType()) ? 2 : 1;
    if (!type.getElementType().isInteger(32) || type.getDimSize(0) != rows || type.getDimSize(1) % 64 ||
        type.getDimSize(1) > data.getNumElements())
      return emitOpError("selection workspace requires bounded 64-lane i32 groups for the selected operand form");
    auto storage = storageFor(getOperation());
    Value inputs[] = {getCondition(), getTrueValue(), getFalseValue(), getOutput()};
    for (Value input : inputs)
      if (isa<BaseMemRefType>(input.getType()) && !storage.disjoint(input, scratch))
        return emitOpError("selection workspace must be independent of data and predicate");
  }
  bool falseValue = same(getTrueValue(), getFalseValue()) ||
      getFalseValue().getType() == cast<MemRefType>(getOutput().getType()).getElementType();
  return condition && falseValue && same(getTrueValue(), getOutput())
      ? success() : emitOpError("selection requires a matching false tile or scalar and a boolean scalar or tile");
}
LogicalResult MaskedFillOp::verify() {
  if (!tile(getMask()) || !same(getInput(), getOutput())) return emitOpError("masked fill requires matching local data tiles");
  auto mask = cast<MemRefType>(getMask().getType()), data = cast<MemRefType>(getOutput().getType());
  auto storage = storageFor(getOperation());
  return mask.getElementType().isInteger(32) && data.getElementType().isF32() &&
      mask.getShape() == data.getShape() && mask.getLayout().isIdentity() && data.getLayout().isIdentity() &&
      data.getNumElements() % 64 == 0 && storage.disjoint(getMask(), getOutput()) && storage.disjoint(getMask(), getInput())
      ? success() : emitOpError("masked fill requires an independent dense i32 predicate and f32 data in 64-lane groups");
}
LogicalResult UnaryOp::verify() {
  if (!same(getInput(), getOutput())) return emitOpError("unary tiles must match");
  if (Value scratch = getScratch()) {
    auto input = cast<MemRefType>(getInput().getType());
    if ((getKind() == UnaryOperator::Exp && !getApproximate() && !getFlushToZero()) ||
        (getKind() == UnaryOperator::Exp2 && getApproximate() && getFlushToZero() &&
         cast<MemRefType>(scratch.getType()).getElementType().isF32())) {
      auto workspace = cast<MemRefType>(scratch.getType());
      auto storage = storageFor(getOperation());
      return tile(scratch) && input.getElementType().isF32() && workspace.getElementType().isF32() &&
          workspace.getLayout().isIdentity() && workspace.getDimSize(0) == 4 && workspace.getDimSize(1) >= 4 &&
          workspace.getDimSize(1) <= input.getNumElements() && storage.disjoint(scratch, getInput()) &&
          storage.disjoint(scratch, getOutput()) && storage.disjoint(getInput(), getOutput())
          ? success() : emitOpError("exp workspace requires four independent f32 rows and disjoint input/output");
    }
    if (!tile(scratch) || !input.getElementType().isF32() || getKind() != UnaryOperator::Exp2 ||
        !getApproximate() || !getFlushToZero()) return emitOpError("unary workspace requires f32 exp2 with explicit FTZ");
    auto type = cast<MemRefType>(scratch.getType());
    auto storage = storageFor(getOperation());
    if (!type.getElementType().isInteger(32) || type.getDimSize(0) != 1 ||
        type.getNumElements() > input.getNumElements() || !storage.disjoint(scratch, getInput()) ||
        !storage.disjoint(scratch, getOutput())) return emitOpError("FTZ workspace requires an independent bounded i32 row");
  }
  return success();
}
LogicalResult BinaryOp::verify() {
  if (!same(getLhs(), getOutput())) return emitOpError("binary input and output tiles must match");
  if (Value scratch = getScratch()) {
    if (getKind() == BinaryOperator::TrueDivide && getApproximate() && getFlushToZero()) {
      if (!tile(scratch) || !same(getLhs(), getRhs()))
        return emitOpError("reciprocal workspace requires matching f32 input tiles");
      auto input = cast<MemRefType>(getLhs().getType());
      auto workspace = cast<MemRefType>(scratch.getType());
      auto storage = storageFor(getOperation());
      return input.getElementType().isF32() && workspace.getLayout().isIdentity() &&
          workspace.getElementType().isInteger(32) && workspace.getDimSize(0) == 2 &&
          workspace.getDimSize(1) >= 1 && workspace.getDimSize(1) <= input.getNumElements() &&
          storage.disjoint(scratch, getLhs()) && storage.disjoint(scratch, getRhs()) &&
          storage.disjoint(scratch, getOutput()) && storage.disjoint(getOutput(), getRhs())
          ? success() : emitOpError("reciprocal requires two independent bounded i32 workspace rows");
    }
    if (!tile(scratch) || !same(getLhs(), getRhs()) ||
        (getKind() != BinaryOperator::MaximumNum && getKind() != BinaryOperator::MinimumNum &&
         getKind() != BinaryOperator::Maximum && getKind() != BinaryOperator::Minimum))
      return emitOpError("destructive extrema workspace requires matching input tiles");
    auto input = cast<MemRefType>(getLhs().getType());
    auto workspace = cast<MemRefType>(scratch.getType());
    auto storage = storageFor(getOperation());
    if ((!input.getElementType().isF16() && !input.getElementType().isF32()) ||
        workspace.getElementType() != input.getElementType() || !workspace.getLayout().isIdentity() ||
        workspace.getDimSize(0) != 1 || workspace.getNumElements() > input.getNumElements() ||
        !storage.disjoint(scratch, getLhs()) || !storage.disjoint(scratch, getRhs()) || !storage.disjoint(scratch, getOutput()))
      return emitOpError("numeric extrema workspace must be an independent bounded f16/f32 row");
  }
  auto lhs = cast<MemRefType>(getLhs().getType());
  auto rhs = dyn_cast<MemRefType>(getRhs().getType());
  bool broadcast = tile(getRhs()) && rhs.getElementType() == lhs.getElementType() &&
      (lhs.getElementType().isF32() || lhs.getElementType().isF16()) &&
      (rhs.getShape() == ArrayRef<int64_t>({lhs.getDimSize(0), 1}) ||
       rhs.getShape() == ArrayRef<int64_t>({1, lhs.getDimSize(1)})) &&
      lhs.getLayout().isIdentity() && rhs.getLayout().isIdentity() && !getApproximate() && !getFlushToZero() &&
      (getKind() == intent::BinaryOperator::Add || getKind() == intent::BinaryOperator::Subtract || getKind() == intent::BinaryOperator::Multiply);
  return broadcast || same(getLhs(), getRhs()) ||
      getRhs().getType() == cast<MemRefType>(getLhs().getType()).getElementType()
      ? success() : emitOpError("binary right operand must be a matching tile, scalar, or same-dtype unit-axis broadcast");
}
LogicalResult CompareOp::verify() {
  if (Value scratch = getScratch()) {
    if (!tile(scratch) || !tile(getLhs()) || !tile(getOutput()))
      return emitOpError("comparison workspace requires local tiles");
    auto workspace = cast<MemRefType>(scratch.getType());
    auto input = cast<MemRefType>(getLhs().getType());
    if (!input.getElementType().isInteger(64) || !input.getLayout().isIdentity() ||
        !cast<MemRefType>(getOutput().getType()).getLayout().isIdentity() ||
        !workspace.getLayout().isIdentity() || !workspace.getElementType().isInteger(32) ||
        workspace.getDimSize(0) != 6 || workspace.getDimSize(1) < 32 ||
        workspace.getDimSize(1) > 65536 || workspace.getDimSize(1) % 32)
      return emitOpError("split-word comparison needs dense i64 operands and i32[6,W] workspace, W a multiple of 32 up to 65536");
    auto storage = storageFor(getOperation());
    if (!storage.disjoint(scratch, getLhs()) || !storage.disjoint(scratch, getRhs()) ||
        !storage.disjoint(scratch, getOutput()))
      return emitOpError("comparison workspace must be independent of operands and output");
  }
  return same(getLhs(), getRhs()) && tile(getOutput()) &&
      cast<MemRefType>(getLhs().getType()).getShape() == cast<MemRefType>(getOutput().getType()).getShape() &&
      cast<MemRefType>(getOutput().getType()).getElementType().isInteger(1)
      ? success() : emitOpError("comparison requires matching input tiles and a boolean output tile");
}
LogicalResult CastOp::verify() {
  return tile(getInput()) && tile(getOutput()) &&
      cast<MemRefType>(getInput().getType()).getShape() == cast<MemRefType>(getOutput().getType()).getShape()
      ? success() : emitOpError("cast must preserve local tile shape");
}
LogicalResult DivideRNOp::verify() {
  auto analysis = storageFor(getOperation());
  if (!same(getLhs(), getOutput()) || !cast<MemRefType>(getLhs().getType()).getElementType().isF32() ||
      (!getRhs().getType().isF32() && !same(getLhs(), getRhs())) || !tile(getScratch()) || !tile(getLaneIndices()))
    return emitOpError("rounded division requires f32 tiles and owned correction workspace");
  auto scratch = cast<MemRefType>(getScratch().getType());
  auto indices = cast<MemRefType>(getLaneIndices().getType());
  if (!scratch.getElementType().isInteger(32) || scratch.getDimSize(0) != 12 ||
      !scratch.getLayout().isIdentity() || !indices.getLayout().isIdentity() ||
      !indices.getElementType().isF32() || indices.getDimSize(0) != 1 ||
      indices.getDimSize(1) != scratch.getDimSize(1) || scratch.getDimSize(1) > (1 << 24))
    return emitOpError("rounded division needs twelve integer rows and an exact lane-index row");
  SmallVector<Value> storage;
  for (Value value : {getOutput(), getScratch(), getLaneIndices()}) {
    Value allocation = analysis.uniqueOrigin(value);
    if (!allocation || !allocation.getDefiningOp<memref::AllocaOp>() ||
        !analysis.disjoint(value, getLhs()) ||
        (isa<BaseMemRefType>(getRhs().getType()) && !analysis.disjoint(value, getRhs())) ||
        llvm::any_of(storage, [&](Value other) { return !analysis.disjoint(value, other); }))
      return emitOpError("division result, correction workspace and indices must have independent owners");
    storage.push_back(value);
  }
  return success();
}
LogicalResult DivideCastOp::verify() {
  auto storage = storageFor(getOperation());
  auto owner = [&](Value value) {
    Value origin = storage.uniqueOrigin(value);
    return origin && origin.getDefiningOp<memref::AllocaOp>() ? origin : Value{};
  };
  Value lhsOwner = owner(getLhs()), rhsOwner = owner(getRhs()), indicesOwner = owner(getLaneIndices());
  if (!lhsOwner || !tile(getLhs()) || (!rhsOwner && !getRhs().getType().isF32()))
    return emitOpError("divide/cast requires owned local inputs or a uniform f32 divisor");
  auto input = cast<MemRefType>(getLhs().getType());
  auto indices = cast<MemRefType>(getLaneIndices().getType());
  if (!indicesOwner || !tile(getLaneIndices()) || !indices.getElementType().isF32() ||
      indices.getDimSize(0) != 1 || indices.getNumElements() != input.getNumElements() ||
      input.getNumElements() > (1 << 24))
    return emitOpError("divide/cast needs an exact f32 lane-index row covering the input tile");
  SmallVector<Value> written;
  for (Value value : {getOutput(), getQuotient(), getBounds(), getAccepted(), getNarrowBounds()}) {
    Value allocation = owner(value);
    if (!allocation || !storage.disjoint(value, getLhs()) ||
        (rhsOwner && !storage.disjoint(value, getRhs())) ||
        !storage.disjoint(value, getLaneIndices()) ||
        llvm::any_of(written, [&](Value other) { return !storage.disjoint(value, other); }))
      return emitOpError("divide/cast output and workspace must have independent owned storage");
    written.push_back(value);
  }
  auto output = cast<MemRefType>(getOutput().getType());
  return (getRhs().getType().isF32() || same(getLhs(), getRhs())) && same(getLhs(), getQuotient()) &&
      same(getLhs(), getBounds()) && same(getLhs(), getAccepted()) &&
      same(getOutput(), getNarrowBounds()) && input.getElementType().isF32() &&
      output.getElementType().isF16() && input.getShape() == output.getShape()
      ? success() : emitOpError("divide/cast requires f32 input/workspace and matching f16 output/workspace");
}
LogicalResult ReduceOp::verify() {
  auto input = cast<MemRefType>(getInput().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  if (input.getElementType().isF16() && output.getElementType().isF32()) {
    auto scratch = cast<MemRefType>(getScratch().getType());
    auto storage = storageFor(getOperation());
    return tile(getInput()) && tile(getOutput()) && tile(getScratch()) && getAxis() == 1 &&
        (getKind() == intent::BinaryOperator::MaximumNum || getKind() == intent::BinaryOperator::MinimumNum) &&
        input.getDimSize(0) < 32 && input.getDimSize(1) >= 4 &&
        input.getLayout().isIdentity() && output.getLayout().isIdentity() && scratch.getLayout().isIdentity() &&
        scratch.getElementType().isF16() && scratch.getShape() == ArrayRef<int64_t>({1, input.getDimSize(1)}) &&
        output.getShape() == ArrayRef<int64_t>({1, input.getDimSize(0)}) && getIdentity().getType().isF32() &&
        storage.disjoint(getInput(), getOutput()) && storage.disjoint(getInput(), getScratch()) && storage.disjoint(getOutput(), getScratch())
        ? success() : emitOpError("widened numeric extrema require f16 rows, independent f16 workspace and f32 result/identity");
  }
  bool shape = getAxis() == 1 && same(getInput(), getScratch()) && input.getDimSize(0) == 1 &&
      output.getNumElements() == 1;
  if (getAxis() == 0 || (getAxis() == 1 && input.getDimSize(0) > 1)) {
    auto storage = storageFor(getOperation());
    auto scratch = cast<MemRefType>(getScratch().getType());
    bool workspace = same(getOutput(), getScratch());
    if (getAxis() == 1 && getKind() != intent::BinaryOperator::Add)
      workspace = tile(getScratch()) && scratch.getLayout().isIdentity() && scratch.getDimSize(0) == 1 &&
          scratch.getElementType() == input.getElementType() &&
          scratch.getDimSize(1) == input.getNumElements() + input.getDimSize(0);
    if (getAxis() == 1 && input.getDimSize(0) > 1 && input.getDimSize(0) < 32 &&
        input.getDimSize(1) >= 1024 && input.getDimSize(1) % 32 == 0)
      workspace |= tile(getScratch()) && scratch.getLayout().isIdentity() &&
          scratch.getElementType() == input.getElementType() &&
          scratch.getShape() == ArrayRef<int64_t>({1, input.getDimSize(1)});
    shape = tile(getInput()) && workspace && output.getDimSize(0) == 1 &&
        output.getDimSize(1) == input.getDimSize(1 - getAxis()) && input.getLayout().isIdentity() &&
        output.getLayout().isIdentity() && storage.disjoint(getInput(), getOutput()) &&
        storage.disjoint(getInput(), getScratch()) && storage.disjoint(getOutput(), getScratch());
  }
  return shape && tile(getOutput()) && input.getElementType() == output.getElementType() &&
      getIdentity().getType() == output.getElementType()
      ? success() : emitOpError("reduce requires matching local rows or columns, explicit workspace and identity");
}
LogicalResult PrepareMatrixViewOp::verify() {
  auto source = cast<MemRefType>(getSource().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  auto input = cast<MemRefType>(getInputSlice().getType());
  auto transpose = cast<MemRefType>(getTransposedSlice().getType());
  auto storage = storageFor(getOperation());
  return source.getMemorySpaceAsInt() == 0 && output.getMemorySpaceAsInt() == matrixSpace &&
      output.hasStaticShape() && output.getRank() == 2 && output.getLayout().isIdentity() &&
      output.getDimSize(0) > 0 && output.getDimSize(1) > 0 && output.getDimSize(1) % 64 == 0 &&
      tile(getInputSlice()) && tile(getTransposedSlice()) && input.getLayout().isIdentity() &&
      transpose.getLayout().isIdentity() && input.getDimSize(0) > 0 && input.getDimSize(1) % 64 == 0 &&
      output.getDimSize(0) % input.getDimSize(0) == 0 && output.getDimSize(1) % input.getDimSize(1) == 0 &&
      transpose.getShape() == ArrayRef<int64_t>({input.getDimSize(1), input.getDimSize(0)}) &&
      source.getElementType() == output.getElementType() && source.getElementType() == input.getElementType() &&
      source.getElementType() == transpose.getElementType() &&
      storage.disjoint(getInputSlice(), getTransposedSlice())
      ? success() : emitOpError("resident matrix preparation requires a full-K WRAM panel and independent local slices in complete 64-channel groups");
}
LogicalResult PrepareMatrixOp::verify() {
  auto input = cast<MemRefType>(getInput().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  auto storage = storageFor(getOperation());
  if (!tile(getInput()) || !input.getLayout().isIdentity() || !output.hasStaticShape() ||
      output.getRank() != 2 || output.getMemorySpaceAsInt() != matrixSpace ||
      input.getElementType() != output.getElementType())
    return emitOpError("matrix preparation requires dense local input and matrix-local output");
  int64_t k = output.getDimSize(0), n = output.getDimSize(1);
  auto workspace = [&](Value value) {
    if (!value || !tile(value)) return false;
    auto type = cast<MemRefType>(value.getType());
    return type.getLayout().isIdentity() && type.getElementType() == input.getElementType() &&
        type.getShape() == ArrayRef<int64_t>({n, k});
  };
  Value transposed = getInput();
  if (getInputTransposed()) {
    if (input.getShape() != ArrayRef<int64_t>({n, k}) || getScratch())
      return emitOpError("transposed matrix input must already contain dense N-by-K storage");
  } else {
    if (input.getShape() != output.getShape() || !workspace(getScratch()) ||
        !storage.disjoint(getScratch(), getInput()))
      return emitOpError("row-major matrix input requires independent transpose workspace");
    transposed = getScratch();
  }
  if (n == 64) return !getReshaped() ? success() : emitOpError("64-row filter needs no reshape workspace");
  return workspace(getReshaped()) && storage.disjoint(getReshaped(), transposed)
      ? success() : emitOpError("matrix filter reshape requires independent dense N-by-K workspace");
}
LogicalResult MatMulOp::verify() {
  auto a = cast<MemRefType>(getLhs().getType()), b = cast<MemRefType>(getRhs().getType());
  auto c = cast<MemRefType>(getAccumulator().getType());
  return tile(getLhs()) && tile(getAccumulator()) && b.hasStaticShape() && b.getRank() == 2 &&
      a.getElementType() == b.getElementType() && c.getElementType().isF32() &&
      a.getDimSize(1) == b.getDimSize(getRhsTransposed() ? 1 : 0) && a.getDimSize(0) == c.getDimSize(0) &&
      b.getDimSize(getRhsTransposed() ? 0 : 1) == c.getDimSize(1) && b.getMemorySpaceAsInt() == nramSpace
      ? success() : emitOpError("matmul requires matching M/K/N tiles and an f32 accumulator");
}
LogicalResult MatrixTileOp::verify() {
  auto a = cast<MemRefType>(getLhs().getType()), b = cast<MemRefType>(getRhs().getType());
  auto c = cast<MemRefType>(getAccumulator().getType());
  auto storage = storageFor(getOperation());
  return tile(getLhs()) && tile(getAccumulator()) && b.hasStaticShape() && b.getRank() == 2 &&
      b.getMemorySpaceAsInt() == matrixSpace && a.getElementType() == b.getElementType() && c.getElementType().isF32() &&
      a.getDimSize(1) == b.getDimSize(0) && a.getDimSize(0) == c.getDimSize(0) &&
      b.getDimSize(1) == c.getDimSize(1) && storage.disjoint(getAccumulator(), getLhs())
      ? success() : emitOpError("matrix tile requires prepared storage, matching M/K/N shapes and an f32 accumulator");
}
FailureOr<DenseI64ArrayAttr>
intent::dsa::queryFullExtentRequirements(func::FuncOp function) {
  Attribute attribute = function->getAttr(fullExtentDimensionsAttr);
  if (!attribute)
    return function.emitError("DSA entry requires explicit full-extent obligations"), failure();
  auto dimensions = dyn_cast<DenseI64ArrayAttr>(attribute);
  if (!dimensions)
    return function.emitError("DSA full-extent obligations require a dense i64 array"), failure();

  auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  if (!configuration || configuration.getTile() <= 0)
    return function.emitError("DSA full-extent obligations require a positive bound tile capacity"), failure();
  auto interface = intent::getPublicInterface(function);
  if (failed(intent::verifyPublicInterface(function, interface))) return failure();

  llvm::SmallDenseSet<int64_t> required, found;
  for (int64_t identity : dimensions.asArrayRef()) {
    if (identity <= 0 || !required.insert(identity).second)
      return function.emitError("DSA full-extent obligations require distinct positive public dimension identities"), failure();
  }
  for (Attribute parameter : interface.getArguments()) {
    auto view = dyn_cast<intent::ViewType>(cast<intent::PublicParameterAttr>(parameter).getType());
    if (!view) continue;
    auto tensor = intent::publicViewTensor(view);
    for (auto [extent, identity] : llvm::zip(
             tensor.getShape(), intent::publicViewDimensions(view).asArrayRef())) {
      if (!required.contains(identity)) continue;
      found.insert(identity);
      if (!ShapedType::isDynamic(extent) && extent > configuration.getTile())
        return function.emitError("DSA full-extent public dimension ")
                   << identity << " exceeds the bound tile capacity "
                   << configuration.getTile(), failure();
    }
  }
  for (int64_t identity : dimensions.asArrayRef())
    if (!found.contains(identity))
      return function.emitError("DSA full-extent obligation names an unknown public dimension: ")
                 << identity, failure();
  return dimensions;
}

LogicalResult intent::dsa::verifyProgram(ModuleOp module) {
  if (failed(mlir::verify(module))) return failure();
  auto functions = llvm::to_vector(module.getOps<func::FuncOp>());
  if (functions.size() != 1) return module.emitError("DSA artifact requires one physical kernel");
  auto function = functions.front();
  auto interface = intent::getPublicInterface(function);
  auto config = function->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  auto requirements = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (failed(intent::verifyPublicInterface(function, interface))) return failure();
  if (!config || !requirements || !requirements.getDisjointOutputs() ||
      interface.getArguments().size() != function.getNumArguments())
    return function.emitError("DSA kernel requires a complete interface and configuration");
  if (failed(queryFullExtentRequirements(function))) return failure();
  for (auto [argument, attribute] : llvm::zip(function.getArguments(), interface.getArguments())) {
    auto storageType = [&](Type logical) -> Type {
      if (auto integer = dyn_cast<IntegerType>(logical))
        return IntegerType::get(function.getContext(), integer.getWidth());
      return logical;
    };
    Type logical = cast<intent::PublicParameterAttr>(attribute).getType();
    if (auto view = dyn_cast<intent::ViewType>(logical)) {
      auto tensor = intent::publicViewTensor(view);
      Type element = tensor.getElementType();
      if (!isStorageElementType(element))
        return function.emitError("DSA view has unsupported numeric storage");
      auto physical = dyn_cast<MemRefType>(argument.getType());
      if (!physical || physical.getElementType() != storageType(element) || physical.getShape() != tensor.getShape() ||
          physical.getMemorySpaceAsInt() != 0)
        return function.emitError("DSA view disagrees with its physical entry ABI");
      SmallVector<int64_t> expectedStrides(tensor.getRank(), ShapedType::kDynamic);
      auto constraints = view.getConstraints();
      if (constraints.getHasStrides())
        for (auto [axis, constraint] : llvm::enumerate(constraints.getStrides()))
          if (auto fixed = dyn_cast<IntegerAttr>(constraint))
            expectedStrides[axis] = fixed.getInt();
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(physical.getStridesAndOffset(strides, offset)) || offset != 0 ||
          strides != expectedStrides)
        return function.emitError("DSA public memory layout disagrees with its bound stride contract");
    } else {
      if (!logical.isF32() && !logical.isIndex() && !logical.isInteger(64) &&
          !logical.isInteger(32) && !logical.isInteger(1))
        return function.emitError("DSA scalar has unsupported native ABI type");
      if (storageType(logical) != argument.getType())
        return function.emitError("DSA scalar disagrees with its physical entry ABI");
    }
  }
  auto groupWidth = function->getAttrOfType<IntegerAttr>("intent_dsa.group_width");
  if (groupWidth && (groupWidth.getInt() != 4 || config.getTasks() < 4 || config.getTasks() % 4))
    return function.emitError("MLU370 cooperative execution requires four compute participants per group");
  ExecutionRelations execution(function);
  auto walk = function.walk([&](Operation *op) {
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != "intent_dsa" && dialect != "arith" && dialect != "math" &&
        dialect != "scf" && dialect != "memref" && dialect != "func") {
      op->emitError("operation is outside the DSA execution program");
      return WalkResult::interrupt();
    }
    if (isa<GroupIdOp, GroupCountOp, LocalIdOp, IsMemoryCoreOp, GroupSynchronizeOp, StageTileOp, GroupGatherRowsOp>(op) && !groupWidth) {
      op->emitError("cooperative operations require a bound execution-group width");
      return WalkResult::interrupt();
    }
    if (auto stage = dyn_cast<StageTileOp>(op)) {
      std::function<bool(Value)> memoryCondition = [&](Value condition) {
        if (condition.getDefiningOp<IsMemoryCoreOp>()) return true;
        if (auto conjunction = condition.getDefiningOp<arith::AndIOp>())
          return memoryCondition(conjunction.getLhs()) || memoryCondition(conjunction.getRhs());
        return false;
      };
      bool memoryOwned = false;
      for (Operation *parent = op->getParentOp(); parent && parent != function; parent = parent->getParentOp())
        if (auto branch = dyn_cast<scf::IfOp>(parent))
          memoryOwned |= memoryCondition(branch.getCondition()) &&
              branch.getThenRegion().isAncestor(op->getParentRegion());
      if (!memoryOwned) {
        op->emitError("shared staging must be owned by the memory participant");
        return WalkResult::interrupt();
      }
    }
    if (auto gather = dyn_cast<GroupGatherRowsOp>(op); gather && !execution.isUniform(gather.getRows())) {
      op->emitError("group gather row count must be uniform"); return WalkResult::interrupt();
    }
    if (isa<GroupSynchronizeOp, GroupGatherRowsOp>(op) && !execution.hasUniformControl(op)) {
      op->emitError("execution-group barriers require uniform structured control");
      return WalkResult::interrupt();
    }
    if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
      auto type = allocation.getType();
      if (type.getMemorySpaceAsInt() == sharedSpace && !groupWidth) {
        op->emitError("shared storage requires cooperative execution");
        return WalkResult::interrupt();
      }
      if (!type.hasStaticShape() || type.getRank() != 2 || !type.getLayout().isIdentity() ||
          (type.getMemorySpaceAsInt() != nramSpace && type.getMemorySpaceAsInt() != matrixSpace &&
           type.getMemorySpaceAsInt() != sharedSpace)) {
        op->emitError("local allocation requires bounded shape and storage ownership");
        return WalkResult::interrupt();
      }
    }
    if (auto view = dyn_cast<memref::ReinterpretCastOp>(op)) {
      if (!isCompleteLocalStorageView(view.getResult()) &&
          !isBoundedContiguousLocalView(view.getResult())) {
        op->emitError("DSA local views require a same-dtype complete reshape or a statically bounded contiguous alias of local storage");
        return WalkResult::interrupt();
      }
    }
    for (Type type : op->getResultTypes()) {
      if (isa<MemRefType>(type) && !isa<memref::AllocaOp, memref::ReinterpretCastOp>(op)) {
        op->emitError("DSA local buffers require explicit storage or an owned contiguous view; escaping control state is not supported");
        return WalkResult::interrupt();
      }
    }
    if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      if (yield->getNumOperands()) {
        op->emitError("DSA control results must be copied into explicit state storage before yielding");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

LogicalResult intent::dsa::verifyRealizedProgram(ModuleOp module) {
  if (failed(verifyProgram(module))) return failure();
  WalkResult result = module.walk([&](Operation *operation) {
    if (!isa<SliceReduceOp, ScanOp, CollectiveYieldOp>(operation))
      return WalkResult::advance();
    operation->emitOpError("ordinary collective must be realized before target lowering");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}
