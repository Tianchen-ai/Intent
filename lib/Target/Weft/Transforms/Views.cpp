#include "Views.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Transforms/Task/Tasks.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::weft_provider {
namespace {

bool isViewDefinition(Operation *operation) {
  return operation && (isAxisView(operation) ||
      isa<memref::SubViewOp, memref::CastOp, memref::ExtractStridedMetadataOp>(operation));
}

IntegerRangePolicy taskIntegerRanges() {
  IntegerRangePolicy policy;
  policy.infer = [](Value value, IntegerRangeAnalysis &analysis)
      -> std::optional<ConstantIntRanges> {
    auto argument = dyn_cast<BlockArgument>(value);
    auto tasks = argument ? dyn_cast<cpu::TasksOp>(argument.getOwner()->getParentOp())
                          : cpu::TasksOp{};
    if (!tasks) return std::nullopt;
    if (argument.getArgNumber())
      return analysis.range(tasks.getCaptures()[argument.getArgNumber() - 1]);
    auto count = analysis.range(tasks.getCount());
    if (!count || !count->smax().isStrictlyPositive()) return std::nullopt;
    return ConstantIntRanges::fromSigned(llvm::APInt(64, 0), count->smax() - 1);
  };
  return policy;
}

bool alignedOffset(Value value, int64_t alignment, IntegerRangeAnalysis &ranges) {
  if (auto constant = getConstantIntValue(value)) return *constant % alignment == 0;
  auto operation = value.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation))
    return false;
  Value lhs = operation->getOperand(0), rhs = operation->getOperand(1);
  auto left = ranges.range(lhs), right = ranges.range(rhs);
  auto kind = isa<arith::AddIOp>(operation) ? BinaryOperator::Add
      : isa<arith::SubIOp>(operation) ? BinaryOperator::Subtract : BinaryOperator::Multiply;
  if (!left || !right || !provesSignedNoWrap(kind, *left, *right)) return false;
  if (kind == BinaryOperator::Multiply)
    return alignedOffset(lhs, alignment, ranges) || alignedOffset(rhs, alignment, ranges);
  return alignedOffset(lhs, alignment, ranges) && alignedOffset(rhs, alignment, ranges);
}

void normalizeContiguousWindows(func::FuncOp function) {
  SmallVector<memref::ReinterpretCastOp> windows;
  function.walk([&](memref::ReinterpretCastOp view) { windows.push_back(view); });
  for (auto window : windows) {
    auto metadata = window.getSource().getDefiningOp<memref::ExtractStridedMetadataOp>();
    if (!metadata || window.getSource() != metadata.getBaseBuffer()) continue;
    auto sourceType = cast<MemRefType>(metadata.getSource().getType());
    auto targetType = window.getType();
    if (!sourceType.hasStaticShape() || !targetType.hasStaticShape() ||
        !targetType.getRank() || sourceType.getRank() < targetType.getRank()) continue;
    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;
    if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)) || sourceOffset != 0)
      continue;
    int64_t elements = 1;
    bool contiguous = true;
    for (int64_t axis = sourceType.getRank() - 1; axis >= 0; --axis) {
      int64_t extent = sourceType.getDimSize(axis);
      if (extent <= 0 || sourceStrides[axis] != elements ||
          elements > std::numeric_limits<int64_t>::max() / extent) {
        contiguous = false; break;
      }
      elements *= extent;
    }
    if (!contiguous) continue;
    unsigned prefix = sourceType.getRank() - targetType.getRank();
    int64_t count = 1;
    for (unsigned axis = 0; axis < targetType.getRank(); ++axis) {
      int64_t extent = targetType.getDimSize(axis), original = sourceType.getDimSize(prefix + axis);
      if (extent <= 0 || getConstantIntValue(window.getMixedSizes()[axis]) != extent ||
          getConstantIntValue(window.getMixedStrides()[axis]) != sourceStrides[prefix + axis] ||
          (axis ? extent != original : original % extent != 0) ||
          count > std::numeric_limits<int64_t>::max() / extent) {
        contiguous = false; break;
      }
      count *= extent;
    }
    if (!contiguous) continue;
    OpFoldResult offset = window.getMixedOffsets().front();
    auto constant = getConstantIntValue(offset);
    if (constant) {
      if (*constant < 0 || *constant > elements - count || *constant % count) continue;
    } else {
      auto value = cast<Value>(offset);
      IntegerRangeAnalysis ranges(taskIntegerRanges());
      auto bounds = ranges.range(value);
      if (!bounds || bounds->smin().isNegative() ||
          bounds->smax().sgt(llvm::APInt(64, elements - count)) ||
          !alignedOffset(value, count, ranges)) continue;
    }
    OpBuilder builder(window);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (unsigned axis = 0; axis < sourceType.getRank(); ++axis) {
      if (constant) {
        offsets.push_back(builder.getIndexAttr((*constant / sourceStrides[axis]) % sourceType.getDimSize(axis)));
      } else {
        Value stride = builder.create<arith::ConstantIndexOp>(window.getLoc(), sourceStrides[axis]);
        Value extent = builder.create<arith::ConstantIndexOp>(window.getLoc(), sourceType.getDimSize(axis));
        Value quotient = builder.create<arith::DivUIOp>(window.getLoc(), cast<Value>(offset), stride);
        offsets.push_back(builder.create<arith::RemUIOp>(window.getLoc(), quotient, extent).getResult());
      }
      sizes.push_back(builder.getIndexAttr(axis < prefix ? 1 : targetType.getDimSize(axis - prefix)));
      strides.push_back(builder.getIndexAttr(1));
    }
    auto replacement = builder.create<memref::SubViewOp>(window.getLoc(), targetType,
        metadata.getSource(), offsets, sizes, strides);
    window.replaceAllUsesWith(replacement.getResult());
    window.erase();
  }
}

class CaptureReifier {
public:
  CaptureReifier(cpu::TasksOp task) : task(task), builder(&task.getBody().front(), task.getBody().front().begin()) {}

  LogicalResult run() {
    SmallVector<Value> captures(task.getCaptures());
    auto &body = task.getBody().front();
    for (auto [number, capture] : llvm::enumerate(captures)) {
      auto argument = body.getArgument(number + 1);
      if (argument.use_empty() || !isa<MemRefType>(capture.getType())) continue;
      auto replacement = materialize(capture);
      if (failed(replacement)) return failure();
      if (argument != *replacement) argument.replaceAllUsesWith(*replacement);
    }
    return success();
  }

private:
  Value capture(Value value) {
    for (auto [number, existing] : llvm::enumerate(task.getCaptures()))
      if (existing == value) {
        Value argument = task.getBody().front().getArgument(number + 1);
        mapping.map(value, argument);
        return argument;
      }
    task.getCapturesMutable().append(value);
    auto argument = task.getBody().front().addArgument(value.getType(), value.getLoc());
    mapping.map(value, argument);
    return argument;
  }

  FailureOr<Value> materialize(Value value) {
    if (auto found = mapping.lookupOrNull(value)) return found;
    Operation *definition = value.getDefiningOp();
    // These values belong to the host descriptor being reified. Capturing an
    // already computed size/stride keeps metadata-only storage out of the task
    // ABI and does not replay its scalar dependencies inside each task.
    if (!isa<MemRefType>(value.getType()) &&
        !isa_and_nonnull<arith::ConstantOp>(definition)) return capture(value);
    bool clone = isViewDefinition(definition) ||
        isa_and_nonnull<arith::ConstantOp>(definition);
    if (!clone) {
      if (isa<MemRefType>(value.getType())) {
        cpu::StorageAnalysis storage(task->getParentOfType<func::FuncOp>());
        auto origins = storage.origins(value);
        if (!origins.complete || origins.values.empty())
          return emitError(value.getLoc(), "Weft task capture has unresolved storage origins"), failure();
        if (!cpu::isContiguousDescriptor(value))
          return emitError(value.getLoc(), "Weft task capture requires a proved contiguous descriptor; its native view ABI does not carry strides"), failure();
      }
      return capture(value);
    }
    for (Value operand : definition->getOperands())
      if (failed(materialize(operand))) return failure();
    builder.clone(*definition, mapping);
    return mapping.lookup(value);
  }

  cpu::TasksOp task;
  OpBuilder builder;
  IRMapping mapping;
};

void captureStaticShapeViews(cpu::TasksOp task) {
  // A contiguous host buffer may be captured through several shape views.
  // Form those descriptors before the task; the native ABI then carries the
  // actual view instead of asking canonical Weft to flatten numeric axes.
  IRMapping hostValues;
  for (auto [argument, capture] : llvm::zip_equal(
           task.getBody().front().getArguments().drop_front(), task.getCaptures()))
    hostValues.map(argument, capture);
  SmallVector<Operation *> views;
  task.walk([&](Operation *operation) {
    if (isStaticShapeView(operation)) views.push_back(operation);
  });
  OpBuilder host(task);
  for (Operation *operation : views) {
    Value result = operation->getResult(0);
    if (!cpu::isContiguousDescriptor(result) ||
        !llvm::all_of(operation->getOperands(), [&](Value operand) {
          return hostValues.contains(operand);
        })) continue;
    cpu::StorageAnalysis storage(task->getParentOfType<func::FuncOp>());
    auto aliases = storage.aliases(result);
    if (!aliases.complete || llvm::any_of(aliases.users, [](Operation *user) {
          return isa<cpu::QuantizeOp, cpu::QuantizedDotOp>(user);
        })) continue;
    // Encoded records retain their original addressable view and field axes.
    // Only dense descriptor geometry crosses this boundary unchanged in bytes.
    Value descriptor = host.clone(*operation, hostValues)->getResult(0);
    task.getCapturesMutable().append(descriptor);
    Value argument = task.getBody().front().addArgument(result.getType(), result.getLoc());
    hostValues.map(argument, descriptor);
    result.replaceAllUsesWith(argument);
    operation->erase();
  }
}

} // namespace

bool isAxisView(Operation *operation) {
  return isa_and_nonnull<memref::ReinterpretCastOp, memref::ExpandShapeOp,
                        memref::CollapseShapeOp, memref::TransposeOp>(operation);
}

Value axisViewSource(Operation *operation) {
  if (auto view = dyn_cast<memref::ReinterpretCastOp>(operation)) return view.getSource();
  if (auto view = dyn_cast<memref::ExpandShapeOp>(operation)) return view.getSrc();
  if (auto view = dyn_cast<memref::CollapseShapeOp>(operation)) return view.getSrc();
  if (auto view = dyn_cast<memref::TransposeOp>(operation)) return view.getIn();
  llvm_unreachable("expected a supported memref axis view");
}

bool isStaticShapeView(Operation *operation) {
  if (!isa_and_nonnull<memref::CollapseShapeOp, memref::ExpandShapeOp>(operation))
    return false;
  return cast<MemRefType>(axisViewSource(operation).getType()).hasStaticShape() &&
         cast<MemRefType>(operation->getResult(0).getType()).hasStaticShape();
}

FailureOr<cpu::ViewAxisProjection> queryAxisView(Value value) {
  auto projection = cpu::queryViewAxisProjection(value);
  if (failed(projection))
    emitError(value.getLoc(), "Weft view requires an offset-preserving axis permutation or unit-axis insertion/removal; general strided reinterpretation and axis flattening are unsupported");
  return projection;
}

LogicalResult reifyTaskViewCaptures(func::FuncOp function) {
  SmallVector<cpu::TasksOp> tasks;
  function.walk([&](cpu::TasksOp task) { tasks.push_back(task); });
  if (tasks.empty()) {
    if (!llvm::hasSingleElement(function.getBody()) ||
        function.getFunctionType().getNumResults())
      return function.emitError("Weft sequential entry requires one void CPU body");
    Block &body = function.front();
    // A one-worker entry is still an ordinary host call. Preserve its
    // invocation allocations instead of turning every intermediate into a
    // target-private stack Value merely because there is no parallel task.
    // Only entry-available sizes may cross the new task boundary.
    llvm::SmallPtrSet<Operation *, 16> hostOperations;
    std::function<bool(Value, llvm::SmallPtrSetImpl<Operation *> &)> entryValue =
        [&](Value value, llvm::SmallPtrSetImpl<Operation *> &dependencies) {
      if (auto argument = dyn_cast<BlockArgument>(value))
        return argument.getOwner() == &body;
      Operation *producer = value.getDefiningOp();
      if (!producer || producer->getBlock() != &body || producer->getNumRegions())
        return false;
      if (hostOperations.contains(producer) || dependencies.contains(producer))
        return true;
      if (!isMemoryEffectFree(producer)) return false;
      if (!llvm::all_of(producer->getOperands(), [&](Value input) {
            return entryValue(input, dependencies);
          })) return false;
      dependencies.insert(producer);
      return true;
    };
    for (auto allocation : body.getOps<memref::AllocOp>()) {
      llvm::SmallPtrSet<Operation *, 16> dependencies;
      if (!llvm::all_of(allocation->getOperands(), [&](Value value) {
            return entryValue(value, dependencies);
          })) continue;
      hostOperations.insert(dependencies.begin(), dependencies.end());
      hostOperations.insert(allocation);
      for (Operation *user : allocation->getUsers())
        if (isa<memref::DeallocOp>(user) && user->getBlock() == &body)
          hostOperations.insert(user);
    }
    SmallVector<Operation *> operations;
    for (Operation &operation : body.without_terminator())
      if (!hostOperations.contains(&operation)) operations.push_back(&operation);
    OpBuilder builder(&body, body.begin());
    Value zero = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
    Value one = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
    auto sequential = builder.create<scf::ParallelOp>(function.getLoc(),
        ValueRange{zero}, ValueRange{one}, ValueRange{one});
    // Dependencies were selected from the original SSA order. Retain that
    // order before the task; releases follow its synchronous completion.
    for (Operation &operation : llvm::make_early_inc_range(body))
      if (hostOperations.contains(&operation) && !isa<memref::DeallocOp>(operation))
        operation.moveBefore(sequential);
    for (Operation *operation : operations)
      operation->moveBefore(sequential.getBody()->getTerminator());
    if (failed(cpu::isolateTasks(function))) return failure();
    function.walk([&](cpu::TasksOp task) { tasks.push_back(task); });
  } else {
    SmallVector<Operation *> preparations;
    function.walk([&](scf::ForOp loop) {
      if (loop->getParentOfType<cpu::TasksOp>() || loop.getNumResults() ||
          loop->getParentOp() != function.getOperation()) return;
      bool hasTasks = false, hasDataAccess = false;
      loop.walk([&](Operation *operation) {
        hasTasks |= isa<cpu::TasksOp>(operation);
        hasDataAccess |= isa<memref::LoadOp, memref::StoreOp>(operation);
      });
      if (!hasTasks && hasDataAccess) preparations.push_back(loop);
    });
    // Invocation-owned storage stays on the host. Its serial numerical
    // preparations execute as joined tasks at their original program points.
    for (Operation *preparation : preparations) {
      OpBuilder builder(preparation);
      Location loc = preparation->getLoc();
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto task = builder.create<scf::ParallelOp>(loc, ValueRange{zero},
          ValueRange{one}, ValueRange{one});
      preparation->moveBefore(task.getBody()->getTerminator());
    }
    if (!preparations.empty()) {
      if (failed(cpu::isolateTasks(function))) return failure();
      tasks.clear();
      function.walk([&](cpu::TasksOp task) { tasks.push_back(task); });
    }
  }
  for (auto task : tasks) if (failed(CaptureReifier(task).run())) return failure();
  normalizeContiguousWindows(function);
  for (auto task : tasks) captureStaticShapeViews(task);
  return success();
}

} // namespace intent::weft_provider
