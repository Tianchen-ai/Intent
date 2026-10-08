#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "Intent/Dialect/DSA/Analysis/UniformValues.h"
#include "Intent/Dialect/DSA/IR/ExecutionRelations.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

using namespace mlir;
namespace intent::dsa {
namespace {
bool supportedScalarBinary(BinaryOperator kind) {
  return kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply;
}
} // namespace
void bindUniformOperands(func::FuncOp function) {
  SmallVector<dsa::BinaryOp> binaries;
  function.walk([&](dsa::BinaryOp binary) { binaries.push_back(binary); });
  bool changed = false;
  for (auto binary : binaries) {
    if (!supportedScalarBinary(binary.getKind()) ||
        binary.getApproximate() || binary.getFlushToZero()) continue;
    Type element = cast<MemRefType>(binary.getLhs().getType()).getElementType();
    if (!element.isF16() && !element.isF32()) continue;
    Value previous = binary.getRhs();
    StorageAnalysis storage(function);
    UniformMemoryAnalysis uniforms(function, storage);
    Value scalar = uniforms.read(previous, binary);
    if (!scalar) continue;
    binary.getRhsMutable().assign(scalar);
    changed = true;
  }
  function.walk([&](dsa::SelectOp select) {
    if (!isa<MemRefType>(select.getFalseValue().getType())) return;
    StorageAnalysis storage(function);
    UniformMemoryAnalysis uniforms(function, storage);
    Value scalar = uniforms.read(select.getFalseValue(), select);
    FloatAttr constant;
    if (scalar && matchPattern(scalar, m_Constant(&constant))) {
      select.getFalseValueMutable().assign(scalar);
      changed = true;
    }
  });
  if (changed)
    eliminateUnreadLocalWrites(function);
}

bool foldUniformLocalValues(func::FuncOp function) {
  SmallVector<Operation *> operations;
  function.walk([&](Operation *operation) {
    if (isa<LoadTileOp, memref::CopyOp, TransposeOp, SelectOp>(operation))
      operations.push_back(operation);
  });
  bool changed = false;
  for (Operation *operation : operations) {
    StorageAnalysis storage(function);
    UniformMemoryAnalysis uniforms(function, storage);
    OpBuilder builder(operation);
    Location location = operation->getLoc();
    if (auto select = dyn_cast<SelectOp>(operation)) {
      Value condition = select.getCondition();
      if (isa<MemRefType>(condition.getType()))
        condition = uniforms.read(condition, select);
      APInt predicate;
      if (!condition || !matchPattern(condition, m_ConstantInt(&predicate))) continue;
      Value source = predicate.isZero() ? select.getFalseValue() : select.getTrueValue();
      if (!isa<MemRefType>(source.getType()))
        builder.create<FillOp>(location, select.getOutput(), source);
      else if (source != select.getOutput())
        builder.create<memref::CopyOp>(location, source, select.getOutput());
      select.erase(); changed = true;
      continue;
    }
    Value output;
    if (auto load = dyn_cast<LoadTileOp>(operation)) {
      if (load.getAsynchronous()) continue;
      output = load.getOutput();
    } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      output = copy.getTarget();
    } else {
      output = cast<TransposeOp>(operation).getOutput();
    }
    auto type = cast<MemRefType>(output.getType());
    if (type.getRank() != 2 || !type.hasStaticShape() || type.getNumElements() <= 0 ||
        type.getMemorySpaceAsInt() != nramSpace || !type.getLayout().isIdentity() ||
        !operation->getNextNode()) continue;
    Value scalar = uniforms.read(output, operation->getNextNode());
    if (!scalar) continue;
    builder.create<FillOp>(location, output, scalar);
    operation->erase(); changed = true;
  }
  return changed;
}

bool eliminateUnreadLocalWrites(func::FuncOp function) {
  SmallVector<memref::AllocaOp> allocations;
  function.walk([&](memref::AllocaOp allocation) {
    if (allocation.getType().getMemorySpaceAsInt() == dsa::nramSpace) allocations.push_back(allocation);
  });
  bool changed = false;
  for (auto allocation : allocations) {
    StorageAnalysis storage(function);
    auto aliases = storage.aliases(allocation);
    if (!aliases.complete)
      continue;
    SmallVector<Operation *> writes;
    bool unread = true;
    for (Operation *user : aliases.users) {
      if (isBufferStorageAliasOperation(user))
        continue;
      bool write = false;
      if (auto fill = dyn_cast<dsa::FillOp>(user))
        write = storage.uniqueOrigin(fill.getOutput()) == allocation;
      if (auto store = dyn_cast<memref::StoreOp>(user))
        write = storage.uniqueOrigin(store.getMemref()) == allocation;
      if (auto copy = dyn_cast<memref::CopyOp>(user))
        write = storage.uniqueOrigin(copy.getTarget()) == allocation &&
                storage.disjoint(allocation, copy.getSource());
      if (auto load = dyn_cast<dsa::LoadTileOp>(user))
        write = !load.getAsynchronous() &&
                storage.uniqueOrigin(load.getOutput()) == allocation &&
                storage.disjoint(allocation, load.getSource());
      if (auto cast = dyn_cast<dsa::CastOp>(user))
        write = storage.uniqueOrigin(cast.getOutput()) == allocation &&
                storage.disjoint(allocation, cast.getInput());
      if (!write) unread = false;
      else if (!llvm::is_contained(writes, user)) writes.push_back(user);
    }
    if (!unread || writes.empty()) continue;
    for (Operation *write : writes) write->erase();
    for (Value alias : llvm::reverse(aliases.values))
      if (alias.use_empty() && alias.getDefiningOp())
        alias.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}

bool batchPointwiseTasks(func::FuncOp function) {
  auto interface = intent::getPublicInterface(function);
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  auto argumentWithAccess = [&](Value value, unsigned access) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = intent::getPublicView(interface, argument.getArgNumber());
    return view && view.getAccess() == access;
  };
  auto constant = [](Value value) -> int64_t {
    auto index = value.getDefiningOp<arith::ConstantIndexOp>();
    return index ? index.value() : -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (auto loop : loops) {
    if (!loop.getLowerBound().getDefiningOp<dsa::TaskIdOp>() ||
        !loop.getStep().getDefiningOp<dsa::TaskCountOp>() || !loop.getInitArgs().empty()) continue;
    Block *body = loop.getBody();
    BlockArgument induction = cast<BlockArgument>(loop.getInductionVar());
    ExecutionRelations execution(function);
    SmallVector<memref::AllocaOp> allocations;
    dsa::LoadTileOp load;
    dsa::StoreTileOp store;
    int64_t capacity = 0;
    bool eligible = true;
    for (Operation &operation : *body) {
      if (auto allocation = dyn_cast<memref::AllocaOp>(operation)) {
        auto type = allocation.getType();
        if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != 1 ||
            !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
            (capacity && capacity != type.getDimSize(1)) ||
            llvm::any_of(allocation.getResult().getUsers(), [&](Operation *user) { return user->getBlock() != body; })) {
          eligible = false; break;
        }
        capacity = type.getDimSize(1);
        allocations.push_back(allocation);
      } else if (auto transfer = dyn_cast<dsa::LoadTileOp>(operation)) {
        if (load || !argumentWithAccess(transfer.getSource(), 0)) { eligible = false; break; }
        load = transfer;
      } else if (auto transfer = dyn_cast<dsa::StoreTileOp>(operation)) {
        if (store || !argumentWithAccess(transfer.getDestination(), 1)) { eligible = false; break; }
        store = transfer;
      } else if (isa<dsa::FillOp, dsa::UnaryOp, dsa::BinaryOp, dsa::CastOp, dsa::CompareOp, dsa::SelectOp>(operation)) {
        if (auto unary = dyn_cast<dsa::UnaryOp>(operation); unary && unary.getScratch()) eligible = false;
        if (auto select = dyn_cast<dsa::SelectOp>(operation); select && select.getScratch()) eligible = false;
        for (Value operand : operation.getOperands()) {
          if (isa<MemRefType>(operand.getType())) {
            auto allocation = operand.getDefiningOp<memref::AllocaOp>();
            if (!allocation || allocation->getBlock() != body) eligible = false;
          } else if (execution.coordinateDependency(operand, induction) !=
                     CoordinateDependency::Independent) eligible = false;
        }
        if (!eligible) break;
      } else if (auto stride = dyn_cast<dsa::StrideOp>(operation)) {
        auto argument = dyn_cast<BlockArgument>(stride.getSource());
        if (!argument || argument.getOwner() != &function.front()) { eligible = false; break; }
      } else if (!isa<scf::YieldOp>(operation) &&
                 (operation.getNumRegions() || !isMemoryEffectFree(&operation) ||
                  operation.getName().getDialectNamespace() != "arith")) {
        eligible = false; break;
      }
    }
    if (!eligible || !load || !store || allocations.empty()) continue;
    int64_t columns = constant(load.getColumns());
    if (columns <= 0 || constant(store.getColumns()) != columns ||
        constant(load.getRows()) != 1 || constant(store.getRows()) != 1 ||
        capacity <= columns || capacity % columns) continue;
    auto matchesStride = [&](Value value, Value resource, int64_t axis) {
      if (auto stride = value.getDefiningOp<dsa::StrideOp>())
        return stride.getSource() == resource && stride.getAxis() == uint64_t(axis);
      auto argument = dyn_cast<BlockArgument>(resource);
      if (!argument || argument.getOwner() != &function.front()) return false;
      auto view = intent::getPublicView(interface, argument.getArgNumber());
      if (!view || !view.getConstraints().getHasStrides()) return false;
      auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis]);
      APInt bits;
      return fixed && matchPattern(value, m_ConstantInt(&bits)) && bits.getSExtValue() == fixed.getInt();
    };
    auto rowCoefficient = [&](Value view, Value offset, Value columnStride) -> Value {
      auto type = cast<MemRefType>(view.getType());
      if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(1) != columns ||
          type.getDimSize(0) != constant(loop.getUpperBound())) return {};
      if (!matchesStride(columnStride, view, 1)) return {};
      auto product = offset.getDefiningOp<arith::MulIOp>();
      if (!product) return {};
      Value coefficient = product.getLhs() == induction ? product.getRhs() :
          product.getRhs() == induction ? product.getLhs() : Value{};
      return coefficient && matchesStride(coefficient, view, 0) ? coefficient : Value{};
    };
    Value inputPitch = rowCoefficient(load.getSource(), load.getOffset(), load.getColumnStride());
    Value outputPitch = rowCoefficient(store.getDestination(), store.getOffset(), store.getColumnStride());
    auto owned = [&](Value value) {
      auto allocation = value.getDefiningOp<memref::AllocaOp>();
      return allocation && llvm::is_contained(allocations, allocation);
    };
    if (!inputPitch || !outputPitch || !owned(load.getOutput()) || !owned(store.getInput())) continue;
    // The runtime rejects overlapping writable view arguments. With read-only
    // input, write-only output and no carried/local scalar state, these rows
    // can share one supply/compute/store while retaining each task's row order.
    int64_t rows = capacity / columns;
    int64_t tasks = config.getTasks(), upper = constant(loop.getUpperBound());
    if (tasks <= 0 || rows > (std::numeric_limits<int64_t>::max() - upper) / tasks) continue;
    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    Value factor = builder.create<arith::ConstantIndexOp>(loc, rows);
    Value oldStep = loop.getStep();
    loop.getStepMutable().assign(builder.create<arith::MulIOp>(loc, oldStep, factor));
    builder.setInsertionPointToStart(body);
    Value remaining = builder.create<arith::SubIOp>(loc, loop.getUpperBound(), induction);
    Value count = builder.create<arith::CeilDivSIOp>(loc, remaining, oldStep);
    count = builder.create<arith::MinSIOp>(loc, count, factor);
    for (auto allocation : allocations) {
      auto type = allocation.getType();
      allocation.getResult().setType(MemRefType::get({rows, columns}, type.getElementType(),
          MemRefLayoutAttrInterface{}, type.getMemorySpace()));
    }
    builder.setInsertionPoint(load);
    load.getRowStrideMutable().assign(builder.create<arith::MulIOp>(loc, inputPitch, oldStep));
    load.getRowsMutable().assign(count);
    builder.setInsertionPoint(store);
    store.getRowStrideMutable().assign(builder.create<arith::MulIOp>(loc, outputPitch, oldStep));
    store.getRowsMutable().assign(count);
    changed = true;
  }
  return changed;
}
} // namespace intent::dsa
