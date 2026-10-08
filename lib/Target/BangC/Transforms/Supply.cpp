#include "PassDetail.h"
#include "Intent/Target/BangC/NativeWorkspace.h"

using namespace mlir;
namespace intent::bangc {
void realizeGatherWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::GatherRowsOp> gathers;
  function.walk([&](dsa::GatherRowsOp gather) { if (!gather.getPlan()) gathers.push_back(gather); });
  DenseMap<int64_t, Value> indicesByRows;
  for (auto gather : gathers) {
    int64_t rows = cast<MemRefType>(gather.getOutput().getType()).getDimSize(0);
    auto workspace = gatherWorkspace(rows);
    if (!workspace) continue;
    OpBuilder b(gather);
    Location loc = gather.getLoc();
    dsa::StorageAnalysis storage(function);
    for (Operation *previous = gather->getPrevNode(); previous; previous = previous->getPrevNode()) {
      auto supply = dyn_cast<dsa::GatherRowsOp>(previous);
      if (supply && supply.getPlan() && supply.getRowOffsets() == gather.getRowOffsets() &&
          supply.getRows() == gather.getRows() &&
          cast<MemRefType>(supply.getOutput().getType()).getDimSize(0) == rows &&
          storage.readStable(gather.getRowOffsets(), supply, gather) &&
          storage.unchangedBetween(supply.getPlan(), supply, gather)) {
        gather.getPlanMutable().assign(supply.getPlan());
        break;
      }
    }
    if (gather.getPlan()) continue;
    Value scratch = allocate(b, loc, b.getI64Type(), workspace->plan, dsa::nramSpace);
    Value indices = indicesByRows.lookup(rows);
    dsa::IotaOp initialize;
    if (!indices) {
      OpBuilder init(function.getContext());
      init.setInsertionPointToStart(&function.front());
      indices = allocate(init, loc, init.getF32Type(), workspace->indices, dsa::nramSpace);
      initialize = init.create<dsa::IotaOp>(loc, indices);
    }
    auto prepare = b.create<dsa::GatherPlanOp>(loc, gather.getRowOffsets(), gather.getRows(), indices, scratch);
    prepare->setAttr("bangc.internal_nram_bytes", b.getI64IntegerAttr(workspace->internalBytes));
    gather.getPlanMutable().assign(scratch);
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() - workspace->internalBytes ||
        nram > 768 * 1024 - workspace->internalBytes) {
      gather.getPlanMutable().clear();
      prepare.erase();
      scratch.getDefiningOp()->erase();
      if (initialize) { initialize.erase(); indices.getDefiningOp()->erase(); }
      continue;
    }
    indicesByRows[rows] = indices;
  }
  // Submit consecutive independent destinations before the common completion
  // fence. The plan is read-only throughout the group, and remains live until
  // every transfer has been issued.
  for (auto gather : gathers) {
    if (!gather.getPlan() || gather.getAsynchronous()) continue;
    dsa::StorageAnalysis storage(function);
    SmallVector<dsa::GatherRowsOp> group{gather};
    Operation *next = gather->getNextNode();
    while (next) {
      if (isa<memref::AllocaOp>(next) || isMemoryEffectFree(next)) { next = next->getNextNode(); continue; }
      auto supply = dyn_cast<dsa::GatherRowsOp>(next);
      if (!supply || supply.getPlan() != gather.getPlan() ||
          llvm::any_of(group, [&](dsa::GatherRowsOp previous) {
            return !storage.disjoint(previous.getOutput(), supply.getOutput()) ||
                !storage.disjoint(previous.getOutput(), supply.getSource()) ||
                !storage.disjoint(supply.getOutput(), previous.getSource()) ||
                !storage.disjoint(supply.getOutput(), previous.getPlan());
          })) break;
      group.push_back(supply);
      next = next->getNextNode();
    }
    if (group.size() < 2) continue;
    OpBuilder b(gather);
    b.create<dsa::SynchronizeOp>(gather.getLoc());
    for (auto supply : group) supply.setAsynchronous(true);
    b.setInsertionPointAfter(group.back());
    b.create<dsa::SynchronizeOp>(gather.getLoc());
  }
}

bool realizeRowBroadcasts(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::LoadTileOp> loads;
  function.walk([&](dsa::LoadTileOp load) { loads.push_back(load); });
  bool changed = false;
  for (auto load : loads) {
    auto source = cast<MemRefType>(load.getSource().getType());
    auto output = cast<MemRefType>(load.getOutput().getType());
    if (load.getAsynchronous() || source.getMemorySpaceAsInt() != dsa::nramSpace ||
        !source.getLayout().isIdentity() || !output.getElementType().isF32() ||
        output.getDimSize(0) < 16 || output.getDimSize(1) < 16 ||
        !matchPattern(load.getRowStride(), m_One()) || !matchPattern(load.getColumnStride(), m_Zero())) continue;
    OpBuilder b(load);
    Value scratch = allocate(b, load.getLoc(), output.getElementType(),
        {output.getDimSize(1), output.getDimSize(0)}, dsa::nramSpace);
    auto broadcast = b.create<dsa::BroadcastRowsOp>(load.getLoc(), load.getSource(), load.getOutput(), scratch,
        load.getOffset(), load.getRows(), load.getColumns());
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024) {
      broadcast.erase(); scratch.getDefiningOp()->erase(); continue;
    }
    load.erase(); changed = true;
  }
  return changed;
}

void coalesceTileLoads(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = intent::getPublicInterface(function);
  auto eligible = [&](dsa::LoadTileOp load) {
    if (!load || load.getAsynchronous()) return false;
    auto argument = dyn_cast<BlockArgument>(load.getSource());
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = intent::getPublicView(interface, argument.getArgNumber());
    return view && view.getAccess() == 0;
  };
  SmallVector<dsa::LoadTileOp> loads;
  function.walk([&](dsa::LoadTileOp load) { if (eligible(load)) loads.push_back(load); });
  for (auto load : loads) {
    if (load.getAsynchronous()) continue;
    dsa::StorageAnalysis storage(function);
    SmallVector<dsa::LoadTileOp> group{load};
    SmallVector<Value> destinations{load.getOutput()};
    Operation *next = load->getNextNode();
    while (next) {
      if (isa<memref::AllocaOp>(next) || isMemoryEffectFree(next)) { next = next->getNextNode(); continue; }
      auto supply = dyn_cast<dsa::LoadTileOp>(next);
      if (!eligible(supply) || llvm::any_of(destinations, [&](Value value) {
            return !storage.disjoint(value, supply.getOutput());
          })) break;
      group.push_back(supply); destinations.push_back(supply.getOutput());
      next = next->getNextNode();
    }
    if (group.size() < 2) continue;
    OpBuilder b(load);
    auto before = b.create<dsa::SynchronizeOp>(load.getLoc());
    for (auto supply : group) supply.setAsynchronous(true);
    b.setInsertionPointAfter(group.back());
    auto after = b.create<dsa::SynchronizeOp>(load.getLoc());
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      for (auto supply : group) supply.setAsynchronous(false);
      before.erase(); after.erase();
    }
  }
}

bool batchIndependentRowPrograms(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = intent::getPublicInterface(function);
  auto viewArgument = [&](Value value, unsigned access) -> intent::ViewType {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return {};
    auto view = intent::getPublicView(interface, argument.getArgNumber());
    return view && view.getAccess() == access ? view : intent::ViewType{};
  };
  auto constant = [](Value value) -> int64_t {
    APInt bits;
    return matchPattern(value, m_ConstantInt(&bits)) ? bits.getSExtValue() : -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (auto loop : loops) {
    if (!loop.getLowerBound().getDefiningOp<dsa::TaskIdOp>() ||
        !loop.getStep().getDefiningOp<dsa::TaskCountOp>() || !loop.getInitArgs().empty()) continue;
    auto *body = loop.getBody();
    Value induction = loop.getInductionVar();
    dsa::LoadTileOp load;
    dsa::StoreTileOp store;
    SmallVector<memref::AllocaOp> allocations;
    SmallVector<dsa::ReduceOp> reductions;
    SmallVector<memref::LoadOp> scalarLoads;
    bool eligible = true;
    for (Operation &op : body->without_terminator()) {
      if (auto allocation = dyn_cast<memref::AllocaOp>(op)) allocations.push_back(allocation);
      else if (auto transfer = dyn_cast<dsa::LoadTileOp>(op)) {
        if (load || transfer.getAsynchronous() || !viewArgument(transfer.getSource(), 0)) eligible = false;
        load = transfer;
      } else if (auto transfer = dyn_cast<dsa::StoreTileOp>(op)) {
        if (store || !viewArgument(transfer.getDestination(), 1)) eligible = false;
        store = transfer;
      } else if (auto reduction = dyn_cast<dsa::ReduceOp>(op)) {
        auto kind = reduction.getKind();
        if (reduction.getAxis() != 1 || (kind != BinaryOperator::Add && kind != BinaryOperator::Maximum &&
            kind != BinaryOperator::MaximumNum && kind != BinaryOperator::Minimum && kind != BinaryOperator::MinimumNum)) eligible = false;
        reductions.push_back(reduction);
      } else if (auto read = dyn_cast<memref::LoadOp>(op)) {
        if (!read.getType().isF32() || read.getMemRefType().getShape() != ArrayRef<int64_t>({1, 1}) ||
            llvm::any_of(read.getIndices(), [&](Value index) { return constant(index) != 0; })) eligible = false;
        for (Operation *user : read.getResult().getUsers()) {
          if (auto fill = dyn_cast<dsa::FillOp>(user); fill && fill.getValue() == read.getResult()) continue;
          if (auto binary = dyn_cast<dsa::BinaryOp>(user); binary && binary.getRhs() == read.getResult()) continue;
          eligible = false;
        }
        scalarLoads.push_back(read);
      } else if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
        if (unary.getScratch()) eligible = false;
      } else if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
        if (binary.getScratch()) eligible = false;
      } else if (auto stride = dyn_cast<dsa::StrideOp>(op)) {
        if (!isa<BlockArgument>(stride.getSource())) eligible = false;
      } else if (!isa<dsa::FillOp, dsa::CastOp, memref::CopyOp>(op) &&
                 (op.getNumRegions() || op.getName().getDialectNamespace() != "arith" || !isMemoryEffectFree(&op))) eligible = false;
    }
    if (!eligible || !load || !store || reductions.empty()) continue;
    int64_t columns = constant(load.getColumns()), upper = constant(loop.getUpperBound());
    if (columns < 1024 || columns % 32 || columns != constant(store.getColumns()) ||
        constant(load.getRows()) != 1 || constant(store.getRows()) != 1 || upper <= 0) continue;
    auto owned = [&](Value value) {
      auto allocation = value.getDefiningOp<memref::AllocaOp>();
      return allocation && llvm::is_contained(allocations, allocation);
    };
    for (auto allocation : allocations) {
      auto type = allocation.getType();
      if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != 1 ||
          (type.getDimSize(1) != 1 && type.getDimSize(1) != columns) || !type.getLayout().isIdentity() ||
          type.getMemorySpaceAsInt() != dsa::nramSpace ||
          llvm::any_of(allocation.getResult().getUsers(), [&](Operation *user) { return user->getBlock() != body; })) eligible = false;
    }
    for (Operation &op : body->without_terminator()) {
      if (isa<dsa::LoadTileOp, dsa::StoreTileOp, dsa::StrideOp>(op)) continue;
      for (Value operand : op.getOperands()) if (isa<MemRefType>(operand.getType()) && !owned(operand)) eligible = false;
    }
    for (auto reduction : reductions) {
      if (cast<MemRefType>(reduction.getInput().getType()).getShape() != ArrayRef<int64_t>({1, columns}) ||
          !cast<MemRefType>(reduction.getInput().getType()).getElementType().isF32() || constant(reduction.getCount()) != columns ||
          !reduction.getScratch().hasOneUse() || reduction.getScratch() == reduction.getInput() ||
          reduction.getScratch() == reduction.getOutput()) eligible = false;
    }
    DenseMap<Value, bool> varying;
    std::function<bool(Value)> depends = [&](Value value) {
      if (value == induction || value.getDefiningOp<dsa::TaskIdOp>()) return true;
      if (auto found = varying.find(value); found != varying.end()) return found->second;
      auto *definition = value.getDefiningOp();
      return varying[value] = definition && llvm::any_of(definition->getOperands(), depends);
    };
    for (Operation &op : body->without_terminator()) if (isa<dsa::FillOp, dsa::UnaryOp, dsa::BinaryOp, dsa::ReduceOp>(op))
      for (Value operand : op.getOperands()) if (!isa<MemRefType>(operand.getType()) && depends(operand)) eligible = false;
    auto pitch = [&](Value resource, Value offset, Value columnStride, unsigned access) -> Value {
      auto type = cast<MemRefType>(resource.getType());
      auto view = viewArgument(resource, access);
      if (!view || type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != upper || type.getDimSize(1) != columns) return {};
      auto matches = [&](Value value, int axis) {
        if (auto stride = value.getDefiningOp<dsa::StrideOp>()) return stride.getSource() == resource && stride.getAxis() == uint64_t(axis);
        if (!view.getConstraints().getHasStrides()) return false;
        auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis]);
        return fixed && constant(value) == fixed.getInt();
      };
      auto product = offset.getDefiningOp<arith::MulIOp>();
      if (!product || !matches(columnStride, 1)) return {};
      Value coefficient = product.getLhs() == induction ? product.getRhs() : product.getRhs() == induction ? product.getLhs() : Value{};
      return coefficient && !depends(coefficient) && matches(coefficient, 0) ? coefficient : Value{};
    };
    Value inputPitch = pitch(load.getSource(), load.getOffset(), load.getColumnStride(), 0);
    Value outputPitch = pitch(store.getDestination(), store.getOffset(), store.getColumnStride(), 1);
    if (!eligible || !inputPitch || !outputPitch || !owned(load.getOutput()) || !owned(store.getInput())) continue;
    for (int64_t rows : {8, 4, 2}) {
      if (config.getTasks() <= 0 || config.getTasks() > upper / rows ||
          upper % (rows * config.getTasks()) || columns > 65536 / rows) continue;
      OpBuilder b(loop); Location loc = loop.getLoc();
      Value rowCount = b.create<arith::ConstantIndexOp>(loc, rows);
      Value batches = b.create<arith::ConstantIndexOp>(loc, upper / rows);
      auto batch = b.create<scf::ForOp>(loc, loop.getLowerBound(), batches, loop.getStep());
      b.setInsertionPointToStart(batch.getBody());
      // Independent logical rows are partitioned into adjacent row blocks.
      // A task owns whole blocks, so each transfer has the original row pitch.
      Value firstRow = b.create<arith::MulIOp>(loc, batch.getInductionVar(), rowCount);
      IRMapping mapping; mapping.map(induction, firstRow);
      auto mapped = [&](Value value) { return mapping.lookupOrDefault(value); };
      auto broadcast = [&](Value source, Value destination) {
        auto type = cast<MemRefType>(destination.getType());
        if (source.getType() == destination.getType()) b.create<memref::CopyOp>(loc, source, destination);
        else {
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0), one = b.create<arith::ConstantIndexOp>(loc, 1);
          Value width = b.create<arith::ConstantIndexOp>(loc, type.getDimSize(1));
          b.create<dsa::LoadTileOp>(loc, source, destination, zero, one, zero, rowCount, width, b.getBoolAttr(false));
        }
      };
      for (Operation &op : body->without_terminator()) {
        if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
          auto type = allocation.getType();
          bool scratch = llvm::any_of(reductions, [&](dsa::ReduceOp reduce) { return reduce.getScratch() == allocation; });
          SmallVector<int64_t> shape = scratch ? SmallVector<int64_t>{1, columns} :
              type.getDimSize(1) == 1 ? SmallVector<int64_t>{1, rows} : SmallVector<int64_t>{rows, columns};
          mapping.map(allocation, allocate(b, loc, type.getElementType(), shape, dsa::nramSpace));
        } else if (auto read = dyn_cast<memref::LoadOp>(op)) mapping.map(read.getResult(), mapped(read.getMemref()));
        else if (auto fill = dyn_cast<dsa::FillOp>(op); fill && isa<MemRefType>(mapped(fill.getValue()).getType()))
          broadcast(mapped(fill.getValue()), mapped(fill.getOutput()));
        else if (auto binary = dyn_cast<dsa::BinaryOp>(op); binary && !isa<MemRefType>(binary.getRhs().getType()) && isa<MemRefType>(mapped(binary.getRhs()).getType())) {
          Value rhs = mapped(binary.getRhs()), output = mapped(binary.getOutput());
          if (rhs.getType() != output.getType()) {
            auto type = cast<MemRefType>(output.getType());
            Value expanded = allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
            broadcast(rhs, expanded); rhs = expanded;
          }
          b.create<dsa::BinaryOp>(loc, mapped(binary.getLhs()), rhs, output, binary.getKindAttr(),
              binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value{});
        } else {
          Operation *cloned = b.clone(op, mapping);
          if (&op == load.getOperation()) {
            auto transfer = cast<dsa::LoadTileOp>(cloned);
            transfer.getRowsMutable().assign(rowCount);
            transfer.getRowStrideMutable().assign(mapped(inputPitch));
          } else if (&op == store.getOperation()) {
            auto transfer = cast<dsa::StoreTileOp>(cloned);
            transfer.getRowsMutable().assign(rowCount);
            transfer.getRowStrideMutable().assign(mapped(outputPitch));
          }
        }
      }
      int64_t nram = 0, wram = 0; measureStorage(function, nram, wram);
      if (nram + 32768 > config.getLocalBytes() || nram + 32768 > 768 * 1024 || wram > 1024 * 1024) {
        batch.erase(); batches.getDefiningOp()->erase(); rowCount.getDefiningOp()->erase(); continue;
      }
      loop.erase(); changed = true; break;
    }
  }
  return changed;
}

void pipelineMatrixLoads(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = intent::getPublicInterface(function);
  auto readOnlyArgument = [&](Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = intent::getPublicView(interface, argument.getArgNumber());
    return view && view.getAccess() == 0;
  };
  auto constant = [](Value value) -> int64_t {
    auto op = value.getDefiningOp<arith::ConstantIndexOp>();
    return op ? op.value() : -1;
  };
  std::function<int64_t(Value)> upperLimit = [&](Value value) -> int64_t {
    int64_t fixed = constant(value);
    if (fixed >= 0) return fixed;
    if (auto minimum = value.getDefiningOp<arith::MinSIOp>()) {
      int64_t lhs = upperLimit(minimum.getLhs()), rhs = upperLimit(minimum.getRhs());
      return lhs >= 0 && rhs >= 0 ? std::min(lhs, rhs) : std::max(lhs, rhs);
    }
    return -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (auto loop : loops) {
    int64_t lower = constant(loop.getLowerBound()), upper = upperLimit(loop.getUpperBound());
    int64_t step = constant(loop.getStep());
    if (!loop.getInitArgs().empty() || lower < 0 || step <= 0 || upper <= lower ||
        step > std::numeric_limits<int64_t>::max() / 2 ||
        upper > std::numeric_limits<int64_t>::max() - 2 * step || upper - lower <= step) continue;
    SmallVector<dsa::LoadTileOp> loads;
    dsa::PrepareMatrixOp prepare;
    dsa::MatrixTileOp matrix;
    dsa::StorageAnalysis storage(function);
    auto owner = [&](Value value) { return storage.uniqueOrigin(value); };
    auto readOnlySource = [&](Value source) {
      if (readOnlyArgument(source)) return true;
      Value origin = owner(source);
      auto allocation = origin ? origin.getDefiningOp<memref::AllocaOp>() : memref::AllocaOp{};
      if (!allocation || allocation.getType().getMemorySpaceAsInt() != dsa::sharedSpace || loop->isAncestor(allocation)) return false;
      return storage.aliases(origin).complete && storage.preservesContents(loop, source);
    };
    auto eligible = loop.walk([&](Operation *op) {
      if (op == loop.getOperation()) return WalkResult::advance();
      if (auto load = dyn_cast<dsa::LoadTileOp>(op)) {
        if (load->getBlock() != loop.getBody() || load.getAsynchronous() || !readOnlySource(load.getSource()))
          return WalkResult::interrupt();
        loads.push_back(load);
        return WalkResult::advance();
      }
      if (auto packing = dyn_cast<dsa::PrepareMatrixOp>(op)) {
        if (prepare) return WalkResult::interrupt();
        prepare = packing;
        return WalkResult::advance();
      }
      if (auto compute = dyn_cast<dsa::MatrixTileOp>(op)) {
        if (matrix) return WalkResult::interrupt();
        matrix = compute;
        return WalkResult::advance();
      }
      if (auto branch = dyn_cast<scf::IfOp>(op))
        return branch.getNumResults() || !branch.getElseRegion().empty()
            ? WalkResult::interrupt() : WalkResult::advance();
      if (auto stride = dyn_cast<dsa::StrideOp>(op))
        return readOnlyArgument(stride.getSource()) ? WalkResult::advance() : WalkResult::interrupt();
      if (isa<dsa::SynchronizeOp, memref::AllocaOp, memref::ReinterpretCastOp, scf::YieldOp>(op))
        return WalkResult::advance();
      if (op->getName().getDialectNamespace() == "arith" && isMemoryEffectFree(op) && !op->getNumRegions())
        return WalkResult::advance();
      return WalkResult::interrupt();
    });
    if (eligible.wasInterrupted() || loads.size() != 2 || !prepare || !matrix ||
        prepare->getBlock() != matrix->getBlock() || !prepare->isBeforeInBlock(matrix) ||
        matrix.getRhs() != prepare.getOutput()) continue;
    dsa::LoadTileOp lhsLoad, rhsLoad;
    for (auto load : loads) {
      if (load.getOutput() == matrix.getLhs()) lhsLoad = load;
      if (load.getOutput() == prepare.getInput()) rhsLoad = load;
    }
    if (!lhsLoad || !rhsLoad || lhsLoad == rhsLoad) continue;
    Operation *computeScope = loop.getBody()->findAncestorOpInBlock(*prepare.getOperation());
    if (!computeScope || !lhsLoad->isBeforeInBlock(computeScope) || !rhsLoad->isBeforeInBlock(computeScope)) continue;
    Value lhs = lhsLoad.getOutput(), rhs = rhsLoad.getOutput(), accumulator = matrix.getAccumulator();
    auto lhsAllocation = lhs.getDefiningOp<memref::AllocaOp>();
    auto rhsAllocation = rhs.getDefiningOp<memref::AllocaOp>();
    auto accAllocation = accumulator.getDefiningOp<memref::AllocaOp>();
    if (!lhsAllocation || !rhsAllocation || !accAllocation || lhs == rhs || lhs == accumulator || rhs == accumulator ||
        lhsAllocation->getBlock() != loop->getBlock() || rhsAllocation->getBlock() != loop->getBlock() ||
        loop->isAncestor(accAllocation)) continue;
    bool privateInputs = true;
    for (Value input : {lhs, rhs}) {
      auto aliases = storage.aliases(input);
      if (!aliases.complete || llvm::any_of(aliases.users, [&](Operation *user) {
            return !loop->isAncestor(user);
          })) privateInputs = false;
    }
    for (Value scratch : {prepare.getOutput(), prepare.getScratch(), prepare.getReshaped()}) {
      if (!scratch) continue;
      Value allocation = owner(scratch);
      if (allocation == rhs) continue;
      if (!allocation || !allocation.getDefiningOp<memref::AllocaOp>() || !loop->isAncestor(allocation.getDefiningOp()))
        privateInputs = false;
    }
    if (!privateInputs) continue;
    auto lhsType = cast<MemRefType>(lhs.getType()), rhsType = cast<MemRefType>(rhs.getType());
    if (!lhsType.getLayout().isIdentity() || !rhsType.getLayout().isIdentity()) continue;
    // Keep storage ownership explicit. Both input slots live across the whole
    // loop; each stage finishes before the shared packing workspace is reused.
    Operation *previous = loop->getPrevNode();
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    Value nextLhs = allocate(b, loc, lhsType.getElementType(), lhsType.getShape(), dsa::nramSpace);
    Value nextRhs = allocate(b, loc, rhsType.getElementType(), rhsType.getShape(), dsa::nramSpace);
    Value doubledStep = b.create<arith::ConstantIndexOp>(loc, 2 * step);
    if (constant(loop.getUpperBound()) < 0) {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
      auto nonempty = b.create<scf::IfOp>(loc, active, false);
      b.setInsertionPointToStart(nonempty.thenBlock());
    }
    auto emitLoads = [&](Value coordinate, Value a, Value rawB) {
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), coordinate);
      mapping.map(lhs, a);
      mapping.map(rhs, rawB);
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (isa<dsa::LoadTileOp>(op)) {
          Operation *copy = b.clone(op, mapping);
          copy->setAttr("asynchronous", b.getBoolAttr(true));
        } else if (isa<dsa::StrideOp>(op) || op.getName().getDialectNamespace() == "arith") {
          b.clone(op, mapping);
        }
      }
    };
    auto emitCompute = [&](Value coordinate, Value a, Value rawB) {
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), coordinate);
      mapping.map(lhs, a);
      mapping.map(rhs, rawB);
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (isa<dsa::LoadTileOp, dsa::SynchronizeOp>(op)) continue;
        Operation *copy = b.clone(op, mapping);
        SmallVector<dsa::SynchronizeOp> fences;
        copy->walk([&](dsa::SynchronizeOp fence) { fences.push_back(fence); });
        for (auto fence : fences) fence.erase();
      }
      b.create<dsa::SynchronizeOp>(loc);
    };
    emitLoads(loop.getLowerBound(), lhs, rhs);
    b.create<dsa::SynchronizeOp>(loc);
    auto pipeline = b.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), doubledStep);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(pipeline.getBody());
      Value k = pipeline.getInductionVar();
      Value next = b.create<arith::AddIOp>(loc, k, loop.getStep());
      Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, loop.getUpperBound());
      auto prefetch = b.create<scf::IfOp>(loc, hasNext, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&prefetch.getThenRegion().front());
        emitLoads(next, nextLhs, nextRhs);
      }
      emitCompute(k, lhs, rhs);
      auto second = b.create<scf::IfOp>(loc, hasNext, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&second.getThenRegion().front());
        Value following = b.create<arith::AddIOp>(loc, next, loop.getStep());
        Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, loop.getUpperBound());
        auto prefetch = b.create<scf::IfOp>(loc, hasFollowing, false);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&prefetch.getThenRegion().front());
          emitLoads(following, lhs, rhs);
        }
        emitCompute(next, nextLhs, nextRhs);
      }
    }
    // Probe with the original loop still attached so its operand uses remain
    // well formed. The two loops are sequential and share the original slots.
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
      continue;
    }
    loop.erase();
  }
}

LogicalResult realizeGroupParticipants(func::FuncOp function) {
  bool collective = false;
  function.walk([&](dsa::GroupGatherRowsOp) { collective = true; });
  if (!collective) return success();
  OpBuilder b(function.getContext());
  b.setInsertionPointToStart(&function.front());
  Location loc = function.getLoc();
  Value memory = b.create<dsa::IsMemoryCoreOp>(loc, b.getI1Type());
  Value compute = b.create<arith::XOrIOp>(loc, memory, b.create<arith::ConstantIntOp>(loc, 1, 1));
  auto containsCollective = [](Operation *operation) {
    bool found = false;
    operation->walk([&](dsa::GroupGatherRowsOp) { found = true; });
    return found;
  };
  std::function<LogicalResult(Block &)> partition = [&](Block &block) -> LogicalResult {
    SmallVector<Operation *> original;
    for (Operation &operation : block) original.push_back(&operation);
    SmallVector<Operation *> pending;
    DenseSet<Operation *> privateOperations;
    auto flush = [&](Operation *before) -> LogicalResult {
      if (pending.empty()) return success();
      for (Operation *operation : pending) for (Value result : operation->getResults())
        for (Operation *user : result.getUsers()) {
          while (user && user->getBlock() != &block) user = user->getParentOp();
          if (!user || !privateOperations.contains(user))
            return operation->emitError("compute-local scalar escapes a collective supply interval");
        }
      b.setInsertionPoint(before);
      auto guard = b.create<scf::IfOp>(before->getLoc(), compute, false);
      for (Operation *operation : pending) operation->moveBefore(guard.thenBlock()->getTerminator());
      pending.clear(); privateOperations.clear();
      return success();
    };
    for (Operation *operation : original) {
      if (containsCollective(operation)) {
        if (failed(flush(operation))) return failure();
        if (!isa<dsa::GroupGatherRowsOp>(operation))
          for (Region &region : operation->getRegions()) for (Block &body : region)
            if (failed(partition(body))) return failure();
        continue;
      }
      if (operation->hasTrait<OpTrait::IsTerminator>()) {
        if (failed(flush(operation))) return failure();
        continue;
      }
      bool available = llvm::all_of(operation->getOperands(), [&](Value value) {
        return !privateOperations.contains(value.getDefiningOp());
      });
      bool scalarLoad = isa<dsa::LoadScalarOp>(operation) && available;
      bool movable = available && !operation->getNumRegions() &&
          (isa<memref::AllocaOp, memref::ReinterpretCastOp>(operation) || isMemoryEffectFree(operation));
      if (scalarLoad) {
        if (failed(flush(operation))) return failure();
      } else if (movable) {
        // Static storage views and scalar expressions can precede the local
        // interval. Their operands are already outside that interval.
        if (!pending.empty()) operation->moveBefore(pending.front());
      } else {
        pending.push_back(operation); privateOperations.insert(operation);
      }
    }
    return success();
  };
  return partition(function.front());
}

LogicalResult composeLocalStorage(func::FuncOp function,
    llvm::function_ref<LogicalResult()> cleanup) {
  while (true) {
    bool changed = dsa::foldUniformLocalValues(function);
    changed |= dsa::reuseLocalMemoryValues(function);
    changed |= dsa::composeLocalTransfers(function);
    changed |= dsa::eliminateUnreadLocalWrites(function);
    if (!changed) return success();
    if (failed(cleanup())) return failure();
  }
}

LogicalResult composeLocalProgram(ModuleOp module,
    llvm::function_ref<LogicalResult()> cleanup) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  if (failed(cleanup())) return failure();
  while (dsa::reuseLocalMemoryValues(function))
    if (failed(cleanup())) return failure();
  if (dsa::forwardIndexExpressions(function) && failed(cleanup())) return failure();
  if (dsa::realizeRangeComparisons(function) && failed(cleanup())) return failure();
  if (dsa::foldRangeCounts(function) && failed(cleanup())) return failure();
  if (realizeAffineRanges(function, config) && failed(cleanup())) return failure();
  if (realizeFullWidthMasks(function, config) && failed(cleanup())) return failure();
  while (dsa::eliminateUnreadLocalWrites(function)) if (failed(cleanup())) return failure();
  if (bindBroadcastOperands(function) && failed(cleanup())) return failure();
  if (realizeRowBroadcasts(function, config) && failed(cleanup())) return failure();
  if (failed(composeLocalStorage(function, cleanup))) return failure();
  while (propagateLocalTransposeSupply(function, config))
    if (failed(cleanup()) || failed(composeLocalStorage(function, cleanup))) return failure();
  if (retainNarrowExtremaInputs(function, config) && failed(cleanup())) return failure();
  if (failed(composeLocalStorage(function, cleanup))) return failure();
  if (dsa::reuseGatherOffsets(function) && failed(cleanup())) return failure();
  if (vectorizeIndexLoops(function, config) && failed(cleanup())) return failure();
  realizeGatherWorkspace(function, config);
  if (dsa::batchPointwiseTasks(function) && failed(cleanup())) return failure();
  if (coarsenLocalPrograms(function, config) && failed(cleanup())) return failure();
  return success();
}

LogicalResult reuseStorageScopes(ModuleOp module,
    llvm::function_ref<LogicalResult()> cleanup) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  if (failed(cleanup()) || failed(composeLocalStorage(function, cleanup))) return failure();
  while (placeInvariantSupply(function, config) || reuseCompletedSupply(function, config)) {
    if (failed(cleanup()) || failed(composeLocalStorage(function, cleanup))) return failure();
  }
  while (initializeMatrixStorage(function)) {
    if (failed(cleanup()) || failed(composeLocalStorage(function, cleanup))) return failure();
  }
  if (reuseConsumedBinaryInputs(function, config) && failed(cleanup())) return failure();
  coalesceTileLoads(function, config);
  return success();
}

LogicalResult scheduleProgramSupply(ModuleOp module) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  // Scalar NRAM accesses preserve program order within a task. Complete them
  // before entering a bulk operation, and complete bulk operations before any
  // scalar consumer or allocation reuse. A fence inside every scalar store or
  // scalar carry copy would serialize each element of broadcast/reduce loops.
  SmallVector<Operation *> localEffects;
  function.walk([&](Operation *op) {
    if (isa<dsa::LoadTileOp, dsa::GatherPlanOp, dsa::GatherRowsOp, dsa::StoreTileOp, dsa::FillOp, dsa::IotaOp,
            dsa::IndexLayoutOp, dsa::IndexBinaryOp, dsa::BroadcastRowsOp, dsa::TransposeOp, dsa::SelectOp, dsa::MaskedFillOp, dsa::UnaryOp,
            dsa::BinaryOp, dsa::CastOp, dsa::CompareOp, dsa::CompareRangeOp, dsa::CompareRampOp, dsa::DivideCastOp, dsa::DivideRNOp, dsa::ReduceOp,
            dsa::PrepareMatrixOp, dsa::PrepareMatrixViewOp, dsa::MatrixTileOp>(op))
      localEffects.push_back(op);
    if (auto copy = dyn_cast<memref::CopyOp>(op))
      if (cast<MemRefType>(copy.getSource().getType()).getNumElements() > 1)
        localEffects.push_back(op);
  });
  DominanceInfo transferDominance(function);
  auto canOverlapPrefetch = [&](Operation *compute) {
    if (!isa<dsa::MatrixTileOp>(compute) || !compute->getPrevNode()) return false;
    Operation *previous = compute->getPrevNode();
    SmallVector<Value> pending;
    bool eligible = true;
    previous->walk([&](Operation *op) {
      if (auto load = dyn_cast<dsa::LoadTileOp>(op)) {
        if (!load.getAsynchronous()) eligible = false;
        pending.push_back(load.getOutput());
      } else if (op->getNumRegions()) {
        if (!isa<scf::IfOp>(op)) eligible = false;
      } else if (!isMemoryEffectFree(op)) eligible = false;
    });
    if (!eligible || pending.empty()) return false;
    dsa::StorageAnalysis storage(function);
    for (Value input : compute->getOperands()) {
      Value allocation = storage.uniqueOrigin(input);
      if (!allocation || !allocation.getDefiningOp<memref::AllocaOp>() ||
          !transferDominance.dominates(allocation.getDefiningOp(), previous)) return false;
      for (Value output : pending) if (!storage.disjoint(output, input)) return false;
    }
    return true;
  };
  for (Operation *op : localEffects) {
    // An explicitly asynchronous transfer is consumed at the next explicit
    // work-unit/group fence. Independent current matrix inputs may execute
    // while the alternate input buffer is being supplied.
    if (auto load = dyn_cast<dsa::LoadTileOp>(op); load && load.getAsynchronous()) continue;
    if (auto gather = dyn_cast<dsa::GatherRowsOp>(op); gather && gather.getAsynchronous()) continue;
    OpBuilder builder(op);
    if ((!op->getPrevNode() || !isa<dsa::SynchronizeOp>(op->getPrevNode())) && !canOverlapPrefetch(op))
      builder.create<dsa::SynchronizeOp>(op->getLoc());
    builder.setInsertionPointAfter(op);
    builder.create<dsa::SynchronizeOp>(op->getLoc());
  }
  // Consecutive native Compute operations obey stream dependencies, including
  // storage reuse. Keep fences at scalar access, transfer and helper boundaries.
  auto nativeCompute = [](Operation *op) {
    if (isa<dsa::IndexBinaryOp, dsa::MaskedFillOp>(op)) return true;
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
      auto type = cast<MemRefType>(binary.getOutput().getType());
      return !binary.getApproximate() && !binary.getFlushToZero() && type.getNumElements() >= 64 &&
          (type.getElementType().isF16() || type.getElementType().isF32()) &&
          (binary.getKind() == BinaryOperator::Add || binary.getKind() == BinaryOperator::Subtract ||
           binary.getKind() == BinaryOperator::Multiply);
    }
    if (auto conversion = dyn_cast<dsa::CastOp>(op)) {
      auto input = cast<MemRefType>(conversion.getInput().getType());
      auto output = cast<MemRefType>(conversion.getOutput().getType());
      return input.getNumElements() >= 64 &&
          ((input.getElementType().isF16() && output.getElementType().isF32()) ||
           (input.getElementType().isF32() && output.getElementType().isF16()));
    }
    if (auto unary = dyn_cast<dsa::UnaryOp>(op))
      return unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() && unary.getFlushToZero() &&
          unary.getScratch() && cast<MemRefType>(unary.getScratch().getType()).getElementType().isInteger(32);
    return false;
  };
  function.walk([&](Operation *compute) {
    if (!nativeCompute(compute)) return;
    SmallVector<Operation *> fences;
    Operation *previous = compute->getPrevNode();
    while (previous) {
      if (isa<dsa::SynchronizeOp>(previous)) fences.push_back(previous);
      else if (!isa<memref::AllocaOp>(previous) && !isMemoryEffectFree(previous)) break;
      previous = previous->getPrevNode();
    }
    if (previous && nativeCompute(previous))
      for (Operation *fence : fences) fence->erase();
  });
  pipelineLocalSupply(function, config);
  pipelineMatrixLoads(function, config);
  SmallVector<dsa::SynchronizeOp> fences;
  function.walk([&](dsa::SynchronizeOp fence) { fences.push_back(fence); });
  for (auto fence : fences) {
    Operation *previous = fence->getPrevNode();
    while (previous && (isa<memref::AllocaOp>(previous) || isMemoryEffectFree(previous)))
      previous = previous->getPrevNode();
    auto earlier = dyn_cast_or_null<dsa::SynchronizeOp>(previous);
    if (!earlier) continue;
    // No command was issued between these waits. A full wait subsumes a local
    // wait; IO completion remains explicit at every asynchronous slot boundary.
    if (!earlier.getLocalOnly() || fence.getLocalOnly()) fence.erase();
    else earlier.erase();
  }
  if (failed(realizeGroupParticipants(function))) return failure();
  return success();
}
} // namespace intent::bangc
