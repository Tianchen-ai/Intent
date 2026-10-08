#include "PassSupport.h"
#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/IR/ExecutionRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::dsa {
namespace {
std::optional<int64_t> integer(Value value) {
  APInt bits;
  if (matchPattern(value, m_ConstantInt(&bits)) && bits.isSignedIntN(64)) return bits.getSExtValue();
  return std::nullopt;
}

bool fourAligned(Value value) {
  if (auto constant = integer(value)) return *constant % 4 == 0;
  if (auto multiply = value.getDefiningOp<arith::MulIOp>())
    return fourAligned(multiply.getLhs()) || fourAligned(multiply.getRhs());
  return false;
}

bool adjacentIntervals(GatherRowsOp gather, ExecutionRelations &relations,
                       StorageAnalysis &storage) {
  if (gather.getAsynchronous() || gather.getPlan() || !relations.hasUniformControl(gather) ||
      !relations.isUniform(gather.getRows()) || !relations.isUniform(gather.getColumns())) return false;
  auto source = cast<MemRefType>(gather.getSource().getType());
  auto output = cast<MemRefType>(gather.getOutput().getType());
  auto view = relations.readonlyView(gather.getSource());
  if (!view || source.getRank() < 2 || !view.getConstraints().getHasStrides() ||
      !output.getLayout().isIdentity() || output.getMemorySpaceAsInt() != nramSpace) return false;
  int64_t rows = output.getDimSize(0), columns = output.getDimSize(1);
  unsigned headAxis = source.getRank() - 2, columnAxis = source.getRank() - 1;
  auto headStride = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[headAxis]);
  auto columnStride = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[columnAxis]);
  if (rows < 64 || rows % 64 || rows > 65536 || columns <= 0 || source.getDimSize(columnAxis) != columns ||
      source.getDimSize(headAxis) <= 0 || source.getDimSize(headAxis) % 4 || !headStride || !columnStride ||
      headStride.getInt() != columns || columnStride.getInt() != 1 ||
      integer(gather.getColumnStride()) != std::optional<int64_t>(1) ||
      integer(gather.getColumns()) != std::optional<int64_t>(columns)) return false;
  int64_t elementBytes = llvm::divideCeil(source.getElementType().getIntOrFloatBitWidth(), 8u);
  if (columns > ((3968 * 1024) / rows - 32) / (4 * elementBytes)) return false;

  // Require one complete, current offset snapshot immediately before the read.
  // No other writer or alias can substitute older lane-dependent row addresses.
  Value offsets = gather.getRowOffsets();
  auto allocation = offsets.getDefiningOp<memref::AllocaOp>();
  auto loop = dyn_cast_or_null<scf::ForOp>(gather->getPrevNode());
  if (!allocation || !loop || !loop.getInitArgs().empty() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
      !matchPattern(loop.getStep(), m_One()) || loop.getUpperBound() != gather.getRows() ||
      !relations.hasUniformControl(loop)) return false;
  auto store = dyn_cast_or_null<memref::StoreOp>(storage.uniqueWriter(offsets));
  if (!store ||
      !detail::sameCompleteView(storage, store.getMemref(), offsets) ||
      store->getBlock() != loop.getBody() || store.getIndices().size() != 2 ||
      !matchPattern(store.getIndices()[0], m_Zero()) ||
      store.getIndices()[1] != loop.getInductionVar())
    return false;
  for (Operation *user : storage.aliases(offsets).users)
    if (user != store && user != gather && !isBufferStorageAliasOperation(user))
      return false;
  Value outputOrigin = storage.uniqueOrigin(gather.getOutput());
  if (!outputOrigin || !outputOrigin.getDefiningOp<memref::AllocaOp>() ||
      !storage.disjoint(outputOrigin, offsets))
    return false;
  auto coefficient = relations.participantCoefficient(store.getValue());
  return coefficient && *coefficient == columns;
}

void realizeGroup(scf::ForOp work, ArrayRef<GatherRowsOp> gathers, ExecutionRelations &relations) {
  SmallVector<std::pair<arith::DivSIOp, int64_t>> quotients;
  work.walk([&](arith::DivSIOp divide) {
    if (auto divisor = relations.groupedQuotientDivisor(divide.getResult()))
      quotients.emplace_back(divide, *divisor);
  });
  auto function = work->getParentOfType<func::FuncOp>();
  OpBuilder b(work);
  Location loc = work.getLoc();
  auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
  auto allocate = [&](Type element, int64_t rows, int64_t columns, int64_t space = nramSpace) -> Value {
    auto type = MemRefType::get({rows, columns}, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
    auto allocation = b.create<memref::AllocaOp>(loc, type); allocation.setAlignment(128); return allocation;
  };
  Value originalTask = work.getInductionVar();
  Value group = b.create<GroupIdOp>(loc, b.getIndexType()), groups = b.create<GroupCountOp>(loc, b.getIndexType());
  Value local = b.create<LocalIdOp>(loc, b.getIndexType()), memory = b.create<IsMemoryCoreOp>(loc, b.getI1Type());
  Value lane = b.create<arith::SelectOp>(loc, memory, index(0), local);
  Value total = b.create<arith::DivSIOp>(loc, work.getUpperBound(), index(4));
  work.setLowerBound(group); work.setUpperBound(total); work.setStep(groups);
  DenseSet<Operation *> groupOperations;
  for (auto [quotient, divisor] : quotients) {
    b.setInsertionPoint(quotient);
    auto grouped = b.create<arith::DivSIOp>(loc, originalTask, index(divisor));
    groupOperations.insert(grouped);
    quotient.replaceAllUsesWith(grouped.getResult()); quotient.erase();
  }
  b.setInsertionPointToStart(work.getBody());
  Value first = b.create<arith::MulIOp>(loc, originalTask, index(4));
  groupOperations.insert(first.getDefiningOp());
  Value task = b.create<arith::AddIOp>(loc, first, lane);
  originalTask.replaceUsesWithIf(task, [&](OpOperand &use) { return !groupOperations.contains(use.getOwner()); });
  function->setAttr("intent_dsa.group_width", b.getI64IntegerAttr(4));
  DenseMap<int64_t, Value> indices;
  for (auto gather : gathers) {
    auto type = cast<MemRefType>(gather.getOutput().getType());
    int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
    Value ramp = indices.lookup(rows);
    if (!ramp) {
      b.setInsertionPointToStart(&function.front());
      ramp = allocate(b.getF32Type(), 1, rows); b.create<IotaOp>(loc, ramp); indices[rows] = ramp;
    }
    b.setInsertionPoint(gather);
    Value descriptor = allocate(b.getI64Type(), 3, rows);
    b.create<GatherPlanOp>(loc, gather.getRowOffsets(), gather.getRows(), ramp, descriptor);
    Value data = allocate(type.getElementType(), rows, 4 * columns, sharedSpace);
    Value metadata = allocate(b.getI64Type(), 4, rows, sharedSpace);
    b.create<GroupGatherRowsOp>(loc, gather.getSource(), gather.getRowOffsets(), descriptor,
        gather.getOutput(), data, metadata, gather.getRows(), lane);
    gather.erase();
  }
}
} // namespace

LogicalResult realizeCollectiveGatherSupply(func::FuncOp function) {
  auto config = function->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  if (function->hasAttr("intent_dsa.group_width") || config.getTasks() < 4 || config.getTasks() % 4) return success();
  SmallVector<scf::ForOp> worksets;
  function.walk([&](scf::ForOp loop) {
    if (loop.getLowerBound().getDefiningOp<TaskIdOp>()) worksets.push_back(loop);
  });
  if (worksets.size() != 1) return success();
  auto work = worksets.front();
  if (work->getParentOp() != function || !work.getStep().getDefiningOp<TaskCountOp>() ||
      !work.getInitArgs().empty() || !fourAligned(work.getUpperBound()) ||
      !work.getLowerBound().hasOneUse() || !work.getStep().hasOneUse()) return success();
  bool isolated = true;
  function.walk([&](Operation *operation) {
    if (!isa<TaskIdOp, TaskCountOp>(operation)) return;
    Value query = operation->getResult(0);
    if (query != work.getLowerBound() && query != work.getStep() && !query.use_empty()) isolated = false;
  });
  if (!isolated) return success();
  ExecutionRelations relations(work, 4);
  StorageAnalysis storage(function);
  SmallVector<GatherRowsOp> gathers;
  work.walk([&](GatherRowsOp gather) {
    if (adjacentIntervals(gather, relations, storage)) gathers.push_back(gather);
  });
  if (!gathers.empty()) realizeGroup(work, gathers, relations);
  return success();
}
} // namespace intent::dsa

namespace intent::dsa {
#define GEN_PASS_DEF_DSACOLLECTIVEGATHERSUPPLY
#include "Intent/Dialect/DSA/Transforms/Passes.h.inc"

namespace {
struct CollectiveGatherSupplyPass : impl::DSACollectiveGatherSupplyBase<CollectiveGatherSupplyPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyRealizedProgram(module)))
      return signalPassFailure();
    auto function = *module.getOps<func::FuncOp>().begin();
    auto result = realizeCollectiveGatherSupply(function);
    if (failed(detail::finishTransform(module, getArgument(), result)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::dsa
