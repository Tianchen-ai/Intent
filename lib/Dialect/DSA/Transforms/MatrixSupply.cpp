#include "PassSupport.h"
#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "Intent/Dialect/DSA/Transforms/MatrixPanels.h"
#include "MatrixSupplyRelations.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <functional>
#include <optional>

using namespace mlir;
namespace intent::dsa {
namespace {
// This match refers only to the existing tiled program, including its epilogue.
// It does not retain a KIR clone or authorize an alternative algorithm.
struct MatrixSupplyMatch {
  scf::ForOp work, reduction;
  MatMulOp matrix;
  LoadTileOp lhs, rhs;
  detail::MatrixWorksetGeometry geometry;
};
std::optional<MatrixSupplyMatch> matchMatrixSupply(scf::ForOp work) {
  if (!work.getLowerBound().getDefiningOp<TaskIdOp>() ||
      !work.getStep().getDefiningOp<TaskCountOp>() || !work.getInitArgs().empty()) return std::nullopt;
  SmallVector<MatMulOp> matrices;
  work.walk([&](MatMulOp matrix) { matrices.push_back(matrix); });
  if (matrices.size() != 1) return std::nullopt;
  MatMulOp matrix = matrices.front();
  auto function = work->getParentOfType<func::FuncOp>();
  // Group launch is function-wide. All observable work must belong to this
  // complete workset; unrelated tasks cannot start executing on the new memory
  // participants merely because one matrix traversal was cooperative.
  if (function->hasAttr("intent_dsa.group_width") ||
      work->getBlock() != &function.front() ||
      llvm::any_of(function.front(), [&](Operation &operation) {
        return &operation != work.getOperation() &&
               !operation.hasTrait<OpTrait::IsTerminator>() &&
               (!isMemoryEffectFree(&operation) || !isSpeculatable(&operation));
      })) return std::nullopt;
  StorageAnalysis storage(function);
  auto reduction = dyn_cast<scf::ForOp>(matrix->getParentOp());
  if (!reduction || reduction->getBlock() != work.getBody() || !reduction.getInitArgs().empty() ||
      matrix.getRhsTransposed()) return std::nullopt;
  LoadTileOp lhs, rhs;
  SmallVector<FillOp> inputInitializers;
  for (Operation &operation : reduction.getBody()->without_terminator()) {
    if (&operation == matrix.getOperation()) continue;
    if (auto load = dyn_cast<LoadTileOp>(operation)) {
      if (load.getAsynchronous() || !isa<BlockArgument>(load.getSource())) return std::nullopt;
      if (detail::sameCompleteView(storage, load.getOutput(),
                                   matrix.getLhs()) &&
          !lhs)
        lhs = load;
      else if (detail::sameCompleteView(storage, load.getOutput(),
                                        matrix.getRhs()) &&
               !rhs)
        rhs = load;
      else return std::nullopt;
    } else if (auto fill = dyn_cast<FillOp>(operation)) {
      inputInitializers.push_back(fill);
    } else if (auto allocation = dyn_cast<memref::AllocaOp>(operation)) {
      if (allocation.getResult() != storage.uniqueOrigin(matrix.getLhs()) &&
          allocation.getResult() != storage.uniqueOrigin(matrix.getRhs()))
        return std::nullopt;
    } else if (!isMemoryEffectFree(&operation)) return std::nullopt;
  }
  if (!lhs || !rhs ||
      !lhs->isBeforeInBlock(matrix) || !rhs->isBeforeInBlock(matrix)) return std::nullopt;
  auto external = [](Value source) {
    auto type = dyn_cast<MemRefType>(source.getType());
    return type && type.getRank() == 2 && type.getMemorySpaceAsInt() == 0;
  };
  if (!external(lhs.getSource()) || !external(rhs.getSource())) return std::nullopt;
  auto lhsSourceType = cast<MemRefType>(lhs.getSource().getType());
  auto rhsSourceType = cast<MemRefType>(rhs.getSource().getType());
  Type element = lhsSourceType.getElementType();
  if (!element.isF16() && !element.isBF16() && !element.isF32()) return std::nullopt;
  auto geometry = detail::queryCompleteMatrixWorkset(work, reduction, matrix, lhs, rhs);
  if (rhsSourceType.getElementType() != element || !geometry) return std::nullopt;
  // The geometry comes from current types, not a bounded profile tuple. This
  // checked native footprint also bounds twice the input rows and panel bytes
  // used by the cooperative double-buffer construction below.
  if (!matrixPanelStorage(element, geometry->rows, geometry->columns, geometry->depth))
    return std::nullopt;
  if (!storage.disjoint(matrix.getLhs(), matrix.getRhs()) ||
      !storage.disjoint(matrix.getAccumulator(), matrix.getLhs()) ||
      !storage.disjoint(matrix.getAccumulator(), matrix.getRhs()))
    return std::nullopt;
  SmallVector<Value> buffers{matrix.getLhs(), matrix.getRhs(), matrix.getAccumulator()};
  SmallVector<SmallVector<int64_t>> shapes{{geometry->rows, geometry->depth},
      {geometry->depth, geometry->columns}, {geometry->rows, geometry->columns}};
  for (auto [buffer, shape] : llvm::zip(buffers, shapes)) {
    Value origin = storage.uniqueOrigin(buffer);
    auto allocation =
        origin ? origin.getDefiningOp<memref::AllocaOp>() : memref::AllocaOp();
    if (!allocation || (allocation->getBlock() != work.getBody() &&
        (buffer == matrix.getAccumulator() ||
         allocation->getBlock() != reduction.getBody()))) return std::nullopt;
    auto type = cast<MemRefType>(buffer.getType());
    if (type.getShape() != ArrayRef<int64_t>(shape) ||
        !type.getLayout().isIdentity() ||
        type.getMemorySpaceAsInt() != nramSpace ||
        !isCompleteStorageViewOf(buffer, origin))
      return std::nullopt;
  }
  // Only the accumulator persists through the K traversal. All other work in
  // the outer body must be its pointwise epilogue or pure coordinate setup.
  FillOp initial;
  for (Operation &operation : work.getBody()->without_terminator()) {
    if (&operation == reduction.getOperation()) break;
    if (auto fill = dyn_cast<FillOp>(operation)) {
      if (detail::sameCompleteView(storage, fill.getOutput(),
                                   matrix.getLhs()) ||
          detail::sameCompleteView(storage, fill.getOutput(),
                                   matrix.getRhs())) {
        inputInitializers.push_back(fill);
        continue;
      }
      FloatAttr zero;
      if (initial ||
          !detail::sameCompleteView(storage, fill.getOutput(),
                                    matrix.getAccumulator()) ||
          !matchPattern(fill.getValue(), m_Constant(&zero)) ||
          !zero.getValue().isZero())
        return std::nullopt;
      initial = fill;
    } else if (!isa<memref::AllocaOp>(operation) && !isMemoryEffectFree(&operation)) return std::nullopt;
  }
  if (!initial) return std::nullopt;
  // A synchronous LoadTile writes the complete private tile, including zero
  // padding. Its input initialization may be in either the task or K scope,
  // provided no other consumer observes that earlier value or aliases storage.
  for (FillOp fill : inputInitializers) {
    LoadTileOp load =
        detail::sameCompleteView(storage, fill.getOutput(), matrix.getLhs())
            ? lhs
        : detail::sameCompleteView(storage, fill.getOutput(), matrix.getRhs())
            ? rhs
            : LoadTileOp{};
    if (!load || (fill->getBlock() == reduction.getBody()
        ? !fill->isBeforeInBlock(load) : !fill->isBeforeInBlock(reduction)))
      return std::nullopt;
  }
  for (LoadTileOp load : {lhs, rhs}) {
    auto aliases = storage.aliases(storage.uniqueOrigin(load.getOutput()));
    if (!aliases.complete)
      return std::nullopt;
    for (Operation *user : aliases.users) {
      if (user == load.getOperation() || user == matrix.getOperation())
        continue;
      if (isBufferStorageAliasOperation(user))
        continue;
      auto fill = dyn_cast<FillOp>(user);
      if (!fill || !llvm::is_contained(inputInitializers, fill)) return std::nullopt;
    }
  }
  DenseSet<Value> available{matrix.getAccumulator(), work.getInductionVar()};
  DenseSet<Value> visiting;
  std::function<bool(Value)> canCapture = [&](Value value) {
    if (available.contains(value)) return true;
    if (auto argument = dyn_cast<BlockArgument>(value))
      return !work->isAncestor(argument.getOwner()->getParentOp()) &&
          argument.getOwner() != work.getBody();
    Operation *definition = value.getDefiningOp();
    if (!definition || !work->isAncestor(definition)) return true;
    if (!visiting.insert(value).second || definition->getNumRegions() || !isMemoryEffectFree(definition)) return false;
    bool legal = llvm::all_of(definition->getOperands(), canCapture);
    visiting.erase(value);
    if (legal) available.insert(value);
    return legal;
  };
  for (Operation *operation = reduction->getNextNode(); operation && !operation->hasTrait<OpTrait::IsTerminator>();
       operation = operation->getNextNode()) {
    if (operation->getNumRegions() ||
        (!isa<memref::AllocaOp, StoreTileOp, UnaryOp, BinaryOp, CastOp, SelectOp, FillOp>(operation) &&
         !isMemoryEffectFree(operation))) return std::nullopt;
    if (!llvm::all_of(operation->getOperands(), canCapture)) return std::nullopt;
    available.insert(operation->result_begin(), operation->result_end());
  }
  return MatrixSupplyMatch{work, reduction, matrix, lhs, rhs, *geometry};
}

class MatrixSupplyRewrite {
public:
  explicit MatrixSupplyRewrite(MatrixSupplyMatch match, ConfigurationAttr config)
      : match(match), config(config), function(match.work->getParentOfType<func::FuncOp>()), b(match.work) {}
  LogicalResult run(std::optional<StreamedMatrixPanel> resident) {
    auto matrix = match.matrix;
    Location loc = matrix.getLoc();
    Value lhsSource = match.lhs.getSource(), rhsSource = match.rhs.getSource();
    Value M = b.createOrFold<memref::DimOp>(loc, lhsSource, 0);
    Value N = b.createOrFold<memref::DimOp>(loc, rhsSource, 1);
    Value K = b.createOrFold<memref::DimOp>(loc, lhsSource, 1);
    int64_t rows = match.geometry.rows, columns = match.geometry.columns;
    int64_t matrixDepth = resident ? resident->depth : match.geometry.depth;
    Value tm = index(loc, rows), tn = index(loc, columns), tk = index(loc, matrixDepth);
    Value gridM = b.create<arith::CeilDivSIOp>(loc, M, tm), gridN = b.create<arith::CeilDivSIOp>(loc, N, tn);
    auto aType = cast<MemRefType>(lhsSource.getType());
    Type matrixElement = aType.getElementType();
    auto emitOutput = [&](Value accumulator, Value m, Value n, Value, Value) {
      IRMapping mapping;
      mapping.map(match.matrix.getAccumulator(), accumulator);
      Value mi = b.create<arith::DivSIOp>(loc, m, tm);
      Value ni = b.create<arith::DivSIOp>(loc, n, tn);
      mapping.map(match.work.getInductionVar(), add(loc, mul(loc, mi, gridN), ni));
      std::function<Value(Value)> operand = [&](Value value) -> Value {
        if (Value replacement = mapping.lookupOrNull(value)) return replacement;
        Operation *definition = value.getDefiningOp();
        if (!definition || !match.work->isAncestor(definition)) return value;
        for (Value input : definition->getOperands()) mapping.map(input, operand(input));
        b.clone(*definition, mapping);
        return mapping.lookup(value);
      };
      for (Operation *operation = match.reduction->getNextNode(); !operation->hasTrait<OpTrait::IsTerminator>();
           operation = operation->getNextNode()) {
        for (Value input : operation->getOperands()) mapping.map(input, operand(input));
        b.clone(*operation, mapping);
      }
      return success();
    };
    if (resident) {
      // A resident RHS panel is reused across output-row blocks. The group's
      // four column programs consume the same streamed LHS row block.
      function->setAttr("intent_dsa.group_width", b.getI64IntegerAttr(4));
      Value groupId = b.create<dsa::GroupIdOp>(loc, b.getIndexType());
      Value groupCount = b.create<dsa::GroupCountOp>(loc, b.getIndexType());
      Value localId = b.create<dsa::LocalIdOp>(loc, b.getIndexType());
      Value memoryCore = b.create<dsa::IsMemoryCoreOp>(loc, b.getI1Type());
      Value computeCore = b.create<arith::XOrIOp>(loc, memoryCore, b.create<arith::ConstantIntOp>(loc, 1, 1));
      Value groupsN = b.create<arith::CeilDivSIOp>(loc, gridN, index(loc, 4));
      auto sharedType = MemRefType::get({rows, matrixDepth}, matrixElement,
          MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(dsa::sharedSpace));
      auto shared = b.create<memref::AllocaOp>(loc, sharedType);
      shared.setAlignment(128);
      auto alternate = b.create<memref::AllocaOp>(loc, sharedType);
      alternate.setAlignment(128);
      auto when = [&](Value condition, const std::function<LogicalResult()> &body) -> LogicalResult {
        auto branch = b.create<scf::IfOp>(loc, condition, false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(branch.thenBlock());
        return body();
      };
      int64_t sliceN = resident->sliceColumns, sliceK = resident->sliceDepth;
      return loop(loc, groupId, groupsN, groupCount, [&](Value group) -> LogicalResult {
        Value ni = add(loc, mul(loc, group, index(loc, 4)), localId);
        Value n0 = mul(loc, ni, tn);
        Value cols = b.create<arith::MaxSIOp>(loc, index(loc, 0), b.create<arith::MinSIOp>(loc, sub(loc, N, n0), tn));
        Value active = b.create<arith::AndIOp>(loc, computeCore,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ni, gridN));
        auto packedType = MemRefType::get({matrixDepth, columns}, matrixElement,
            MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(dsa::matrixSpace));
        auto packed = b.create<memref::AllocaOp>(loc, packedType);
        packed.setAlignment(128);
        if (failed(when(active, [&]() -> LogicalResult {
          Value raw = allocate(loc, matrixElement, sliceK, sliceN);
          Value transposed = allocate(loc, matrixElement, sliceN, sliceK);
          Value br = stride(loc, rhsSource, 0), bc = stride(loc, rhsSource, 1);
          b.create<dsa::PrepareMatrixViewOp>(loc, rhsSource, packed, raw, transposed, mul(loc, n0, bc), br, bc, cols);
          return success();
        }))) return failure();
        auto supply = [&](Value m0, Value slot) -> LogicalResult {
          return when(memoryCore, [&]() -> LogicalResult {
            Value rows = b.create<arith::MinSIOp>(loc, sub(loc, M, m0), tm);
            Value ar = stride(loc, lhsSource, 0), ac = stride(loc, lhsSource, 1);
            b.create<dsa::StageTileOp>(loc, lhsSource, slot, mul(loc, m0, ar), ar, ac, rows, K);
            return success();
          });
        };
        const bool pipelineLocal = resident->pipeline;
        Value local0, local1;
        if (pipelineLocal) {
          local0 = allocate(loc, matrixElement, rows, matrixDepth);
          local1 = allocate(loc, matrixElement, rows, matrixDepth);
        }
        auto compute = [&](Value m0, Value slot, Value nextShared = Value{}, Value nextLocal = Value{}, Value nextRow = Value{}) -> LogicalResult {
          return when(active, [&]() -> LogicalResult {
            Value rows = b.create<arith::MinSIOp>(loc, sub(loc, M, m0), tm);
            Value lhs = slot;
            if (!pipelineLocal) {
              lhs = allocate(loc, matrixElement, match.geometry.rows, matrixDepth);
              b.create<dsa::LoadTileOp>(loc, slot, lhs, index(loc, 0), tk, index(loc, 1), rows, K);
            }
            Value accumulator = allocate(loc, b.getF32Type(), match.geometry.rows, columns);
            b.create<dsa::FillOp>(loc, accumulator, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)));
            if (nextShared && failed(when(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nextRow, M), [&]() {
              Value nextRows = b.create<arith::MinSIOp>(loc, sub(loc, M, nextRow), tm);
              b.create<dsa::LoadTileOp>(loc, nextShared, nextLocal, index(loc, 0), tk, index(loc, 1), nextRows, K, b.getBoolAttr(true));
              return success();
            }))) return failure();
            b.create<dsa::MatrixTileOp>(loc, lhs, packed, accumulator);
            return emitOutput(accumulator, m0, n0, rows, cols);
          });
        };
        if (failed(supply(index(loc, 0), shared))) return failure();
        if (pipelineLocal) {
          Value second = tm;
          if (failed(when(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, second, M),
                          [&]() { return supply(second, alternate); }))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          if (failed(when(active, [&]() {
            Value rows = b.create<arith::MinSIOp>(loc, M, tm);
            b.create<dsa::LoadTileOp>(loc, shared, local0, index(loc, 0), tk, index(loc, 1), rows, K);
            return success();
          }))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          return loop(loc, index(loc, 0), M, index(loc, 2 * rows), [&](Value m0) -> LogicalResult {
            Value next = add(loc, m0, tm), following = add(loc, next, tm);
            Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, M);
            Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, M);
            if (failed(when(hasFollowing, [&]() { return supply(following, shared); }))) return failure();
            if (failed(compute(m0, local0, alternate, local1, next))) return failure();
            b.create<dsa::GroupSynchronizeOp>(loc);
            return when(hasNext, [&]() -> LogicalResult {
              Value third = add(loc, following, tm);
              if (failed(when(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, third, M),
                              [&]() { return supply(third, alternate); }))) return failure();
              if (failed(compute(next, local1, shared, local0, following))) return failure();
              b.create<dsa::GroupSynchronizeOp>(loc);
              return success();
            });
          });
        }
        b.create<dsa::GroupSynchronizeOp>(loc);
        return loop(loc, index(loc, 0), M, index(loc, 2 * rows), [&](Value m0) -> LogicalResult {
          Value next = add(loc, m0, tm);
          Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, M);
          if (failed(when(hasNext, [&]() { return supply(next, alternate); }))) return failure();
          if (failed(compute(m0, shared))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          return when(hasNext, [&]() -> LogicalResult {
            Value following = add(loc, next, tm);
            Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, M);
            if (failed(when(hasFollowing, [&]() { return supply(following, shared); }))) return failure();
            if (failed(compute(next, alternate))) return failure();
            b.create<dsa::GroupSynchronizeOp>(loc);
            return success();
          });
        });
      });
    }
    {
      // Four independent M tiles reuse one RHS panel supplied by the group's
      // memory participant. Local matrix computation remains explicit below.
      function->setAttr("intent_dsa.group_width", b.getI64IntegerAttr(4));
      Value groupId = b.create<dsa::GroupIdOp>(loc, b.getIndexType());
      Value groupCount = b.create<dsa::GroupCountOp>(loc, b.getIndexType());
      Value localId = b.create<dsa::LocalIdOp>(loc, b.getIndexType());
      Value memoryCore = b.create<dsa::IsMemoryCoreOp>(loc, b.getI1Type());
      Value computeCore = b.create<arith::XOrIOp>(loc, memoryCore, b.create<arith::ConstantIntOp>(loc, 1, 1));
      Value groupsM = b.create<arith::CeilDivSIOp>(loc, gridM, index(loc, 4));
      Type element = aType.getElementType();
      int64_t panelBytes = match.geometry.depth * columns * (element.getIntOrFloatBitWidth() / 8);
      int64_t stageTiles = std::min<int64_t>(8, (3968 * 1024) / (2 * panelBytes));
      if (!stageTiles) return matrix.emitError("matrix panel exceeds execution-group shared storage");
      int64_t stageRows = stageTiles * match.geometry.depth;
      auto sharedType = MemRefType::get({stageRows, columns}, element,
          MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(dsa::sharedSpace));
      auto shared = b.create<memref::AllocaOp>(loc, sharedType);
      shared.setAlignment(128);
      auto alternate = b.create<memref::AllocaOp>(loc, sharedType);
      alternate.setAlignment(128);
      auto when = [&](Value condition, const std::function<LogicalResult()> &body) -> LogicalResult {
        auto branch = b.create<scf::IfOp>(loc, condition, false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(branch.thenBlock());
        return body();
      };
      int64_t localDepth = match.geometry.depth;
      if (aType.hasStaticShape() && aType.getDimSize(1) > 0 &&
          aType.getDimSize(1) % stageRows == 0)
        localDepth = selectMatrixPanelDepth(function, config, element,
            match.geometry.rows, columns, stageRows, localDepth);
      Value localStep = index(loc, localDepth);
      return loop(loc, groupId, mul(loc, groupsM, gridN), groupCount, [&](Value group) -> LogicalResult {
        Value mi = add(loc, mul(loc, b.create<arith::DivSIOp>(loc, group, gridN), index(loc, 4)), localId);
        Value ni = b.create<arith::RemSIOp>(loc, group, gridN);
        Value m0 = mul(loc, mi, tm), n0 = mul(loc, ni, tn);
        Value rows = b.create<arith::MaxSIOp>(loc, index(loc, 0),
            b.create<arith::MinSIOp>(loc, sub(loc, M, m0), tm));
        Value cols = b.create<arith::MinSIOp>(loc, sub(loc, N, n0), tn);
        Value active = b.create<arith::AndIOp>(loc, computeCore,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, mi, gridM));
        Value accumulator = allocate(loc, b.getF32Type(), match.geometry.rows, columns);
        if (failed(when(active, [&]() {
          b.create<dsa::FillOp>(loc, accumulator, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)));
          return success();
        }))) return failure();
        auto supply = [&](Value stage, Value slot) -> LogicalResult {
          return when(memoryCore, [&]() {
            Value stageDepth = b.create<arith::MinSIOp>(loc, sub(loc, K, stage), index(loc, stageRows));
            Value br = stride(loc, rhsSource, 0), bc = stride(loc, rhsSource, 1);
            b.create<dsa::StageTileOp>(loc, rhsSource, slot, add(loc, mul(loc, stage, br), mul(loc, n0, bc)),
                br, bc, stageDepth, cols);
            return success();
          });
        };
        auto compute = [&](Value stage, Value slot) -> LogicalResult {
          return when(active, [&]() {
            Value stageDepth = b.create<arith::MinSIOp>(loc, sub(loc, K, stage), index(loc, stageRows));
            Value lhs = allocate(loc, element, match.geometry.rows, localDepth);
            Value rhs = allocate(loc, element, localDepth, columns);
            return loop(loc, index(loc, 0), stageDepth, localStep, [&](Value within) -> LogicalResult {
              Value depth = b.create<arith::MaxSIOp>(loc, index(loc, 0),
                  b.create<arith::MinSIOp>(loc, sub(loc, stageDepth, within), localStep));
              Value k0 = add(loc, stage, within);
              Value ar = stride(loc, lhsSource, 0), ac = stride(loc, lhsSource, 1);
              b.create<dsa::LoadTileOp>(loc, lhsSource, lhs, add(loc, mul(loc, m0, ar), mul(loc, k0, ac)), ar, ac, rows, depth);
              b.create<dsa::LoadTileOp>(loc, slot, rhs, mul(loc, within, tn), tn, index(loc, 1), depth, cols);
              b.create<dsa::MatMulOp>(loc, lhs, rhs, accumulator, rows, depth, cols);
              return success();
            });
          });
        };
        if (failed(supply(index(loc, 0), shared))) return failure();
        b.create<dsa::GroupSynchronizeOp>(loc);
        if (failed(loop(loc, index(loc, 0), K, index(loc, 2 * stageRows), [&](Value stage) -> LogicalResult {
          Value next = add(loc, stage, index(loc, stageRows));
          Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, K);
          if (failed(when(hasNext, [&]() { return supply(next, alternate); }))) return failure();
          if (failed(compute(stage, shared))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          return when(hasNext, [&]() -> LogicalResult {
            Value following = add(loc, next, index(loc, stageRows));
            Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, K);
            if (failed(when(hasFollowing, [&]() { return supply(following, shared); }))) return failure();
            if (failed(compute(next, alternate))) return failure();
            // Inactive compute participants also reach both group barriers.
            b.create<dsa::GroupSynchronizeOp>(loc);
            return success();
          });
        }))) return failure();
        return when(active, [&]() { return emitOutput(accumulator, m0, n0, rows, cols); });
      });
    }
  }
private:
  Value index(Location loc, int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
  Value add(Location loc, Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value sub(Location loc, Value a, Value c) { return b.create<arith::SubIOp>(loc, a, c); }
  Value mul(Location loc, Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value stride(Location loc, Value source, unsigned axis) {
    auto function = match.work->getParentOfType<func::FuncOp>();
    auto interface = intent::getPublicInterface(function);
    auto argument = cast<BlockArgument>(source);
    auto view = intent::getPublicView(interface, argument.getArgNumber());
    if (view.getConstraints().getHasStrides())
      if (auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis])) return index(loc, fixed.getInt());
    return b.create<StrideOp>(loc, b.getIndexType(), source, axis);
  }
  Value allocate(Location loc, Type element, int64_t rows, int64_t columns) {
    auto type = MemRefType::get({rows, columns}, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(nramSpace));
    auto allocation = b.create<memref::AllocaOp>(loc, type); allocation.setAlignment(128); return allocation;
  }
  LogicalResult loop(Location loc, Value lower, Value upper, Value step,
                     const std::function<LogicalResult(Value)> &body) {
    auto loop = b.create<scf::ForOp>(loc, lower, upper, step);
    OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(loop.getBody());
    return body(loop.getInductionVar());
  }
  MatrixSupplyMatch match;
  ConfigurationAttr config;
  func::FuncOp function;
  OpBuilder b;
};
} // namespace

LogicalResult realizeMatrixSupply(func::FuncOp function) {
  auto config = function->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  SmallVector<scf::ForOp> worksets;
  function.walk([&](scf::ForOp loop) {
    if (loop.getLowerBound().getDefiningOp<TaskIdOp>()) worksets.push_back(loop);
  });
  for (auto work : worksets) {
    auto match = matchMatrixSupply(work);
    if (!match) continue;
    auto a = cast<MemRefType>(match->lhs.getSource().getType());
    auto c = cast<MemRefType>(match->rhs.getSource().getType());
    Type element = a.getElementType();
    std::optional<StreamedMatrixPanel> resident;
    if (a.hasStaticShape() && c.hasStaticShape() &&
        config.getTasks() >= 4 && config.getTasks() % 4 == 0 &&
        c.getDimSize(1) / match->geometry.columns >= config.getTasks())
      resident = selectStreamedMatrixPanel(function, config, element,
          match->geometry.rows, match->geometry.columns, a.getDimSize(1), 0);
    bool shared = config.getTasks() >= 4 && config.getTasks() % 4 == 0 &&
        a.getDimSize(0) / match->geometry.rows >= 4;
    if (!resident && !shared) continue;
    if (failed(MatrixSupplyRewrite(*match, config).run(resident))) return failure();
    work.erase();
  }
  return success();
}
} // namespace intent::dsa

namespace intent::dsa {
#define GEN_PASS_DEF_DSAMATRIXSUPPLY
#include "Intent/Dialect/DSA/Transforms/Passes.h.inc"

namespace {
struct MatrixSupplyPass : impl::DSAMatrixSupplyBase<MatrixSupplyPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyRealizedProgram(module)))
      return signalPassFailure();
    auto function = *module.getOps<func::FuncOp>().begin();
    auto result = realizeMatrixSupply(function);
    if (failed(detail::finishTransform(module, getArgument(), result)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::dsa
