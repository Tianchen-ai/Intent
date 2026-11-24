#include "../legalize/Matrix.h"
#include "../legalize/Ordered.h"
#include "../legalize/Rows.h"
#include "compilerinfra/Assembly.h"
#include "compilerinfra/Contract.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <climits>

#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;

namespace {
struct WeftTaskContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          WeftTaskContractModel, intentdsl::wefthost::TaskContractOp> {
  Type getAccumulatorType(Operation *op) const {
    auto acc = op->getAttrOfType<StringAttr>("acc_type");
    if (!acc)
      return {};
    if (acc.getValue() == "f32")
      return Float32Type::get(op->getContext());
    if (acc.getValue() == "si32" || acc.getValue() == "i32")
      return IntegerType::get(op->getContext(), 32);
    return {};
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if ((op->getNumOperands() != 2 && op->getNumOperands() != 3) ||
        op->getNumResults() != 1)
      return failure();
    auto kind = op->getAttrOfType<StringAttr>("kind");
    auto inTys = op->getAttrOfType<ArrayAttr>("in_types");
    auto valTy = op->getAttrOfType<StringAttr>("val_type");
    if (!kind || !inTys || inTys.size() != op->getNumOperands() || !valTy)
      return failure();
    if (op->getNumOperands() == 3)
      return success(kind.getValue() == "fma");
    return success(kind.getValue() == "outer_contract" ||
                   kind.getValue() == "contract");
  }
};

struct HostParallelTilesBranchModel
    : public RegionBranchOpInterface::ExternalModel<
          HostParallelTilesBranchModel,
          intentdsl::wefthost::HostParallelTilesOp> {
  OperandRange getEntrySuccessorOperands(Operation *op,
                                         RegionBranchPoint) const {
    return OperandRange(op->operand_end(), op->operand_end());
  }
  void getEntrySuccessorRegions(
      Operation *op, ArrayRef<Attribute>,
      SmallVectorImpl<RegionSuccessor> &regions) const {
    getSuccessorRegions(op, RegionBranchPoint::parent(), regions);
  }
  void getSuccessorRegions(Operation *op, RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) const {
    if (point.isParent()) {
      regions.emplace_back(&op->getRegion(0));
      regions.emplace_back();
    } else {
      regions.emplace_back();
    }
  }
};

struct HostParallelTilesRangeModel
    : public InferIntRangeInterface::ExternalModel<
          HostParallelTilesRangeModel,
          intentdsl::wefthost::HostParallelTilesOp> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange>,
                                     SetIntLatticeFn setResultRanges) const {
    if (op->getNumRegions() == 0 || op->getRegion(0).empty())
      return;
    Block &bb = op->getRegion(0).front();
    if (bb.getNumArguments() < 10)
      return;
    auto ext = op->getAttrOfType<DenseI64ArrayAttr>("extents");
    auto tile = op->getAttrOfType<DenseI64ArrayAttr>("tile");
    if (!ext || ext.size() < 4 || !tile || tile.size() < 2)
      return;
    int64_t M = ext[0], N = ext[1], G = ext[3];
    int64_t mr = tile[0], nr = tile[1];
    int64_t mult = std::max<int64_t>(
        1, op->getAttrOfType<IntegerAttr>("multiplicity").getInt());
    int64_t H = std::max<int64_t>(
        1, op->getAttrOfType<IntegerAttr>("head_extent").getInt());
    int64_t chunks = std::max<int64_t>(
        1, op->getAttrOfType<IntegerAttr>("chunk_extent").getInt());
    bool routed = op->getAttrOfType<BoolAttr>("routed").getValue();
    auto setBound = [&](unsigned idx, int64_t hi) {
      int64_t clampedHi = std::max<int64_t>(0, hi);
      APInt loVal(64, 0, /*isSigned=*/true);
      APInt hiVal(64, uint64_t(clampedHi), /*isSigned=*/true);
      setResultRanges(bb.getArgument(idx),
                      ConstantIntRanges(loVal, hiVal, loVal, hiVal));
    };
    auto extentHi = [](int64_t e) -> int64_t { return e <= 1 ? 0 : e - 1; };
    setBound(0, extentHi(G));
    setBound(1, extentHi(N));
    setBound(2, extentHi(M));
    setBound(3, extentHi(M));
    setBound(4, extentHi(routed ? M * mult : M));
    setBound(5, extentHi(std::max<int64_t>(1, G / (H * chunks))));
    setBound(6, extentHi(chunks));
    setBound(7, extentHi(H));
    setBound(8, std::max<int64_t>(0, mr));
    setBound(9, std::max<int64_t>(0, nr));
  }
};
} // namespace

void intentdsl::wefthost::WeftHostDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  MLIRContext *ctx = getContext();
  TaskContractOp::attachInterface<WeftTaskContractModel>(*ctx);
  HostParallelTilesOp::attachInterface<HostParallelTilesBranchModel>(*ctx);
  HostParallelTilesOp::attachInterface<HostParallelTilesRangeModel>(*ctx);
}

namespace CI_UNIT_NAMESPACE {
LogicalResult lowerWeftMatrix(func::FuncOp, bool);
bool hasWeftMatrixProgram(ModuleOp);
LogicalResult verifyWeftMatrix(ModuleOp, const compilerinfra::TargetProfile &,
                               bool);
void publishWeftMatrixABI(OpPassManager &);

void dialect(DialectRegistry &r) {
  r.insert<intentdsl::wefthost::WeftHostDialect, arith::ArithDialect,
           math::MathDialect>();
}

namespace {
struct Lower : PassWrapper<Lower, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Lower)
  std::string phase;
  intentdsl::CallFacts facts;
  bool grouped;
  int64_t tailVecWidth;
  int64_t privateStackBytes;
  Lower(StringRef p, const intentdsl::CallFacts &f, bool g, int64_t tv,
        int64_t ps)
      : phase(p.str()), facts(f), grouped(g), tailVecWidth(tv),
        privateStackBytes(ps) {}
  void runOnOperation() final {
    ModuleOp m = getOperation();
    if (hasWeftMatrixProgram(m) || hasWeftRowProgram(m) ||
        hasWeftOwnerProgram(m))
      return;
    auto f = *m.getOps<func::FuncOp>().begin();
    if (phase == "legal") {
      if (intentdsl::cpu::ownerTask(f) &&
          failed(bindWeftMemberStores(f.front().front(), grouped, true)))
        signalPassFailure();
      return;
    }
    if (phase != "realized")
      return;

    getContext().getOrLoadDialect<intentdsl::wefthost::WeftHostDialect>();
    getContext().getOrLoadDialect<arith::ArithDialect>();
    getContext().getOrLoadDialect<math::MathDialect>();
    if (intentdsl::cpu::matrixTask(f)) {
      if (failed(lowerWeftMatrix(f, grouped)))
        signalPassFailure();
    } else if (intentdsl::cpu::ownerTask(f)) {
      if (failed(lowerWeftOrdered(f, facts, grouped)))
        signalPassFailure();
    } else {
      if (failed(
              lowerWeftRow(f, facts, grouped, tailVecWidth, privateStackBytes)))
        signalPassFailure();
    }
  }
};

struct PublishWeftRowAndOrderedABI
    : PassWrapper<PublishWeftRowAndOrderedABI, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PublishWeftRowAndOrderedABI)
  void runOnOperation() final {
    ModuleOp m = getOperation();
    bool isRow = hasWeftRowProgram(m);
    bool isOwner = hasWeftOwnerProgram(m);
    if (!isRow && !isOwner)
      return;
    auto fn = *m.getOps<func::FuncOp>().begin();
    auto &prog = fn.front().front();
    OpBuilder b(m.getContext());
    fn->setAttr("weft_host.entry", b.getStringAttr("intent_entry"));
    llvm::SmallVector<char> buf;
    llvm::raw_svector_ostream os(buf);
    llvm::Error err = isRow ? describeWeftRowPackage(fn, prog, os)
                            : describeWeftOrderedPackage(fn, prog, os);
    if (err) {
      m.emitError() << llvm::toString(std::move(err));
      signalPassFailure();
      return;
    }
    m->setAttr("weft_host.package",
               b.getStringAttr(StringRef(buf.data(), buf.size())));
  }
};
} // namespace

void lower(OpPassManager &pm, StringRef phase,
           const intentdsl::CallFacts &facts, bool grouped,
           int64_t tailVecWidth, int64_t privateStackBytes) {
  pm.addPass(std::make_unique<Lower>(phase, facts, grouped, tailVecWidth,
                                     privateStackBytes));
}

void publishWeftABI(OpPassManager &pm) {
  publishWeftMatrixABI(pm);
  pm.addPass(std::make_unique<PublishWeftRowAndOrderedABI>());
}

LogicalResult verifyHost(ModuleOp m, const compilerinfra::TargetProfile &p,
                         StringRef phase) {
  bool callable = (phase == "callable");
  if (hasWeftMatrixProgram(m)) {
    if (phase == "legal")
      return m.emitError("unexpected weft_host.matrix_program at weft@legal");
    return verifyWeftMatrix(m, p, callable);
  }
  if (hasWeftRowProgram(m)) {
    if (phase == "legal")
      return m.emitError("unexpected weft_host.row_program at weft@legal");
    return verifyWeftRow(m, p, callable);
  }
  if (hasWeftOwnerProgram(m)) {
    if (phase == "legal")
      return m.emitError("unexpected weft_host.owner_program at weft@legal");
    return verifyWeftOrdered(m, p, callable);
  }
  if (phase != "legal")
    return m.emitError("unlowered CPU task past weft@legal");

  if (failed(intentdsl::cpu::verifyTask(m, p, "weft-task-ssa")))
    return failure();
  auto fn = *m.getOps<func::FuncOp>().begin();
  if (intentdsl::cpu::ownerTask(fn)) {
    if (failed(bindWeftMemberStores(fn.front().front(),
                                    p.services.get<RecordFields>()->grouped,
                                    false)))
      return failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
