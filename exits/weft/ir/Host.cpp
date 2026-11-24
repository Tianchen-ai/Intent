#include "../legalize/Matrix.h"
#include "../legalize/Ordered.h"
#include "../legalize/Rows.h"
#include "compilerinfra/Assembly.h"
#include "compilerinfra/Contract.h"
#include "compilerinfra/Proofs.h"
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
#include "mlir/Interfaces/LoopLikeInterface.h"
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

template <typename ConcreteOp>
struct WeftRegionBranchModel
    : public RegionBranchOpInterface::ExternalModel<
          WeftRegionBranchModel<ConcreteOp>, ConcreteOp> {
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
      for (Region &r : op->getRegions())
        if (!r.empty())
          regions.emplace_back(&r);
      regions.emplace_back();
    } else {
      regions.emplace_back();
    }
  }
};

struct HostDefI64RangeModel
    : public InferIntRangeInterface::ExternalModel<
          HostDefI64RangeModel, intentdsl::wefthost::HostDefI64Op> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange>,
                                     SetIntLatticeFn setResultRanges) const {
    auto mb = op->getAttrOfType<IntegerAttr>("max_bound");
    if (!mb || mb.getInt() < 0) {
      setResultRanges(op->getResult(0), ConstantIntRanges::maxRange(64));
      return;
    }
    APInt loVal(64, 0, /*isSigned=*/true);
    APInt hiVal(64, uint64_t(mb.getInt()), /*isSigned=*/true);
    setResultRanges(op->getResult(0),
                    ConstantIntRanges(loVal, hiVal, loVal, hiVal));
  }
};

struct WeftTaskForLoopModel
    : public LoopLikeOpInterface::ExternalModel<
          WeftTaskForLoopModel, intentdsl::wefthost::TaskForOp> {
  SmallVector<Region *> getLoopRegions(Operation *op) const {
    return {&op->getRegion(0)};
  }
  std::optional<SmallVector<Value>> getLoopInductionVars(Operation *op) const {
    if (op->getRegion(0).empty() ||
        op->getRegion(0).front().getNumArguments() < 1)
      return std::nullopt;
    return SmallVector<Value>{op->getRegion(0).front().getArgument(0)};
  }
  std::optional<SmallVector<OpFoldResult>>
  getLoopLowerBounds(Operation *op) const {
    return SmallVector<OpFoldResult>{OpFoldResult(op->getOperand(0))};
  }
  std::optional<SmallVector<OpFoldResult>>
  getLoopUpperBounds(Operation *op) const {
    return SmallVector<OpFoldResult>{OpFoldResult(op->getOperand(1))};
  }
  std::optional<SmallVector<OpFoldResult>> getLoopSteps(Operation *op) const {
    return SmallVector<OpFoldResult>{OpFoldResult(op->getOperand(2))};
  }
  OperandRange getInits(Operation *op) const {
    return cast<intentdsl::wefthost::TaskForOp>(op).getInits();
  }
  Block::BlockArgListType getRegionIterArgs(Operation *op) const {
    if (op->getRegion(0).empty() ||
        op->getRegion(0).front().getNumArguments() < 1)
      return {};
    return op->getRegion(0).front().getArguments().drop_front(1);
  }
  std::optional<MutableArrayRef<OpOperand>>
  getYieldedValuesMutable(Operation *op) const {
    if (op->getRegion(0).empty() || op->getRegion(0).front().empty())
      return std::nullopt;
    Operation *term = op->getRegion(0).front().getTerminator();
    if (!term || term->getName().getStringRef() != "weft_host.task_yield")
      return std::nullopt;
    return term->getOpOperands();
  }
};

struct WeftTaskForRangeModel
    : public InferIntRangeInterface::ExternalModel<
          WeftTaskForRangeModel, intentdsl::wefthost::TaskForOp> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange> argRanges,
                                     SetIntLatticeFn setResultRanges) const {
    if (op->getNumRegions() == 0 || op->getRegion(0).empty() ||
        op->getRegion(0).front().getNumArguments() < 1)
      return;
    Value iv = op->getRegion(0).front().getArgument(0);
    unsigned bw = ConstantIntRanges::getStorageBitwidth(iv.getType());
    if (bw == 0)
      return;
    if (argRanges.size() < 3 || argRanges[0].isUninitialized() ||
        argRanges[1].isUninitialized() || argRanges[2].isUninitialized()) {
      setResultRanges(iv, ConstantIntRanges::maxRange(bw));
      return;
    }
    const auto &lbR = argRanges[0].getValue();
    const auto &ubR = argRanges[1].getValue();
    const auto &stR = argRanges[2].getValue();
    APInt lbLo = lbR.smin().sextOrTrunc(bw);
    APInt lbHi = lbR.smax().sextOrTrunc(bw);
    APInt ubHi = ubR.smax().sextOrTrunc(bw);
    APInt stLo = stR.smin().sextOrTrunc(bw);
    APInt stHi = stR.smax().sextOrTrunc(bw);
    if (lbLo.isNegative() || !stLo.isStrictlyPositive()) {
      setResultRanges(iv, ConstantIntRanges::maxRange(bw));
      return;
    }
    if (ubHi.sle(lbLo)) {
      setResultRanges(iv, ConstantIntRanges(lbLo, lbLo, lbLo, lbLo));
      return;
    }
    APInt one(bw, 1, /*isSigned=*/true);
    APInt hi = ubHi - one;
    if (lbLo == lbHi && stLo == stHi) {
      APInt trips = (ubHi - one - lbLo).sdiv(stLo);
      hi = lbLo + trips * stLo;
    }
    setResultRanges(iv, ConstantIntRanges(lbLo, hi, lbLo, hi));
  }
};

struct WeftTaskSubviewTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          WeftTaskSubviewTiledAccessModel, intentdsl::wefthost::TaskSubviewOp> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < 2)
      return failure();
    auto tileAttr = op->getAttrOfType<IntegerAttr>("static_extent");
    auto extAttr = op->getAttrOfType<IntegerAttr>("container_extent");
    if (!tileAttr || tileAttr.getInt() <= 0 || !extAttr || extAttr.getInt() <= 0)
      return failure();
    compilerinfra::AccessAxis ax;
    ax.container = op->getOperand(0);
    ax.containerDim = 0;
    ax.staticExtent = extAttr.getInt();
    ax.coord = op->getOperand(1);
    ax.tileSize = tileAttr.getInt();
    ax.step = 1;
    axes.push_back(ax);
    return success();
  }
};

struct WeftAddressLinearModel
    : public compilerinfra::AddressOpInterface::ExternalModel<
          WeftAddressLinearModel, intentdsl::wefthost::AddressOp> {
  LogicalResult
  getLinearAddress(Operation *op,
                   compilerinfra::LinearAddressDescriptor &out) const {
    auto strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
    auto indices = op->getOperands();
    if (!strides || strides.size() != indices.size())
      return failure();
    out = compilerinfra::LinearAddressDescriptor{};
    out.coordinates.assign(indices.begin(), indices.end());
    out.staticStrides.assign(strides.asArrayRef().begin(),
                             strides.asArrayRef().end());
    out.constantOffset = 0;
    return success();
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
  MatrixProgramOp::attachInterface<WeftRegionBranchModel<MatrixProgramOp>>(
      *ctx);
  RowProgramOp::attachInterface<WeftRegionBranchModel<RowProgramOp>>(*ctx);
  OwnerProgramOp::attachInterface<WeftRegionBranchModel<OwnerProgramOp>>(*ctx);
  TaskKernelOp::attachInterface<WeftRegionBranchModel<TaskKernelOp>>(*ctx);
  TaskScanOp::attachInterface<WeftRegionBranchModel<TaskScanOp>>(*ctx);
  TaskForOp::attachInterface<WeftRegionBranchModel<TaskForOp>>(*ctx);
  TaskForOp::attachInterface<WeftTaskForLoopModel>(*ctx);
  TaskForOp::attachInterface<WeftTaskForRangeModel>(*ctx);
  TaskIfOp::attachInterface<WeftRegionBranchModel<TaskIfOp>>(*ctx);
  TaskSubquantLevelOp::attachInterface<
      WeftRegionBranchModel<TaskSubquantLevelOp>>(*ctx);
  TaskSubviewOp::attachInterface<WeftTaskSubviewTiledAccessModel>(*ctx);
  AddressOp::attachInterface<WeftAddressLinearModel>(*ctx);
  HostFnOp::attachInterface<WeftRegionBranchModel<HostFnOp>>(*ctx);
  HostRowFnOp::attachInterface<WeftRegionBranchModel<HostRowFnOp>>(*ctx);
  HostOwnerFnOp::attachInterface<WeftRegionBranchModel<HostOwnerFnOp>>(*ctx);
  HostIfOp::attachInterface<WeftRegionBranchModel<HostIfOp>>(*ctx);
  HostOmpForOp::attachInterface<WeftRegionBranchModel<HostOmpForOp>>(*ctx);
  HostForOp::attachInterface<WeftRegionBranchModel<HostForOp>>(*ctx);
  HostWhileOp::attachInterface<WeftRegionBranchModel<HostWhileOp>>(*ctx);
  HostDefI64Op::attachInterface<HostDefI64RangeModel>(*ctx);
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
