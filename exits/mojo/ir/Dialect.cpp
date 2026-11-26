#include "compilerinfra/Contract.h"
#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <climits>
#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace intentdsl::rowmojo;

namespace {
using EffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

template <typename ConcreteOp>
struct AllRegionsBranchModel
    : public RegionBranchOpInterface::ExternalModel<
          AllRegionsBranchModel<ConcreteOp>, ConcreteOp> {
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

struct MojoForRangeLoopModel
    : public LoopLikeOpInterface::ExternalModel<MojoForRangeLoopModel,
                                                MojoForRange> {
  SmallVector<Region *> getLoopRegions(Operation *op) const {
    return {&op->getRegion(0)};
  }
  std::optional<SmallVector<Value>> getLoopInductionVars(Operation *op) const {
    if (op->getRegion(0).empty() || op->getRegion(0).front().getNumArguments() < 1)
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
};

struct MojoForRangeRangeModel
    : public InferIntRangeInterface::ExternalModel<MojoForRangeRangeModel,
                                                   MojoForRange> {
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

struct MojoSimdFmaContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoSimdFmaContract, MojoSimdFma> {
  Type getAccumulatorType(Operation *op) const {
    if (op->getNumResults() != 1)
      return {};
    Type elem = getElementTypeOrSelf(op->getResult(0).getType());
    if (elem.isF32() || elem.isSignlessInteger(32))
      return elem;
    return {};
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();
    auto vecTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!vecTy || vecTy.getRank() != 1 || vecTy.getDimSize(0) < 1 ||
        vecTy.isScalable())
      return failure();
    if (op->getOperand(0).getType() != vecTy ||
        op->getOperand(1).getType() != vecTy ||
        op->getOperand(2).getType() != vecTy)
      return failure();
    auto isInt = op->getAttrOfType<BoolAttr>("is_int");
    if (!isInt)
      return failure();
    Type elem = vecTy.getElementType();
    if (isInt.getValue())
      return success(elem.isSignlessInteger(32));
    return success(elem.isF32());
  }
  LogicalResult verifyLayoutAndAddressSpace(Operation *op) const {
    auto lhsSt = op->getAttrOfType<DenseI64ArrayAttr>("lhs_strides");
    auto rhsSt = op->getAttrOfType<DenseI64ArrayAttr>("rhs_strides");
    if (!lhsSt || lhsSt.size() != 2 || !rhsSt || rhsSt.size() != 2)
      return failure();
    return success(lhsSt[0] >= 0 && lhsSt[1] >= 0 && rhsSt[0] >= 0 &&
                   rhsSt[1] >= 0);
  }
};

struct MojoVnniDot4Contract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoVnniDot4Contract, MojoVnniDot4> {
  Type getAccumulatorType(Operation *op) const {
    if (op->getNumResults() != 1)
      return {};
    Type elem = getElementTypeOrSelf(op->getResult(0).getType());
    if (elem.isSignlessInteger(32))
      return elem;
    return {};
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();
    auto accTy = dyn_cast<VectorType>(op->getOperand(0).getType());
    auto lhsTy = dyn_cast<VectorType>(op->getOperand(1).getType());
    auto rhsTy = dyn_cast<VectorType>(op->getOperand(2).getType());
    auto outTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!accTy || !lhsTy || !rhsTy || !outTy || accTy.isScalable() ||
        lhsTy.isScalable() || rhsTy.isScalable() || outTy.isScalable())
      return failure();
    if (accTy != outTy || accTy.getRank() != 1 || accTy.getDimSize(0) != 16 ||
        !accTy.getElementType().isSignlessInteger(32))
      return failure();
    if (lhsTy.getRank() != 1 || lhsTy.getDimSize(0) != 64 ||
        !lhsTy.getElementType().isUnsignedInteger(8))
      return failure();
    if (rhsTy.getRank() != 1 || rhsTy.getDimSize(0) != 64 ||
        !rhsTy.getElementType().isSignlessInteger(8))
      return failure();
    return success();
  }
  LogicalResult verifyLayoutAndAddressSpace(Operation *op) const {
    auto lhsSt = op->getAttrOfType<DenseI64ArrayAttr>("lhs_strides");
    auto pack = op->getAttrOfType<StringAttr>("rhs_packing");
    auto panelW = op->getAttrOfType<IntegerAttr>("rhs_panel_width");
    auto kExt = op->getAttrOfType<IntegerAttr>("k_extent");
    if (!lhsSt || lhsSt.size() != 2 || lhsSt[1] != 1)
      return failure();
    if (!pack || pack.getValue() != "signed-dot4-panels")
      return failure();
    if (!panelW || panelW.getInt() < 16 || (panelW.getInt() % 16) != 0)
      return failure();
    if (!kExt || kExt.getInt() < 4 || (kExt.getInt() % 4) != 0)
      return failure();
    return success();
  }
};

struct MojoQ4KContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoQ4KContract, MojoQ4KSuperblock> {
  Type getAccumulatorType(Operation *op) const {
    return IntegerType::get(op->getContext(), 32);
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if (op->getNumOperands() != 4 || op->getNumResults() != 1)
      return failure();
    for (Value operand : op->getOperands())
      if (!operand.getType().isIndex())
        return failure();
    auto outTy = dyn_cast<VectorType>(op->getResult(0).getType());
    return success(outTy && outTy.getRank() == 1 && outTy.getDimSize(0) == 1 &&
                   outTy.getElementType().isF32());
  }
  LogicalResult verifyLayoutAndAddressSpace(Operation *op) const {
    auto bs = op->getAttrOfType<IntegerAttr>("byte_stride");
    if (!bs || bs.getInt() < 1)
      return failure();
    if (op->getAttr("kir.storage") ==
        TypeAttr::get(intentdsl::kir::Q4KType::get(op->getContext())))
      return success();
    auto fmt = op->getAttrOfType<StringAttr>("format");
    const auto *spec =
        fmt ? intentdsl::kir::lookupQuantFormat(fmt.getValue()) : nullptr;
    return success(spec &&
                   spec->roleLayout ==
                       intentdsl::kir::QuantRoleLayout::RecordGemv);
  }
};

template <typename ConcreteOp>
struct MojoF32AccContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoF32AccContract<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
};

template <typename ConcreteOp>
struct MojoI32AccContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoI32AccContract<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return IntegerType::get(op->getContext(), 32);
  }
};

struct MojoPrefixContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoPrefixContract, MojoValuePrefix> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
};
struct MojoBrgemmContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoBrgemmContract, MojoBrgemmCall> {
  Type getAccumulatorType(Operation *op) const {
    if (auto impl = op->getAttrOfType<StringAttr>("ci.implementation")) {
      StringRef s = impl.getValue();
      if (s.ends_with("-s8") || s.ends_with("-u8s8"))
        return IntegerType::get(op->getContext(), 32);
    }
    return Float32Type::get(op->getContext());
  }
};

struct MojoScanContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoScanContract, MojoSimdScanSum> {
  Type getAccumulatorType(Operation *op) const {
    auto dt = op->getAttrOfType<StringAttr>("dtype");
    if (dt && dt.getValue() == "int32")
      return IntegerType::get(op->getContext(), 32);
    if (dt && dt.getValue() == "int64")
      return IntegerType::get(op->getContext(), 64);
    Type elem = getElementTypeOrSelf(op->getResult(0).getType());
    if (elem.isInteger(32) || elem.isInteger(64) || elem.isF32())
      return elem;
    return Float32Type::get(op->getContext());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if (op->getNumOperands() != 2 || op->getNumResults() != 2)
      return failure();
    auto wAttr = op->getAttrOfType<IntegerAttr>("width");
    if (!wAttr || wAttr.getInt() < 1)
      return failure();
    Type elem = op->getOperand(1).getType();
    if (!(elem.isF32() || elem.isSignlessInteger(32) ||
          elem.isSignlessInteger(64)) ||
        op->getResult(1).getType() != elem)
      return failure();
    if (wAttr.getInt() == 1 && !isa<ShapedType>(op->getOperand(0).getType()))
      return success(op->getOperand(0).getType() == elem &&
                     op->getResult(0).getType() == elem);
    auto inVec = dyn_cast<VectorType>(op->getOperand(0).getType());
    auto outVec = dyn_cast<VectorType>(op->getResult(0).getType());
    return success(inVec && outVec && inVec == outVec && inVec.getRank() == 1 &&
                   inVec.getDimSize(0) == wAttr.getInt() &&
                   inVec.getElementType() == elem);
  }
};

struct MojoTransposeContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoTransposeContract, MojoTranspose8x8> {};

struct MojoOwnerCoordRangeModel
    : public InferIntRangeInterface::ExternalModel<MojoOwnerCoordRangeModel,
                                                   MojoOwnerCoord> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange>,
                                     SetIntLatticeFn setResultRanges) const {
    auto ext = op->getAttrOfType<IntegerAttr>("extent");
    if (!ext || ext.getInt() <= 0) {
      setResultRanges(op->getResult(0), ConstantIntRanges::maxRange(64));
      return;
    }
    APInt lo(64, 0, /*isSigned=*/true);
    APInt hi(64, uint64_t(ext.getInt() - 1), /*isSigned=*/true);
    setResultRanges(op->getResult(0), ConstantIntRanges(lo, hi, lo, hi));
  }
};

struct MojoPanelFnBoundsModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          MojoPanelFnBoundsModel, MojoPanelFn> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front())
      return failure();
    auto dtypes = op->getAttrOfType<ArrayAttr>("param_dtypes");
    auto bounds = op->getAttrOfType<DenseI64ArrayAttr>("tile_bounds");
    if (!dtypes || !bounds || bounds.size() != 2)
      return failure();
    unsigned idx = arg.getArgNumber();
    if (idx != dtypes.size() && idx != dtypes.size() + 1)
      return failure();
    int64_t hi = bounds[idx - dtypes.size()];
    if (hi < 0)
      return failure();
    out = compilerinfra::RegionIndexBounds{};
    out.staticLower = 0;
    out.staticUpperExclusive = hi + 1;
    out.staticStep = 1;
    return success();
  }
};

struct MojoOwnerFnBoundsModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          MojoOwnerFnBoundsModel, MojoOwnerFn> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front())
      return failure();
    auto ownersAttr = op->getAttrOfType<IntegerAttr>("owners");
    if (!ownersAttr || ownersAttr.getInt() < 0)
      return failure();
    Block &bb = op->getRegion(0).front();
    unsigned nArgs = bb.getNumArguments();
    if (nArgs < 2)
      return failure();
    unsigned idx = arg.getArgNumber();
    int64_t owners = ownersAttr.getInt();
    if (idx == nArgs - 2) {
      out = compilerinfra::RegionIndexBounds{};
      out.staticLower = 0;
      out.staticUpperExclusive = std::max<int64_t>(1, owners);
      out.staticStep = 1;
      return success();
    }
    if (idx == nArgs - 1) {
      out = compilerinfra::RegionIndexBounds{};
      out.staticLower = 0;
      out.staticUpperExclusive = owners + 1;
      out.staticStep = 1;
      return success();
    }
    return failure();
  }
};

struct MojoWhileLtBoundsModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          MojoWhileLtBoundsModel, MojoWhileLt> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front() ||
        arg.getArgNumber() != 0 || op->getNumOperands() < 3)
      return failure();
    out = compilerinfra::RegionIndexBounds{};
    out.dynamicLower = op->getOperand(0);
    out.dynamicUpperExclusive = op->getOperand(1);
    out.dynamicStep = op->getOperand(2);
    return success();
  }
};

struct MojoWhileLtRangeModel
    : public InferIntRangeInterface::ExternalModel<MojoWhileLtRangeModel,
                                                   MojoWhileLt> {
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
      APInt clampHi = llvm::APIntOps::smax(lbLo, lbHi);
      setResultRanges(iv, ConstantIntRanges(lbLo, clampHi, lbLo, clampHi));
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

bool flattenMojoLinearTerms(Operation *op, int64_t scale,
                            SmallVectorImpl<Value> &coords,
                            SmallVectorImpl<int64_t> &strides,
                            unsigned depth = 0) {
  if (depth > 8)
    return false;
  auto stAttr = op->getAttrOfType<DenseI64ArrayAttr>("strides");
  auto idxOps = op->getOperands().drop_front();
  if (!stAttr || stAttr.size() != idxOps.size())
    return false;
  for (auto [idxVal, st] : llvm::zip(idxOps, stAttr.asArrayRef())) {
    __int128 prod = __int128(scale) * __int128(st);
    if (prod < INT64_MIN || prod > INT64_MAX)
      return false;
    int64_t scaledSt = static_cast<int64_t>(prod);
    if (Operation *def = idxVal.getDefiningOp();
        def && def->getName().getStringRef() == "mojo_row.address") {
      if (!flattenMojoLinearTerms(def, scaledSt, coords, strides, depth + 1))
        return false;
    } else {
      coords.push_back(idxVal);
      strides.push_back(scaledSt);
    }
  }
  return true;
}

struct MojoAddressLinearModel
    : public compilerinfra::AddressOpInterface::ExternalModel<
          MojoAddressLinearModel, MojoAddress> {
  LogicalResult
  getLinearAddress(Operation *op,
                   compilerinfra::LinearAddressDescriptor &out) const {
    if (op->getNumOperands() < 1)
      return failure();
    out = compilerinfra::LinearAddressDescriptor{};
    out.baseContainer = op->getOperand(0);
    out.constantOffset = 0;
    if (!flattenMojoLinearTerms(op, 1, out.coordinates, out.staticStrides))
      return failure();
    return success();
  }
};

struct MojoAddressRangeModel
    : public InferIntRangeInterface::ExternalModel<MojoAddressRangeModel,
                                                   MojoAddress> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange> argRanges,
                                     SetIntLatticeFn setResultRanges) const {
    Value res = op->getResult(0);
    unsigned bw = ConstantIntRanges::getStorageBitwidth(res.getType());
    if (bw == 0 || bw > 64)
      return;
    auto strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
    if (!strides || argRanges.size() != strides.size() + 1) {
      setResultRanges(res, ConstantIntRanges::maxRange(bw));
      return;
    }
    __int128 sumLo = 0, sumHi = 0;
    for (size_t i = 0; i < strides.size(); ++i) {
      if (argRanges[i + 1].isUninitialized()) {
        setResultRanges(res, ConstantIntRanges::maxRange(bw));
        return;
      }
      const auto &r = argRanges[i + 1].getValue();
      if (r.smin().isMinSignedValue() && r.smax().isMaxSignedValue()) {
        setResultRanges(res, ConstantIntRanges::maxRange(bw));
        return;
      }
      __int128 st = strides[i];
      __int128 p1 = __int128(r.smin().getSExtValue()) * st;
      __int128 p2 = __int128(r.smax().getSExtValue()) * st;
      if (p1 > p2)
        std::swap(p1, p2);
      sumLo += p1;
      sumHi += p2;
      if (sumLo < INT64_MIN || sumHi > INT64_MAX) {
        setResultRanges(res, ConstantIntRanges::maxRange(bw));
        return;
      }
    }
    APInt sLo(bw, static_cast<int64_t>(sumLo), /*isSigned=*/true);
    APInt sHi(bw, static_cast<int64_t>(sumHi), /*isSigned=*/true);
    if (sumLo >= 0)
      setResultRanges(res, ConstantIntRanges(sLo, sHi, sLo, sHi));
    else
      setResultRanges(res, ConstantIntRanges::fromSigned(sLo, sHi));
  }
};

struct MojoAddressTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          MojoAddressTiledAccessModel, MojoAddress> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < 2)
      return failure();
    Value base = op->getOperand(0);
    auto coords = op->getOperands().drop_front();
    auto extents = op->getAttrOfType<DenseI64ArrayAttr>("extents");
    auto tileAxes = op->getAttrOfType<DenseI64ArrayAttr>("tile_axes");
    auto tileSizes = op->getAttrOfType<DenseI64ArrayAttr>("tile_sizes");
    if (!extents || extents.size() != coords.size())
      return failure();
    if (tileAxes && tileSizes) {
      if (tileAxes.empty() || tileAxes.size() != tileSizes.size())
        return failure();
      for (auto [axIdx, tileSz] :
           llvm::zip(tileAxes.asArrayRef(), tileSizes.asArrayRef())) {
        if (axIdx < 0 || static_cast<size_t>(axIdx) >= coords.size() ||
            tileSz <= 0)
          return failure();
        compilerinfra::AccessAxis ax;
        ax.container = base;
        ax.containerDim = static_cast<unsigned>(axIdx);
        if (extents[axIdx] >= 0)
          ax.staticExtent = extents[axIdx];
        ax.coord = coords[axIdx];
        ax.tileSize = tileSz;
        ax.step = 1;
        axes.push_back(ax);
      }
      return success();
    }
    for (size_t d = 0; d < coords.size(); ++d) {
      compilerinfra::AccessAxis ax;
      ax.container = base;
      ax.containerDim = static_cast<unsigned>(d);
      if (extents[d] >= 0)
        ax.staticExtent = extents[d];
      ax.coord = coords[d];
      ax.tileSize = 1;
      ax.step = 1;
      axes.push_back(ax);
    }
    return success();
  }
};

struct MojoLoadSliceTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          MojoLoadSliceTiledAccessModel, MojoLoadSlice> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < 2)
      return failure();
    Operation *addrOp = op->getOperand(1).getDefiningOp();
    auto widthAttr = op->getAttrOfType<IntegerAttr>("width");
    if (!addrOp || addrOp->getName().getStringRef() != "mojo_row.address" ||
        !widthAttr || widthAttr.getInt() <= 0)
      return failure();
    auto coords = addrOp->getOperands().drop_front();
    auto extents = addrOp->getAttrOfType<DenseI64ArrayAttr>("extents");
    if (coords.empty() || !extents || extents.size() != coords.size())
      return failure();
    int64_t sliceAxis = static_cast<int64_t>(coords.size()) - 1;
    if (auto sa = op->getAttrOfType<IntegerAttr>("slice_axis"))
      sliceAxis = sa.getInt();
    if (sliceAxis < 0 || static_cast<size_t>(sliceAxis) >= coords.size())
      return failure();
    for (size_t d = 0; d < coords.size(); ++d) {
      compilerinfra::AccessAxis ax;
      ax.container = addrOp->getOperand(0);
      ax.containerDim = static_cast<unsigned>(d);
      if (extents[d] >= 0)
        ax.staticExtent = extents[d];
      ax.coord = coords[d];
      ax.tileSize =
          (static_cast<int64_t>(d) == sliceAxis) ? widthAttr.getInt() : 1;
      ax.step = 1;
      axes.push_back(ax);
    }
    return success();
  }
};

struct MojoStoreSliceTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          MojoStoreSliceTiledAccessModel, MojoStoreSlice> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < 3)
      return failure();
    Operation *addrOp = op->getOperand(1).getDefiningOp();
    auto vt = dyn_cast<ShapedType>(op->getOperand(2).getType());
    if (!addrOp || addrOp->getName().getStringRef() != "mojo_row.address" ||
        !vt || vt.getRank() != 1 || vt.getDimSize(0) <= 0)
      return failure();
    auto coords = addrOp->getOperands().drop_front();
    auto extents = addrOp->getAttrOfType<DenseI64ArrayAttr>("extents");
    if (coords.empty() || !extents || extents.size() != coords.size())
      return failure();
    int64_t sliceAxis = static_cast<int64_t>(coords.size()) - 1;
    if (auto sa = op->getAttrOfType<IntegerAttr>("slice_axis"))
      sliceAxis = sa.getInt();
    if (sliceAxis < 0 || static_cast<size_t>(sliceAxis) >= coords.size())
      return failure();
    for (size_t d = 0; d < coords.size(); ++d) {
      compilerinfra::AccessAxis ax;
      ax.container = addrOp->getOperand(0);
      ax.containerDim = static_cast<unsigned>(d);
      if (extents[d] >= 0)
        ax.staticExtent = extents[d];
      ax.coord = coords[d];
      ax.tileSize =
          (static_cast<int64_t>(d) == sliceAxis) ? vt.getDimSize(0) : 1;
      ax.step = 1;
      axes.push_back(ax);
    }
    return success();
  }
};

template <typename ConcreteOp>
struct MojoVecAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          MojoVecAccessModel<ConcreteOp>, ConcreteOp> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < 2)
      return failure();
    auto addrOp = op->getOperand(1).getDefiningOp<MojoAddress>();
    if (!addrOp)
      return failure();
    if (auto tiled = dyn_cast<compilerinfra::TiledAccessOpInterface>(
            addrOp.getOperation());
        addrOp->hasAttr("tile_axes") && addrOp->hasAttr("tile_sizes"))
      return tiled.getAccessedAxes(axes);
    auto coords = addrOp->getOperands().drop_front();
    auto extents = addrOp->getAttrOfType<DenseI64ArrayAttr>("extents");
    auto widthAttr = op->getAttrOfType<IntegerAttr>("width");
    auto stepAttr = op->getAttrOfType<IntegerAttr>("lane_stride");
    if (coords.empty() || !extents || extents.size() != coords.size() ||
        !widthAttr || widthAttr.getInt() <= 0 || !stepAttr ||
        stepAttr.getInt() <= 0)
      return failure();
    for (size_t d = 0; d < coords.size(); ++d) {
      compilerinfra::AccessAxis ax;
      ax.container = addrOp->getOperand(0);
      ax.containerDim = static_cast<unsigned>(d);
      if (extents[d] >= 0)
        ax.staticExtent = extents[d];
      ax.coord = coords[d];
      bool isLaneAxis = (d + 1 == coords.size());
      ax.tileSize = isLaneAxis ? widthAttr.getInt() : 1;
      ax.step = isLaneAxis ? stepAttr.getInt() : 1;
      axes.push_back(ax);
    }
    return success();
  }
};

struct MojoDynAddressLinearModel
    : public compilerinfra::AddressOpInterface::ExternalModel<
          MojoDynAddressLinearModel, MojoDynAddress> {
  LogicalResult
  getLinearAddress(Operation *op,
                   compilerinfra::LinearAddressDescriptor &out) const {
    auto termsAttr = op->getAttrOfType<IntegerAttr>("terms");
    if (!termsAttr || termsAttr.getInt() < 0)
      return failure();
    size_t n = static_cast<size_t>(termsAttr.getInt());
    if (op->getNumOperands() != 1 + 2 * n)
      return failure();
    out = compilerinfra::LinearAddressDescriptor{};
    out.baseContainer = op->getOperand(0);
    out.constantOffset = 0;
    for (size_t i = 0; i < n; ++i) {
      out.coordinates.push_back(op->getOperand(1 + i));
      out.dynamicStrides.push_back(op->getOperand(1 + n + i));
    }
    return success();
  }
};
} // namespace

void MojoQ4KSuperblock::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

void MojoPackValueTable::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void MojoTransposeWeights::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}
void MojoValuePrefix::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void MojoBrgemmCall::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void MojoTranspose8x8::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void intentdsl::rowmojo::RowMojoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  MLIRContext *ctx = getContext();
  MojoSimdFma::attachInterface<MojoSimdFmaContract>(*ctx);
  MojoVnniDot4::attachInterface<MojoVnniDot4Contract>(*ctx);
  MojoQ4KSuperblock::attachInterface<MojoQ4KContract>(*ctx);
  MojoBrgemmCall::attachInterface<MojoBrgemmContract>(*ctx);
  MojoValuePrefix::attachInterface<MojoPrefixContract>(*ctx);
  MojoUniform::attachInterface<MojoI32AccContract<MojoUniform>>(*ctx);
  MojoReduceSum16x16::attachInterface<MojoF32AccContract<MojoReduceSum16x16>>(
      *ctx);
  MojoWelfordVecStep::attachInterface<MojoF32AccContract<MojoWelfordVecStep>>(
      *ctx);
  MojoWelfordVecReduce::attachInterface<
      MojoF32AccContract<MojoWelfordVecReduce>>(*ctx);
  MojoWelfordScalarStep::attachInterface<
      MojoF32AccContract<MojoWelfordScalarStep>>(*ctx);
  MojoTranspose8x8::attachInterface<MojoTransposeContract>(*ctx);
  MojoSimdScanSum::attachInterface<MojoScanContract>(*ctx);
  MojoOwnerCoord::attachInterface<MojoOwnerCoordRangeModel>(*ctx);

  MojoMatrixProgram::attachInterface<AllRegionsBranchModel<MojoMatrixProgram>>(
      *ctx);
  MojoRowProgram::attachInterface<AllRegionsBranchModel<MojoRowProgram>>(*ctx);
  MojoOrderedProgram::attachInterface<
      AllRegionsBranchModel<MojoOrderedProgram>>(*ctx);
  MojoPanelFn::attachInterface<AllRegionsBranchModel<MojoPanelFn>>(*ctx);
  MojoPanelFn::attachInterface<MojoPanelFnBoundsModel>(*ctx);
  MojoRowFn::attachInterface<AllRegionsBranchModel<MojoRowFn>>(*ctx);
  MojoOwnerFn::attachInterface<AllRegionsBranchModel<MojoOwnerFn>>(*ctx);
  MojoOwnerFn::attachInterface<MojoOwnerFnBoundsModel>(*ctx);
  MojoIfFullTile::attachInterface<AllRegionsBranchModel<MojoIfFullTile>>(*ctx);
  MojoWhileLt::attachInterface<AllRegionsBranchModel<MojoWhileLt>>(*ctx);
  MojoWhileLt::attachInterface<MojoWhileLtBoundsModel>(*ctx);
  MojoWhileLt::attachInterface<MojoWhileLtRangeModel>(*ctx);
  MojoIf::attachInterface<AllRegionsBranchModel<MojoIf>>(*ctx);
  MojoForRange::attachInterface<AllRegionsBranchModel<MojoForRange>>(*ctx);
  MojoForRange::attachInterface<MojoForRangeLoopModel>(*ctx);
  MojoForRange::attachInterface<MojoForRangeRangeModel>(*ctx);
  MojoAddress::attachInterface<MojoAddressLinearModel>(*ctx);
  MojoAddress::attachInterface<MojoAddressRangeModel>(*ctx);
  MojoAddress::attachInterface<MojoAddressTiledAccessModel>(*ctx);
  MojoDynAddress::attachInterface<MojoDynAddressLinearModel>(*ctx);
  MojoLoadSlice::attachInterface<MojoLoadSliceTiledAccessModel>(*ctx);
  MojoStoreSlice::attachInterface<MojoStoreSliceTiledAccessModel>(*ctx);
  MojoLoadVec::attachInterface<MojoVecAccessModel<MojoLoadVec>>(*ctx);
  MojoStoreVec::attachInterface<MojoVecAccessModel<MojoStoreVec>>(*ctx);
}
