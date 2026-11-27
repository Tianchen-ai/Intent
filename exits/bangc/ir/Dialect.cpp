#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "Dialect.cpp.inc"
#include "compilerinfra/Proofs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <climits>
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace intentdsl::bangc_src;

namespace {
using EffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

void addRead(EffectList &effects, Operation *op, unsigned idx) {
  if (idx < op->getNumOperands()) {
    OpOperand &opnd = op->getOpOperand(idx);
    if (isa<MemRefType>(opnd.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &opnd);
  }
}

void addWrite(EffectList &effects, Operation *op, unsigned idx) {
  if (idx < op->getNumOperands()) {
    OpOperand &opnd = op->getOpOperand(idx);
    if (isa<MemRefType>(opnd.get().getType()))
      effects.emplace_back(MemoryEffects::Write::get(), &opnd);
  }
}

void addReadWrite(EffectList &effects, Operation *op, unsigned idx) {
  addRead(effects, op, idx);
  addWrite(effects, op, idx);
}

bool hasMemSpace(Value v, int64_t expected) {
  auto m = dyn_cast<MemRefType>(v.getType());
  if (!m)
    return false;
  auto attr = dyn_cast_or_null<IntegerAttr>(m.getMemorySpace());
  int64_t space = attr ? attr.getInt() : 0;
  return space == expected;
}

bool isMomentTupleType(Type t) {
  auto tup = dyn_cast<TupleType>(t);
  return tup && tup.size() == 3 && tup.getType(0).isInteger(32) &&
         tup.getType(1).isF32() && tup.getType(2).isF32();
}

ConstantIntRanges nonNegRange64(int64_t lo, int64_t hi) {
  APInt loVal(64, uint64_t(std::max<int64_t>(0, lo)), /*isSigned=*/true);
  APInt hiVal(64, uint64_t(std::max<int64_t>(lo, hi)), /*isSigned=*/true);
  return ConstantIntRanges(loVal, hiVal, loVal, hiVal);
}

template <typename ConcreteOp>
struct F32AccContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          F32AccContract<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
};

template <typename ConcreteOp>
struct I32AccContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          I32AccContract<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return IntegerType::get(op->getContext(), 32);
  }
};

template <typename ConcreteOp>
struct I16AccContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          I16AccContract<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return IntegerType::get(op->getContext(), 16);
  }
};

struct WramConvContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          WramConvContract, WramConvOp> {
  Type getAccumulatorType(Operation *op) const {
    auto mode = op->getAttrOfType<StringAttr>("mode");
    if (mode && mode.getValue() == "i8")
      return IntegerType::get(op->getContext(), 32);
    return Float32Type::get(op->getContext());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    auto mode = op->getAttrOfType<StringAttr>("mode");
    auto bn = op->getAttrOfType<IntegerAttr>("bn");
    auto bk = op->getAttrOfType<IntegerAttr>("bk");
    if (!mode || !bn || bn.getInt() != 64 || !bk || op->getNumOperands() != 4)
      return failure();
    if (mode == "i8") {
      if (bk.getInt() > 1024)
        return failure();
      auto acc = dyn_cast<MemRefType>(op->getOperand(0).getType());
      auto part = dyn_cast<MemRefType>(op->getOperand(1).getType());
      auto lhs = dyn_cast<MemRefType>(op->getOperand(2).getType());
      auto flt = dyn_cast<MemRefType>(op->getOperand(3).getType());
      return success(acc && part && lhs && flt &&
                     acc.getElementType().isInteger(32) &&
                     part.getElementType().isF32() &&
                     lhs.getElementType().getIntOrFloatBitWidth() == 8 &&
                     flt.getElementType().getIntOrFloatBitWidth() == 8);
    }
    auto acc = dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto lhs = dyn_cast<MemRefType>(op->getOperand(1).getType());
    auto flt = dyn_cast<MemRefType>(op->getOperand(2).getType());
    if (!acc || !lhs || !flt || !acc.getElementType().isF32() ||
        !op->getOperand(3).getType().isInteger(1))
      return failure();
    if (mode == "f16")
      return success(lhs.getElementType().isF16() &&
                     flt.getElementType().isF16());
    if (mode == "bf16")
      return success(lhs.getElementType().isBF16() &&
                     flt.getElementType().isBF16());
    return failure();
  }
  LogicalResult verifyLayoutAndAddressSpace(Operation *op) const {
    auto mode = op->getAttrOfType<StringAttr>("mode");
    if (!mode || op->getNumOperands() != 4)
      return failure();
    if (mode == "i8")
      return success(hasMemSpace(op->getOperand(0), 1) &&
                     hasMemSpace(op->getOperand(1), 1) &&
                     hasMemSpace(op->getOperand(2), 1) &&
                     hasMemSpace(op->getOperand(3), 3));
    return success(hasMemSpace(op->getOperand(0), 1) &&
                   hasMemSpace(op->getOperand(1), 1) &&
                   hasMemSpace(op->getOperand(2), 3));
  }
};

struct RowReduceContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          RowReduceContract, RowReduceOp> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
  LogicalResult verifyLayoutAndAddressSpace(Operation *op) const {
    return success((hasMemSpace(op->getOperand(0), 1) ||
                    hasMemSpace(op->getOperand(0), 2)) &&
                   hasMemSpace(op->getOperand(1), 1) &&
                   hasMemSpace(op->getOperand(2), 1));
  }
};

struct ScanSumContract
    : public compilerinfra::ContractOpInterface::ExternalModel<ScanSumContract,
                                                               ScanSumOp> {
  Type getAccumulatorType(Operation *op) const {
    auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
    return dst ? dst.getElementType() : Type{};
  }
};

struct TileTransposeContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          TileTransposeContract, TileTransposeOp> {
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto src = dyn_cast<MemRefType>(op->getOperand(1).getType());
    if (!dst || !src || dst.getElementType() != src.getElementType())
      return failure();
    unsigned bits = dst.getElementTypeBitWidth();
    return success(bits == 8 || bits == 16 || bits == 32);
  }
};

struct SummaryReduceContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          SummaryReduceContract, SummaryReduceOp> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    return success(isMomentTupleType(op->getResult(0).getType()));
  }
};

struct SummaryMergeContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          SummaryMergeContract, SummaryMergeOp> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    return success(isMomentTupleType(op->getOperand(0).getType()) &&
                   isMomentTupleType(op->getOperand(1).getType()) &&
                   isMomentTupleType(op->getResult(0).getType()));
  }
};

struct CnnlMatmulContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          CnnlMatmulContract, CnnlMatmulOp> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    auto mm = cast<CnnlMatmulOp>(op);
    unsigned nOps = op->getNumOperands();
    int64_t ls = mm.getLhsSlot(), rs = mm.getRhsSlot(), ds = mm.getDstSlot();
    if (ls < 0 || rs < 0 || ds < 0 || ls >= nOps || rs >= nOps || ds >= nOps ||
        ds == ls || ds == rs)
      return failure();
    auto lTy = dyn_cast<MemRefType>(op->getOperand(ls).getType());
    auto rTy = dyn_cast<MemRefType>(op->getOperand(rs).getType());
    auto dTy = dyn_cast<MemRefType>(op->getOperand(ds).getType());
    if (!lTy || !rTy || !dTy || lTy.getRank() < 2 || lTy.getRank() > 3 ||
        rTy.getRank() < 2 || rTy.getRank() > 3 || dTy.getRank() < 2 ||
        dTy.getRank() > 3)
      return failure();
    Type elem = lTy.getElementType();
    if (elem != rTy.getElementType() || elem != dTy.getElementType() ||
        mm.getInDtype() != mm.getOutDtype())
      return failure();
    int32_t dt = mm.getInDtype();
    bool dtOk = (dt == 1 && elem.isF16()) || (dt == 2 && elem.isF32()) ||
                (dt == 17 && elem.isBF16());
    return success(dtOk && mm.getBatch() > 0 && mm.getM() > 0 &&
                   mm.getN() > 0 && mm.getK() > 0 && mm.getM() <= INT_MAX &&
                   mm.getN() <= INT_MAX && mm.getK() <= INT_MAX);
  }
  LogicalResult verifyLayoutAndAddressSpace(Operation *op) const {
    auto mm = cast<CnnlMatmulOp>(op);
    return success(hasMemSpace(op->getOperand(mm.getLhsSlot()), 0) &&
                   hasMemSpace(op->getOperand(mm.getRhsSlot()), 0) &&
                   hasMemSpace(op->getOperand(mm.getDstSlot()), 0));
  }
};

struct CnnlRowContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          CnnlRowContract, CnnlRowOp> {
  Type getAccumulatorType(Operation *op) const {
    auto kind = cast<CnnlRowOp>(op).getKind();
    if (kind == "transpose" || kind == "gather_rows")
      return {};
    return Float32Type::get(op->getContext());
  }
  arith::FastMathFlags getRequiredFastMath(Operation *op) const {
    StringRef k = cast<CnnlRowOp>(op).getKind();
    if (k == "softmax")
      return arith::FastMathFlags::afn | arith::FastMathFlags::arcp |
             arith::FastMathFlags::reassoc;
    if (k == "layer_norm")
      return arith::FastMathFlags::afn | arith::FastMathFlags::reassoc;
    return arith::FastMathFlags::none;
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    auto rop = cast<CnnlRowOp>(op);
    StringRef k = rop.getKind();
    if (k == "gather_rows") {
      if (op->getNumOperands() != 3)
        return failure();
      auto table = dyn_cast<MemRefType>(op->getOperand(0).getType());
      auto ids = dyn_cast<MemRefType>(op->getOperand(1).getType());
      auto out = dyn_cast<MemRefType>(op->getOperand(2).getType());
      auto bytes = rop.getElemBytes();
      return success(table && ids && out && table.getRank() == 2 &&
                     ids.getRank() == 1 && out.getRank() == 2 &&
                     (ids.getElementType().isSignlessInteger(32) ||
                      ids.getElementType().isSignlessInteger(64)) &&
                     table.getElementType() == out.getElementType() &&
                     (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8) &&
                     table.getElementTypeBitWidth() == 8 * bytes);
    }
    if (k == "transpose") {
      if (op->getNumOperands() != 2)
        return failure();
      auto x = dyn_cast<MemRefType>(op->getOperand(0).getType());
      auto y = dyn_cast<MemRefType>(op->getOperand(1).getType());
      if (!x || !y || x.getRank() != 2 || y.getRank() != 2 ||
          x.getElementType() != y.getElementType())
        return failure();
      int32_t eb = rop.getElemBytes();
      return success((eb == 1 || eb == 2 || eb == 4) &&
                     static_cast<int32_t>(x.getElementTypeBitWidth()) ==
                         eb * 8);
    }
    int32_t dt = rop.getDtypeCode();
    auto matchesDt = [&](Type t) {
      return (dt == 1 && t.isF16()) || (dt == 2 && t.isF32()) ||
             (dt == 17 && t.isBF16());
    };
    if (k == "softmax") {
      if (op->getNumOperands() != 2)
        return failure();
      auto x = dyn_cast<MemRefType>(op->getOperand(0).getType());
      auto y = dyn_cast<MemRefType>(op->getOperand(1).getType());
      return success(x && y && x.getRank() == 2 && y.getRank() == 2 &&
                     x.getElementType() == y.getElementType() &&
                     matchesDt(x.getElementType()));
    }
    if (k == "layer_norm") {
      if (op->getNumOperands() != 5 || !op->getOperand(4).getType().isF32())
        return failure();
      auto x = dyn_cast<MemRefType>(op->getOperand(0).getType());
      auto g = dyn_cast<MemRefType>(op->getOperand(1).getType());
      auto b = dyn_cast<MemRefType>(op->getOperand(2).getType());
      auto y = dyn_cast<MemRefType>(op->getOperand(3).getType());
      return success(x && g && b && y && x.getRank() == 2 && g.getRank() == 1 &&
                     b.getRank() == 1 && y.getRank() == 2 &&
                     x.getElementType() == g.getElementType() &&
                     x.getElementType() == b.getElementType() &&
                     x.getElementType() == y.getElementType() &&
                     matchesDt(x.getElementType()));
    }
    return failure();
  }
  LogicalResult verifyLayoutAndAddressSpace(Operation *op) const {
    for (Value v : op->getOperands())
      if (isa<MemRefType>(v.getType()) && !hasMemSpace(v, 0))
        return failure();
    return success();
  }
};

struct AddressLinearModel
    : public compilerinfra::AddressOpInterface::ExternalModel<
          AddressLinearModel, AddressOp> {
  LogicalResult
  getLinearAddress(Operation *op,
                   compilerinfra::LinearAddressDescriptor &out) const {
    auto addr = cast<AddressOp>(op);
    auto coords = addr.getCoords();
    auto strides = addr.getStrides();
    if (coords.size() != strides.size())
      return failure();
    out = compilerinfra::LinearAddressDescriptor{};
    out.baseContainer = addr.getBase();
    out.coordinates.assign(coords.begin(), coords.end());
    out.dynamicStrides.assign(strides.begin(), strides.end());
    out.constantOffset = 0;
    return success();
  }
};

struct AddressTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          AddressTiledAccessModel, AddressOp> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    auto addr = cast<AddressOp>(op);
    auto tileAxes = op->getAttrOfType<DenseI64ArrayAttr>("bangc.tile_axes");
    auto tileSizes = op->getAttrOfType<DenseI64ArrayAttr>("bangc.tile_sizes");
    if (!tileAxes || !tileSizes || tileAxes.empty() ||
        tileAxes.size() != tileSizes.size())
      return failure();
    Value base = addr.getBase();
    auto coords = addr.getCoords();
    DenseI64ArrayAttr viewShape;
    if (auto barg = dyn_cast<BlockArgument>(base)) {
      if (auto fn = op->getParentOfType<func::FuncOp>();
          fn && barg.getOwner() == &fn.front())
        viewShape =
            fn.getArgAttrOfType<DenseI64ArrayAttr>(barg.getArgNumber(),
                                                   "block.shape");
    }
    for (auto [axIdx, tileSz] :
         llvm::zip(tileAxes.asArrayRef(), tileSizes.asArrayRef())) {
      if (axIdx < 0 || static_cast<size_t>(axIdx) >= coords.size() ||
          tileSz <= 0)
        return failure();
      compilerinfra::AccessAxis ax;
      ax.container = base;
      ax.containerDim = static_cast<unsigned>(axIdx);
      if (viewShape && static_cast<size_t>(axIdx) < viewShape.size() &&
          viewShape[axIdx] >= 0)
        ax.staticExtent = viewShape[axIdx];
      ax.coord = coords[axIdx];
      ax.tileSize = tileSz;
      ax.step = 1;
      axes.push_back(ax);
    }
    return success();
  }
};
} // namespace

void NramBufferOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0));
}
void WramBufferOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0));
}
void SramBufferOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0));
}
void NramAllocOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0));
}
void LocalVarOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0));
}
void MomentArrayOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get());
}
void MomentGetOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}
void MomentSetOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

void WramConvOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  if (op->getNumOperands() == 4) {
    if (getMode() == "i8") {
      addReadWrite(effects, op, 0);
      addWrite(effects, op, 1);
      addRead(effects, op, 2);
      addRead(effects, op, 3);
    } else {
      addReadWrite(effects, op, 0);
      addRead(effects, op, 1);
      addRead(effects, op, 2);
    }
  }
}

void CycleMulOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addRead(effects, op, 2);
}

void VecMinScalarOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
}

void Bf16PanelGuardOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addRead(effects, op, 0);
  addReadWrite(effects, op, 1);
  addReadWrite(effects, op, 2);
}

void Bf16IeeeDotOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addReadWrite(effects, op, 0);
  addRead(effects, op, 1);
  addRead(effects, op, 2);
}

void Bf16AccNarrowOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void SummaryReduceOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addRead(effects, op, 0);
  addReadWrite(effects, op, 1);
}

void Memcpy1DOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void Memcpy2DOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void LoadScalarOp::getEffects(EffectList &effects) {
  addRead(effects, getOperation(), 0);
}

void StoreScalarOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
}

void SyncOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void FillOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
}

void KeyHistogramOp::getEffects(EffectList &effects) {
  auto *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addReadWrite(effects, op, 2);
  addReadWrite(effects, op, 3);
  addRead(effects, op, 4);
}

void AtomicAddOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
}

void VecBinaryOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addRead(effects, op, 2);
}

void VecScalarOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void VecCastOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void PositiveHalfOrNanOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void RowReduceOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
}

void RowExpOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
}

void RowExp2Op::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
}

void RowLogOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
}

void VecCmpMaskOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
  addRead(effects, op, 3);
}

void VecBitSelectOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
  addRead(effects, op, 3);
  addRead(effects, op, 4);
}

void QuantInitLutOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
}

void QuantInitRecordLutOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
}

void QuantE8m0GuardOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addRead(effects, op, 0);
  addRead(effects, op, 2);
  addReadWrite(effects, op, 4);
}

void QuantDecodeBlockScaledOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addReadWrite(effects, op, 2);
  addRead(effects, op, 3);
  addReadWrite(effects, op, 4);
  addReadWrite(effects, op, 5);
  addReadWrite(effects, op, 6);
}

void QuantDecodeGroupedRhsOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addRead(effects, op, 2);
  addRead(effects, op, 3);
  addRead(effects, op, 4);
  addReadWrite(effects, op, 5);
  addReadWrite(effects, op, 6);
  addReadWrite(effects, op, 7);
}

void QuantRecordGemvOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addReadWrite(effects, op, 0);
  addRead(effects, op, 1);
  addReadWrite(effects, op, 2);
  addRead(effects, op, 3);
  addRead(effects, op, 4);
  addReadWrite(effects, op, 5);
  addReadWrite(effects, op, 6);
  addReadWrite(effects, op, 7);
  addReadWrite(effects, op, 8);
}

void Reduce32SumOp::getEffects(EffectList &effects) {
  addReadWrite(effects, getOperation(), 0);
}

void SumpoolOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void TileTransposeOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
}

void Bf16TileOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  unsigned nBufs = getBuffers().size();
  if (nBufs >= 1)
    addWrite(effects, op, 0);
  if (nBufs >= 2)
    addRead(effects, op, 1);
  if (nBufs >= 3)
    addReadWrite(effects, op, 2);
}

void Bf16ProductOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addRead(effects, op, 2);
  addReadWrite(effects, op, 3);
}

void VecSqrtRnOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addReadWrite(effects, op, 2);
}

void VecSqrtOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
  addRead(effects, getOperation(), 1);
}

void VecDivApproxOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
  for (unsigned i : {1u, 2u})
    if (isa<MemRefType>(getOperation()->getOperand(i).getType()))
      addRead(effects, getOperation(), i);
  if (isa<MemRefType>(getLhs().getType()) && isa<MemRefType>(getRhs().getType()))
    addReadWrite(effects, getOperation(), 3);
}

static void quotientEffects(Operation *op, EffectList &effects) {
  addReadWrite(effects, op, 3);
  addWrite(effects, op, 0);
  for (unsigned i : {1u, 2u})
    if (isa<MemRefType>(op->getOperand(i).getType()))
      addRead(effects, op, i);
}

void VecDivRnOp::getEffects(EffectList &effects) { quotientEffects(getOperation(), effects); }
void VecGuardedDivOp::getEffects(EffectList &effects) { quotientEffects(getOperation(), effects); }

void FinitePanelOp::getEffects(EffectList &effects) {
  addRead(effects, getOperation(), 0);
  addWrite(effects, getOperation(), 1);
}
void ReserveTicketsOp::getEffects(EffectList &effects) {
  auto *op = getOperation();
  addReadWrite(effects, op, 0);
  addRead(effects, op, 1);
  for (unsigned i : {2u, 3u, 4u})
    addReadWrite(effects, op, i);
}
void MemberRampOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
}
void StableFilterOp::getEffects(EffectList &effects) {
  addWrite(effects, getOperation(), 0);
  addRead(effects, getOperation(), 1);
  addRead(effects, getOperation(), 2);
  addWrite(effects, getOperation(), 3);
}
void OrderedSelectOp::getEffects(EffectList &effects) {
  auto *op = getOperation();
  addReadWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
  addRead(effects, op, 2);
  addReadWrite(effects, op, 3);
}
void ScanSumOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addRead(effects, op, 1);
  addReadWrite(effects, op, 2);
}

void PhiloxUniformOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addWrite(effects, op, 0);
  addReadWrite(effects, op, 1);
}

void CnnlMatmulOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  addRead(effects, op, static_cast<unsigned>(getLhsSlot()));
  addRead(effects, op, static_cast<unsigned>(getRhsSlot()));
  addWrite(effects, op, static_cast<unsigned>(getDstSlot()));
}

void CnnlRowOp::getEffects(EffectList &effects) {
  Operation *op = getOperation();
  if (getKind() == "gather_rows") {
    addRead(effects, op, 0);
    addRead(effects, op, 1);
    addWrite(effects, op, 2);
  } else if (getKind() == "layer_norm") {
    addRead(effects, op, 0);
    addRead(effects, op, 1);
    addRead(effects, op, 2);
    addWrite(effects, op, 3);
  } else {
    addRead(effects, op, 0);
    addWrite(effects, op, 1);
  }
}

void SymbolOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                 SetIntRangeFn setResultRange) {
  StringRef sym = getName();
  Operation *self = getOperation();
  if (auto mat = self->getParentOfType<MatrixProgramOp>()) {
    if (sym == "BM") {
      setResultRange(getValue(), nonNegRange64(mat.getTileM(), mat.getTileM()));
      return;
    }
    if (sym == "BN") {
      setResultRange(getValue(), nonNegRange64(mat.getTileN(), mat.getTileN()));
      return;
    }
    if (sym == "BK") {
      setResultRange(getValue(), nonNegRange64(mat.getTileK(), mat.getTileK()));
      return;
    }
    if (sym == "grid") {
      if (auto extAttr = mat->getAttrOfType<DenseI64ArrayAttr>("extents")) {
        auto ext = extAttr.asArrayRef();
        if (ext.size() >= 4 && mat.getTileM() > 0 && mat.getTileN() > 0 &&
            ext[0] >= 0 && ext[1] >= 0 && ext[3] >= 0) {
          __int128 gm = llvm::divideCeil(uint64_t(ext[0]),
                                         uint64_t(mat.getTileM()));
          __int128 gn = llvm::divideCeil(uint64_t(ext[1]),
                                         uint64_t(mat.getTileN()));
          __int128 g = gm * gn * __int128(ext[3]);
          if (g >= 0 && g <= INT64_MAX) {
            setResultRange(getValue(),
                           nonNegRange64(static_cast<int64_t>(g),
                                         static_cast<int64_t>(g)));
            return;
          }
        }
      }
    }
    if (auto tasks = mat->getAttrOfType<IntegerAttr>("participants");
        tasks && tasks.getInt() > 0) {
      if (sym == "taskId") {
        setResultRange(getValue(), nonNegRange64(0, tasks.getInt() - 1));
        return;
      }
      if (sym == "taskDim") {
        setResultRange(getValue(), nonNegRange64(1, tasks.getInt()));
        return;
      }
      if (sym == "clusterId") {
        int64_t nCl = std::max<int64_t>(1, tasks.getInt() / 4);
        setResultRange(getValue(), nonNegRange64(0, nCl - 1));
        return;
      }
      if (sym == "clusterDim") {
        int64_t nCl = std::max<int64_t>(1, tasks.getInt() / 4);
        setResultRange(getValue(), nonNegRange64(nCl, nCl));
        return;
      }
    }
  }
  if (auto ord = self->getParentOfType<OrderedProgramOp>()) {
    if (sym == "grid" && ord.getOwnerTile() > 0 && ord.getOwners() >= 0) {
      int64_t g = static_cast<int64_t>(llvm::divideCeil(
          uint64_t(ord.getOwners()), uint64_t(ord.getOwnerTile())));
      setResultRange(getValue(), nonNegRange64(g, g));
      return;
    }
    int64_t tasks = ord.getParticipants();
    if (tasks > 0) {
      if (sym == "taskId") {
        setResultRange(getValue(), nonNegRange64(0, tasks - 1));
        return;
      }
      if (sym == "taskDim") {
        setResultRange(getValue(), nonNegRange64(1, tasks));
        return;
      }
    }
  }
  auto fn = self->getParentOfType<func::FuncOp>();
  if (fn) {
    auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
    auto layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
    if (configs && layout && layout.getAs<ArrayAttr>("rows") &&
        !layout.getAs<ArrayAttr>("rows").empty()) {
      int64_t batch = layout.getAs<IntegerAttr>("batch")
                          ? layout.getAs<IntegerAttr>("batch").getInt()
                          : 1;
      int64_t minR = 0, maxR = 0, minW = 0, maxW = 0, minP = 0, maxP = 0,
              minB = 0, maxB = 0;
      bool first = true;
      for (auto a : layout.getAs<ArrayAttr>("rows")) {
        auto alloc = cast<DictionaryAttr>(a);
        auto id = alloc.getAs<IntegerAttr>("id");
        auto p = alloc.getAs<IntegerAttr>("pitch").getInt();
        auto bytes = alloc.getAs<IntegerAttr>("bytes").getInt();
        for (auto ca : configs) {
          auto row = cast<DictionaryAttr>(ca);
          if (row.get("id") != id)
            continue;
          auto r = row.getAs<IntegerAttr>("rows").getInt();
          auto w = row.getAs<IntegerAttr>("width").getInt();
          if (first) {
            minR = maxR = r;
            minW = maxW = w;
            minP = maxP = p;
            minB = maxB = bytes;
            first = false;
          } else {
            minR = std::min(minR, r);
            maxR = std::max(maxR, r);
            minW = std::min(minW, w);
            maxW = std::max(maxW, w);
            minP = std::min(minP, p);
            maxP = std::max(maxP, p);
            minB = std::min(minB, bytes);
            maxB = std::max(maxB, bytes);
          }
        }
      }
      if (!first && maxR <= INT32_MAX / std::max<int64_t>(1, batch) &&
          maxW <= INT32_MAX && maxP <= INT32_MAX && maxB <= INT32_MAX) {
        if (sym == "R") {
          setResultRange(getValue(), nonNegRange64(minR, maxR));
          return;
        }
        if (sym == "B") {
          setResultRange(getValue(), nonNegRange64(batch, batch));
          return;
        }
        if (sym == "EB") {
          setResultRange(getValue(),
                         nonNegRange64(minR * batch, maxR * batch));
          return;
        }
        if (sym == "W") {
          setResultRange(getValue(), nonNegRange64(minW, maxW));
          return;
        }
        if (sym == "P") {
          setResultRange(getValue(), nonNegRange64(minP, maxP));
          return;
        }
        if (sym == "BYTES") {
          setResultRange(getValue(), nonNegRange64(minB, maxB));
          return;
        }
      }
    }
  }
  setResultRange(getValue(), ConstantIntRanges::maxRange(64));
}

void BangcSrcDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  auto *ctx = getContext();
  WramConvOp::attachInterface<WramConvContract>(*ctx);
  CycleMulOp::attachInterface<F32AccContract<CycleMulOp>>(*ctx);
  VecMinScalarOp::attachInterface<F32AccContract<VecMinScalarOp>>(*ctx);
  Bf16PanelGuardOp::attachInterface<I16AccContract<Bf16PanelGuardOp>>(*ctx);
  Bf16IeeeDotOp::attachInterface<F32AccContract<Bf16IeeeDotOp>>(*ctx);
  Bf16AccNarrowOp::attachInterface<F32AccContract<Bf16AccNarrowOp>>(*ctx);
  I32ToBf16Op::attachInterface<I32AccContract<I32ToBf16Op>>(*ctx);
  KeyHistogramOp::attachInterface<I32AccContract<KeyHistogramOp>>(*ctx);
  RowReduceOp::attachInterface<RowReduceContract>(*ctx);
  RowExpOp::attachInterface<F32AccContract<RowExpOp>>(*ctx);
  RowExp2Op::attachInterface<F32AccContract<RowExp2Op>>(*ctx);
  RowLogOp::attachInterface<F32AccContract<RowLogOp>>(*ctx);
  VecCmpMaskOp::attachInterface<I32AccContract<VecCmpMaskOp>>(*ctx);
  VecBitSelectOp::attachInterface<I32AccContract<VecBitSelectOp>>(*ctx);
  QuantInitLutOp::attachInterface<I32AccContract<QuantInitLutOp>>(*ctx);
  QuantInitRecordLutOp::attachInterface<I16AccContract<QuantInitRecordLutOp>>(
      *ctx);
  QuantE8m0GuardOp::attachInterface<I16AccContract<QuantE8m0GuardOp>>(*ctx);
  QuantDecodeBlockScaledOp::attachInterface<
      F32AccContract<QuantDecodeBlockScaledOp>>(*ctx);
  QuantDecodeGroupedRhsOp::attachInterface<
      F32AccContract<QuantDecodeGroupedRhsOp>>(*ctx);
  QuantRecordGemvOp::attachInterface<F32AccContract<QuantRecordGemvOp>>(*ctx);
  Reduce32SumOp::attachInterface<F32AccContract<Reduce32SumOp>>(*ctx);
  SumpoolOp::attachInterface<F32AccContract<SumpoolOp>>(*ctx);
  TileTransposeOp::attachInterface<TileTransposeContract>(*ctx);
  Bf16TileOp::attachInterface<F32AccContract<Bf16TileOp>>(*ctx);
  Bf16ProductOp::attachInterface<F32AccContract<Bf16ProductOp>>(*ctx);
  VecSqrtRnOp::attachInterface<F32AccContract<VecSqrtRnOp>>(*ctx);
  VecSqrtOp::attachInterface<F32AccContract<VecSqrtOp>>(*ctx);
  VecDivApproxOp::attachInterface<F32AccContract<VecDivApproxOp>>(*ctx);
  VecGuardedDivOp::attachInterface<F32AccContract<VecGuardedDivOp>>(*ctx);
  VecDivRnOp::attachInterface<F32AccContract<VecDivRnOp>>(*ctx);
  DivRnOp::attachInterface<F32AccContract<DivRnOp>>(*ctx);
  Bf16ScalarOp::attachInterface<F32AccContract<Bf16ScalarOp>>(*ctx);
  FinitePanelOp::attachInterface<I16AccContract<FinitePanelOp>>(*ctx);
  ReserveTicketsOp::attachInterface<I32AccContract<ReserveTicketsOp>>(*ctx);
  MemberRampOp::attachInterface<I32AccContract<MemberRampOp>>(*ctx);
  StableFilterOp::attachInterface<I32AccContract<StableFilterOp>>(*ctx);
  OrderedSelectOp::attachInterface<F32AccContract<OrderedSelectOp>>(*ctx);
  ScanSumOp::attachInterface<ScanSumContract>(*ctx);
  SummaryReduceOp::attachInterface<SummaryReduceContract>(*ctx);
  SummaryMergeOp::attachInterface<SummaryMergeContract>(*ctx);
  PhiloxUniformOp::attachInterface<I32AccContract<PhiloxUniformOp>>(*ctx);
  PhiloxAtOp::attachInterface<I32AccContract<PhiloxAtOp>>(*ctx);
  CnnlMatmulOp::attachInterface<CnnlMatmulContract>(*ctx);
  CnnlRowOp::attachInterface<CnnlRowContract>(*ctx);
  AddressOp::attachInterface<AddressLinearModel>(*ctx);
  AddressOp::attachInterface<AddressTiledAccessModel>(*ctx);
}
