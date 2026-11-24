#include "intentdsl/exits/triton/ir/Dialect.h"
#include "TileShape.h"
#include "compilerinfra/Contract.h"
#include "Dialect.cpp.inc"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace {
using namespace mlir;

struct TritonDotContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          TritonDotContractModel, intentdsl::itsrc::TileDot> {
  Type getAccumulatorType(Operation *op) const {
    return getElementTypeOrSelf(op->getResult(0).getType());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if (op->getNumResults() != 1 ||
        (op->getNumOperands() != 2 && op->getNumOperands() != 3))
      return failure();
    auto trAttr = op->getAttrOfType<BoolAttr>("transpose_rhs");
    if (!trAttr)
      return failure();
    auto l = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto r = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto m = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!l || !r || !m || l.getRank() != 2 || r.getRank() != 2 ||
        m.getRank() != 2 || !l.hasStaticShape() || !r.hasStaticShape() ||
        !m.hasStaticShape())
      return failure();
    bool tr = trAttr.getValue();
    return success(l.getDimSize(1) == r.getDimSize(tr ? 1 : 0) &&
                   m.getDimSize(0) == l.getDimSize(0) &&
                   m.getDimSize(1) == r.getDimSize(tr ? 0 : 1));
  }
};

struct TritonScaledMMAContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          TritonScaledMMAContractModel, intentdsl::itsrc::ScaledMMA> {
  Type getAccumulatorType(Operation *op) const {
    return getElementTypeOrSelf(op->getResult(0).getType());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if (op->getNumResults() != 1 || op->getNumOperands() != 5)
      return failure();
    auto l = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto ls = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto r = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
    auto rs = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
    auto m = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!l || !ls || !r || !rs || !m || l.getRank() != 2 || ls.getRank() != 2 ||
        r.getRank() != 2 || rs.getRank() != 2 || m.getRank() != 2 ||
        !l.hasStaticShape() || !ls.hasStaticShape() || !r.hasStaticShape() ||
        !rs.hasStaticShape() || !m.hasStaticShape() ||
        op->getOperand(4).getType() != m)
      return failure();
    return success(l.getDimSize(1) > 0 && l.getDimSize(1) % 32 == 0 &&
                   l.getDimSize(1) == r.getDimSize(0) &&
                   m.getDimSize(0) == l.getDimSize(0) &&
                   m.getDimSize(1) == r.getDimSize(1) &&
                   ls.getDimSize(0) == l.getDimSize(0) &&
                   ls.getDimSize(1) == l.getDimSize(1) / 32 &&
                   rs.getDimSize(0) == r.getDimSize(1) &&
                   rs.getDimSize(1) == l.getDimSize(1) / 32);
  }
};

template <typename ConcreteOp>
struct TritonPureContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          TritonPureContractModel<ConcreteOp>, ConcreteOp> {};

struct TritonTilePidRangeModel
    : public InferIntRangeInterface::ExternalModel<TritonTilePidRangeModel,
                                                   intentdsl::itsrc::TilePid> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange>,
                                     SetIntLatticeFn setResultRanges) const {
    int64_t hi = INT32_MAX;
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (auto *prog = op->getParentOp();
        prog && axisAttr &&
        prog->getName().getStringRef() == "triton_src.tile_program") {
      int64_t ax = axisAttr.getInt();
      auto mode = prog->getAttrOfType<StringAttr>("calling_convention");
      if (mode && mode.getValue() == "matrix") {
        auto dims = prog->getAttrOfType<DenseI64ArrayAttr>("block.extents");
        auto tile = prog->getAttrOfType<DenseI64ArrayAttr>("block.output_tile");
        if (dims && dims.size() == 4 && tile && tile.size() == 3) {
          if (ax == 0 && tile[0] > 0)
            hi = std::max<int64_t>(0, llvm::divideCeil(dims[0], tile[0]) - 1);
          else if (ax == 1 && tile[1] > 0)
            hi = std::max<int64_t>(0, llvm::divideCeil(dims[1], tile[1]) - 1);
          else if (ax == 2)
            hi = std::max<int64_t>(0, dims[3] - 1);
        }
      } else if (mode && mode.getValue() == "owners" && ax == 0) {
        auto owners = prog->getAttrOfType<IntegerAttr>("block.owners");
        auto tile = prog->getAttrOfType<IntegerAttr>("block.owner_tile");
        if (owners && tile && tile.getInt() > 0)
          hi = std::max<int64_t>(
              0, llvm::divideCeil(owners.getInt(), tile.getInt()) - 1);
      }
    }
    hi = std::min<int64_t>(hi, INT32_MAX);
    setResultRanges(op->getResult(0),
                    ConstantIntRanges::fromSigned(APInt(64, 0), APInt(64, hi)));
  }
};

struct TritonTileReadRangeModel
    : public InferIntRangeInterface::ExternalModel<
          TritonTileReadRangeModel, intentdsl::itsrc::TileRead> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange>,
                                     SetIntLatticeFn setResultRanges) const {
    Value res = op->getResult(0);
    if (!isa<ShapedType>(res.getType())) {
      if (auto it = dyn_cast<IntegerType>(res.getType());
          it && it.getWidth() > 0)
        setResultRanges(res, ConstantIntRanges::maxRange(it.getWidth()));
    }
  }
};
} // namespace

void intentdsl::itsrc::ItsrcDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  MLIRContext *ctx = getContext();
  TileDot::attachInterface<TritonDotContractModel>(*ctx);
  ScaledMMA::attachInterface<TritonScaledMMAContractModel>(*ctx);
  TileReduce::attachInterface<TritonPureContractModel<TileReduce>>(*ctx);
  RowReduce::attachInterface<TritonPureContractModel<RowReduce>>(*ctx);
  BinCount::attachInterface<TritonPureContractModel<BinCount>>(*ctx);
  reduce::attachInterface<TritonPureContractModel<reduce>>(*ctx);
  Dialect_ScanSum::attachInterface<TritonPureContractModel<Dialect_ScanSum>>(
      *ctx);
  Dialect_summary_reduceOp::attachInterface<
      TritonPureContractModel<Dialect_summary_reduceOp>>(*ctx);
  Dialect_summary_mergeOp::attachInterface<
      TritonPureContractModel<Dialect_summary_mergeOp>>(*ctx);
  TilePid::attachInterface<TritonTilePidRangeModel>(*ctx);
  TileRead::attachInterface<TritonTileReadRangeModel>(*ctx);
}
