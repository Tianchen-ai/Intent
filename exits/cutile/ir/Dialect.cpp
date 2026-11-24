#include "compilerinfra/Assembly.h"
#include "compilerinfra/Contract.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace {
using namespace mlir;

template <typename ConcreteOp>
struct CutileMatrixContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          CutileMatrixContractModel<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return getElementTypeOrSelf(op->getResult(0).getType());
  }
};

template <typename ConcreteOp>
struct CutilePureContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          CutilePureContractModel<ConcreteOp>, ConcreteOp> {};

struct CutileTilePidRangeModel
    : public InferIntRangeInterface::ExternalModel<CutileTilePidRangeModel,
                                                   intentdsl::cutile::TilePid> {
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
        prog->getName().getStringRef() == "ct_src.tile_program") {
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

struct CutileTileReadRangeModel
    : public InferIntRangeInterface::ExternalModel<
          CutileTileReadRangeModel, intentdsl::cutile::TileRead> {
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

void intentdsl::cutile::TileSource::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  MLIRContext *ctx = getContext();
  TileDot::attachInterface<CutileMatrixContractModel<TileDot>>(*ctx);
  NativeScaleMMA::attachInterface<CutileMatrixContractModel<NativeScaleMMA>>(
      *ctx);
  TileReduce::attachInterface<CutilePureContractModel<TileReduce>>(*ctx);
  RowReduce::attachInterface<CutilePureContractModel<RowReduce>>(*ctx);
  Reduce::attachInterface<CutilePureContractModel<Reduce>>(*ctx);
  ScanSum::attachInterface<CutilePureContractModel<ScanSum>>(*ctx);
  summary_reduceOp::attachInterface<CutilePureContractModel<summary_reduceOp>>(
      *ctx);
  summary_mergeOp::attachInterface<CutilePureContractModel<summary_mergeOp>>(
      *ctx);
  Uniform::attachInterface<CutilePureContractModel<Uniform>>(*ctx);
  TilePid::attachInterface<CutileTilePidRangeModel>(*ctx);
  TileRead::attachInterface<CutileTileReadRangeModel>(*ctx);
}

namespace CI_UNIT_NAMESPACE {
void registerIR(mlir::DialectRegistry &r) {
  r.insert<intentdsl::cutile::TileSource>();
}
} // namespace CI_UNIT_NAMESPACE
