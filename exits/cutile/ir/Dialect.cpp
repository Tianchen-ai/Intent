#include "compilerinfra/Assembly.h"
#include "compilerinfra/Contract.h"
#include "compilerinfra/Proofs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

template <typename ConcreteOp, bool RequireReassoc>
struct CutileAtomicContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          CutileAtomicContractModel<ConcreteOp, RequireReassoc>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return op->getNumOperands() >= 3
               ? getElementTypeOrSelf(op->getOperand(2).getType())
               : Type{};
  }
  arith::FastMathFlags getRequiredFastMath(Operation *op) const {
    if constexpr (!RequireReassoc)
      return arith::FastMathFlags::none;
    Type acc = getAccumulatorType(op);
    return (acc && isa<FloatType>(acc)) ? arith::FastMathFlags::reassoc
                                        : arith::FastMathFlags::none;
  }
};

struct CutileLibraryGemmContractModel
    : public compilerinfra::ContractOpInterface::ExternalModel<
          CutileLibraryGemmContractModel, intentdsl::cutile::LibraryGemm> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
  LogicalResult verifyShapeAndTypes(Operation *op) const {
    if (op->getNumResults() != 0 || op->getNumOperands() != 3)
      return failure();
    auto a = dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto b = dyn_cast<MemRefType>(op->getOperand(1).getType());
    auto c = dyn_cast<MemRefType>(op->getOperand(2).getType());
    if (!a || !b || !c || a.getRank() != 2 || b.getRank() != 2 ||
        c.getRank() != 2)
      return failure();
    Type el = a.getElementType();
    return success((el.isF16() || el.isBF16() || el.isF32()) &&
                   b.getElementType() == el && c.getElementType() == el);
  }
};

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
      } else if (mode && mode.getValue() == "split_dequant") {
        auto ext = prog->getAttrOfType<DenseI64ArrayAttr>("block.dequant_extents");
        auto dqTile = prog->getAttrOfType<DenseI64ArrayAttr>("block.dequant_tile");
        if (ext && ext.size() == 2 && dqTile && dqTile.size() == 2) {
          if (dqTile[1] == 1 && ax == 0 && dqTile[0] > 0)
            hi = std::max<int64_t>(
                0, llvm::divideCeil(ext[0] * ext[1], dqTile[0]) - 1);
          else if (ax == 0 && dqTile[0] > 0)
            hi = std::max<int64_t>(0, llvm::divideCeil(ext[0], dqTile[0]) - 1);
          else if (ax == 1 && dqTile[1] > 0)
            hi = std::max<int64_t>(0, llvm::divideCeil(ext[1], dqTile[1]) - 1);
        }
      } else if (mode && mode.getValue() == "owners" && ax == 0) {
        auto owners = prog->getAttrOfType<IntegerAttr>("block.owners");
        auto tile = prog->getAttrOfType<IntegerAttr>("block.owner_tile");
        if (owners && tile && tile.getInt() > 0)
          hi = std::max<int64_t>(
              0, llvm::divideCeil(owners.getInt(), tile.getInt()) - 1);
      } else if (mode && mode.getValue() == "owners" && ax == 1) {
        if (auto fs = prog->getAttrOfType<IntegerAttr>("block.fold_split");
            fs && fs.getInt() > 0)
          hi = std::max<int64_t>(0, fs.getInt() - 1);
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

std::optional<int64_t> resolveCutileViewExtent(Operation *ctxOp, Value view,
                                               int64_t dim) {
  if (!view || dim < 0)
    return std::nullopt;
  if (auto st = dyn_cast<ShapedType>(view.getType());
      st && dim < st.getRank() && !st.isDynamicDim(dim))
    return st.getDimSize(dim);
  if (auto barg = dyn_cast<BlockArgument>(view)) {
    if (auto fn = ctxOp->getParentOfType<func::FuncOp>();
        fn && barg.getOwner() == &fn.front()) {
      for (StringRef attrName : {"ct_src.shape", "block.shape", "kir.shape"}) {
        if (auto sh =
                fn.getArgAttrOfType<DenseI64ArrayAttr>(barg.getArgNumber(),
                                                       attrName);
            sh && static_cast<size_t>(dim) < sh.size() && sh[dim] >= 0)
          return sh[dim];
      }
    }
  }
  return std::nullopt;
}

struct CutileOwnerSpaceIndexedRegionModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          CutileOwnerSpaceIndexedRegionModel,
          intentdsl::cutile::OrderedOwnerSpace> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front())
      return failure();
    unsigned idx = arg.getArgNumber();
    auto axes = op->getAttrOfType<DenseI64ArrayAttr>("axes");
    if (!axes || idx >= axes.size() || idx >= op->getNumOperands())
      return failure();
    auto ext = resolveCutileViewExtent(op, op->getOperand(idx), axes[idx]);
    if (!ext || *ext <= 0)
      return failure();
    int64_t step = 1;
    if (auto pair = op->getAttrOfType<IntegerAttr>("block.pair_axis")) {
      if (auto tile = op->getAttrOfType<IntegerAttr>("block.owner_tile");
          tile && pair.getInt() == static_cast<int64_t>(idx) &&
          tile.getInt() > 1) {
        auto owners = op->getAttrOfType<IntegerAttr>("block.owners");
        if (*ext < tile.getInt() || (*ext % tile.getInt()) != 0 || !owners ||
            owners.getInt() <= 0 || (owners.getInt() % tile.getInt()) != 0)
          return failure();
        step = tile.getInt();
      }
    }
    out = compilerinfra::RegionIndexBounds{};
    out.staticLower = 0;
    out.staticUpperExclusive = *ext;
    out.staticStep = step;
    return success();
  }
};

template <typename ConcreteOp, bool IsWrite>
struct CutileSliceTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          CutileSliceTiledAccessModel<ConcreteOp, IsWrite>, ConcreteOp> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < (IsWrite ? 2u : 1u))
      return failure();
    Value view = op->getOperand(IsWrite ? 1 : 0);
    Value vecVal = IsWrite ? op->getOperand(0) : op->getResult(0);
    auto mt = dyn_cast<MemRefType>(view.getType());
    auto vt = dyn_cast<RankedTensorType>(vecVal.getType());
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    auto coords = op->getOperands().drop_front(IsWrite ? 2 : 1);
    if (!mt || !vt || vt.getRank() != 1 || !axisAttr)
      return failure();
    int64_t sliceAxis = axisAttr.getInt();
    if (sliceAxis < 0 || sliceAxis >= mt.getRank() ||
        coords.size() + 1 != static_cast<size_t>(mt.getRank()))
      return failure();

    Value pairArg;
    int64_t ownerTile = 1;
    if (auto space =
            dyn_cast_or_null<intentdsl::cutile::OrderedOwnerSpace>(
                op->getParentOp())) {
      if (auto pair = space->getAttrOfType<IntegerAttr>("block.pair_axis")) {
        if (auto tile = space->getAttrOfType<IntegerAttr>("block.owner_tile");
            tile && tile.getInt() > 1 && space.getRegion().hasOneBlock()) {
          Block &bb = space.getRegion().front();
          if (pair.getInt() >= 0 &&
              static_cast<unsigned>(pair.getInt()) < bb.getNumArguments()) {
            pairArg = bb.getArgument(pair.getInt());
            ownerTile = tile.getInt();
          }
        }
      }
    }

    size_t coordPos = 0;
    for (int64_t d = 0; d < mt.getRank(); ++d) {
      compilerinfra::AccessAxis ax;
      ax.container = view;
      ax.containerDim = static_cast<unsigned>(d);
      ax.staticExtent = resolveCutileViewExtent(op, view, d);
      if (d == sliceAxis) {
        ax.entireAxisFromZero = true;
        int64_t width = !vt.isDynamicDim(0) ? vt.getDimSize(0)
                                            : ax.staticExtent.value_or(0);
        if (width <= 0)
          return failure();
        ax.tileSize = width;
        ax.step = 1;
      } else {
        Value c = coords[coordPos++];
        ax.coord = c;
        ax.tileSize = (pairArg && c == pairArg) ? ownerTile : 1;
        ax.step = 1;
      }
      axes.push_back(ax);
    }
    return success();
  }
};

struct CutileTileAxisRangeModel
    : public InferIntRangeInterface::ExternalModel<CutileTileAxisRangeModel,
                                                   intentdsl::cutile::TileAxis> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange>,
                                     SetIntLatticeFn setResultRanges) const {
    auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto it = rt ? dyn_cast<IntegerType>(rt.getElementType()) : IntegerType{};
    if (!rt || rt.getRank() != 1 || rt.getDimSize(0) <= 0 || !it ||
        it.getWidth() == 0 || it.getWidth() > 64)
      return;
    unsigned bw = it.getWidth();
    APInt lo(bw, 0), hi(bw, static_cast<uint64_t>(rt.getDimSize(0) - 1));
    setResultRanges(op->getResult(0),
                    ConstantIntRanges::fromSigned(lo, hi).intersection(
                        ConstantIntRanges::fromUnsigned(lo, hi)));
  }
};

struct CutileTileAddressModel
    : public compilerinfra::AddressOpInterface::ExternalModel<
          CutileTileAddressModel, intentdsl::cutile::TileAddr> {
  LogicalResult
  getLinearAddress(Operation *op,
                   compilerinfra::LinearAddressDescriptor &out) const {
    auto strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
    auto coords = op->getOperands().drop_front();
    if (op->getNumOperands() < 1 || !strides ||
        strides.size() != coords.size())
      return failure();
    out = compilerinfra::LinearAddressDescriptor{};
    out.baseContainer = op->getOperand(0);
    out.coordinates.assign(coords.begin(), coords.end());
    out.staticStrides.assign(strides.asArrayRef().begin(),
                             strides.asArrayRef().end());
    out.constantOffset = 0;
    return success();
  }
};

struct CutileStripeLoopIndexedRegionModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          CutileStripeLoopIndexedRegionModel, intentdsl::cutile::StripeLoop> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front() || arg.getArgNumber() != 0)
      return failure();
    auto count = op->getAttrOfType<IntegerAttr>("count");
    if (!count || count.getInt() <= 0)
      return failure();
    out = compilerinfra::RegionIndexBounds{};
    out.staticLower = 0;
    out.staticUpperExclusive = count.getInt();
    out.staticStep = 1;
    return success();
  }
};

struct CutileBinPanelsIndexedRegionModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          CutileBinPanelsIndexedRegionModel, intentdsl::cutile::BinPanels> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front() ||
        arg.getArgNumber() != 0 || op->getNumOperands() != 1)
      return failure();
    out = compilerinfra::RegionIndexBounds{};
    out.staticLower = 0;
    out.dynamicUpperExclusive = op->getOperand(0);
    out.staticStep = 1;
    return success();
  }
};

struct CutilePanelFoldIndexedRegionModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          CutilePanelFoldIndexedRegionModel,
          intentdsl::cutile::NativePanelsFold> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front() ||
        arg.getArgNumber() != 0 ||
        !isa<IndexType, IntegerType>(arg.getType()))
      return failure();
    auto bound = op->getAttrOfType<IntegerAttr>("bound");
    auto step = op->getAttrOfType<IntegerAttr>("step");
    if (!bound || !step || bound.getInt() <= 0 || step.getInt() <= 0)
      return failure();
    out = compilerinfra::RegionIndexBounds{};
    out.staticLower = 0;
    out.staticUpperExclusive = bound.getInt();
    out.staticStep = step.getInt();
    return success();
  }
};

struct CutileMemberLoadAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          CutileMemberLoadAccessModel, intentdsl::cutile::MemberLoad> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < 2)
      return failure();
    Value view = op->getOperand(0);
    auto mt = dyn_cast<MemRefType>(view.getType());
    auto coords = op->getOperands().drop_front(1);
    if (!mt || coords.size() != static_cast<size_t>(mt.getRank()))
      return failure();
    for (auto [d, c] : llvm::enumerate(coords)) {
      if (!isa<IndexType, IntegerType>(c.getType()))
        return failure();
      compilerinfra::AccessAxis ax;
      ax.container = view;
      ax.containerDim = static_cast<unsigned>(d);
      ax.staticExtent =
          resolveCutileViewExtent(op, view, static_cast<int64_t>(d));
      ax.coord = c;
      ax.tileSize = 1;
      ax.step = 1;
      axes.push_back(ax);
    }
    return success();
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
  LibraryGemm::attachInterface<CutileLibraryGemmContractModel>(*ctx);
  TileAtomicAdd::attachInterface<
      CutileAtomicContractModel<TileAtomicAdd, true>>(*ctx);
  TileAtomicFetchAdd::attachInterface<
      CutileAtomicContractModel<TileAtomicFetchAdd, true>>(*ctx);
  TileAtomicXchg::attachInterface<
      CutileAtomicContractModel<TileAtomicXchg, false>>(*ctx);
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
  TileAxis::attachInterface<CutileTileAxisRangeModel>(*ctx);
  TileRead::attachInterface<CutileTileReadRangeModel>(*ctx);
  TileAddr::attachInterface<CutileTileAddressModel>(*ctx);
  OrderedOwnerSpace::attachInterface<CutileOwnerSpaceIndexedRegionModel>(*ctx);
  StripeLoop::attachInterface<CutileStripeLoopIndexedRegionModel>(*ctx);
  BinPanels::attachInterface<CutileBinPanelsIndexedRegionModel>(*ctx);
  NativePanelsFold::attachInterface<CutilePanelFoldIndexedRegionModel>(*ctx);
  OrderedSliceRead::attachInterface<
      CutileSliceTiledAccessModel<OrderedSliceRead, false>>(*ctx);
  OrderedSliceWrite::attachInterface<
      CutileSliceTiledAccessModel<OrderedSliceWrite, true>>(*ctx);
  MemberLoad::attachInterface<CutileMemberLoadAccessModel>(*ctx);
}

namespace CI_UNIT_NAMESPACE {
void registerIR(mlir::DialectRegistry &r) {
  r.insert<intentdsl::cutile::TileSource>();
}
} // namespace CI_UNIT_NAMESPACE
