#include "intentdsl/levels/execution/block/ir/Dialect.h"
#include "Dialect.cpp.inc"
#include "compilerinfra/Proofs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace {
using namespace mlir;
using namespace intentdsl::iblock;

std::optional<int64_t> resolveViewExtent(Operation *ctxOp, Value view,
                                         int64_t dim) {
  if (!view || dim < 0)
    return std::nullopt;
  if (auto st = dyn_cast<ShapedType>(view.getType());
      st && dim < st.getRank() && !st.isDynamicDim(dim))
    return st.getDimSize(dim);
  if (auto barg = dyn_cast<BlockArgument>(view)) {
    if (auto fn = ctxOp->getParentOfType<func::FuncOp>();
        fn && barg.getOwner() == &fn.front()) {
      for (StringRef attrName : {"block.shape", "kir.shape"}) {
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

struct OwnerSpaceIndexedRegionModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          OwnerSpaceIndexedRegionModel, OrderedOwnerSpace> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front())
      return failure();
    unsigned idx = arg.getArgNumber();
    auto axes = op->getAttrOfType<DenseI64ArrayAttr>("axes");
    if (!axes || idx >= axes.size() || idx >= op->getNumOperands())
      return failure();
    auto ext = resolveViewExtent(op, op->getOperand(idx), axes[idx]);
    if (!ext)
      return failure();
    int64_t step = 1;
    if (auto pair = op->getAttrOfType<IntegerAttr>("block.pair_axis")) {
      if (auto tile = op->getAttrOfType<IntegerAttr>("block.owner_tile");
          tile && pair.getInt() == static_cast<int64_t>(idx) &&
          tile.getInt() > 1)
        step = tile.getInt();
    }
    out = compilerinfra::RegionIndexBounds{};
    out.staticLower = 0;
    out.staticUpperExclusive = *ext;
    out.staticStep = step;
    return success();
  }
};

template <typename ConcreteOp, bool IsWrite>
struct SliceTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          SliceTiledAccessModel<ConcreteOp, IsWrite>, ConcreteOp> {
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
    if (auto space = dyn_cast_or_null<OrderedOwnerSpace>(op->getParentOp())) {
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
      ax.staticExtent = resolveViewExtent(op, view, d);
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
} // namespace

void intentdsl::iblock::IblockDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  auto *ctx = getContext();
  OrderedOwnerSpace::attachInterface<OwnerSpaceIndexedRegionModel>(*ctx);
  OrderedSliceRead::attachInterface<
      SliceTiledAccessModel<OrderedSliceRead, false>>(*ctx);
  OrderedSliceWrite::attachInterface<
      SliceTiledAccessModel<OrderedSliceWrite, true>>(*ctx);
}
