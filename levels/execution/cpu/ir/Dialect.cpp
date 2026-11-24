#include "intentdsl/levels/execution/cpu/ir/Dialect.h"
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
using namespace intentdsl::cpu;

std::optional<int64_t> resolveCpuViewExtent(Operation *ctxOp, Value view,
                                            int64_t dim) {
  if (!view || dim < 0)
    return std::nullopt;
  if (auto st = dyn_cast<ShapedType>(view.getType());
      st && dim < st.getRank() && !st.isDynamicDim(dim))
    return st.getDimSize(dim);
  if (auto barg = dyn_cast<BlockArgument>(view)) {
    if (auto fn = ctxOp->getParentOfType<func::FuncOp>();
        fn && barg.getOwner() == &fn.front()) {
      for (StringRef attrName : {"cpu.shape", "kir.shape"}) {
        if (auto sh = fn.getArgAttrOfType<DenseI64ArrayAttr>(
                barg.getArgNumber(), attrName);
            sh && static_cast<size_t>(dim) < sh.size() && sh[dim] >= 0)
          return sh[dim];
      }
    }
  }
  return std::nullopt;
}

std::pair<Value, int64_t> findEnclosingInnerOwnerBatch(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto space = dyn_cast<OrderedOwnerSpace>(p)) {
      auto inner = space->getAttrOfType<IntegerAttr>("cpu.innermost_owner");
      auto batch = space->getAttrOfType<IntegerAttr>("cpu.owner_batch");
      if (inner && batch && batch.getInt() > 1 &&
          space.getRegion().hasOneBlock()) {
        Block &bb = space.getRegion().front();
        if (inner.getInt() >= 0 &&
            static_cast<unsigned>(inner.getInt()) < bb.getNumArguments())
          return {bb.getArgument(inner.getInt()), batch.getInt()};
      }
      break;
    }
  }
  return {{}, 1};
}

struct CpuOwnerSpaceIndexedRegionModel
    : public compilerinfra::IndexedRegionOpInterface::ExternalModel<
          CpuOwnerSpaceIndexedRegionModel, OrderedOwnerSpace> {
  LogicalResult getRegionArgBounds(Operation *op, BlockArgument arg,
                                   compilerinfra::RegionIndexBounds &out) const {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
        arg.getOwner() != &op->getRegion(0).front())
      return failure();
    unsigned idx = arg.getArgNumber();
    auto axes = op->getAttrOfType<DenseI64ArrayAttr>("axes");
    if (!axes || idx >= axes.size() || idx >= op->getNumOperands())
      return failure();
    auto ext = resolveCpuViewExtent(op, op->getOperand(idx), axes[idx]);
    if (!ext || *ext <= 0)
      return failure();
    int64_t step = 1;
    int64_t upperExcl = *ext;
    if (auto inner = op->getAttrOfType<IntegerAttr>("cpu.innermost_owner")) {
      if (auto batch = op->getAttrOfType<IntegerAttr>("cpu.owner_batch");
          batch && inner.getInt() == static_cast<int64_t>(idx) &&
          batch.getInt() > 1) {
        int64_t t = batch.getInt();
        int64_t groups = (*ext + t - 1) / t;
        upperExcl = groups * t;
        step = t;
      }
    }
    out = compilerinfra::RegionIndexBounds{};
    out.staticLower = 0;
    out.staticUpperExclusive = upperExcl;
    out.staticStep = step;
    return success();
  }
};

template <typename ConcreteOp, bool IsWrite>
struct CpuSliceTiledAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          CpuSliceTiledAccessModel<ConcreteOp, IsWrite>, ConcreteOp> {
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

    auto [innerArg, ownerBatch] = findEnclosingInnerOwnerBatch(op);
    size_t coordPos = 0;
    for (int64_t d = 0; d < mt.getRank(); ++d) {
      compilerinfra::AccessAxis ax;
      ax.container = view;
      ax.containerDim = static_cast<unsigned>(d);
      ax.staticExtent = resolveCpuViewExtent(op, view, d);
      if (d == sliceAxis) {
        ax.entireAxisFromZero = true;
        int64_t width =
            !vt.isDynamicDim(0) ? vt.getDimSize(0) : ax.staticExtent.value_or(0);
        if (width <= 0)
          return failure();
        ax.tileSize = width;
        ax.step = 1;
      } else {
        Value c = coords[coordPos++];
        ax.coord = c;
        ax.tileSize = (innerArg && c == innerArg) ? ownerBatch : 1;
        ax.step = 1;
      }
      axes.push_back(ax);
    }
    return success();
  }
};

template <typename ConcreteOp, unsigned ViewOperand, unsigned CoordStart,
          unsigned TrailingOperands = 0>
struct CpuScalarElementAccessModel
    : public compilerinfra::TiledAccessOpInterface::ExternalModel<
          CpuScalarElementAccessModel<ConcreteOp, ViewOperand, CoordStart,
                                      TrailingOperands>,
          ConcreteOp> {
  LogicalResult
  getAccessedAxes(Operation *op,
                  SmallVectorImpl<compilerinfra::AccessAxis> &axes) const {
    if (op->getNumOperands() < CoordStart + TrailingOperands ||
        ViewOperand >= op->getNumOperands())
      return failure();
    Value view = op->getOperand(ViewOperand);
    auto mt = dyn_cast<MemRefType>(view.getType());
    auto coords = op->getOperands()
                      .drop_front(CoordStart)
                      .drop_back(TrailingOperands);
    if (!mt || coords.size() != static_cast<size_t>(mt.getRank()))
      return failure();
    auto [innerArg, ownerBatch] = findEnclosingInnerOwnerBatch(op);
    for (auto [d, c] : llvm::enumerate(coords)) {
      if (!isa<IndexType, IntegerType>(c.getType()))
        return failure();
      compilerinfra::AccessAxis ax;
      ax.container = view;
      ax.containerDim = static_cast<unsigned>(d);
      ax.staticExtent =
          resolveCpuViewExtent(op, view, static_cast<int64_t>(d));
      ax.coord = c;
      ax.tileSize = (innerArg && c == innerArg) ? ownerBatch : 1;
      ax.step = 1;
      axes.push_back(ax);
    }
    return success();
  }
};
} // namespace

void intentdsl::cpu::CpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  auto *ctx = getContext();
  OrderedOwnerSpace::attachInterface<CpuOwnerSpaceIndexedRegionModel>(*ctx);
  OrderedSliceRead::attachInterface<
      CpuSliceTiledAccessModel<OrderedSliceRead, false>>(*ctx);
  OrderedSliceWrite::attachInterface<
      CpuSliceTiledAccessModel<OrderedSliceWrite, true>>(*ctx);
  read::attachInterface<CpuScalarElementAccessModel<read, 0, 1>>(*ctx);
  OwnerReadOp::attachInterface<CpuScalarElementAccessModel<OwnerReadOp, 0, 1>>(
      *ctx);
  OwnerWriteOp::attachInterface<
      CpuScalarElementAccessModel<OwnerWriteOp, 0, 1, 1>>(*ctx);
  Dialect_MemberLoad::attachInterface<
      CpuScalarElementAccessModel<Dialect_MemberLoad, 0, 1>>(*ctx);
  BufferLoad::attachInterface<CpuScalarElementAccessModel<BufferLoad, 0, 1>>(
      *ctx);
  BufferStore::attachInterface<CpuScalarElementAccessModel<BufferStore, 1, 2>>(
      *ctx);
}
