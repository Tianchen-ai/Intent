#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Target/CuTile/Transforms/Access/ArrayViews.h"
#include "TileIndices.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::cutile {
namespace {

DenseI64ArrayAttr collapseGroups(TileLoadOp load) {
  auto view = load.getResource().getType();
  if (view.getAccess() != 0 || view.getRank() <= 2 ||
      !isa<BlockArgument>(load.getResource()))
    return {};
  SmallVector<int64_t> ends;
  unsigned collapsed = 0;
  for (unsigned axis = 1; axis < view.getRank(); ++axis) {
    auto extent =
        dyn_cast<gpu::PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
    bool fullInner = extent &&
        extent.getKind() ==
            gpu::PhysicalExprKind::Constant &&
        extent.getValue() > 1 &&
        load.getResult().getType().getShape()[axis] == extent &&
        matchPattern(load.getTileIndices()[axis], m_Zero());
    if (fullInner && view.getRank() - collapsed > 2)
      ++collapsed;
    else
      ends.push_back(axis);
  }
  ends.push_back(view.getRank());
  return collapsed ? DenseI64ArrayAttr::get(load.getContext(), ends)
                   : DenseI64ArrayAttr();
}

gpu::FragmentType collapsedTile(gpu::FragmentType source,
                                ArrayRef<int64_t> ends) {
  MLIRContext *context = source.getContext();
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  unsigned begin = 0;
  for (int64_t end : ends) {
    Attribute extent = source.getShape()[begin];
    for (unsigned axis = begin + 1; axis < static_cast<unsigned>(end); ++axis)
      extent = gpu::PhysicalExprAttr::get(
          context, gpu::PhysicalExprKind::Multiply, 0,
          StringAttr::get(context),
          ArrayAttr::get(context, {extent, source.getShape()[axis]}));
    auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[begin]);
    mappings.push_back(gpu::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        end == begin + 1 ? mapping.getDimensionId() : 0, shape.size(),
        end != begin + 1 || mapping.getDerived()));
    shape.push_back(extent);
    begin = end;
  }
  return gpu::FragmentType::get(
      context, source.getElementType(), ArrayAttr::get(context, shape),
      ArrayAttr::get(context, mappings), source.getValidity(), source.getOwner());
}

} // namespace

LogicalResult collapseArrayViews(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<TileLoadOp> loads;
  kernel.walk([&](TileLoadOp load) { loads.push_back(load); });
  SmallVector<ArrayViewOp> views;
  for (TileLoadOp load : loads) {
    auto groups = collapseGroups(load);
    if (!groups)
      continue;
    auto tile = collapsedTile(load.getResult().getType(), groups.asArrayRef());
    auto reassociation =
        gpu::inferReshapeReassociation(tile, load.getResult().getType());
    if (failed(reassociation))
      return load.emitOpError("cannot restore a collapsed native array tile");
    ArrayViewOp array;
    for (ArrayViewOp candidate : views)
      if (candidate.getBase() == load.getResource() &&
          candidate.getGroupEndsAttr() == groups)
        array = candidate;
    if (!array) {
      OpBuilder declarations(&kernel.getBody().front(),
                             kernel.getBody().front().begin());
      array = declarations.create<ArrayViewOp>(
          load.getLoc(), load.getResource(), groups);
      views.push_back(array);
    }
    OpBuilder builder(load);
    auto choice = builder.create<scf::IfOp>(
        load.getLoc(), TypeRange{load.getResult().getType()}, array.getEligible(),
        /*withElseRegion=*/true);
    OpBuilder folded = choice.getThenBodyBuilder();
    SmallVector<Value> indices;
    unsigned begin = 0;
    for (int64_t end : groups.asArrayRef()) {
      indices.push_back(load.getTileIndices()[begin]);
      begin = end;
    }
    Value fullTiles = load.getFullTiles()
        ? materializeFullTileCondition(folded, load.getLoc(), array.getResult(), tile)
        : Value();
    auto native = folded.create<TileLoadOp>(
        load.getLoc(), tile, array.getResult(), load.getAllowTma(), indices,
        load.getLatencyPolicy(), fullTiles);
    auto restored = folded.create<gpu::ReshapeOp>(
        load.getLoc(), load.getResult().getType(), native.getResult(),
        *reassociation);
    folded.create<scf::YieldOp>(load.getLoc(), restored.getResult());
    OpBuilder original = choice.getElseBodyBuilder();
    auto unchanged = cast<TileLoadOp>(original.clone(*load));
    original.create<scf::YieldOp>(load.getLoc(), unchanged.getResult());
    load.getResult().replaceAllUsesWith(choice.getResult(0));
    load.erase();
  }
  return success();
}

} // namespace intent::cutile
