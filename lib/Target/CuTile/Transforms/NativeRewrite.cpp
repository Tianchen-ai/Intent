#include "NativeRewrite.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cassert>

using namespace mlir;

namespace intent::cutile {

Type withElementType(Type type, Type elementType) {
  auto fragment = dyn_cast<gpu::FragmentType>(type);
  if (!fragment)
    return elementType;
  return gpu::FragmentType::get(
      fragment.getContext(), elementType, fragment.getShape(),
      fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
}

Value repeatTileAxis(OpBuilder &builder, Location location, Value value,
                     unsigned axis, gpu::PhysicalExprAttr requested) {
  auto source = cast<gpu::FragmentType>(value.getType());
  auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
  if (extent == requested)
    return value;
  auto context = builder.getContext();
  auto expression = [&](gpu::PhysicalExprKind kind,
                         ArrayRef<Attribute> operands) {
    return gpu::PhysicalExprAttr::get(context, kind, 0,
                                     builder.getStringAttr(""),
                                     builder.getArrayAttr(operands));
  };
  auto unit = gpu::PhysicalExprAttr::get(
      context, gpu::PhysicalExprKind::Constant, 1,
      builder.getStringAttr(""), builder.getArrayAttr({}));
  auto repeat = expression(gpu::PhysicalExprKind::FloorDiv,
      {expression(gpu::PhysicalExprKind::Maximum, {extent, requested}), extent});
  auto expanded = expression(gpu::PhysicalExprKind::Multiply, {repeat, extent});
  SmallVector<Attribute> splitShape, splitMaps, splitGroups, mergeGroups;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  uint64_t repeatSource = gpu::nextPhysicalAxisIdentities(kernel).first;
  for (unsigned original = 0; original < source.getShape().size(); ++original) {
    SmallVector<int64_t> splitAxes;
    if (original == axis) {
      splitAxes.push_back(splitShape.size());
      splitMaps.push_back(gpu::AxisMapAttr::get(
          context, repeatSource, 0, 0, splitShape.size(), true));
      splitShape.push_back(unit);
    }
    splitAxes.push_back(splitShape.size());
    auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[original]);
    splitMaps.push_back(gpu::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), splitShape.size(), mapping.getDerived()));
    splitShape.push_back(source.getShape()[original]);
    splitGroups.push_back(gpu::ReshapeGroupAttr::get(
        context, builder.getDenseI64ArrayAttr({original}),
        builder.getDenseI64ArrayAttr(splitAxes)));
    mergeGroups.push_back(gpu::ReshapeGroupAttr::get(
        context, builder.getDenseI64ArrayAttr(splitAxes),
        builder.getDenseI64ArrayAttr({original})));
  }
  auto fragment = [&](ArrayRef<Attribute> shape, ArrayRef<Attribute> mappings) {
    return gpu::FragmentType::get(context, source.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
        source.getValidity(), source.getOwner());
  };
  Value split = builder.create<gpu::ReshapeOp>(
      location, fragment(splitShape, splitMaps), value,
      builder.getArrayAttr(splitGroups));
  splitShape[axis] = repeat;
  Value repeated = builder.create<gpu::BroadcastOp>(
      location, fragment(splitShape, splitMaps), split);
  SmallVector<Attribute> resultShape(source.getShape().getValue());
  resultShape[axis] = expanded;
  return builder.create<gpu::ReshapeOp>(
      location, fragment(resultShape, source.getAxisMaps().getValue()), repeated,
      builder.getArrayAttr(mergeGroups));
}

void NativeFormRewriter::replace(Operation *operation, ValueRange values) {
  assert(operation->getNumResults() == values.size());
  replacements.push_back({operation, llvm::to_vector(values)});
}
void NativeFormRewriter::erase(Operation *operation) {
  assert(operation->use_empty());
  replacements.push_back({operation, {}});
}
void NativeFormRewriter::commit() {
  for (Replacement &replacement : replacements)
    if (!replacement.values.empty())
      replacement.operation->getResults().replaceAllUsesWith(replacement.values);
  // Destroy inner operations before their owners even if a future native form
  // consumes nested input operations in a different construction order.
  auto depth = [](Operation *operation) {
    unsigned result = 0;
    while ((operation = operation->getParentOp())) ++result;
    return result;
  };
  llvm::stable_sort(replacements, [&](const Replacement &a, const Replacement &b) {
    return depth(a.operation) > depth(b.operation);
  });
  for (Replacement &replacement : replacements)
    replacement.operation->erase();
  replacements.clear();
}

} // namespace intent::cutile
