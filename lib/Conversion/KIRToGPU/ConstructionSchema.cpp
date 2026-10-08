#include "ConstructionSchema.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::detail {
namespace {

gpu::AxisMapAttr atAxis(gpu::AxisMapAttr source, unsigned axis,
                       std::optional<int64_t> dimension = std::nullopt) {
  return gpu::AxisMapAttr::get(source.getContext(), source.getSourceId(),
      source.getSourceAxis(), dimension.value_or(source.getDimensionId()), axis,
      source.getDerived());
}

} // namespace

LogicalResult alignPointwiseOperands(
    OpBuilder &builder, Operation *canonical, CanonicalKernelAnalysis &analysis,
    func::FuncOp kernel, gpu::FragmentType logicalSeed,
    MutableArrayRef<Value> values, ArrayRef<unsigned> operandNumbers) {
  if (values.size() != operandNumbers.size() || values.empty())
    return failure();
  if (llvm::none_of(values, [](Value value) {
        return isa<gpu::FragmentType>(value.getType());
      }))
    return success();

  SmallVector<TensorOperandProjection, 3> relations;
  if (isa<RankedTensorType>(canonical->getResult(0).getType())) {
    auto queried = analysis.operandProjections(cast<OpResult>(canonical->getResult(0)));
    if (failed(queried)) return failure();
    relations = *queried;
  }
  unsigned logicalRank = logicalSeed ? logicalSeed.getShape().size() : 0;
  SmallVector<Attribute> shape, mappings;
  SmallVector<SmallVector<unsigned>> projections(values.size());
  uint64_t owner = 0;
  uint64_t validity = 0;
  // Execution prefixes belong to the current workset. Only their actual
  // physical coordinate identities can merge; logical shape equality cannot.
  for (auto [index, value] : llvm::enumerate(values)) {
    auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
    if (!fragment) continue;
    if (owner && (owner != fragment.getOwner() || validity != fragment.getValidity()))
      return failure();
    owner = fragment.getOwner();
    validity = fragment.getValidity();
    auto tensor = dyn_cast<RankedTensorType>(canonical->getOperand(operandNumbers[index]).getType());
    unsigned rank = tensor ? tensor.getRank() : 0;
    if (rank > fragment.getShape().size()) return failure();
    unsigned prefix = fragment.getShape().size() - rank;
    for (unsigned axis = 0; axis < prefix; ++axis) {
      auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
      auto found = llvm::find_if(mappings, [&](Attribute other) {
        auto candidate = cast<gpu::AxisMapAttr>(other);
        return gpu::sourceAxisIdentity(mapping) == gpu::sourceAxisIdentity(candidate) &&
               mapping.getDimensionId() == candidate.getDimensionId();
      });
      unsigned target = found - mappings.begin();
      if (found == mappings.end()) {
        shape.push_back(fragment.getShape()[axis]);
        mappings.push_back(atAxis(mapping, target));
      } else if (shape[target] != fragment.getShape()[axis]) {
        return failure();
      }
      projections[index].push_back(target);
    }
  }
  unsigned prefixRank = shape.size();
  if (logicalSeed) {
    shape.append(logicalSeed.getShape().begin(), logicalSeed.getShape().end());
    for (Attribute attribute : logicalSeed.getAxisMaps())
      mappings.push_back(atAxis(cast<gpu::AxisMapAttr>(attribute), mappings.size()));
  }
  SmallVector<bool> chosen(logicalRank, false);
  for (auto [index, value] : llvm::enumerate(values)) {
    auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
    auto tensor = dyn_cast<RankedTensorType>(canonical->getOperand(operandNumbers[index]).getType());
    if (!fragment || !tensor) continue;
    auto relation = llvm::find_if(relations, [&](const auto &relation) {
      return relation.operandNumber == operandNumbers[index];
    });
    if (relation == relations.end() || relation->resultAxes.size() != tensor.getRank())
      return failure();
    unsigned sourcePrefix = fragment.getShape().size() - tensor.getRank();
    for (auto [axis, mapped] : llvm::enumerate(relation->resultAxes)) {
      if (!mapped || *mapped >= logicalRank) return failure();
      unsigned target = prefixRank + *mapped;
      projections[index].push_back(target);
      // Unit broadcast axes do not donate an execution extent or coordinate.
      if (tensor.getDimSize(axis) == 1) continue;
      auto candidate = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[sourcePrefix + axis]);
      auto selected = cast<gpu::AxisMapAttr>(mappings[target]);
      if (!chosen[*mapped] || (selected.getDerived() && !candidate.getDerived())) {
        mappings[target] = atAxis(candidate, target, selected.getDimensionId());
        shape[target] = fragment.getShape()[sourcePrefix + axis];
        chosen[*mapped] = true;
      }
    }
  }
  if (shape.empty()) return failure();
  Type element = logicalSeed ? logicalSeed.getElementType() : values.front().getType();
  if (auto fragment = dyn_cast<gpu::FragmentType>(element)) element = fragment.getElementType();
  auto target = gpu::FragmentType::get(builder.getContext(), element,
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings), validity, owner);

  // Rebind only axes for which the canonical operation proved correspondence.
  // Keep source extents until the common solver has selected their authority.
  for (auto [index, value] : llvm::enumerate(values)) {
    auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
    if (!fragment) continue;
    if (projections[index].size() != fragment.getShape().size()) return failure();
    SmallVector<Attribute> reboundMaps, groups;
    for (auto [axis, mapped] : llvm::enumerate(projections[index])) {
      reboundMaps.push_back(atAxis(cast<gpu::AxisMapAttr>(mappings[mapped]), axis));
      auto singleton = builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)});
      groups.push_back(gpu::ReshapeGroupAttr::get(builder.getContext(), singleton, singleton));
    }
    auto rebound = gpu::FragmentType::get(builder.getContext(), fragment.getElementType(),
        fragment.getShape(), builder.getArrayAttr(reboundMaps), validity, owner);
    if (rebound != fragment)
      values[index] = builder.create<gpu::ReshapeOp>(canonical->getLoc(), rebound,
          value, builder.getArrayAttr(groups));
  }
  auto schema = gpu::queryValueSchema(kernel, target, values);
  if (failed(schema)) return failure();
  for (Value &value : values) {
    Type element = value.getType();
    if (auto fragment = dyn_cast<gpu::FragmentType>(element)) element = fragment.getElementType();
    auto destination = gpu::FragmentType::get(builder.getContext(), element,
        schema->getShape(), schema->getAxisMaps(), schema->getValidity(), schema->getOwner());
    auto projected = gpu::projectPhysicalValueToSchema(builder, canonical->getLoc(), value, destination);
    if (failed(projected)) return failure();
    value = *projected;
  }
  return success();
}

} // namespace intent::detail
