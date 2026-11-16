#include "AccessForms.h"
#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/ProgramInterface.h"
#include "Intent/Dialect/Block/Analysis/UniformValues.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::triton::detail {
constexpr llvm::StringLiteral tensorDescriptorChoice =
    "USE_TENSOR_DESCRIPTOR";
constexpr llvm::StringLiteral tensorDescriptorEligibility =
    "TENSOR_DESCRIPTOR_ELIGIBLE";
struct DescriptorOffset {
  SmallVector<std::pair<Value, bool>> terms;

  void add(Value value, bool subtract = false) {
    if (auto binary = value.getDefiningOp<block::BinaryOp>();
        binary && (binary.getOperatorKind() == BinaryOperator::Add ||
                   binary.getOperatorKind() == BinaryOperator::Subtract)) {
      add(binary.getLhs(), subtract);
      add(binary.getRhs(), subtract !=
          (binary.getOperatorKind() == BinaryOperator::Subtract));
      return;
    }
    auto opposite = llvm::find_if(terms, [&](auto term) {
      return term.second != subtract &&
             block::samePhysicalScalarExpression(term.first, value);
    });
    if (opposite != terms.end()) terms.erase(opposite);
    else terms.emplace_back(value, subtract);
  }
};

struct DescriptorAccessPlan {
  SmallVector<DescriptorOffset> offsets;
  SmallVector<int64_t> blockAxes;
};

bool isUnitStep(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  return constant && constant.value() == 1;
}

bool isZeroValue(Value value) {
  Attribute constant =
      UniformValueAnalysis(block::describeUniformValue).evaluate(value);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant))
    return integer.getValue().isZero();
  // Native zero padding produces positive floating zero.
  if (auto floating = dyn_cast<FloatAttr>(constant))
    return floating.getValue().isZero() && !floating.getValue().isNegative();
  return false;
}

std::optional<DescriptorAccessPlan>
planDescriptorAccess(block::AccessOpInterface access,
                const block::PhysicalAccessBoundaryFact &boundaryFact) {
  Value resource = access.getAccessResource();
  auto coordinates = access.getAccessCoordinates();
  auto sourceAxes = access.getAccessSourceAxes();
  Type valueType = access.getAccessValueType();
  Value valid = access.getAccessValidity();
  Value fill = access.getAccessFill();
  auto view = dyn_cast<block::ViewType>(resource.getType());
  auto fragment = dyn_cast<block::FragmentType>(valueType);
  if (!view || !fragment || fragment.getShape().empty() || !isa<BlockArgument>(resource) ||
      coordinates.size() != view.getRank() ||
      sourceAxes.size() != view.getRank())
    return std::nullopt;

  DescriptorAccessPlan plan;
  plan.offsets.resize(view.getRank());
  plan.blockAxes.assign(fragment.getShape().size(), -1);
  llvm::SmallBitVector seenViewAxes(view.getRank());
  llvm::SmallBitVector seenFragmentAxes(fragment.getShape().size());
  for (auto [coordinateIndex, coordinate] : llvm::enumerate(coordinates)) {
    int64_t viewAxis = sourceAxes[coordinateIndex];
    if (viewAxis < 0 || viewAxis >= static_cast<int64_t>(view.getRank()) ||
        seenViewAxes.test(viewAxis))
      return std::nullopt;
    seenViewAxes.set(viewAxis);
    if (Value scalar = block::uniformScalarSource(coordinate);
        scalar && scalar.getType().isIndex()) {
      plan.offsets[viewAxis].add(scalar);
      continue;
    }
    auto coordinateType = dyn_cast<block::FragmentType>(coordinate.getType());
    if (!coordinateType)
      return std::nullopt;
    auto projection = block::queryAccessCoordinateProjection(access, coordinateIndex);
    if (!projection.isExact())
      return std::nullopt;
    // Broadcasting a Cartesian range does not make the access indirect.
    // Compose the actual projections rather than matching equal extents or
    // dropping a reshape that might change the coordinate's varying axis.
    while (auto broadcast = coordinate.getDefiningOp<block::BroadcastOp>()) {
      auto source = dyn_cast<block::FragmentType>(broadcast.getValue().getType());
      if (!source)
        return std::nullopt;
      auto relations = block::queryFragmentOperandRelations(broadcast.getOperation());
      if (failed(relations) || !relations->front().hasCompatibleExtents())
        return std::nullopt;
      for (std::optional<unsigned> &axis : projection.targetToSource) {
        if (!axis)
          continue;
        const auto *group = relations->front().groupForResultAxis(*axis);
        if (!group || group->sourceAxes.size() > 1)
          return std::nullopt;
        axis = group->sourceAxes.empty()
                   ? std::nullopt
                   : std::optional<unsigned>(group->sourceAxes.front());
      }
      coordinate = broadcast.getValue();
      coordinateType = source;
    }
    // A translated Cartesian range is still a contiguous block. Keep the
    // origin as a read-only additive expression until descriptor materialization;
    // cancellation also exposes loop IVs hidden by repeated-domain rebasing.
    auto &offset = plan.offsets[viewAxis];
    bool translated = false;
    while (auto binary = coordinate.getDefiningOp<block::BinaryOp>()) {
      auto kind = binary.getOperatorKind();
      if (kind != BinaryOperator::Add && kind != BinaryOperator::Subtract)
        break;
      Value scalar = block::uniformScalarSource(binary.getRhs());
      Value varying = binary.getLhs();
      if (!scalar && kind == BinaryOperator::Add) {
        scalar = block::uniformScalarSource(binary.getLhs());
        varying = binary.getRhs();
      }
      if (!scalar || !scalar.getType().isIndex() ||
          varying.getType() != coordinate.getType())
        break;
      offset.add(scalar, kind == BinaryOperator::Subtract);
      coordinate = varying;
      translated = true;
    }
    auto range = coordinate.getDefiningOp<block::MakeRangeOp>();
    if (coordinateType.getShape().size() != 1 ||
        !coordinateType.getElementType().isIndex() || !range ||
        !isUnitStep(range.getStep()))
      return std::nullopt;
    std::optional<unsigned> selected;
    for (auto [fragmentAxis, sourceAxis] :
         llvm::enumerate(projection.targetToSource)) {
      if (sourceAxis) {
        if (selected)
          return std::nullopt;
        selected = fragmentAxis;
      }
    }
    if (!selected || seenFragmentAxes.test(*selected) ||
        coordinateType.getShape()[0] != fragment.getShape()[*selected])
      return std::nullopt;
    auto extent = cast<block::PhysicalExprAttr>(fragment.getShape()[*selected]);
    if (!isTritonFragmentExtent(extent))
      return std::nullopt;
    seenFragmentAxes.set(*selected);
    plan.blockAxes[*selected] = viewAxis;
    offset.add(range.getStart());
    // Descriptor bounds operate on a non-wrapping window. Fixed-width index
    // addition is valid under rebasing, but a translated window must not cross
    // the signed boundary. An aligned power-of-two window proves this for every
    // represented origin, without assuming runtime tensor dimensions.
    if (translated &&
        (!block::IndexRelations().powerOfTwo(range.getExtent()) ||
         !llvm::all_of(offset.terms, [&](auto term) {
           return block::IndexRelations().multipleOf(term.first, range.getExtent());
         })))
      return std::nullopt;
  }
  if (seenViewAxes.count() != view.getRank() ||
      seenFragmentAxes.count() != fragment.getShape().size())
    return std::nullopt;
  if (!boundaryFact.isExact())
    return std::nullopt;
  if (valid) {
    if (fill && !isZeroValue(fill))
      return std::nullopt;
  } else if (fill) {
    return std::nullopt;
  }
  return plan;
}

struct DescriptorAccessCandidates {
  SmallVector<std::pair<block::LoadOp, DescriptorAccessPlan>, 4> loads;
  SmallVector<std::pair<block::StoreOp, DescriptorAccessPlan>, 4> stores;
};

LogicalResult orientPointerLoads(func::FuncOp kernel) {
  llvm::DenseMap<Operation *, SmallVector<block::ContractOp>> simtRhsConsumers;
  kernel.walk([&](block::ContractOp contract) {
    if (!block::uniformElementType(contract.getLhs().getType()).isF32() ||
        !block::uniformElementType(contract.getRhs().getType()).isF32())
      return;
    SmallVector<Value> pending{contract.getRhs()};
    llvm::DenseSet<Operation *> visited;
    while (!pending.empty()) {
      Operation *producer = pending.pop_back_val().getDefiningOp();
      if (!producer || !visited.insert(producer).second)
        continue;
      if (auto load = dyn_cast<block::LoadOp>(producer)) {
        simtRhsConsumers[load].push_back(contract);
        continue;
      }
      if (isa<block::BroadcastOp, block::ReshapeOp, block::TransposeOp, block::CastOp,
              block::UnaryOp, block::BinaryOp, block::SelectOp>(producer))
        llvm::append_range(pending, producer->getOperands());
    }
  });
  SmallVector<block::LoadOp> loads;
  kernel.walk([&](block::LoadOp load) {
    auto access = cast<block::AccessOpInterface>(load.getOperation());
    if (isa<block::ViewType>(access.getAccessResource().getType()) &&
        isa<block::FragmentType>(load.getResult().getType()))
      loads.push_back(load);
  });
  for (block::LoadOp load : loads) {
    auto access = cast<block::AccessOpInterface>(load.getOperation());
    auto result = cast<block::FragmentType>(load.getResult().getType());
    unsigned rank = result.getShape().size();
    if (rank < 2)
      continue;
    block::PhysicalProgramAnalysis analysis(kernel);
    SmallVector<int64_t> viewAxes(rank, -1);
    Value innermostCoordinate;
    int64_t innermostViewAxis = -1;
    bool exact = true;
    for (auto [position, coordinate] : llvm::enumerate(access.getAccessCoordinates())) {
      auto type = dyn_cast<block::FragmentType>(coordinate.getType());
      if (!type)
        continue;
      auto projection = block::queryAccessCoordinateProjection(access, position);
      if (!projection.isExact()) {
        exact = false;
        break;
      }
      unsigned varying = 0;
      for (auto [axis, source] : llvm::enumerate(projection.targetToSource)) {
        if (!source)
          continue;
        auto extent = cast<block::PhysicalExprAttr>(type.getShape()[*source]);
        if (extent.getKind() == block::PhysicalExprKind::Constant &&
            extent.getValue() == 1)
          continue;
        auto ranges = analysis.axisRanges(coordinate, *source);
        if (!ranges.isExact() || !ranges.blockers.empty() ||
            !ranges.accesses.empty()) {
          exact = false;
          break;
        }
        if (ranges.roots.empty())
          continue;
        if (++varying != 1) {
          exact = false;
          break;
        }
        // A broadcast axis does not vary. Reshape-derived coordinates may
        // share one varying fragment axis across several resource axes.
        viewAxes[axis] = std::max(viewAxes[axis], access.getAccessSourceAxes()[position]);
        if (viewAxes[axis] > innermostViewAxis) {
          innermostViewAxis = viewAxes[axis];
          innermostCoordinate = coordinate;
        }
      }
      if (!exact)
        break;
    }
    // Leave a unit-step innermost view coordinate to native coalescing. A
    // unit-step coordinate on an outer view axis can still be strided.
    auto view = cast<block::ViewType>(access.getAccessResource().getType());
    if (innermostCoordinate && innermostViewAxis + 1 == view.getRank()) {
      auto range = block::stripAdditiveProjection(innermostCoordinate)
                       .getDefiningOp<block::MakeRangeOp>();
      if (range && block::isUnitStepRange(range))
        continue;
    }
    auto sameSchema = [&](Type element) {
      return block::FragmentType::get(kernel.getContext(), element,
          result.getShape(), result.getAxisMaps(), result.getValidity(),
          result.getOwner());
    };
    for (Value operand : {access.getAccessValidity(), access.getAccessFill()})
      if (operand)
        if (auto fragment = dyn_cast<block::FragmentType>(operand.getType()))
          exact &= block::queryBroadcastProjection(
              fragment, sameSchema(fragment.getElementType())).isExact();
    if (!exact)
      continue;
    SmallVector<int64_t> permutation;
    for (unsigned axis = 0; axis < rank; ++axis)
      permutation.push_back(axis);
    llvm::stable_sort(permutation, [&](int64_t lhs, int64_t rhs) {
      return viewAxes[lhs] > viewAxes[rhs];
    });
    if (llvm::all_of(llvm::enumerate(permutation), [](auto entry) {
          return static_cast<int64_t>(entry.index()) == entry.value();
        }))
      continue;
    // IEEE f32 dot uses SIMT FMA. Its RHS shared tile has no native swizzle;
    // making K the leading load axis can serialize the free-axis reads.
    auto leadingRanges = analysis.axisRanges(load.getResult(), permutation.front());
    bool reductionLeading = llvm::any_of(simtRhsConsumers.lookup(load),
        [&](block::ContractOp contract) {
          if (!leadingRanges.isExact() || leadingRanges.roots.empty())
            return false;
          auto axes = analysis.rangeAxes(contract.getRhs(), leadingRanges.roots);
          return axes.isExact() &&
                 llvm::any_of(axes.fragmentAxes, [&](unsigned axis) {
                   return llvm::is_contained(contract.getRhsReductionAxes(), axis);
                 });
        });
    if (reductionLeading)
      continue;
    OpBuilder builder(load);
    auto permutedType = [&](block::FragmentType type, ArrayRef<int64_t> order) {
      SmallVector<Attribute> shape, mappings;
      for (auto [position, axis] : llvm::enumerate(order)) {
        shape.push_back(type.getShape()[axis]);
        auto mapping = cast<block::AxisMapAttr>(type.getAxisMaps()[axis]);
        mappings.push_back(block::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), position, mapping.getDerived()));
      }
      return block::FragmentType::get(kernel.getContext(),
          type.getElementType(), builder.getArrayAttr(shape),
          builder.getArrayAttr(mappings), type.getValidity(), type.getOwner());
    };
    auto transpose = [&](Value value, ArrayRef<int64_t> order) -> Value {
      auto target = permutedType(cast<block::FragmentType>(value.getType()), order);
      return builder.create<block::TransposeOp>(load.getLoc(), target, value, order);
    };
    auto orientValue = [&](Value operand) -> Value {
      if (!operand)
        return {};
      auto type = dyn_cast<block::FragmentType>(operand.getType());
      if (!type)
        return operand;
      auto target = sameSchema(type.getElementType());
      Value expanded = operand;
      if (type != target)
        expanded = builder.create<block::BroadcastOp>(load.getLoc(), target, operand);
      return transpose(expanded, permutation);
    };
    SmallVector<Value> coordinates;
    for (unsigned position = 0; position < access.getAccessCoordinates().size(); ++position) {
      auto projected = block::materializeAccessCoordinate(builder, access, position);
      if (failed(projected))
        return load.emitOpError("cannot materialize its proven coordinate projection");
      coordinates.push_back(isa<block::FragmentType>((*projected).getType())
          ? transpose(*projected, permutation) : *projected);
    }
    // Coordinate occurrences can share one SSA value while varying over
    // different Cartesian axes. Bind every slot, independently of fill/validity.
    Value validity = orientValue(access.getAccessValidity());
    Value fill = orientValue(access.getAccessFill());
    auto oriented = cast<block::LoadOp>(builder.clone(*load));
    auto orientedAccess = cast<block::AccessOpInterface>(oriented.getOperation());
    orientedAccess.getAccessCoordinatesMutable().assign(coordinates);
    if (validity)
      orientedAccess.getAccessValidityMutable().assign(validity);
    if (fill)
      orientedAccess.getAccessFillMutable()->assign(fill);
    oriented.getResult().setType(permutedType(result, permutation));
    SmallVector<int64_t> inverse(rank);
    for (auto [position, axis] : llvm::enumerate(permutation))
      inverse[axis] = position;
    Value restored = transpose(oriented.getResult(), inverse);
    load.getResult().replaceAllUsesWith(restored);
    load.erase();
  }
  return success();
}

bool descriptorStrideAvailable(func::FuncOp kernel, block::ViewType view,
                               unsigned axis) {
  auto stride = cast<block::PhysicalExprAttr>(view.getLayout().getStrides()[axis]);
  if (stride.getKind() == block::PhysicalExprKind::Constant)
    return stride.getValue() > 0;
  return bool(block::resolveArgument(kernel, stride.getArgumentReference()));
}



bool descriptorAccessEligible(func::FuncOp kernel, Value viewValue,
                              ArrayRef<DescriptorOffset> offsets,
                              ArrayRef<int64_t> blockAxes,
                              block::FragmentType fragment) {
  auto view = cast<block::ViewType>(viewValue.getType());
  unsigned blockRank = fragment.getShape().size();
  if (view.getRank() < 1 || view.getRank() > 5 || blockRank == 0 ||
      !llvm::is_contained(blockAxes, static_cast<int64_t>(view.getRank()) - 1))
    return false;
  std::optional<int64_t> elementBytes =
      descriptorElementBytes(view.getElementType());
  auto layout = view.getLayout();
  if (!elementBytes || 16 % *elementBytes != 0)
    return false;
  auto strides = layout.getStrides();
  for (unsigned axis = 0; axis < view.getRank(); ++axis)
    if (!descriptorStrideAvailable(kernel, view, axis))
      return false;
  auto last = cast<block::PhysicalExprAttr>(strides[strides.size() - 1]);
  if (last.getKind() == block::PhysicalExprKind::Constant && last.getValue() != 1)
    return false;
  for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
    auto stride = cast<block::PhysicalExprAttr>(strides[axis]);
    if (stride.getKind() == block::PhysicalExprKind::Constant &&
        stride.getValue() % (16 / *elementBytes) != 0)
      return false;
  }
  // TMA requires aligned coordinates, not constant coordinates. Tile starts
  // and reduction-loop induction values preserve power-of-two divisibility,
  // including under fixed-width index arithmetic.
  // Descriptor configs already require a power-of-two contiguous block of
  // at least 16 bytes. Its extent parameter is aligned in this branch even
  // when the shared parameter domain also includes smaller pointer tiles.
  unsigned contiguousAxis = std::distance(
      blockAxes.begin(), llvm::find(blockAxes, view.getRank() - 1));
  auto extent =
      cast<block::PhysicalExprAttr>(fragment.getShape()[contiguousAxis]);
  llvm::SmallDenseSet<StringAttr, 4> alignedBlockParameters;
  if (extent.getKind() ==
      block::PhysicalExprKind::Parameter)
    alignedBlockParameters.insert(extent.getParameterReference().getName());
  else {
    // In the descriptor branch min(P, capacity) is at least the required
    // alignment. Each operand is therefore at least that large; when every
    // leaf has a power-of-two domain, each leaf is aligned as well. Keep this
    // conditional fact local to the existing descriptor eligibility contract.
    std::function<bool(block::PhysicalExprAttr)> collectAlignedLeaves =
        [&](block::PhysicalExprAttr expression) {
      if (expression.getKind() == block::PhysicalExprKind::Constant)
        return expression.getValue() > 0 &&
               llvm::isPowerOf2_64(expression.getValue());
      if (expression.getKind() == block::PhysicalExprKind::Parameter) {
        auto parameter = block::lookupParameter(
            kernel, expression.getParameterReference());
        if (!parameter ||
            parameter.getRole() == block::ParameterRole::ResidentWorkers ||
            parameter.getCandidates().empty() ||
            !llvm::all_of(parameter.getCandidates().asArrayRef(),
                          [](int64_t value) {
                            return value > 0 && llvm::isPowerOf2_64(value);
                          }))
          return false;
        alignedBlockParameters.insert(parameter.getName());
        return true;
      }
      return expression.getKind() == block::PhysicalExprKind::Minimum &&
             expression.getOperands().size() == 2 &&
             llvm::all_of(expression.getOperands(), [&](Attribute operand) {
               return collectAlignedLeaves(cast<block::PhysicalExprAttr>(operand));
             });
    };
    if (!collectAlignedLeaves(extent)) alignedBlockParameters.clear();
  }
  block::IndexRelations relations;
  return llvm::all_of(offsets.back().terms, [&](auto term) {
    return relations.multipleOf(term.first, 16 / *elementBytes,
        [&](block::ParameterAttr parameter) {
          return alignedBlockParameters.contains(parameter.getName());
        });
  });
}

void copyOrigin(Operation *source, Operation *target) {
  if (Attribute origin = source->getAttr(block::originAttr))
    target->setAttr(block::originAttr, origin);
}

FailureOr<Value> descriptorStrideValue(OpBuilder &builder, func::FuncOp kernel,
                                       Location location, block::ViewType view,
                                       unsigned axis) {
  auto stride = cast<block::PhysicalExprAttr>(view.getLayout().getStrides()[axis]);
  if (stride.getKind() == block::PhysicalExprKind::Constant)
    return Value(builder.create<arith::ConstantIndexOp>(location,
                                                        stride.getValue()));
  if (auto argument = block::resolveArgument(kernel, stride.getArgumentReference()))
    return Value(argument);
  return failure();
}

SmallVector<Value> materializeDescriptorOffsets(
    OpBuilder &builder, Location location, ArrayRef<DescriptorOffset> offsets) {
  // Descriptor shapes are at most INT32_MAX and blocks at most 2^20
  // elements. A start outside signed i32 therefore denotes an entirely
  // padded block; preserve that fact before converting to native offsets.
  Value minimum = builder.create<arith::ConstantIndexOp>(
      location, std::numeric_limits<int32_t>::min());
  Value maximum = builder.create<arith::ConstantIndexOp>(
      location, std::numeric_limits<int32_t>::max());
  SmallVector<Value> result;
  for (const DescriptorOffset &expression : offsets) {
    Value offset;
    for (auto [value, subtract] : expression.terms) {
      if (!offset && !subtract) {
        offset = value;
        continue;
      }
      if (!offset) offset = builder.create<arith::ConstantIndexOp>(location, 0);
      offset = builder.createOrFold<block::BinaryOp>(
          location, builder.getIndexType(), offset, value,
          subtract ? BinaryOperator::Subtract : BinaryOperator::Add);
    }
    if (!offset) offset = builder.create<arith::ConstantIndexOp>(location, 0);
    Value lower = builder.create<block::CompareOp>(
        location, builder.getI1Type(), offset, minimum, ComparePredicate::Ge);
    Value upper = builder.create<block::CompareOp>(
        location, builder.getI1Type(), offset, maximum, ComparePredicate::Le);
    Value inside = builder.create<block::BinaryOp>(
        location, builder.getI1Type(), lower, upper, BinaryOperator::LogicalAnd);
    Value native = builder.create<block::SelectOp>(
        location, builder.getIndexType(), inside, offset, minimum);
    result.push_back(builder.create<block::CastOp>(
        location, builder.getI32Type(), native));
  }
  return result;
}

bool feedsIndirectAccessCoordinates(Value value) {
  SmallVector<Value> pending{value};
  llvm::SmallDenseSet<Value, 32> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      if (auto access = dyn_cast<block::AccessOpInterface>(user);
          access && (access.getAccessKind() == block::AccessKind::Load ||
                     access.getAccessKind() == block::AccessKind::Store)) {
        if (llvm::is_contained(access.getAccessCoordinates(), current))
          return true;
        continue;
      }
      if (user->getNumRegions() == 0 && isMemoryEffectFree(user))
        pending.append(user->getResults().begin(), user->getResults().end());
    }
  }
  return false;
}

bool feedsMatrixOperand(Value value) {
  SmallVector<Value> pending{value};
  llvm::SmallDenseSet<Value, 32> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second) continue;
    for (Operation *user : current.getUsers()) {
      if (isa<block::ContractOp, block::ScaledContractOp>(user)) return true;
      if (isa<block::BroadcastOp, block::ReshapeOp, block::TransposeOp, block::CastOp>(user))
        llvm::append_range(pending, user->getResults());
    }
  }
  return false;
}

// A stable read phase. No analysis or boundary cache survives the returned
// candidate list: rewrites only consume the already-proved access operands.
DescriptorAccessCandidates readDescriptorAccesses(func::FuncOp kernel) {
  DescriptorAccessCandidates candidates;
  block::PhysicalProgramAnalysis analysis(kernel);
  kernel.walk([&](Operation *operation) {
    auto access = dyn_cast<block::AccessOpInterface>(operation);
    if (!access || (access.getAccessKind() != block::AccessKind::Load &&
                   access.getAccessKind() != block::AccessKind::Store))
      return;
    // Auxiliary vectors use ordinary loads: a separate TMA transaction and
    // shared-memory wait compete with the matrix pipeline for little data.
    // A vector projected directly into a contraction remains eligible.
    auto fragment = dyn_cast<block::FragmentType>(access.getAccessValueType());
    if (access.getAccessKind() == block::AccessKind::Load && fragment &&
        fragment.getShape().size() == 1 &&
        !feedsMatrixOperand(access.getAccessResult()))
      return;
    auto plan = planDescriptorAccess(access, analysis.boundaryValidity(operation));
    if (!plan ||
        // Address-producing reads stay in their synchronous dependency chain.
        (access.getAccessKind() == block::AccessKind::Load &&
         feedsIndirectAccessCoordinates(access.getAccessResult())) ||
        !descriptorAccessEligible(
            kernel, access.getAccessResource(), plan->offsets, plan->blockAxes,
            cast<block::FragmentType>(access.getAccessValueType())))
      return;
    if (access.getAccessKind() == block::AccessKind::Load)
      candidates.loads.emplace_back(cast<block::LoadOp>(operation), std::move(*plan));
    else
      candidates.stores.emplace_back(cast<block::StoreOp>(operation), std::move(*plan));
  });
  return candidates;
}

FailureOr<TensorDescriptorChoiceOp>
materializeTensorDescriptorForms(
    func::FuncOp kernel, ArrayRef<TritonLocalOptions> localOptions) {
  auto declaration = block::lookupParameter(
      kernel, StringAttr::get(kernel.getContext(), tensorDescriptorChoice));
  if (!declaration || !declaration.getValueType().isInteger(1) ||
      declaration.getRole() != block::ParameterRole::ProviderOption ||
      declaration.getPhase() != block::ConfigurationBindingPhase::Provider)
    return kernel.emitError("Triton access forms require their declared boolean provider option"), failure();
  if (!llvm::is_contained(declaration.getCandidates().asArrayRef(), int64_t(1)))
    return TensorDescriptorChoiceOp();
  auto capabilities =
      kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
  auto device = nvidia::deviceOf(kernel);
  if (!device || device.getComputeCapabilityMajor() < 9)
    return TensorDescriptorChoiceOp();
  bool hasContraction = false;
  kernel.walk([&](Operation *operation) {
    hasContraction |= isa<block::ContractOp, block::ScaledContractOp>(operation);
  });
  if (!hasContraction)
    return TensorDescriptorChoiceOp();

  auto candidates = readDescriptorAccesses(kernel);
  auto &[loads, stores] = candidates;
  if (loads.empty() && stores.empty())
    return TensorDescriptorChoiceOp();

  OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
  entry.create<TensorDescriptorAllocatorOp>(
      kernel.getLoc(), entry.getI64IntegerAttr(0), entry.getI64IntegerAttr(1),
      entry.getI64IntegerAttr(2), entry.getStringAttr("launch"),
      entry.getStringAttr("torch_cuda_current_device"));
  struct DescriptorPlan {
    Value view;
    block::FragmentType fragment;
    block::FragmentType orderedFragment;
    block::FragmentType nativeFragment;
    SmallVector<int64_t> blockAxes;
    SmallVector<int64_t> permutation;
    TensorDescriptorOp descriptor;
  };
  SmallVector<DescriptorPlan> descriptors;
  auto [nextSource, nextDimension] = block::nextPhysicalAxisIdentities(kernel);
  auto descriptorFor = [&](Value viewValue, block::FragmentType fragment,
                           ArrayRef<int64_t> blockAxes)
      -> FailureOr<DescriptorPlan> {
    for (const DescriptorPlan &plan : descriptors)
      if (plan.view == viewValue && plan.fragment == fragment &&
          ArrayRef<int64_t>(plan.blockAxes) == blockAxes)
        return plan;
    auto view = cast<block::ViewType>(viewValue.getType());
    SmallVector<Value> dimensions;
    for (unsigned axis = 0; axis < view.getRank(); ++axis)
      dimensions.push_back(entry.create<block::DimOp>(
          kernel.getLoc(), entry.getIndexType(), viewValue, axis));
    SmallVector<Value> strides;
    SmallVector<int64_t> alignedAxes;
    for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
      auto stride = descriptorStrideValue(entry, kernel, kernel.getLoc(), view,
                                          axis);
      if (failed(stride))
        return failure();
      strides.push_back(*stride);
      alignedAxes.push_back(axis);
    }
    strides.push_back(entry.create<arith::ConstantIndexOp>(kernel.getLoc(), 1));
    SmallVector<Attribute> shape, mappings, orderedShape, orderedMappings;
    SmallVector<int64_t> permutation;
    for (unsigned axis = 0; axis < view.getRank(); ++axis) {
      auto found = llvm::find(blockAxes, axis);
      if (found != blockAxes.end()) {
        unsigned fragmentAxis = std::distance(blockAxes.begin(), found);
        shape.push_back(fragment.getShape()[fragmentAxis]);
        auto mapping = cast<block::AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
        mappings.push_back(block::AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), axis, mapping.getDerived()));
        orderedShape.push_back(fragment.getShape()[fragmentAxis]);
        orderedMappings.push_back(block::AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), permutation.size(), mapping.getDerived()));
        permutation.push_back(fragmentAxis);
      } else {
        shape.push_back(block::PhysicalExprAttr::get(
            kernel.getContext(),
            block::PhysicalExprKind::Constant, 1,
            entry.getStringAttr(""), entry.getArrayAttr({})));
        mappings.push_back(block::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, axis, true));
      }
    }
    auto nativeFragment = block::FragmentType::get(
        kernel.getContext(), fragment.getElementType(), entry.getArrayAttr(shape),
        entry.getArrayAttr(mappings), fragment.getValidity(), fragment.getOwner());
    auto orderedFragment = block::FragmentType::get(
        kernel.getContext(), fragment.getElementType(),
        entry.getArrayAttr(orderedShape), entry.getArrayAttr(orderedMappings),
        fragment.getValidity(), fragment.getOwner());
    SmallVector<Value> descriptorBlockShape;
    for (Attribute extent : nativeFragment.getShape())
      descriptorBlockShape.push_back(entry.create<block::PhysicalExprOp>(
          kernel.getLoc(), entry.getIndexType(),
          cast<block::PhysicalExprAttr>(extent)));
    std::optional<int64_t> elementBytes =
        descriptorElementBytes(view.getElementType());
    if (!elementBytes || 16 % *elementBytes != 0)
      return failure();
    int64_t maximumBlockElements = maxTritonTensorElements;
    if (llvm::all_of(localOptions, [](const TritonLocalOptions &options) {
          return options.ctas == 1;
        })) {
      // Single-CTA TMA materializes the entire descriptor block in shared
      // memory. Other buffers and barriers remain the provider's responsibility.
      maximumBlockElements = std::min(
          maximumBlockElements,
          capabilities.getLocalStorageBudgetBytes() / *elementBytes);
    }
    SmallVector<int64_t> initialBlockShape(view.getRank(), 1);
    initialBlockShape.back() = 16 / *elementBytes;
    auto descriptor = entry.create<TensorDescriptorOp>(
        kernel.getLoc(), viewValue.getType(), viewValue, dimensions, strides,
        descriptorBlockShape, blockAxes,
        initialBlockShape, "strided", "zero", alignedAxes,
        ArrayRef<int64_t>{static_cast<int64_t>(view.getRank()) - 1},
        /*requirePositiveShape=*/true,
        /*requirePositiveStrides=*/true,
        /*requirePowerOfTwoBlockShape=*/true, /*alignment=*/16,
        /*minimumContiguousBytes=*/16,
        /*pipelineBlockAlignment=*/1,
        /*maximumShapeExtent=*/std::numeric_limits<int32_t>::max(),
        maximumBlockElements);
    descriptors.push_back(
        {viewValue, fragment, orderedFragment, nativeFragment,
         SmallVector<int64_t>(blockAxes.begin(), blockAxes.end()),
         permutation, descriptor});
    return descriptors.back();
  };
  for (auto &[load, plan] : loads) {
    auto access = cast<block::AccessOpInterface>(load.getOperation());
    if (failed(descriptorFor(access.getAccessResource(),
                            cast<block::FragmentType>(access.getAccessValueType()),
                            plan.blockAxes)))
      return load.emitOpError(
          "could not declare its tensor-descriptor runtime contract");
  }
  for (auto &[store, plan] : stores) {
    auto access = cast<block::AccessOpInterface>(store.getOperation());
    if (failed(descriptorFor(access.getAccessResource(),
                            cast<block::FragmentType>(access.getAccessValueType()),
                            plan.blockAxes)))
      return store.emitOpError(
          "could not declare its tensor-descriptor runtime contract");
  }
  SmallVector<Value> descriptorValues;
  for (DescriptorPlan &plan : descriptors)
    descriptorValues.push_back(plan.descriptor.getResult());
  auto choice = entry.create<TensorDescriptorChoiceOp>(
      kernel.getLoc(), entry.getI1Type(), descriptorValues,
      entry.getStringAttr("host"), entry.getStringAttr("all_eligible"),
      declaration.getReference(),
      entry.getStringAttr(tensorDescriptorEligibility));
  auto prepareBranch = [](Region &region) {
    Block &block = region.front();
    if (!block.empty() && isa<scf::YieldOp>(block.back()))
      block.back().erase();
    return OpBuilder(&block, block.end());
  };

  auto reshape = [&](OpBuilder &builder, Location location, Value value,
                     block::FragmentType target) -> FailureOr<Value> {
    if (value.getType() == target)
      return value;
    auto reassociation = block::inferReshapeReassociation(
        cast<block::FragmentType>(value.getType()), target);
    if (failed(reassociation))
      return failure();
    return Value(builder.create<block::ReshapeOp>(location, target, value,
                                               *reassociation));
  };
  for (auto &[load, plan] : loads) {
    auto access = cast<block::AccessOpInterface>(load.getOperation());
    OpBuilder builder(load);
    auto fragment = cast<block::FragmentType>(load.getResult().getType());
    auto descriptor = descriptorFor(access.getAccessResource(), fragment, plan.blockAxes);
    if (failed(descriptor))
      return load.emitOpError(
          "could not materialize the declared tensor-descriptor ABI");
    SmallVector<Value> descriptorOffsets =
        materializeDescriptorOffsets(builder, load.getLoc(), plan.offsets);
    if (load->getParentOfType<scf::ForOp>())
      descriptor->descriptor.setPipelineBlockAlignmentAttr(
          builder.getI64IntegerAttr(128));
    auto conditional = builder.create<scf::IfOp>(
        load.getLoc(), TypeRange{load.getResult().getType()}, choice.getResult(),
        /*withElseRegion=*/true);
    OpBuilder descriptorBuilder = prepareBranch(conditional.getThenRegion());
    auto descriptorLoad = descriptorBuilder.create<DescriptorLoadOp>(
        load.getLoc(), descriptor->nativeFragment,
        descriptor->descriptor.getResult(), descriptorOffsets);
    copyOrigin(load, descriptorLoad);
    auto result = reshape(descriptorBuilder, load.getLoc(),
                          descriptorLoad.getResult(), descriptor->orderedFragment);
    if (failed(result))
      return load.emitOpError("cannot remove descriptor unit axes");
    if (!llvm::is_sorted(descriptor->permutation)) {
      SmallVector<int64_t> inverse(descriptor->permutation.size());
      for (auto [axis, original] : llvm::enumerate(descriptor->permutation))
        inverse[original] = axis;
      result = Value(descriptorBuilder.create<block::TransposeOp>(
          load.getLoc(), fragment, *result,
          descriptorBuilder.getDenseI64ArrayAttr(inverse)));
    }
    descriptorBuilder.create<scf::YieldOp>(load.getLoc(), *result);

    OpBuilder pointerBuilder = prepareBranch(conditional.getElseRegion());
    auto pointer = cast<block::LoadOp>(pointerBuilder.clone(*load));
    pointerBuilder.create<scf::YieldOp>(load.getLoc(), pointer.getResult());
    load.getResult().replaceAllUsesWith(conditional.getResult(0));
    load.erase();
  }

  for (auto &[store, plan] : stores) {
    auto access = cast<block::AccessOpInterface>(store.getOperation());
    OpBuilder builder(store);
    auto descriptor = descriptorFor(
        access.getAccessResource(), cast<block::FragmentType>(access.getAccessPayloads().front().getType()),
        plan.blockAxes);
    if (failed(descriptor))
      return store.emitOpError(
          "could not materialize the declared tensor-descriptor ABI");
    SmallVector<Value> descriptorOffsets =
        materializeDescriptorOffsets(builder, store.getLoc(), plan.offsets);
    auto conditional = builder.create<scf::IfOp>(
        store.getLoc(), TypeRange{}, choice.getResult(),
        /*withElseRegion=*/true);
    OpBuilder descriptorBuilder = prepareBranch(conditional.getThenRegion());
    Value ordered = access.getAccessPayloads().front();
    if (!llvm::is_sorted(descriptor->permutation))
      ordered = descriptorBuilder.create<block::TransposeOp>(
          store.getLoc(), descriptor->orderedFragment, ordered,
          descriptorBuilder.getDenseI64ArrayAttr(descriptor->permutation));
    auto value = reshape(descriptorBuilder, store.getLoc(), ordered,
                         descriptor->nativeFragment);
    if (failed(value))
      return store.emitOpError("cannot insert descriptor unit axes");
    auto descriptorStore = descriptorBuilder.create<DescriptorStoreOp>(
        store.getLoc(), descriptor->descriptor.getResult(), descriptorOffsets,
        *value);
    copyOrigin(store, descriptorStore);
    descriptorBuilder.create<scf::YieldOp>(store.getLoc());

    OpBuilder pointerBuilder = prepareBranch(conditional.getElseRegion());
    pointerBuilder.clone(*store);
    pointerBuilder.create<scf::YieldOp>(store.getLoc());
    store.erase();
  }
  return choice;
}


} // namespace intent::triton::detail
