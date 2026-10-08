#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace intent::gpu {

Type uniformElementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type)) return fragment.getElementType();
  return type;
}

Value uniformScalarSource(Value value) {
  while (value && isa<FragmentType>(value.getType())) {
    UniformExpression expression = describeUniformValue(value);
    if (expression.kind != UniformKind::Forward ||
        expression.operands.size() != 1)
      return {};
    value = expression.operands.front();
  }
  return value && value.getType().isIntOrIndexOrFloat() ? value : Value();
}

namespace {

class UniformAxisQuery {
public:
  using Axis = std::pair<Value, unsigned>;

  bool query(Value current, unsigned axis) {
    if (!current) return false;
    auto fragment = dyn_cast<FragmentType>(current.getType());
    if (!fragment)
      return current.getType().isIntOrIndexOrFloat();
    if (axis >= fragment.getShape().size()) return false;
    Axis key{current, axis};
    if (auto found = bindings.find(key); found != bindings.end())
      return found->second;
    if (auto found = known.find(key); found != known.end())
      return found->second;
    if (!active.insert(key).second) return false;
    auto finish = [&](bool result) {
      active.erase(key);
      known[key] = result;
      return result;
    };
    Operation *operation = current.getDefiningOp();
    if (!operation || !isMemoryEffectFree(operation))
      return finish(false);
    if (auto reduce = dyn_cast<ReduceOp>(operation))
      return finish(reduction(reduce, axis));
    // Scan's prefix axis is not uniform merely because its input is uniform.
    // Other region producers need their own value and projection contract.
    if (operation->getNumRegions()) return finish(false);
    UniformExpression expression = describeUniformValue(current);
    switch (expression.kind) {
    case UniformKind::Unknown:
      // Constant evaluability is not numerical lane semantics. Approximate
      // pointwise operations preserve uniformity without becoming foldable.
      if (isLaneWisePointwiseOperation(operation)) break;
      return finish(false);
    case UniformKind::Join:
    case UniformKind::Aggregate:
    case UniformKind::Extract:
    case UniformKind::Fold:
    case UniformKind::Contract:
      return finish(false);
    case UniformKind::Constant:
      return finish(static_cast<bool>(expression.literal));
    default:
      break;
    }
    if (expression.operands.empty()) return finish(false);

    // Extracting a known product field is a value-forwarding fact. It need not
    // have the operation's ordinary fragment operand relation (its operand is
    // a record), but it preserves the exact forwarded field type and axes.
    if (expression.kind == UniformKind::Forward &&
        !isa<FragmentOpInterface>(operation) &&
        expression.operands.size() == 1 &&
        expression.operands.front().getType() == current.getType())
      return finish(query(expression.operands.front(), axis));

    auto relations = queryFragmentOperandRelations(cast<OpResult>(current));
    if (failed(relations)) return finish(false);
    for (const auto &relation : *relations)
      if (llvm::is_contained(relation.invariantResultAxes, axis))
        return finish(false);
    for (auto [slot, operand] : llvm::enumerate(operation->getOperands())) {
      if (!isa<FragmentType>(operand.getType())) {
        if (!operand.getType().isIntOrIndexOrFloat()) return finish(false);
        continue;
      }
      auto relation = llvm::find_if(*relations, [&](const auto &relation) {
        return relation.operandNumber == slot;
      });
      if (relation == relations->end()) return finish(false);
      const auto *group = relation->groupForResultAxis(axis);
      if (!group) return finish(false);
      // This is an explicit replication edge, not a guess from a one-lane
      // physical type. In particular, the input may vary on its other axes.
      if (group->kind == FragmentAxisRelationKind::Broadcast) continue;
      if (!llvm::all_of(group->sourceAxes, [&](unsigned sourceAxis) {
            return query(operand, sourceAxis);
          }))
        return finish(false);
    }
    return finish(true);
  }

private:
  bool reduction(ReduceOp reduce, unsigned resultAxis) {
    // The structured projection owns one component and no captures. Its helper
    // is checked by the same lane proof used by the actual reconstruction.
    if (reduce.getSources().size() != 1 ||
        reduce.getIdentities().size() != 1 ||
        !reduce.getCaptures().empty() || reduce.getNumResults() != 1 ||
        !llvm::hasSingleElement(reduce.getCombine()))
      return false;
    Block &body = reduce.getCombine().front();
    if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>() ||
        failed(proveLaneWiseHelper(reduce.getCombine())))
      return false;
    auto source = dyn_cast<FragmentType>(reduce.getSources().front().getType());
    auto result = dyn_cast<FragmentType>(reduce.getResult(0).getType());
    if (!source || !result) return false;
    SmallVector<unsigned> freeAxes;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
        freeAxes.push_back(axis);
    if (freeAxes.size() != result.getShape().size() ||
        resultAxis >= freeAxes.size())
      return false;

    Value input = reduce.getSources().front();
    Value identity = reduce.getIdentities().front();
    // Identity participates even for an empty reduction. The source test is
    // on the exact retained axis, never on the reduced member axis.
    if (!query(input, freeAxes[resultAxis]) || !query(identity, resultAxis))
      return false;
    UniformAxisQuery combine;
    for (auto [axis, sourceAxis] : llvm::enumerate(freeAxes)) {
      bool uniform = query(input, sourceAxis) && query(identity, axis);
      combine.bindings[{body.getArgument(0), axis}] = uniform;
      combine.bindings[{body.getArgument(1), axis}] = uniform;
    }
    // Check the actual yielded computation under those formal bindings.
    // A separate query prevents assumptions or cached helper facts from
    // escaping into another component or invocation of the analysis.
    return combine.query(body.back().getOperand(0), resultAxis);
  }

  DenseMap<Axis, bool> bindings;
  DenseMap<Axis, bool> known;
  llvm::SmallDenseSet<Axis> active;
};

} // namespace

SmallVector<bool> uniformFragmentAxes(Value value) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment) return {};
  UniformAxisQuery query;
  SmallVector<bool> result;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis)
    result.push_back(query.query(value, axis));
  return result;
}

UniformExpression describeUniformValue(Value value) {
  UniformExpression result = describeScalarValue(value);
  result.type = uniformElementType(value.getType());
  Operation *op = value.getDefiningOp();
  if (!op) return result;
  using K = UniformKind;
  if (auto physical = dyn_cast<PhysicalExprOp>(op)) {
    if (auto literal = constantPhysicalExpression(physical.getExpression())) {
      result.kind = K::Constant;
      result.literal = IntegerAttr::get(value.getType(), *literal);
    }
    return result;
  }
  if (isa<BroadcastOp, SplatOp, ReshapeOp, TransposeOp>(op)) result.kind = K::Forward;
  else if (isa<JoinOp>(op)) result.kind = K::Join;
  else if (isa<MakeRecordOp>(op)) result.kind = K::Aggregate;
  else if (auto extract = dyn_cast<ExtractOp>(op)) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      result.kind = K::Forward;
      result.operands = {record.getFields()[extract.getField()]};
      return result;
    }
    result.kind = K::Extract;
    result.result = extract.getField();
  } else if (isa<SelectOp>(op)) result.kind = K::Select;
  else if (isa<CastOp>(op)) result.kind = K::Cast;
  else if (isa<BitcastOp>(op)) result.kind = K::Bitcast;
  else if (auto unary = dyn_cast<UnaryOp>(op)) {
    return describeUniformUnary(result.type, unary.getOperatorKind(), unary.getInput(),
                                 unary.getApproximate(), unary.getFlushToZero());
  } else if (auto binary = dyn_cast<BinaryOp>(op)) {
    return describeUniformBinary(result.type, binary.getOperatorKind(), binary.getLhs(), binary.getRhs(),
                                  binary.getApproximate(), binary.getFlushToZero());
  } else if (auto compare = dyn_cast<CompareOp>(op)) {
    return describeUniformCompare(result.type, uniformElementType(compare.getLhs().getType()),
                                   compare.getPredicate(), compare.getLhs(), compare.getRhs());
  } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
    bool nonempty = false;
    if (auto source = dyn_cast<FragmentType>(reduce.getSources().front().getType())) {
      auto kernel = reduce->getParentOfType<func::FuncOp>();
      nonempty = kernel && llvm::all_of(reduce.getAxes(), [&](int64_t axis) {
        return axis >= 0 && static_cast<unsigned>(axis) < source.getShape().size() &&
            isKnownPositiveExtent(cast<PhysicalExprAttr>(source.getShape()[axis]), kernel);
      });
    }
    return describeStructuredReduction(cast<OpResult>(value), result.type, nonempty);
  } else if (isa<ContractOp>(op)) result.kind = K::Contract;
  result.operands.assign(op->operand_begin(), op->operand_end());
  return result;
}

}
