#include "RegionSummary.h"
#include "RegionSources.h"
#include "Intent/Analysis/RegionSemantics.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;

namespace intent::gpu::region {

std::optional<bool> booleanConstant(Value value) {
  UniformValueAnalysis facts(describeUniformValue);
  return uniformBoolean(facts.evaluate(value));
}

bool isRecordField(Value value, BlockArgument record, unsigned field) {
  auto extract = value.getDefiningOp<ExtractOp>();
  return extract && extract.getRecord() == record && extract.getField() == field;
}

namespace {

bool isMembershipReduction(Value value, Value membershipPredicate) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  auto reduce = value.getDefiningOp<ReduceOp>();
  auto result = dyn_cast<OpResult>(value);
  if (!reduce || !result || reduce.getSources().size() != 1 ||
      reduce.getIdentities().size() != 1 || reduce.getCaptures().size() != 0 ||
      reduce.getNumResults() != 1 || result.getResultNumber() != 0 ||
      reduce.getSources().front() != membershipPredicate ||
      !llvm::hasSingleElement(reduce.getCombine()))
    return false;
  std::optional<bool> identity =
      booleanConstant(reduce.getIdentities().front());
  auto yield = dyn_cast<YieldOp>(reduce.getCombine().front().getTerminator());
  if (!identity || *identity || !yield || yield.getValues().size() != 1 ||
      reduce.getCombine().front().getNumArguments() != 2)
    return false;
  Block &combine = reduce.getCombine().front();
  return isBooleanUnion(UniformValueAnalysis(describeUniformValue), yield.getValues()[0],
                        combine.getArgument(0), combine.getArgument(1));
}

} // namespace

std::optional<SummaryEmptinessPlan>
summaryEmptinessPlan(RegionFoldOp fold, ValueRange identities,
                     Value membershipPredicate) {
  if (identities.size() != 1 || fold.getSummarize().empty() ||
      fold.getCombine().empty())
    return std::nullopt;
  auto identity = identities.front().getDefiningOp<MakeRecordOp>();
  auto summarizeYield =
      dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  auto combineYield =
      dyn_cast<YieldOp>(fold.getCombine().front().getTerminator());
  auto summary = summarizeYield && summarizeYield.getValues().size() == 1
                     ? summarizeYield.getValues().front().getDefiningOp<MakeRecordOp>()
                     : MakeRecordOp();
  auto combined = combineYield && combineYield.getValues().size() == 1
                      ? combineYield.getValues().front().getDefiningOp<MakeRecordOp>()
                      : MakeRecordOp();
  Block &combine = fold.getCombine().front();
  if (!identity || !summary || !combined || combine.getNumArguments() != 2 ||
      identity.getFields().size() != summary.getFields().size() ||
      identity.getFields().size() != combined.getFields().size())
    return std::nullopt;
  auto fullType = dyn_cast<RecordType>(identities.front().getType());
  if (!fullType || fullType.getFieldTypes().size() != identity.getFields().size())
    return std::nullopt;

  std::optional<unsigned> selected;
  for (unsigned field = 0; field < identity.getFields().size(); ++field) {
    std::optional<bool> identityValue =
        booleanConstant(identity.getFields()[field]);
    auto summaryType = dyn_cast<FragmentType>(summary.getFields()[field].getType());
    auto merged = combined.getFields()[field].getDefiningOp<BinaryOp>();
    if (!identityValue || *identityValue || !summaryType ||
        !summaryType.getElementType().isInteger(1) || !merged ||
        !isMembershipReduction(summary.getFields()[field],
                               membershipPredicate) ||
        (merged.getOperatorKind() != BinaryOperator::LogicalOr &&
         merged.getOperatorKind() != BinaryOperator::BitwiseOr))
      continue;
    BlockArgument lhs = combine.getArgument(0);
    BlockArgument rhs = combine.getArgument(1);
    bool fieldsMatch =
        (isRecordField(merged.getLhs(), lhs, field) &&
         isRecordField(merged.getRhs(), rhs, field)) ||
        (isRecordField(merged.getRhs(), lhs, field) &&
         isRecordField(merged.getLhs(), rhs, field));
    if (!fieldsMatch || selected)
      return std::nullopt;
    UniformValueAnalysis values(describeUniformValue);
    if (!isBooleanUnion(values, merged.getResult(), merged.getLhs(), merged.getRhs()) ||
        !hasTrueStateInvariant(values, merged.getResult(), merged.getLhs()))
      return std::nullopt;
    selected = field;
  }
  if (!selected)
    return std::nullopt;

  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  for (unsigned field = 0; field < fullType.getFieldTypes().size(); ++field) {
    if (field == *selected)
      continue;
    names.push_back(fullType.getFieldNames()[field]);
    types.push_back(fullType.getFieldTypes()[field]);
  }
  auto payload = RecordType::get(
      fold.getContext(), ArrayAttr::get(fold.getContext(), names),
      ArrayAttr::get(fold.getContext(), types), fullType.getOwner());
  return SummaryEmptinessPlan{*selected, fullType, payload,
                              summary.getFields()[*selected]};
}

Value allTrueValue(OpBuilder &builder, Location location, Type type) {
  Value truth = builder.create<arith::ConstantOp>(
      location, builder.getI1Type(), builder.getBoolAttr(true));
  if (auto fragment = dyn_cast<FragmentType>(type))
    return builder.create<SplatOp>(location, fragment, truth);
  return truth;
}

FailureOr<Value> stripOptionalRecord(OpBuilder &builder, Location location,
                                     Value value,
                                     const SummaryEmptinessPlan &plan) {
  auto record = value.getDefiningOp<MakeRecordOp>();
  SmallVector<Value> fields;
  unsigned payloadField = 0;
  for (unsigned field = 0; field < plan.fullType.getFieldTypes().size(); ++field) {
    if (field == plan.optionalField)
      continue;
    Type type = cast<TypeAttr>(plan.payloadType.getFieldTypes()[payloadField++])
                    .getValue();
    fields.push_back(record ? record.getFields()[field]
                            : Value(builder.create<ExtractOp>(location, type,
                                                              value, field)));
  }
  return Value(builder.create<MakeRecordOp>(location, plan.payloadType, fields));
}

Value restoreOptionalRecord(OpBuilder &builder, Location location, Value payload,
                            const SummaryEmptinessPlan &plan,
                            Value optionalValidity) {
  SmallVector<Value> fields;
  unsigned payloadField = 0;
  for (unsigned field = 0; field < plan.fullType.getFieldTypes().size(); ++field) {
    Type type = cast<TypeAttr>(plan.fullType.getFieldTypes()[field]).getValue();
    if (field == plan.optionalField) {
      fields.push_back(optionalValidity
                           ? optionalValidity
                           : allTrueValue(builder, location, type));
      continue;
    }
    fields.push_back(builder.create<ExtractOp>(location, type, payload,
                                               payloadField++));
  }
  return builder.create<MakeRecordOp>(location, plan.fullType, fields);
}

FailureOr<Value> restoreOnlineRecord(OpBuilder &builder, Location location,
                                     Value payload,
                                     const SummaryEmptinessPlan &plan,
                                     unsigned massField,
                                     std::optional<unsigned> encodedMaximum) {
  unsigned payloadField = massField - (plan.optionalField < massField);
  auto massType = cast<FragmentType>(
      cast<TypeAttr>(plan.payloadType.getFieldTypes()[payloadField]).getValue());
  Value mass = builder.create<ExtractOp>(location, massType, payload,
                                         payloadField);
  Value scalarZero = builder.create<arith::ConstantOp>(
      location, massType.getElementType(),
      builder.getZeroAttr(massType.getElementType()));
  Value zero = builder.create<SplatOp>(location, massType, scalarZero);
  Value nonempty = builder.create<CompareOp>(
      location, predicateType(massType), mass, zero, ComparePredicate::Ne);
  Type validityType = cast<TypeAttr>(
      plan.fullType.getFieldTypes()[plan.optionalField]).getValue();
  FailureOr<Value> validity = projectPhysicalValueToSchema(
      builder, location, nonempty, validityType);
  if (failed(validity))
    return failure();
  Value restored = restoreOptionalRecord(builder, location, payload, plan, *validity);
  if (encodedMaximum) {
    auto record = restored.getDefiningOp<MakeRecordOp>();
    OpBuilder::InsertionGuard insertion(builder);
    builder.setInsertionPoint(record);
    Value maximum = record.getFields()[*encodedMaximum];
    Value canonicalZero = builder.create<SplatOp>(
        location, maximum.getType(), scalarZero);
    Value canonicalMaximum = builder.create<SelectOp>(
        location, maximum.getType(), *validity, maximum, canonicalZero);
    record->setOperand(*encodedMaximum, canonicalMaximum);
  }
  return restored;
}

void simplifyKnownRecordValues(func::FuncOp kernel) {
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> candidates;
    kernel.walk([&](Operation *operation) {
      if (isa<ExtractOp, SelectOp, BinaryOp>(operation))
        candidates.push_back(operation);
    });
    for (Operation *operation : candidates) {
      if (!operation->getBlock())
        continue;
      if (auto extract = dyn_cast<ExtractOp>(operation)) {
        auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
        if (!record || extract.getField() >= record.getFields().size())
          continue;
        extract.getResult().replaceAllUsesWith(
            record.getFields()[extract.getField()]);
        extract.erase();
        changed = true;
        continue;
      }
      if (auto select = dyn_cast<SelectOp>(operation)) {
        std::optional<bool> condition =
            booleanConstant(select.getCondition());
        if (!condition)
          continue;
        select.getResult().replaceAllUsesWith(
            *condition ? select.getTrueValue() : select.getFalseValue());
        select.erase();
        changed = true;
        continue;
      }
      auto binary = cast<BinaryOp>(operation);
      BinaryOperator kind = binary.getOperatorKind();
      if (kind != BinaryOperator::LogicalAnd &&
          kind != BinaryOperator::BitwiseAnd &&
          kind != BinaryOperator::LogicalOr &&
          kind != BinaryOperator::BitwiseOr)
        continue;
      std::optional<bool> lhs =
          booleanConstant(binary.getLhs());
      std::optional<bool> rhs =
          booleanConstant(binary.getRhs());
      Value replacement;
      bool conjunction = kind == BinaryOperator::LogicalAnd ||
                         kind == BinaryOperator::BitwiseAnd;
      if (conjunction) {
        if (lhs && !*lhs)
          replacement = binary.getLhs();
        else if (rhs && !*rhs)
          replacement = binary.getRhs();
        else if (lhs && *lhs)
          replacement = binary.getRhs();
        else if (rhs && *rhs)
          replacement = binary.getLhs();
      } else {
        if (lhs && *lhs)
          replacement = binary.getLhs();
        else if (rhs && *rhs)
          replacement = binary.getRhs();
        else if (lhs && !*lhs)
          replacement = binary.getRhs();
        else if (rhs && !*rhs)
          replacement = binary.getLhs();
      }
      if (!replacement)
        continue;
      binary.getResult().replaceAllUsesWith(replacement);
      binary.erase();
      changed = true;
    }
  } while (changed);
}

void foldKnownRecordProjections(Operation *structured) {
  for (Region &region : structured->getRegions())
    region.walk([&](ExtractOp extract) {
      if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
        extract.getResult().replaceAllUsesWith(
            record.getFields()[extract.getField()]);
        extract.erase();
      }
    });
}

void forwardUnusedRecordFields(RegionFoldOp fold) {
  if (fold.getNumResults() != 1 ||
      !isa<RecordType>(fold.getResult(0).getType()))
    return;
  Block &combine = fold.getCombine().front();
  if (combine.getNumArguments() != 2)
    return;
  auto yield = cast<YieldOp>(combine.getTerminator());
  auto record = yield.getValues().front().getDefiningOp<MakeRecordOp>();
  if (!record)
    return;
  llvm::SmallDenseSet<unsigned> live;
  SmallVector<unsigned> pending;
  auto require = [&](unsigned field) {
    if (live.insert(field).second)
      pending.push_back(field);
  };
  for (Operation *user : fold.getResult(0).getUsers()) {
    auto extract = dyn_cast<ExtractOp>(user);
    if (!extract)
      return;
    require(extract.getField());
  }
  while (!pending.empty()) {
    unsigned field = pending.pop_back_val();
    llvm::SmallPtrSet<Value, 16> visited;
    std::function<bool(Value)> collect = [&](Value value) {
      if (!visited.insert(value).second)
        return true;
      if (value == combine.getArgument(0) ||
          value == combine.getArgument(1))
        return false;
      if (auto extract = value.getDefiningOp<ExtractOp>();
          extract && (extract.getRecord() == combine.getArgument(0) ||
                      extract.getRecord() == combine.getArgument(1))) {
        require(extract.getField());
        return true;
      }
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != &combine)
        return true;
      if (definition->getNumRegions())
        return false;
      return llvm::all_of(definition->getOperands(), collect);
    };
    if (!collect(record.getFields()[field]))
      return;
  }
  OpBuilder builder(record);
  for (auto [field, value] : llvm::enumerate(record.getFields())) {
    if (live.contains(field))
      continue;
    // Unobserved fields cannot feed any observed combine field. Carry their
    // existing value through, so ordinary DCE can remove their summary work
    // without changing the record schema or the observed recurrence.
    Value unchanged = builder.create<ExtractOp>(
        record.getLoc(), value.getType(), combine.getArgument(0), field);
    record->setOperand(field, unchanged);
  }
}

} // namespace intent::gpu::region
