#include "Intent/Target/CuTile/Transforms/Control/Loops.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::cutile {
namespace {
bool fitsNativeLoopBound(Value value, unsigned depth = 0) {
  if (depth >= 32)
    return false;
  value = stripIndexIdentities(value);
  if (value.getDefiningOp<gpu::ProgramIdOp>())
    return true;
  if (auto expression = value.getDefiningOp<gpu::PhysicalExprOp>()) {
    auto bounds = gpu::queryPositiveExtentBounds(
        expression.getExpression(), expression->getParentOfType<func::FuncOp>());
    return bounds && llvm::isInt<32>(bounds->second);
  }
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (integer && (integer.getWidth() < 32 ||
                  (integer.getWidth() == 32 && !integer.isUnsigned())))
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
    if (!attribute)
      return false;
    return integer && integer.isUnsigned()
               ? attribute.getValue().getActiveBits() < 32
               : attribute.getValue().isSignedIntN(32);
  }
  if (auto upper = gpu::queryNonNegativeIndexUpperBound(value)) {
    auto bounds = gpu::queryPositiveExtentBounds(
        upper, value.getParentRegion()->getParentOfType<func::FuncOp>());
    if (bounds && llvm::isInt<32>(bounds->second))
      return true;
  }
  if (auto parameter = value.getDefiningOp<gpu::ParameterOp>())
    return llvm::all_of(parameter.getDeclaration().getCandidates().asArrayRef(),
                        [](int64_t candidate) { return llvm::isInt<32>(candidate); });
  if (auto bound = value.getDefiningOp<gpu::RangeBoundOp>())
    if (auto range = bound.getRange().getDefiningOp<gpu::RangeOp>())
      return fitsNativeLoopBound(bound.getBound() == 0 ? range.getStart()
                                 : bound.getBound() == 1 ? range.getStop()
                                                        : range.getStep(),
                                 depth + 1);
  if (auto cast = value.getDefiningOp<gpu::CastOp>())
    if (value.getType().isIndex() ||
        (integer && integer.getWidth() == 64 && !integer.isUnsigned()))
      return fitsNativeLoopBound(cast.getValue(), depth + 1);
  if (auto clamp = value.getDefiningOp<gpu::BinaryOp>()) {
    auto kind = clamp.getOperatorKind();
    if (kind != BinaryOperator::Minimum && kind != BinaryOperator::Maximum)
      return false;
    BinaryOperator opposite = kind == BinaryOperator::Minimum
                                  ? BinaryOperator::Maximum
                                  : BinaryOperator::Minimum;
    for (auto [limit, nestedValue] :
         {std::pair{clamp.getLhs(), clamp.getRhs()},
          std::pair{clamp.getRhs(), clamp.getLhs()}}) {
      auto nested = nestedValue.getDefiningOp<gpu::BinaryOp>();
      if (fitsNativeLoopBound(limit, depth + 1) && nested &&
          nested.getOperatorKind() == opposite &&
          (fitsNativeLoopBound(nested.getLhs(), depth + 1) ||
           fitsNativeLoopBound(nested.getRhs(), depth + 1)))
        return true;
    }
  }
  return false;
}

std::optional<int64_t> maximumNativeLoopStep(Value value) {
  value = stripIndexIdentities(value);
  if (auto expression = value.getDefiningOp<gpu::PhysicalExprOp>()) {
    auto bounds = gpu::queryPositiveExtentBounds(
        expression.getExpression(), expression->getParentOfType<func::FuncOp>());
    if (bounds && llvm::isInt<32>(bounds->second))
      return bounds->second;
  }
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
    if (attribute && attribute.getValue().isSignedIntN(32) &&
        attribute.getInt() > 0)
      return attribute.getInt();
  }
  if (auto parameter = value.getDefiningOp<gpu::ParameterOp>()) {
    ArrayRef<int64_t> candidates =
        parameter.getDeclaration().getCandidates().asArrayRef();
    if (!candidates.empty() && llvm::all_of(candidates, [](int64_t candidate) {
          return candidate > 0 && llvm::isInt<32>(candidate);
        }))
      return *llvm::max_element(candidates);
  }
  return std::nullopt;
}

} // namespace

void realizeWideLoops(func::FuncOp kernel) {
  SmallVector<scf::ForOp> loops;
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    Type type = loop.getInductionVar().getType();
    if (type.isIndex() || type.isInteger(64))
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    OpBuilder builder(loop);
    Location location = loop.getLoc();
    std::optional<int64_t> maximumStep = maximumNativeLoopStep(loop.getStep());
    auto isInductionQuotient = [&](Operation &operation) {
      auto binary = dyn_cast<gpu::BinaryOp>(operation);
      return binary && binary.getOperatorKind() == BinaryOperator::FloorDivide &&
             binary.getLhs() == loop.getInductionVar() &&
             gpu::samePhysicalScalarExpression(binary.getRhs(), loop.getStep());
    };
    bool useTileCounter = maximumStep && *maximumStep > 1 &&
                          gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
                          gpu::IndexRelations().multipleOf(loop.getLowerBound(), loop.getStep()) &&
                          llvm::any_of(loop.getBody()->without_terminator(),
                                       isInductionQuotient);
    auto nativeLoop = [&](OpBuilder &nested) {
      Value lowerBound = loop.getLowerBound();
      Value upperBound = loop.getUpperBound();
      if (useTileCounter) {
        Type wideType = loop.getInductionVar().getType();
        Value one = nested.create<arith::ConstantOp>(
            location, wideType, nested.getIntegerAttr(wideType, 1));
        Value positive = nested.create<gpu::CompareOp>(
            location, nested.getI1Type(), upperBound, loop.getLowerBound(),
            ComparePredicate::Gt);
        Value distance = nested.create<gpu::SelectOp>(
            location, wideType, positive, upperBound, loop.getLowerBound());
        Value adjustment = nested.create<gpu::BinaryOp>(
            location, wideType, loop.getStep(), one, BinaryOperator::Subtract);
        Value rounded = nested.create<gpu::BinaryOp>(
            location, wideType, distance, adjustment, BinaryOperator::Add);
        upperBound = nested.create<gpu::BinaryOp>(
            location, wideType, rounded, loop.getStep(), BinaryOperator::FloorDivide);
        lowerBound = nested.create<gpu::BinaryOp>(
            location, wideType, lowerBound, loop.getStep(), BinaryOperator::FloorDivide);
      }
      Value lower = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), lowerBound);
      Value upper = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), upperBound);
      Value step = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), loop.getStep());
      if (useTileCounter)
        step = nested.create<arith::ConstantIntOp>(location, 1, 32);
      auto replacement = nested.create<scf::ForOp>(
          location, lower, upper, step, loop.getInitArgs(),
          [&](OpBuilder &body, Location bodyLoc, Value induction,
              ValueRange carried) {
            IRMapping mapping;
            Value counter = body.create<gpu::CastOp>(
                bodyLoc, loop.getInductionVar().getType(), induction);
            Value wide = counter;
            if (useTileCounter)
              wide = body.create<gpu::BinaryOp>(
                  bodyLoc, wide.getType(), counter, loop.getStep(), BinaryOperator::Multiply);
            mapping.map(loop.getInductionVar(), wide);
            mapping.map(loop.getRegionIterArgs(), carried);
            for (Operation &operation : loop.getBody()->without_terminator()) {
              // IV = counter * step, so its exact tile quotient is the counter.
              if (useTileCounter && isInductionQuotient(operation)) {
                Value quotient = counter;
                Type resultType = operation.getResult(0).getType();
                if (quotient.getType() != resultType)
                  quotient = body.create<gpu::CastOp>(bodyLoc, resultType, quotient);
                mapping.map(operation.getResult(0), quotient);
                continue;
              }
              body.clone(operation, mapping);
            }
            SmallVector<Value> yielded;
            for (Value value :
                 cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands())
              yielded.push_back(mapping.lookupOrDefault(value));
            body.create<scf::YieldOp>(bodyLoc, yielded);
          });
      replacement->setAttrs(loop->getAttrs());
      return replacement;
    };
    auto wideLoop = [&](OpBuilder &nested) {
      OpBuilder::InsertionGuard guard(nested);
      SmallVector<Value> initial{loop.getLowerBound()};
      llvm::append_range(initial, loop.getInitArgs());
      SmallVector<Type> types;
      for (Value value : initial)
        types.push_back(value.getType());
      SmallVector<Location> locations(types.size(), location);
      auto replacement = nested.create<scf::WhileOp>(location, types, initial);
      if (Attribute origin = loop->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      Block *before = nested.createBlock(&replacement.getBefore(), {}, types, locations);
      Value condition = nested.create<gpu::CompareOp>(
          location, nested.getI1Type(), before->getArgument(0),
          loop.getUpperBound(), ComparePredicate::Lt);
      nested.create<scf::ConditionOp>(location, condition, before->getArguments());
      Block *after = nested.createBlock(&replacement.getAfter(), {}, types, locations);
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), after->getArgument(0));
      mapping.map(loop.getRegionIterArgs(), after->getArguments().drop_front());
      for (Operation &operation : loop.getBody()->without_terminator())
        nested.clone(operation, mapping);
      Value next = nested.create<gpu::BinaryOp>(
          location, types.front(), after->getArgument(0), loop.getStep(),
          BinaryOperator::Add);
      SmallVector<Value> yielded{next};
      for (Value value : cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands())
        yielded.push_back(mapping.lookupOrDefault(value));
      nested.create<scf::YieldOp>(location, yielded);
      return replacement;
    };
    // Unit steps keep the terminating increment in range as well as the body IV.
    bool unitStep = maximumStep && *maximumStep == 1;
    bool boundedStep = unitStep;
    if (!boundedStep && maximumStep)
      if (auto upper = gpu::queryNonNegativeIndexUpperBound(loop.getUpperBound())) {
        auto bounds = gpu::queryPositiveExtentBounds(upper, kernel);
        boundedStep = bounds &&
                      bounds->second <= int64_t{INT32_MAX} - *maximumStep + 1;
      }
    if (boundedStep && fitsNativeLoopBound(loop.getLowerBound()) &&
        fitsNativeLoopBound(loop.getUpperBound())) {
      auto replacement = nativeLoop(builder);
      loop.replaceAllUsesWith(replacement.getResults());
      loop.erase();
      continue;
    }
    auto integer = dyn_cast<IntegerType>(loop.getInductionVar().getType());
    // Runtime scalar bounds can use the same checked native loop as specialized
    // bounds.  The predicate proves the i32 range before narrowing; genuinely
    // wide domains retain the original explicit i64 induction state.
    if (maximumStep && (!integer || !integer.isUnsigned())) {
      Value minimum = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(), INT32_MIN));
      Value maximum = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(), INT32_MAX));
      // The last body IV is at most upper - 1; its increment must also fit.
      Value maximumUpper = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(),
                                 int64_t{INT32_MAX} - *maximumStep + 1));
      Value condition;
      for (Value bound : {loop.getLowerBound(), loop.getUpperBound()}) {
        if (bound != loop.getUpperBound() && fitsNativeLoopBound(bound))
          continue;
        Value limit = bound == loop.getUpperBound() ? maximumUpper : maximum;
        Value fits;
        if (gpu::PhysicalExprAttr upperBound =
                gpu::queryNonNegativeIndexUpperBound(bound)) {
          Value symbolic = builder.create<gpu::PhysicalExprOp>(
              location, builder.getIndexType(), upperBound);
          fits = builder.create<gpu::CompareOp>(
              location, builder.getI1Type(), symbolic, limit, ComparePredicate::Le);
        } else {
          Value lower = builder.create<gpu::CompareOp>(
              location, builder.getI1Type(), bound, minimum, ComparePredicate::Ge);
          Value upper = builder.create<gpu::CompareOp>(
              location, builder.getI1Type(), bound, limit, ComparePredicate::Le);
          fits = builder.create<gpu::BinaryOp>(
              location, builder.getI1Type(), lower, upper, BinaryOperator::LogicalAnd);
        }
        condition = condition ? Value(builder.create<gpu::BinaryOp>(
                                    location, builder.getI1Type(), condition, fits,
                                    BinaryOperator::LogicalAnd))
                              : fits;
      }
      auto replacement = builder.create<scf::IfOp>(
          location, condition,
          [&](OpBuilder &nested, Location nestedLocation) {
            auto native = nativeLoop(nested);
            nested.create<scf::YieldOp>(nestedLocation, native.getResults());
          },
          [&](OpBuilder &nested, Location nestedLocation) {
            auto wide = wideLoop(nested);
            nested.create<scf::YieldOp>(nestedLocation, wide.getResults().drop_front());
          });
      loop.replaceAllUsesWith(replacement.getResults());
      loop.erase();
      continue;
    }
    // cuTile range always has an i32 IV. Unproven bounds retain explicit wide state.
    auto replacement = wideLoop(builder);
    loop.replaceAllUsesWith(replacement.getResults().drop_front());
    loop.erase();
  }
}

void preserveNativeIndexValues(func::FuncOp kernel) {
  kernel.walk([](Operation *operation) {
    if (!isa<TileLoadOp, TileStoreOp, TileAtomicAddOp>(operation))
      return;
    for (OpOperand &operand : operation->getOpOperands()) {
      auto cast = operand.get().getDefiningOp<gpu::CastOp>();
      if (cast && cast.getResult().getType().isIndex() &&
          cast.getValue().getType().isSignlessInteger(32))
        operand.set(cast.getValue());
    }
  });
}

} // namespace intent::cutile
