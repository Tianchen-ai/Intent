#include "Values.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Target/Triton/Analysis/Contractions.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::triton::detail {
void canonicalizeBroadcastProjections(func::FuncOp kernel) {
  SmallVector<Value> pending;
  kernel.walk([&](gpu::ContractOp contract) {
    pending.push_back(contract.getLhs());
    pending.push_back(contract.getRhs());
  });
  llvm::DenseSet<Operation *> visited;
  SmallVector<gpu::ReshapeOp> candidates;
  while (!pending.empty()) {
    Operation *producer = pending.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    llvm::append_range(pending, producer->getOperands());
    for (Region &region : producer->getRegions())
      for (Block &block : region)
        llvm::append_range(pending, block.getTerminator()->getOperands());
    if (auto reshape = dyn_cast<gpu::ReshapeOp>(producer))
      candidates.push_back(reshape);
  }
  // A load's address and predicate producers are outside this additional scope.
  kernel.walk([&](gpu::ReshapeOp reshape) {
    auto source = cast<gpu::FragmentType>(reshape.getValue().getType());
    Value resource;
    if (auto load = reshape.getValue().getDefiningOp<gpu::LoadOp>())
      resource = load.getResource();
    else
      return;
    // A vector resource may be loaded across several execution axes.
    auto view = dyn_cast<gpu::ViewType>(resource.getType());
    if ((source.getShape().size() == 1 || (view && view.getRank() == 1)) &&
        visited.insert(reshape).second)
      candidates.push_back(reshape);
  });
  SmallVector<gpu::ReshapeOp> projections;
  for (gpu::ReshapeOp reshape : candidates) {
    auto relations = gpu::queryFragmentOperandRelations(reshape);
    if (succeeded(relations) && relations->front().isUnitAxisInsertion())
      projections.push_back(reshape);
  }
  // Expand-dims preserves dot input alignment and avoids redundant register
  // copies (and loads) when broadcasting a loaded vector. Leave unrelated
  // high-rank and reduction result layout choices free.
  for (gpu::ReshapeOp reshape : projections) {
    OpBuilder builder(reshape);
    auto broadcast = builder.create<gpu::BroadcastOp>(
        reshape.getLoc(), reshape.getResult().getType(), reshape.getValue());
    broadcast->setDiscardableAttrs(llvm::to_vector(reshape->getDiscardableAttrs()));
    reshape.getResult().replaceAllUsesWith(broadcast.getResult());
    reshape.erase();
  }
}

void selectContractForms(func::FuncOp kernel) {
  SmallVector<gpu::ContractOp> contracts;
  kernel.walk([&](gpu::ContractOp contract) { contracts.push_back(contract); });
  for (gpu::ContractOp contract : contracts) {
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    bool ieeeFp32 = lhs.getElementType().isF32() &&
                    rhs.getElementType().isF32() &&
                    contract.getResult().getType().getElementType().isF32();
    if (lhs.getShape().empty())
      continue;
    auto reductionExtent =
        dyn_cast<gpu::PhysicalExprAttr>(
            lhs.getShape()[lhs.getShape().size() - 1]);
    if (!reductionExtent)
      continue;
    OpBuilder builder(contract);
    auto expandedFits = [&]() -> Value {
      Value count = builder.create<gpu::PhysicalExprOp>(
          contract.getLoc(), builder.getIndexType(), expandedContractionElements(contract));
      Value maximum = builder.create<arith::ConstantIndexOp>(
          contract.getLoc(), maxTritonTensorElements);
      return builder.create<gpu::CompareOp>(contract.getLoc(), builder.getI1Type(),
                                           count, maximum, ComparePredicate::Le);
    };
    if (reductionExtent.getKind() ==
        gpu::PhysicalExprKind::Constant) {
      if (reductionExtent.getValue() < expansionReductionThreshold) {
        contract->setAttr(contractFormAttr,
                          StringAttr::get(kernel.getContext(), "multiply_sum"));
        continue;
      }
      if (!ieeeFp32)
        continue;
    }
    // IEEE dot has no matrix reuse along a unit free axis. Prefer its parallel
    // reduction expansion there, as well as when estimated staging is too large.
    // The provider still validates the native form's actual resource usage.
    Value canExpand = expandedFits();
    Value extent = builder.create<gpu::PhysicalExprOp>(
        contract.getLoc(), builder.getIndexType(), reductionExtent);
    Value minimum = builder.create<arith::ConstantIndexOp>(
        contract.getLoc(), expansionReductionThreshold);
    Value small = builder.create<gpu::CompareOp>(
        contract.getLoc(), builder.getI1Type(), extent, minimum,
        ComparePredicate::Lt);
    if (ieeeFp32) {
      Value footprint = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 0);
      for (auto operand : {lhs, rhs}) {
        Value bytes = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 4);
        for (Attribute dimension : operand.getShape()) {
          Value width = builder.create<gpu::PhysicalExprOp>(
              contract.getLoc(), builder.getIndexType(),
              cast<gpu::PhysicalExprAttr>(dimension));
          bytes = builder.create<gpu::BinaryOp>(contract.getLoc(),
              builder.getIndexType(), bytes, width, BinaryOperator::Multiply);
        }
        footprint = builder.create<gpu::BinaryOp>(contract.getLoc(),
            builder.getIndexType(), footprint, bytes, BinaryOperator::Add);
      }
      auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
      Value capacity = builder.create<arith::ConstantIndexOp>(
          contract.getLoc(), capabilities.getMaxDynamicSharedMemoryPerBlock());
      Value preferExpansion = builder.create<gpu::CompareOp>(contract.getLoc(),
          builder.getI1Type(), footprint, capacity, ComparePredicate::Gt);
      auto shape = contract.getAccumulator().getType().getShape().getValue();
      for (Attribute dimension : shape.take_back(std::min<size_t>(2, shape.size()))) {
        Value width = builder.create<gpu::PhysicalExprOp>(
            contract.getLoc(), builder.getIndexType(),
            cast<gpu::PhysicalExprAttr>(dimension));
        Value one = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 1);
        Value unit = builder.create<gpu::CompareOp>(contract.getLoc(),
            builder.getI1Type(), width, one, ComparePredicate::Eq);
        preferExpansion = builder.create<gpu::BinaryOp>(contract.getLoc(),
            builder.getI1Type(), preferExpansion, unit, BinaryOperator::LogicalOr);
      }
      preferExpansion = builder.create<gpu::BinaryOp>(contract.getLoc(),
          builder.getI1Type(), preferExpansion, canExpand, BinaryOperator::LogicalAnd);
      small = builder.create<gpu::BinaryOp>(contract.getLoc(), builder.getI1Type(),
          small, preferExpansion, BinaryOperator::LogicalOr);
    }
    auto choice = builder.create<scf::IfOp>(
        contract.getLoc(), TypeRange{contract.getResult().getType()}, small, true);
    builder.setInsertionPointToStart(&choice.getThenRegion().front());
    auto expanded = cast<gpu::ContractOp>(builder.clone(*contract));
    expanded->setAttr(contractFormAttr,
                      StringAttr::get(kernel.getContext(), "multiply_sum"));
    builder.create<scf::YieldOp>(contract.getLoc(), expanded.getResult());
    builder.setInsertionPointToStart(&choice.getElseRegion().front());
    auto native = cast<gpu::ContractOp>(builder.clone(*contract));
    builder.create<scf::YieldOp>(contract.getLoc(), native.getResult());
    contract.getResult().replaceAllUsesWith(choice.getResult(0));
    contract.erase();
  }

  contracts.clear();
  kernel.walk([&](gpu::ContractOp contract) {
    auto form = contract->getAttrOfType<StringAttr>(contractFormAttr);
    if (form && form.getValue() == "multiply_sum" &&
        contract.getAccumulator().getType().getElementType().isF32())
      contracts.push_back(contract);
  });
  for (gpu::ContractOp contract : contracts) {
    OpBuilder builder(contract);
    auto shape = contract.getAccumulator().getType().getShape().getValue();
    // A unit matrix axis offers no second free axis to amortize serial K work.
    // Keep its legal multiply/reduce expansion parallel along K.
    ArrayRef<Attribute> matrixAxes =
        shape.take_back(std::min<size_t>(2, shape.size()));
    if (llvm::any_of(matrixAxes, [&](Attribute dimension) {
          auto extent = evaluateCompileTimeExpression(
              cast<gpu::PhysicalExprAttr>(dimension));
          return extent && *extent == 1;
        }))
      continue;
    auto elements = cast<gpu::PhysicalExprAttr>(shape.front());
    for (Attribute extent : shape.drop_front())
      elements = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          gpu::PhysicalExprKind::Multiply, 0,
          builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
    auto fmaForm = builder.getStringAttr("fma");
    // Serial K accumulation needs enough independent output elements.
    if (auto count = evaluateCompileTimeExpression(elements)) {
      if (*count >= 256)
        contract->setAttr(contractFormAttr, fmaForm);
      continue;
    }
    Value count = builder.create<gpu::PhysicalExprOp>(
        contract.getLoc(), builder.getIndexType(), elements);
    Value limit = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 256);
    Value wide = builder.create<gpu::CompareOp>(
        contract.getLoc(), builder.getI1Type(), count, limit,
        ComparePredicate::Ge);
    for (Attribute dimension : matrixAxes) {
      auto extent = cast<gpu::PhysicalExprAttr>(dimension);
      if (evaluateCompileTimeExpression(extent))
        continue;
      Value width = builder.create<gpu::PhysicalExprOp>(
          contract.getLoc(), builder.getIndexType(), extent);
      Value one = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 1);
      Value multiple = builder.create<gpu::CompareOp>(
          contract.getLoc(), builder.getI1Type(), width, one, ComparePredicate::Gt);
      wide = builder.create<gpu::BinaryOp>(
          contract.getLoc(), builder.getI1Type(), wide, multiple,
          BinaryOperator::LogicalAnd);
    }
    auto choice = builder.create<scf::IfOp>(
        contract.getLoc(), TypeRange{contract.getResult().getType()}, wide, true);
    builder.setInsertionPointToStart(&choice.getThenRegion().front());
    auto fma = cast<gpu::ContractOp>(builder.clone(*contract));
    fma->setAttr(contractFormAttr, fmaForm);
    builder.create<scf::YieldOp>(contract.getLoc(), fma.getResult());
    builder.setInsertionPointToStart(&choice.getElseRegion().front());
    auto reduction = cast<gpu::ContractOp>(builder.clone(*contract));
    builder.create<scf::YieldOp>(contract.getLoc(), reduction.getResult());
    contract.getResult().replaceAllUsesWith(choice.getResult(0));
    contract.erase();
  }
}

bool hasMapRelation(Type type, gpu::FragmentType result) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragment.getShape() == result.getShape() &&
           fragment.getAxisMaps() == result.getAxisMaps() &&
           fragment.getValidity() == result.getValidity() &&
           fragment.getOwner() == result.getOwner();
  return isa<IntegerType, IndexType, FloatType>(type);
}

bool isScalarizableMapProducer(Operation *operation,
                              gpu::FragmentType result) {
  if (!llvm::all_of(operation->getOperandTypes(), [&](Type type) {
        return hasMapRelation(type, result);
      }) || !llvm::all_of(operation->getResultTypes(), [&](Type type) {
        return hasMapRelation(type, result);
      }))
    return false;
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    APInt lower, upper, step;
    if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        !step.isStrictlyPositive())
      return false;
    if (lower.slt(upper)) {
      bool overflow = false;
      (void)(upper - 1).sadd_ov(step, overflow);
      if (overflow)
        return false;
    }
    return llvm::all_of(loop.getBody()->without_terminator(),
                       [&](Operation &nested) {
                         return isScalarizableMapProducer(&nested, result);
                       });
  }
  return isa<arith::ConstantOp, gpu::SplatOp, gpu::BroadcastOp,
             gpu::UnaryOp, gpu::BinaryOp, gpu::CompareOp, gpu::SelectOp,
             gpu::CastOp, gpu::BitcastOp>(operation);
}

SmallVector<Value> mapProducerInputs(Operation *operation) {
  llvm::SetVector<Value> inputs;
  inputs.insert(operation->operand_begin(), operation->operand_end());
  for (Region &region : operation->getRegions())
    getUsedValuesDefinedAbove(region, inputs);
  return SmallVector<Value>(inputs.begin(), inputs.end());
}

bool hasExpensiveMapProducer(Operation *operation) {
  if (isa<scf::ForOp>(operation))
    return true;
  if (auto binary = dyn_cast<gpu::BinaryOp>(operation))
    return binary.getOperatorKind() == BinaryOperator::TrueDivide ||
           binary.getOperatorKind() == BinaryOperator::Power;
  auto unary = dyn_cast<gpu::UnaryOp>(operation);
  if (!unary)
    return false;
  switch (unary.getOperatorKind()) {
  case UnaryOperator::Exp:
  case UnaryOperator::Exp2:
  case UnaryOperator::Log:
  case UnaryOperator::Log1p:
  case UnaryOperator::Lgamma:
  case UnaryOperator::Sin:
  case UnaryOperator::Asin:
  case UnaryOperator::Cos:
  case UnaryOperator::Erf:
  case UnaryOperator::Erfc:
  case UnaryOperator::I0:
  case UnaryOperator::Tanh:
    return !unary.getApproximate();
  default:
    return false;
  }
}

void sinkSelectProducers(func::FuncOp kernel) {
  SmallVector<gpu::SelectOp> selects;
  bool changed = false;
  kernel.walk([&](gpu::SelectOp select) { selects.push_back(select); });
  // Start at consumers so nested selects stay inside one scalar callback.
  for (gpu::SelectOp select : llvm::reverse(selects)) {
    auto resultType = dyn_cast<gpu::FragmentType>(select.getType());
    if (!resultType || select->use_empty())
      continue;
    Block *block = select->getBlock();
    llvm::SmallPtrSet<Operation *, 32> slices[2];
    for (unsigned arm = 0; arm < 2; ++arm) {
      SmallVector<Value> pending{select->getOperand(arm + 1)};
      while (!pending.empty()) {
        Operation *producer = pending.pop_back_val().getDefiningOp();
        if (!producer || producer->getBlock() != block ||
            (!isa<arith::ConstantOp>(producer) &&
             !llvm::any_of(producer->getResultTypes(), [](Type type) {
               return isa<gpu::FragmentType>(type);
             })) ||
            !isScalarizableMapProducer(producer, resultType) ||
            !slices[arm].insert(producer).second)
          continue;
        llvm::append_range(pending, mapProducerInputs(producer));
      }
    }
    // Shared producers and values used outside the selected arm stay eager.
    // Only a closed, finite, effect-free slice can move under the condition.
    llvm::SmallPtrSet<Operation *, 32> exclusive[2];
    for (unsigned arm = 0; arm < 2; ++arm)
      for (Operation *producer : slices[arm])
        if (!slices[1 - arm].contains(producer))
          exclusive[arm].insert(producer);
    for (unsigned arm = 0; arm < 2; ++arm) {
      bool changed;
      do {
        SmallVector<Operation *> retained;
        for (Operation *producer : exclusive[arm])
          if (llvm::any_of(producer->getResults(), [&](Value value) {
                return llvm::any_of(value.getUses(), [&](OpOperand &use) {
                  if (use.getOwner() == select)
                    return use.getOperandNumber() != arm + 1;
                  Operation *owner = use.getOwner();
                  while (owner->getBlock() != block && owner->getParentOp())
                    owner = owner->getParentOp();
                  return !exclusive[arm].contains(owner);
                });
              }))
            retained.push_back(producer);
        for (Operation *producer : retained)
          exclusive[arm].erase(producer);
        changed = !retained.empty();
      } while (changed);
    }
    // Without branch frequencies, keep short expressions predicated. A scalar
    // callback must avoid a loop or multiple library calls to pay for control.
    auto profitable = [&](const auto &slice) {
      return llvm::any_of(slice, [](Operation *operation) {
               return isa<scf::ForOp>(operation);
             }) || llvm::count_if(slice, hasExpensiveMapProducer) >= 2;
    };
    if (!profitable(exclusive[0]) && !profitable(exclusive[1]))
      continue;

    llvm::SetVector<Value> captures;
    captures.insert(select.getCondition());
    for (unsigned arm = 0; arm < 2; ++arm) {
      auto capture = [&](Value value) {
        if (!exclusive[arm].contains(value.getDefiningOp()))
          captures.insert(value);
      };
      capture(select->getOperand(arm + 1));
      for (Operation &operation : *block)
        if (exclusive[arm].contains(&operation))
          for (Value input : mapProducerInputs(&operation))
            capture(input);
    }
    if (!llvm::all_of(captures, [&](Value value) {
          return hasMapRelation(value.getType(), resultType);
        }) || !llvm::any_of(captures, [](Value value) {
          return isa<gpu::FragmentType>(value.getType());
        }))
      continue;
    OpBuilder builder(select);
    auto map = builder.create<MapElementwiseOp>(select.getLoc(), resultType,
                                                captures.getArrayRef());
    if (Attribute origin = select->getAttr(gpu::originAttr))
      map->setAttr(gpu::originAttr, origin);
    SmallVector<Type> types;
    for (Value value : captures)
      types.push_back(gpu::scalarCallbackType(value.getType()));
    Block *body = builder.createBlock(&map.getBody(), {}, types,
        SmallVector<Location>(types.size(), select.getLoc()));
    IRMapping mapping;
    mapping.map(captures.getArrayRef(), body->getArguments());
    auto conditional = builder.create<scf::IfOp>(select.getLoc(),
        TypeRange{resultType.getElementType()},
        mapping.lookup(select.getCondition()), /*withElseRegion=*/true);
    for (unsigned arm = 0; arm < 2; ++arm) {
      Block &branch = conditional->getRegion(arm).front();
      builder.setInsertionPointToStart(&branch);
      IRMapping branchMapping(mapping);
      for (Operation &operation : *block) {
        if (!exclusive[arm].contains(&operation))
          continue;
        Operation *clone = builder.clone(operation, branchMapping);
        clone->walk([&](Operation *nested) {
          for (Value result : nested->getResults())
            result.setType(gpu::scalarCallbackType(result.getType()));
          for (Region &region : nested->getRegions())
            for (Block &block : region)
              for (BlockArgument argument : block.getArguments())
              argument.setType(gpu::scalarCallbackType(argument.getType()));
        });
      }
      builder.create<scf::YieldOp>(select.getLoc(),
          branchMapping.lookup(select->getOperand(arm + 1)));
    }
    builder.setInsertionPointToEnd(body);
    builder.create<gpu::YieldOp>(select.getLoc(), conditional.getResults());
    SmallVector<Operation *> broadcasts;
    map.walk([&](Operation *operation) {
      if (isa<gpu::SplatOp, gpu::BroadcastOp>(operation))
        broadcasts.push_back(operation);
    });
    for (Operation *broadcast : llvm::reverse(broadcasts)) {
      broadcast->getResult(0).replaceAllUsesWith(broadcast->getOperand(0));
      broadcast->erase();
    }
    select.getResult().replaceAllUsesWith(map.getResult());
    select.erase();
    changed = true;
  }
  if (changed)
    gpu::eraseDeadPhysicalValues(kernel);
}


} // namespace intent::triton::detail
