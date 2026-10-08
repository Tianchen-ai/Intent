#include "Intent/Dialect/GPU/Transforms/Mapping/UniformBranches.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
namespace intent::gpu {

scf::IfOp independentUniformBranches(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space || space.size() != 1 ||
      constantPhysicalExpression(cast<PhysicalExprAttr>(space[0])) != 1)
    return {};
  auto group = dyn_cast_or_null<ExecutionGroupOp>(
      kernel.front().getTerminator()->getPrevNode());
  auto conditional = group ? dyn_cast_or_null<scf::IfOp>(
      group.getBody().front().getTerminator()->getPrevNode()) : scf::IfOp();
  if (!conditional || conditional.getElseRegion().empty() ||
      !conditional->use_empty() ||
      !isLaunchUniformScalar(conditional.getCondition(), kernel))
    return {};
  // Branch-local tuning parameters are renamed and bound independently.
  // The dispatch condition must retain one shared ABI meaning.
  SmallVector<Value> conditions{conditional.getCondition()};
  llvm::SmallPtrSet<Operation *, 16> visited;
  bool referencesParameter = false;
  AttrTypeWalker expressions;
  expressions.addWalk([&](PhysicalExprAttr expression) {
    referencesParameter |= expression.getKind() ==
                           PhysicalExprKind::Parameter;
  });
  while (!conditions.empty()) {
    Operation *producer = conditions.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (isa<ParameterOp>(producer))
      return {};
    expressions.walk(producer->getAttrDictionary());
    if (referencesParameter)
      return {};
    llvm::append_range(conditions, producer->getOperands());
  }
  unsigned mappings = 0;
  bool hasStores[2] = {false, false};
  WalkResult eligible = kernel.walk([&](Operation *operation) {
    if (operation == kernel || operation == conditional)
      return WalkResult::advance();
    if (isa<ExecutionGroupOp>(operation)) {
      ++mappings;
      return WalkResult::advance();
    }
    if (isa<ExecutionGroupYieldOp, AssumeInBoundsOp>(operation))
      return WalkResult::advance();
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!isa<ViewType>(store.getResource().getType()))
        return WalkResult::interrupt();
      for (unsigned index = 0; index < 2; ++index)
        if (conditional->getRegion(index).isAncestor(store->getParentRegion())) {
          hasStores[index] = true;
          return WalkResult::advance();
        }
      return WalkResult::interrupt();
    }
    if (isa<scf::IfOp, scf::ForOp, scf::WhileOp, ContractOp,
            ScaledContractOp, SparseContractOp, RegionFoldOp,
            RegionScanOp>(operation))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadOp>(operation)) {
      auto view = dyn_cast<ViewType>(load.getResource().getType());
      return view && view.getAccess() == 0 ? WalkResult::advance()
                                         : WalkResult::interrupt();
    }
    return isMemoryEffectFree(operation) ? WalkResult::advance()
                                        : WalkResult::interrupt();
  });
  if (eligible.wasInterrupted() || mappings != 1 || !hasStores[0] || !hasStores[1])
    return {};
  SmallVector<PhysicalSourceAxis> reducedAxes[2];
  for (unsigned index = 0; index < 2; ++index)
    conditional->getRegion(index).walk([&](ReduceOp reduction) {
      for (Value source :
           reduction.getSources()) {
        auto type = cast<FragmentType>(source.getType());
        for (int64_t axis : reduction.getAxes()) {
          auto sourceAxis =
              sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[axis]));
          if (!llvm::is_contained(reducedAxes[index], sourceAxis))
            reducedAxes[index].push_back(sourceAxis);
        }
      }
    });
  bool sameAxes =
      reducedAxes[0].size() == reducedAxes[1].size() &&
      llvm::all_of(reducedAxes[0], [&](PhysicalSourceAxis axis) {
        return llvm::is_contained(reducedAxes[1], axis);
      });
  return sameAxes ? scf::IfOp() : conditional;
}

FailureOr<bool> realizeUniformBranches(ModuleOp module, func::FuncOp kernel,
    scf::IfOp conditional, function_ref<LogicalResult(ModuleOp)> runTransforms) {
  // Requirements describe the current whole program. This rewrite has no
  // representation for moving their control domain into either branch.
  if (auto set = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
      set && !set.getRequirements().empty())
    return false;
  // Optimize each mutually exclusive region with the existing kernel passes,
  // then rejoin their physical programs under one launch and the original ABI.
  SmallVector<OwningOpRef<ModuleOp>> branches;
  // Clone before attaching either temporary module so a later branch never
  // captures the earlier branch's transient program. OwningOpRef erases each
  // attached module on success, rejection, or pass failure.
  SmallVector<func::FuncOp> functions;
  SmallVector<scf::IfOp> choices;
  for (unsigned index = 0; index < 2; ++index) {
    IRMapping mapping;
    OwningOpRef<ModuleOp> branch(cast<ModuleOp>(module->clone(mapping)));
    (*branch)->setAttr(SymbolTable::getSymbolAttrName(),
        StringAttr::get(module.getContext(), "intent_uniform_branch_" + std::to_string(index)));
    functions.push_back(cast<func::FuncOp>(mapping.lookup(kernel.getOperation())));
    choices.push_back(cast<scf::IfOp>(mapping.lookup(conditional.getOperation())));
    branches.push_back(std::move(branch));
  }
  for (unsigned index = 0; index < 2; ++index) {
    auto &branch = branches[index];
    module.getBody()->push_back(branch->getOperation());
    func::FuncOp function = functions[index];
    scf::IfOp choice = choices[index];
    Block &body = choice->getRegion(index).front();
    for (Operation &operation :
         llvm::make_early_inc_range(body.without_terminator()))
      operation.moveBefore(choice);
    choice.erase();
    SmallVector<Attribute> effects;
    function.walk([&](StoreOp store) {
      effects.push_back(store->getAttr(originAttr));
    });
    function->setAttr(effectOriginsAttr,
                      ArrayAttr::get(module.getContext(), effects));
    eraseDeadPhysicalValues(function);
    if (failed(runTransforms(*branch)))
      return failure();
    if (auto set = function->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
        set && !set.getRequirements().empty())
      return false;
    eraseUnusedParameters(function);
    auto space = function->getAttrOfType<ArrayAttr>(programSpaceAttr);
    bool separateResources =
        function.getFunctionType() != kernel.getFunctionType();
    function.walk([&](BufferOp) { separateResources = true; });
    function.walk([&](ExecutionGroupOp mapping) {
      separateResources |=
          space.size() != 1 ||
          mapping.getSegmentLength() != space[0];
    });
    if (separateResources)
      return false;

    // Branch-local tuning symbols remain distinct after the region programs
    // are rejoined. External ABI identities stay shared.
    auto qualify = [&](StringAttr name) {
      return StringAttr::get(
          module.getContext(),
          "G" + std::to_string(index) + "_" + name.getValue().str());
    };
    if (failed(renameParameters(function, qualify))) return failure();
  }

  OwningOpRef<func::FuncOp> combinedOwner(cast<func::FuncOp>(kernel->cloneWithoutRegions()));
  func::FuncOp combined = *combinedOwner;
  combined->setAttr(parametersAttr, ArrayAttr::get(module.getContext(), {}));
  combined->removeAttr(configurationsAttr);
  SmallVector<ParameterAttr> declarations;
  for (auto &branch : branches) {
    auto function = *getPhysicalKernel(*branch);
    for (Attribute attribute : getParameterDeclarations(function))
      declarations.push_back(cast<ParameterAttr>(attribute));
  }
  // Declaration insertion is prepend, matching the former entry-block builder.
  // Import in reverse so the merged domain order remains branch0 then branch1.
  for (ParameterAttr declaration : llvm::reverse(declarations))
    if (failed(declareParameter(combined, declaration))) return failure();
  module.getBody()->push_back(combined);
  Block *entry = combined.addEntryBlock();
  OpBuilder builder(entry, entry->begin());
  IRMapping conditionMapping;
  conditionMapping.map(kernel.getArguments(), combined.getArguments());
  std::function<Value(Value)> cloneCondition = [&](Value value) -> Value {
    if (Value mapped = conditionMapping.lookupOrNull(value))
      return mapped;
    Operation *producer = value.getDefiningOp();
    for (Value operand : producer->getOperands())
      cloneCondition(operand);
    builder.clone(*producer, conditionMapping);
    return conditionMapping.lookup(value);
  };
  Value condition = cloneCondition(conditional.getCondition());
  Location location = conditional.getLoc();
  Value program = builder.create<ProgramIdOp>(location, builder.getIndexType(), 0);
  auto expression = [&](PhysicalExprKind kind, int64_t value,
                        ArrayRef<Attribute> operands = {}) {
    return PhysicalExprAttr::get(module.getContext(), kind,
        value, builder.getStringAttr(""), builder.getArrayAttr(operands));
  };
  PhysicalExprAttr zero = expression(PhysicalExprKind::Constant, 0);
  PhysicalExprAttr offset = zero;
  PhysicalExprAttr launchCondition = queryLaunchExpression(condition);
  for (auto [index, branch] : llvm::enumerate(branches)) {
    auto function = *getPhysicalKernel(*branch);
    auto space = function->getAttrOfType<ArrayAttr>(programSpaceAttr);
    auto length = cast<PhysicalExprAttr>(space[0]);
    if (launchCondition)
      length = expression(PhysicalExprKind::Select, 0,
          {launchCondition, index == 0 ? length : zero,
           index == 0 ? zero : length});
    IRMapping mapping;
    mapping.map(function.getArguments(), combined.getArguments());
    Value begin = builder.create<PhysicalExprOp>(
        location, builder.getIndexType(), offset);
    Value count = builder.create<PhysicalExprOp>(
        location, builder.getIndexType(), length);
    Value end = builder.create<BinaryOp>(
        location, builder.getIndexType(), begin, count, BinaryOperator::Add);
    Value lower = builder.create<CompareOp>(
        location, builder.getI1Type(), program, begin, ComparePredicate::Ge);
    Value upper = builder.create<CompareOp>(
        location, builder.getI1Type(), program, end, ComparePredicate::Lt);
    Value active = builder.create<BinaryOp>(
        location, builder.getI1Type(), lower, upper, BinaryOperator::LogicalAnd);
    Value selected = condition;
    if (index == 1)
      selected = builder.create<UnaryOp>(
          location, builder.getI1Type(), condition, UnaryOperator::Not);
    active = builder.create<BinaryOp>(
        location, builder.getI1Type(), active, selected, BinaryOperator::LogicalAnd);
    auto dispatch = builder.create<scf::IfOp>(location, active, false);
    OpBuilder nested = OpBuilder::atBlockBegin(&dispatch.getThenRegion().front());
    Value local = nested.create<BinaryOp>(
        location, builder.getIndexType(), program, begin, BinaryOperator::Subtract);
    for (Operation &operation : function.front().without_terminator()) {
      if (auto id = dyn_cast<ProgramIdOp>(operation)) {
        mapping.map(id.getResult(), local);
        continue;
      }
      Operation *cloned = nested.clone(operation, mapping);
      cloned->walk([&](ExecutionGroupOp group) {
        group.setGroupId(index);
        group.setSegmentOffsetAttr(offset);
        group.setSegmentLengthAttr(length);
      });
    }
    offset = index == 0 ? length
                        : expression(PhysicalExprKind::Add, 0, {offset, length});
  }
  builder.create<func::ReturnOp>(location);
  combined->setAttr(programSpaceAttr, builder.getArrayAttr({offset}));
  combined->setAttr(gridRankAttr, builder.getI64IntegerAttr(1));
  kernel.erase();
  combinedOwner.release();
  branches.clear();
  if (failed(verifyGPUProgram(module)))
    return failure();
  return true;
}

} // namespace intent::gpu
