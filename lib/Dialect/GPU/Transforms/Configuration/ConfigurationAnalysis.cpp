#include "ConfigurationPolicy.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu::configuration {
namespace {

bool expressionReferencesParameter(PhysicalExprAttr expression,
                                   StringAttr parameter) {
  if (expression.getKind() ==
          PhysicalExprKind::Parameter &&
      expression.getParameterReference().getName() == parameter)
    return true;
  return llvm::any_of(expression.getOperands(), [&](Attribute operand) {
    return expressionReferencesParameter(cast<PhysicalExprAttr>(operand),
                                         parameter);
  });
}

bool expressionHasParameter(PhysicalExprAttr expression) {
  if (expression.getKind() ==
      PhysicalExprKind::Parameter)
    return true;
  return llvm::any_of(expression.getOperands(), [](Attribute operand) {
    return expressionHasParameter(cast<PhysicalExprAttr>(operand));
  });
}

bool isBlockedReductionFreeAxis(func::FuncOp kernel, ParameterAttr parameter) {
  auto role = parameter.getRole();
  if (role != ParameterRole::OwnershipM && role != ParameterRole::OwnershipN)
    return false;
  StringAttr name = parameter.getName();
  bool found = false;
  kernel.walk([&](ReduceOp reduce) {
    if (found || reduce.getAxes().empty())
      return;
    // Every component contributes a live free-axis footprint. Its tuning
    // preference does not require the other components to share its schema.
    for (Value source :
         reduce.getSources()) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      bool blockedReduction = llvm::all_of(reduce.getAxes(), [&](int64_t axis) {
        if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
          return false;
        auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
        auto kind = extent.getKind();
        return (kind == PhysicalExprKind::Constant && extent.getValue() > 0) ||
               expressionHasParameter(extent);
      });
      if (!blockedReduction)
        continue;
      for (auto [axis, extent] : llvm::enumerate(fragment.getShape())) {
        if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
          continue;
        found |= expressionReferencesParameter(cast<PhysicalExprAttr>(extent),
                                               name);
      }
    }
  });
  return found;
}

bool axisReferencesParameter(FragmentType fragment, int64_t axis,
                             StringAttr parameter) {
  return axis >= 0 && axis < static_cast<int64_t>(fragment.getShape().size()) &&
         expressionReferencesParameter(
             cast<PhysicalExprAttr>(fragment.getShape()[axis]), parameter);
}

bool fragmentReferencesParameter(FragmentType fragment, StringAttr parameter) {
  return llvm::any_of(fragment.getShape(), [&](Attribute extent) {
    return expressionReferencesParameter(cast<PhysicalExprAttr>(extent),
                                         parameter);
  });
}

bool reducedAxesReferenceParameter(FragmentType fragment,
                                   ArrayRef<int64_t> axes,
                                   StringAttr parameter) {
  return llvm::any_of(axes, [&](int64_t axis) {
    return axisReferencesParameter(fragment, axis, parameter);
  });
}

bool freeAxesReferenceParameter(FragmentType fragment, ArrayRef<int64_t> axes,
                                StringAttr parameter) {
  return llvm::any_of(llvm::enumerate(fragment.getShape()), [&](auto indexed) {
    return !llvm::is_contained(axes, static_cast<int64_t>(indexed.index())) &&
           expressionReferencesParameter(
               cast<PhysicalExprAttr>(indexed.value()), parameter);
  });
}

bool valueDependsOn(Value value, Value producer,
                    llvm::DenseSet<Value> &visited) {
  if (value == producer)
    return true;
  if (!visited.insert(value).second)
    return false;

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument.getArgNumber() == 0)
      return false;
    unsigned resultIndex = argument.getArgNumber() - 1;
    if (resultIndex >= loop.getInitArgs().size())
      return false;
    if (valueDependsOn(loop.getInitArgs()[resultIndex], producer, visited))
      return true;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return valueDependsOn(yield.getOperand(resultIndex), producer, visited);
  }

  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (auto loop = dyn_cast<scf::ForOp>(definition)) {
    unsigned resultIndex = cast<OpResult>(value).getResultNumber();
    if (resultIndex >= loop.getInitArgs().size())
      return false;
    if (valueDependsOn(loop.getInitArgs()[resultIndex], producer, visited))
      return true;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return valueDependsOn(yield.getOperand(resultIndex), producer, visited);
  }
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOn(operand, producer, visited);
  });
}

bool isOnlineMomentOwnership(func::FuncOp kernel, ParameterAttr parameter) {
  auto role = parameter.getRole();
  if (role != ParameterRole::OwnershipM && role != ParameterRole::OwnershipN)
    return false;
  StringAttr name = parameter.getName();
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    if (found || !loop->hasAttr(reductionSourcesAttr) ||
        loop.getNumResults() < 3)
      return;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getNumResults())
      return;
    SmallVector<unsigned> carryingResults;
    for (auto [index, result] : llvm::enumerate(loop.getResults())) {
      auto fragment = dyn_cast<FragmentType>(result.getType());
      if (fragment && fragmentReferencesParameter(fragment, name))
        carryingResults.push_back(index);
    }
    if (carryingResults.size() != 1)
      return;
    unsigned carry = carryingResults.front();
    loop.getBody()->walk([&](ContractOp contract) {
      auto result = dyn_cast<FragmentType>(contract.getResult().getType());
      if (found || !result || !fragmentReferencesParameter(result, name))
        return;
      llvm::DenseSet<Value> visited;
      found = valueDependsOn(yield.getOperand(carry), contract.getResult(),
                             visited);
    });
  });
  return found;
}

unsigned contractionOwnershipBitWidth(func::FuncOp kernel,
                                      ParameterAttr parameter) {
  auto role = parameter.getRole();
  if (role != ParameterRole::OwnershipM && role != ParameterRole::OwnershipN)
    return 0;
  StringAttr name = parameter.getName();
  unsigned width = 0;
  auto include = [&](FragmentType result, FragmentType lhs, FragmentType rhs) {
    if (fragmentReferencesParameter(result, name))
      width = std::max({width, lhs.getElementType().getIntOrFloatBitWidth(),
                        rhs.getElementType().getIntOrFloatBitWidth()});
  };
  kernel.walk([&](ContractOp contract) {
    include(contract.getResult().getType(), contract.getLhs().getType(),
            contract.getRhs().getType());
  });
  if (!width)
    return 0;
  kernel.walk([&](ScaledContractOp contract) {
    include(contract.getResult().getType(), contract.getLhs().getType(),
            contract.getRhs().getType());
  });
  kernel.walk([&](SparseContractOp contract) {
    include(contract.getResult().getType(), contract.getCompressed().getType(),
            contract.getRhs().getType());
  });
  return width;
}

bool isMultiAxisReduction(func::FuncOp kernel, ParameterAttr parameter) {
  StringAttr name = parameter.getName();
  SmallVector<ReduceOp> reductions;
  kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp inner : reductions) {
    bool selectsAxis = llvm::any_of(
        inner.getSources(), [&](Value value) {
          auto type = dyn_cast<FragmentType>(value.getType());
          return type && reducedAxesReferenceParameter(type, inner.getAxes(), name);
        });
    if (!selectsAxis)
      continue;
    for (ReduceOp outer : reductions) {
      if (outer == inner)
        continue;
      for (Value result : inner.getResults()) {
        auto resultType = dyn_cast<FragmentType>(result.getType());
        if (!resultType)
          continue;
        for (Value source : outer.getSources()) {
          auto sourceType = dyn_cast<FragmentType>(source.getType());
          if (!sourceType)
            continue;
          bool reducesFreeAxis = llvm::any_of(outer.getAxes(), [&](int64_t axis) {
            auto mapping = cast<AxisMapAttr>(sourceType.getAxisMaps()[axis]);
            return llvm::any_of(resultType.getAxisMaps(), [&](Attribute attribute) {
              auto retained = cast<AxisMapAttr>(attribute);
              return sourceAxisIdentity(mapping) == sourceAxisIdentity(retained) &&
                     mapping.getDimensionId() == retained.getDimensionId();
            });
          });
          llvm::DenseSet<Value> visited;
          if (reducesFreeAxis && valueDependsOn(source, result, visited))
            return true;
        }
      }
    }
  }
  return false;
}

bool isStatefulReduction(func::FuncOp kernel, ParameterAttr parameter) {
  auto role = parameter.getRole();
  if (role != ParameterRole::Reduction)
    return false;
  bool found = false;
  // A chunked multi-component reduction carries one coupled accumulator state.
  // Its chunk controls that state directly, so it needs a profile distinct from
  // ordinary scalar reductions even though both use the Reduction role.
  kernel.walk([&](scf::ForOp loop) {
    auto step = queryLaunchExpression(loop.getStep());
    if (found || !step ||
        !expressionReferencesParameter(step, parameter.getName()) ||
        !loop->hasAttr(reductionSourcesAttr) || loop.getNumResults() < 2)
      return;
    loop.getBody()->walk([&](ReduceOp reduce) {
      found |= reduce.getSources().size() > 1 &&
               reduce.getNumResults() == loop.getNumResults();
    });
  });
  return found;
}

bool isRegionContractionParameter(func::FuncOp kernel, ParameterAttr parameter) {
  SmallVector<StringAttr> segments;
  for (Attribute declaration : getParameterDeclarations(kernel)) {
    auto schema = cast<ParameterAttr>(declaration);
    if (schema.getCategory() ==
            ParameterCategory::RegionContraction &&
        schema.getRole() == ParameterRole::ScanChunk)
      segments.push_back(schema.getName());
  }
  if (segments.empty())
    return false;
  bool found = false;
  bool allRegion = true;
  kernel.walk([&](Operation *operation) {
    if (!isa<ContractOp, ScaledContractOp, SparseContractOp>(operation))
      return;
    auto references = [&](Type type) {
      auto fragment = dyn_cast<FragmentType>(type);
      return fragment && fragmentReferencesParameter(
                             fragment, parameter.getName());
    };
    if (!llvm::any_of(operation->getOperandTypes(), references) &&
        !llvm::any_of(operation->getResultTypes(), references))
      return;
    found = true;
    bool inRegion = false;
    for (Operation *parent = operation->getParentOp();
         parent && parent != kernel.getOperation(); parent = parent->getParentOp())
      if (auto loop = dyn_cast<scf::ForOp>(parent))
        for (Value bound : {loop.getLowerBound(), loop.getUpperBound(),
                            loop.getStep()})
          if (auto expression = queryLaunchExpression(bound))
            inRegion |= llvm::any_of(segments, [&](StringAttr segment) {
              return expressionReferencesParameter(expression, segment);
            });
    allRegion &= inRegion;
  });
  return found && allRegion;
}

TuningClass tuningClass(func::FuncOp kernel, ParameterAttr parameter,
                       bool reductionRow, unsigned contractionWidth) {
  ParameterAttr schema = parameter;
  switch (schema.getCategory()) {
  case ParameterCategory::Reduction:
    if (isMultiAxisReduction(kernel, parameter))
      return TuningClass::MultiAxisReduction;
    if (isStatefulReduction(kernel, parameter))
      return TuningClass::StatefulReduction;
    return schema.getRole() ==
                       ParameterRole::ReductionOuter ||
                   schema.getRole() == ParameterRole::ReductionInner
               ? TuningClass::MultiAxisReduction
               : TuningClass::Reduction;
  case ParameterCategory::Scan:
    return TuningClass::Scan;
  case ParameterCategory::Contraction:
    return isRegionContractionParameter(kernel, parameter)
               ? TuningClass::RegionContraction
               : TuningClass::Contraction;
  case ParameterCategory::PersistentContraction:
    return TuningClass::PersistentContraction;
  case ParameterCategory::Histogram:
    return TuningClass::Histogram;
  case ParameterCategory::RegionReduction:
    return TuningClass::RegionReduction;
  case ParameterCategory::RegionContraction:
    return TuningClass::RegionContraction;
  case ParameterCategory::Execution:
    return TuningClass::Execution;
  case ParameterCategory::Pointwise:
    if (isOnlineMomentOwnership(kernel, parameter))
      return TuningClass::OnlineMoment;
    // A reduction retains this free-axis footprint across its chunks even
    // when a contraction produces its input. Keep the smaller row candidates.
    if (reductionRow)
      return TuningClass::PointwiseReduction;
    if (contractionWidth)
      return TuningClass::Contraction;
    return TuningClass::Pointwise;
  case ParameterCategory::Coverage:
  case ParameterCategory::Provider:
    return TuningClass::Pointwise;
  }
  llvm_unreachable("unknown physical parameter category");
}

SmallVector<CorrelatedProfileParameters>
correlatedReductionContractionParameters(
    func::FuncOp kernel, ArrayRef<ParameterAttr> parameters,
    const ConfigurationFacts &facts) {
  SmallVector<CorrelatedProfileParameters> correlated;
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || !capabilities.getMatrixUnits() ||
      facts.hasFixedPointwiseLocal)
    return correlated;

  SmallVector<ParameterAttr> pointwise;
  SmallVector<ParameterAttr> reductions;
  SmallVector<ParameterAttr> contractions;
  SmallVector<ContractOp> contracts;
  for (ParameterAttr parameter : parameters) {
    ParameterAttr schema = parameter;
    auto role = schema.getRole();
    auto kind = facts.classifications.find(parameter)->second.kind;
    if ((kind == TuningClass::Pointwise ||
         kind == TuningClass::PointwiseReduction) &&
        (role == ParameterRole::OwnershipM ||
         role == ParameterRole::OwnershipN) &&
        schema.getElementBitWidth() > 16)
      pointwise.push_back(parameter);
    if (kind == TuningClass::Reduction && role == ParameterRole::Reduction)
      reductions.push_back(parameter);
    if (kind == TuningClass::Contraction && role == ParameterRole::Reduction &&
        schema.getElementBitWidth() <= 16)
      contractions.push_back(parameter);
  }
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });

  kernel.walk([&](ReduceOp reduce) {
    if (reduce.getSources().size() != 1 || reduce.getNumResults() != 1 ||
        reduce.getAxes().empty())
      return;
    Value source = reduce.getSources().front();
    auto sourceType = dyn_cast<FragmentType>(source.getType());
    auto resultType = dyn_cast<FragmentType>(reduce.getResult(0).getType());
    if (!sourceType || !resultType)
      return;

    for (ParameterAttr pointwiseParameter : pointwise) {
      StringAttr pointwiseName = pointwiseParameter.getName();
      if (!freeAxesReferenceParameter(sourceType, reduce.getAxes(),
                                      pointwiseName) ||
          !fragmentReferencesParameter(resultType, pointwiseName))
        continue;
      for (ParameterAttr reductionParameter : reductions) {
        StringAttr reductionName = reductionParameter.getName();
        if (!reducedAxesReferenceParameter(sourceType, reduce.getAxes(),
                                           reductionName))
          continue;
        for (ContractOp contract : contracts) {
          auto lhsType = dyn_cast<FragmentType>(contract.getLhs().getType());
          auto rhsType = dyn_cast<FragmentType>(contract.getRhs().getType());
          auto contractResultType =
              dyn_cast<FragmentType>(contract.getResult().getType());
          if (!lhsType || !rhsType || !contractResultType ||
              lhsType.getShape().size() != 2 ||
              rhsType.getShape().size() != 2 ||
              !contract.getLhsBatchAxes().empty() ||
              !contract.getRhsBatchAxes().empty() ||
              contract.getLhsReductionAxes().size() != 1 ||
              contract.getRhsReductionAxes().size() != 1 ||
              !freeAxesReferenceParameter(lhsType,
                                          contract.getLhsReductionAxes(),
                                          pointwiseName) ||
              !freeAxesReferenceParameter(rhsType,
                                          contract.getRhsReductionAxes(),
                                          reductionName) ||
              !fragmentReferencesParameter(contractResultType,
                                           pointwiseName) ||
              !fragmentReferencesParameter(contractResultType, reductionName))
            continue;
          llvm::DenseSet<Value> visited;
          if (!valueDependsOn(source, contract.getResult(), visited))
            continue;
          for (ParameterAttr contractionParameter : contractions) {
            StringAttr contractionName =
                contractionParameter.getName();
            if (!reducedAxesReferenceParameter(
                    lhsType, contract.getLhsReductionAxes(), contractionName) ||
                !reducedAxesReferenceParameter(
                    rhsType, contract.getRhsReductionAxes(), contractionName))
              continue;
            CorrelatedProfileParameters profile{pointwiseParameter,
                                                reductionParameter,
                                                contractionParameter};
            if (!llvm::any_of(correlated, [&](const auto &existing) {
                  return existing.pointwise == profile.pointwise &&
                         existing.reduction == profile.reduction &&
                         existing.contraction == profile.contraction;
                }))
              correlated.push_back(profile);
          }
        }
      }
    }
  });
  return correlated;
}

bool carriesContraction(
    Value value, ArrayRef<unsigned> path, scf::ForOp scope,
    llvm::DenseMap<Value, SmallVector<SmallVector<unsigned>>> &visited) {
  auto &paths = visited[value];
  if (llvm::any_of(paths, [&](const auto &seen) {
        return ArrayRef<unsigned>(seen) == path;
      })) return false;
  paths.emplace_back(path.begin(), path.end());
  Type component = value.getType();
  for (unsigned field : path) {
    auto record = dyn_cast<RecordType>(component);
    if (!record || field >= record.getFieldTypes().size()) return false;
    component = cast<TypeAttr>(record.getFieldTypes()[field]).getValue();
  }
  auto type = dyn_cast<FragmentType>(component);
  // Follow matrix data, not a predicate derived from a contraction. In
  // particular, Select's condition does not make its other operands matrix
  // accumulators. Casts between floating element types still carry that data.
  if (!type || !isa<FloatType>(type.getElementType())) return false;
  if (auto record = value.getDefiningOp<MakeRecordOp>())
    return !path.empty() && carriesContraction(
        record.getFields()[path.front()], path.drop_front(), scope, visited);
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    SmallVector<unsigned> nested{static_cast<unsigned>(extract.getField())};
    llvm::append_range(nested, path);
    return carriesContraction(extract.getRecord(), nested, scope, visited);
  }

  Operation *definition = value.getDefiningOp();
  Operation *owner = definition ? definition : value.getParentBlock()->getParentOp();
  if (!owner || !scope->isAncestor(owner)) return false;
  if (definition && isa<ContractOp, ScaledContractOp, SparseContractOp>(definition))
    return path.empty();

  auto incoming = queryControlFlowIncoming(value);
  if (incoming.complete && !incoming.edges.empty())
    return llvm::any_of(incoming.edges, [&](const ControlFlowEdge &edge) {
      return edge.operand && carriesContraction(
          edge.operand->get(), path, scope, visited);
    });
  if (!path.empty() || !definition || definition->getNumRegions() ||
      !isMemoryEffectFree(definition)) return false;
  auto relations = queryFragmentOperandRelations(cast<OpResult>(value));
  if (failed(relations)) return false;
  return llvm::any_of(*relations, [&](const FragmentOperandRelation &relation) {
    // A reduced/broadcast statistic may depend on matrix values without
    // retaining a matrix accumulator. Require the actual data lane relation.
    return relation.invariantResultAxes.empty() &&
           relation.preservesNonUnitAxes() &&
           carriesContraction(definition->getOperand(relation.operandNumber),
                              {}, scope, visited);
  });
}

unsigned carriedMatrixCount(Value value, Type type, ArrayRef<unsigned> path,
                            scf::ForOp scope) {
  if (auto record = dyn_cast<RecordType>(type)) {
    unsigned count = 0;
    for (auto [field, attribute] : llvm::enumerate(record.getFieldTypes())) {
      SmallVector<unsigned> nested(path);
      nested.push_back(field);
      count += carriedMatrixCount(value, cast<TypeAttr>(attribute).getValue(),
                                  nested, scope);
    }
    return count;
  }
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment || fragment.getShape().size() < 2 ||
      !isa<FloatType>(fragment.getElementType())) return 0;
  llvm::DenseMap<Value, SmallVector<SmallVector<unsigned>>> visited;
  return carriesContraction(value, path, scope, visited);
}

bool hasMultipleRegionMatrixAccumulators(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    auto segment = queryParameter(loop.getStep());
    if (!segment || segment.getCategory() !=
                        ParameterCategory::RegionContraction)
      return;
    unsigned matrices = 0;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (Value value : yield.getResults())
      matrices += carriedMatrixCount(value, value.getType(), {}, loop);
    found |= matrices > 1;
  });
  return found;
}

SmallVector<ContractionFreeExtent>
contractionFreeExtents(func::FuncOp kernel, ArrayRef<ParameterAttr> parameters) {
  // An enclosing reduction's chunk can be a free axis of its inner contract.
  llvm::StringMap<ParameterAttr> freeAxisParameters;
  for (ParameterAttr parameter : parameters)
    freeAxisParameters[parameter.getName().getValue()] = parameter;
  SmallVector<ContractionFreeExtent> groups;
  kernel.walk([&](ContractOp contract) {
    SmallVector<ContractionFreeExtent> contractGroups;
    bool factorizedFreeAxes = false;
    auto collectSide = [&](Value value, ArrayRef<int64_t> reduction,
                           ArrayRef<int64_t> batch, ParameterRole role) {
      ContractionFreeExtent group;
      group.role = role;
      unsigned factors = 0;
      std::function<bool(PhysicalExprAttr)> collect = [&](PhysicalExprAttr extent) {
        auto kind = extent.getKind();
        if (kind == PhysicalExprKind::Constant) {
          factors += extent.getValue() > 1;
          return extent.getValue() > 0;
        }
        if (kind == PhysicalExprKind::Parameter) {
          auto found = freeAxisParameters.find(extent.getParameterReference().getName().getValue());
          if (found == freeAxisParameters.end()) return false;
          ++factors;
          if (!llvm::is_contained(group.parameters, found->second))
            group.parameters.push_back(found->second);
          return true;
        }
        return kind == PhysicalExprKind::Multiply &&
               llvm::all_of(extent.getOperands(), [&](Attribute operand) {
                 return collect(cast<PhysicalExprAttr>(operand));
               });
      };
      auto fragment = cast<FragmentType>(value.getType());
      // Batch lanes resident in this fragment also multiply its accumulator
      // footprint. Consume the row budget before shrinking the matrix M axis;
      // a batch distributed one per program contributes only its unit extent.
      if (role == ParameterRole::OwnershipM)
        for (int64_t axis : batch) {
          auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
          if (!collect(extent)) return;
          group.extents.push_back(extent);
        }
      for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
        if (llvm::is_contained(reduction, static_cast<int64_t>(axis)) ||
            llvm::is_contained(batch, static_cast<int64_t>(axis)))
          continue;
        auto extent = cast<PhysicalExprAttr>(attribute);
        if (!collect(extent)) return;
        group.extents.push_back(extent);
      }
      if (!group.parameters.empty()) {
        factorizedFreeAxes |= factors > 1;
        contractGroups.push_back(std::move(group));
      }
    };
    collectSide(contract.getLhs(), contract.getLhsReductionAxes(),
                contract.getLhsBatchAxes(), ParameterRole::OwnershipM);
    collectSide(contract.getRhs(), contract.getRhsReductionAxes(),
                contract.getRhsBatchAxes(), ParameterRole::OwnershipN);
    if (factorizedFreeAxes)
      for (auto &group : contractGroups)
        groups.push_back(std::move(group));
  });
  return groups;
}

bool hasSmallRegionRows(func::FuncOp kernel, ArrayRef<ParameterAttr> parameters) {
  bool found = false;
  for (ParameterAttr parameter : parameters) {
    ParameterAttr schema = parameter;
    if (schema.getCategory() !=
            ParameterCategory::RegionContraction ||
        schema.getRole() != ParameterRole::OwnershipM)
      continue;
    bool small = false;
    kernel.walk([&](MakeRangeOp range) {
      auto fragment = cast<FragmentType>(range.getResult().getType());
      if (fragment.getShape().size() != 1 ||
          !axisReferencesParameter(fragment, 0, schema.getName()))
        return;
      auto start = range.getLogicalStart().getDefiningOp<arith::ConstantIndexOp>();
      auto stop = range.getLogicalStop().getDefiningOp<arith::ConstantIndexOp>();
      if (start && stop && start.value() == 0 &&
          stop.value() > 0 && stop.value() <= 8)
        small = true;
    });
    if (!small)
      return false;
    found = true;
  }
  return found;
}

} // namespace

ConfigurationFacts analyzeConfigurationPolicy(
    func::FuncOp kernel, ArrayRef<ParameterAttr> parameters,
    const FragmentResourceAnalysis &resources) {
  ConfigurationFacts facts;
  facts.parameters.assign(parameters.begin(), parameters.end());
  for (ParameterAttr parameter : parameters) {
    auto schema = parameter;
    bool reductionRow = isBlockedReductionFreeAxis(kernel, parameter);
    unsigned contractionWidth = contractionOwnershipBitWidth(kernel, parameter);
    TuningClass kind =
        tuningClass(kernel, parameter, reductionRow, contractionWidth);
    unsigned width = schema.getElementBitWidth();
    if ((kind == TuningClass::Contraction ||
         kind == TuningClass::PersistentContraction) && contractionWidth)
      width = contractionWidth;
    facts.classifications.try_emplace(
        parameter, ParameterClassification{kind, width, reductionRow});
  }
  facts.hasTwoAxisPointwiseOwnership = llvm::any_of(
      parameters, [](ParameterAttr parameter) {
        ParameterAttr schema = parameter;
        return schema.getCategory() ==
                   ParameterCategory::Pointwise &&
               schema.getRole() ==
               ParameterRole::OwnershipM;
      });
  facts.hasFixedPointwiseLocal = llvm::any_of(
      parameters, [](ParameterAttr parameter) {
        ParameterAttr schema = parameter;
        return schema.getCategory() ==
                   ParameterCategory::Pointwise &&
               schema.getCandidates().size() == 1 &&
               parameter.getBinding().getPointwiseLocal();
      });
  facts.pointwiseOnlyProgram = true;
  kernel.walk([&](Operation *operation) {
    facts.pointwiseOnlyProgram &=
        !isa<ContractOp, ReduceOp, ScanOp, RegionFoldOp, RegionScanOp,
             ScaledContractOp, SparseContractOp, HistogramOp, ScatterReduceOp>(
            operation);
  });

  if (facts.pointwiseOnlyProgram && !facts.hasTwoAxisPointwiseOwnership)
    for (ParameterAttr parameter : parameters) {
      auto schema = parameter;
      if (schema.getCategory() != ParameterCategory::Pointwise ||
          schema.getRole() != ParameterRole::OwnershipN)
        continue;
      auto localExtent = [&](PhysicalExprAttr extent) -> std::optional<int64_t> {
        if (extent.getKind() != PhysicalExprKind::Parameter ||
            extent.getParameterReference().getName() != schema.getName())
          return std::nullopt;
        return 1;
      };
      int64_t multiplicity = 1;
      for (FragmentType fragment : resources.valueTypes()) {
        if (!fragmentReferencesParameter(fragment, schema.getName()))
          continue;
        int64_t elements = 1;
        for (Attribute dimension : fragment.getShape()) {
          auto extent = evaluatePhysicalExpression(
              cast<PhysicalExprAttr>(dimension), localExtent);
          if (!extent)
            continue;
          if (*extent <= 0 ||
              elements > std::numeric_limits<int64_t>::max() / *extent) {
            elements = 1;
            break;
          }
          elements *= *extent;
        }
        multiplicity = std::max(multiplicity, elements);
      }
      facts.pointwiseLocalMultiplicity[parameter] = multiplicity;
    }

  if (facts.pointwiseOnlyProgram)
    for (ParameterAttr parameter : parameters) {
      ParameterAttr schema = parameter;
      if (schema.getCategory() ==
              ParameterCategory::Pointwise &&
          schema.getRole() == ParameterRole::OwnershipM &&
          llvm::is_contained(schema.getCandidates().asArrayRef(), 1) &&
          llvm::any_of(schema.getCandidates().asArrayRef(),
                       [](int64_t extent) { return extent > 1; }))
        facts.pointwiseRowAxes.push_back(parameter);
    }
  SmallVector<unsigned> rowLeaders;
  for (unsigned index = 0; index < facts.pointwiseRowAxes.size(); ++index)
    rowLeaders.push_back(index);
  auto leader = [&](unsigned index) {
    while (rowLeaders[index] != index)
      index = rowLeaders[index];
    return index;
  };
  kernel.walk([&](Operation *operation) {
    for (Value result : operation->getResults()) {
      auto fragment = dyn_cast<FragmentType>(result.getType());
      if (!fragment)
        continue;
      std::optional<unsigned> first;
      for (auto [index, parameter] : llvm::enumerate(facts.pointwiseRowAxes)) {
        if (!fragmentReferencesParameter(fragment,
                                          parameter.getName()))
          continue;
        if (first)
          rowLeaders[leader(index)] = leader(*first);
        else
          first = index;
      }
    }
  });

  for (auto [index, parameter] : llvm::enumerate(facts.pointwiseRowAxes))
    facts.rowGroups[leader(index)].push_back(parameter);
  // These dataflow relations bind producer and consumer axes from one row.
  facts.correlatedProfiles =
      correlatedReductionContractionParameters(kernel, parameters, facts);

  facts.smallRegionRows = hasSmallRegionRows(kernel, parameters);
  facts.multipleRegionMatrixAccumulators =
      hasMultipleRegionMatrixAccumulators(kernel);
  facts.freeExtents = contractionFreeExtents(kernel, parameters);
  return facts;
}

} // namespace intent::gpu::configuration
