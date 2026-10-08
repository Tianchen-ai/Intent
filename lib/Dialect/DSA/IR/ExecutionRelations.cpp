#include "Intent/Dialect/DSA/IR/ExecutionRelations.h"
#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::dsa {

struct ExecutionRelations::Impl {
  Impl(func::FuncOp function, scf::ForOp taskLoop = {}, int64_t groupWidth = 0)
      : function(function), taskLoop(taskLoop), groupWidth(groupWidth),
        interface(getPublicInterface(function)), storage(function, storagePolicy()) {}

  std::optional<int64_t> integer(Value value) {
    auto range = ranges.range(value);
    auto constant = range ? range->getConstantValue() : std::nullopt;
    if (!constant || !constant->isSignedIntN(64)) return std::nullopt;
    return constant->getSExtValue();
  }

  bool taskIdentity(Value value) {
    if (!taskLoop) return false;
    while (auto divide = value.getDefiningOp<arith::DivSIOp>()) {
      if (integer(divide.getRhs()) != 1) break;
      value = divide.getLhs();
    }
    return value == taskLoop.getInductionVar();
  }

  std::optional<int64_t> groupedDivisor(Value value) {
    auto divide = value.getDefiningOp<arith::DivSIOp>();
    if (!divide || groupWidth <= 0 || !taskIdentity(divide.getLhs()))
      return std::nullopt;
    auto divisor = integer(divide.getRhs());
    return divisor && *divisor > 0 && *divisor % groupWidth == 0
               ? std::optional<int64_t>(*divisor / groupWidth)
               : std::nullopt;
  }

  ViewType readonlyView(Value value) {
    if (auto known = readonly.find(value); known != readonly.end())
      return known->second;
    auto argument = dyn_cast_or_null<BlockArgument>(storage.uniqueOrigin(value));
    ViewType result;
    if (argument && argument.getOwner() == &function.front()) {
      auto view = getPublicView(interface, argument.getArgNumber());
      if (view && view.getAccess() == 0 &&
          storage.preservesContents(function, argument))
        result = view;
    }
    readonly[value] = result;
    return result;
  }

  bool uniformLoop(scf::ForOp loop) {
    return loop != taskLoop && uniform(loop.getLowerBound()) &&
           uniform(loop.getUpperBound()) && uniform(loop.getStep());
  }

  bool uniformControl(Operation *operation) {
    for (Operation *parent = operation->getParentOp(); parent && parent != function;
         parent = parent->getParentOp()) {
      if (parent == taskLoop) return true;
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        if (!uniformLoop(loop)) return false;
      } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
        if (!uniform(branch.getCondition())) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  bool uniformForwarding(Value value) {
    Operation *owner = value.getDefiningOp();
    if (auto argument = dyn_cast<BlockArgument>(value))
      owner = argument.getOwner()->getParentOp();
    // Forwarded values also depend on which edge executes. Uniform operands
    // from two divergent branches do not necessarily denote the same value.
    if (auto loop = dyn_cast_or_null<scf::ForOp>(owner)) {
      if (!uniformLoop(loop)) return false;
    } else if (auto branch = dyn_cast_or_null<scf::IfOp>(owner)) {
      if (!uniform(branch.getCondition())) return false;
    } else {
      return false;
    }
    auto incoming = queryControlFlowIncoming(value);
    if (!incoming.complete || incoming.edges.empty()) return false;
    bool grounded = false;
    for (const auto &edge : incoming.edges) {
      if (!edge.operand) return false;
      Value source = edge.operand->get();
      if (source == value) continue;
      if (!uniform(source)) return false;
      grounded = true;
    }
    return grounded;
  }

  bool uniform(Value value) {
    if (auto known = uniforms.find(value); known != uniforms.end())
      return known->second;
    if (!activeUniforms.insert(value).second) return false;
    auto leave = llvm::make_scope_exit([&] { activeUniforms.erase(value); });
    auto infer = [&]() {
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() == &function.front()) return true;
        auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
        if (loop && argument == loop.getInductionVar()) return uniformLoop(loop);
        return uniformForwarding(value);
      }
      Operation *definition = value.getDefiningOp();
      if (!definition) return false;
      if (isa<GroupIdOp, GroupCountOp, TaskCountOp, arith::ConstantOp>(definition))
        return true;
      if (groupedDivisor(value)) return true;
      if (auto load = dyn_cast<LoadScalarOp>(definition))
        return readonlyView(load.getSource()) && uniform(load.getSource()) &&
               uniform(load.getOffset());
      if (definition->getNumRegions()) return uniformForwarding(value);
      if (isa<memref::DimOp, StrideOp>(definition) ||
          isa<arith::ArithDialect>(definition->getDialect()) ||
          isBufferStorageAliasOperation(definition))
        return llvm::all_of(definition->getOperands(),
                            [&](Value input) { return uniform(input); });
      return false;
    };
    bool result = infer();
    uniforms[value] = result;
    return result;
  }

  bool uniformValue(Value value) {
    auto difference = coefficient(value);
    return difference && *difference == 0;
  }

  bool uniformBuffer(Value value) {
    if (readonlyView(value)) return uniform(value);
    value = storage.uniqueOrigin(value);
    if (!value) return false;
    if (auto known = buffers.find(value); known != buffers.end()) return known->second;
    if (!value.getDefiningOp<memref::AllocaOp>() ||
        !activeBuffers.insert(value).second) return false;
    auto leave = llvm::make_scope_exit([&] { activeBuffers.erase(value); });
    auto infer = [&]() {
      auto aliases = storage.aliases(value);
      if (!aliases.complete) return false;
      for (Value alias : aliases.values) {
        if (alias == value) continue;
        Operation *definition = alias.getDefiningOp();
        if (!definition || definition->getNumRegions() ||
            !isBufferStorageAliasOperation(definition)) return false;
        for (Value parameter : definition->getOperands())
          if (!isa<MemRefType>(parameter.getType()) && !uniformValue(parameter))
            return false;
      }
      auto accesses = storage.accesses(value);
      if (!accesses.complete) return false;
      llvm::SmallPtrSet<Operation *, 8> checked;
      for (const auto &access : accesses.entries) {
        if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(access.effect.getEffect()))
          continue;
        if (!isa<MemoryEffects::Write>(access.effect.getEffect())) return false;
        Operation *writer = access.operation;
        if (!checked.insert(writer).second) continue;
        if (!isa<memref::StoreOp, FillOp, IotaOp, LoadTileOp, GatherRowsOp,
                 UnaryOp, BinaryOp, CastOp, SelectOp, CompareOp, CompareRangeOp,
                 CompareRampOp, IndexBinaryOp, IndexLayoutOp, TransposeOp>(writer) ||
            !uniformControl(writer)) return false;
        auto effects = storage.effects(writer);
        if (!effects.complete || effects.ordered) return false;
        for (Value operand : writer->getOperands()) {
          if (!isa<MemRefType>(operand.getType())) {
            if (!uniformValue(operand)) return false;
            continue;
          }
          bool reads = false, writes = false;
          for (const auto &entry : effects.entries)
            if (entry.effect.getValue() == operand) {
              reads |= isa<MemoryEffects::Read>(entry.effect.getEffect());
              writes |= isa<MemoryEffects::Write>(entry.effect.getEffect());
            }
          if ((!writes || reads) && !uniformBuffer(operand)) return false;
        }
      }
      return !checked.empty();
    };
    bool result = infer();
    buffers[value] = result;
    return result;
  }

  std::optional<int64_t> coefficient(Value value) {
    if (auto known = coefficients.find(value); known != coefficients.end())
      return known->second;
    if (!activeCoefficients.insert(value).second) return std::nullopt;
    auto leave = llvm::make_scope_exit([&] { activeCoefficients.erase(value); });
    auto infer = [&]() -> std::optional<int64_t> {
      if (uniform(value)) return 0;
      if (taskIdentity(value) || value.getDefiningOp<LocalIdOp>()) return 1;
      Operation *definition = value.getDefiningOp();
      if (!definition) return std::nullopt;
      if (auto remainder = dyn_cast<arith::RemSIOp>(definition)) {
        auto divisor = integer(remainder.getRhs());
        if (groupWidth > 0 && taskIdentity(remainder.getLhs()) && divisor &&
            *divisor > 0 && *divisor % groupWidth == 0) return 1;
      }
      if (auto load = dyn_cast<memref::LoadOp>(definition))
        return uniformBuffer(load.getMemref()) &&
                       llvm::all_of(load.getIndices(), [&](Value index) { return uniformValue(index); })
                   ? std::optional<int64_t>(0) : std::nullopt;
      if (auto load = dyn_cast<LoadScalarOp>(definition))
        return readonlyView(load.getSource()) && uniform(load.getSource()) &&
                       uniformValue(load.getOffset())
                   ? std::optional<int64_t>(0) : std::nullopt;
      if (!isa<arith::ArithDialect>(definition->getDialect()) || definition->getNumRegions())
        return std::nullopt;
      if (llvm::all_of(definition->getOperands(), [&](Value input) { return uniformValue(input); }))
        return 0;
      return foldIntegerDifference(describeScalarValue(value),
          [&](Value input) { return coefficient(input); },
          [&](Value input) { return integer(input); }, /*indexBitWidth=*/64);
    };
    auto result = infer();
    if (result && *result != 0 && !value.getType().isIndex() && !value.getType().isInteger(64))
      result.reset();
    coefficients[value] = result;
    return result;
  }

  CoordinateDependency dependency(Value value, BlockArgument coordinate) {
    if (value == coordinate) return CoordinateDependency::Dependent;
    auto key = std::make_pair(value, Value(coordinate));
    if (auto known = dependencies.find(key); known != dependencies.end()) return known->second;
    if (!activeDependencies.insert(key).second) return CoordinateDependency::Unknown;
    auto leave = llvm::make_scope_exit([&] { activeDependencies.erase(key); });
    auto infer = [&]() {
      Region *scope = coordinate.getOwner()->getParent();
      if (!scope->isAncestor(value.getParentRegion())) return CoordinateDependency::Independent;
      SmallVector<Value> inputs;
      if (isa<BlockArgument>(value) || value.getDefiningOp()->getNumRegions()) {
        auto incoming = queryControlFlowIncoming(value);
        if (!incoming.complete || incoming.edges.empty()) return CoordinateDependency::Unknown;
        for (const auto &edge : incoming.edges) {
          if (!edge.operand) return CoordinateDependency::Unknown;
          if (edge.operand->get() != value) inputs.push_back(edge.operand->get());
        }
        Operation *owner = value.getDefiningOp();
        if (auto argument = dyn_cast<BlockArgument>(value)) owner = argument.getOwner()->getParentOp();
        if (auto branch = dyn_cast<scf::IfOp>(owner)) inputs.push_back(branch.getCondition());
        else if (auto loop = dyn_cast<scf::ForOp>(owner))
          llvm::append_range(inputs, ValueRange{loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
        else return CoordinateDependency::Unknown;
        if (inputs.empty()) return CoordinateDependency::Unknown;
      } else {
        Operation *definition = value.getDefiningOp();
        if (!isMemoryEffectFree(definition)) return CoordinateDependency::Unknown;
        llvm::append_range(inputs, definition->getOperands());
      }
      bool unknown = false;
      for (Value input : inputs) {
        auto relation = dependency(input, coordinate);
        if (relation == CoordinateDependency::Dependent) return relation;
        unknown |= relation == CoordinateDependency::Unknown;
      }
      return unknown ? CoordinateDependency::Unknown : CoordinateDependency::Independent;
    };
    auto result = infer();
    dependencies[key] = result;
    return result;
  }

  func::FuncOp function;
  scf::ForOp taskLoop;
  int64_t groupWidth;
  InterfaceAttr interface;
  BufferStorageAnalysis storage;
  IntegerRangeAnalysis ranges;
  DenseMap<Value, ViewType> readonly;
  DenseMap<Value, bool> uniforms, buffers;
  DenseMap<Value, std::optional<int64_t>> coefficients;
  DenseSet<Value> activeUniforms, activeBuffers, activeCoefficients;
  DenseMap<std::pair<Value, Value>, CoordinateDependency> dependencies;
  DenseSet<std::pair<Value, Value>> activeDependencies;
};

ExecutionRelations::ExecutionRelations(func::FuncOp function)
    : impl(std::make_unique<Impl>(function)) {}
ExecutionRelations::ExecutionRelations(scf::ForOp taskLoop, int64_t groupWidth)
    : impl(std::make_unique<Impl>(taskLoop->getParentOfType<func::FuncOp>(), taskLoop, groupWidth)) {}
ExecutionRelations::~ExecutionRelations() = default;
bool ExecutionRelations::isUniform(Value value) { return impl->uniform(value); }
bool ExecutionRelations::hasUniformControl(Operation *operation) { return impl->uniformControl(operation); }
std::optional<int64_t> ExecutionRelations::participantCoefficient(Value value) { return impl->coefficient(value); }
std::optional<int64_t> ExecutionRelations::groupedQuotientDivisor(Value value) { return impl->groupedDivisor(value); }
ViewType ExecutionRelations::readonlyView(Value value) { return impl->readonlyView(value); }
CoordinateDependency ExecutionRelations::coordinateDependency(Value value, BlockArgument coordinate) {
  return impl->dependency(value, coordinate);
}

} // namespace intent::dsa
