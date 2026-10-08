#include "Intent/Dialect/CPU/Transforms/Collective/Collectives.h"
#include "Intent/Dialect/CPU/Transforms/Control/Traversals.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "Intent/Dialect/CPU/Transforms/Task/Tasks.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Legalize.h"
#include "Quantization.h"
#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Bufferization.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::mojo {
namespace {

bool directAtomicAdd(Type element) {
  return element.isF32() || element.isF64() || element.isSignlessInteger(32) || element.isSignlessInteger(64);
}

bool needsFloatingPointEnvironment(Operation *scope) {
  auto floating = [](Type type) {
    if (auto vector = dyn_cast<VectorType>(type)) type = vector.getElementType();
    return isa<FloatType>(type);
  };
  return scope->walk([&](Operation *operation) {
    // Calls may depend on the caller's FP state even with an integer-only ABI.
    if (isa<func::CallOp, cpu::InvokeOp>(operation) ||
        llvm::any_of(operation->getOperandTypes(), floating) ||
        llvm::any_of(operation->getResultTypes(), floating))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }).wasInterrupted();
}

void promotePrivateScratch(func::FuncOp function, int64_t budget) {
  cpu::PhysicalProgramAnalysis physical(function);
  auto allocations = physical.allocations();
  llvm::DenseMap<Operation *, int64_t> remaining;
  llvm::DenseSet<Operation *> unavailable;
  auto alignmentOf = [](auto allocation) -> int64_t {
    Type element = allocation.getType().getElementType();
    int64_t bytes = element.isIndex() ? 8 : (element.getIntOrFloatBitWidth() + 7) / 8;
    return allocation.getAlignment().value_or(bytes);
  };
  auto fits = [](int64_t bytes, int64_t alignment, int64_t available) {
    return bytes >= 0 && bytes <= available && alignment > 0 && alignment - 1 <= available - bytes;
  };
  // Count every static slot, including mutually exclusive lifetimes, to bound
  // each automatic allocation scope without relying on stack coloring. Task
  // dispatches have separate worker frames from their enclosing function.
  for (auto facts : allocations) {
    if (!facts.stack) continue;
    auto allocation = facts.value.getDefiningOp<memref::AllocaOp>();
    Operation *owner = allocation->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (!owner || unavailable.contains(owner)) continue;
    int64_t &available = remaining.try_emplace(owner, budget).first->second;
    int64_t alignment = alignmentOf(allocation);
    if (!facts.bytes || !fits(*facts.bytes, alignment, available)) {
      unavailable.insert(owner);
      continue;
    }
    available -= *facts.bytes + alignment - 1;
  }
  for (auto facts : allocations) {
    auto allocation = facts.value.getDefiningOp<memref::AllocOp>();
    if (!allocation || !facts.bytes || *facts.bytes <= 0 || *facts.bytes > 1024 ||
        !allocation.getType().getLayout().isIdentity()) continue;
    Operation *owner = allocation->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (!owner || unavailable.contains(owner)) continue;
    int64_t &available = remaining.try_emplace(owner, budget).first->second;
    int64_t alignment = alignmentOf(allocation);
    if (!fits(*facts.bytes, alignment, available)) continue;
    cpu::StorageAnalysis storage(function);
    auto lifetime = storage.lifetime(allocation);
    if (!lifetime || !lifetime->aliases.complete) continue;
    if (!llvm::all_of(lifetime->aliases.users, [&](Operation *user) {
          return user == lifetime->end || storage.effects(user).complete;
        })) continue;
    OpBuilder builder(allocation);
    auto stack = builder.create<memref::AllocaOp>(allocation.getLoc(), allocation.getType(),
        ValueRange{}, allocation.getAlignmentAttr());
    lifetime->end.erase();
    allocation.replaceAllUsesWith(stack.getResult());
    allocation.erase();
    available -= *facts.bytes + alignment - 1;
  }
}

LogicalResult expandAtomicUpdates(ModuleOp module) {
  SmallVector<memref::GenericAtomicRMWOp> genericUpdates;
  module.walk([&](memref::GenericAtomicRMWOp operation) { genericUpdates.push_back(operation); });
  for (auto operation : genericUpdates) {
    if (!operation.getResult().use_empty())
      return operation.emitError("Mojo generic atomic update currently requires a discarded scatter result");
    OpBuilder builder(operation);
    Location loc = operation.getLoc();
    Block &body = operation.getRegion().front();
    Value yielded = cast<memref::AtomicYieldOp>(body.getTerminator()).getResult();
    auto *combine = yielded.getDefiningOp();
    if (combine && isa<arith::AddFOp, arith::AddIOp>(combine) && directAtomicAdd(yielded.getType()) &&
        std::distance(body.begin(), body.end()) == 2 &&
        llvm::is_contained(combine->getOperands(), body.getArgument(0))) {
      Value value = combine->getOperand(combine->getOperand(0) == body.getArgument(0) ? 1 : 0);
      if (value != body.getArgument(0)) {
        builder.create<cpu::AtomicRMWOp>(loc, value.getType(), operation.getMemref(), value,
            operation.getIndices(), AtomicOrdering::Relaxed, AtomicRMWKind::Add, false);
        operation.erase();
        continue;
      }
    }
    Type type = operation.getResult().getType();
    Value initial = builder.create<cpu::AtomicLoadOp>(loc, type, operation.getMemref(), operation.getIndices(), AtomicOrdering::Relaxed);
    Value pending = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    auto retry = builder.create<scf::WhileOp>(loc, TypeRange{type, builder.getI1Type()}, ValueRange{initial, pending});
    auto *before = builder.createBlock(&retry.getBefore(), {}, {type, builder.getI1Type()}, {loc, loc});
    Value no = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    Value again = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, before->getArgument(1), no);
    builder.create<scf::ConditionOp>(loc, again, before->getArguments());
    auto *after = builder.createBlock(&retry.getAfter(), {}, {type, builder.getI1Type()}, {loc, loc});
    IRMapping mapping;
    mapping.map(body.getArgument(0), after->getArgument(0));
    for (Operation &instruction : body.without_terminator()) builder.clone(instruction, mapping);
    auto exchange = builder.create<cpu::AtomicCompareExchangeOp>(loc, type, builder.getI1Type(), operation.getMemref(),
        after->getArgument(0), mapping.lookupOrDefault(yielded), operation.getIndices(), AtomicOrdering::Relaxed);
    builder.create<scf::YieldOp>(loc, exchange.getResults());
    operation.erase();
  }
  SmallVector<cpu::AtomicRMWOp> updates;
  module.walk([&](cpu::AtomicRMWOp operation) {
    if (operation.getKind() != AtomicRMWKind::Add || !directAtomicAdd(operation.getValue().getType()))
      updates.push_back(operation);
  });
  for (auto operation : updates) {
    OpBuilder builder(operation);
    Location loc = operation.getLoc();
    Type type = operation.getValue().getType();
    Value initial = builder.create<cpu::AtomicLoadOp>(loc, type, operation.getTarget(), operation.getIndices(), AtomicOrdering::Relaxed);
    Value pending = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    auto retry = builder.create<scf::WhileOp>(loc, TypeRange{type, builder.getI1Type()}, ValueRange{initial, pending});
    auto *before = builder.createBlock(&retry.getBefore(), {}, {type, builder.getI1Type()}, {loc, loc});
    Value no = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    Value again = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, before->getArgument(1), no);
    builder.create<scf::ConditionOp>(loc, again, before->getArguments());
    auto *after = builder.createBlock(&retry.getAfter(), {}, {type, builder.getI1Type()}, {loc, loc});
    Value old = after->getArgument(0), value = operation.getValue(), desired;
    switch (operation.getKind()) {
    case AtomicRMWKind::Exchange: desired = value; break;
    case AtomicRMWKind::Maximum:
      desired = isa<FloatType>(type) ? Value(builder.create<arith::MaximumFOp>(loc, old, value))
          : operation.getUnsignedInteger() ? Value(builder.create<arith::MaxUIOp>(loc, old, value))
          : Value(builder.create<arith::MaxSIOp>(loc, old, value)); break;
    case AtomicRMWKind::Minimum:
      desired = isa<FloatType>(type) ? Value(builder.create<arith::MinimumFOp>(loc, old, value))
          : operation.getUnsignedInteger() ? Value(builder.create<arith::MinUIOp>(loc, old, value))
          : Value(builder.create<arith::MinSIOp>(loc, old, value)); break;
    case AtomicRMWKind::BitwiseAnd: desired = builder.create<arith::AndIOp>(loc, old, value); break;
    case AtomicRMWKind::BitwiseOr: desired = builder.create<arith::OrIOp>(loc, old, value); break;
    case AtomicRMWKind::BitwiseXor: desired = builder.create<arith::XOrIOp>(loc, old, value); break;
    case AtomicRMWKind::Add:
      desired = isa<FloatType>(type) ? Value(builder.create<arith::AddFOp>(loc, old, value))
                                    : Value(builder.create<arith::AddIOp>(loc, old, value)); break;
    }
    auto exchange = builder.create<cpu::AtomicCompareExchangeOp>(loc, type, builder.getI1Type(),
        operation.getTarget(), old, desired, operation.getIndices(), operation.getOrdering());
    builder.create<scf::YieldOp>(loc, exchange.getResults());
    operation.getOldValue().replaceAllUsesWith(retry.getResult(0));
    operation.erase();
  }
  return success();
}

LogicalResult finishNativeProgram(ModuleOp module) {
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    cpu::localizeTaskConstants(function);
  if (failed(verifySourceProgram(module))) return failure();
  // Source legality closes every implementation's expansion. The executable
  // loops, vectors and resources now own the selected behavior; retained
  // candidate summaries remain available for artifact inspection.
  module.walk([](Operation *operation) {
    operation->removeAttr("intent_cpu.implementation");
  });
  return success();
}

}

LogicalResult prepareNativeProgram(ModuleOp module) {
  if (failed(cpu::lowerOwnership(module))) return failure();
  auto capabilities = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  if (!capabilities || (capabilities.getVectorBits() != 256 && capabilities.getVectorBits() != 512))
    return module.emitError("Mojo native currently requires an AVX2 or AVX512 CPU capability");
  auto implementations = cpu::lookupImplementationProvider(module, "mojo");
  if (failed(implementations)) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(cpu::materializeTaskDispatches(function))) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto programBinding = function->getAttrOfType<cpu::ImplementationAttr>("intent_cpu.implementation");
    if (!programBinding)
      return function.emitError("Mojo materialization requires an explicit program implementation");
    if (failed(materializeQuantizedComputations(function, **implementations)) ||
        failed(materializeRegisterContractions(function)) ||
        failed(cpu::materializeStructuredComputations(function, [&](Operation *operation) {
          // Preparation and fills have no source computation to inherit from.
          // The explicitly selected program implementation owns these loops.
          if (!operation->hasAttr("intent_cpu.implementation"))
            operation->setAttr("intent_cpu.implementation", programBinding);
          return (**implementations).materialize(operation);
        }))) return failure();
    // Implementation supplies may introduce additional independent worksets.
    // Form their worker intervals before terminal dispatch, keeping the same
    // grain-one assignment without repeating scheduling inside each callback.
    SmallVector<scf::ParallelOp> worksets;
    function.walk([&](scf::ParallelOp workset) {
      if (!workset->getParentOfType<scf::ParallelOp>()) worksets.push_back(workset);
    });
    for (scf::ParallelOp workset : worksets)
      if (failed(cpu::partitionWorkset(workset, 1))) return failure();
    if (failed(cpu::isolateTasks(function)) ||
        failed(cpu::materializeTaskDispatches(function))) return failure();
  }
  return success();
}

LogicalResult fusePrivateComputations(ModuleOp module) {
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(cpu::fuseIntermediateBuffers(function))) return failure();
  return success();
}

LogicalResult vectorizeNativeProgram(ModuleOp module, bool fuseTraversals) {
  auto implementations = cpu::lookupImplementationProvider(module, "mojo");
  if (failed(implementations)) return failure();
  auto capabilities = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto programBinding = function->getAttrOfType<cpu::ImplementationAttr>("intent_cpu.implementation");
    if (!programBinding)
      return function.emitError("Mojo vectorization requires an explicit program implementation");
    function.walk([&](scf::ForOp loop) {
      if (!loop->hasAttr("intent_cpu.implementation"))
        loop->setAttr("intent_cpu.implementation", programBinding);
    });
    if (fuseTraversals && failed(cpu::fuseSharedTraversals(function))) return failure();
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    auto configuration = function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration");
    for (scf::ForOp loop : loops) {
      auto binding = loop->getAttrOfType<cpu::ImplementationAttr>("intent_cpu.implementation");
      auto implementation = (**implementations).verifyBinding(binding, capabilities, configuration, loop);
      if (failed(implementation)) return failure();
      if (!(*implementation)->vectorize)
        return loop.emitError("selected Mojo implementation has no loop vectorization");
      if (failed((*implementation)->vectorize(loop, binding))) return failure();
    }
  }
  return success();
}

LogicalResult finalizeNativeProgram(ModuleOp module) {
  auto capabilities = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  if (!capabilities)
    return module.emitError("Mojo finalization requires CPU capabilities");
  SmallVector<arith::AddFOp> additions;
  module.walk([&](arith::AddFOp operation) { additions.push_back(operation); });
  for (auto addition : additions) {
    if ((addition.getFastmath() & arith::FastMathFlags::contract) == arith::FastMathFlags::none)
      continue;
    for (unsigned side = 0; side != 2; ++side) {
      auto product = addition->getOperand(side).getDefiningOp<arith::MulFOp>();
      if (!product || !product->hasOneUse() || product->getBlock() != addition->getBlock() ||
          product.getType() != addition.getType() ||
          (product.getFastmath() & arith::FastMathFlags::contract) == arith::FastMathFlags::none)
        continue;
      OpBuilder builder(addition);
      auto fused = builder.create<math::FmaOp>(addition.getLoc(), product.getLhs(),
          product.getRhs(), addition->getOperand(1 - side));
      fused.setFastmath(addition.getFastmath() & product.getFastmath());
      addition.replaceAllUsesWith(fused.getResult());
      addition.erase();
      product.erase();
      break;
    }
  }
  SmallVector<math::RsqrtOp> roots;
  module.walk([&](math::RsqrtOp operation) { roots.push_back(operation); });
  for (auto operation : roots) {
    OpBuilder b(operation);
    Type type = operation.getType();
    TypedAttr one;
    if (auto vector = dyn_cast<VectorType>(type))
      one = DenseElementsAttr::get(vector, b.getFloatAttr(vector.getElementType(), 1.0));
    else one = b.getFloatAttr(type, 1.0);
    Value root = b.create<math::SqrtOp>(operation.getLoc(), operation.getOperand());
    Value unit = b.create<arith::ConstantOp>(operation.getLoc(), type, one);
    Value result = b.create<arith::DivFOp>(operation.getLoc(), unit, root);
    operation.getResult().replaceAllUsesWith(result);
    operation.erase();
  }
  RewritePatternSet integerDivision(module.getContext());
  arith::populateCeilFloorDivExpandOpsPatterns(integerDivision);
  if (failed(applyPatternsGreedily(module, std::move(integerDivision)))) return failure();
  if (failed(expandAtomicUpdates(module))) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (failed(cpu::optimizeMemoryAccesses(function)) ||
        failed(cpu::reuseScratchStorage(
            function, cpu::ScratchRepresentation::LinearCapacity))) return failure();
    promotePrivateScratch(function, capabilities.getPrivateBytes());
  }
  SmallVector<Block *> scopes;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    if (needsFloatingPointEnvironment(function)) scopes.push_back(&function.front());
    function.walk([&](cpu::TaskDispatchOp dispatch) {
      if (needsFloatingPointEnvironment(dispatch)) scopes.push_back(&dispatch.getBody().front());
    });
  }
  if (scopes.empty()) return finishNativeProgram(module);
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto enter = builder.create<func::FuncOp>(module.getLoc(), "intent_cpu_enter_ieee",
      builder.getFunctionType({}, {builder.getI32Type()}));
  auto leave = builder.create<func::FuncOp>(module.getLoc(), "intent_cpu_leave_ieee",
      builder.getFunctionType({builder.getI32Type()}, {}));
  enter.setPrivate(); leave.setPrivate();
  enter->setAttr("cpu.external_runtime", builder.getUnitAttr());
  leave->setAttr("cpu.external_runtime", builder.getUnitAttr());
  for (Block *scope : scopes) {
    builder.setInsertionPointToStart(scope);
    Value previous = builder.create<func::CallOp>(module.getLoc(), enter, ValueRange{}).getResult(0);
    builder.setInsertionPoint(scope->getTerminator());
    builder.create<func::CallOp>(module.getLoc(), leave, ValueRange{previous});
  }
  return finishNativeProgram(module);
}

}
