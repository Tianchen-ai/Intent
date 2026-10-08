#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Analysis/UniformValues.h"
#include "Intent/Dialect/CPU/Analysis/RegionPredicates.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <memory>

using namespace mlir;
namespace intent::cpu {
namespace {

class UniformComputations {
public:
  explicit UniformComputations(func::FuncOp function)
      : function(function) {}

  void run(Block &block, UniformBindings initial) {
    UniformMemoryAnalysis memory(currentStorage());
    memory.setFacts(std::move(initial));
    for (Operation &operation : llvm::make_early_inc_range(block)) {
      memory.setStorageAnalysis(currentStorage());
      if (isa<linalg::FillOp>(operation)) {
        memory.visit(&operation);
        continue;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
        Attribute constant = memory.read(copy.getSource());
        memory.visit(copy.getOperation());
        if (constant) {
          OpBuilder builder(copy);
          Value scalar = builder.create<arith::ConstantOp>(copy.getLoc(),
              cast<MemRefType>(copy.getTarget().getType()).getElementType(),
              cast<TypedAttr>(constant));
          builder.create<linalg::FillOp>(copy.getLoc(), ValueRange{scalar},
                                         ValueRange{copy.getTarget()});
          copy.erase();
          storageSnapshot.reset();
        }
        continue;
      }
      if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
        if (generic.getNumResults()) {
          memory.invalidate(generic);
          continue;
        }
        auto facts = memory.visit(generic);
        // Multi-output operations contribute facts as one computation. Rewriting
        // or decomposing their coupled state is a separate legality decision.
        if (generic.getOutputs().size() != 1) continue;
        Value output = generic.getOutputs()[0];
        if (facts.canSubstituteInputs && foldCoordinateReduction(generic)) {
          storageSnapshot.reset();
          memory.setStorageAnalysis(currentStorage());
          memory.write(output, {});
          continue;
        }
        Attribute constant = facts.outputs[0];
        if (!isMatrixContraction(generic) && !constant && facts.canSubstituteInputs) {
          OpBuilder builder = OpBuilder::atBlockBegin(&generic.getRegion().front());
          for (auto [argument, input] : llvm::zip(generic.getRegion().front().getArguments(), generic.getInputs())) {
            Attribute known = facts.operands.lookup(input);
            if (!known || argument.use_empty()) continue;
            Value scalar = builder.create<arith::ConstantOp>(generic.getLoc(), argument.getType(), cast<TypedAttr>(known));
            argument.replaceAllUsesWith(scalar);
          }
        }
        if (!constant) continue;
        OpBuilder builder(generic);
        Value scalar = builder.create<arith::ConstantOp>(generic.getLoc(), cast<MemRefType>(output.getType()).getElementType(), cast<TypedAttr>(constant));
        builder.create<linalg::FillOp>(generic.getLoc(), ValueRange{scalar}, ValueRange{output});
        generic.erase();
        storageSnapshot.reset();
        continue;
      }
      if (isa<memref::AllocOp, memref::AllocaOp, memref::CastOp, memref::SubViewOp, memref::DimOp>(operation)) continue;
      if (isa<memref::DeallocOp, memref::StoreOp>(operation)) {
        memory.visit(&operation);
        continue;
      }
      // A repeated body cannot inherit a pre-loop constant for mutated storage.
      if (isa<scf::ForOp, scf::WhileOp>(operation)) memory.invalidate(&operation);
      for (Region &region : operation.getRegions())
        for (Block &nested : region) run(nested, memory.getFacts());
      memory.setStorageAnalysis(currentStorage());
      memory.invalidate(&operation);
    }
  }

private:
  StorageAnalysis &currentStorage() {
    if (!storageSnapshot) storageSnapshot = std::make_unique<StorageAnalysis>(function);
    return *storageSnapshot;
  }

  bool foldCoordinateReduction(linalg::GenericOp reduction) {
    if (reduction.getNumDpsInputs() != 1 || reduction.getNumReductionLoops() != 1 ||
        reduction.getNumLoops() != 1) return false;
    Value input = reduction.getInputs()[0], output = reduction.getOutputs()[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = cast<MemRefType>(output.getType());
    if (!inputType || inputType.getRank() != 1 || !inputType.getElementType().isInteger(1) ||
        outputType.getRank() != 0 || !outputType.getElementType().isInteger(1) ||
        !input.getDefiningOp<memref::AllocOp>()) return false;
    auto maps = reduction.getIndexingMapsArray();
    if (!maps.front().isIdentity() || maps.back().getNumResults() != 0) return false;
    Block &body = reduction.getRegion().front();
    Operation *combine = body.getTerminator()->getOperand(0).getDefiningOp();
    if (!combine || !isa<arith::AndIOp, arith::OrIOp>(combine) ||
        !((combine->getOperand(0) == body.getArgument(0) && combine->getOperand(1) == body.getArgument(1)) ||
          (combine->getOperand(1) == body.getArgument(0) && combine->getOperand(0) == body.getArgument(1)))) return false;
    bool disjunction = isa<arith::OrIOp>(combine);

    linalg::GenericOp predicate;
    for (Operation *user : input.getUsers()) {
      auto writer = dyn_cast<linalg::GenericOp>(user);
      if (!writer || !llvm::is_contained(writer.getOutputs(), input)) continue;
      if (predicate) return false;
      predicate = writer;
    }
    if (!predicate || predicate.getNumResults() || predicate.getOutputs().size() != 1 ||
        predicate.getNumReductionLoops() || predicate.getNumLoops() != 1 ||
        !predicate.getIndexingMapsArray().back().isIdentity() ||
        !currentStorage().unchangedBetween(input, predicate, reduction)) return false;
    Block &predicateBody = predicate.getRegion().front();
    if (!predicateBody.getArguments().back().use_empty() ||
        llvm::any_of(predicateBody.without_terminator(), [](Operation &operation) {
          return operation.getNumRegions() || !isMemoryEffectFree(&operation);
        })) return false;
    auto comparison = predicateBody.getTerminator()->getOperand(0).getDefiningOp<arith::CmpIOp>();
    if (!comparison || !matchPattern(comparison.getRhs(), m_Zero())) return false;
    auto coordinate = dyn_cast<BlockArgument>(comparison.getLhs());
    if (!coordinate || coordinate.getOwner() != &predicateBody ||
        coordinate.getArgNumber() >= predicate.getNumDpsInputs() ||
        !predicate.getIndexingMapsArray()[coordinate.getArgNumber()].isIdentity()) return false;
    auto sequence = coordinateSequence(predicate.getInputs()[coordinate.getArgNumber()], predicate);
    if (!sequence || sequence->origin) return false;
    if (comparison.getPredicate() == arith::CmpIPredicate::sge && sequence->nonnegativeOffsets) {
      if (!disjunction) {
        // Every member is true; conjunction leaves the incoming state intact,
        // including an empty domain.
        reduction.erase();
        return true;
      }
    } else if (comparison.getPredicate() != arith::CmpIPredicate::eq || !sequence->startsAtOrigin) {
      return false;
    }
    OpBuilder builder(reduction);
    Location loc = reduction.getLoc();
    Value extent = builder.create<memref::DimOp>(loc, input, 0);
    Value bound = builder.create<arith::ConstantIndexOp>(loc, disjunction ? 0 : 1);
    Value aggregate = builder.create<arith::CmpIOp>(loc,
        disjunction ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sle, extent, bound);
    Value initial = builder.create<memref::LoadOp>(loc, output, ValueRange{});
    Value result = disjunction ? Value(builder.create<arith::OrIOp>(loc, initial, aggregate))
                               : Value(builder.create<arith::AndIOp>(loc, initial, aggregate));
    builder.create<memref::StoreOp>(loc, result, output, ValueRange{});
    reduction.erase();
    return true;
  }

  func::FuncOp function;
  std::unique_ptr<StorageAnalysis> storageSnapshot;
};

}

LogicalResult foldUniformComputations(func::FuncOp function) {
  UniformComputations(function).run(function.front(), UniformBindings());
  return success();
}

}
