#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Bufferization.h"
#include "CollectiveBufferization.h"
#include "../Structure/ProducerReuse.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace intent::cpu {
namespace {

constexpr StringLiteral readBorrowedAttr = "intent_cpu.borrowed";

bool borrowExternalInput(ReadOp operation) {
  auto decision = operation->getAttrOfType<BoolAttr>(readBorrowedAttr);
  assert(decision && "CPU read bufferization requires an explicit borrowing decision");
  return decision.getValue();
}

void classifyReadStorage(ModuleOp module) {
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    StorageAnalysis storage(function);
    auto requirements = function->getAttrOfType<EntryRequirementsAttr>(
        entryRequirementsAttr);
    function.walk([&](ReadOp read) {
      auto source = storage.externalView(read.getSource());
      // Inputs may alias other inputs, but their tensor values are immutable
      // and the native entry excludes writable aliases. Record the decision
      // before native bufferization starts rebuilding structured regions.
      bool borrow = source && source.getAccess() == 0 && requirements &&
                    requirements.getDisjointOutputs();
      read->setAttr(readBorrowedAttr, BoolAttr::get(module.getContext(), borrow));
    });
  }
}

struct ReadBufferization final
    : bufferization::BufferizableOpInterface::ExternalModel<ReadBufferization,
                                                            ReadOp> {
  bool isWritable(Operation *operation, Value,
                  const bufferization::AnalysisState &) const {
    return !borrowExternalInput(cast<ReadOp>(operation));
  }

  FailureOr<BaseMemRefType> getBufferType(
      Operation *operation, Value,
      const bufferization::BufferizationOptions &,
      SmallVector<Value> &) const {
    auto read = cast<ReadOp>(operation);
    auto result = read.getResult().getType();
    auto memory = cast<MemRefType>(read.getSource().getType());
    return MemRefType::get(result.getShape(), result.getElementType(),
        borrowExternalInput(read) ? memory.getLayout() : MemRefLayoutAttrInterface(),
        memory.getMemorySpace());
  }

  LogicalResult bufferize(
      Operation *operation, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options) const {
    auto read = cast<ReadOp>(operation);
    SmallVector<Value> invocationStack;
    auto bufferType = getBufferType(operation, read.getResult(), options,
                                    invocationStack);
    if (failed(bufferType)) return failure();
    auto type = cast<MemRefType>(*bufferType);
    Value result = read.getSource();
    if (borrowExternalInput(read)) {
      if (result.getType() != type)
        result = rewriter.create<memref::CastOp>(read.getLoc(), type, result);
    } else {
      SmallVector<Value> dimensions;
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        if (type.isDynamicDim(axis))
          dimensions.push_back(rewriter.create<memref::DimOp>(read.getLoc(),
                                                             result, axis));
      auto allocation = options.createAlloc(rewriter, read.getLoc(), type,
                                            dimensions);
      if (failed(allocation)) return failure();
      if (failed(options.createMemCpy(rewriter, read.getLoc(), result,
                                     *allocation))) return failure();
      result = *allocation;
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, operation, result);
    return success();
  }
};

struct WriteBufferization final
    : bufferization::BufferizableOpInterface::ExternalModel<WriteBufferization,
                                                            WriteOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                             const bufferization::AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return false;
  }

  bufferization::AliasingValueList getAliasingValues(
      Operation *, OpOperand &, const bufferization::AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(
      Operation *operation, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options) const {
    auto write = cast<WriteOp>(operation);
    auto source = bufferization::getBuffer(rewriter, write.getSource(), options);
    if (failed(source) ||
        failed(options.createMemCpy(rewriter, write.getLoc(), *source,
                                    write.getDestination()))) return failure();
    rewriter.eraseOp(operation);
    return success();
  }
};

// The native ownership driver visits each helper block before its parent. It
// initially gives memref formals ownership arguments. A CPU helper borrows all
// inputs and destination slots, so those indicators are statically false; its
// own allocations are still released by the native terminator processing.
template <typename Op>
struct BorrowedHelperOwnership final
    : bufferization::BufferDeallocationOpInterface::ExternalModel<
          BorrowedHelperOwnership<Op>, Op> {
  FailureOr<Operation *> process(
      Operation *operation, bufferization::DeallocationState &,
      const bufferization::DeallocationOptions &) const {
    for (Region &region : operation->getRegions()) {
      if (!llvm::hasSingleElement(region))
        return operation->emitError("CPU helper ownership requires one block"),
               failure();
      Block &body = region.front();
      unsigned count = llvm::count_if(body.getArgumentTypes(), [](Type type) {
        return isa<BaseMemRefType>(type);
      });
      unsigned originalCount = body.getNumArguments() - count;
      if (!llvm::all_of(TypeRange(body.getArgumentTypes()).take_back(count),
                       [](Type type) { return type.isInteger(1); }))
        return operation->emitError(
            "CPU helper ownership arguments do not match the native protocol"),
               failure();
      if (!count) continue;
      OpBuilder builder = OpBuilder::atBlockBegin(&body);
      Value borrowed = builder.create<arith::ConstantOp>(operation->getLoc(),
                                                        builder.getBoolAttr(false));
      for (BlockArgument argument : body.getArguments().take_back(count))
        argument.replaceAllUsesWith(borrowed);
      body.eraseArguments(originalCount, count);
    }
    return operation;
  }
};

template <typename Op>
struct HelperReturnOwnership final
    : bufferization::BufferDeallocationOpInterface::ExternalModel<
          HelperReturnOwnership<Op>, Op> {
  FailureOr<Operation *> process(
      Operation *operation, bufferization::DeallocationState &state,
      const bufferization::DeallocationOptions &) const {
    SmallVector<Value> ownerships;
    return bufferization::deallocation_impl::insertDeallocOpForReturnLike(
        state, operation, {}, ownerships);
  }
};

template <typename... Ops>
void registerHelperOwnership(MLIRContext *context) {
  (Ops::template attachInterface<BorrowedHelperOwnership<Ops>>(*context), ...);
}

template <typename... Ops>
void registerHelperReturns(MLIRContext *context) {
  (Ops::template attachInterface<HelperReturnOwnership<Ops>>(*context), ...);
}

LogicalResult noTensorValues(ModuleOp module) {
  auto status = module.walk([&](Operation *operation) {
    auto tensor = [](Type type) { return isa<TensorType>(type); };
    if (llvm::any_of(operation->getOperandTypes(), tensor) ||
        llvm::any_of(operation->getResultTypes(), tensor)) {
      operation->emitError("CPU value bufferization left an unresolved tensor");
      return WalkResult::interrupt();
    }
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        if (llvm::any_of(block.getArgumentTypes(), tensor)) {
          operation->emitError("CPU value bufferization left a tensor formal");
          return WalkResult::interrupt();
        }
    return WalkResult::advance();
  });
  return failure(status.wasInterrupted());
}

LogicalResult lowerLexicalReleases(ModuleOp module) {
  bufferization::DeallocHelperMap helpers;
  RewritePatternSet patterns(module.getContext());
  bufferization::populateBufferizationDeallocLoweringPattern(patterns, helpers);
  ConversionTarget target(*module.getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addDynamicallyLegalOp<bufferization::DeallocOp>(
      [](bufferization::DeallocOp operation) {
        return operation.getMemrefs().size() != 1 ||
               !operation.getRetained().empty() ||
               !operation.getMemrefs()[0].getDefiningOp<memref::AllocOp>() ||
               !matchPattern(operation.getConditions()[0], m_One());
      });
  return applyPartialConversion(module, target, std::move(patterns));
}

LogicalResult expandReshapes(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  memref::populateExpandOpsPatterns(patterns);
  ConversionTarget target(*module.getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<memref::ReshapeOp>();
  return applyPartialConversion(module, target, std::move(patterns));
}

} // namespace

void registerValueBufferizationInterfaces(DialectRegistry &registry) {
  registerCollectiveBufferizationInterfaces(registry);
  registry.addExtension(+[](MLIRContext *context, IntentCPUDialect *) {
    ReadOp::attachInterface<ReadBufferization>(*context);
    WriteOp::attachInterface<WriteBufferization>(*context);
    registerHelperOwnership<ReduceOp, SliceReduceOp, ScanOp, RegionFoldOp,
                            RegionScanOp>(context);
    registerHelperReturns<YieldOp, SliceReduceYieldOp, ScanYieldOp,
                          RegionYieldOp>(context);
  });
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    // The atomic update region exchanges only scalars. Its destination remains
    // borrowed by the update; the scalar yield transfers no buffer ownership.
    registerHelperOwnership<memref::GenericAtomicRMWOp>(context);
    registerHelperReturns<memref::AtomicYieldOp>(context);
  });
}

LogicalResult bufferizeValues(ModuleOp module) {
  PassManager normalize(module.getContext());
  normalize.addPass(createCanonicalizerPass());
  normalize.addPass(createCSEPass());
  if (failed(normalize.run(module))) return failure();
  RewritePatternSet fusion(module.getContext());
  linalg::populateElementwiseOpsFusionPatterns(fusion, [](OpOperand *operand) {
    auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
    auto consumer = dyn_cast<linalg::GenericOp>(operand->getOwner());
    if (!producer || !consumer || consumer.getOutputs().size() != 1 ||
        consumer.getNumReductionLoops()) return false;
    // Snapshot reads remain at their original execution point. The standard
    // tensor fusion legality checks indexing, but does not protect captured
    // mutable memref reads in a generic operation's payload.
    if (producer.getRegion().walk([](Operation *nested) {
      return isMemoryEffectFree(nested) ? WalkResult::advance()
                                       : WalkResult::interrupt();
    }).wasInterrupted()) return false;
    // Preserving producer results would create a new multi-output generic.
    // That representation is not supported by every CPU implementation.
    if (!linalg::getPreservedProducerResults(producer, consumer, operand).empty())
      return false;
    auto result = cast<OpResult>(operand->get());
    unsigned resultNumber = result.getResultNumber();
    AffineMap outputMap = producer.getMatchingIndexingMap(
        producer.getDpsInitOperand(resultNumber));
    if (!outputMap.isPermutation()) return false;
    AffineMap coordinates = inversePermutation(outputMap).compose(
        consumer.getMatchingIndexingMap(operand));
    StorageAnalysis storage(producer->getParentOfType<func::FuncOp>());
    auto payload = analyzeProducerResult(producer, storage, resultNumber);
    if (failed(payload)) return false;
    bool removesProducer = llvm::all_of(producer->getResults(), [&](Value value) {
      return llvm::all_of(value.getUses(), [&](OpOperand &use) { return &use == operand; });
    });
    Value yielded = producer.getRegion().front().getTerminator()->getOperand(resultNumber);
    return shouldFuseProducerResult(*payload, yielded, coordinates,
                                    consumer.getStaticLoopRanges(), removesProducer);
  });
  if (failed(applyPatternsGreedily(module, std::move(fusion)))) return failure();
  classifyReadStorage(module);
  normalize.clear();
  normalize.addPass(bufferization::createEmptyTensorEliminationPass());
  normalize.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  if (failed(normalize.run(module))) return failure();

  bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  options.allowReturnAllocsFromLoops = true;
  if (failed(bufferization::runOneShotBufferize(module, options))) return failure();

  PassManager cleanup(module.getContext());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  if (failed(cleanup.run(module)) || failed(noTensorValues(module)) ||
      failed(expandReshapes(module))) return failure();

  // Recompute releases for the complete program, including author-owned mutable
  // buffers, tensor temporaries and memrefs transferred by structured control.
  SmallVector<memref::DeallocOp> releases;
  module.walk([&](memref::DeallocOp release) { releases.push_back(release); });
  for (auto release : releases) release.erase();
  for (auto function : module.getOps<func::FuncOp>())
    if (!function.isExternal() &&
        failed(bufferization::deallocateBuffersOwnershipBased(function, {})))
      return failure();

  PassManager ownership(module.getContext());
  ownership.addPass(bufferization::createBufferDeallocationSimplificationPass());
  ownership.addPass(createCanonicalizerPass());
  ownership.addPass(createCSEPass());
  if (failed(ownership.run(module)) || failed(lowerLexicalReleases(module)))
    return failure();
  PassManager lexical(module.getContext());
  lexical.addPass(createCanonicalizerPass());
  lexical.addPass(createCSEPass());
  return lexical.run(module);
}

LogicalResult lowerOwnership(ModuleOp module) {
  PassManager lowering(module.getContext());
  lowering.addPass(createBufferizationToMemRefPass());
  lowering.addPass(createInlinerPass());
  lowering.addPass(createSymbolDCEPass());
  lowering.addPass(createCanonicalizerPass());
  lowering.addPass(createCSEPass());
  return lowering.run(module);
}

} // namespace intent::cpu
