#include "CollectiveBufferization.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace intent::cpu;

namespace {

OperandRange destinations(Operation *operation) {
  return TypeSwitch<Operation *, OperandRange>(operation)
      .Case<SliceReduceOp, ScanOp>([](auto op) { return op.getOutputs(); })
      .Case<RegionFoldOp, RegionScanOp>([](auto op) {
        return cast<RegionOpInterface>(op.getOperation()).getDestinations();
      })
      .Case<HistogramOp, QuantizeOp, QuantizedDotOp>([](auto op) {
        return op->getOperands().take_back(1);
      });
}

unsigned destinationBegin(Operation *operation) {
  return operation->getNumOperands() - destinations(operation).size();
}

bool scalarCombine(Operation *operation) {
  ValueRange identities;
  if (auto reduce = dyn_cast<SliceReduceOp>(operation))
    identities = reduce.getIdentities();
  else if (auto scan = dyn_cast<ScanOp>(operation))
    identities = scan.getInitials();
  else
    return false;
  return llvm::none_of(identities, [](Value value) {
    return isa<ShapedType>(value.getType());
  });
}

MemRefType slicedBufferType(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType(),
      StridedLayoutAttr::get(type.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)));
}

// This is the calling convention of a bufferized helper, not an alias relation:
// incoming state formals describe scratch selected by collective realization.
FailureOr<Type> helperArgumentType(Operation *operation, BlockArgument argument,
                                  const BufferizationOptions &options,
                                  SmallVector<Value> &invocationStack) {
  Type type = argument.getType();
  if (auto program = dyn_cast<RegionOpInterface>(operation)) {
    auto schema = program.getRegionSchema(*argument.getOwner()->getParent());
    if (failed(schema)) return failure();
    const auto &entry = (*schema)[argument.getArgNumber()];
    if (entry.kind == RegionArgumentKind::Captures) {
      if (!isa<TensorType>(type)) return type;
      auto buffer = bufferization::getBufferType(entry.prototype, options,
                                                 invocationStack);
      if (failed(buffer)) return failure();
      return Type(*buffer);
    }
    if (entry.sliceAxis)
      return Type(slicedBufferType(cast<RankedTensorType>(type)));
    return Type(collectiveStateType(type));
  }

  ValueRange sources, captures;
  if (auto reduce = dyn_cast<SliceReduceOp>(operation)) {
    sources = reduce.getSources();
    captures = reduce.getCaptures();
  } else {
    auto scan = cast<ScanOp>(operation);
    sources = scan.getSources();
    captures = scan.getCaptures();
  }
  unsigned index = argument.getArgNumber(), count = sources.size();
  if (index >= 2 * count) {
    if (!isa<TensorType>(type)) return type;
    auto buffer = bufferization::getBufferType(captures[index - 2 * count],
                                               options, invocationStack);
    if (failed(buffer)) return failure();
    return Type(*buffer);
  }
  if (scalarCombine(operation)) return type;
  if (index >= count && isa<RankedTensorType>(type))
    return Type(slicedBufferType(cast<RankedTensorType>(type)));
  return Type(collectiveStateType(type));
}

struct HelperSignature {
  SmallVector<Type> arguments;
  SmallVector<Type> yielded;
  bool destinationPassing;
};

FailureOr<SmallVector<HelperSignature>>
helperSignatures(Operation *operation, const BufferizationOptions &options) {
  SmallVector<HelperSignature> result;
  for (Region &region : operation->getRegions()) {
    HelperSignature signature;
    signature.destinationPassing = !scalarCombine(operation);
    for (BlockArgument argument : region.front().getArguments()) {
      SmallVector<Value> stack;
      auto type = helperArgumentType(operation, argument, options, stack);
      if (failed(type)) return failure();
      signature.arguments.push_back(*type);
    }
    llvm::append_range(signature.yielded,
                      region.front().getTerminator()->getOperandTypes());
    result.push_back(std::move(signature));
  }
  return result;
}

LogicalResult convertHelper(Region &region, const HelperSignature &signature,
                            RewriterBase &rewriter,
                            const BufferizationOptions &options) {
  Block &body = region.front();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body);
  for (auto [argument, type] : llvm::zip_equal(body.getArguments(),
                                             signature.arguments)) {
    Type oldType = argument.getType();
    if (type == oldType) continue;
    argument.setType(type);
    Value replacement;
    if (isa<TensorType>(oldType)) {
      // Analysis is already complete. This bridge reconnects existing tensor
      // users until their own models bufferize; it makes no restrict promise.
      replacement = rewriter.create<ToTensorOp>(argument.getLoc(), oldType,
                                                argument);
    } else {
      replacement = rewriter.create<memref::LoadOp>(argument.getLoc(), argument,
                                                    ValueRange{});
    }
    argument.replaceAllUsesExcept(replacement, replacement.getDefiningOp());
  }
  if (!signature.destinationPassing) return success();

  Operation *terminator = body.getTerminator();
  rewriter.setInsertionPoint(terminator);
  for (auto [value, type] : llvm::zip_equal(terminator->getOperands(),
                                          signature.yielded)) {
    auto destinationType = collectiveStateType(type);
    // Emission destinations are source slices, while summary/state scratch is
    // dense. Their operand schema owns this distinction after bufferization.
    if (auto program = dyn_cast<RegionOpInterface>(region.getParentOp()))
      if (&region == program.getEmitRegion())
        destinationType = slicedBufferType(cast<RankedTensorType>(type));
    BlockArgument output = body.addArgument(destinationType, terminator->getLoc());
    if (isa<TensorType>(type)) {
      auto buffer = getBuffer(rewriter, value, options);
      if (failed(buffer)) return failure();
      rewriter.create<memref::CopyOp>(terminator->getLoc(), *buffer, output);
    } else {
      rewriter.create<memref::StoreOp>(terminator->getLoc(), value, output,
                                      ValueRange{});
    }
  }
  rewriter.modifyOpInPlace(terminator, [&] { terminator->setOperands({}); });
  return success();
}

LogicalResult bufferizeCollective(Operation *operation, RewriterBase &rewriter,
                                  const BufferizationOptions &options) {
  if (!operation->getNumResults()) return success();
  auto signatures = helperSignatures(operation, options);
  if (failed(signatures)) return failure();
  SmallVector<Value> operands, outputs;
  unsigned begin = destinationBegin(operation);
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    Value value = operand;
    if (index >= begin &&
        !isa<TensorType>(operation->getResult(index - begin).getType())) {
      auto type = cast<RankedTensorType>(operand.getType());
      auto allocation = options.createAlloc(rewriter, operation->getLoc(),
          MemRefType::get({}, type.getElementType()), {});
      if (failed(allocation)) return failure();
      value = *allocation;
    } else if (isa<TensorType>(operand.getType())) {
      auto buffer = getBuffer(rewriter, operand, options);
      if (failed(buffer)) return failure();
      value = *buffer;
    }
    operands.push_back(value);
    if (index >= begin) outputs.push_back(value);
  }

  OperationState state(operation->getLoc(), operation->getName());
  state.addOperands(operands);
  state.addAttributes(operation->getAttrs());
  if (isa<RegionScanOp>(operation))
    state.attributes.set("resultSegmentSizes", rewriter.getDenseI32ArrayAttr({0, 0}));
  for (unsigned index = 0; index < operation->getNumRegions(); ++index)
    state.addRegion();
  Operation *bufferized = rewriter.create(state);
  for (auto [index, region] : llvm::enumerate(operation->getRegions())) {
    Region &target = bufferized->getRegion(index);
    rewriter.inlineRegionBefore(region, target, target.end());
    if (failed(convertHelper(target, (*signatures)[index], rewriter, options)))
      return failure();
  }
  rewriter.setInsertionPointAfter(bufferized);
  SmallVector<Value> replacements;
  for (auto [result, output] : llvm::zip_equal(operation->getResults(), outputs)) {
    if (isa<TensorType>(result.getType())) replacements.push_back(output);
    else replacements.push_back(rewriter.create<memref::LoadOp>(
        operation->getLoc(), output, ValueRange{}));
  }
  replaceOpWithBufferizedValues(rewriter, operation, replacements);
  return success();
}

template <typename Op>
struct CollectiveModel
    : BufferizableOpInterface::ExternalModel<CollectiveModel<Op>, Op> {
  bool bufferizesToMemoryRead(Operation *operation, OpOperand &operand,
                             const AnalysisState &) const {
    return operand.getOperandNumber() < destinationBegin(operation);
  }
  bool bufferizesToMemoryWrite(Operation *operation, OpOperand &operand,
                              const AnalysisState &) const {
    return operand.getOperandNumber() >= destinationBegin(operation);
  }
  AliasingValueList getAliasingValues(Operation *operation, OpOperand &operand,
                                    const AnalysisState &) const {
    unsigned begin = destinationBegin(operation);
    if (operand.getOperandNumber() < begin) return {};
    Value result = operation->getResult(operand.getOperandNumber() - begin);
    if (!isa<TensorType>(result.getType())) return {};
    return {{result, BufferRelation::Equivalent}};
  }
  bool isWritable(Operation *, Value value, const AnalysisState &) const {
    return isa<OpResult>(value);
  }
  bool isRepetitiveRegion(Operation *, unsigned) const { return true; }
  FailureOr<BaseMemRefType>
  getBufferType(Operation *operation, Value value,
                const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto type = helperArgumentType(operation, argument, options, invocationStack);
      if (failed(type)) return failure();
      return cast<BaseMemRefType>(*type);
    }
    unsigned index = cast<OpResult>(value).getResultNumber();
    return bufferization::getBufferType(destinations(operation)[index], options,
                                        invocationStack);
  }
  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeCollective(operation, rewriter, options);
  }
};

// A helper yield returns computed values to its owning collective, which copies
// them into independent destinations. It is not an alias of the owner's result.
template <typename Op>
struct YieldModel
    : BufferizableOpInterface::ExternalModel<YieldModel<Op>, Op> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &, const AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &, const AnalysisState &) const {
    return false;
  }
  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                    const AnalysisState &) const { return {}; }
  LogicalResult bufferize(Operation *, RewriterBase &,
                          const BufferizationOptions &) const {
    // The parent's model consumes the yield while converting the signature.
    return success();
  }
};

} // namespace

void intent::cpu::registerCollectiveBufferizationInterfaces(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, IntentCPUDialect *) {
    SliceReduceOp::attachInterface<CollectiveModel<SliceReduceOp>>(*context);
    ScanOp::attachInterface<CollectiveModel<ScanOp>>(*context);
    RegionFoldOp::attachInterface<CollectiveModel<RegionFoldOp>>(*context);
    RegionScanOp::attachInterface<CollectiveModel<RegionScanOp>>(*context);
    HistogramOp::attachInterface<CollectiveModel<HistogramOp>>(*context);
    QuantizeOp::attachInterface<CollectiveModel<QuantizeOp>>(*context);
    QuantizedDotOp::attachInterface<CollectiveModel<QuantizedDotOp>>(*context);
    SliceReduceYieldOp::attachInterface<YieldModel<SliceReduceYieldOp>>(*context);
    ScanYieldOp::attachInterface<YieldModel<ScanYieldOp>>(*context);
    RegionYieldOp::attachInterface<YieldModel<RegionYieldOp>>(*context);
  });
}
