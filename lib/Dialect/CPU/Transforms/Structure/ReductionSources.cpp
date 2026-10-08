#include "ReductionSources.h"
#include "ProducerReuse.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "../Vector/ProducerVectorization.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool projected(AffineMap map) {
  return !map.getNumSymbols() && llvm::all_of(map.getResults(), [](AffineExpr expression) {
    return isa<AffineDimExpr, AffineConstantExpr>(expression);
  });
}

SmallVector<Value> project(OpBuilder &builder, Location location, AffineMap map,
                           ValueRange coordinates) {
  SmallVector<Value> result;
  for (AffineExpr expression : map.getResults()) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression))
      result.push_back(coordinates[dimension.getPosition()]);
    else
      result.push_back(builder.create<arith::ConstantIndexOp>(
          location, cast<AffineConstantExpr>(expression).getValue()));
  }
  return result;
}

bool contiguousLane(Value memory) {
  auto type = cast<MemRefType>(memory.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.getRank() && !type.getElementType().isInteger(1) &&
      succeeded(type.getStridesAndOffset(strides, offset)) && strides.back() == 1;
}

} // namespace

struct ReductionSources::Impl {
  struct Source {
    Value memory;
    linalg::GenericOp producer;
    ProducerReplay payload;
    AffineMap projection;
    std::optional<unsigned> lane;
    SmallVector<Value> varying;
    SmallVector<Source *> inputs;
    SmallVector<bool> vectorInputs;
  };

  bool uniform(Source &source) const {
    if (!source.payload.reads.empty()) return false;
    Block &body = source.producer.getRegion().front();
    for (Value value : source.payload.frontier) {
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument || argument.getOwner() != &body ||
          argument.getArgNumber() >= source.producer.getInputs().size() ||
          isa<MemRefType>(source.producer.getInputs()[argument.getArgNumber()].getType()))
        return false;
    }
    return llvm::all_of(source.payload.nodes, [](Operation *operation) {
      return !operation->getNumRegions() && isMemoryEffectFree(operation) &&
          isSpeculatable(operation);
    });
  }

  explicit Impl(linalg::GenericOp consumer, Value completedSupply,
                ArrayRef<Operation *> supplyReaders)
      : consumer(consumer), function(consumer->getParentOfType<func::FuncOp>()), storage(function),
        completedSupply(completedSupply), supplyReaders(supplyReaders) {
    auto maps = consumer.getIndexingMapsArray();
    for (auto [number, input] : llvm::enumerate(consumer.getInputs())) {
      auto type = dyn_cast<MemRefType>(input.getType());
      roots.push_back(type ? collect(input, consumer, maps[number],
          type.getRank() ? std::optional<unsigned>(type.getRank() - 1) : std::nullopt) : nullptr);
    }
  }

  Source *collect(Value memory, linalg::GenericOp reader, AffineMap coordinates,
                  std::optional<unsigned> lane) {
    auto allocation = memory.getDefiningOp<memref::AllocOp>();
    if (!allocation || !projected(coordinates)) return nullptr;
    auto current = findCurrentBufferWrite(memory, reader, storage);
    auto producer = current ? dyn_cast<linalg::GenericOp>(current->operation) : linalg::GenericOp{};
    if (!producer || producer == reader || producer->getBlock() != reader->getBlock() ||
        !producer->isBeforeInBlock(reader) || producer.getOutputs().size() != 1 ||
        !completelyWritesBuffer(producer, memory) ||
        !producer.getRegion().front().getArguments().back().use_empty()) return nullptr;
    // No snapshot is duplicated: this completed version's observations belong
    // to one consumer, even when another version later reuses the allocation.
    auto observations = queryBufferVersionObservations(memory, producer, storage);
    bool sharedSupply = memory == completedSupply;
    if (failed(observations) || !storage.preserves(reader, memory)) return nullptr;
    if (sharedSupply) {
      if (observations->reads.empty() || llvm::any_of(observations->reads,
          [&](Operation *read) { return !llvm::is_contained(supplyReaders, read); })) return nullptr;
    } else if (observations->reads.size() != 1 || observations->reads.front() != reader) return nullptr;
    for (const auto &known : sources)
      if (known->memory == memory && known->producer == producer)
        return known->lane == lane ? known.get() : nullptr;
    auto readerMaps = reader.getIndexingMapsArray();
    for (auto [number, input] : llvm::enumerate(reader.getInputs()))
      if (input == memory && readerMaps[number] != coordinates) return nullptr;
    auto projection = fullOutputProjection(memory, producer.getIndexingMapsArray().back());
    auto payload = analyzeProducerResult(producer, storage);
    if (failed(projection) || failed(payload) ||
        !canReplayProducerAt(*payload, producer, consumer, storage, consumer.getOutputs()))
      return nullptr;
    Value result = producer.getRegion().front().getTerminator()->getOperand(0);
    if (sharedSupply) {
      if (!convertedLoadSource(*payload, result) ||
          !isNonRepeatingProjection(projection->compose(coordinates),
                                    reader.getStaticLoopRanges())) return nullptr;
    } else if (!shouldFuseProducerResult(*payload, result, projection->compose(coordinates),
                                        reader.getStaticLoopRanges(), true)) return nullptr;
    auto source = std::make_unique<Source>();
    source->memory = memory;
    source->producer = producer;
    source->payload = std::move(*payload);
    source->projection = *projection;
    source->lane = lane;
    auto maps = producer.getIndexingMapsArray();
    if (!llvm::all_of(maps, projected)) return nullptr;
    std::optional<unsigned> loopLane;
    if (lane)
      if (auto dimension = dyn_cast<AffineDimExpr>(maps.back().getResult(*lane)))
        loopLane = dimension.getPosition();
    Block &body = producer.getRegion().front();
    for (auto [number, input] : llvm::enumerate(producer.getInputs())) {
      Source *nested = nullptr;
      bool varying = false;
      if (isa<MemRefType>(input.getType()) &&
          llvm::is_contained(source->payload.frontier, body.getArgument(number))) {
        for (auto [axis, expression] : llvm::enumerate(maps[number].getResults())) {
          auto dimension = dyn_cast<AffineDimExpr>(expression);
          if (!loopLane || !dimension || dimension.getPosition() != *loopLane) continue;
          if (varying || axis + 1 != maps[number].getNumResults()) return nullptr;
          varying = true;
        }
        auto inputType = cast<MemRefType>(input.getType());
        nested = collect(input, producer, maps[number], varying
            ? std::optional<unsigned>(inputType.getRank() - 1) : std::nullopt);
        // A contiguous temporary must not become a strided scalar gather merely
        // because its numerical producer can be replayed.
        if (varying && !nested && !contiguousLane(input)) return nullptr;
      }
      source->inputs.push_back(nested);
      source->vectorInputs.push_back(varying);
      if (varying) source->varying.push_back(body.getArgument(number));
    }
    body.walk([&](linalg::IndexOp index) {
      if (loopLane && index.getDim() == *loopLane &&
          llvm::is_contained(source->payload.frontier, index.getResult()))
        source->varying.push_back(index.getResult());
    });
    if (lane && !ProducerVectorization(source->payload, {}, source->varying).canWiden(result))
      return nullptr;
    Source *selected = source.get();
    sources.push_back(std::move(source));
    return selected;
  }

  FailureOr<Value> emit(OpBuilder &builder, Source &source,
                        ValueRange coordinates, int64_t lanes) {
    for (const auto &known : emitted)
      if (known.source == &source && known.lanes == lanes &&
          known.coordinates.size() == coordinates.size() &&
          llvm::all_of(llvm::zip(known.coordinates, coordinates), [](auto pair) {
            auto [lhs, rhs] = pair;
            if (lhs == rhs) return true;
            auto a = getConstantIntValue(lhs), b = getConstantIntValue(rhs);
            return a && b && *a == *b;
          })) return known.value;
    Location location = source.producer.getLoc();
    SmallVector<Value> position = project(builder, location, source.projection, coordinates);
    auto maps = source.producer.getIndexingMapsArray();
    Block &body = source.producer.getRegion().front();
    IRMapping scalars, vectors;
    for (auto [number, input] : llvm::enumerate(source.producer.getInputs())) {
      Value formal = body.getArgument(number);
      if (!llvm::is_contained(source.payload.frontier, formal)) continue;
      Value value = input;
      int64_t inputLanes = lanes && source.vectorInputs[number] ? lanes : 0;
      if (isa<MemRefType>(input.getType())) {
        auto indices = project(builder, location, maps[number], position);
        if (Source *nested = source.inputs[number]) {
          auto result = emit(builder, *nested, indices, inputLanes);
          if (failed(result)) return failure();
          value = *result;
        } else if (inputLanes) {
          auto type = VectorType::get({inputLanes}, cast<MemRefType>(input.getType()).getElementType());
          value = builder.create<vector::LoadOp>(location, type, input, indices);
        } else {
          value = builder.create<memref::LoadOp>(location, input, indices);
        }
      }
      if (inputLanes) vectors.map(formal, value);
      else scalars.map(formal, value);
    }
    body.walk([&](linalg::IndexOp index) {
      if (!llvm::is_contained(source.payload.frontier, index.getResult())) return;
      Value scalar = position[index.getDim()];
      scalars.map(index.getResult(), scalar);
      if (lanes && llvm::is_contained(source.varying, index.getResult())) {
        auto type = VectorType::get({lanes}, builder.getIndexType());
        Value start = builder.create<vector::BroadcastOp>(location, type, scalar);
        Value steps = builder.create<vector::StepOp>(location, type);
        vectors.map(index.getResult(), builder.create<arith::AddIOp>(location, start, steps));
      }
    });
    Value result = body.getTerminator()->getOperand(0);
    auto replayed = lanes ? ProducerVectorization(source.payload, {}, source.varying)
        .materialize(result, lanes, builder, scalars, vectors)
        : materializeProducerValue(source.payload, result, builder, scalars);
    if (failed(replayed)) return failure();
    emitted.push_back({&source, llvm::to_vector(coordinates), lanes, *replayed});
    return *replayed;
  }

  struct Emitted {
    Source *source;
    SmallVector<Value> coordinates;
    int64_t lanes;
    Value value;
  };

  linalg::GenericOp consumer;
  func::FuncOp function;
  StorageAnalysis storage;
  SmallVector<Source *> roots;
  SmallVector<std::unique_ptr<Source>> sources;
  SmallVector<Emitted> emitted;
  Value completedSupply;
  SmallVector<Operation *> supplyReaders;
};

ReductionSources::ReductionSources(linalg::GenericOp consumer, Value completedSupply,
                                   ArrayRef<Operation *> supplyReaders)
    : impl(std::make_unique<Impl>(consumer, completedSupply, supplyReaders)) {}
ReductionSources::~ReductionSources() = default;

bool ReductionSources::hasReplays() const {
  return llvm::any_of(impl->roots, [](auto *source) { return source; });
}
bool ReductionSources::replays(unsigned input) const { return impl->roots[input]; }
void ReductionSources::beginMember() { impl->emitted.clear(); }

FailureOr<bool> ReductionSources::foldUniformMembers() {
  auto consumer = impl->consumer;
  auto maps = consumer.getIndexingMapsArray();
  auto preservesDomain = [&](ArrayRef<unsigned> removed) {
    AffineMap before = consumer.getShapesToLoopsMap();
    auto remainingMaps = maps;
    SmallVector<std::pair<Value, unsigned>> oldDimensions, newDimensions;
    for (auto [slot, operand] : llvm::enumerate(consumer->getOperands())) {
      bool omitted = llvm::is_contained(removed, slot);
      if (omitted)
        remainingMaps[slot] = AffineMap::get(consumer.getNumLoops(), 0, {}, consumer.getContext());
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type) continue;
      for (unsigned axis = 0; axis < type.getRank(); ++axis) {
        oldDimensions.emplace_back(operand, axis);
        if (!omitted) newDimensions.emplace_back(operand, axis);
      }
    }
    // This is GenericOp's native shape-to-loop inference, applied to the actual
    // remaining operands. Map coverage alone says nothing about dynamic sizes.
    AffineMap after = inversePermutation(concatAffineMaps(remainingMaps, consumer.getContext()));
    if (!before || !after || before.getNumResults() != after.getNumResults()) return false;
    for (auto [oldBound, newBound] : llvm::zip(before.getResults(), after.getResults())) {
      auto oldDimension = dyn_cast<AffineDimExpr>(oldBound);
      auto newDimension = dyn_cast<AffineDimExpr>(newBound);
      if (!oldDimension || !newDimension ||
          oldDimension.getPosition() >= oldDimensions.size() ||
          newDimension.getPosition() >= newDimensions.size()) return false;
      auto [oldValue, oldAxis] = oldDimensions[oldDimension.getPosition()];
      auto [newValue, newAxis] = newDimensions[newDimension.getPosition()];
      if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(oldValue, oldAxis),
                            ValueBoundsConstraintSet::Variable(newValue, newAxis))) return false;
    }
    return true;
  };
  SmallVector<unsigned> selected;
  for (auto [number, source] : llvm::enumerate(impl->roots)) {
    if (!source || !impl->uniform(*source)) continue;
    selected.push_back(number);
    if (!preservesDomain(selected)) selected.pop_back();
  }
  if (selected.empty()) return false;
  OpBuilder builder(consumer);
  SmallVector<Value> replacements;
  beginMember();
  for (unsigned input : selected) {
    auto type = cast<MemRefType>(consumer.getInputs()[input].getType());
    Value zero = builder.create<arith::ConstantIndexOp>(consumer.getLoc(), 0);
    SmallVector<Value> coordinates(type.getRank(), zero);
    auto value = materialize(builder, input, coordinates);
    if (failed(value)) return failure();
    replacements.push_back(*value);
  }
  for (auto [input, value] : llvm::zip(selected, replacements)) {
    consumer.getInputsMutable()[input].set(value);
    maps[input] = AffineMap::get(consumer.getNumLoops(), 0, {}, consumer.getContext());
  }
  consumer.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
  eraseUnusedProducers();
  return true;
}

FailureOr<Value> ReductionSources::materialize(OpBuilder &builder, unsigned input,
                                              ValueRange coordinates, int64_t lanes) {
  return impl->emit(builder, *impl->roots[input], coordinates, lanes);
}

void ReductionSources::eraseUnusedProducers() {
  // Children were planned before their parent. Delete consumers before their
  // sources, after the original reduction itself has been replaced.
  for (auto &source : llvm::reverse(impl->sources)) {
    // Uniform-member folding may have replaced only some roots. Re-query the
    // current observations rather than deleting other still-used versions.
    StorageAnalysis current(impl->function);
    auto observations = queryBufferVersionObservations(source->memory, source->producer, current);
    if (succeeded(observations) && observations->reads.empty())
      eraseReplayedProducerVersion(source->producer);
  }
}

} // namespace intent::cpu
