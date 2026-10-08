#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "IntegerSources.h"
#include "ProducerReuse.h"
#include "../Storage/AccessAliases.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool integer(Type type) { return isa<IndexType, IntegerType>(type); }

bool projectedCoordinates(AffineMap map) {
  return !map.getNumSymbols() && llvm::all_of(map.getResults(), [](AffineExpr expr) {
    return isa<AffineDimExpr, AffineConstantExpr>(expr);
  });
}

struct IntegerProducer {
  linalg::GenericOp operation;
  ProducerReplay payload;
  AffineMap outputProjection;
  BufferVersionObservations observations;
};

std::optional<IntegerProducer> integerProducer(memref::LoadOp load,
                                               StorageAnalysis &storage) {
  if (!integer(load.getType())) return std::nullopt;
  auto allocation = load.getMemref().getDefiningOp<memref::AllocOp>();
  if (!allocation) return std::nullopt;
  auto current = findCurrentBufferWrite(allocation, load, storage);
  auto producer = current ? dyn_cast<linalg::GenericOp>(current->operation) : linalg::GenericOp{};
  if (!producer || producer.getNumResults() || producer.getOutputs().size() != 1 ||
      producer.getNumReductionLoops() || producer->getBlock() != allocation->getBlock() ||
      !completelyWritesBuffer(producer, allocation))
    return std::nullopt;
  auto projection = fullOutputProjection(allocation, producer.getIndexingMapsArray().back());
  if (failed(projection)) return std::nullopt;
  Block &body = producer.getRegion().front();
  if (!body.getArguments().back().use_empty()) return std::nullopt;
  if (llvm::any_of(producer.getIndexingMapsArray(), [](AffineMap map) {
        return !projectedCoordinates(map);
      })) return std::nullopt;
  auto observations = queryBufferVersionObservations(allocation, producer, storage);
  if (failed(observations) || !llvm::is_contained(observations->reads, load.getOperation()))
    return std::nullopt;
  auto payload = analyzeProducerResult(producer, storage);
  if (failed(payload) ||
      !canReplayProducerAt(*payload, producer, load, storage, ValueRange{allocation}))
    return std::nullopt;
  return IntegerProducer{producer, std::move(*payload), *projection, std::move(*observations)};
}

Value replay(const ProducerReplayGroup &group, const IntegerProducer &source) {
  auto producer = source.operation;
  OpBuilder builder(group.anchor);
  Location loc = group.anchor->getLoc();
  auto maps = producer.getIndexingMapsArray();
  ArrayRef<Value> coordinates = group.coordinates;
  IRMapping mapping;
  Block &body = producer.getRegion().front();
  for (auto [number, input] : llvm::enumerate(producer.getInputs())) {
    if (!llvm::is_contained(source.payload.frontier, body.getArgument(number))) continue;
    Value value = input;
    if (isa<MemRefType>(input.getType())) {
      SmallVector<Value> indices;
      for (AffineExpr expr : maps[number].getResults()) {
        if (auto dim = dyn_cast<AffineDimExpr>(expr)) indices.push_back(coordinates[dim.getPosition()]);
        else indices.push_back(builder.create<arith::ConstantIndexOp>(loc, cast<AffineConstantExpr>(expr).getValue()));
      }
      value = builder.create<memref::LoadOp>(loc, input, indices);
    }
    mapping.map(body.getArgument(number), value);
  }
  body.walk([&](linalg::IndexOp index) {
    if (llvm::is_contained(source.payload.frontier, index.getResult()))
      mapping.map(index.getResult(), coordinates[index.getDim()]);
  });
  auto cloned = materializeProducerValue(source.payload,
      body.getTerminator()->getOperand(0), builder, mapping);
  assert(succeeded(cloned) && "integer replay must bind the complete payload frontier");
  return *cloned;
}

} // namespace

LogicalResult foldIntegerSources(func::FuncOp function) {
  bool changed;
  do {
    changed = false;
    if (failed(foldPrivateAccessAliases(function))) return failure();
    SmallVector<memref::LoadOp> loads;
    function.walk([&](memref::LoadOp load) { if (integer(load.getType())) loads.push_back(load); });
    for (auto load : loads) {
      // Every successful replay changes both uses and read placement. Queries
      // deliberately do not survive that rewrite or a dead-buffer deletion.
      StorageAnalysis storage(function);
      auto producer = integerProducer(load, storage);
      if (!producer) continue;
      Value buffer = load.getMemref();
      SmallVector<ProducerReplayUse> uses;
      bool removesProducer = true;
      for (Operation *user : producer->observations.reads) {
        auto read = dyn_cast<memref::LoadOp>(user);
        if (!read || read.getMemref() != buffer) { removesProducer = false; continue; }
        SmallVector<Value> coordinates;
        for (AffineExpr expression : producer->outputProjection.getResults())
          coordinates.push_back(read.getIndices()[cast<AffineDimExpr>(expression).getPosition()]);
        uses.push_back({read, std::move(coordinates), 1});
      }
      auto groups = groupProducerReplays(producer->payload,
          producer->operation.getRegion().front().getTerminator()->getOperand(0),
          producer->operation, producer->operation->getBlock(), uses,
          removesProducer, storage);
      if (failed(groups)) continue;
      for (const auto &group : *groups) {
        Value replacement = replay(group, *producer);
        for (Operation *read : group.uses) {
          read->getResult(0).replaceAllUsesWith(replacement);
          read->erase();
        }
      }
      if (removesProducer) eraseReplayedProducerVersion(producer->operation);
      changed = true;
      // Uses, producer lifetime and possible scalar bindings have changed.
      // Recollect the next batch from the live program, never from this list.
      break;
    }
    if (changed) eraseDeadPrivateBuffers(function);
  } while (changed);
  return success();
}

} // namespace intent::cpu
