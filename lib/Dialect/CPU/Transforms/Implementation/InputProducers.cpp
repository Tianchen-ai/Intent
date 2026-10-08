#include "InputProducers.h"
#include "InputWindows.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "../Structure/ProducerReuse.h"
#include "../Vector/ProducerVectorization.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::cpu {
namespace {

Value readMemory(Operation *operation) {
  if (auto load = dyn_cast<memref::LoadOp>(operation)) return load.getMemref();
  if (auto copy = dyn_cast<memref::CopyOp>(operation)) return copy.getSource();
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  if (!generic || generic.getNumResults() || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1 || generic.getNumReductionLoops() ||
      !generic.getRegion().front().getArguments().back().use_empty()) return {};
  for (AffineMap map : generic.getIndexingMapsArray())
    if (!map.isIdentity()) return {};
  for (Operation &nested : generic.getRegion().front().without_terminator())
    if (nested.getNumRegions() || !isMemoryEffectFree(&nested) ||
        !isSpeculatable(&nested)) return {};
  return generic.getInputs()[0];
}

struct DescriptorPath {
  Value root;
  SmallVector<Value> views;
};

std::optional<DescriptorPath> descriptorPath(Value memory) {
  DescriptorPath path;
  while (!memory.getDefiningOp<memref::AllocOp>()) {
    path.views.push_back(memory);
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) {
      memory = cast.getSource();
    } else if (auto view = memory.getDefiningOp<memref::SubViewOp>()) {
      // Compose the actual descriptor, including dropped dimensions. Unit
      // strides retain the original source-coordinate member domain.
      if (llvm::any_of(view.getMixedStrides(), [](OpFoldResult stride) {
            return getConstantIntValue(stride) != 1;
          })) return std::nullopt;
      memory = view.getSource();
    } else {
      auto projection = queryViewAxisProjection(memory);
      if (failed(projection) || projection->source == memory) return std::nullopt;
      memory = projection->source;
    }
  }
  path.root = memory;
  return path;
}

SmallVector<Value> sourceCoordinates(OpBuilder &builder, Location loc,
    const DescriptorPath &path, ValueRange coordinates) {
  SmallVector<Value> result(coordinates);
  for (Value value : path.views) {
    if (value.getDefiningOp<memref::CastOp>()) continue;
    if (auto view = value.getDefiningOp<memref::SubViewOp>()) {
      SmallVector<Value> source;
      auto dropped = view.getDroppedDims();
      unsigned axis = 0;
      for (auto [number, offset] : llvm::enumerate(view.getMixedOffsets())) {
        Value begin = getValueOrCreateConstantIndexOp(builder, loc, offset);
        source.push_back(dropped.test(number) ? begin
            : add(builder, loc, begin, result[axis++]));
      }
      result = std::move(source);
      continue;
    }
    auto projection = *queryViewAxisProjection(value);
    SmallVector<Value> source(cast<MemRefType>(projection.source.getType()).getRank(),
                              index(builder, loc, 0));
    for (auto [axis, input] : llvm::enumerate(projection.sourceAxes))
      if (input) source[*input] = result[axis];
    result = std::move(source);
  }
  return result;
}

// Substituting an element access can remove its intermediate write/read without
// repeating numerical work. Index-table reads stay as stored coordinate inputs;
// arithmetic on loaded numerical values never enters this qualification.
bool accessOnly(const ProducerReplay &payload, Value result) {
  auto load = result.getDefiningOp<memref::LoadOp>();
  if (!load) return isLoadReplacement(payload, result);
  llvm::SmallPtrSet<Operation *, 16> addressOperations;
  std::function<bool(Value)> address = [&](Value value) {
    if (llvm::is_contained(payload.frontier, value) ||
        llvm::is_contained(payload.externalValues, value)) return true;
    Operation *operation = value.getDefiningOp();
    if (!operation || !llvm::is_contained(payload.nodes, operation)) return false;
    if (!addressOperations.insert(operation).second) return true;
    if (auto coordinate = dyn_cast<memref::LoadOp>(operation)) {
      if (!coordinate.getType().isIntOrIndex()) return false;
    } else if (!isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp,
                    arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
                    arith::ExtUIOp, arith::TruncIOp>(operation)) {
      return false;
    }
    return llvm::all_of(operation->getOperands(), address);
  };
  return llvm::all_of(load->getOperands(), address);
}

bool hasIndirectAddress(linalg::GenericOp producer,
                        const ProducerReplay &payload) {
  llvm::SmallPtrSet<Operation *, 16> visiting;
  std::function<bool(Value)> readsElement = [&](Value value) {
    if (auto formal = dyn_cast<BlockArgument>(value))
      return formal.getOwner() == &producer.getRegion().front() &&
          formal.getArgNumber() < producer.getInputs().size() &&
          isa<MemRefType>(producer.getInputs()[formal.getArgNumber()].getType());
    Operation *operation = value.getDefiningOp();
    if (!operation || !llvm::is_contained(payload.nodes, operation)) return false;
    if (isa<memref::LoadOp>(operation)) return true;
    if (!visiting.insert(operation).second) return false;
    bool found = llvm::any_of(operation->getOperands(), readsElement);
    visiting.erase(operation);
    return found;
  };
  return llvm::any_of(payload.nodes, [&](Operation *operation) {
    auto load = dyn_cast<memref::LoadOp>(operation);
    return load && llvm::any_of(load.getIndices(), readsElement);
  });
}

std::optional<unsigned> preparationLane(Operation *read) {
  Value memory = readMemory(read);
  auto type = cast<MemRefType>(memory.getType());
  if (!type.getRank()) return std::nullopt;
  auto load = dyn_cast<memref::LoadOp>(read);
  if (!load) return type.getRank() - 1;
  auto loop = read->getParentOfType<scf::ForOp>();
  if (!loop) return std::nullopt;
  std::optional<unsigned> lane;
  for (auto [axis, coordinate] : llvm::enumerate(load.getIndices())) {
    auto dependencies = inputWindowDependencies(ValueRange{coordinate}, loop,
                                                ValueRange{loop.getInductionVar()});
    if (!dependencies) return std::nullopt;
    if (dependencies->frontier.empty()) continue;
    if (lane) return std::nullopt;
    lane = axis;
  }
  return lane;
}

std::optional<unsigned> rootAxis(const DescriptorPath &path, unsigned axis) {
  for (Value value : path.views) {
    if (value.getDefiningOp<memref::CastOp>()) continue;
    if (auto view = value.getDefiningOp<memref::SubViewOp>()) {
      auto dropped = view.getDroppedDims();
      unsigned actual = 0, result = 0;
      for (; actual < dropped.size(); ++actual) {
        if (dropped.test(actual)) continue;
        if (result++ == axis) break;
      }
      if (actual == dropped.size()) return std::nullopt;
      axis = actual;
    } else {
      auto projection = *queryViewAxisProjection(value);
      if (!projection.sourceAxes[axis]) return std::nullopt;
      axis = *projection.sourceAxes[axis];
    }
  }
  return axis;
}

bool vectorizesAlong(linalg::GenericOp producer, const ProducerReplay &payload,
                     unsigned axis) {
  Value coordinate;
  SmallVector<Value> varying, guarded;
  Block &body = producer.getRegion().front();
  auto maps = producer.getIndexingMapsArray();
  for (Value frontier : payload.frontier) {
    if (auto index = frontier.getDefiningOp<linalg::IndexOp>()) {
      if (index.getDim() == axis) {
        if (!coordinate) coordinate = frontier;
        else varying.push_back(frontier);
      }
      continue;
    }
    auto formal = dyn_cast<BlockArgument>(frontier);
    if (!formal || formal.getOwner() != &body ||
        formal.getArgNumber() >= producer.getInputs().size()) continue;
    unsigned number = formal.getArgNumber();
    auto type = dyn_cast<MemRefType>(producer.getInputs()[number].getType());
    if (!type || !llvm::any_of(maps[number].getResults(), [&](AffineExpr expression) {
          return expression.isFunctionOfDim(axis);
        })) continue;
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!type.getRank() || failed(type.getStridesAndOffset(strides, offset)) ||
        (strides.back() != 1 && !ShapedType::isDynamic(strides.back()))) return false;
    for (auto [dimension, expression] : llvm::enumerate(maps[number].getResults()))
      if (dimension + 1 == type.getRank()
              ? expression != getAffineDimExpr(axis, producer.getContext())
              : expression.isFunctionOfDim(axis)) return false;
    varying.push_back(frontier);
  }
  return ProducerVectorization(payload, coordinate, varying).canWiden(
      body.getTerminator()->getOperand(0), &guarded);
}

bool preservesAccessVectorization(linalg::GenericOp producer,
    const ProducerReplay &payload, AffineMap output, const DescriptorPath &path,
    Operation *read) {
  if (!hasIndirectAddress(producer, payload)) return true;
  auto type = cast<MemRefType>(path.root.getType());
  if (!type.getRank() || !producer.getNumLoops()) return true;
  // Match the original structured traversal's actual innermost loop and store,
  // not merely the allocation's shape. A transposed preparation must not turn
  // that SIMD producer into scalar indirect gathers to remove a temporary.
  unsigned original = producer.getNumLoops() - 1;
  auto writeMap = producer.getIndexingMapsArray().back();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) || strides.back() != 1 ||
      writeMap.getResults().back() != getAffineDimExpr(original, producer.getContext()) ||
      llvm::any_of(writeMap.getResults().drop_back(), [&](AffineExpr expression) {
        return expression.isFunctionOfDim(original);
      }) || !vectorizesAlong(producer, payload, original)) return true;
  auto loopAxis = [&](unsigned memoryAxis) -> std::optional<unsigned> {
    for (auto [axis, expression] : llvm::enumerate(output.getResults()))
      if (auto dimension = dyn_cast<AffineDimExpr>(expression);
          dimension && dimension.getPosition() == memoryAxis) return axis;
    return std::nullopt;
  };
  auto lane = preparationLane(read);
  auto physical = lane ? rootAxis(path, *lane) : std::nullopt;
  auto transported = physical ? loopAxis(*physical) : std::nullopt;
  return transported && (*transported == original ||
                         vectorizesAlong(producer, payload, *transported));
}

bool sameCompleteDescriptor(Value source, Value root) {
  auto path = descriptorPath(source);
  if (!path || path->root != root) return false;
  for (Value value : path->views) {
    auto view = value.getDefiningOp<memref::SubViewOp>();
    if (!view) continue;
    for (auto [axis, offset] : llvm::enumerate(view.getMixedOffsets()))
      if (getConstantIntValue(offset) != 0 || !haveEqualExtents(
              ValueBoundsConstraintSet::Variable(view.getSource(), axis),
              ValueBoundsConstraintSet::Variable(view.getMixedSizes()[axis])))
        return false;
  }
  return true;
}

bool onceAfter(linalg::GenericOp producer, Block *block) {
  // The preparation itself visits every source element once. Its placement may
  // be guarded, but a loop between it and the defining version would repeat it.
  while (block != producer->getBlock()) {
    Operation *owner = block->getParentOp();
    if (!isa_and_nonnull<scf::IfOp>(owner)) return false;
    block = owner->getBlock();
  }
  return true;
}

struct Replay {
  linalg::GenericOp producer;
  const ProducerReplay &payload;
  AffineMap output;
  const DescriptorPath &path;
};

FailureOr<Value> replay(OpBuilder &builder, Location loc, const Replay &source,
                        ValueRange coordinates) {
  auto actual = sourceCoordinates(builder, loc, source.path, coordinates);
  auto loops = affine::expandAffineMap(builder, loc, source.output, actual);
  if (!loops) return failure();
  auto producer = source.producer;
  Block &body = producer.getRegion().front();
  auto maps = producer.getIndexingMapsArray();
  IRMapping mapping;
  for (auto [number, input] : llvm::enumerate(producer.getInputs())) {
    Value formal = body.getArgument(number);
    if (!llvm::is_contained(source.payload.frontier, formal)) continue;
    Value value = input;
    if (isa<MemRefType>(input.getType())) {
      auto indices = affine::expandAffineMap(builder, loc, maps[number], *loops);
      if (!indices) return failure();
      value = builder.create<memref::LoadOp>(loc, input, *indices);
    }
    mapping.map(formal, value);
  }
  body.walk([&](linalg::IndexOp coordinate) {
    if (llvm::is_contained(source.payload.frontier, coordinate.getResult()))
      mapping.map(coordinate.getResult(), (*loops)[coordinate.getDim()]);
  });
  return materializeProducerValue(source.payload,
      body.getTerminator()->getOperand(0), builder, mapping);
}

LogicalResult replaceRead(Operation *read, const Replay &source) {
  OpBuilder builder(read);
  Location loc = read->getLoc();
  if (auto load = dyn_cast<memref::LoadOp>(read)) {
    auto value = replay(builder, loc, source, load.getIndices());
    if (failed(value)) return failure();
    load.replaceAllUsesWith(*value);
    load.erase();
    return success();
  }
  Value input = readMemory(read), destination;
  auto generic = dyn_cast<linalg::GenericOp>(read);
  if (auto copy = dyn_cast<memref::CopyOp>(read)) destination = copy.getTarget();
  else destination = generic.getOutputs()[0];
  auto type = cast<MemRefType>(input.getType());
  SmallVector<Value> sizes, coordinates;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    sizes.push_back(builder.createOrFold<memref::DimOp>(loc, input, axis));
  auto zero = index(builder, loc, 0);
  LogicalResult status = success();
  std::function<void(unsigned)> emit = [&](unsigned axis) {
    if (axis < sizes.size()) {
      loop(builder, loc, zero, sizes[axis], 1, [&](Value coordinate) {
        coordinates.push_back(coordinate);
        emit(axis + 1);
        coordinates.pop_back();
      });
      return;
    }
    auto value = replay(builder, loc, source, coordinates);
    if (failed(value)) { status = failure(); return; }
    Value stored = *value;
    if (generic) {
      IRMapping mapping;
      Block &body = generic.getRegion().front();
      mapping.map(body.getArgument(0), stored);
      for (Operation &operation : body.without_terminator())
        builder.clone(operation, mapping);
      stored = mapping.lookupOrDefault(body.getTerminator()->getOperand(0));
    }
    builder.create<memref::StoreOp>(loc, stored, destination, coordinates);
  };
  emit(0);
  if (failed(status)) return failure();
  read->erase();
  return success();
}

} // namespace

unsigned InputProducerCopies::begin(Block *block, Value source) {
  preparations.push_back({block, source, {}});
  return preparations.size() - 1;
}

void InputProducerCopies::record(unsigned preparation, Operation *read) {
  preparations[preparation].reads.push_back(read);
}

LogicalResult InputProducerCopies::fold(func::FuncOp function) {
  SmallVector<linalg::GenericOp> producers;
  function.walk([&](linalg::GenericOp producer) {
    if (producer.getOutputs().size() == 1 &&
        producer.getOutputs()[0].getDefiningOp<memref::AllocOp>())
      producers.push_back(producer);
  });
  for (auto producer : llvm::reverse(producers)) {
    Value memory = producer.getOutputs()[0];
    if (producer.getNumResults() || producer.getNumReductionLoops() ||
        !producer.getRegion().front().getArguments().back().use_empty() ||
        !completelyWritesBuffer(producer, memory)) continue;
    StorageAnalysis storage(function);
    auto observations = queryBufferVersionObservations(memory, producer, storage);
    auto projection = fullOutputProjection(memory, producer.getIndexingMapsArray().back());
    auto payload = analyzeProducerResult(producer, storage);
    if (failed(observations) || observations->reads.empty() ||
        failed(projection) || failed(payload)) continue;
    struct Use { Operation *read; DescriptorPath path; };
    SmallVector<Use, 0> uses;
    SmallVector<unsigned> traversals;
    bool complete = true;
    for (Operation *read : observations->reads) {
      std::optional<unsigned> preparation;
      for (auto [number, candidate] : llvm::enumerate(preparations))
        if (llvm::is_contained(candidate.reads, read)) { preparation = number; break; }
      Value input = preparation ? readMemory(read) : Value{};
      auto path = input ? descriptorPath(input) : std::nullopt;
      auto current = path ? findCurrentBufferWrite(path->root, read, storage) : std::nullopt;
      if (!path || path->root != memory || !current || current->operation != producer ||
          !canReplayProducerAt(*payload, producer, read, storage) ||
          !preservesAccessVectorization(producer, *payload, *projection, *path, read)) {
        complete = false; break;
      }
      if (!llvm::is_contained(traversals, *preparation)) traversals.push_back(*preparation);
      uses.push_back({read, std::move(*path)});
    }
    if (!complete) continue;
    Value result = producer.getRegion().front().getTerminator()->getOperand(0);
    if (!accessOnly(*payload, result)) {
      if (traversals.size() != 1) continue;
      const Preparation &preparation = preparations[traversals.front()];
      if (!sameCompleteDescriptor(preparation.source, memory) ||
          !onceAfter(producer, preparation.block) ||
          !shouldFuseProducerResult(*payload, result, *projection,
              cast<MemRefType>(memory.getType()).getShape(), true)) continue;
    }
    for (const Use &use : uses) {
      Replay source{producer, *payload, *projection, use.path};
      if (failed(replaceRead(use.read, source)))
        return producer.emitError("prepared input producer lost its proved coordinate replay");
      for (Preparation &preparation : preparations)
        llvm::erase(preparation.reads, use.read);
    }
    // The rewrite above changed aliases and effects. Never carry its query
    // snapshot into deletion, nor delete a still-observed later version.
    StorageAnalysis current(function);
    auto remaining = queryBufferVersionObservations(memory, producer, current);
    if (succeeded(remaining) && remaining->reads.empty())
      eraseReplayedProducerVersion(producer);
  }
  return success();
}

} // namespace intent::cpu
