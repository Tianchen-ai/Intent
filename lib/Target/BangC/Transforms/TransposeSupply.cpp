#include "PassDetail.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::bangc {
namespace {

enum class SupplyKind { Frontier, Cast, Binary, Slice, Broadcast };
struct SupplyNode {
  SupplyKind kind;
  Value value;
  Operation *operation;
  Operation *read;
  SmallVector<unsigned> inputs;
  Value shortInput;
};

MemRefType transposeType(MemRefType type) {
  return MemRefType::get({type.getDimSize(1), type.getDimSize(0)},
      type.getElementType(), MemRefLayoutAttrInterface{}, type.getMemorySpace());
}

// These nodes refer to actual storage versions at their read points. No result
// is reused merely because an allocation SSA name appears again after a write.
class TransposeSupply {
public:
  TransposeSupply(func::FuncOp function, Operation *terminal)
      : function(function), terminal(terminal), storage(function),
        dominance(function), relations(function) {}

  bool query() {
    Value input;
    if (auto transpose = dyn_cast<dsa::TransposeOp>(terminal)) {
      input = transpose.getInput();
      auto type = cast<MemRefType>(input.getType());
      if (!relations.equal(transpose.getRows(), type.getDimSize(0)) ||
          !relations.equal(transpose.getColumns(), type.getDimSize(1)))
        return reject("transpose does not cover its complete input");
      destination = transpose.getOutput();
      auto allocation = destination.getDefiningOp<memref::AllocaOp>();
      if (!allocation || allocation.getType() != transposeType(type) ||
          storage.uniqueWriter(destination) != terminal ||
          !storage.aliases(destination).complete)
        return reject("transpose destination is not a closed single-writer allocation");
    } else {
      auto prepare = cast<dsa::PrepareMatrixOp>(terminal);
      if (prepare.getInputTransposed()) return reject("matrix input is already transposed");
      input = prepare.getInput();
    }
    root = collect(input, terminal, false);
    if (!root) return false;
    if (frontiers != 1) return reject("supply has more than one transpose frontier");
    if (removedBroadcastBytes == 0) return reject("supply has no removable broadcast");
    if (removedBroadcastBytes <= extraTransposeBytes(input))
      return reject("broadcast savings do not cover the wider transpose");

    llvm::SmallPtrSet<Operation *, 16> replaced;
    for (const SupplyNode &node : nodes)
      if (node.kind != SupplyKind::Frontier) replaced.insert(node.operation);
    replaced.insert(terminal);
    for (const SupplyNode &node : nodes) {
      if (node.kind == SupplyKind::Frontier) continue;
      auto allocation = storage.uniqueOrigin(node.value);
      if (!allocation) return reject("supply has no unique storage origin", node.operation);
      auto accesses = storage.accesses(allocation);
      if (!accesses.complete) return reject("allocation access set is incomplete", node.operation);
      for (const auto &entry : accesses.entries) {
        if (!isa<MemoryEffects::Read>(entry.effect.getEffect())) continue;
        Operation *reader = entry.operation;
        Value memory = entry.effect.getValue();
        if (!memory) return reject("allocation has an unbound read", reader);
        Operation *writer = storage.lastWriterBefore(memory, reader);
        if (writer == node.operation &&
            (!replaced.contains(reader) || !dominance.properlyDominates(writer, reader)))
          return reject("selected storage version has an unselected or undominated reader", reader);
        // A read before this writer in a repeating scope can still observe the
        // previous iteration's version. Only a known different writer closes
        // that path; allocation placement and static read dominance do not.
        if (!writer)
          return reject("reader may observe the selected version across a backedge", reader);
        if (writer != node.operation) {
          OpOperand *output = dsa::detail::completeLocalOutput(writer);
          if (!output || !dsa::isCompleteStorageViewOf(output->get(), allocation))
            return reject("intervening writer does not replace the complete storage version", writer);
        }
      }
    }
    if (destination) {
      Operation *anchor = nodes[*root].operation;
      for (Operation *user : storage.aliases(destination).users)
        if (user != terminal && !dominance.properlyDominates(anchor, user))
          return reject("replacement does not dominate a transpose destination user", user);
    }
    return true;
  }

  bool apply(dsa::ConfigurationAttr config) {
    struct Insertions : OpBuilder::Listener {
      SmallVector<Operation *> operations;
      void notifyOperationInserted(Operation *operation, OpBuilder::InsertPoint) override {
        operations.push_back(operation);
      }
    } inserted;
    OpBuilder builder(function.getContext(), &inserted);
    SmallVector<Value> values;
    for (const SupplyNode &node : nodes) {
      builder.setInsertionPoint(node.kind == SupplyKind::Frontier ? node.read : node.operation);
      Location loc = node.operation->getLoc();
      auto originalType = cast<MemRefType>(node.value.getType());
      if (node.kind == SupplyKind::Broadcast) {
        auto shortType = MemRefType::get({1, originalType.getDimSize(0)}, originalType.getElementType(),
            MemRefLayoutAttrInterface{}, originalType.getMemorySpace());
        auto view = dsa::materializeCollectiveView(builder, loc, node.shortInput, shortType);
        assert(succeeded(view) && "complete short supply has unchanged element count");
        values.push_back(*view);
        continue;
      }
      auto type = transposeType(originalType);
      Value output = allocate(builder, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
      if (node.kind == SupplyKind::Frontier) {
        Value rows = builder.create<arith::ConstantIndexOp>(loc, originalType.getDimSize(0));
        Value columns = builder.create<arith::ConstantIndexOp>(loc, originalType.getDimSize(1));
        builder.create<dsa::TransposeOp>(loc, node.value, output, rows, columns);
      } else if (node.kind == SupplyKind::Cast) {
        auto copy = cast<dsa::CastOp>(builder.clone(*node.operation));
        copy.getInputMutable().assign(values[node.inputs[0]]);
        copy.getOutputMutable().assign(output);
      } else if (node.kind == SupplyKind::Binary) {
        Value rhs = node.inputs.size() == 2 ? values[node.inputs[1]] : node.shortInput;
        if (auto memory = dyn_cast<MemRefType>(rhs.getType()); memory && node.inputs.size() == 1) {
          auto view = dsa::materializeCollectiveView(builder, loc, rhs, transposeType(memory));
          assert(succeeded(view) && "compact transpose only changes unit axes");
          rhs = *view;
        }
        auto copy = cast<dsa::BinaryOp>(builder.clone(*node.operation));
        copy.getLhsMutable().assign(values[node.inputs[0]]);
        copy.getRhsMutable().assign(rhs);
        copy.getOutputMutable().assign(output);
        copy->removeAttr("bangc.implementation");
        if (auto rhsType = dyn_cast<MemRefType>(rhs.getType()); rhsType && rhsType != type)
          copy->setAttr("bangc.implementation", builder.getStringAttr(
              rhsType.getDimSize(0) == 1 ? "cycle" : "row_scalar"));
      } else {
        auto load = cast<dsa::LoadTileOp>(node.operation);
        Value source = values[node.inputs[0]];
        auto sourceType = cast<MemRefType>(load.getSource().getType());
        Value oldStride = builder.create<arith::ConstantIndexOp>(loc, sourceType.getDimSize(1));
        Value newStride = builder.create<arith::ConstantIndexOp>(loc, sourceType.getDimSize(0));
        Value row = builder.create<arith::DivSIOp>(loc, load.getOffset(), oldStride);
        Value column = builder.create<arith::RemSIOp>(loc, load.getOffset(), oldStride);
        Value offset = builder.create<arith::AddIOp>(loc, row,
            builder.create<arith::MulIOp>(loc, column, newStride));
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        auto copy = cast<dsa::LoadTileOp>(builder.clone(*load));
        copy.getSourceMutable().assign(source);
        copy.getOutputMutable().assign(output);
        copy.getOffsetMutable().assign(offset);
        copy.getRowStrideMutable().assign(newStride);
        copy.getColumnStrideMutable().assign(one);
        copy.getRowsMutable().assign(load.getColumns());
        copy.getColumnsMutable().assign(load.getRows());
      }
      values.push_back(output);
    }

    SmallVector<Value> originalTerminalOperands(terminal->getOperands());
    auto originalTerminalAttrs = terminal->getAttrDictionary();
    Value result = values[*root];
    if (auto prepare = dyn_cast<dsa::PrepareMatrixOp>(terminal)) {
      prepare.getInputMutable().assign(result);
      prepare.getScratchMutable().clear();
      prepare.setInputTransposed(true);
    }
    llvm::SmallPtrSet<Operation *, 16> removed;
    for (const auto &node : nodes)
      if (node.kind != SupplyKind::Frontier) removed.insert(node.operation);
    if (destination) removed.insert(terminal);
    struct Saved {
      Operation *operation, *next;
      Block *block;
      SmallVector<Value> operands;
    };
    SmallVector<Saved> saved;
    function.walk([&](Operation *operation) {
      if (removed.contains(operation))
        saved.push_back({operation, operation->getNextNode(), operation->getBlock(),
                         llvm::to_vector(operation->getOperands())});
    });
    for (auto &entry : saved) {
      entry.operation->setOperands(ValueRange{});
      entry.operation->remove();
    }
    SmallVector<OpOperand *> destinationUses;
    if (destination) {
      for (OpOperand &use : destination.getUses()) destinationUses.push_back(&use);
      for (OpOperand *use : destinationUses) use->set(result);
    }

    bool fits = storageFitsBudget(function, config, measureStorage(function));
    if (!fits) {
      for (OpOperand *use : destinationUses) use->set(destination);
      if (!destination) {
        terminal->setOperands(originalTerminalOperands);
        terminal->setAttrs(originalTerminalAttrs);
      }
      for (auto &entry : saved) entry.operation->setOperands(entry.operands);
      for (auto &entry : llvm::reverse(saved)) {
        OpBuilder restore(function.getContext());
        if (entry.next) restore.setInsertionPoint(entry.next);
        else restore.setInsertionPointToEnd(entry.block);
        restore.insert(entry.operation);
      }
      for (Operation *operation : llvm::reverse(inserted.operations)) operation->erase();
      return reject("replacement exceeds local storage budget");
    }
    for (auto &entry : saved) entry.operation->destroy();
    return true;
  }

  void emitRejection() {
    auto module = function->getParentOfType<ModuleOp>();
    auto options = module->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr);
    if (!options.getOptimizationRemarks() || rejection.empty()) return;
    auto remark = terminal->emitRemark("transpose-supply: ");
    remark << rejection;
    if (rejectedAt && rejectedAt != terminal)
      remark.attachNote(rejectedAt->getLoc()) << "supply operation: " << rejectedAt->getName();
  }

private:
  bool reject(StringRef reason, Operation *operation = nullptr) {
    if (rejection.empty()) {
      rejection = reason;
      rejectedAt = operation;
    }
    return false;
  }

  std::optional<unsigned> rejectNode(StringRef reason, Operation *operation) {
    reject(reason, operation);
    return std::nullopt;
  }

  bool local(Value value) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape() || type.getNumElements() <= 0 ||
        !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
        (!type.getElementType().isF16() && !type.getElementType().isBF16() &&
         !type.getElementType().isF32()))
      return false;
    Value origin = storage.uniqueOrigin(value);
    return origin && origin.getDefiningOp<memref::AllocaOp>() &&
        storage.aliases(origin).complete && dsa::isCompleteStorageViewOf(value, origin);
  }

  bool nestedScope(Operation *writer, Operation *reader) {
    Operation *cursor = reader;
    while (cursor->getBlock() != writer->getBlock()) {
      auto loop = dyn_cast<scf::ForOp>(cursor->getParentOp());
      if (!loop) return false;
      auto lower = relations.interval(loop.getLowerBound());
      auto upper = relations.interval(loop.getUpperBound());
      auto step = relations.interval(loop.getStep());
      if (!lower || !upper || !step || step->first <= 0 || lower->second >= upper->first)
        return false;
      cursor = loop;
    }
    return writer->isBeforeInBlock(cursor);
  }

  bool rectangle(dsa::LoadTileOp load) {
    auto source = cast<MemRefType>(load.getSource().getType());
    if (!local(load.getSource()) || !relations.equal(load.getRowStride(), source.getDimSize(1)) ||
        !relations.equal(load.getColumnStride(), 1)) return false;
    auto offset = relations.interval(load.getOffset());
    auto rows = relations.interval(load.getRows()), columns = relations.interval(load.getColumns());
    if (!offset || !rows || !columns || offset->first < 0 || rows->first < 0 || columns->first < 0)
      return false;
    // Full rows/columns are kept separate from the linear descriptor offset.
    // Only a single known starting coordinate can authorize swapping its axes.
    if (offset->first != offset->second) return false;
    int64_t row = offset->first / source.getDimSize(1);
    int64_t column = offset->first % source.getDimSize(1);
    return rows->second <= source.getDimSize(0) - row &&
           columns->second <= source.getDimSize(1) - column;
  }

  std::optional<unsigned> collect(Value value, Operation *read, bool shortOperand) {
    if (!local(value)) return rejectNode("supply is not a complete local tile", read);
    Operation *writer = storage.lastWriterBefore(value, read);
    if (!writer) return rejectNode("current supply writer is not known", read);
    if (auto found = known.find({value, writer}); found != known.end()) {
      SupplyNode &node = nodes[found->second];
      // A compact broadcast keeps its source view, not the expanded snapshot.
      // Every replacement read must still observe the original source version.
      if (node.kind == SupplyKind::Broadcast &&
          !storage.readStable(node.shortInput, node.operation, read))
        return rejectNode("compact broadcast source changed before a reader", read);
      if (node.kind == SupplyKind::Frontier && node.read != read) {
        if (dominance.properlyDominates(node.read, read)) {
          if (!storage.readStable(value, node.read, read))
            return rejectNode("transpose frontier changed before a reader", read);
        } else if (node.read->getBlock() == read->getBlock() &&
                   dominance.properlyDominates(read, node.read) &&
                   storage.readStable(value, read, node.read)) {
          // Use an already selected earlier read in the same execution scope.
          // Do not repair dominance by speculating into an enclosing scope.
          node.read = read;
        } else return rejectNode("transpose frontier has no dominating read anchor", read);
      }
      return found->second;
    }
    auto type = cast<MemRefType>(value.getType());
    Value written;
    if (auto op = dyn_cast<dsa::CastOp>(writer)) written = op.getOutput();
    if (auto op = dyn_cast<dsa::BinaryOp>(writer)) written = op.getOutput();
    if (auto op = dyn_cast<dsa::LoadTileOp>(writer)) written = op.getOutput();
    if (auto op = dyn_cast<dsa::BroadcastRowsOp>(writer)) written = op.getOutput();
    if (!written || written.getType() != type || !local(written))
      return rejectNode("writer does not define the complete typed supply", writer);
    if (writer->getNumRegions() || writer->getNumResults() ||
        dsa::requiredCompletion(writer) != dsa::CompletionScope::Immediate ||
        !nestedScope(writer, read))
      return rejectNode("writer completion or execution scope cannot be transported", writer);

    SupplyNode node{SupplyKind::Frontier, value, writer, read, {}, {}};
    if (auto broadcast = dyn_cast<dsa::BroadcastRowsOp>(writer); broadcast && shortOperand) {
      auto source = cast<MemRefType>(broadcast.getInput().getType());
      if (!relations.equal(broadcast.getRows(), type.getDimSize(0)) ||
          !relations.equal(broadcast.getColumns(), type.getDimSize(1)) ||
          !relations.equal(broadcast.getOffset(), 0) ||
          !local(broadcast.getInput()) || source.getNumElements() != type.getDimSize(0) ||
          !storage.readStable(broadcast.getInput(), writer, read))
        return rejectNode("broadcast is not a complete stable compact supply", writer);
      auto scratchUses = storage.accesses(broadcast.getScratch());
      if (!scratchUses.complete || llvm::any_of(scratchUses.entries, [](const auto &entry) {
            return isa<MemoryEffects::Read>(entry.effect.getEffect());
          })) return rejectNode("broadcast scratch has an observable read", writer);
      node.kind = SupplyKind::Broadcast;
      node.shortInput = broadcast.getInput();
      removedBroadcastBytes += 2 * type.getNumElements() * (type.getElementType().getIntOrFloatBitWidth() / 8);
    } else if (auto conversion = dyn_cast<dsa::CastOp>(writer); conversion && !shortOperand) {
      auto input = collect(conversion.getInput(), writer, false);
      if (!input) return std::nullopt;
      node.kind = SupplyKind::Cast;
      node.inputs.push_back(*input);
    } else if (auto binary = dyn_cast<dsa::BinaryOp>(writer); binary && !shortOperand &&
               !binary.getScratch() && supportedScalarBinary(binary.getKind()) &&
               !binary.getApproximate() && !binary.getFlushToZero() &&
               (type.getElementType().isF16() || type.getElementType().isF32())) {
      auto lhs = collect(binary.getLhs(), writer, false);
      if (!lhs) return std::nullopt;
      node.kind = SupplyKind::Binary;
      node.inputs.push_back(*lhs);
      auto rhs = dyn_cast<MemRefType>(binary.getRhs().getType());
      if (!rhs) node.shortInput = binary.getRhs();
      else if (rhs != type) {
        if (!local(binary.getRhs()) || rhs.getDimSize(1) != 1 ||
            rhs.getDimSize(0) != type.getDimSize(0))
          return rejectNode("binary compact operand does not follow the transposed axis", writer);
        node.shortInput = binary.getRhs();
      } else {
        auto input = collect(binary.getRhs(), writer, true);
        if (!input) return std::nullopt;
        node.inputs.push_back(*input);
      }
    } else if (auto load = dyn_cast<dsa::LoadTileOp>(writer); load && !shortOperand &&
               !load.getAsynchronous()) {
      Operation *producer = local(load.getSource()) ? storage.lastWriterBefore(load.getSource(), writer) : nullptr;
      if (producer && isa<dsa::CastOp, dsa::BinaryOp>(producer) && rectangle(load)) {
        auto input = collect(load.getSource(), writer, false);
        if (!input) return std::nullopt;
        node.kind = SupplyKind::Slice;
        node.inputs.push_back(*input);
      } else {
        ++frontiers;
        frontierBytes += type.getNumElements() * (type.getElementType().getIntOrFloatBitWidth() / 8);
      }
    } else return rejectNode("writer has no supported local axis transport", writer);

    unsigned position = nodes.size();
    nodes.push_back(std::move(node));
    known[{value, writer}] = position;
    return position;
  }

  uint64_t extraTransposeBytes(Value input) {
    auto type = cast<MemRefType>(input.getType());
    uint64_t original = type.getNumElements() * (type.getElementType().getIntOrFloatBitWidth() / 8);
    return frontierBytes > original ? frontierBytes - original : 0;
  }

  func::FuncOp function;
  Operation *terminal;
  dsa::StorageAnalysis storage;
  DominanceInfo dominance;
  dsa::LocalSupplyRelations relations;
  SmallVector<SupplyNode> nodes;
  DenseMap<std::pair<Value, Operation *>, unsigned> known;
  std::optional<unsigned> root;
  Value destination;
  unsigned frontiers = 0;
  uint64_t removedBroadcastBytes = 0, frontierBytes = 0;
  StringRef rejection;
  Operation *rejectedAt = nullptr;
};

} // namespace

bool propagateLocalTransposeSupply(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<Operation *> consumers;
  function.walk([&](Operation *operation) {
    if (isa<dsa::TransposeOp, dsa::PrepareMatrixOp>(operation)) consumers.push_back(operation);
  });
  for (Operation *consumer : consumers) {
    TransposeSupply supply(function, consumer);
    if (supply.query() && supply.apply(config)) return true;
    supply.emitRejection();
  }
  return false;
}

} // namespace intent::bangc
