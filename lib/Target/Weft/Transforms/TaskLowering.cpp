#include "TaskConversion.h"
#include "TaskInterface.h"
#include "Views.h"
#include "ScalarValues.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
using namespace task_detail;

TaskConversion::TaskConversion(func::FuncOp function, ModuleOp output,
             const llvm::DenseMap<Value, intent::QuantFormat> &formats,
             const cpu::ImplementationRegistry &implementations)
    : sourceFunction(function), relations(function),
      output(output), b(output.getContext()),
      formats(formats), implementations(implementations) {
  auto interface = getPublicInterface(function);
  for (auto [argument, schema] : llvm::zip(function.getArguments(), interface.getArguments())) {
    auto view = dyn_cast<intent::ViewType>(cast<PublicParameterAttr>(schema).getType());
    if (!view) continue;
    auto memory = cast<MemRefType>(argument.getType());
    for (unsigned axis = 0; axis < memory.getRank(); ++axis)
      if (memory.isDynamicDim(axis)) extentIds.try_emplace(publicViewDimensions(view)[axis], extentIds.size() + 1);
  }
  initializeAxes();
}

void TaskConversion::initializeAxes() {
  relations = cpu::AxisRelations(sourceFunction);
  axisProjection.clear();
  auto join = [&](Value lhs, Value rhs) {
    int64_t a = physicalAxis(relations.axes(lhs).back());
    int64_t c = physicalAxis(relations.axes(rhs).back());
    if (a != c) axisProjection[std::max(a, c)] = std::min(a, c);
  };
  sourceFunction.walk([&](cpu::QuantizeOp op) { join(op.getInput(), op.getOutput()); });
  sourceFunction.walk([&](cpu::QuantizedDotOp op) { join(op.getLhs(), op.getRhs()); });
  nextAxis = 1;
  sourceFunction.walk([&](Operation *operation) {
    for (Value value : operation->getOperands())
      if (isa<MemRefType>(value.getType()))
        for (int64_t axis : relations.axes(value)) nextAxis = std::max(nextAxis, axis + 1);
  });
}

SmallVector<Value> TaskConversion::shapeArguments(cpu::TasksOp task, OpBuilder &builder) {
  auto function = sourceFunction;
  SmallVector<Value> result(extentIds.size());
  auto interface = getPublicInterface(function);
  for (BlockArgument argument : function.getArguments()) {
    auto memory = dyn_cast<MemRefType>(argument.getType());
    if (!memory) continue;
    auto ids = publicViewDimensions(getPublicView(interface, argument.getArgNumber()));
    for (auto [axis, extent] : llvm::enumerate(memory.getShape()))
      if (ShapedType::isDynamic(extent))
        result[extentIds.at(ids[axis]) - 1] = builder.create<memref::DimOp>(function.getLoc(), argument, axis);
  }
  for (const CapturedExtent &extent : capturedExtents)
    result.push_back(builder.create<memref::DimOp>(task.getLoc(),
        task.getCaptures()[extent.argument - 1], extent.axis));
  return result;
}

std::optional<int64_t> TaskConversion::capturedShapeSymbol(
    const ValueBoundsConstraintSet::Variable &extent) {
  for (auto [number, captured] : llvm::enumerate(capturedExtents))
    for (Value memory : {currentTask.getCaptures()[captured.argument - 1],
                         Value(currentTask.getBody().front().getArgument(captured.argument))})
      if (cpu::haveEqualExtents(extent,
              ValueBoundsConstraintSet::Variable(memory, captured.axis)))
        return extentIds.size() + number + 1;
  return std::nullopt;
}

LogicalResult TaskConversion::lower(cpu::TasksOp tasks, StringRef name,
                    SmallVectorImpl<unsigned> &argumentPositions) {
  storage = std::make_unique<cpu::StorageAnalysis>(sourceFunction);
  currentTask = tasks;
  capturedExtents.clear();
  localShapeBindings.clear();
  values.clear(); locals.clear(); localReferences.clear(); controlOwners.clear(); shapeValues.clear(); readOnlySupplies.clear(); viewAxes.clear();
  Location loc = tasks.getLoc();
  SmallVector<Attribute> names, accesses, symbols;
  SmallVector<int64_t> aliases;
  SmallVector<Type> types;
  auto memoryEffects = storage->effects(tasks);
  if (!memoryEffects.complete)
    return tasks.emitError("Weft task requires complete storage effects");
  auto formatStatus = tasks.walk([&](Operation *operation) {
    for (Value memory : operation->getOperands()) {
      if (!isa<MemRefType>(memory.getType())) continue;
      auto origins = storage->origins(memory);
      std::optional<intent::QuantFormat> selected;
      bool denseOrigin = false;
      for (Value origin : origins.values) {
        auto found = formats.find(origin);
        if (found == formats.end()) { denseOrigin = true; continue; }
        if (selected && *selected != found->second) {
          operation->emitError("Weft view merges incompatible encoded storage");
          return WalkResult::interrupt();
        }
        selected = found->second;
      }
      if (selected && (!origins.complete || denseOrigin)) {
        operation->emitError("Weft encoded view requires one representation across all storage origins");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (formatStatus.wasInterrupted()) return failure();
  for (unsigned position : argumentPositions) {
    Value capture = position == 0 ? tasks.getBody().front().getArgument(0) : tasks.getCaptures()[position - 1];
    names.push_back(b.getStringAttr(position == 0 ? "coordinate" : "capture_" + std::to_string(position - 1)));
    if (auto memory = dyn_cast<MemRefType>(capture.getType())) {
      if (!cpu::isContiguousDescriptor(capture))
        return tasks.emitError("native task capture requires a proved contiguous descriptor");
      auto ids = memoryAxes(capture);
      llvm::SmallSet<int64_t, 8> unique(ids.begin(), ids.end());
      if (unique.size() != ids.size())
        return tasks.emitError("Weft view requires independent logical axes; this reuse needs an explicit axis projection");
      SmallVector<int64_t> dimensions;
      for (auto [axis, extent] : llvm::enumerate(memory.getShape())) {
        if (ShapedType::isDynamic(extent)) {
          auto symbol = shapeSymbol(ValueBoundsConstraintSet::Variable(capture, axis));
          if (!symbol) {
            capturedExtents.push_back({position, static_cast<unsigned>(axis)});
            symbol = extentIds.size() + capturedExtents.size();
          }
          dimensions.push_back(-*symbol);
        } else dimensions.push_back(extent);
      }
      auto format = quantizedFormat(capture);
      if (format) {
        int64_t bytes = *format == intent::QuantFormat::Q4K ? 144 : 292;
        if (memory.getShape().back() != bytes)
          return tasks.emitError("encoded capture requires a statically complete contiguous record byte span");
        dimensions.back() = 256;
      }
      types.push_back(wk::ViewType::get(b.getContext(), encoding(capture), array(dimensions), array(ids)));
      bool reads = false, writes = false;
      for (const cpu::StorageEffect &entry : memoryEffects.entries) {
        Value affected = entry.effect.getValue();
        if (!affected && isa<cpu::AtomicLoadOp, cpu::AtomicStoreOp,
                             cpu::AtomicRMWOp, cpu::AtomicCompareExchangeOp>(entry.operation))
          continue;
        if (!affected || !storage->disjoint(affected, capture)) {
          reads |= isa<MemoryEffects::Read>(entry.effect.getEffect());
          writes |= isa<MemoryEffects::Write>(entry.effect.getEffect());
        }
      }
      accesses.push_back(b.getStringAttr(reads ? (writes ? "readwrite" : "read")
                                             : (writes ? "write" : "none")));
      aliases.push_back(0);
    } else {
      types.push_back(scalarView(capture.getType()));
      accesses.push_back(b.getStringAttr("read"));
      aliases.push_back(0);
    }
  }
  for (unsigned symbol = 1; symbol <= extentIds.size() + capturedExtents.size(); ++symbol)
    symbols.push_back(b.getStringAttr("shape_" + std::to_string(symbol)));
  b.setInsertionPointToEnd(output.getBody());
  auto kernel = b.create<wk::KernelOp>(loc, name, b.getArrayAttr(names),
      b.getArrayAttr(accesses), array(aliases), b.getArrayAttr(symbols), "Intent CPU task");
  Block *body = new Block;
  kernel.getBody().push_back(body);
  for (Type type : types) body->addArgument(type, loc);
  b.setInsertionPointToStart(body);
  for (Attribute symbol : symbols)
    shapeValues.push_back(b.create<wk::SymbolOp>(loc, b.getIndexType(), cast<StringAttr>(symbol),
        b.getStringAttr("shape"), array({})));
  b.create<wk::RootDomainOp>(loc,
      wk::DomainType::get(b.getContext(), "root", 0, -1, 0, "root", "exact"));
  for (auto [position, target] : llvm::zip(argumentPositions, body->getArguments())) {
    Value source = tasks.getBody().front().getArgument(position);
    if (isa<MemRefType>(source.getType())) values.map(source, target);
    else {
      Value scalar = b.create<wk::SliceOp>(loc,
          sliceType(target, {}, {}),
          target, ValueRange{index(loc, 0)}, b.getArrayAttr({b.getStringAttr("index")}));
      Type storage = source.getType().isIndex()
          ? Type(IntegerType::get(b.getContext(), 64, IntegerType::Signed))
          : nativeScalarType(source.getType());
      Value loaded = b.create<wk::AdmitOp>(loc, storage, scalar);
      if (source.getType().isIndex()) loaded = b.create<wk::CastOp>(loc, b.getIndexType(), loaded);
      values.map(source, loaded);
    }
  }
  if (failed(block(tasks.getBody().front()))) return failure();
  b.create<wk::ReturnOp>(loc, ValueRange{});
  return finalizeTaskInterface(kernel, argumentPositions);
}

LogicalResult TaskConversion::block(Block &source) {
  auto enclosingSupplies = readOnlySupplies;
  for (Operation &operation : source.without_terminator())
    if (failed(lower(&operation))) {
      readOnlySupplies = std::move(enclosingSupplies);
      return failure();
    }
  // A supplied immutable input may serve later consumers in this block, but
  // a branch/loop-local SSA value cannot escape into its enclosing scope.
  readOnlySupplies = std::move(enclosingSupplies);
  return success();
}

LogicalResult TaskConversion::lower(Operation *operation) {
  operandReads.clear();
  auto expanded = expandSelected(operation);
  if (failed(expanded)) return failure();
  if (*expanded) return success();
  if (isa<memref::AllocOp, memref::AllocaOp>(operation))
    return allocateLocal(operation->getResult(0));
  if (isa<memref::SubViewOp, memref::CastOp>(operation) || isAxisView(operation)) {
    Value result = operation->getResult(0);
    if (isAxisView(operation) && !(isLocal(result) && isStaticShapeView(operation)) &&
        failed(queryAxisView(result))) return failure();
    if (!isLocal(result) && failed(view(result))) return failure();
    return success();
  }
  if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(operation)) return lower(metadata);
  if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) { locals.erase(localRoot(dealloc.getMemref())); return success(); }
  if (auto release = dyn_cast<bufferization::DeallocOp>(operation)) return lower(release);
  if (auto dim = dyn_cast<memref::DimOp>(operation)) return lower(dim);
  if (auto copy = dyn_cast<memref::CopyOp>(operation)) return lower(copy);
  if (auto fill = dyn_cast<linalg::FillOp>(operation)) return lower(fill);
  if (auto store = dyn_cast<memref::StoreOp>(operation)) return lower(store);
  if (auto genericOp = dyn_cast<linalg::GenericOp>(operation)) return generic(genericOp);
  if (auto reduce = dyn_cast<cpu::ReduceOp>(operation)) return reduction(reduce);
  if (auto conditional = dyn_cast<scf::IfOp>(operation)) return lower(conditional);
  if (auto loop = dyn_cast<scf::ForOp>(operation)) return lower(loop);
  if (auto loop = dyn_cast<scf::WhileOp>(operation)) return lower(loop);
  if (auto load = dyn_cast<memref::LoadOp>(operation)) return lower(load);
  if (auto atomic = dyn_cast<cpu::AtomicRMWOp>(operation)) return lower(atomic);
  if (operation->getNumResults() != 1) return operation->emitError("CPU task operation has no Weft representation");
  auto value = expression(operation, values);
  if (failed(value)) return failure();
  values.map(operation->getResult(0), *value);
  return success();
}

TaskLowering::TaskLowering(func::FuncOp function, ModuleOp output,
    const llvm::DenseMap<Value, intent::QuantFormat> &formats,
    const cpu::ImplementationRegistry &implementations)
    : implementation(std::make_unique<TaskConversion>(function, output, formats, implementations)) {}

TaskLowering::~TaskLowering() = default;

LogicalResult TaskLowering::normalizeComputations() {
  return implementation->normalizeComputations();
}

SmallVector<Value> TaskLowering::shapeArguments(cpu::TasksOp task, OpBuilder &builder) {
  return implementation->shapeArguments(task, builder);
}

LogicalResult TaskLowering::lower(cpu::TasksOp task, StringRef name,
                                  SmallVectorImpl<unsigned> &argumentPositions) {
  return implementation->lower(task, name, argumentPositions);
}


} // namespace intent::weft_provider
