#include "Lowering.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Target/DSA/BangC/Physical/IR/Views.h"
#include "Intent/Serialization/NativeABI.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

namespace intent::bangc::block_lowering {
Lowering::Lowering(func::FuncOp source, ModuleOp output, physical::ConfigurationAttr config,
    DictionaryAttr parameters, DictionaryAttr shapes, DictionaryAttr strides,
    block::ProgramInterface interface)
    : loc(source.getLoc()), source(source), output(output), b(output.getContext()),
      config(config), parameters(parameters), shapes(shapes), strides(strides), interface(interface) {}

FailureOr<int64_t> Lowering::constant(block::PhysicalExprAttr expression) {
  auto result = block::evaluatePhysicalExpression(expression, [&](block::PhysicalExprAttr leaf) -> std::optional<int64_t> {
    if (leaf.getKind() == block::PhysicalExprKind::Parameter) {
      auto name = leaf.getParameterReference().getName();
      if (auto value = parameters.getAs<IntegerAttr>(name)) return value.getInt();
      auto declaration = block::lookupParameter(source, name);
      if (!declaration || !declaration.isDeferred()) return std::nullopt;
      auto binding = declaration.getBinding();
      std::optional<int64_t> bound;
      if (binding.getCoverageBound()) {
        auto value = constant(binding.getCoverageBound());
        if (succeeded(value)) bound = *value;
      } else if (binding.getDimension()) {
        auto argument = interface.dimension(binding.getDimension().getInt());
        auto found = argument ? staticArguments.find(block::getArgumentReference(argument)) : staticArguments.end();
        if (found != staticArguments.end()) bound = found->second;
      }
      if (!bound || *bound < 0 || *bound > (int64_t(1) << 60)) return std::nullopt;
      for (int64_t candidate : declaration.getCandidates().asArrayRef())
        if (candidate >= *bound) return candidate;
      return std::nullopt;
    }
    auto found = staticArguments.find(leaf.getArgumentReference());
    return found == staticArguments.end() ? std::nullopt : std::optional<int64_t>(found->second);
  });
  if (!result) return failure();
  return *result;
}

FailureOr<SmallVector<int64_t>> Lowering::shape(Type type) {
  ArrayAttr expressions;
  if (auto fragment = dyn_cast<block::FragmentType>(type)) expressions = fragment.getShape();
  else if (auto buffer = dyn_cast<block::BufferType>(type)) expressions = buffer.getShape();
  else return SmallVector<int64_t>();
  SmallVector<int64_t> result;
  for (Attribute attribute : expressions) {
    auto extent = constant(cast<block::PhysicalExprAttr>(attribute));
    if (failed(extent) || *extent <= 0)
      return emitError(loc, "BANG C local storage requires a positive specialized fragment extent"), failure();
    result.push_back(*extent);
  }
  return result;
}

FailureOr<Value> Lowering::expression(block::PhysicalExprAttr expr) {
  if (auto fixed = constant(expr); succeeded(fixed)) return index(*fixed);
  if (expr.getKind() == block::PhysicalExprKind::Dimension || expr.getKind() == block::PhysicalExprKind::ScalarABI) {
    Value result = get(interface.resolve(expr));
    if (!result) return failure();
    return result;
  }
  SmallVector<Value> operands;
  for (Attribute child : expr.getOperands()) {
    auto value = expression(cast<block::PhysicalExprAttr>(child));
    if (failed(value)) return failure();
    operands.push_back(*value);
  }
  using Kind = block::PhysicalExprKind;
  switch (expr.getKind()) {
  case Kind::Add: return Value(b.createOrFold<arith::AddIOp>(loc, operands[0], operands[1]));
  case Kind::Multiply: return Value(b.createOrFold<arith::MulIOp>(loc, operands[0], operands[1]));
  case Kind::Subtract: return Value(b.createOrFold<arith::SubIOp>(loc, operands[0], operands[1]));
  case Kind::FloorDiv: return Value(b.create<arith::FloorDivSIOp>(loc, operands[0], operands[1]));
  case Kind::CeilDiv: return Value(b.create<arith::CeilDivSIOp>(loc, operands[0], operands[1]));
  case Kind::Minimum: return Value(b.create<arith::MinSIOp>(loc, operands[0], operands[1]));
  case Kind::Maximum: return Value(b.create<arith::MaxSIOp>(loc, operands[0], operands[1]));
  case Kind::Select: return Value(b.create<arith::SelectOp>(loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, operands[0], index(0)), operands[1], operands[2]));
  default: return emitError(loc, "BANG C physical expression is not bound by this configuration"), failure();
  }
}

LogicalResult Lowering::run() {
  if (auto options = readCompileOptions(source); succeeded(options) && options->getOptimizationRemarks())
    source.emitRemark("BANG C Block configuration: ") << parameters;
  DenseMap<int64_t, int64_t> dimensions;
  for (Attribute attribute : interface.getPublicInterface().getArguments()) {
    auto parameter = cast<PublicParameterAttr>(attribute);
    auto view = dyn_cast<intent::ViewType>(parameter.getType());
    auto binding = shapes.getAs<DenseI64ArrayAttr>(parameter.getName());
    if (!view || !binding) continue;
    auto tensor = publicViewTensor(view);
    auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
    if (binding.size() != tensor.getRank()) return source.emitError("BANG C shape binding rank mismatch");
    for (auto [axis, extent] : llvm::enumerate(binding.asArrayRef())) {
      if (extent < 0 || identities[axis] <= 0) continue;
      auto [item, inserted] = dimensions.try_emplace(identities[axis], extent);
      if (!inserted && item->second != extent) return source.emitError("BANG C bound dimension has conflicting extents");
    }
  }
  SmallVector<Type> argumentTypes;
  SmallVector<Attribute> publicParameters;
  SmallVector<Value> originals;
  for (auto parameter : interface.getPublicInterface().getArguments()) {
    auto publicParameter = cast<PublicParameterAttr>(parameter);
    unsigned ordinal = publicParameters.size();
    auto original = interface.publicArgument(ordinal);
    originals.push_back(original);
    if (auto viewType = dyn_cast<intent::ViewType>(publicParameter.getType())) {
      auto tensor = publicViewTensor(viewType);
      SmallVector<int64_t> sizes(tensor.getShape());
      auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
      for (auto [axis, identity] : llvm::enumerate(identities.asArrayRef()))
        if (auto found = dimensions.find(identity); found != dimensions.end()) sizes[axis] = found->second;
      auto selected = shapes.getAs<DenseI64ArrayAttr>(publicParameter.getName());
      if (selected) {
        if (selected.size() != sizes.size()) return source.emitError("BANG C specialization rank mismatch");
        for (auto [axis, extent] : llvm::enumerate(selected.asArrayRef())) {
          if (extent < 0) continue;
          if (!ShapedType::isDynamic(sizes[axis]) && sizes[axis] != extent)
            return source.emitError("BANG C specialization contradicts a static extent");
          sizes[axis] = extent;
        }
      }
      SmallVector<int64_t> steps(sizes.size(), ShapedType::kDynamic);
      auto constraints = viewType.getConstraints();
      if (constraints.getHasStrides())
        for (auto [axis, attribute] : llvm::enumerate(constraints.getStrides()))
          if (auto value = dyn_cast<IntegerAttr>(attribute)) steps[axis] = value.getInt();
      if (auto selectedStrides = strides.getAs<DenseI64ArrayAttr>(publicParameter.getName())) {
        if (selectedStrides.size() != steps.size()) return source.emitError("BANG C specialized stride rank mismatch");
        SmallVector<Attribute> encoded;
        for (auto [axis, step] : llvm::enumerate(selectedStrides.asArrayRef())) {
          if (!ShapedType::isDynamic(steps[axis]) && steps[axis] != step)
            return source.emitError("BANG C specialization contradicts a stride constraint");
          steps[axis] = step;
          encoded.push_back(b.getI64IntegerAttr(step));
        }
        constraints = ViewConstraintsAttr::get(b.getContext(), true, b.getArrayAttr(encoded),
            constraints.getAlias(), constraints.getNoalias());
      }
      auto type = MemRefType::get(sizes, tensor.getElementType(), StridedLayoutAttr::get(b.getContext(), 0, steps));
      argumentTypes.push_back(type);
      extents[original] = sizes;
      auto specialized = intent::ViewType::get(b.getContext(),
          RankedTensorType::get(sizes, tensor.getElementType(), tensor.getEncoding()), viewType.getAccess(), constraints);
      publicParameters.push_back(PublicParameterAttr::get(b.getContext(), publicParameter.getName(), specialized));
    } else {
      argumentTypes.push_back(original.getType());
      publicParameters.push_back(parameter);
    }
  }
  b.setInsertionPointToEnd(output.getBody());
  entry = b.create<func::FuncOp>(loc, source.getName(), b.getFunctionType(argumentTypes, {}));
  entry->setAttr(interfaceAttr, InterfaceAttr::get(b.getContext(), b.getArrayAttr(publicParameters)));
  entry->setAttr(physical::entryRequirementsAttr, physical::EntryRequirementsAttr::get(b.getContext(), true));
  entry->setAttr("intent_bangc.configuration", config);
  entry->setAttr(physical::fullExtentDimensionsAttr, b.getDenseI64ArrayAttr({}));
  entry.addEntryBlock();
  b.setInsertionPointToStart(&entry.front());
  for (auto [original, argument] : llvm::zip(originals, entry.getArguments())) values.map(original, argument);
  for (auto argument : interface.arguments()) {
    auto binding = argument.binding;
    if (binding.getKind() == block::ArgumentKind::Public) continue;
    if (binding.getKind() == block::ArgumentKind::Workspace)
      return source.emitError("BANG C invocation workspace requires an external runtime binding");
    Value resource = get(interface.resolve(binding.getSource()));
    auto type = cast<MemRefType>(resource.getType());
    unsigned axis = binding.getAxis().getInt();
    int64_t fixed = ShapedType::kDynamic;
    Value value;
    if (binding.getKind() == block::ArgumentKind::Dimension) {
      fixed = type.getDimSize(axis);
      value = b.createOrFold<memref::DimOp>(loc, resource, axis);
    } else {
      SmallVector<int64_t> steps; int64_t offset;
      if (failed(type.getStridesAndOffset(steps, offset))) return failure();
      fixed = steps[axis];
      value = ShapedType::isDynamic(fixed) ? Value(b.create<physical::StrideOp>(loc, b.getIndexType(), resource, axis)) : index(fixed);
    }
    values.map(argument.value, value);
    if (!ShapedType::isDynamic(fixed)) staticArguments[binding.getReference()] = fixed;
  }
  auto space = source->getAttrOfType<ArrayAttr>(block::programSpaceAttr);
  Value total = index(1);
  SmallVector<Value> gridDimensions;
  for (Attribute attribute : space) {
    auto value = expression(cast<block::PhysicalExprAttr>(attribute));
    if (failed(value)) return failure();
    gridDimensions.push_back(*value);
    total = b.createOrFold<arith::MulIOp>(loc, total, *value);
  }
  Value task = b.create<physical::TaskIdOp>(loc, b.getIndexType());
  Value tasks = b.create<physical::TaskCountOp>(loc, b.getIndexType());
  auto loop = b.create<scf::ForOp>(loc, task, total, tasks);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value linear = loop.getInductionVar();
    programCoordinates.resize(gridDimensions.size());
    for (unsigned axis = 0; axis < gridDimensions.size(); ++axis) {
      programCoordinates[axis] = b.create<arith::RemSIOp>(loc, linear, gridDimensions[axis]);
      linear = b.create<arith::DivSIOp>(loc, linear, gridDimensions[axis]);
    }
    if (failed(operations(source.front()))) return failure();
  }
  b.create<func::ReturnOp>(loc);
  return success();
}

LogicalResult Lowering::operations(Block &sourceBlock) {
  for (Operation &op : sourceBlock.without_terminator())
    if (failed(operation(&op))) return failure();
  return success();
}

LogicalResult Lowering::operation(Operation *op) {
  loc = op->getLoc();
  if (op->getNumResults() == 1 && !materializing.contains(op))
    if (auto fragment = dyn_cast<block::FragmentType>(op->getResult(0).getType());
        fragment && (fragment.getElementType().isIndex() || fragment.getElementType().isInteger(1)))
      if (isa<block::MakeRangeOp, block::SplatOp, block::BroadcastOp, block::BinaryOp,
              block::UnaryOp, block::CompareOp, block::SelectOp, block::CastOp, block::BitcastOp>(op)) {
        deferred[op] = b.getInsertionBlock();
        return success();
      }
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    values.map(constant.getResult(), b.clone(*op)->getResult(0));
    return success();
  }
  if (auto parameter = dyn_cast<block::ParameterOp>(op)) {
    auto expression = block::PhysicalExprAttr::get(b.getContext(), block::PhysicalExprKind::Parameter,
        0, parameter.getReference(), b.getArrayAttr({}));
    auto fixed = constant(expression);
    if (failed(fixed)) return op->emitError("BANG C parameter lacks a concrete configuration binding: ") << parameter.getDeclaration();
    values.map(parameter.getResult(), index(*fixed));
    return success();
  }
  if (auto expr = dyn_cast<block::PhysicalExprOp>(op)) {
    auto value = expression(expr.getExpression());
    if (failed(value)) return failure();
    values.map(expr.getResult(), *value);
    return success();
  }
  if (auto id = dyn_cast<block::ProgramIdOp>(op)) {
    values.map(id.getResult(), programCoordinates[id.getAxis()]);
    return success();
  }
  if (auto coordinate = dyn_cast<block::WorksetCoordinateOp>(op)) {
    values.map(coordinate.getResult(), get(coordinate.getCoordinate()));
    return success();
  }
  if (auto dim = dyn_cast<block::DimOp>(op)) {
    values.map(dim.getResult(), b.createOrFold<memref::DimOp>(loc, get(dim.getView()), dim.getAxis()));
    return success();
  }
  if (auto range = dyn_cast<block::RangeOp>(op)) {
    records[range.getResult()] = {get(range.getStart()), get(range.getStop()), get(range.getStep())};
    return success();
  }
  if (auto bound = dyn_cast<block::RangeBoundOp>(op)) {
    values.map(bound.getResult(), records.lookup(bound.getRange())[bound.getBound()]);
    return success();
  }
  if (auto record = dyn_cast<block::MakeRecordOp>(op)) {
    records[record.getResult()] = flatten(record.getFields());
    return success();
  }
  if (auto extract = dyn_cast<block::ExtractOp>(op)) {
    auto type = cast<block::RecordType>(extract.getRecord().getType());
    unsigned offset = 0;
    for (unsigned field = 0; field < extract.getField(); ++field)
      offset += leaves(cast<TypeAttr>(type.getFieldTypes()[field]).getValue()).size();
    auto fields = records.lookup(extract.getRecord());
    unsigned count = leaves(extract.getResult().getType()).size();
    if (isa<block::RecordType>(extract.getResult().getType()))
      records[extract.getResult()] = llvm::to_vector(ArrayRef<Value>(fields).slice(offset, count));
    else values.map(extract.getResult(), fields[offset]);
    return success();
  }
  if (auto buffer = dyn_cast<block::BufferOp>(op)) {
    auto type = buffer.getResult().getType();
    if (type.getScope().getValue() == block::BufferScope::InvocationWorkspace) {
      if (buffer.getInitialValue()) return buffer.emitError("invocation workspace initialization requires explicit owner writes");
      auto dimensions = shape(type);
      if (failed(dimensions)) return failure();
      Type element = type.getElementType().isIndex() ? b.getI64Type() : type.getElementType();
      auto memoryType = MemRefType::get(*dimensions, element);
      unsigned ordinal = entry.getNumArguments();
      entry.insertArgument(ordinal, memoryType, b.getDictionaryAttr({b.getNamedAttr(nativeWorkspaceAttr, b.getUnitAttr())}), loc);
      Value memory = entry.getArgument(ordinal);
      values.map(buffer.getResult(), memory);
      extents[memory] = *dimensions;
      return success();
    }
    Value memory = allocate(type);
    if (!memory || (buffer.getInitialValue() && failed(copy(get(buffer.getInitialValue()), memory)))) return failure();
    values.map(buffer.getResult(), memory);
    return success();
  }
  if (auto overlap = dyn_cast<block::ViewOverlapOp>(op)) {
    auto span = [&](Value original) {
      Value memory = get(original);
      auto type = cast<MemRefType>(memory.getType());
      Value base = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memory);
      Value low = index(0), high = index(0);
      Value present = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
      int64_t bytes = std::max<int64_t>(1, type.getElementType().getIntOrFloatBitWidth() / 8);
      for (unsigned axis = 0; axis < type.getRank(); ++axis) {
        Value size = b.createOrFold<memref::DimOp>(loc, memory, axis);
        present = b.create<arith::AndIOp>(loc, present,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, size, index(0)));
        Value steps = b.create<arith::SubIOp>(loc, b.create<arith::MaxSIOp>(loc, size, index(1)), index(1));
        Value stride = b.create<physical::StrideOp>(loc, b.getIndexType(), memory, axis);
        Value distance = b.create<arith::MulIOp>(loc, steps, stride);
        low = b.create<arith::AddIOp>(loc, low, b.create<arith::MinSIOp>(loc, distance, index(0)));
        high = b.create<arith::AddIOp>(loc, high, b.create<arith::MaxSIOp>(loc, distance, index(0)));
      }
      low = b.create<arith::AddIOp>(loc, base, b.create<arith::MulIOp>(loc, low, index(bytes)));
      high = b.create<arith::AddIOp>(loc, base, b.create<arith::MulIOp>(loc,
          b.create<arith::AddIOp>(loc, high, index(1)), index(bytes)));
      return SmallVector<Value>{low, high, present};
    };
    auto left = span(overlap.getLhs()), right = span(overlap.getRhs());
    Value active = b.create<arith::AndIOp>(loc, left[2], right[2]);
    Value crossed = b.create<arith::AndIOp>(loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, left[0], right[1]),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, right[0], left[1]));
    values.map(overlap.getResult(), b.create<arith::AndIOp>(loc, active, crossed));
    return success();
  }
  if (auto histogram = dyn_cast<block::HistogramOp>(op)) {
    Value result = allocate(histogram.getResult().getType());
    if (!result || failed(copy(b.create<arith::ConstantIntOp>(loc, 0, 32), result))) return failure();
    Value input = get(histogram.getValues()), valid = get(histogram.getValid());
    if (failed(each(extents[input], [&](ValueRange lane) -> LogicalResult {
      auto bin = castScalarValue(b, loc, read(input, lane), b.getIndexType());
      if (failed(bin)) return failure();
      Value bounds = b.create<arith::AndIOp>(loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, *bin, index(0)),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, *bin, get(histogram.getBins())));
      Value active = b.create<arith::AndIOp>(loc, bounds, read(valid, lane));
      auto branch = b.create<scf::IfOp>(loc, active, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      Value increment = b.create<arith::AddIOp>(loc, read(result, ValueRange{*bin}), b.create<arith::ConstantIntOp>(loc, 1, 32));
      return write(increment, result, ValueRange{*bin});
    }))) return failure();
    values.map(histogram.getResult(), result);
    return success();
  }
  if (auto decode = dyn_cast<block::DelinearizeOp>(op)) {
    Value linear = get(decode.getLinear());
    for (unsigned i = decode.getExtents().size(); i-- > 0;) {
      Value size = get(decode.getExtents()[i]);
      values.map(decode.getCoordinates()[i], b.create<arith::RemSIOp>(loc, linear, size));
      linear = b.create<arith::DivSIOp>(loc, linear, size);
    }
    return success();
  }
  if (auto group = dyn_cast<block::ExecutionGroupOp>(op)) {
    auto offset = expression(group.getSegmentOffset()), length = expression(group.getSegmentLength());
    if (failed(offset) || failed(length)) return failure();
    Value linear = get(group.getLinear());
    Value begin = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, linear, *offset);
    Value end = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, linear,
        b.create<arith::AddIOp>(loc, *offset, *length));
    auto selected = b.create<scf::IfOp>(loc, b.create<arith::AndIOp>(loc, begin, end), false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(selected.thenBlock());
    linear = b.create<arith::SubIOp>(loc, linear, *offset);
    for (unsigned axis = group.getExtents().size(); axis-- > 0;) {
      Value extent = get(group.getExtents()[axis]);
      values.map(group.getCoordinates()[axis], b.create<arith::RemSIOp>(loc, linear, extent));
      linear = b.create<arith::DivSIOp>(loc, linear, extent);
    }
    return operations(group.getBody().front());
  }
  if (isa<block::AssumeInBoundsOp>(op)) return success();
  if (isa<block::AccessOpInterface>(op)) return access(op);
  if (isa<block::BinaryOp, block::UnaryOp, block::CompareOp, block::SelectOp,
          block::CastOp, block::BitcastOp>(op)) return elementwise(op);
  if (isa<block::SplatOp, block::BroadcastOp, block::ReshapeOp, block::TransposeOp,
          block::JoinOp, block::MakeRangeOp>(op)) return projection(op);
  if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) return control(op);
  if (isa<block::ReduceOp, block::ScanOp>(op)) return collective(op);
  if (auto contract = dyn_cast<block::ContractOp>(op)) return contraction(contract);
  if (auto contract = dyn_cast<block::ScaledContractOp>(op)) return scaledContraction(contract);
  return op->emitError("Block operation has no BANG C storage realization");
}
}

namespace intent::bangc {
using namespace mlir;
LogicalResult lowerBlockToBangC(ModuleOp module, physical::ConfigurationAttr configuration,
    DictionaryAttr parameters, DictionaryAttr shapes, DictionaryAttr strides) {
  auto kernel = block::getPhysicalKernel(module);
  if (failed(kernel)) return failure();
  auto interface = block::ProgramInterface::read(*kernel);
  if (failed(interface)) return failure();
  OwningOpRef<ModuleOp> target = ModuleOp::create(module.getLoc());
  (*target)->setAttrs(module->getAttrs());
  block_lowering::Lowering lowering(*kernel, *target, configuration, parameters, shapes, strides, *interface);
  if (failed(lowering.run())) return failure();
  realizeIntegerStorage(*target);
  module.getBodyRegion().takeBody(target->getBodyRegion());
  module->setAttrs((*target)->getAttrs());
  return success();
}
}
