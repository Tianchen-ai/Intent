#include "Intent/Serialization/NativeSource.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent {

NativeSourceEmitter::NativeSourceEmitter(llvm::raw_ostream &output,
                                         NativeSourceSyntax syntax)
    : SourceEmitter(output, syntax == NativeSourceSyntax::Mojo ? 4 : 2),
      syntax(syntax) {}

LogicalResult verifyNativeMemoryType(Operation *owner, MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!type || mlir::failed(type.getStridesAndOffset(strides, offset)))
    return owner->emitOpError("native memory requires a ranked strided descriptor");
  return success();
}

LogicalResult verifyNativeAllocation(Operation *operation, bool requireStaticShape) {
  auto type = dyn_cast<MemRefType>(operation->getResult(0).getType());
  if (!type || !type.getLayout().isIdentity() ||
      (requireStaticShape && !type.hasStaticShape()))
    return operation->emitOpError("native allocation requires an identity layout")
           << (requireStaticShape ? " and a static shape" : "");
  return success();
}

LogicalResult NativeSourceEmitter::bindEntryMemory(Value value,
                                                   llvm::StringRef base) {
  auto type = cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)) || offset != 0)
    return emitError(value.getLoc(), "native entry requires a strided view with zero descriptor offset");
  MemoryDescriptor descriptor{base.str(), "0", {}, {}};
  for (int64_t size : type.getShape())
    descriptor.sizes.push_back(ShapedType::isDynamic(size) ? "" : std::to_string(size));
  for (int64_t stride : strides)
    descriptor.strides.push_back(ShapedType::isDynamic(stride) ? "" : std::to_string(stride));
  memories[value] = std::move(descriptor);
  reserveName(base);
  return success();
}

MemoryDescriptor NativeSourceEmitter::allocationDescriptor(
    MemRefType type, llvm::StringRef base, ValueRange dynamicSizes) {
  MemoryDescriptor descriptor{base.str(), "0", {}, {}};
  unsigned dynamic = 0;
  for (int64_t size : type.getShape())
    descriptor.sizes.push_back(ShapedType::isDynamic(size)
        ? valueString(dynamicSizes[dynamic++]) : std::to_string(size));
  descriptor.strides.resize(type.getRank());
  std::string stride = "1";
  for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
    descriptor.strides[axis] = stride;
    stride = "(" + stride + ") * (" + descriptor.sizes[axis] + ")";
  }
  return descriptor;
}

std::string NativeSourceEmitter::foldIndex(OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value)) return valueString(ssa);
  return std::to_string(cast<IntegerAttr>(cast<Attribute>(value)).getInt());
}

std::string NativeSourceEmitter::memoryOffset(Value memory, ValueRange indices) {
  const auto &descriptor = memories.at(memory);
  std::string offset = descriptor.offset;
  if (indices.empty()) return offset;
  for (auto [index, stride] : llvm::zip_equal(indices, descriptor.strides))
    offset += " + (" + valueString(index) + ") * (" + stride + ")";
  return offset;
}

std::string NativeSourceEmitter::memoryPointer(Value memory, ValueRange indices) {
  const auto &descriptor = memories.at(memory);
  // No coordinates means the view origin, including its descriptor offset.
  auto offset = indices.empty() ? descriptor.offset : memoryOffset(memory, indices);
  return offset == "0" ? descriptor.base : offsetPointer(descriptor.base, offset);
}

SmallVector<std::string> NativeSourceEmitter::components(Value value) {
  if (isa<MemRefType>(value.getType())) return memories.at(value).components();
  return {valueString(value)};
}

SmallVector<Type> NativeSourceEmitter::componentTypes(Type type) {
  if (auto memory = dyn_cast<MemRefType>(type)) {
    SmallVector<Type> result{memory};
    result.append(1 + 2 * memory.getRank(), IndexType::get(type.getContext()));
    return result;
  }
  return {type};
}

void NativeSourceEmitter::declareComponent(llvm::StringRef name, Type type,
                                          std::optional<llvm::StringRef> initial) {
  auto declaration = syntax == NativeSourceSyntax::Mojo
      ? "var " + name.str() + ": " + nativeType(type)
      : nativeType(type) + " " + name.str();
  if (initial) declaration += " = " + initial->str();
  line(declaration + (syntax == NativeSourceSyntax::C ? ";" : ""));
}

void NativeSourceEmitter::assignComponent(llvm::StringRef name,
                                         llvm::StringRef expression) {
  line(name + " = " + expression + (syntax == NativeSourceSyntax::C ? ";" : ""));
}

void NativeSourceEmitter::bindExpression(Value value, llvm::StringRef expression) {
  auto name = newName();
  declareComponent(name, value.getType(), expression);
  bind(value, name);
}

void NativeSourceEmitter::declare(Value value, Value initial) {
  SmallVector<std::string> fields;
  auto sources = initial ? components(initial) : SmallVector<std::string>{};
  for (auto [index, type] : llvm::enumerate(componentTypes(value.getType()))) {
    fields.push_back(newName());
    declareComponent(fields.back(), type, initial
        ? std::optional<llvm::StringRef>(sources[index]) : std::nullopt);
  }
  if (auto memory = dyn_cast<MemRefType>(value.getType()))
    memories[value] = MemoryDescriptor::fromComponents(memory, fields);
  else bind(value, fields.front());
}

void NativeSourceEmitter::alias(Value result, Value source) {
  if (isa<MemRefType>(result.getType())) memories[result] = memories.at(source);
  else bind(result, valueString(source));
}

void NativeSourceEmitter::transfer(ValueRange from, ValueRange to) {
  SmallVector<std::pair<std::string, std::string>> assignments;
  for (auto [source, target] : llvm::zip_equal(from, to)) {
    auto sources = components(source), targets = components(target);
    for (auto [index, type] : llvm::enumerate(componentTypes(source.getType()))) {
      auto temporary = newName("next_");
      declareComponent(temporary, type, sources[index]);
      assignments.emplace_back(targets[index], temporary);
    }
  }
  for (const auto &[destination, temporary] : assignments)
    assignComponent(destination, temporary);
}

LogicalResult NativeSourceEmitter::emitNativeBlock(Block &block) {
  auto before = emittedLines;
  for (Operation &operation : block.without_terminator())
    if (mlir::failed(emitNativeOperation(&operation))) return failure();
  if (syntax == NativeSourceSyntax::Mojo && before == emittedLines) line("pass");
  return success();
}

const OperationEmitters<NativeSourceEmitter> &NativeSourceEmitter::metadataEmitters() {
  static const auto handlers = [] {
    OperationEmitters<NativeSourceEmitter> table;
    auto memoryTypes = [](Operation *operation) -> LogicalResult {
      for (Type type : llvm::concat<Type>(operation->getOperandTypes(), operation->getResultTypes())) {
        if (isa<UnrankedMemRefType>(type))
          return operation->emitOpError("native memory metadata requires ranked descriptors");
        if (auto memory = dyn_cast<MemRefType>(type))
          if (mlir::failed(verifyNativeMemoryType(operation, memory))) return failure();
      }
      return success();
    };
    table.add<memref::DimOp>([=](memref::DimOp operation) -> LogicalResult {
      if (mlir::failed(memoryTypes(operation))) return failure();
      if (!operation.getConstantIndex())
        return operation.emitOpError("native memory dimension requires a constant axis");
      return success();
    }, [](memref::DimOp operation, NativeSourceEmitter &out) {
      out.bind(operation.getResult(), out.memories.at(operation.getSource()).sizes[*operation.getConstantIndex()]);
      return success();
    });
    table.add<memref::CastOp>(memoryTypes, [](memref::CastOp operation, NativeSourceEmitter &out) {
      out.alias(operation.getResult(), operation.getSource());
      return success();
    });
    table.add<memref::TransposeOp>(memoryTypes,
        [](memref::TransposeOp operation, NativeSourceEmitter &out) {
      const auto source = out.memories.at(operation.getIn());
      MemoryDescriptor target{source.base, source.offset, {}, {}};
      for (AffineExpr expression : operation.getPermutation().getResults()) {
        unsigned axis = cast<AffineDimExpr>(expression).getPosition();
        target.sizes.push_back(source.sizes[axis]);
        target.strides.push_back(source.strides[axis]);
      }
      out.memories[operation.getResult()] = std::move(target);
      return success();
    });
    table.add<memref::ExtractStridedMetadataOp>(memoryTypes,
        [](memref::ExtractStridedMetadataOp operation, NativeSourceEmitter &out) {
      const auto source = out.memories.at(operation.getSource());
      out.memories[operation.getBaseBuffer()] = {source.base, "0", {}, {}};
      out.bind(operation.getOffset(), source.offset);
      for (auto [result, size] : llvm::zip_equal(operation.getSizes(), source.sizes)) out.bind(result, size);
      for (auto [result, stride] : llvm::zip_equal(operation.getStrides(), source.strides)) out.bind(result, stride);
      return success();
    });
    table.add<memref::ReinterpretCastOp>(memoryTypes,
        [](memref::ReinterpretCastOp operation, NativeSourceEmitter &out) {
      // Reinterpret offsets are relative to the storage base, not the view origin.
      MemoryDescriptor target{out.memories.at(operation.getSource()).base,
                              out.foldIndex(operation.getMixedOffsets().front()), {}, {}};
      for (auto size : operation.getMixedSizes()) target.sizes.push_back(out.foldIndex(size));
      for (auto stride : operation.getMixedStrides()) target.strides.push_back(out.foldIndex(stride));
      out.memories[operation.getResult()] = std::move(target);
      return success();
    });
    table.add<memref::SubViewOp>(memoryTypes,
        [](memref::SubViewOp operation, NativeSourceEmitter &out) {
      const auto source = out.memories.at(operation.getSource());
      MemoryDescriptor target{source.base, source.offset, {}, {}};
      auto offsets = operation.getMixedOffsets(), sizes = operation.getMixedSizes(),
           steps = operation.getMixedStrides();
      auto dropped = operation.getDroppedDims();
      for (unsigned axis = 0; axis < offsets.size(); ++axis) {
        target.offset += " + (" + out.foldIndex(offsets[axis]) + ") * (" + source.strides[axis] + ")";
        if (!dropped.test(axis)) {
          target.sizes.push_back(out.foldIndex(sizes[axis]));
          target.strides.push_back("(" + source.strides[axis] + ") * (" + out.foldIndex(steps[axis]) + ")");
        }
      }
      target.offset = "(" + target.offset + ")";
      out.memories[operation.getResult()] = std::move(target);
      return success();
    });
    table.add<memref::ExtractAlignedPointerAsIndexOp>(memoryTypes,
        [](memref::ExtractAlignedPointerAsIndexOp operation, NativeSourceEmitter &out) {
      out.bindExpression(operation.getResult(), out.pointerAsIndex(out.memories.at(operation.getSource()).base));
      return success();
    });
    return table;
  }();
  return handlers;
}

const OperationEmitters<NativeSourceEmitter> &NativeSourceEmitter::controlEmitters() {
  static const auto handlers = [] {
    OperationEmitters<NativeSourceEmitter> table;
    auto check = [](Operation *operation) -> LogicalResult {
      for (Type type : llvm::concat<Type>(operation->getOperandTypes(), operation->getResultTypes()))
        if (auto memory = dyn_cast<MemRefType>(type))
          if (mlir::failed(verifyNativeMemoryType(operation, memory))) return failure();
      return success();
    };
    table.add<scf::ForOp>(check, [](scf::ForOp loop, NativeSourceEmitter &out) {
      for (auto [argument, initial] : llvm::zip_equal(loop.getRegionIterArgs(), loop.getInitArgs()))
        out.declare(argument, initial);
      auto iv = out.newName();
      out.bind(loop.getInductionVar(), iv);
      auto lower = out.valueString(loop.getLowerBound()), upper = out.valueString(loop.getUpperBound()),
           step = out.valueString(loop.getStep());
      out.line(out.syntax == NativeSourceSyntax::Mojo
          ? "for " + iv + " in range(" + lower + ", " + upper + ", " + step + "):"
          : "for (" + out.nativeType(loop.getInductionVar().getType()) + " " + iv + " = " + lower +
            "; " + iv + " < " + upper + "; " + iv + " += " + step + ") {");
      ++out.indent;
      if (mlir::failed(out.emitNativeBlock(*loop.getBody()))) return failure();
      out.transfer(loop.getBody()->getTerminator()->getOperands(), loop.getRegionIterArgs());
      --out.indent;
      if (out.syntax == NativeSourceSyntax::C) out.line("}");
      for (auto [result, argument] : llvm::zip_equal(loop.getResults(), loop.getRegionIterArgs())) out.alias(result, argument);
      return success();
    });
    table.add<scf::IfOp>(check, [](scf::IfOp condition, NativeSourceEmitter &out) {
      for (Value result : condition.getResults()) out.declare(result);
      auto branch = [&](Block &block) {
        ++out.indent;
        if (mlir::failed(out.emitNativeBlock(block))) return failure();
        out.transfer(block.getTerminator()->getOperands(), condition.getResults());
        --out.indent;
        return success();
      };
      auto predicate = out.valueString(condition.getCondition());
      out.line(out.syntax == NativeSourceSyntax::Mojo ? "if " + predicate + ":" : "if (" + predicate + ") {");
      if (mlir::failed(branch(*condition.thenBlock()))) return failure();
      if (!condition.getElseRegion().empty()) {
        out.line(out.syntax == NativeSourceSyntax::Mojo ? "else:" : "} else {");
        if (mlir::failed(branch(*condition.elseBlock()))) return failure();
      }
      if (out.syntax == NativeSourceSyntax::C) out.line("}");
      return success();
    });
    table.add<scf::WhileOp>(check, [](scf::WhileOp loop, NativeSourceEmitter &out) {
      Block &before = loop.getBefore().front(), &after = loop.getAfter().front();
      for (auto [argument, initial] : llvm::zip_equal(before.getArguments(), loop.getInits())) out.declare(argument, initial);
      for (Value argument : after.getArguments()) out.declare(argument);
      out.line(out.syntax == NativeSourceSyntax::Mojo ? "while True:" : "while (1) {");
      ++out.indent;
      if (mlir::failed(out.emitNativeBlock(before))) return failure();
      auto condition = cast<scf::ConditionOp>(before.getTerminator());
      out.transfer(condition.getArgs(), after.getArguments());
      auto predicate = out.valueString(condition.getCondition());
      if (out.syntax == NativeSourceSyntax::Mojo) {
        out.line("if not " + predicate + ":");
        ++out.indent; out.line("break"); --out.indent;
      } else out.line("if (!(" + predicate + ")) break;");
      if (mlir::failed(out.emitNativeBlock(after))) return failure();
      out.transfer(after.getTerminator()->getOperands(), before.getArguments());
      --out.indent;
      if (out.syntax == NativeSourceSyntax::C) out.line("}");
      for (auto [result, argument] : llvm::zip_equal(loop.getResults(), after.getArguments())) out.alias(result, argument);
      return success();
    });
    return table;
  }();
  return handlers;
}

} // namespace intent
