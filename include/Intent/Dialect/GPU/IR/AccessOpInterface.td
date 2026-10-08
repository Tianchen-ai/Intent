#ifndef INTENT_DIALECT_GPU_IR_ACCESSOPINTERFACE_TD
#define INTENT_DIALECT_GPU_IR_ACCESSOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def IntentGPU_AccessOpInterface : OpInterface<"AccessOpInterface"> {
  let cppNamespace = "::intent::gpu";
  let description = [{
    Current physical access coordinates and value schema. The accessed operand
    is a memory resource for memory operations and an immutable SSA fragment for
    Gather. MemoryEffectsOpInterface, not this interface, defines memory effects.
    Atomic ordering, sharing and collision semantics remain on the concrete op.
  }];
  let methods = [
    InterfaceMethod<"The existing access semantics.", "::intent::gpu::AccessKind", "getAccessKind">,
    InterfaceMethod<"The actual resource/source operand, retaining its operand identity.",
      "::mlir::OpOperand &", "getAccessResourceOperand">,
    InterfaceMethod<"Current coordinate operands.", "::mlir::OperandRange", "getAccessCoordinates",
      (ins), [{ return $_op.getCoordinates(); }]>,
    InterfaceMethod<"Coordinate operands for in-place schema updates.",
      "::mlir::MutableOperandRange", "getAccessCoordinatesMutable",
      (ins), [{ return $_op.getCoordinatesMutable(); }]>,
    InterfaceMethod<"Resource axes indexed by the coordinate operands.",
      "::llvm::ArrayRef<int64_t>", "getAccessSourceAxes",
      (ins), [{ return $_op.getSourceAxes(); }]>,
    InterfaceMethod<"Replace the explicit coordinate-to-resource axis mapping.",
      "void", "setAccessSourceAxes", (ins "::llvm::ArrayRef<int64_t>":$axes),
      [{ $_op.setSourceAxes(axes); }]>,
    InterfaceMethod<"Payload operands. Compare-exchange order is expected, desired.",
      "::mlir::MutableOperandRange", "getAccessPayloadsMutable">,
    InterfaceMethod<"Type whose lanes define the access predicate; CAS uses expected, not its record result.",
      "::mlir::Type", "getAccessValueType">,
    InterfaceMethod<"Current active predicate.", "::mlir::Value", "getAccessValidity",
      (ins), [{ return $_op.getValid(); }]>,
    InterfaceMethod<"The active predicate operand group.", "::mlir::MutableOperandRange", "getAccessValidityMutable",
      (ins), [{ return $_op.getValidMutable(); }]>,
    InterfaceMethod<"Explicit inactive result of a plain read, when present.",
      "::mlir::Value", "getAccessFill", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"The fill group exists only for load and gather.",
      "std::optional<::mlir::MutableOperandRange>", "getAccessFillMutable",
      (ins), [{}], [{ return std::nullopt; }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::Value getAccessResource() { return getAccessResourceOperand().get(); }
    ::mlir::OperandRange getAccessPayloads() { return getAccessPayloadsMutable(); }
    ::mlir::Value getAccessResult() {
      return getOperation()->getNumResults() ? getOperation()->getResult(0) : ::mlir::Value();
    }
    bool isMemoryAccess() { return getAccessKind() != ::intent::gpu::AccessKind::Gather; }
    bool writesMemory();
    // Keep resource, source axes, operation identity, attributes, result types and
    // ordering unchanged. Coordinates must still match the explicit source axes.
    // All operand ranges are copied before mutation, so current op ranges may be
    // passed directly. No implicit projection, fill, or type repair is performed.
    ::mlir::LogicalResult updateAccessOperands(::mlir::ValueRange coordinates,
        ::mlir::ValueRange payloads, ::mlir::Value valid, ::mlir::Value fill);
  }];
  let verify = [{
    auto operation = ::llvm::cast<ConcreteOp>($_op);
    if (::mlir::failed(operation.verifyAccessOperandGroups()))
      return ::mlir::failure();
    return ::intent::gpu::verifyAccessSchema(
        ::llvm::cast<::intent::gpu::AccessOpInterface>($_op));
  }];
}

#endif
