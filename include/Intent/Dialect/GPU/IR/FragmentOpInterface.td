#ifndef INTENT_DIALECT_GPU_IR_FRAGMENTOPINTERFACE_TD
#define INTENT_DIALECT_GPU_IR_FRAGMENTOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def IntentGPU_FragmentOpInterface : OpInterface<"FragmentOpInterface"> {
  let cppNamespace = "::intent::gpu";
  let description = [{
    Current operation-owned coordinate and extent relations between fragment
    operands and results. The relations preserve positional occurrences and
    row-major reassociation groups, but do not assert numerical equivalence.
  }];
  let methods = [
    InterfaceMethod<"Snapshot this result's relations to its operand slots.",
      "::mlir::FailureOr<::llvm::SmallVector<::intent::gpu::FragmentOperandRelation>>",
      "getFragmentOperandRelations", (ins "unsigned":$resultNumber,
        "::mlir::TypeRange":$operandTypes, "::mlir::Type":$resultType)>
  ];
}

#endif
