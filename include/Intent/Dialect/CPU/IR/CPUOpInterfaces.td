#ifndef INTENT_DIALECT_CPU_IR_CPUOPINTERFACES_TD
#define INTENT_DIALECT_CPU_IR_CPUOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def IntentCPU_RegionOpInterface : OpInterface<"RegionOpInterface"> {
  let cppNamespace = "::intent::cpu";
  let description = [{
    Destination-passing region computations. Operand segments own the public
    helper schema; argument relations describe types and sliced coordinates,
    never aliasing or equivalence between an initial value and its destination.
  }];
  let methods = [
    InterfaceMethod<"Source storage", "::mlir::OperandRange", "getSources">,
    InterfaceMethod<"Summary identities", "::mlir::OperandRange", "getIdentities">,
    InterfaceMethod<"Captured values", "::mlir::OperandRange", "getCaptures">,
    InterfaceMethod<"Initial scan state", "::mlir::OperandRange", "getInitialStates",
      (ins), [{}], [{ return $_op->getOperands().slice(0, 0); }]>,
    InterfaceMethod<"Fold summary destinations", "::mlir::OperandRange", "getSummaryOutputs",
      (ins), [{}], [{ return $_op->getOperands().slice(0, 0); }]>,
    InterfaceMethod<"Emitted scan destinations", "::mlir::OperandRange", "getEmittedOutputs",
      (ins), [{}], [{ return $_op->getOperands().slice(0, 0); }]>,
    InterfaceMethod<"Final scan-state destinations", "::mlir::OperandRange", "getFinalStateOutputs",
      (ins), [{}], [{ return $_op->getOperands().slice(0, 0); }]>,
    InterfaceMethod<"Logical source member axis", "uint64_t", "getAxis">,
    InterfaceMethod<"Summary helper", "::mlir::Region &", "getSummarize">,
    InterfaceMethod<"Summary combination helper", "::mlir::Region &", "getCombine">,
    InterfaceMethod<"State application helper", "::mlir::Region *", "getApplyRegion",
      (ins), [{}], [{ return nullptr; }]>,
    InterfaceMethod<"Slice emission helper", "::mlir::Region *", "getEmitRegion",
      (ins), [{}], [{ return nullptr; }]>,
    InterfaceMethod<"Whether this operation emits ordered prefixes", "bool", "isScan",
      (ins), [{}], [{ return $_op.getApplyRegion() != nullptr; }]>,
    InterfaceMethod<"All destinations in ABI order", "::mlir::OperandRange", "getDestinations",
      (ins), [{}], [{
        if (!$_op.getSummaryOutputs().empty()) return $_op.getSummaryOutputs();
        return $_op->getOperands().take_back(
            $_op.getEmittedOutputs().size() + $_op.getFinalStateOutputs().size());
      }]>,
    InterfaceMethod<"Final summary or state destinations", "::mlir::OperandRange", "getFinalDestinations",
      (ins), [{}], [{
        return $_op.isScan() ? $_op.getFinalStateOutputs() : $_op.getSummaryOutputs();
      }]>,
    InterfaceMethod<"Initial values corresponding to final destinations", "::mlir::OperandRange", "getInitialValues",
      (ins), [{}], [{
        return $_op.isScan() ? $_op.getInitialStates() : $_op.getIdentities();
      }]>,
    InterfaceMethod<"Current helper argument groups and their type relations",
      "::mlir::FailureOr<::llvm::SmallVector<::intent::cpu::RegionArgumentSchema>>", "getRegionSchema",
      (ins "::mlir::Region &":$region), [{}], [{
        return ::intent::cpu::detail::regionArgumentSchema($_op, region);
      }]>,
    InterfaceMethod<"Arguments with one declared helper role", "::mlir::Block::BlockArgListType", "getRegionArguments",
      (ins "::mlir::Region &":$region, "::intent::cpu::RegionArgumentKind":$kind), [{}], [{
        return ::intent::cpu::detail::regionArguments($_op, region, kind);
      }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::FailureOr<::llvm::SmallVector<::mlir::Value>> bindRegionArguments(
        ::mlir::Region &region,
        ::llvm::function_ref<::mlir::ValueRange(::intent::cpu::RegionArgumentKind)> values);
  }];
}

#endif
