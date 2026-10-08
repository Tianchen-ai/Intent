#ifndef INTENT_DIALECT_INDEXEDACCESSOPINTERFACE_TD
#define INTENT_DIALECT_INDEXEDACCESSOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def Intent_IndexedAccessOpInterface : OpInterface<"IndexedAccessOpInterface"> {
  let cppNamespace = "::intent";
  let description = [{
    Logical indexed access to an external view, a local buffer or an immutable
    tensor. Index-relation positions refer only to the named indices segment.
    Queries describe operands; they do not imply memory effects, collision
    freedom, reordering permission or a physical access implementation.
  }];
  let methods = [
    InterfaceMethod<"The indexed source or destination.", "::mlir::Value",
      "getAccessSource", (ins), [{}], [{ return $_op.getSource(); }]>,
    InterfaceMethod<"SSA inputs referenced by the index relation, in local slot order.",
      "::mlir::OperandRange", "getIndexOperands", (ins), [{}], [{ return $_op.getIndices(); }]>,
    InterfaceMethod<"The logical coordinate relation.", "::intent::IndexRelationAttr",
      "getIndexRelation", (ins), [{}], [{ return $_op.getIndex(); }]>,
    InterfaceMethod<"The value supplied to an ordinary store, scatter or atomic update.",
      "::mlir::Value", "getStoredValue", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"An explicit read predicate, if present.", "::mlir::Value",
      "getAccessValidity", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"The inactive-read value paired with the predicate.", "::mlir::Value",
      "getAccessFill", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"The expected value of compare-exchange.", "::mlir::Value",
      "getCompareValue", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"The desired value of compare-exchange.", "::mlir::Value",
      "getReplacementValue", (ins), [{}], [{ return {}; }]>
  ];
  let verify = [{
    if constexpr (ConcreteOp::template hasTrait<
                      ::mlir::OpTrait::AttrSizedOperandSegments>()) {
      // ODS checks group types and optional arities, but a fixed group in an
      // explicit segment table must also contain exactly one source.
      auto operation = ::llvm::cast<ConcreteOp>($_op);
      if (operation.getODSOperands(0).size() != 1)
        return operation.emitOpError("indexed access requires one source operand");
    }
    return ::mlir::success();
  }];
}

#endif
