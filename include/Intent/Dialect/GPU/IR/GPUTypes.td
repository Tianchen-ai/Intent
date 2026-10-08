#ifndef INTENT_DIALECT_GPU_IR_GPUTYPES_TD
#define INTENT_DIALECT_GPU_IR_GPUTYPES_TD

include "Intent/Dialect/GPU/IR/GPUDialect.td"
include "Intent/Dialect/GPU/IR/GPUAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def IntentGPU_ViewType : TypeDef<IntentGPU_Dialect, "View"> {
  let mnemonic = "view";
  let summary = "One physical external-view ABI resource";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "uint32_t":$access,
    "uint64_t":$sourceId,
    "::intent::gpu::ViewLayoutAttr":$layout
  );
  let assemblyFormat =
      "`<` $elementType `,` $access `,` $sourceId `,` $layout `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    uint32_t getRank() const { return getLayout().getExtents().size(); }
  }];
}

def IntentGPU_FragmentType : TypeDef<IntentGPU_Dialect, "Fragment"> {
  let mnemonic = "fragment";
  let summary = "A parameterized value fragment owned by one physical program";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::ArrayAttr":$shape,
    "::mlir::ArrayAttr":$axisMaps,
    "uint32_t":$validity,
    "uint64_t":$owner
  );
  let assemblyFormat =
      "`<` $elementType `,` $shape `,` $axisMaps `,` $validity `,` $owner `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_RangeType : TypeDef<IntentGPU_Dialect, "Range"> {
  let mnemonic = "range";
  let summary = "A runtime physical traversal range with logical provenance";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$sourceAxis,
    "int64_t":$dimensionId,
    "bool":$derived
  );
  let assemblyFormat =
      "`<` $sourceId `,` $sourceAxis `,` $dimensionId `,` $derived `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_BufferType : TypeDef<IntentGPU_Dialect, "Buffer"> {
  let mnemonic = "buffer";
  let summary = "A physical mutable resource with explicit scope and ownership";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::ArrayAttr":$shape,
    "::intent::gpu::BufferScopeAttr":$scope,
    "uint64_t":$instance,
    "uint64_t":$owner,
    "::intent::gpu::BufferInitializationAttr":$initialization,
    "uint32_t":$visibility
  );
  let assemblyFormat =
      "`<` $elementType `,` $shape `,` $scope `,` $instance `,` $owner `,` $initialization `,` $visibility `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isInvocationWorkspace() const {
      return getScope().getValue() == BufferScope::InvocationWorkspace;
    }
  }];
}

def IntentGPU_RecordType : TypeDef<IntentGPU_Dialect, "Record"> {
  let mnemonic = "record";
  let summary = "A physical product value with stable named components";
  let parameters = (ins
    "::mlir::ArrayAttr":$fieldNames,
    "::mlir::ArrayAttr":$fieldTypes,
    "uint64_t":$owner
  );
  let assemblyFormat = "`<` $fieldNames `,` $fieldTypes `,` $owner `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_ViewValue : DialectType<IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::ViewType>($_self)">,
    "physical external view", "::intent::gpu::ViewType">;
def IntentGPU_FragmentValue : DialectType<IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::FragmentType>($_self)">,
    "physical fragment", "::intent::gpu::FragmentType">;
def IntentGPU_RangeValue : DialectType<IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::RangeType>($_self)">,
    "physical range", "::intent::gpu::RangeType">;
def IntentGPU_BufferValue : DialectType<IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::BufferType>($_self)">,
    "physical buffer", "::intent::gpu::BufferType">;
def IntentGPU_RecordValue : DialectType<IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::RecordType>($_self)">,
    "physical record", "::intent::gpu::RecordType">;

def IntentGPU_ScalarValue : AnyTypeOf<[AnyInteger, AnyFloat, Index]>;
def IntentGPU_DataValue : AnyTypeOf<[
  IntentGPU_ScalarValue, IntentGPU_FragmentValue, IntentGPU_RecordValue
]>;
def IntentGPU_ResourceValue : AnyTypeOf<[IntentGPU_ViewValue, IntentGPU_BufferValue]>;
def IntentGPU_Value : AnyTypeOf<[
  IntentGPU_DataValue, IntentGPU_ResourceValue, IntentGPU_RangeValue
]>;

#endif
