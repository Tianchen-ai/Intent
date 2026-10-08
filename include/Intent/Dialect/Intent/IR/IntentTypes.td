#ifndef INTENT_DIALECT_INTENT_IR_INTENTTYPES_TD
#define INTENT_DIALECT_INTENT_IR_INTENTTYPES_TD

include "Intent/Dialect/Intent/IR/IntentDialect.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def Intent_LogicalIndexType : TypeDef<Intent_Dialect, "LogicalIndex"> {
  let mnemonic = "logical_index";
  let summary = "A coordinate tied to one logical domain source and axis";
  let parameters = (ins "uint64_t":$sourceId, "uint32_t":$axis);
  let assemblyFormat = "`<` $sourceId `,` $axis `>`";
  let genVerifyDecl = 1;
}

def Intent_DomainType : TypeDef<Intent_Dialect, "Domain"> {
  let mnemonic = "domain";
  let summary = "A logical iteration domain with stable provenance";
  let parameters = (ins
    "uint32_t":$flavor,
    "uint32_t":$rank,
    "uint64_t":$originId
  );
  let assemblyFormat = "`<` $flavor `,` $rank `,` $originId `>`";
  let genVerifyDecl = 1;
}

def Intent_RegionType : TypeDef<Intent_Dialect, "Region"> {
  let mnemonic = "region";
  let summary = "A source-derived logical subregion";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$rank,
    "uint64_t":$originId
  );
  let assemblyFormat = "`<` $sourceId `,` $rank `,` $originId `>`";
  let genVerifyDecl = 1;
}

def Intent_BufferType : TypeDef<Intent_Dialect, "Buffer"> {
  let mnemonic = "buffer";
  let summary = "A kernel-local logical mutable buffer";
  let parameters = (ins "::mlir::Type":$tensor, "uint64_t":$originId);
  let assemblyFormat = "`<` $tensor `,` $originId `>`";
  let genVerifyDecl = 1;
}

def Intent_TupleType : TypeDef<Intent_Dialect, "Tuple"> {
  let mnemonic = "tuple";
  let summary = "An immutable ordered structural product";
  let parameters = (ins "::mlir::ArrayAttr":$componentTypes);
  let assemblyFormat = "`<` $componentTypes `>`";
  let genVerifyDecl = 1;
}

def Intent_RecordType : TypeDef<Intent_Dialect, "Record"> {
  let mnemonic = "record";
  let summary = "An immutable ordered named-field product";
  let parameters = (ins
    "::mlir::ArrayAttr":$fieldNames,
    "::mlir::ArrayAttr":$fieldTypes
  );
  let assemblyFormat = "`<` $fieldNames `,` $fieldTypes `>`";
  let genVerifyDecl = 1;
}

def Intent_ConstexprType : TypeDef<Intent_Dialect, "Constexpr"> {
  let mnemonic = "constexpr";
  let summary = "A compile-time specialization value";
  let parameters = (ins "::mlir::Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
  let genVerifyDecl = 1;
}

def Intent_EnumType : TypeDef<Intent_Dialect, "Enum"> {
  let mnemonic = "enum";
  let summary = "A constexpr-only closed named enumeration";
  let parameters = (ins
    "::mlir::StringAttr":$name,
    "::mlir::ArrayAttr":$memberNames,
    "::mlir::DenseI64ArrayAttr":$memberValues
  );
  let assemblyFormat = "`<` $name `,` $memberNames `,` $memberValues `>`";
  let genVerifyDecl = 1;
}

def Intent_ViewType : TypeDef<Intent_Dialect, "View"> {
  let mnemonic = "view";
  let summary = "An external strided logical view and its access direction";
  let parameters = (ins "::mlir::Type":$tensor, "uint32_t":$access,
                        "::intent::ViewConstraintsAttr":$constraints);
  let assemblyFormat = "`<` $tensor `,` $access `,` $constraints `>`";
  let genVerifyDecl = 1;
}

def Intent_LogicalIndexValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::LogicalIndexType>($_self)">,
    "logical index",
    "::intent::LogicalIndexType">;
def Intent_DomainValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::DomainType>($_self)">,
    "logical domain",
    "::intent::DomainType">;
def Intent_RegionValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::RegionType>($_self)">,
    "logical subregion",
    "::intent::RegionType">;
def Intent_BufferValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::BufferType>($_self)">,
    "logical buffer",
    "::intent::BufferType">;
def Intent_RecordValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::RecordType>($_self)">,
    "typed record",
    "::intent::RecordType">;
def Intent_TupleValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::TupleType>($_self)">,
    "typed tuple",
    "::intent::TupleType">;
def Intent_ConstexprValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::ConstexprType>($_self)">,
    "constexpr value",
    "::intent::ConstexprType">;
def Intent_EnumValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::EnumType>($_self)">,
    "constexpr enum",
    "::intent::EnumType">;
def Intent_ViewValue : DialectType<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::ViewType>($_self)">,
    "external view",
    "::intent::ViewType">;

def Intent_ScalarValue : AnyTypeOf<[AnyInteger, AnyFloat, Index]>;
def Intent_DataValue : AnyTypeOf<[
  Intent_ScalarValue,
  AnyRankedTensor,
  Intent_TupleValue,
  Intent_RecordValue,
  Intent_LogicalIndexValue
]>;
def Intent_IterationValue : AnyTypeOf<[Intent_DomainValue, Intent_RegionValue]>;
def Intent_AddressableValue : AnyTypeOf<[Intent_ViewValue, Intent_BufferValue]>;
def Intent_Value : AnyTypeOf<[
  Intent_DataValue,
  Intent_LogicalIndexValue,
  Intent_IterationValue,
  Intent_BufferValue,
  Intent_ConstexprValue,
  Intent_EnumValue,
  Intent_ViewValue
]>;

#endif
