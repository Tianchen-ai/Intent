#ifndef INTENT_DIALECT_INTENT_IR_INTENTATTRS_TD
#define INTENT_DIALECT_INTENT_IR_INTENTATTRS_TD

include "Intent/Dialect/Intent/IR/IntentDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class Intent_I32Enum<string name, string description,
                     list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::intent";
}

class Intent_I32EnumAttr<string mnemonic, Intent_I32Enum enumInfo>
    : EnumAttr<Intent_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def Intent_NumericsMode : Intent_I32Enum<"NumericsMode", "Compilation numerical permissions", [
  I32EnumAttrCase<"Source", 0, "source">,
  I32EnumAttrCase<"RelaxedNormalization", 1, "relaxed_normalization">
]>;

def Intent_CompileOptionsAttr : AttrDef<Intent_Dialect, "CompileOptions"> {
  let mnemonic = "compile_options";
  let summary = "Bound numerical permissions and optional optimization controls";
  let parameters = (ins EnumParameter<Intent_NumericsMode>:$numerics,
      "bool":$onlineReduction, "bool":$optimizationRemarks);
  let assemblyFormat = "`<` $numerics `,` $onlineReduction `,` $optimizationRemarks `>`";
}

def Intent_UnaryOperator : Intent_I32Enum<"UnaryOperator",
    "Canonical unary arithmetic semantics", [
      I32EnumAttrCase<"Negate", 0, "negate">,
      I32EnumAttrCase<"Not", 1, "not">,
      I32EnumAttrCase<"Exp", 2, "exp">,
      I32EnumAttrCase<"Exp2", 3, "exp2">,
      I32EnumAttrCase<"Log", 4, "log">,
      I32EnumAttrCase<"Sin", 5, "sin">,
      I32EnumAttrCase<"Cos", 6, "cos">,
      I32EnumAttrCase<"Floor", 7, "floor">,
      I32EnumAttrCase<"Erf", 8, "erf">,
      I32EnumAttrCase<"Rsqrt", 9, "rsqrt">,
      I32EnumAttrCase<"Sigmoid", 10, "sigmoid">,
      I32EnumAttrCase<"Tanh", 11, "tanh">,
      I32EnumAttrCase<"Abs", 12, "abs">,
      I32EnumAttrCase<"Sqrt", 13, "sqrt">,
      I32EnumAttrCase<"Lgamma", 14, "lgamma">,
      I32EnumAttrCase<"Log1p", 15, "log1p">,
      I32EnumAttrCase<"Erfc", 16, "erfc">,
      I32EnumAttrCase<"I0", 17, "i0">,
      I32EnumAttrCase<"Asin", 18, "asin">
    ]>;
def Intent_UnaryOperatorAttr :
    Intent_I32EnumAttr<"unary_operator", Intent_UnaryOperator>;

def Intent_BinaryOperator : Intent_I32Enum<"BinaryOperator",
    "Canonical binary arithmetic semantics", [
      I32EnumAttrCase<"Add", 0, "add">,
      I32EnumAttrCase<"Subtract", 1, "subtract">,
      I32EnumAttrCase<"Multiply", 2, "multiply">,
      I32EnumAttrCase<"TrueDivide", 3, "true_divide">,
      I32EnumAttrCase<"FloorDivide", 4, "floor_divide">,
      I32EnumAttrCase<"Remainder", 5, "remainder">,
      I32EnumAttrCase<"Power", 6, "power">,
      I32EnumAttrCase<"Maximum", 7, "maximum">,
      I32EnumAttrCase<"Minimum", 8, "minimum">,
      I32EnumAttrCase<"MaximumNum", 9, "maximum_num">,
      I32EnumAttrCase<"MinimumNum", 10, "minimum_num">,
      I32EnumAttrCase<"LogicalAnd", 11, "logical_and">,
      I32EnumAttrCase<"LogicalOr", 12, "logical_or">,
      I32EnumAttrCase<"BitwiseAnd", 13, "bitwise_and">,
      I32EnumAttrCase<"BitwiseOr", 14, "bitwise_or">,
      I32EnumAttrCase<"BitwiseXor", 15, "bitwise_xor">,
      I32EnumAttrCase<"LeftShift", 16, "left_shift">,
      I32EnumAttrCase<"RightShift", 17, "right_shift">
    ]>;
def Intent_BinaryOperatorAttr :
    Intent_I32EnumAttr<"binary_operator", Intent_BinaryOperator>;

def Intent_ComparePredicate : Intent_I32Enum<"ComparePredicate",
    "Canonical comparison predicate", [
      I32EnumAttrCase<"Eq", 0, "eq">,
      I32EnumAttrCase<"Ne", 1, "ne">,
      I32EnumAttrCase<"Lt", 2, "lt">,
      I32EnumAttrCase<"Le", 3, "le">,
      I32EnumAttrCase<"Gt", 4, "gt">,
      I32EnumAttrCase<"Ge", 5, "ge">
    ]>;
def Intent_ComparePredicateAttr :
    Intent_I32EnumAttr<"compare_predicate", Intent_ComparePredicate>;

def Intent_AtomicOrdering : Intent_I32Enum<"AtomicOrdering",
    "Canonical atomic memory ordering", [
      I32EnumAttrCase<"Relaxed", 0, "relaxed">,
      I32EnumAttrCase<"Acquire", 1, "acquire">,
      I32EnumAttrCase<"Release", 2, "release">,
      I32EnumAttrCase<"AcquireRelease", 3, "acq_rel">
    ]>;
def Intent_AtomicOrderingAttr :
    Intent_I32EnumAttr<"atomic_ordering", Intent_AtomicOrdering>;

def Intent_AtomicRMWKind : Intent_I32Enum<"AtomicRMWKind",
    "Canonical atomic read-modify-write operation", [
      I32EnumAttrCase<"Exchange", 0, "exchange">,
      I32EnumAttrCase<"Add", 1, "add">,
      I32EnumAttrCase<"Maximum", 2, "maximum">,
      I32EnumAttrCase<"Minimum", 3, "minimum">,
      I32EnumAttrCase<"BitwiseAnd", 4, "bitwise_and">,
      I32EnumAttrCase<"BitwiseOr", 5, "bitwise_or">,
      I32EnumAttrCase<"BitwiseXor", 6, "bitwise_xor">
    ]>;
def Intent_AtomicRMWKindAttr :
    Intent_I32EnumAttr<"atomic_rmw_kind", Intent_AtomicRMWKind>;

def Intent_ScaledFormat : Intent_I32Enum<"ScaledFormat",
    "Canonical block-scaled operand format", [
      I32EnumAttrCase<"E2M1", 0, "e2m1">,
      I32EnumAttrCase<"E4M3", 1, "e4m3">,
      I32EnumAttrCase<"E8M0", 2, "e8m0">
    ]>;
def Intent_ScaledFormatAttr :
    Intent_I32EnumAttr<"scaled_format", Intent_ScaledFormat>;

def Intent_QuantFormat : Intent_I32Enum<"QuantFormat",
    "Canonical quantized record format", [
      I32EnumAttrCase<"Q4K", 0, "q4_k">,
      I32EnumAttrCase<"Q8K", 1, "q8_k">
    ]>;
def Intent_QuantFormatAttr :
    Intent_I32EnumAttr<"quant_format", Intent_QuantFormat>;

def Intent_TensorShapeAttr : AttrDef<Intent_Dialect, "TensorShape"> {
  let mnemonic = "tensor_shape";
  let summary = "Stable identities for logical tensor dimensions";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$dimensions);
  let assemblyFormat = "`<` $dimensions `>`";
  let genVerifyDecl = 1;
}

def Intent_ShapeExprAttr : AttrDef<Intent_Dialect, "ShapeExpr"> {
  let mnemonic = "shape_expr";
  let summary = "One typed logical shape expression";
  let parameters = (ins
    "uint32_t":$kind,
    "int64_t":$dimension,
    "int64_t":$payload
  );
  let assemblyFormat = "`<` $kind `,` $dimension `,` $payload `>`";
  let genVerifyDecl = 1;
}

def Intent_ShapeRelationAttr : AttrDef<Intent_Dialect, "ShapeRelation"> {
  let mnemonic = "shape_relation";
  let summary = "Typed relation from SSA extents to a logical tensor shape";
  let parameters = (ins "::mlir::ArrayAttr":$axes);
  let assemblyFormat = "`<` $axes `>`";
  let genVerifyDecl = 1;
}

def Intent_IndexTermAttr : AttrDef<Intent_Dialect, "IndexTerm"> {
  let mnemonic = "index_term";
  let summary = "One typed coordinate expression in an index relation";
  let parameters = (ins
    "uint32_t":$kind,
    "::mlir::DenseI64ArrayAttr":$operandPositions,
    "::mlir::DenseI64ArrayAttr":$staticValues
  );
  let assemblyFormat =
      "`<` $kind `,` $operandPositions `,` $staticValues `>`";
  let genVerifyDecl = 1;
}

def Intent_IndexRelationAttr : AttrDef<Intent_Dialect, "IndexRelation"> {
  let mnemonic = "index_relation";
  let summary = "Typed source-to-result logical coordinate relation";
  let parameters = (ins
    "uint32_t":$sourceRank,
    "uint32_t":$resultRank,
    "::mlir::DenseI64ArrayAttr":$resultDimensions,
    "::mlir::ArrayAttr":$terms
  );
  let assemblyFormat =
      "`<` $sourceRank `,` $resultRank `,` $resultDimensions `,` $terms `>`";
  let genVerifyDecl = 1;
}

def Intent_ViewConstraintsAttr : AttrDef<Intent_Dialect, "ViewConstraints"> {
  let mnemonic = "view_constraints";
  let summary = "Typed logical stride and alias constraints for an external view";
  let parameters = (ins
    "bool":$hasStrides,
    "::mlir::ArrayAttr":$strides,
    "::mlir::StringAttr":$alias,
    "bool":$noalias
  );
  let assemblyFormat =
      "`<` $hasStrides `,` $strides `,` $alias `,` $noalias `>`";
  let genVerifyDecl = 1;
}

def Intent_ParameterAttr : AttrDef<Intent_Dialect, "Parameter"> {
  let mnemonic = "parameter";
  let summary = "Canonical function argument name and source origin";
  let parameters = (ins "::mlir::StringAttr":$name, "int64_t":$originId);
  let assemblyFormat = "`<` $name `,` $originId `>`";
  let genVerifyDecl = 1;
}

def Intent_PublicParameterAttr : AttrDef<Intent_Dialect, "PublicParameter"> {
  let mnemonic = "public_parameter";
  let summary = "One runtime author parameter in declaration order";
  let parameters = (ins "::mlir::StringAttr":$name, "::mlir::Type":$type);
  let assemblyFormat = "`<` $name `,` $type `>`";
  let genVerifyDecl = 1;
}

def Intent_InterfaceAttr : AttrDef<Intent_Dialect, "Interface"> {
  let mnemonic = "interface";
  let summary = "The public invocation contract, independent of physical argument packing";
  let parameters = (ins "::mlir::ArrayAttr":$arguments);
  let assemblyFormat = "`<` $arguments `>`";
  let genVerifyDecl = 1;
}

def Intent_FunctionKindAttr : AttrDef<Intent_Dialect, "FunctionKind"> {
  let mnemonic = "function_kind";
  let summary = "Canonical kernel or helper function role";
  let parameters = (ins "uint32_t":$kind);
  let assemblyFormat = "`<` $kind `>`";
  let genVerifyDecl = 1;
}

def Intent_SparseFormatAttr : AttrDef<Intent_Dialect, "SparseFormat"> {
  let mnemonic = "sparse_format";
  let summary = "Closed canonical structured-sparsity format schema";
  let parameters = (ins "uint32_t":$kind, "uint32_t":$compressionAxis);
  let assemblyFormat = "`<` $kind `,` $compressionAxis `>`";
  let genVerifyDecl = 1;
}

def Intent_TensorShape : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::TensorShapeAttr>($_self)">,
    "typed tensor dimension identities"> {
  let storageType = "::intent::TensorShapeAttr";
  let returnType = "::intent::TensorShapeAttr";
  let convertFromStorage = "$_self";
}
def Intent_ShapeRelation : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::ShapeRelationAttr>($_self)">,
    "typed logical shape relation"> {
  let storageType = "::intent::ShapeRelationAttr";
  let returnType = "::intent::ShapeRelationAttr";
  let convertFromStorage = "$_self";
}
def Intent_IndexRelation : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::IndexRelationAttr>($_self)">,
    "typed logical index relation"> {
  let storageType = "::intent::IndexRelationAttr";
  let returnType = "::intent::IndexRelationAttr";
  let convertFromStorage = "$_self";
}
def Intent_SparseFormat : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::SparseFormatAttr>($_self)">,
    "typed sparse format schema"> {
  let storageType = "::intent::SparseFormatAttr";
  let returnType = "::intent::SparseFormatAttr";
  let convertFromStorage = "$_self";
}
#endif
