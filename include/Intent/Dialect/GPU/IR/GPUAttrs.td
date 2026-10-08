#ifndef INTENT_DIALECT_GPU_IR_GPUATTRS_TD
#define INTENT_DIALECT_GPU_IR_GPUATTRS_TD

include "Intent/Dialect/GPU/IR/GPUDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class IntentGPU_I32Enum<string name, string description,
                        list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::intent::gpu";
}

class IntentGPU_I32EnumAttr<string mnemonic, IntentGPU_I32Enum enumInfo>
    : EnumAttr<IntentGPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def IntentGPU_AtomicSharingDomain : IntentGPU_I32Enum<
    "AtomicSharingDomain", "Logical participants in one physical atomic object", [
      I32EnumAttrCase<"ProgramInstance", 0, "program_instance">,
      I32EnumAttrCase<"KernelInvocation", 1, "kernel_invocation">
    ]>;
def IntentGPU_AtomicSharingDomainAttr : IntentGPU_I32EnumAttr<
    "atomic_sharing", IntentGPU_AtomicSharingDomain>;

def IntentGPU_BufferScope : IntentGPU_I32Enum<
    "BufferScope", "Logical allocation scope of a physical mutable buffer", [
      I32EnumAttrCase<"ProgramPrivate", 0, "program_private">,
      I32EnumAttrCase<"IterationPrivate", 1, "iteration_private">,
      I32EnumAttrCase<"InvocationWorkspace", 2, "invocation_workspace">
    ]>;
def IntentGPU_BufferScopeAttr : IntentGPU_I32EnumAttr<
    "buffer_scope", IntentGPU_BufferScope>;

def IntentGPU_BufferInitialization : IntentGPU_I32Enum<
    "BufferInitialization", "Initialization obligation of a physical buffer", [
      I32EnumAttrCase<"FullValue", 0, "full_value">,
      I32EnumAttrCase<"FirstWrite", 1, "first_write">
    ]>;
def IntentGPU_BufferInitializationAttr : IntentGPU_I32EnumAttr<
    "buffer_initialization", IntentGPU_BufferInitialization>;

def IntentGPU_ParameterRefAttr : AttrDef<IntentGPU_Dialect, "ParameterRef"> {
  let mnemonic = "parameter_ref";
  let summary = "A reference to one kernel-owned compile-time declaration";
  let parameters = (ins "::mlir::StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_ArgumentKind : IntentGPU_I32Enum<
    "ArgumentKind", "Binding of one physical function argument", [
      I32EnumAttrCase<"Public", 0, "public">,
      I32EnumAttrCase<"Dimension", 1, "dimension">,
      I32EnumAttrCase<"Stride", 2, "stride">,
      I32EnumAttrCase<"Workspace", 3, "workspace">
    ]>;

def IntentGPU_ArgumentRefAttr : AttrDef<IntentGPU_Dialect, "ArgumentRef"> {
  let mnemonic = "argument_ref";
  let summary = "Stable identity of a kernel-owned physical argument";
  let parameters = (ins "uint64_t":$id);
  let assemblyFormat = "`<` $id `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_ArgumentBindingAttr : AttrDef<IntentGPU_Dialect, "ArgumentBinding"> {
  let mnemonic = "argument_binding";
  let summary = "Public or private origin of a physical function argument";
  let parameters = (ins
    "::intent::gpu::ArgumentRefAttr":$reference,
    EnumParameter<IntentGPU_ArgumentKind>:$kind,
    OptionalParameter<"::mlir::IntegerAttr">:$publicOrdinal,
    OptionalParameter<"::intent::gpu::ArgumentRefAttr">:$source,
    OptionalParameter<"::mlir::IntegerAttr">:$axis,
    OptionalParameter<"::mlir::IntegerAttr">:$dimension
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_PhysicalExprKind : IntentGPU_I32Enum<
    "PhysicalExprKind", "Typed physical integer expression kind", [
      I32EnumAttrCase<"Constant", 0, "constant">,
      I32EnumAttrCase<"Parameter", 1, "parameter">,
      I32EnumAttrCase<"Dimension", 2, "dimension">,
      I32EnumAttrCase<"ScalarABI", 3, "scalar_abi">,
      I32EnumAttrCase<"Add", 4, "add">,
      I32EnumAttrCase<"Multiply", 5, "multiply">,
      I32EnumAttrCase<"CeilDiv", 6, "ceil_div">,
      I32EnumAttrCase<"Minimum", 7, "minimum">,
      I32EnumAttrCase<"Select", 8, "select">,
      I32EnumAttrCase<"Subtract", 9, "subtract">,
      I32EnumAttrCase<"FloorDiv", 10, "floor_div">,
      I32EnumAttrCase<"Maximum", 11, "maximum">,
      I32EnumAttrCase<"NextPowerOfTwo", 12, "next_power_of_two">
    ]>;

def IntentGPU_ParameterRole : IntentGPU_I32Enum<
    "ParameterRole", "Semantic role of a physical compile-time parameter", [
      I32EnumAttrCase<"OwnershipM", 0, "ownership_m">,
      I32EnumAttrCase<"OwnershipN", 1, "ownership_n">,
      I32EnumAttrCase<"Reduction", 2, "reduction">,
      I32EnumAttrCase<"ScanChunk", 3, "scan_chunk">,
      I32EnumAttrCase<"ProviderWarps", 4, "provider_warps">,
      I32EnumAttrCase<"ProviderStages", 5, "provider_stages">,
      I32EnumAttrCase<"ProviderCTAs", 6, "provider_ctas">,
      I32EnumAttrCase<"TraversalWorkers", 8, "traversal_workers">,
      I32EnumAttrCase<"TraversalGroup", 9, "traversal_group">,
      I32EnumAttrCase<"ResidentWorkers", 10, "resident_workers">,
      I32EnumAttrCase<"FullCoverage", 11, "full_coverage">,
      I32EnumAttrCase<"ReductionOuter", 12, "reduction_outer">,
      I32EnumAttrCase<"ReductionInner", 13, "reduction_inner">,
      I32EnumAttrCase<"ProviderAccessForm", 14, "provider_access_form">,
      I32EnumAttrCase<"ProviderOccupancy", 15, "provider_occupancy">,
      I32EnumAttrCase<"ProviderLoadPolicy", 16, "provider_load_policy">
    ]>;

def IntentGPU_ParameterCategory : IntentGPU_I32Enum<
    "ParameterCategory", "Physical parameter decision family", [
      I32EnumAttrCase<"Pointwise", 0, "pointwise">,
      I32EnumAttrCase<"Reduction", 1, "reduction">,
      I32EnumAttrCase<"Scan", 2, "scan">,
      I32EnumAttrCase<"Contraction", 3, "contraction">,
      I32EnumAttrCase<"Execution", 4, "execution">,
      I32EnumAttrCase<"Coverage", 5, "coverage">,
      I32EnumAttrCase<"Provider", 6, "provider">,
      I32EnumAttrCase<"RegionReduction", 7, "region_reduction">,
      I32EnumAttrCase<"RegionContraction", 8, "region_contraction">,
      I32EnumAttrCase<"PersistentContraction", 9, "persistent_contraction">,
      I32EnumAttrCase<"Histogram", 10, "histogram">
    ]>;

def IntentGPU_PhysicalExprAttr : AttrDef<IntentGPU_Dialect, "PhysicalExpr"> {
  let mnemonic = "expr";
  let summary = "Typed launch or compile-time physical integer expression";
  let parameters = (ins
    EnumParameter<IntentGPU_PhysicalExprKind>:$kind,
    "int64_t":$value,
    "::mlir::Attribute":$symbol,
    "::mlir::ArrayAttr":$operands
  );
  let assemblyFormat = "`<` $kind `,` $value `,` $symbol `,` $operands `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::intent::gpu::ParameterRefAttr getParameterReference() const;
    ::intent::gpu::ArgumentRefAttr getArgumentReference() const;
  }];
}

def IntentGPU_ConfigurationBindingPhase : IntentGPU_I32Enum<
    "ConfigurationBindingPhase", "When a compile-time declaration is bound", [
      I32EnumAttrCase<"Shared", 0, "shared">,
      I32EnumAttrCase<"Provider", 1, "provider">,
      I32EnumAttrCase<"Deferred", 2, "deferred">
    ]>;

def IntentGPU_ParameterBindingAttr : AttrDef<IntentGPU_Dialect, "ParameterBinding"> {
  let mnemonic = "parameter_binding";
  let summary = "Declared physical source and coverage binding for a parameter";
  let parameters = (ins
    OptionalParameter<"::mlir::IntegerAttr">:$dimension,
    OptionalParameter<"::intent::gpu::PhysicalSourceAttr">:$source,
    OptionalParameter<"::intent::gpu::PhysicalExprAttr">:$coverageBound,
    OptionalParameter<"::mlir::ArrayAttr">:$group,
    "bool":$pointwiseChunk,
    "bool":$pointwiseLocal
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ParameterBindingAttr withDimension(::mlir::IntegerAttr value) const;
    ParameterBindingAttr withSource(::intent::gpu::PhysicalSourceAttr value) const;
    ParameterBindingAttr withCoverageBound(::intent::gpu::PhysicalExprAttr value) const;
    ParameterBindingAttr withGroup(::mlir::ArrayAttr value) const;
    ParameterBindingAttr withPointwiseChunk(bool value) const;
    ParameterBindingAttr withPointwiseLocal(bool value) const;
  }];
}

def IntentGPU_ParameterAttr : AttrDef<IntentGPU_Dialect, "Parameter"> {
  let mnemonic = "parameter";
  let summary = "One typed compile-time physical parameter and its legal candidates";
  let parameters = (ins
    "::mlir::StringAttr":$name,
    "::mlir::Type":$valueType,
    EnumParameter<IntentGPU_ParameterRole>:$role,
    EnumParameter<IntentGPU_ParameterCategory>:$category,
    "uint32_t":$elementBitWidth,
    "::mlir::DenseI64ArrayAttr":$candidates,
    EnumParameter<IntentGPU_ConfigurationBindingPhase>:$phase,
    "::intent::gpu::ParameterBindingAttr":$binding
  );
  let assemblyFormat =
      "`<` $name `,` $valueType `,` $role `,` $category `,` $elementBitWidth `,` $candidates `,` $phase `,` $binding `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::intent::gpu::ParameterRefAttr getReference() const;
    bool isExtent() const;
    bool isDeferred() const;
    ParameterAttr withName(::mlir::StringAttr value) const;
    ParameterAttr withCandidates(::mlir::DenseI64ArrayAttr value) const;
    ParameterAttr withPhase(ConfigurationBindingPhase value) const;
    ParameterAttr withBinding(ParameterBindingAttr value) const;
  }];
}

def IntentGPU_ConfigurationStage : IntentGPU_I32Enum<
    "ConfigurationStage", "Binding completeness of the current candidate set", [
      I32EnumAttrCase<"Shared", 0, "shared">,
      I32EnumAttrCase<"Complete", 1, "complete">
    ]>;

def IntentGPU_ConfigurationRequirementKind : IntentGPU_I32Enum<
    "ConfigurationRequirementKind", "What a candidate requirement establishes", [
      I32EnumAttrCase<"Legality", 0, "legality">,
      I32EnumAttrCase<"NominalBudget", 1, "nominal_budget">
    ]>;

def IntentGPU_ConfigurationRequirementMetric : IntentGPU_I32Enum<
    "ConfigurationRequirementMetric", "Units of a structural candidate requirement", [
      I32EnumAttrCase<"FragmentElements", 0, "fragment_elements">,
      I32EnumAttrCase<"FragmentRegisterWords", 1, "fragment_register_words">,
      I32EnumAttrCase<"FragmentBytes", 2, "fragment_bytes">,
      I32EnumAttrCase<"ResidentWorkers", 3, "resident_workers">
    ]>;

def IntentGPU_ConfigurationRequirementPredicate : IntentGPU_I32Enum<
    "ConfigurationRequirementPredicate", "A finite candidate quantity test", [
      I32EnumAttrCase<"LessEqual", 0, "less_equal">,
      I32EnumAttrCase<"Positive", 1, "positive">,
      I32EnumAttrCase<"PowerOfTwo", 2, "power_of_two">,
      I32EnumAttrCase<"MultipleOf", 3, "multiple_of">,
      I32EnumAttrCase<"Equal", 4, "equal">
    ]>;

def IntentGPU_ConfigurationRequirementAttr : AttrDef<IntentGPU_Dialect, "ConfigurationRequirement"> {
  let mnemonic = "configuration_requirement";
  let summary = "An explicit predicate on candidate quantities with an optional boolean activation";
  let parameters = (ins
    EnumParameter<IntentGPU_ConfigurationRequirementKind>:$kind,
    EnumParameter<IntentGPU_ConfigurationRequirementMetric>:$metric,
    EnumParameter<IntentGPU_ConfigurationRequirementPredicate>:$predicate,
    "::intent::gpu::PhysicalExprAttr":$usage,
    OptionalParameter<"::intent::gpu::PhysicalExprAttr">:$limit,
    OptionalParameter<"::intent::gpu::ParameterRefAttr">:$activation,
    "::mlir::StringAttr":$message
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_ConfigurationSetAttr : AttrDef<IntentGPU_Dialect, "ConfigurationSet"> {
  let mnemonic = "configurations";
  let summary = "Ordered candidate bindings for the current physical program";
  let parameters = (ins EnumParameter<IntentGPU_ConfigurationStage>:$stage,
                        "::mlir::ArrayAttr":$rows,
                        "::mlir::ArrayAttr":$requirements);
  let assemblyFormat = "`<` $stage `,` $rows `,` $requirements `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_TuningProfileTableAttr : AttrDef<IntentGPU_Dialect, "TuningProfileTable"> {
  let mnemonic = "tuning_profile_table";
  let summary = "Ordered column schema and finite candidate rows by family";
  let parameters = (ins "::mlir::ArrayAttr":$columns,
                        "::mlir::DictionaryAttr":$families);
  let assemblyFormat = "`<` $columns `,` $families `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_TuningProfilesAttr : AttrDef<IntentGPU_Dialect, "TuningProfiles"> {
  let mnemonic = "tuning_profiles";
  let summary = "Resolved, ordered shared and provider candidate profile tables";
  let parameters = (ins "::mlir::DictionaryAttr":$spaces);
  let assemblyFormat = "`<` $spaces `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_AxisMapAttr : AttrDef<IntentGPU_Dialect, "AxisMap"> {
  let mnemonic = "axis_map";
  let summary = "Fragment axis to immutable logical source-axis mapping";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$sourceAxis,
    "int64_t":$dimensionId,
    "uint32_t":$fragmentAxis,
    "bool":$derived
  );
  let assemblyFormat =
      "`<` $sourceId `,` $sourceAxis `,` $dimensionId `,` $fragmentAxis `,` $derived `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_PhysicalSourceAttr : AttrDef<IntentGPU_Dialect, "PhysicalSource"> {
  let mnemonic = "source";
  let summary = "One immutable logical source-axis identity";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$sourceAxis,
    "bool":$derived
  );
  let assemblyFormat = "`<` $sourceId `,` $sourceAxis `,` $derived `>`";
}

def IntentGPU_ReshapeGroupAttr : AttrDef<IntentGPU_Dialect, "ReshapeGroup"> {
  let mnemonic = "reshape_group";
  let summary = "One contiguous row-major physical reshape reassociation group";
  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$sourceAxes,
    "::mlir::DenseI64ArrayAttr":$resultAxes
  );
  let assemblyFormat = "`<` $sourceAxes `,` $resultAxes `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_ViewLayoutAttr : AttrDef<IntentGPU_Dialect, "ViewLayout"> {
  let mnemonic = "view_layout";
  let summary = "Host-visible physical view shape and stride expressions";
  let parameters = (ins
    "::mlir::ArrayAttr":$extents,
    "::mlir::DenseI64ArrayAttr":$dimensionIds,
    "::mlir::ArrayAttr":$strides
  );
  let assemblyFormat =
      "`<` $extents `,` $dimensionIds `,` $strides `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_CapabilitiesAttr : AttrDef<IntentGPU_Dialect, "Capabilities"> {
  let mnemonic = "capabilities";
  let summary = "Selected GPU resources and provider primitives used by shared lowering";
  let parameters = (ins
    "int64_t":$computeUnits,
    "int64_t":$sharedMemoryPerUnit,
    "int64_t":$maxDynamicSharedMemoryPerBlock,
    "int64_t":$registersPerUnit,
    "int64_t":$maxThreadsPerBlock,
    "int64_t":$computeCapabilityMajor,
    "int64_t":$computeCapabilityMinor,
    "int64_t":$singleToDoublePrecisionPerfRatio,
    "bool":$matrixUnits,
    "bool":$dynamicVectorWidth,
    "bool":$nativeTupleReductions,
    "bool":$nativeTupleReductionRequiresConstantIdentity,
    "bool":$nativeFragmentGather
  );
  let assemblyFormat =
      "`<` $computeUnits `,` $sharedMemoryPerUnit `,` $maxDynamicSharedMemoryPerBlock `,` $registersPerUnit `,` $maxThreadsPerBlock `,` $computeCapabilityMajor `,` $computeCapabilityMinor `,` $singleToDoublePrecisionPerfRatio `,` $matrixUnits `,` $dynamicVectorWidth `,` $nativeTupleReductions `,` $nativeTupleReductionRequiresConstantIdentity `,` $nativeFragmentGather `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_PhysicalExpr : DialectAttr<
    IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::PhysicalExprAttr>($_self)">,
    "typed physical expression"> {
  let storageType = "::intent::gpu::PhysicalExprAttr";
  let returnType = "::intent::gpu::PhysicalExprAttr";
  let convertFromStorage = "$_self";
}

#endif
