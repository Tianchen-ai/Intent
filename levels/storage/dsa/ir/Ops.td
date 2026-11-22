include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SupplyDialect : Dialect {
  let name = "dsa";
  let cppNamespace = "::intentdsl::dsa";
}

def MatrixSetup : Op<SupplyDialect,"panel_supply",[SingleBlock]> {
 let arguments=(ins I64Attr:$participants,I64Attr:$scratch_bytes);
 let regions=(region AnyRegion:$transfers);
}
def MatrixStage : Op<SupplyDialect,"stage_operand"> {
 let arguments=(ins AnyMemRef:$source, I64Attr:$slot, StrAttr:$access);
 let results=(outs Index:$completion);
}
def MatrixReady : Op<SupplyDialect,"panel_ready",[Terminator]> {
 let arguments=(ins Variadic<Index>:$completed);
}

class Inst<string mnemonic, dag inspec, dag outspec, list<Trait> traits = []>
    : Op<SupplyDialect, mnemonic, traits> {
  let arguments = inspec;
  let results = outspec;
}

// The issue region runs lookahead iterations ahead of its compute region.
// Its returned fragment tokens are ready only after the corresponding await.
def Phase : Inst<"phase", (ins Variadic<AnyType>:$initial, StrAttr:$axis,
                               I32Attr:$slots, I32Attr:$lookahead,
                               StrAttr:$visibility),
                          (outs Variadic<AnyType>:$summary),
                          [RecursiveMemoryEffects]> {
  let regions = (region AnyRegion:$supply, AnyRegion:$compute, AnyRegion:$schedule);
}
def LoadShared : Inst<"load_shared", (ins Index:$start,Index:$limit), (outs)>;
def Window : Inst<"window", (ins AnyMemRef:$domain), (outs IndexTensor:$rows, IndexTensor:$columns, I1Tensor:$mask)>;
def OwnerRows : Inst<"owner_rows", (ins AnyMemRef:$domain), (outs IndexTensor:$rows)>;
// Project a flattened instance-row coordinate into one logical prefix axis.
def MemberAxis : Inst<"member_axis",(ins IndexTensor:$flat,AnyMemRef:$domain,I64Attr:$axis),(outs IndexTensor:$members)>;
def SlabAxis : Inst<"slab_axis",(ins IndexTensor:$flat,AnyMemRef:$domain,I64Attr:$axis),(outs IndexTensor:$members)>;
def OwnerRead : Inst<"owner_read",(ins Arg<AnyMemRef,"owner view",[MemRead]>:$view,IndexTensor:$rows),(outs AnyTensor:$values)>;
def OwnerWrite : Inst<"owner_write",(ins Arg<AnyMemRef,"owner view",[MemWrite]>:$view,IndexTensor:$rows,AnyTensor:$values),(outs I1:$done)>;
def Copy : Inst<"copy_in", (ins Arg<AnyMemRef, "device view", [MemRead]>:$source,
                                I1Tensor:$mask, Variadic<IndexTensor>:$coordinates), (outs AnyTensor:$pending)>;
def Ready : Inst<"await", (ins AnyTensor:$pending), (outs AnyTensor:$value)>;
def Write : Inst<"write_back", (ins Arg<AnyMemRef, "device view", [MemWrite]>:$dest,
                                     AnyTensor:$value, I1Tensor:$mask, Variadic<IndexTensor>:$coordinates, BoolAttr:$async), (outs I1:$completion)>;
def Finish : Inst<"yield", (ins Variadic<AnyType>:$values), (outs), [Terminator]>;
def Send : Inst<"issued", (ins Variadic<AnyType>:$pending), (outs), [Terminator]>;
def Drain : Inst<"drain", (ins Variadic<I1>:$completions), (outs I1:$done)>;
def Complete : Inst<"complete", (ins I1:$done, Variadic<AnyType>:$values), (outs), [Terminator]>;
def GroupLoop : Inst<"task_groups",(ins Index:$first,Index:$end,Index:$step),(outs),[RecursiveMemoryEffects]> {let regions=(region AnyRegion:$body);}
def Cursor : Inst<"range",(ins StrAttr:$source),(outs Index:$start,Index:$limit,Index:$step)>;
def SlotZero : Inst<"slot_zero",(ins),(outs Index:$slot)>;
def Next : Inst<"next",(ins Index:$current,Index:$step),(outs Index:$next)>;
def Flip : Inst<"other_slot",(ins Index:$slot),(outs Index:$alternate)>;
def Submit : Inst<"issue_if_in_bounds",(ins Index:$position,Index:$end,Index:$slot),(outs)>;
def Wait : Inst<"wait_slot",(ins Index:$slot),(outs)>;
def Execute : Inst<"compute_members",(ins Index:$position,Index:$slot),(outs)>;
def Iterate : Inst<"iterate",(ins Index:$start,Index:$end,Index:$step,Index:$initialSlot),(outs),[RecursiveMemoryEffects]> {let regions=(region AnyRegion:$body);}
def Latch : Inst<"advance",(ins Index:$slot),(outs),[Terminator]>;
def End : Inst<"end",(ins),(outs),[Terminator]>;
def Summary : Inst<"reduce", (ins AnyTensor:$value, I1Tensor:$mask, StrAttr:$algebra),
                             (outs AnyTensor:$result)>;
def LookupAxis : Inst<"lookup_axis",(ins AnyMemRef:$indices,AnyTensor:$rows),(outs AnyTensor:$coordinates)>;
def SupplyDialect_ScatterDrain : Op<SupplyDialect,"scatter_fragment"> {let arguments=(ins AnyMemRef:$output,AnyMemRef:$counts,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I1Tensor:$predicate,I32Tensor:$prefix,I32Tensor:$values,I32Attr:$fill);let results=(outs I1:$done);}
def SupplyIndexedPut : Op<SupplyDialect,"unique_put"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$indices,I32Tensor:$values,I1Tensor:$valid,UnitAttr:$in_bounds);let results=(outs I1:$done);}
def SupplyDialect_PublishOffsets : Op<SupplyDialect,"publish_offsets"> {let arguments=(ins AnyMemRef:$offsets,AnyMemRef:$total,AnyTensor:$columns,I1Tensor:$active,I32Tensor:$prefix);let results=(outs I1:$done);}
def SupplyDialect_RangesAt : Op<SupplyDialect,"fill_ranges_at"> {let arguments=(ins AnyMemRef:$out,AnyMemRef:$offsets,AnyTensor:$columns,I1Tensor:$active,I64Attr:$unit,UnitAttr:$disjoint_in_bounds);let results=(outs I1:$done);}
