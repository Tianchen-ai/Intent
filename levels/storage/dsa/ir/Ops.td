include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SupplyDialect : Dialect {
  let name = "dsa";
  let cppNamespace = "::intentdsl::dsa";
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
  let regions = (region AnyRegion:$supply, AnyRegion:$compute);
}
def Copy : Inst<"copy_in", (ins Arg<AnyMemRef, "device view", [MemRead]>:$source,
                                BoolAttr:$matrix), (outs AnyTensor:$pending)>;
def Ready : Inst<"await", (ins AnyTensor:$pending), (outs AnyTensor:$value)>;
def Write : Inst<"write_back", (ins Arg<AnyMemRef, "device view", [MemWrite]>:$dest,
                                     AnyTensor:$value), (outs)>;
def Finish : Inst<"yield", (ins Variadic<AnyType>:$values), (outs), [Terminator]>;
def Send : Inst<"issued", (ins Variadic<AnyType>:$pending), (outs), [Terminator]>;
def Drain : Inst<"drain", (ins), (outs)>;
def Summary : Inst<"reduce", (ins AnyTensor:$value, StrAttr:$algebra),
                             (outs AnyTensor:$result)>;
