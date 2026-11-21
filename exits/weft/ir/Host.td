include "mlir/IR/OpBase.td"

def WeftHost_Dialect : Dialect {
  let name = "weft_host";
  let cppNamespace = "::intentdsl::wefthost";
}

// Dispatch disjoint intervals of the domain to one serial task instance per
// interval. Matrix captures advance with the interval; vectors and scalars are shared.
def DispatchRows : Op<WeftHost_Dialect, "dispatch_rows", [IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins Variadic<AnyType>:$captures, I64Attr:$domain_arg,
                       ArrayAttr:$rows, StrAttr:$task_symbol);
  let results = (outs Index:$completion);
  let regions = (region AnyRegion:$task);
}

def PanelSweep : Op<WeftHost_Dialect, "sweep", [SingleBlock]> {
  let arguments=(ins StrAttr:$combine, StrAttr:$tail);
  let results=(outs Variadic<AnyType>:$summary);
  let regions=(region AnyRegion:$panels);
}
def PanelRead : Op<WeftHost_Dialect,"read_panel"> {
  let arguments=(ins AnyMemRef:$source,Index:$row,Index:$offset,Index:$extent);
  let results=(outs AnyTensor:$value);
}
def PanelWrite : Op<WeftHost_Dialect,"write_panel"> {
  let arguments=(ins AnyMemRef:$dest,Index:$row,Index:$offset,Index:$extent,AnyTensor:$value);
}
def PanelYield : Op<WeftHost_Dialect,"yield",[Terminator]> {
  let arguments=(ins Variadic<AnyType>:$partial);
}
