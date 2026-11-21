include "mlir/IR/OpBase.td"

def WeftHost_Dialect : Dialect {
  let name = "weft_host";
  let cppNamespace = "::intentdsl::wefthost";
}

// Dispatch disjoint intervals of the domain to one serial task instance per
// interval. Matrix captures advance with the interval; vectors and scalars are shared.
def DispatchRows : Op<WeftHost_Dialect, "dispatch_rows", [IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins Variadic<AnyType>:$captures, I64Attr:$domain_arg,
                       DenseI64ArrayAttr:$rows, StrAttr:$task_symbol);
  let results = (outs Index:$completion);
  let regions = (region AnyRegion:$task);
}
