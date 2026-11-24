include "mlir/Interfaces/SideEffectInterfaces.td"

// Each coordinate identifies an independent logical owner. Its body can keep
// SSA values across ordered scf.for iterations and read or write captured views.
multiclass OrderedOwners<Dialect d> {
  def OwnerSpace : Op<d,"owner_space",[SingleBlock]> {
    let arguments=(ins Variadic<AnyMemRef>:$views,DenseI64ArrayAttr:$axes);
    let regions=(region AnyRegion:$body);
  }
  def SliceRead : Op<d,"slice_read"> {
    let arguments=(ins Arg<AnyMemRef, "", [MemRead]>:$view,Variadic<Index>:$indices,I64Attr:$axis);
    let results=(outs AnyRankedTensor:$values);
  }
  def SliceWrite : Op<d,"slice_write"> {
    let arguments=(ins AnyRankedTensor:$values,Arg<AnyMemRef, "", [MemWrite]>:$view,Variadic<Index>:$indices,I64Attr:$axis);
  }
  def LaneSplat : Op<d,"lane_splat",[Pure]> {
    let arguments=(ins AnyType:$scalar,AnyMemRef:$shapeView,I64Attr:$axis);
    let results=(outs AnyRankedTensor:$values);
  }
  def LaneSum : Op<d,"lane_sum",[Pure]> {
    let arguments=(ins AnyRankedTensor:$values);
    let results=(outs F32:$sum);
  }
}
