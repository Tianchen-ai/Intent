// A panel tests both active E8M0 scale tiles and executes exactly one arm.
multiclass ScaleDispatch<Dialect d> {
  def Select : Op<d,"scale_dispatch"> {
    let arguments=(ins AnyMemRef:$left, AnyMemRef:$right, DenseI64ArrayAttr:$interval,
                       StrAttr:$on_match, StrAttr:$otherwise);
  }
}
