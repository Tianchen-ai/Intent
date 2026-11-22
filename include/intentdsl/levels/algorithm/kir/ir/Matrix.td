// Logical operations shared by execution nodes which preserve the value region.
multiclass MatrixExpressions<Dialect d> {
  def Domain : Op<d,"matrix_domain",[SingleBlock]> {
    let arguments=(ins AnyMemRef:$output); let regions=(region AnyRegion:$body);
  }
  def Contract : Op<d,"contract"> {
    let arguments=(ins AnyMemRef:$lhs, AnyMemRef:$rhs, Variadic<AnyMemRef>:$segments,
                       I64Attr:$lhs_k,I64Attr:$rhs_k,I64Attr:$lhs_batch,I64Attr:$rhs_batch);
    let results=(outs AnyRankedTensor:$sums);
  }
  def ScaleTranspose : Op<d,"transpose_scale"> {
    let arguments=(ins AnyMemRef:$input);let results=(outs AnyRankedTensor:$scale);
  }
  def WindowContract : Op<d,"window_contract",[SingleBlock,IsolatedFromAbove]> {
    let arguments=(ins AnyMemRef:$lhs,AnyMemRef:$rhs,AnyMemRef:$coefficients,
                       AnyMemRef:$prefix,I64Attr:$head_group);
    let results=(outs AnyRankedTensor:$sums);
    let regions=(region AnyRegion:$map);
  }
  def PanelValue : Op<d,"panel_value",[Terminator]> {let arguments=(ins BF16:$element);}
  def ScaledContract : Op<d,"scaled_contract"> {
    let arguments=(ins AnyMemRef:$lhs,AnyMemRef:$leftScale,AnyMemRef:$rhs,AnyRankedTensor:$rightScale);
    let results=(outs AnyRankedTensor:$sums);
  }
  def Q8Encode : Op<d,"quantize_q8k"> {
    let arguments=(ins AnyMemRef:$values);let results=(outs AnyRankedTensor:$records);
  }
  def RecordDot : Op<d,"q4k_dot"> {
    let arguments=(ins AnyMemRef:$weights,AnyRankedTensor:$activation);
    let results=(outs AnyRankedTensor:$sum);
  }
  def AddColumns : Op<d,"add_columns"> {
    let arguments=(ins AnyRankedTensor:$input,AnyMemRef:$bias);let results=(outs AnyRankedTensor:$result);
  }
  def ConvertMatrix : Op<d,"convert_matrix"> {
    let arguments=(ins AnyRankedTensor:$input);let results=(outs AnyRankedTensor:$result);
  }
  def Publish : Op<d,"publish_matrix"> {
    let arguments=(ins AnyMemRef:$output,AnyRankedTensor:$values);
  }
}

// Materialized by execution models. One complete accumulator is yielded after
// all K panels; only this value is visible to bias, conversion or publication.
multiclass MatrixPanelLoop<Dialect d> {
  def Fold : Op<d,"panel_fold",[SingleBlock]> {
    let arguments=(ins I64Attr:$bound,I64Attr:$step);
    let results=(outs AnyRankedTensor:$accumulated);let regions=(region AnyRegion:$panel);
  }
  def Yield : Op<d,"panel_yield",[Terminator]> {let arguments=(ins AnyRankedTensor:$partial);}
}
