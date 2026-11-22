multiclass ColumnTiles<Dialect d> {
  def ReadMatrix : Op<d,"read_owner_matrix"> {let arguments=(ins AnyMemRef:$view,Index:$batch);let results=(outs AnyRankedTensor:$matrix);}
  def Column : Op<d,"column",[Pure]> {let arguments=(ins AnyRankedTensor:$matrix,Index:$index);let results=(outs AnyRankedTensor:$column);}
  def Element : Op<d,"element",[Pure]> {let arguments=(ins AnyRankedTensor:$column,Index:$index);let results=(outs F32:$value);}
  def LowerColumn : Op<d,"lower_column",[Pure]> {let arguments=(ins AnyRankedTensor:$residual,F32:$diagonal,Index:$index);let results=(outs AnyRankedTensor:$values);}
  def Outer : Op<d,"outer_product",[Pure]> {let arguments=(ins AnyRankedTensor:$left,AnyRankedTensor:$right);let results=(outs AnyRankedTensor:$matrix);}
  def StoreColumn : Op<d,"store_column"> {let arguments=(ins AnyMemRef:$view,Index:$batch,Index:$column,AnyRankedTensor:$values);}
}
