multiclass ScoreTiles<Dialect d> {
  def ScoreContract : Op<d,"score_contract",[MemoryEffects<[MemRead]>]> {
    let arguments=(ins AnyMemRef:$query,AnyMemRef:$key,F32:$scale,Index:$batch,Index:$head,Index:$row,I64Attr:$group);
    let results=(outs AnyRankedTensor:$scores);
  }
  def CausalExp : Op<d,"causal_exp",[Pure]> {
    let arguments=(ins AnyRankedTensor:$scores,Index:$row);
    let results=(outs AnyRankedTensor:$weights,AnyRankedTensor:$mass);
  }
  def ValueContract : Op<d,"value_contract",[MemoryEffects<[MemRead]>]> {
    let arguments=(ins AnyRankedTensor:$weights,AnyMemRef:$values,Index:$batch,Index:$head,I64Attr:$group);
    let results=(outs AnyRankedTensor:$weighted);
  }
  def NormalizeStore : Op<d,"normalize_store",[MemoryEffects<[MemWrite]>]> {
    let arguments=(ins AnyRankedTensor:$weighted,AnyRankedTensor:$mass,AnyMemRef:$out,Index:$batch,Index:$head,Index:$row);
  }
}
