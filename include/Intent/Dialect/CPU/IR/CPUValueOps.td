#ifndef INTENT_DIALECT_CPU_IR_CPUVALUEOPS_TD
#define INTENT_DIALECT_CPU_IR_CPUVALUEOPS_TD

def IntentCPU_ReadOp : IntentCPU_Op<"read"> {
  let summary = "Immutable tensor snapshot of the current contents of a view";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def IntentCPU_WriteOp : IntentCPU_Op<"write"> {
  let summary = "Copy an immutable tensor value into an explicit destination";
  let arguments = (ins AnyRankedTensor:$source,
      Arg<AnyMemRef, "", [MemWrite]>:$destination);
  let hasVerifier = 1;
}

#endif
