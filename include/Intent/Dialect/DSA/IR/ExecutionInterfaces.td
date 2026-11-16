#ifndef INTENT_DSA_EXECUTION_INTERFACES_TD
#define INTENT_DSA_EXECUTION_INTERFACES_TD
include "mlir/IR/OpBase.td"
def IntentDSA_AsyncEffectOpInterface : OpInterface<"AsyncEffectOpInterface"> {
  let cppNamespace = "::intent::dsa";
  let description = [{
    An issued operation whose memory effects can outlive the operation's lexical
    position. The completion domain identifies the actual target engine/order
    domain and participant scope. Empty means synchronous completion. Memory
    effects identify buffers; this interface defines when those uses complete.
  }];
  let methods = [InterfaceMethod<"Pending completion, if execution is asynchronous.",
    "std::optional<::intent::dsa::CompletionDomain>", "getPendingCompletion">];
}
def IntentDSA_CompletionOpInterface : OpInterface<"CompletionOpInterface"> {
  let cppNamespace = "::intent::dsa";
  let description = [{
    Completion boundary for explicitly identified engine and participant domains.
    It does not imply a particular vendor barrier or participant count. Control
    legality and target realization must guarantee the declared completion.
  }];
  let methods = [InterfaceMethod<"Domains completed before following operations execute.",
    "::llvm::SmallVector<::intent::dsa::CompletionDomain>", "getCompletedDomains">];
}
#endif
