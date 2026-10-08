#ifndef INTENT_DIALECT_CPU_TRANSFORMS_FINALIZEDCANDIDATES_H
#define INTENT_DIALECT_CPU_TRANSFORMS_FINALIZEDCANDIDATES_H

#include "mlir/IR/BuiltinOps.h"

namespace intent::cpu {

// Call only after provider realization has consumed the configuration and
// implementation bindings into a complete native function. Remove equivalent,
// unreferenced candidate entries, retaining the first entry and its metadata.
// ABI, nested operations/attributes, symbol references and SSA relations must
// match; only entry names, consumed top-level binding metadata and locations
// are excluded. Separate host/task modules require their own closed proof.
void deduplicateFinalizedCandidates(mlir::ModuleOp module);

} // namespace intent::cpu

#endif
