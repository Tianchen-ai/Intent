#ifndef INTENT_ANALYSIS_CONTROLFLOW_H
#define INTENT_ANALYSIS_CONTROLFLOW_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {

enum class ControlFlowEdgeKind {
  Entry,          // Parent operand -> region argument (initial state).
  RegionTransfer, // Terminator operand -> another invocation's region argument.
  Exit,           // Terminator operand -> parent result.
  Bypass,         // Parent operand -> parent result, e.g. a zero-trip loop.
  Branch          // CFG terminator operand -> successor block argument.
};

// One actual forwarding slot. Equal SSA values in two operands do not merge
// their destinations. RegionTransfer includes self-edges and both directions
// of a while; source/target regions preserve their distinct control boundaries.
struct ControlFlowEdge {
  mlir::OpOperand *operand;
  mlir::Value target;
  mlir::Region *sourceRegion;
  mlir::Region *targetRegion;
  ControlFlowEdgeKind kind;
};

struct ControlFlowEdges {
  llvm::SmallVector<ControlFlowEdge> edges;
  // False if an incoming value is produced rather than forwarded, or an
  // interface does not describe the complete requested boundary. A produced
  // value has a null operand; it never establishes a value/alias equivalence.
  bool complete = false;
};

// Construction can query existing values before filling sibling regions.
// Native region interfaces may inspect terminators even for an entry-edge query.
bool hasCompleteControlFlowRegions(mlir::Operation *operation);

// Current-IR queries only. Results borrow OpOperand slots and are invalidated
// by operand/region mutation. A known condition/non-forwarded operand has an
// empty, complete outgoing relation; an unsupported owner is incomplete.
ControlFlowEdges queryControlFlowIncoming(mlir::Value target);
ControlFlowEdges queryControlFlowOutgoing(mlir::OpOperand &operand);

} // namespace intent
#endif
