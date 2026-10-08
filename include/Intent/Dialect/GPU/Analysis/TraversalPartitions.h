#ifndef INTENT_DIALECT_GPU_ANALYSIS_TRAVERSALPARTITIONS_H
#define INTENT_DIALECT_GPU_ANALYSIS_TRAVERSALPARTITIONS_H

namespace mlir {
class Operation;
namespace scf { class ForOp; }
}

namespace intent::gpu {
class PhysicalProgramAnalysis;

// Prove disjoint active Store member domains from their actual access ranges
// and matched control paths. An ancestor loop/If shape alone is not evidence;
// scalar effects without an actual traversal-coordinate anchor are unknown.
bool areDisjointTraversalPartitions(mlir::Operation *first,
                                   mlir::Operation *second,
                                   PhysicalProgramAnalysis &analysis);

// Check before cloning that every observable write has the current range and
// control structure consumed by the traversal-partition proof above.
bool canPeelTraversalEffects(mlir::scf::ForOp loop,
                             PhysicalProgramAnalysis &analysis);

} // namespace intent::gpu
#endif
