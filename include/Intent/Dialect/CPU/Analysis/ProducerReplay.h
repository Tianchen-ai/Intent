#ifndef INTENT_DIALECT_CPU_ANALYSIS_PRODUCERREPLAY_H
#define INTENT_DIALECT_CPU_ANALYSIS_PRODUCERREPLAY_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::linalg { class GenericOp; }

namespace intent::cpu {
class StorageAnalysis;

/// A short-lived description of the current payload that will be cloned.
/// Regions belong to their containing operation: their reads are never lifted
/// out of the original control flow. Frontier values must be explicitly bound
/// by the coordinate adapter before cloning. Recompute after changing source IR.
struct ProducerReplay {
  mlir::Operation *scope;
  llvm::SmallVector<mlir::Value> frontier;
  llvm::SmallVector<mlir::Value> externalValues;
  llvm::SmallVector<mlir::Value> reads;
  llvm::SmallVector<mlir::Operation *> operations;
  llvm::SmallVector<mlir::Operation *> nodes;
};

mlir::FailureOr<ProducerReplay> analyzeProducerPayload(
    mlir::Block &body, mlir::ValueRange frontier,
    mlir::ValueRange implicitReads, StorageAnalysis &storage);
mlir::FailureOr<ProducerReplay> analyzeProducerValue(
    mlir::Value value, mlir::Operation *scope, mlir::ValueRange frontier,
    StorageAnalysis &storage);
/// Include implicit operand reads and make used element formals and index
/// results explicit frontier bindings for the consumer's indexing-map adapter.
mlir::FailureOr<ProducerReplay>
analyzeLinalgProducer(mlir::linalg::GenericOp producer, StorageAnalysis &storage);

/// `from` is the complete original execution scope, including implicit linalg
/// reads/writes; `to` is the existing operation before which replay is inserted.
/// Local descriptors may be reconstructed, but their actual backing storage
/// must remain available and stable throughout both scopes and the interval.
bool canReplayProducerAt(const ProducerReplay &payload, mlir::Operation *from,
                         mlir::Operation *to, StorageAnalysis &storage,
                         mlir::ValueRange disjointFrom = {});
} // namespace intent::cpu

#endif
