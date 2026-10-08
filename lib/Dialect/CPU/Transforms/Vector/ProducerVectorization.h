#ifndef INTENT_CPU_TRANSFORMS_VECTOR_PRODUCERVECTORIZATION_H
#define INTENT_CPU_TRANSFORMS_VECTOR_PRODUCERVECTORIZATION_H

#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include <optional>

namespace intent::cpu {

// A pass-local SIMD adapter for an already-proved scalar producer. Effects,
// source stability and full-width bounds remain the caller's responsibility.
class ProducerVectorization {
public:
  ProducerVectorization(const ProducerReplay &payload, mlir::Value coordinate,
                        mlir::ValueRange varyingFrontier = {})
      : payload(payload), coordinate(coordinate),
        varyingFrontier(varyingFrontier.begin(), varyingFrontier.end()) {}

  bool dependsOn(mlir::Value value, mlir::Value input) const;
  bool isUniform(mlir::Value value) const;
  std::optional<int64_t> coefficient(mlir::Value value) const;
  static bool isElementType(mlir::Type type);

  // A null list requires statically unit-stride loads. Otherwise append the
  // external descriptors whose innermost stride the caller must guard.
  bool canWiden(mlir::Value value,
                llvm::SmallVectorImpl<mlir::Value> *guardedMemories = nullptr) const;

  mlir::FailureOr<mlir::Value> materialize(
      mlir::Value value, int64_t width, mlir::OpBuilder &builder,
      mlir::IRMapping &scalars, mlir::IRMapping &vectors,
      mlir::OpBuilder *invariants = nullptr) const;
  mlir::FailureOr<mlir::Value> materializeScalar(
      mlir::Value value, mlir::OpBuilder &builder, mlir::IRMapping &scalars,
      mlir::OpBuilder *invariants = nullptr) const;

private:
  const ProducerReplay &payload;
  mlir::Value coordinate;
  llvm::SmallVector<mlir::Value> varyingFrontier;
};

} // namespace intent::cpu

#endif
