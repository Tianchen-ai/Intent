#ifndef INTENT_CPU_TRANSFORMS_REDUCTIONSOURCES_H
#define INTENT_CPU_TRANSFORMS_REDUCTIONSOURCES_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <memory>

namespace intent::cpu {

// Source members and partial states are different inputs to a reduction. This
// adapter only supplies the former; it never specializes the combine's formals.
class ReductionSources {
public:
  explicit ReductionSources(mlir::linalg::GenericOp consumer,
      mlir::Value completedSupply = {},
      llvm::ArrayRef<mlir::Operation *> supplyReaders = {});
  ~ReductionSources();
  bool hasReplays() const;
  bool replays(unsigned input) const;
  void beginMember();
  mlir::FailureOr<bool> foldUniformMembers();
  mlir::FailureOr<mlir::Value> materialize(
      mlir::OpBuilder &builder, unsigned input, mlir::ValueRange coordinates,
      int64_t lanes = 0);
  void eraseUnusedProducers();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace intent::cpu
#endif
