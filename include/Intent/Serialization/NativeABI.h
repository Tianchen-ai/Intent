#ifndef INTENT_SERIALIZATION_NATIVEABI_H
#define INTENT_SERIALIZATION_NATIVEABI_H

#include "Intent/Dialect/Intent/IR/Interface.h"
#include <optional>
#include <utility>

namespace intent {

enum class NativeSlotRole { Pointer, Extent, Stride, Scalar };

struct NativeSlot {
  NativeSlotRole role;
  unsigned parameter;
  std::optional<unsigned> axis;
  mlir::Type carrier;
  mlir::Type element;

  std::string name() const;
};

struct NativeABI {
  llvm::SmallVector<NativeSlot> slots;
  llvm::json::Array serialize() const;
};

enum class NativeViewLayout { Strided, Contiguous };

struct NativeViewRequirements {
  NativeViewLayout layout;
  int64_t alignment;
};

struct NativeEntryRequirements {
  llvm::SmallVector<std::pair<unsigned, NativeViewRequirements>> views;
  llvm::SmallVector<std::pair<unsigned, unsigned>> disjoint;

  llvm::json::Object serialize() const;
};

// Export the current family's physical entry requirements separately from the
// author's public view constraints. The provider queries its typed IR facts;
// this function owns parameter enumeration and writable-view pair expansion.
mlir::FailureOr<NativeEntryRequirements> queryNativeEntryRequirements(
    mlir::func::FuncOp function, InterfaceAttr interface,
    bool disjointWritableViews,
    llvm::function_ref<mlir::FailureOr<NativeViewRequirements>(unsigned, ViewType)> viewRequirements);

// CPU/DSA native entry flattening. The family has already verified each public
// argument's physical storage; only the provider decides its scalar C carrier.
mlir::FailureOr<NativeABI> queryNativeABI(
    mlir::func::FuncOp function, InterfaceAttr interface,
    llvm::function_ref<mlir::FailureOr<mlir::Type>(mlir::Type)> scalarCarrier);

} // namespace intent
#endif
