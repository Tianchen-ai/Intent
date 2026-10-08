#ifndef INTENT_SERIALIZATION_MEMORYDESCRIPTOR_H
#define INTENT_SERIALIZATION_MEMORYDESCRIPTOR_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace intent {

// A ranked memref carries its allocation base independently of the view offset.
// Control flow transfers every field; deallocation always consumes the base.
// These are source expressions for facts already present in the physical IR.
struct MemoryDescriptor {
  std::string base;
  std::string offset;
  llvm::SmallVector<std::string> sizes;
  llvm::SmallVector<std::string> strides;

  llvm::SmallVector<std::string> components() const {
    llvm::SmallVector<std::string> result{base, offset};
    llvm::append_range(result, sizes);
    llvm::append_range(result, strides);
    return result;
  }

  static MemoryDescriptor fromComponents(
      mlir::MemRefType type, llvm::ArrayRef<std::string> fields) {
    assert(fields.size() == 2 + 2 * type.getRank());
    unsigned rank = type.getRank();
    return {fields[0], fields[1],
            {fields.begin() + 2, fields.begin() + 2 + rank},
            {fields.begin() + 2 + rank, fields.end()}};
  }
};

} // namespace intent
#endif
