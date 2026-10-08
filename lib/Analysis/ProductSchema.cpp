#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;

namespace intent {

ArrayAttr getProductComponents(Type type) {
  if (auto record = dyn_cast<RecordType>(type))
    return record.getFieldTypes();
  if (auto tuple = dyn_cast<intent::TupleType>(type))
    return tuple.getComponentTypes();
  return {};
}

Type getProductComponentType(Type type, ArrayRef<unsigned> fieldPath) {
  for (unsigned field : fieldPath) {
    auto components = getProductComponents(type);
    if (!components || field >= components.size()) return {};
    type = cast<TypeAttr>(components[field]).getValue();
  }
  return type;
}

namespace {

void visitLeaves(Type type, SmallVectorImpl<unsigned> &path,
                 function_ref<void(Type, ArrayRef<unsigned>)> visit) {
  if (auto components = getProductComponents(type)) {
    for (auto [position, component] : llvm::enumerate(components)) {
      path.push_back(position);
      visitLeaves(cast<TypeAttr>(component).getValue(), path, visit);
      path.pop_back();
    }
  } else {
    visit(type, path);
  }
}

size_t leafCount(Type type) {
  size_t count = 0;
  walkProductLeaves(type, [&](Type, ArrayRef<unsigned>) { ++count; });
  return count;
}

} // namespace

void walkProductLeaves(
    Type type, function_ref<void(Type, ArrayRef<unsigned>)> visit) {
  SmallVector<unsigned, 4> path;
  visitLeaves(type, path, visit);
}

void appendProductLeafTypes(Type type, SmallVectorImpl<Type> &types) {
  walkProductLeaves(type, [&](Type leaf, ArrayRef<unsigned>) {
    types.push_back(leaf);
  });
}

void appendProductLeafTypes(TypeRange roots, SmallVectorImpl<Type> &types) {
  for (Type type : roots)
    appendProductLeafTypes(type, types);
}

FailureOr<ProductLeafRange>
getProductLeafRange(Type type, ArrayRef<unsigned> fieldPath) {
  size_t offset = 0;
  for (unsigned field : fieldPath) {
    auto components = getProductComponents(type);
    if (!components || field >= components.size())
      return failure();
    for (unsigned preceding = 0; preceding < field; ++preceding)
      offset += leafCount(cast<TypeAttr>(components[preceding]).getValue());
    type = cast<TypeAttr>(components[field]).getValue();
  }
  return ProductLeafRange{offset, leafCount(type)};
}

SmallVector<ProductLeafRange> getProductLeafRanges(TypeRange roots) {
  SmallVector<ProductLeafRange> ranges;
  size_t offset = 0;
  for (Type root : roots) {
    size_t size = leafCount(root);
    ranges.push_back({offset, size});
    offset += size;
  }
  return ranges;
}

std::string getProductPathName(Type type, ArrayRef<unsigned> fieldPath) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  for (auto [depth, field] : llvm::enumerate(fieldPath)) {
    auto components = getProductComponents(type);
    assert(components && field < components.size() && "invalid product field path");
    if (depth)
      stream << '.';
    if (auto record = dyn_cast<RecordType>(type))
      stream << cast<StringAttr>(record.getFieldNames()[field]).getValue();
    else
      stream << field;
    type = cast<TypeAttr>(components[field]).getValue();
  }
  return result;
}

} // namespace intent
