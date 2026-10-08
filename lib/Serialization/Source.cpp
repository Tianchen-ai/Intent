#include "Intent/Serialization/Source.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent {

SourceEmitter::SourceEmitter(llvm::raw_ostream &output, unsigned indentationWidth)
    : output(output), indentationWidth(indentationWidth) {}

std::string SourceEmitter::valueString(Value value) {
  auto found = values.find(value);
  if (found != values.end())
    return found->second;
  emitError(value.getLoc(), "source translation encountered an unmapped SSA value");
  failed = true;
  return {};
}

void SourceEmitter::bind(Value value, llvm::StringRef name) {
  values[value] = name.str();
  reserveName(name);
}

void SourceEmitter::reserveName(llvm::StringRef name) {
  reservedNames.insert(name);
}

std::string SourceEmitter::newName(llvm::StringRef prefix) {
  std::string name;
  do {
    name = prefix.str() + std::to_string(counter++);
  } while (reservedNames.contains(name) || llvm::any_of(values, [&](const auto &entry) {
    return entry.second == name;
  }));
  reserveName(name);
  return name;
}

void SourceEmitter::line(const llvm::Twine &text, unsigned explicitIndent) {
  unsigned level = explicitIndent == ~0U ? indent : explicitIndent;
  output.indent(level * indentationWidth) << text << '\n';
  ++emittedLines;
}

LogicalResult SourceEmitter::failedResult(Operation *operation,
                                         const llvm::Twine &message) {
  operation->emitOpError() << message;
  failed = true;
  return failure();
}

SourceEmitter::ScopedValues::ScopedValues(SourceEmitter &emitter)
    : emitter(emitter), savedCounter(emitter.counter) {
  saved.swap(emitter.values);
  savedReservedNames.swap(emitter.reservedNames);
  emitter.counter = 0;
}

SourceEmitter::ScopedValues::~ScopedValues() {
  saved.swap(emitter.values);
  savedReservedNames.swap(emitter.reservedNames);
  emitter.counter = savedCounter;
}

} // namespace intent
