#ifndef INTENT_SERIALIZATION_SOURCE_H
#define INTENT_SERIALIZATION_SOURCE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>

namespace intent {

// Source state belongs to one translation. It neither infers missing values nor
// decides target semantics; operation handlers supply the verified spelling.
class SourceEmitter {
public:
  explicit SourceEmitter(llvm::raw_ostream &output, unsigned indentationWidth = 4);
  virtual ~SourceEmitter() = default;

  std::string valueString(mlir::Value value);
  void bind(mlir::Value value, llvm::StringRef name);
  void reserveName(llvm::StringRef name);
  std::string newName(llvm::StringRef prefix = "v");
  void line(const llvm::Twine &text, unsigned explicitIndent = ~0U);
  mlir::LogicalResult failedResult(mlir::Operation *operation,
                                   const llvm::Twine &message);
  bool hasFailed() const { return failed; }

  // Helpers/functions have independent SSA namespaces. A nested translation
  // restores its caller's bindings even when it reports a failure.
  class ScopedValues {
  public:
    explicit ScopedValues(SourceEmitter &emitter);
    ~ScopedValues();
    ScopedValues(const ScopedValues &) = delete;
    ScopedValues &operator=(const ScopedValues &) = delete;

  private:
    SourceEmitter &emitter;
    llvm::DenseMap<mlir::Value, std::string> saved;
    llvm::StringSet<> savedReservedNames;
    unsigned savedCounter;
  };

protected:
  llvm::raw_ostream &output;
  llvm::DenseMap<mlir::Value, std::string> values;
  unsigned indent = 0;
  unsigned counter = 0;
  uint64_t emittedLines = 0;
  bool failed = false;

private:
  llvm::StringSet<> reservedNames;
  unsigned indentationWidth;
};

// One closed surface table serves both terminal legality and actual emission.
// Registration uses operation classes; names only identify registered MLIR ops.
template <typename Context> class OperationEmitters {
public:
  using Dependencies = llvm::SmallVector<std::string>;

  template <typename Op, typename Check, typename Emit>
  void add(Check check, Emit emit,
           std::function<mlir::FailureOr<Dependencies>(Op)> dependencies = {}) {
    auto inserted = handlers.try_emplace(Op::getOperationName(),
        handler<Op>(std::move(check), std::move(emit), std::move(dependencies)));
    assert(inserted.second && "source operation was registered twice");
    (void)inserted;
  }

  template <typename Op, typename Check, typename Emit>
  void replace(Check check, Emit emit,
               std::function<mlir::FailureOr<Dependencies>(Op)> dependencies = {}) {
    auto found = handlers.find(Op::getOperationName());
    assert(found != handlers.end() && "cannot replace an unregistered source operation");
    found->second = handler<Op>(std::move(check), std::move(emit), std::move(dependencies));
  }

  bool contains(mlir::Operation *operation) const {
    return handlers.count(operation->getName().getStringRef());
  }

  mlir::LogicalResult verify(mlir::Operation *operation) const {
    auto found = handlers.find(operation->getName().getStringRef());
    if (found == handlers.end())
      return operation->emitOpError("has no registered source translation");
    return found->second.check(operation);
  }

  mlir::LogicalResult emit(mlir::Operation *operation, Context &context) const {
    auto found = handlers.find(operation->getName().getStringRef());
    if (found == handlers.end())
      return operation->emitOpError("has no registered source translation");
    if (mlir::failed(found->second.check(operation)))
      return mlir::failure();
    return found->second.emit(operation, context);
  }

  // Dependencies are terminal source spellings owned by the same handler as
  // legality and emission. Recompute them from the current program; no import
  // inventory or execution decision survives an IR rewrite.
  mlir::FailureOr<Dependencies> collectDependencies(mlir::Operation *root) const {
    std::set<std::string> required;
    auto walk = root->walk([&](mlir::Operation *operation) {
      auto found = handlers.find(operation->getName().getStringRef());
      if (found == handlers.end()) {
        operation->emitOpError("has no registered source translation");
        return mlir::WalkResult::interrupt();
      }
      auto dependencies = found->second.dependencies(operation);
      if (mlir::failed(dependencies)) return mlir::WalkResult::interrupt();
      required.insert(dependencies->begin(), dependencies->end());
      return mlir::WalkResult::advance();
    });
    if (walk.wasInterrupted()) return mlir::failure();
    return Dependencies(required.begin(), required.end());
  }

private:
  struct Handler {
    std::function<mlir::LogicalResult(mlir::Operation *)> check;
    std::function<mlir::LogicalResult(mlir::Operation *, Context &)> emit;
    std::function<mlir::FailureOr<Dependencies>(mlir::Operation *)> dependencies;
  };

  template <typename Op, typename Check, typename Emit>
  static Handler handler(Check check, Emit emit,
      std::function<mlir::FailureOr<Dependencies>(Op)> dependencies) {
    return {
        [check = std::move(check)](mlir::Operation *operation) {
          return check(mlir::cast<Op>(operation));
        },
        [emit = std::move(emit)](mlir::Operation *operation, Context &context) {
          return emit(mlir::cast<Op>(operation), context);
        },
        [dependencies = std::move(dependencies)](mlir::Operation *operation)
            -> mlir::FailureOr<Dependencies> {
          return dependencies ? dependencies(mlir::cast<Op>(operation))
                              : mlir::FailureOr<Dependencies>(Dependencies{});
        }};
  }

  llvm::StringMap<Handler> handlers;
};

} // namespace intent
#endif
