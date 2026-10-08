#ifndef INTENT_ANALYSIS_INTEGERRELATIONS_H
#define INTENT_ANALYSIS_INTEGERRELATIONS_H

#include "Intent/Analysis/UniformValues.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <functional>
#include <utility>

namespace intent {

using IntegerDifference = std::optional<int64_t>;
using IntegerDifferenceQuery = llvm::function_ref<IntegerDifference(mlir::Value)>;

// Fold one scalar integer operation's variation coefficient from facts supplied
// by its consumer. The supported value domain is i64 or an explicitly bound
// 64-bit index. Arithmetic values retain their modulo semantics; coefficient
// arithmetic itself is checked and must fit int64_t. Unknown is never zero.
// This query proves neither value bounds nor control/memory independence.
IntegerDifference foldIntegerDifference(
    const UniformExpression &expression, IntegerDifferenceQuery difference,
    llvm::function_ref<std::optional<int64_t>(mlir::Value)> constant,
    unsigned indexBitWidth);

// Adapters describe existing typed SSA values or attributes. This view never
// constructs an arithmetic program and does not assign facts to unknown nodes.
enum class IntegerOrderKind {
  Add, Subtract, Multiply, FloorDivide, Minimum, Maximum
};

template <typename Node> struct IntegerOrderBinary {
  IntegerOrderKind kind;
  Node lhs;
  Node rhs;
};

template <typename Node> struct IntegerRoundDown {
  Node dividend;
  Node divisor;
};

template <typename Node> struct IntegerOrderCallbacks {
  // Zero means unsupported; comparisons require the same signed width, at most
  // 64 bits. Unsigned ordering and a cast that changes the interpreted integer
  // must not be described as a forwarding signed node.
  std::function<unsigned(Node)> signedWidth;
  std::function<std::optional<IntegerOrderBinary<Node>>(Node)> binary;
  std::function<std::optional<int64_t>(Node)> constant;
  std::function<bool(Node, Node)> same;
  std::function<bool(Node)> nonnegative;
  std::function<bool(Node)> positive;
  // SSA modular arithmetic needs an actual no-overflow proof. Checked host
  // arithmetic may supply its defined-input contract; it must not silently
  // bestow that contract on an ordinary SSA add.
  std::function<bool(Node)> noSignedWrap;
  // Additional current-domain facts, e.g. independent integer intervals.
  // `strict` asks for <; false asks for <=. Unknown must return false.
  std::function<bool(Node, Node, bool strict)> knownOrder;
};

// One signed-order rule set for adapters with different current-IR carriers.
// Scope/ownership, configuration selection and coordinate membership remain
// adapter facts. Recreate adapters when their underlying IR snapshot changes.
template <typename Node> class IntegerOrder {
public:
  explicit IntegerOrder(IntegerOrderCallbacks<Node> callbacks)
      : callbacks(std::move(callbacks)) {}

  bool atMost(Node lhs, Node rhs) const { return compare(lhs, rhs, false, 0); }
  bool lessThan(Node lhs, Node rhs) const { return compare(lhs, rhs, true, 0); }

  // For x >= 0 and s > 0, floor(x/s)*s is aligned and lies in [0,x].
  // The product therefore fits the signed type even when independent ranges
  // for the quotient and divisor cannot establish that correlation.
  std::optional<IntegerRoundDown<Node>> roundDown(Node value) const {
    auto product = binary(value);
    if (!product || product->kind != IntegerOrderKind::Multiply)
      return std::nullopt;
    for (unsigned side = 0; side < 2; ++side) {
      Node quotient = side ? product->rhs : product->lhs;
      Node divisor = side ? product->lhs : product->rhs;
      auto division = binary(quotient);
      if (division && division->kind == IntegerOrderKind::FloorDivide &&
          same(division->rhs, divisor) && positive(divisor) &&
          nonnegative(division->lhs))
        return IntegerRoundDown<Node>{division->lhs, divisor};
    }
    return std::nullopt;
  }

private:
  unsigned width(Node value) const {
    unsigned result = callbacks.signedWidth ? callbacks.signedWidth(value) : 0;
    return result <= 64 ? result : 0;
  }

  bool same(Node lhs, Node rhs) const {
    return lhs == rhs || (callbacks.same && callbacks.same(lhs, rhs));
  }

  std::optional<IntegerOrderBinary<Node>> binary(Node value) const {
    unsigned bits = width(value);
    if (!bits || !callbacks.binary) return std::nullopt;
    auto result = callbacks.binary(value);
    if (!result || width(result->lhs) != bits || width(result->rhs) != bits)
      return std::nullopt;
    return result;
  }

  std::optional<int64_t> constant(Node value) const {
    return callbacks.constant ? callbacks.constant(value) : std::nullopt;
  }

  bool nonnegative(Node value) const {
    if (auto number = constant(value)) return *number >= 0;
    return callbacks.nonnegative && callbacks.nonnegative(value);
  }

  bool positive(Node value) const {
    if (auto number = constant(value)) return *number > 0;
    return callbacks.positive && callbacks.positive(value);
  }

  bool compare(Node lhs, Node rhs, bool strict, unsigned depth) const {
    unsigned bits = width(lhs);
    if (!bits || width(rhs) != bits || depth >= 32) return false;
    if (same(lhs, rhs)) return !strict;
    auto left = constant(lhs), right = constant(rhs);
    if (left && right) return strict ? *left < *right : *left <= *right;
    if (left && *left == 0 && (strict ? positive(rhs) : nonnegative(rhs)))
      return true;
    if (callbacks.knownOrder && callbacks.knownOrder(lhs, rhs, strict))
      return true;
    auto prove = [&](Node a, Node b, bool less) {
      return compare(a, b, less, depth + 1);
    };
    if (auto operation = binary(lhs)) {
      switch (operation->kind) {
      case IntegerOrderKind::Minimum:
        return prove(operation->lhs, rhs, strict) ||
               prove(operation->rhs, rhs, strict);
      case IntegerOrderKind::Maximum:
        return prove(operation->lhs, rhs, strict) &&
               prove(operation->rhs, rhs, strict);
      case IntegerOrderKind::Subtract:
        // The difference of two nonnegative signed values cannot overflow.
        if (nonnegative(operation->rhs) &&
            (nonnegative(operation->lhs) ||
             (callbacks.noSignedWrap && callbacks.noSignedWrap(lhs))) &&
            (prove(operation->lhs, rhs, strict) ||
             (strict && positive(operation->rhs) &&
              prove(operation->lhs, rhs, false))))
          return true;
        break;
      default:
        break;
      }
      if (auto rounded = roundDown(lhs))
        if (prove(rounded->dividend, rhs, strict)) return true;
    }
    if (auto operation = binary(rhs)) {
      switch (operation->kind) {
      case IntegerOrderKind::Maximum:
        return prove(lhs, operation->lhs, strict) ||
               prove(lhs, operation->rhs, strict);
      case IntegerOrderKind::Minimum:
        return prove(lhs, operation->lhs, strict) &&
               prove(lhs, operation->rhs, strict);
      case IntegerOrderKind::Add:
        if (callbacks.noSignedWrap && callbacks.noSignedWrap(rhs)) {
          for (unsigned side = 0; side < 2; ++side) {
            Node base = side ? operation->rhs : operation->lhs;
            Node increment = side ? operation->lhs : operation->rhs;
            if ((nonnegative(increment) && prove(lhs, base, strict)) ||
                (strict && positive(increment) && prove(lhs, base, false)))
              return true;
          }
        }
        break;
      default:
        break;
      }
    }
    return false;
  }

  IntegerOrderCallbacks<Node> callbacks;
};

} // namespace intent
#endif
