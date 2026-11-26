#pragma once
#include "compilerinfra/Contract.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace intentdsl::kir {

struct FastMathPolicy {
  bool relaxedNormalization = false;
  bool explicitFastMath = false;
  mlir::arith::FastMathFlags defaultFlags = mlir::arith::FastMathFlags::none;
  std::string denormalFpMath = "ieee";
};

inline bool validDenormalMode(llvm::StringRef mode) {
  if (mode.empty() || mode.starts_with(",") || mode.ends_with(","))
    return false;
  return llvm::parseDenormalFPAttribute(mode).isValid();
}

inline std::string canonicalDenormalName(llvm::DenormalMode mode) {
  if (mode == llvm::DenormalMode::getIEEE())
    return "ieee";
  if (mode == llvm::DenormalMode::getPreserveSign())
    return "preserve-sign";
  if (mode == llvm::DenormalMode::getPositiveZero())
    return "positive-zero";
  if (mode == llvm::DenormalMode::getDynamic())
    return "dynamic";
  return mode.str();
}

inline llvm::StringRef getDenormalFpMath(mlir::Operation *op) {
  if (!op)
    return "ieee";
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("denormal-fp-math"))
    return attr.getValue();
  if (auto fn = mlir::dyn_cast<mlir::func::FuncOp>(op))
    if (auto attr = fn->getAttrOfType<mlir::StringAttr>("denormal-fp-math"))
      return attr.getValue();
  if (auto fn = op->getParentOfType<mlir::func::FuncOp>())
    if (auto attr = fn->getAttrOfType<mlir::StringAttr>("denormal-fp-math"))
      return attr.getValue();
  return "ieee";
}

inline bool allowsSubnormalFtz(mlir::Operation *op) {
  return compilerinfra::allowsSubnormalFlush(op);
}

inline mlir::arith::FastMathFlags getFastMathFlags(mlir::Operation *op) {
  if (!op)
    return mlir::arith::FastMathFlags::none;
  if (auto attr =
          op->getAttrOfType<mlir::arith::FastMathFlagsAttr>("fastmath"))
    return attr.getValue();
  return mlir::arith::FastMathFlags::none;
}

inline void setFastMathFlags(mlir::Operation *op,
                             mlir::arith::FastMathFlags flags) {
  if (!op)
    return;
  if (flags == mlir::arith::FastMathFlags::none && !op->getAttr("fastmath"))
    return;
  op->getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
  op->setAttr("fastmath",
              mlir::arith::FastMathFlagsAttr::get(op->getContext(), flags));
}

inline void addFastMathFlags(mlir::Operation *op,
                             mlir::arith::FastMathFlags flags) {
  if (op && flags != mlir::arith::FastMathFlags::none)
    setFastMathFlags(op, getFastMathFlags(op) | flags);
}

inline bool hasFastMath(mlir::Operation *op,
                        mlir::arith::FastMathFlags required) {
  return mlir::arith::bitEnumContainsAll(getFastMathFlags(op), required);
}

inline bool hasAnyFastMath(mlir::Operation *op,
                           mlir::arith::FastMathFlags mask) {
  return (getFastMathFlags(op) & mask) != mlir::arith::FastMathFlags::none;
}

// NVIDIA .ftz preserves the sign of a flushed operand and result.
inline bool allowsSignedFtz(mlir::Operation *op) {
  auto mode = compilerinfra::effectiveDenormalMode(op);
  bool ignoreSign = compilerinfra::hasFastMath(op, mlir::arith::FastMathFlags::nsz);
  auto compatible = [=](auto kind) {
    return kind == llvm::DenormalMode::PreserveSign ||
           (kind == llvm::DenormalMode::PositiveZero && ignoreSign);
  };
  return compatible(mode.Input) && compatible(mode.Output);
}

inline bool isFloatingMathOp(mlir::Operation *op) {
  if (!op || op->getNumResults() != 1)
    return false;
  auto elem = mlir::getElementTypeOrSelf(op->getResult(0).getType());
  if (!mlir::isa<mlir::FloatType>(elem))
    return false;
  auto ns = op->getName().getDialectNamespace();
  auto code = op->getName().stripDialect();
  if (ns == "arith")
    return llvm::is_contained(
        llvm::ArrayRef<llvm::StringRef>{"addf", "subf", "mulf", "divf",
                                        "maxnumf", "minnumf", "minimumf"},
        code);
  if (ns == "math")
    return llvm::is_contained(
        llvm::ArrayRef<llvm::StringRef>{"exp", "exp2", "log", "sqrt", "rsqrt",
                                        "absf"},
        code);
  if (ns == "kir")
    return llvm::is_contained(
        llvm::ArrayRef<llvm::StringRef>{
            "add", "sub", "mul", "div", "exp", "exp2", "rsqrt", "sqrt",
            "maximum", "sum", "max_num", "lane_sum", "ordered_sum", "contract",
            "scaled_contract", "window_contract", "q4k_dot", "quant_contract"},
        code);
  return false;
}

inline llvm::Expected<FastMathPolicy>
parseFastMathPolicy(const llvm::json::Object &options) {
  using FMF = mlir::arith::FastMathFlags;
  FastMathPolicy policy;
  policy.explicitFastMath = options.get("fpmath_mode") || options.get("fast_math");
  if (options.get("numerics")) {
    auto mode = options.getString("numerics");
    if (!mode || (*mode != "source" && *mode != "relaxed_normalization"))
      return llvm::createStringError("numerics must be source or relaxed_normalization");
    policy.relaxedNormalization = *mode == "relaxed_normalization";
  }
  if (options.get("float32_matmul_precision")) {
    auto precision = options.getString("float32_matmul_precision");
    if (precision != "highest" && precision != "high")
      return llvm::createStringError("float32_matmul_precision must be highest or high");
  }
  if (auto mode = options.getString("fpmath_mode")) {
    if (*mode == "strict") {
      policy.defaultFlags = FMF::none;
    } else if (*mode == "relaxed") {
      policy.defaultFlags = FMF::reassoc | FMF::contract | FMF::afn;
    } else if (*mode == "bf16") {
      policy.defaultFlags = FMF::contract;
      policy.denormalFpMath = "preserve-sign";
    } else if (*mode == "fast") {
      policy.defaultFlags = FMF::fast;
      policy.denormalFpMath = "preserve-sign";
    } else {
      return llvm::createStringError("invalid fpmath_mode '" + mode->str() +
                                     "'");
    }
  }
  if (auto fast = options.getBoolean("fast_math"); fast && *fast) {
    policy.defaultFlags = policy.defaultFlags | FMF::fast;
    if (policy.denormalFpMath == "ieee")
      policy.denormalFpMath = "preserve-sign";
  }
  if (auto denorm = options.getString("denormal_fp_math")) {
    if (!validDenormalMode(*denorm))
      return llvm::createStringError("invalid denormal_fp_math '" +
                                     denorm->str() + "'");
    policy.denormalFpMath = denorm->str();
  }
  return policy;
}

} // namespace intentdsl::kir
