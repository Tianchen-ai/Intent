#pragma once
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include <cmath>
namespace intentdsl {
// Effective input facts for the current JIT specialization.
struct CallFacts {
  llvm::json::Array arguments;
  std::string localEstimator;
  int64_t participants;
  std::pair<double, double> geometry(mlir::Operation *op) const {
    auto fn = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!fn)
      fn = op->getParentOfType<mlir::func::FuncOp>();
    auto extent = [&](int64_t slot, int64_t axis) -> double {
      if (slot < 0 || size_t(slot) >= arguments.size())
        return 0;
      auto *arg = arguments[slot].getAsObject();
      auto *shape = arg ? arg->getArray("shape") : nullptr;
      if (!shape || axis < 0 || size_t(axis) >= shape->size())
        return 0;
      return (*shape)[axis].getAsInteger().value_or(0);
    };
    std::vector<double> shape;
    if (auto bounds =
            fn->getAttrOfType<mlir::DenseI64ArrayAttr>("kir.iteration")) {
      for (unsigned i = 0; i + 3 < bounds.size(); i += 4)
        shape.push_back(bounds[i] < 0
                            ? bounds[i + 3]
                            : extent(bounds[i], bounds[i + 1]) / bounds[i + 2]);
    } else {
      int64_t slot = domainSlot(fn);
      if (slot < 0 || size_t(slot) >= arguments.size())
        return {0, 0};
      auto *arg = arguments[slot].getAsObject();
      auto *dims = arg ? arg->getArray("shape") : nullptr;
      if (dims)
        for (auto &d : *dims)
          shape.push_back(d.getAsInteger().value_or(0));
    }
    if (shape.empty())
      return {0, 0};
    double outer = 1;
    for (unsigned i = 0; i + 1 < shape.size(); ++i)
      outer *= shape[i];
    return {outer, shape.back()};
  }
};
inline llvm::Error bindPerformance(compilerinfra::TargetProfile &p,
                                   const llvm::json::Object &options,
                                   std::string name, std::string calibration,
                                   int64_t participants) {
  const auto *arguments = options.getArray("arguments");
  if (!arguments)
    return llvm::createStringError("missing bound argument facts");
  p.services.calibrationProfile = calibration;
  if (auto err = p.services.facts.query(
          "participants", {participants, "count", "runtime",
                           llvm::json::Object{{"source", "native discovery"}}}))
    return err;
  return p.services.provide<CallFacts>(std::make_shared<CallFacts>(
      CallFacts{*arguments, std::move(name), participants}));
}
inline std::optional<double> quantity(compilerinfra::CostContext &ctx,
                                      llvm::StringRef name,
                                      llvm::StringRef unit) {
  auto *fact = ctx.fact(name, unit);
  return fact ? fact->value.getAsNumber() : std::nullopt;
}
} // namespace intentdsl
