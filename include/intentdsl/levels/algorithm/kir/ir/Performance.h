#pragma once
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include <cmath>
namespace intentdsl {
// Effective input facts for the current JIT specialization.
struct CallFacts {
  llvm::json::Array arguments;
  std::string localEstimator;
  int64_t participants;
  const llvm::json::Array *shapeOf(mlir::Value v) const {
    auto arg=mlir::dyn_cast<mlir::BlockArgument>(v);
    if(!arg || arg.getArgNumber()>=arguments.size())return nullptr;
    auto *desc=arguments[arg.getArgNumber()].getAsObject();
    return desc?desc->getArray("shape"):nullptr;
  }
  double viewBytes(mlir::Value v) const {
    auto *shape=shapeOf(v);
    if(!shape)return 0;
    double bytes=mlir::cast<mlir::MemRefType>(v.getType()).getElementTypeBitWidth()/8.;
    for(auto &extent:*shape)bytes*=extent.getAsInteger().value_or(0);
    return bytes;
  }
  llvm::Expected<double> axisStride(mlir::Operation *op, unsigned slot, unsigned axis) const {
    auto fn=op->getParentOfType<mlir::func::FuncOp>();
    auto map=fn.getArgAttrOfType<mlir::AffineMapAttr>(slot,"kir.indexing");
    if(auto src=fn.getArgAttrOfType<mlir::IntegerAttr>(slot,"kir.source"))slot=src.getInt();
    if(slot>=arguments.size())return llvm::createStringError("missing bound input for access estimate");
    auto *record=arguments[slot].getAsObject();
    auto *strides=record?record->getArray("strides"):nullptr;
    if(!strides)return llvm::createStringError("missing physical strides");
    if(!map){
      if(axis>=strides->size() || !(*strides)[axis].getAsInteger())return llvm::createStringError("invalid inner stride");
      return double(*(*strides)[axis].getAsInteger());
    }
    auto affine=map.getValue();
    if(axis>=affine.getNumDims() || strides->size()!=affine.getNumResults())return llvm::createStringError("invalid projected rank");
    double stride=0;
    for(auto [i,expr]:llvm::enumerate(affine.getResults())){
      llvm::SmallVector<int64_t> terms(affine.getNumDims()+1,0);
      auto physical=(*strides)[i].getAsInteger();
      if(!physical || !coefficients(expr,terms))return llvm::createStringError("invalid affine access estimate");
      stride+=double(terms[axis])* *physical;
    }
    return stride;
  }
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
    if(auto a=fn->getAttrOfType<mlir::IntegerAttr>("kir.owner_axis")){
      double n=1;
      for(auto [i,v]:llvm::enumerate(shape))if(int64_t(i)!=a.getInt())n*=v;
      return {shape[a.getInt()],n};
    }
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
  auto fmPolicy = kir::parseFastMathPolicy(options);
  if (!fmPolicy)
    return fmPolicy.takeError();
  if (auto err = p.services.provide<kir::FastMathPolicy>(
          std::make_shared<kir::FastMathPolicy>(std::move(*fmPolicy))))
    return err;
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
