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
  double keyTraffic(mlir::Operation *root, llvm::StringRef partition="count_keys",
                    llvm::StringRef merge="column_sum") const {
    double bytes=0;
    root->walk([&](mlir::Operation *op){
      auto n=op->getName().stripDialect();
      if(n!=partition && n!=merge)return;
      auto source=mlir::dyn_cast<mlir::BlockArgument>(op->getOperand(0));
      if(!source || source.getArgNumber()>=arguments.size())return;
      auto *view=arguments[source.getArgNumber()].getAsObject();auto *shape=view?view->getArray("shape"):nullptr;
      if(!shape)return;
      double elements=1;
      for(auto &d:*shape)elements*=d.getAsInteger().value_or(0);
      bytes+=elements*mlir::cast<mlir::MemRefType>(source.getType()).getElementTypeBitWidth()/8;
    });
    return bytes;
  }
  // Estimate total visits from a packed offsets domain. A subrange of the
  // packed input may be shorter. This estimate is never a legality bound.
  double segmentReads(mlir::Operation *program, llvm::StringRef memberRegion="ordered_sum") const {
    double bytes=0;
    program->walk([&](mlir::Operation *fold){
      auto name=fold->getName().stripDialect();
      if(name!=memberRegion)return;
      auto &bb=fold->getRegion(0).front();
      double count=0, featureCount=1, bytesPerTerm=0;
      for(auto &read:bb){
        if(read.getName().stripDialect()!="member_load")continue;
        auto source=mlir::dyn_cast<mlir::BlockArgument>(read.getOperand(0));
        if(!source || source.getArgNumber()>=arguments.size())continue;
        auto type=mlir::cast<mlir::MemRefType>(source.getType());
        bytesPerTerm+=type.getElementTypeBitWidth()/8.;
        if(read.getOperand(1)!=bb.getArgument(0))continue;
        auto *record=arguments[source.getArgNumber()].getAsObject();auto *shape=record?record->getArray("shape"):nullptr;
        if(!shape || shape->empty())continue;
        count=std::max(count,double(shape->front().getAsInteger().value_or(0)));
        if(read.getNumOperands()==3 && read.getOperand(2)==bb.getArgument(1) && shape->size()==2)
          featureCount=std::max(featureCount,double(shape->back().getAsInteger().value_or(0)));
      }
      bytes+=count*featureCount*bytesPerTerm;
    });
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
