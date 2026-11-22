#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/JSON.h"

namespace intentdsl {
// The owning operand is preserved when the logical map becomes a block or task.
inline int64_t domainSlot(mlir::Operation *anchor) {
  auto fn = mlir::dyn_cast<mlir::func::FuncOp>(anchor);
  if (!fn)
    fn = anchor->getParentOfType<mlir::func::FuncOp>();
  if (auto slot = fn->getAttrOfType<mlir::IntegerAttr>("kir.domain_arg"))
    return slot.getInt();
  auto &region = fn.front().front();
  if (region.getNumOperands())
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(region.getOperand(0)))
      if (arg.getOwner() == &fn.front())
        return arg.getArgNumber();
  return -1;
}

inline bool coefficients(mlir::AffineExpr expr,
                         llvm::MutableArrayRef<int64_t> out,
                         int64_t scale = 1) {
  if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    out.back() += scale * c.getValue();
    return true;
  }
  if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (d.getPosition() + 1 >= out.size())
      return false;
    out[d.getPosition()] += scale;
    return true;
  }
  auto b = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!b)
    return false;
  if (expr.getKind() == mlir::AffineExprKind::Add)
    return coefficients(b.getLHS(), out, scale) &&
           coefficients(b.getRHS(), out, scale);
  auto k = mlir::dyn_cast<mlir::AffineConstantExpr>(b.getRHS());
  return expr.getKind() == mlir::AffineExprKind::Mul && k &&
         coefficients(b.getLHS(), out, scale * k.getValue());
}
inline mlir::LogicalResult verifyDomain(mlir::func::FuncOp fn) {
  using namespace mlir;
  if (!fn || fn.empty() || fn.front().empty())
    return mlir::failure();
  auto slot = domainSlot(fn);
  if (slot < 0 || size_t(slot) >= fn.getNumArguments() ||
      !isa<MemRefType>(fn.getArgument(slot).getType()))
    return fn.emitError("logical domain requires a bound view");
  if (!fn->hasAttr("kir.iteration")) {
    if (fn->hasAttr("kir.public_abi"))
      return fn.emitError("indexed signature without an iteration domain");
    if (fn->hasAttr("kir.owner_axis")) {
      auto axis=fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
      auto view=cast<MemRefType>(fn.getArgument(slot).getType());
      if(!axis || axis.getInt()<0 || axis.getInt()>=view.getRank() || view.getRank()<2)
        return fn.emitError("owner axis must select one axis of a ranked member domain");
    }
    return success();
  }
  auto abi = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abi)
    return fn.emitError("missing projected signature");
  auto data = llvm::json::parse(abi.getValue());
  if (!data) {
    llvm::consumeError(data.takeError());
    return fn.emitError("invalid projected signature");
  }
  auto *args = data->getAsArray();
  if (!args || args->size() != fn.getNumArguments())
    return fn.emitError("invalid projected slots");

  auto bounds = fn->getAttrOfType<DenseI64ArrayAttr>("kir.iteration");
  auto publicABI = fn->getAttrOfType<StringAttr>("kir.public_abi");
  if (!bounds || bounds.empty() || bounds.size() % 4 || bounds.size() > 16 ||
      !publicABI)
    return fn.emitError("invalid indexed domain");
  auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  if(owner && (owner.getInt()<0 || size_t(owner.getInt())>=bounds.size()/4))
    return fn.emitError("owner axis outside projected domain");
  auto parsed = llvm::json::parse(publicABI.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return fn.emitError("invalid indexed signature");
  }
  auto *slots = parsed->getAsArray();
  if (!slots)
    return failure();
  auto validSlot = [&](int64_t i) {
    return i >= 0 && size_t(i) < slots->size() && (*slots)[i].getAsObject() &&
           (*slots)[i].getAsObject()->getArray("dims");
  };
  for (int i = 0; i < bounds.size(); i += 4) {
    if (bounds[i] == -1) {
      if (bounds[i + 3] < 0 || bounds[i + 1] != -1 || bounds[i + 2] != 1)
        return fn.emitError("invalid constant extent");
    } else if (!validSlot(bounds[i]) || bounds[i + 1] < 0 ||
               size_t(bounds[i + 1]) >= (*slots)[bounds[i]]
                                            .getAsObject()
                                            ->getArray("dims")
                                            ->size() ||
               bounds[i + 2] <= 0)
      return fn.emitError("invalid extent projection");
  }
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    auto source = fn.getArgAttrOfType<IntegerAttr>(i, "kir.source");
    auto layout = fn.getArgAttrOfType<AffineMapAttr>(i, "kir.indexing");
    auto view = dyn_cast<MemRefType>(fn.getArgument(i).getType());
    if (!source || !validSlot(source.getInt()))
      return fn.emitError("invalid projected binding");
    auto *pub = (*slots)[source.getInt()].getAsObject();
    auto *priv = (*args)[i].getAsObject();
    if (!priv)
      return fn.emitError("invalid projected slot descriptor");
    if (!view) {
      if (layout || pub->getString("direction") != "Scalar" || *priv != *pub)
        return fn.emitError("scalar capture must preserve the public parameter");
      continue;
    }
    bool owned=bool(fn.getArgAttr(i,"kir.owner_projection"));
    if (!layout || (owned ? (!owner || view.getRank()!=1 || !priv->getArray("dims") ||
                             priv->getArray("dims")->size()!=1)
                         : view.getRank()!=bounds.size()/4))
      return fn.emitError("projected view rank differs from the iteration domain");
    auto map = layout.getValue();
    if (pub->getString("dtype") != priv->getString("dtype") ||
        pub->getString("direction") != priv->getString("direction") ||
        map.getNumDims() != bounds.size()/4 || map.getNumSymbols() ||
        map.getNumResults() != pub->getArray("dims")->size())
      return fn.emitError("indexed view disagrees with its source");
    for (auto expr : map.getResults()) {
      SmallVector<int64_t> terms(map.getNumDims() + 1, 0);
      if (!coefficients(expr, terms))
        return fn.emitError("nonlinear indexed binding");
      if(owned)
        for(unsigned axis=0;axis<map.getNumDims();++axis)
          if(int64_t(axis)!=owner.getInt() && terms[axis]!=0)
            return fn.emitError("owner projection varies along a reduction axis");
    }
  }

  return success();
}
inline llvm::json::Object publishDomain(mlir::func::FuncOp fn) {
  auto slot = domainSlot(fn);
  auto rank =
      mlir::cast<mlir::MemRefType>(fn.getArgument(slot).getType()).getRank();
  llvm::json::Array outer;
  for (int64_t axis = 0; axis < rank - 1; ++axis)
    outer.push_back(axis);
  llvm::json::Object domain{
      {"slot", slot}, {"outer", std::move(outer)}, {"inner", rank - 1}};
  if (auto axis=fn->getAttrOfType<mlir::IntegerAttr>("kir.owner_axis")) {
    llvm::json::Array members;
    for(int i=0; i<rank; ++i) if(i!=axis.getInt()) members.push_back(i);
    domain["outer"]=llvm::json::Array{axis.getInt()};
    domain["inner"]=std::move(members);
  }
  if (auto bounds =
          fn->getAttrOfType<mlir::DenseI64ArrayAttr>("kir.iteration")) {
    llvm::json::Array extents, views;
    for (size_t i = 0; i < bounds.size(); i += 4)
      extents.emplace_back(llvm::json::Array{bounds[i], bounds[i + 1],
                                             bounds[i + 2], bounds[i + 3]});
    for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
      llvm::json::Array coords;
      if (!mlir::isa<mlir::MemRefType>(fn.getArgument(i).getType())) {
        views.emplace_back(llvm::json::Object{{"source",fn.getArgAttrOfType<mlir::IntegerAttr>(i,"kir.source").getInt()}});
        continue;
      }
      auto map = fn.getArgAttrOfType<mlir::AffineMapAttr>(i, "kir.indexing")
                     .getValue();
      for (auto expr : map.getResults()) {
        llvm::SmallVector<int64_t> terms(rank + 1, 0);
        coefficients(expr, terms);
        llvm::json::Array polynomial;
        for (auto v : terms)
          polynomial.emplace_back(v);
        coords.emplace_back(std::move(polynomial));
      }
      llvm::json::Object binding{
          {"source",
           fn.getArgAttrOfType<mlir::IntegerAttr>(i, "kir.source").getInt()},
          {"coordinates", std::move(coords)}};
      if(fn.getArgAttr(i,"kir.owner_projection"))
        binding["axis"]=fn->getAttrOfType<mlir::IntegerAttr>("kir.owner_axis").getInt();
      views.emplace_back(std::move(binding));
    }
    domain["extents"] = std::move(extents);
    domain["views"] = std::move(views);
    auto signature = llvm::json::parse(
        fn->getAttrOfType<mlir::StringAttr>("kir.public_abi").getValue());
    domain["public_abi"] = std::move(*signature);
  }
  return domain;
}
} // namespace intentdsl
