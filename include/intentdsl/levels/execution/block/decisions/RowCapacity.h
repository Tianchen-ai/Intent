#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace intentdsl {
struct RowCapacity {
  llvm::json::Array bindings;
  llvm::Error bind(const llvm::json::Object &options) {
    const auto *facts=options.getArray("arguments");
    if(!facts)return llvm::createStringError("missing per-argument shape/stride facts");
    bindings=*facts;
    return llvm::Error::success();
  }
  llvm::Expected<int64_t> width(mlir::Operation *op) const {
    auto fn=mlir::dyn_cast<mlir::func::FuncOp>(op);if(!fn)fn=op->getParentOfType<mlir::func::FuncOp>();
    if(auto bounds=fn->getAttrOfType<mlir::DenseI64ArrayAttr>("kir.iteration")){
      if(bounds.empty() || bounds.size()%4)return llvm::createStringError("invalid logical bounds");
      auto inner=bounds.asArrayRef().take_back(4);
      if(inner[0]<0)return inner[3];
      if(size_t(inner[0])>=bindings.size())return llvm::createStringError("absent domain binding");
      auto *v=bindings[inner[0]].getAsObject();auto *shape=v?v->getArray("shape"):nullptr;
      if(!shape || inner[1]<0 || size_t(inner[1])>=shape->size() || inner[2]<=0)return llvm::createStringError("invalid domain axis");
      auto extent=(*shape)[inner[1]].getAsInteger();
      if(!extent || *extent<0 || *extent%inner[2])return llvm::createStringError("inexact domain extent");
      return *extent/inner[2];
    }
    auto slot=domainSlot(op);
    if(slot<0 || size_t(slot)>=bindings.size())return llvm::createStringError("logical map refers to an absent binding");
    auto *view=bindings[slot].getAsObject();
    auto *dims=view?view->getArray("shape"):nullptr;
    if(!dims || dims->empty() || !dims->back().getAsInteger() || *dims->back().getAsInteger()<0)
      return llvm::createStringError("logical inner axis has no bound extent");
    return *dims->back().getAsInteger();
  }
  // Rectangular tile consumers cannot span a carry between two projected
  // outer axes in one transfer. Other consumers linearize lanes separately.
  bool rectangularProjectedRows = false;
  llvm::SmallVector<unsigned> independentWidths;
  bool acceptsGroup(mlir::Operation *op, unsigned count) const {
    if (count == 1) return true;
    auto fn = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!fn) fn = op->getParentOfType<mlir::func::FuncOp>();
    auto bounds = fn->getAttrOfType<mlir::DenseI64ArrayAttr>("kir.iteration");
    if (!bounds || bounds.size() <= 8 || !rectangularProjectedRows) return true;
    auto axis = bounds.asArrayRef().take_back(8).take_front(4);
    if(axis[0]<0) return axis[3]%count == 0;
    if(size_t(axis[0])>=bindings.size() || axis[2]<=0) return false;
    auto *record=bindings[axis[0]].getAsObject();
    auto *shape=record?record->getArray("shape"):nullptr;
    if(!shape || size_t(axis[1])>=shape->size()) return false;
    auto dim=(*shape)[axis[1]].getAsInteger();
    return dim && (*dim/axis[2])%count==0;
  }
  struct Row {
    unsigned id, rows, width;
    std::string payload;
  };
  llvm::SmallVector<Row> rows;
  llvm::SmallVector<unsigned> streamingWidths;
};
} // namespace intentdsl
