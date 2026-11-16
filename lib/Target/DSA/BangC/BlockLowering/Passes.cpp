#include "Intent/Target/DSA/BangC/Physical/IR/DSAOps.h"
#include "Intent/Target/DSA/BangC/ConstructionPasses.h"
#include "Intent/Target/DSA/BangC/Compilation.h"
#include "Intent/Target/DSA/BangC/BlockLowering.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
using namespace mlir;
namespace intent::bangc {
#define GEN_PASS_DEF_CONSTRUCTBANGC
#include "Intent/Target/DSA/BangC/Construction.h.inc"
namespace {
class ConstructBangCPass : public impl::ConstructBangCBase<ConstructBangCPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    auto configurations = compiler::readDSAConfigurations(module, directory.getValue(), overrides.getValue());
    auto hardware = dyn_cast_or_null<physical::CapabilitiesAttr>(parseAttribute(capabilities, &getContext()));
    auto shapeBindings = dyn_cast_or_null<DictionaryAttr>(parseAttribute(shapes, &getContext()));
    auto strideBindings = dyn_cast_or_null<DictionaryAttr>(parseAttribute(strides, &getContext()));
    if (failed(configurations) || !hardware || !shapeBindings || !strideBindings)
      return signalPassFailure();
    SmallVector<OwningOpRef<ModuleOp>> candidates;
    StringAttr entryName;
    for (auto [index, attribute] : llvm::enumerate(*configurations)) {
      OwningOpRef<ModuleOp> candidate(cast<ModuleOp>(module->clone()));
      if (failed(constructBlockStorage(*candidate, cast<physical::ConfigurationAttr>(attribute), hardware,
                                       shapeBindings, strideBindings)))
        return signalPassFailure();
      (*candidate)->setAttr("intent_bangc.capabilities", hardware);
      func::FuncOp entry;
      for (auto function : candidate->getOps<func::FuncOp>())
        if (!function.isExternal()) { entry = function; break; }
      if (!entry) {
        module.emitError("BANG C configuration has no executable entry");
        return signalPassFailure();
      }
      if (!entryName) entryName = StringAttr::get(module.getContext(), entry.getName());
      SymbolTable symbols(*candidate);
      if (failed(symbols.rename(entry, entryName.getValue().str() + "_config_" + std::to_string(index))))
        return signalPassFailure();
      (*candidate)->setAttr(SymbolTable::getSymbolAttrName(),
                           StringAttr::get(module.getContext(), "config_" + std::to_string(index)));
      candidates.push_back(std::move(candidate));
    }
    module.getBody()->clear();
    module->setAttr("intent.entry_name", entryName);
    for (auto &candidate : candidates) module.getBody()->push_back(candidate.release());
  }
};
}
#define GEN_PASS_REGISTRATION
#include "Intent/Target/DSA/BangC/Construction.h.inc"
void registerConstructionPasses() { registerBangCConstructionPasses(); }
}
