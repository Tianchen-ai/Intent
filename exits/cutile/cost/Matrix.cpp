#include "../capabilities/Index.h"
#include "intentdsl/levels/execution/block/ir/CollectiveIssue.h"
#include "intentdsl/levels/execution/block/ir/MatrixIssue.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
namespace CI_UNIT_NAMESPACE {
void matrixIssues(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {intentdsl::block::collectiveSteps(), COMPILERINFRA_SOURCE});
  u.costModels.push_back(
      {{"matrix-issues", "levels/execution/block::contraction", "1",
        [](mlir::Operation *o, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx) {
          auto caps =
              ctx.target.services.get<intentdsl::cutile::ArrayIndices>();
          auto estimate = intentdsl::block::matrixIssues(
              o, c, ctx, caps->scaledMma ? 16 : 64, 8, 16);
          auto machine = ctx.target.services.get<intentdsl::block::OwnerMachine>();
          if (!machine || machine->residentRegisters <= 0)
            return estimate;
          for (auto &dot : o->getRegion(0).front()) {
            if (dot.getName().stripDialect() != "contract" || dot.getNumOperands() != 4)
              continue;
            auto tile = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
            double m = tile[0], n = tile[1], k = tile[2];
            // Indirect rows form a masked gather panel. Its 64-bit addresses
            // and gathered values coexist with the accumulator, while
            // the contiguous other operand is transferred as a matrix tile.
            auto regWords = [&](unsigned i) {
              return mlir::cast<mlir::MemRefType>(dot.getOperand(i).getType())
                         .getElementTypeBitWidth() / 32.;
            };
            double accumWords = mlir::cast<mlir::ShapedType>(dot.getResult(0).getType())
                                    .getElementTypeBitWidth() / 32.;
            double live = m * n * accumWords +
                          m * k * (2. + regWords(0)) + n * k * regWords(1);
            double spill = 2. * sizeof(uint32_t) *
                           std::max(0., live - machine->residentRegisters);
            estimate.terms["indirect_panel_live_words"] = {live, "reg32"};
            estimate.terms["indirect_panel_spill_bytes"] = {spill, "B"};
            auto bw = intentdsl::quantity(ctx, "read_1_shared_bw", "B/ns");
            if (bw && spill > 0) {
              auto facts = ctx.target.services.get<intentdsl::CallFacts>();
              estimate.timeNs = estimate.timeNs.value_or(0.) +
                                spill * facts->participants / *bw;
            }
          }
          return estimate;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
