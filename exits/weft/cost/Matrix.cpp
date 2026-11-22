#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
namespace CI_UNIT_NAMESPACE {
void nativePanelCost(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {{"native-panels", "levels/execution/cpu::matrix-partition", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          const auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto &dot = intentdsl::cpu::calculation(*op);
          compilerinfra::CostEstimate e;
          e.scope = "one native task";
          e.bottleneck = "task operand supply";
          if (dot.getName().stripDialect() == "q4k_dot")
            return e;
          auto g = intentdsl::kir::contractionShape(
              dot, op->getOperand(0),
              *ctx.target.services.get<intentdsl::CallFacts>());
          if (!g)
            return g.takeError();
          auto machine = ctx.target.services.get<intentdsl::cpu::RowMachine>();
          bool blocked = !machine->boundWidening && !g->grouped &&
                         g->m % t[0] == 0 && g->n % t[1] == 0;
          double rows = blocked ? std::min<int64_t>(t[0], 4) : 1;
          double reloads =
              (std::ceil(t[0] / rows) - 1) * t[1] * g->k *
              mlir::cast<mlir::MemRefType>(dot.getOperand(0).getType())
                  .getElementTypeBitWidth() /
              8;
          e.terms["native_simultaneous_rows"] = {rows, "count"};
          e.terms["rhs_reloads"] = {reloads, "B"};
          auto bw = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns");
          if (bw)
            e.timeNs = reloads / *bw;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
