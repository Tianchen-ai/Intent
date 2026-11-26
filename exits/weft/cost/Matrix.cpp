#include "../legalize/Matrix.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include <algorithm>
#include <cmath>

namespace CI_UNIT_NAMESPACE {
void nativePanelCost(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {{"native-panels", "levels/execution/cpu::matrix-partition", "2",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          const auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto &dot = intentdsl::cpu::calculation(*op);
          compilerinfra::CostEstimate e;
          e.scope = "one native task";
          e.bottleneck = "SIMD/RVV register file and local array supply";
          if (dot.getName().stripDialect() == "q4k_dot")
            return e;
          if (dot.getName().stripDialect() == "quant_contract") {
            auto fmt = dot.getAttrOfType<mlir::StringAttr>("format");
            const auto *spec =
                fmt ? intentdsl::kir::lookupQuantFormat(fmt.getValue())
                    : nullptr;
            if (spec && spec->roleLayout ==
                            intentdsl::kir::QuantRoleLayout::RecordGemv)
              return e;
          }
          auto g = intentdsl::kir::contractionShape(
              dot, op->getOperand(0),
              *ctx.target.services.get<intentdsl::CallFacts>());
          if (!g)
            return g.takeError();
          auto machine = ctx.target.services.get<intentdsl::cpu::RowMachine>();
          auto *rf = ctx.target.services.get<RecordFields>();
          int64_t workers = machine->workers;
          double columns = std::ceil(double(t[1]) / t[3]);
          int64_t rawVecRegs =
              rf ? rf->vectorRegisters
                 : (machine->boundWidening ? 32 : (t[3] == 16 ? 32 : 16));
          int64_t vecRegs = machine->boundWidening
                                ? std::max<int64_t>(8, rawVecRegs - 4)
                                : rawVecRegs;
          mlir::Type outElem =
              mlir::cast<mlir::MemRefType>(op->getOperand(0).getType())
                  .getElementType();
          mlir::Type lhsElem =
              mlir::cast<mlir::MemRefType>(dot.getOperand(0).getType())
                  .getElementType();
          bool rvvSoftBf16Out =
              machine->boundWidening && outElem.isBF16() &&
              !(rf && rf->zvfbfmin);
          bool rvvSoftBf16In =
              machine->boundWidening && lhsElem.isBF16() &&
              !(rf && rf->zvfbfmin);
          double epilogueCastGroups =
              rvvSoftBf16Out ? (4.0 * columns + 1.0) : 0.0;
          double inCastExtra = rvvSoftBf16In ? columns : 0.0;
          double live =
              std::max(epilogueCastGroups,
                       t[0] * columns +
                           (machine->boundWidening
                                ? 3.0 * columns + 4.0 + inCastExtra
                                : columns + 2.0));
          double spill = std::max(0.0, live - vecRegs) * t[3] * 4.0 * 2.0 * g->k;

          // When N > tile_n and N % tile_n != 0, Weft task.weftir emits a
          // dynamic [-2] column slice whose K-loop accumulator materializes
          // in local stack storage rather than staying in vector registers.
          bool dynamicColTail = (g->n > t[1]) && (g->n % t[1] != 0);

          int64_t packCode = t.size() > 4 ? t[4] : 0;
          double rawElemBytes =
              (dot.getName().stripDialect() == "scaled_contract" ||
               dot.getName().stripDialect() == "quant_contract")
                  ? 4.0
                  : double(lhsElem.getIntOrFloatBitWidth()) / 8.0;
          double inBytes = (packCode == 3 ||
                            (dot.getName().stripDialect() == "window_contract" &&
                             packCode == 1))
                               ? 4.0
                               : rawElemBytes;
          bool inF32 = (inBytes == 4.0 && !lhsElem.isInteger(32));

          double maxLiveRows = std::max(1.0, std::floor(16.0 / std::max(1.0, columns)));
          double simultaneousRows =
              inF32 ? std::min<double>(t[0], maxLiveRows)
                    : (machine->boundWidening
                           ? (rawElemBytes == 1.0 ? std::min<double>(t[0], 2.0)
                                                  : 1.0)
                           : std::min<double>(t[0], maxLiveRows));
          double reloads =
              (std::ceil(double(t[0]) / simultaneousRows) - 1.0) * t[1] * g->k *
              inBytes;

          // On x86, Weft native outer_contract materializes local ArrayStorage +
          // validity masks for admitted operands and output; on RVV, static tiles
          // stream directly into vector registers while dynamic [-2] column tails
          // spill the f32 accumulator array per K step.
          double arrayCopyBytes =
              machine->boundWidening
                  ? (dynamicColTail ? 2.0 * double(t[0]) * t[1] * 4.0 * g->k
                                    : 0.0)
                  : ((double(t[0]) + double(t[1])) * g->k * (inBytes + 1.0) +
                     double(t[0]) * t[1] * 5.0);

          double inLoopWiden =
              !inF32 ? double(g->k) * (columns + t[0]) *
                           (machine->boundWidening ? 1.5 : 1.0)
                     : 0.0;
          double l1Loads = double(g->k) * (t[0] + columns);

          e.terms["vector_register_budget"] = {double(vecRegs), "count"};
          e.terms["native_simultaneous_rows"] = {simultaneousRows, "count"};
          e.terms["rhs_reloads"] = {reloads, "B"};
          e.terms["local_array_bytes"] = {arrayCopyBytes, "B"};
          e.terms["spill_bytes_per_task"] = {spill, "B"};
          auto bw = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns"),
               writeBw = intentdsl::quantity(ctx, "write_1_private_bw", "B/ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          if (bw && writeBw && rate) {
            double widths = workers * t[3];
            e.timeNs = reloads / *bw +
                       ((spill + arrayCopyBytes) / 2.0) * workers *
                           (1.0 / *bw + 1.0 / *writeBw) +
                       ((l1Loads * 0.6 + inLoopWiden) * widths) / *rate;
          } else if (bw) {
            e.timeNs = (reloads + spill + arrayCopyBytes) / *bw;
          } else {
            double effRate = std::max(1.0, double(machine->vectorBits) / 16.0);
            e.timeNs = (reloads + spill + arrayCopyBytes) / 16.0 +
                       ((l1Loads * 0.6 + inLoopWiden) * t[3]) / effRate;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});

  u.costModels.push_back(
      {{"q4k-panels", "levels/execution/cpu::matrix-partition", "2",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          const auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto &dot = intentdsl::cpu::calculation(*op);
          auto g = intentdsl::kir::contractionShape(
              dot, op->getOperand(0),
              *ctx.target.services.get<intentdsl::CallFacts>());
          if (!g)
            return g.takeError();
          compilerinfra::CostEstimate e;
          e.scope = "one native record task";
          e.bottleneck = "packed weight decoding and subgroup dot products";
          const auto *machine =
              ctx.target.services.get<intentdsl::cpu::RowMachine>();
          const auto *fields = ctx.target.services.get<RecordFields>();
          int64_t rows = std::min(t[0], g->m);
          double dotWork = 2. * rows * g->k;
          e.terms["subgroup_ops"] = {dotWork, "FLOP"};
          double work = dotWork;
          auto format = dot.getAttrOfType<mlir::StringAttr>("format");
          const auto *spec =
              format ? intentdsl::kir::lookupQuantFormat(format.getValue())
                     : nullptr;
          if (spec &&
              spec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv &&
              !fields->grouped) {
            int64_t lanes = t[3], bits = lanes * 32;
            int64_t parallel = fields->decodedRows(*spec, bits, t[0]);
            double blocks = double(g->k) / spec->blockElements;
            double groups = double(spec->blockElements) /
                            RecordFields::subgroupWidth(*spec);
            // Each input row supplies one activation record to all weight
            // rows in its register block. Byte-plane reconstruction performs
            // a shift and mask per plane, then merges adjacent planes.
            double weightVectors =
                rows * blocks * llvm::divideCeil(spec->blockElements * 8, bits);
            double decode = weightVectors * (3 * spec->numBitPlanes - 1);
            double scales = rows * blocks * std::ceil(groups / lanes) *
                            (1 + (spec->zeroBias != 0));
            double activationLoads =
                llvm::divideCeil(rows, parallel) * blocks *
                llvm::divideCeil(spec->blockElements * 8, bits);
            e.terms["record_rows_per_group"] = {double(parallel), "count"};
            e.terms["record_decode_vectors"] = {decode, "count"};
            e.terms["record_scale_vectors"] = {scales, "count"};
            e.terms["activation_vector_loads"] = {activationLoads, "count"};
            work += (decode + scales + activationLoads) * lanes;
          }
          auto rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          double effRate =
              rate ? *rate / machine->workers : machine->vectorBits / 8.;
          // The common partition estimator scales the per-task issue time
          // by the number of waves and separately accounts for record traffic.
          e.timeNs = 25. + work / effRate;
          return e;
        }},
       COMPILERINFRA_SOURCE});

  u.costModels.push_back(
      {{"brgemm-panels", "levels/execution/cpu::matrix-partition", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          const auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto *caps = ctx.target.services.get<intentdsl::cpu::MatrixMachine>();
          const auto &impl = caps->contracts[t[5]];
          auto &dot = intentdsl::cpu::calculation(*op);
          auto g = intentdsl::kir::contractionShape(
              dot, op->getOperand(0),
              *ctx.target.services.get<intentdsl::CallFacts>());
          if (!g)
            return g.takeError();
          compilerinfra::CostEstimate e;
          e.scope = "one Weft zero-copy BRGeMM task";
          auto read = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns"),
               write = intentdsl::quantity(ctx, "write_1_private_bw", "B/ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          auto workers =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;
          double jobs = std::ceil(double(g->m) / t[0]) *
                        std::ceil(double(g->n) / t[1]) * g->batches;
          double waves = std::max(1.0, std::ceil(jobs / workers));
          constexpr double kTlsLookupNs = 8.0;

          if (impl.dtype == "q4k") {
            bool amx = ctx.target.features.count("amx_int8") > 0 && t[0] >= 16;
            auto launch = intentdsl::quantity(ctx, "launch", "ns"),
                 sync = intentdsl::quantity(ctx, "sync", "ns");
            double amxEnterLeaveNs =
                (launch ? *launch * 4.0 : 11000.0) +
                (sync ? *sync * 2.0 : 1400.0);
            double planNs = sync ? *sync * 1.5 : 1000.0;
            double amxCtx = amx ? amxEnterLeaveNs : planNs;
            e.bottleneck = "per-subgroup ukernel call, 16-col padding, and nibble unpack";
            double subgroups = double(g->k) / 32.0;
            double unpackBytes = double(t[0]) * g->k * 2.0;
            e.terms["subgroup_ukernel_calls"] = {subgroups, "count"};
            e.terms["nibble_stack_bytes"] = {unpackBytes, "B"};
            if (read && write && rate) {
              double callCost = launch ? std::min(*launch * 0.05, 25.0) : 20.0;
              double ops = double(t[0]) * subgroups * 16.0;
              e.timeNs = amxCtx / waves + subgroups * callCost +
                         (unpackBytes / 2.0) * workers *
                             (1.0 / *read + 1.0 / *write) +
                         (ops * workers) / *rate;
            } else {
              auto *machine =
                  ctx.target.services.get<intentdsl::cpu::RowMachine>();
              double effRate = double(machine->vectorBits) / 8.0;
              double ops = double(t[0]) * subgroups * 16.0;
              e.timeNs = amxCtx / waves + subgroups * 20.0 +
                         unpackBytes / 32.0 + ops / effRate;
            }
            return e;
          }

          if (impl.dtype == "bf16" || impl.dtype == "s8") {
            e.bottleneck = "AMX zero-copy pack32 BRGeMM";
            double columns = std::ceil(double(t[1]) / t[3]);
            double tileOutBytes = double(t[0]) * t[1] * 4.0;
            e.terms["tile_output_bytes"] = {tileOutBytes, "B"};
            if (write && rate) {
              double widths = workers * t[3];
              e.timeNs = kTlsLookupNs + 120.0 / waves +
                         (tileOutBytes * workers) / *write +
                         (double(t[0]) * columns * widths * 0.25) / *rate;
            }
            return e;
          }

          e.bottleneck = "zero-copy JIT AVX-512 BRGeMM microkernel";
          double columns = std::ceil(double(t[1]) / t[3]);
          double live = std::min<double>(t[3] == 16 ? 32.0 : 16.0,
                                         t[0] * columns + columns + 2.0);
          double aCacheLines = double(t[0]) / double(t[3]);
          double l1Loads = double(g->k) * (aCacheLines + columns);
          double inTileConvert =
              (impl.dtype == "f16" && !ctx.target.features.count("amx_fp16"))
                  ? double(g->k) * (t[0] + columns)
                  : 0.0;
          double tileOutBytes = double(t[0]) * t[1] * 4.0;
          e.terms["live_vectors"] = {live, "count"};
          if (read && write && rate) {
            double widths = workers * t[3];
            e.timeNs = kTlsLookupNs + 120.0 / waves +
                       (tileOutBytes * workers) / *write +
                       ((l1Loads * 0.25 + inTileConvert) * widths) / *rate;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});

  u.costModels.push_back(
      {{"row-primitive", "levels/execution/cpu::partition", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          compilerinfra::CostEstimate e;
          e.scope = "row task";
          e.bottleneck = "segment gather and horizontal reduction";
          auto d = mlir::cast<mlir::DictionaryAttr>(c.value);
          auto segAttr = d.getAs<mlir::IntegerAttr>("segment_width");
          if (!segAttr)
            return e;
          mlir::Operation *segOp = nullptr;
          op->walk([&](mlir::Operation *v) {
            if (v->getName().stripDialect() == "ordered_sum") {
              segOp = v;
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          });
          if (!segOp || segOp->getNumRegions() != 1 ||
              segOp->getRegion(0).empty())
            return e;
          auto *facts = ctx.target.services.get<intentdsl::CallFacts>();
          auto *rm = ctx.target.services.get<intentdsl::cpu::RowMachine>();
          if (!facts || !rm)
            return e;
          int64_t segW = segAttr.getInt();
          double nativeLanes = std::max(1.0, double(rm->vectorBits) / 32.0);
          double offLen = 0.0, directNnz = 0.0;
          if (auto *oShape = facts->shapeOf(segOp->getOperand(0));
              oShape && !oShape->empty())
            offLen = double((*oShape)[0].getAsInteger().value_or(0));
          double ops = 0.0;
          auto &segBb = segOp->getRegion(0).front();
          for (mlir::Operation &scOp : segBb) {
            auto code = scOp.getName().stripDialect();
            if (code == "member_load" &&
                scOp.getOperand(1) == segBb.getArgument(0)) {
              if (auto *mShape = facts->shapeOf(scOp.getOperand(0));
                  mShape && !mShape->empty())
                directNnz = std::max(
                    directNnz, double((*mShape)[0].getAsInteger().value_or(0)));
            } else if (code == "mul" || code == "add" || code == "cast") {
              ops += 1.0;
            }
          }
          ops = std::max(2.0, ops);
          double numRows = offLen > 1.0 ? (offLen - 1.0)
                                        : std::max(1.0, facts->geometry(op).first);
          double avgLen = directNnz > 0.0 ? (directNnz / numRows) : double(segW);
          double vecSteps = std::floor(avgLen / double(segW));
          double tailElems = avgLen - vecSteps * double(segW);
          double reduceSteps = std::log2(std::max(1.0, double(segW)));
          double laneOpsPerRow =
              vecSteps * (nativeLanes * (ops + 1.0) + double(segW)) +
              tailElems * nativeLanes * (ops + 1.0) +
              reduceSteps * nativeLanes;
          double totalLaneOps = numRows * laneOpsPerRow;
          e.terms["segment_width"] = {double(segW), "lane"};
          e.terms["avg_segment_length"] = {avgLen, "element"};
          e.terms["segment_vec_steps"] = {numRows * vecSteps, "count"};
          e.terms["segment_scalar_tail"] = {numRows * tailElems, "element"};
          e.terms["segment_reduce_steps"] = {numRows * reduceSteps,
                                             "instruction"};
          auto rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          double effRate = rate.value_or(std::max(1.0, nativeLanes * 2.0));
          e.timeNs = totalLaneOps / effRate;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
