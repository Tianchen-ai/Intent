#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/OrderedValue.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <numeric>
using namespace mlir;
namespace intentdsl::cpu {
static bool sameInterval(Value v, BlockArgument head, Value projected) {
  if (projected && v == projected)
    return true;
  if (v == head)
    return false;
  if (auto a = dyn_cast<BlockArgument>(v)) {
    if (auto loop = dyn_cast<scf::ForOp>(a.getOwner()->getParentOp()))
      return v == loop.getInductionVar() &&
             sameInterval(loop.getLowerBound(), head, projected) &&
             sameInterval(loop.getUpperBound(), head, projected) &&
             sameInterval(loop.getStep(), head, projected);
    return true;
  }
  auto *p = v.getDefiningOp();
  if (auto ld = dyn_cast<memref::LoadOp>(p))
    if (llvm::any_of(ld.getMemref().getUsers(), [](Operation *u) {
          return isa<memref::StoreOp>(u) ||
                 u->getName().stripDialect() == "slice_write";
        }))
      return false;
  if (auto branch = dyn_cast<scf::IfOp>(p))
    return branch.getNumResults() == 1 &&
           sameInterval(branch.getCondition(), head, projected) &&
           sameInterval(branch.thenYield().getOperand(0), head, projected) &&
           sameInterval(branch.elseYield().getOperand(0), head, projected);
  return (isa<memref::LoadOp, memref::DimOp>(p) ||
          p->getName().stripDialect() == "slice_read" ||
          p->getName().stripDialect() == "lane_splat" ||
          p->getName().getDialectNamespace() == "arith" ||
          p->getName().getDialectNamespace() == "math") &&
         llvm::all_of(p->getOperands(), [&](Value x) {
           return sameInterval(x, head, projected);
         });
}
bool invariantOwnerValue(Value v, Operation &space, const OwnerReuse &r) {
  return sameInterval(v, space.getRegion(0).front().getArgument(r.axis),
                      r.quotient);
}
std::optional<OwnerReuse> sharedInterval(Operation *space,
                                         const CallFacts &facts) {
  auto &body = space->getRegion(0).front();
  auto axes = space->getAttrOfType<DenseI64ArrayAttr>("axes");
  // Interleaving is legal for read-only folds followed by publications, and
  // for folds whose only in-loop write is a single disjoint per-(owner, step)
  // scalar publication to a write-only output view.
  llvm::SmallDenseSet<Value, 8> readViews;
  llvm::DenseMap<Value, unsigned> writeSites;
  body.walk([&](Operation *op) {
    if (auto ld = dyn_cast<memref::LoadOp>(op))
      readViews.insert(ld.getMemref());
    else if (op->getName().stripDialect() == "slice_read")
      readViews.insert(op->getOperand(0));
    else if (auto st = dyn_cast<memref::StoreOp>(op))
      ++writeSites[st.getMemref()];
    else if (op->getName().stripDialect() == "slice_write")
      ++writeSites[op->getOperand(1)];
  });
  auto unitTrailingStride = [&](Value view) {
    auto ba = dyn_cast<BlockArgument>(view);
    if (!ba || ba.getArgNumber() >= facts.arguments.size())
      return false;
    auto *obj = facts.arguments[ba.getArgNumber()].getAsObject();
    auto *st = obj ? obj->getArray("strides") : nullptr;
    return st && !st->empty() && (*st)[st->size() - 1].getAsInteger() == 1;
  };
  auto isDisjointStepStore = [&](memref::StoreOp st) {
    auto parentFor = dyn_cast<scf::ForOp>(st->getParentOp());
    if (!parentFor || parentFor->getBlock() != &body ||
        st->getBlock() != parentFor.getBody())
      return false;
    Value dst = st.getMemref();
    if (!isa<BlockArgument>(dst) || readViews.contains(dst) ||
        writeSites.lookup(dst) != 1)
      return false;
    auto idxs = st.getIndices();
    if (!llvm::is_contained(idxs, parentFor.getInductionVar()))
      return false;
    return llvm::all_of(body.getArguments(), [&](BlockArgument ob) {
      return llvm::is_contained(idxs, Value(ob));
    });
  };
  bool simpleWrites = !body.walk([&](Operation *op) {
                             if (op->getName().stripDialect() == "slice_write")
                               return op->getBlock() == &body
                                          ? WalkResult::advance()
                                          : WalkResult::interrupt();
                             if (auto st = dyn_cast<memref::StoreOp>(op))
                               return (op->getBlock() == &body ||
                                       isDisjointStepStore(st))
                                          ? WalkResult::advance()
                                          : WalkResult::interrupt();
                             if (isa<scf::ForOp>(op) && op->getBlock() != &body)
                               return WalkResult::interrupt();
                             return WalkResult::advance();
                           })
                           .wasInterrupted();

  auto evaluateCandidate = [&](BlockArgument h, int64_t groupVal,
                               Value quotient) -> std::optional<OwnerReuse> {
    if (groupVal < 2)
      return {};
    auto axis = h.getArgNumber();
    auto *shape = facts.shapeOf(space->getOperand(axis));
    if (!shape)
      return {};
    auto extent = (*shape)[axes[axis]].getAsInteger();
    if (!extent || *extent < 2)
      return {};
    if (!simpleWrites) {
      if (quotient)
        return {};
      bool badBounds =
          body.walk([&](scf::ForOp loop) {
                return (sameInterval(loop.getLowerBound(), h, Value{}) &&
                        sameInterval(loop.getUpperBound(), h, Value{}) &&
                        sameInterval(loop.getStep(), h, Value{}))
                           ? WalkResult::advance()
                           : WalkResult::interrupt();
              })
              .wasInterrupted();
      if (badBounds)
        return {};
      llvm::DenseMap<Value, unsigned> ownerAxisOf;
      bool badAccess =
          body.walk([&](Operation *op) {
                Value view;
                SmallVector<Value, 4> coords;
                if (auto ld = dyn_cast<memref::LoadOp>(op)) {
                  view = ld.getMemref();
                  coords.assign(ld.getIndices().begin(), ld.getIndices().end());
                } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
                  view = st.getMemref();
                  coords.assign(st.getIndices().begin(), st.getIndices().end());
                } else if (op->getName().stripDialect() == "slice_read" ||
                           op->getName().stripDialect() == "slice_write") {
                  bool wr = op->getName().stripDialect() == "slice_write";
                  view = op->getOperand(wr ? 1 : 0);
                  auto rest = op->getOperands().drop_front(wr ? 2 : 1);
                  coords.assign(rest.begin(), rest.end());
                  int64_t ax = op->getAttrOfType<IntegerAttr>("axis").getInt();
                  coords.insert(coords.begin() + ax, Value{});
                } else {
                  return WalkResult::advance();
                }
                if (!writeSites.contains(view))
                  return WalkResult::advance();
                if (llvm::count(coords, Value(h)) != 1)
                  return WalkResult::interrupt();
                unsigned pos = llvm::find(coords, Value(h)) - coords.begin();
                auto [it, inserted] = ownerAxisOf.try_emplace(view, pos);
                return (inserted || it->second == pos)
                           ? WalkResult::advance()
                           : WalkResult::interrupt();
              })
              .wasInterrupted();
      if (badAccess)
        return {};
    }
    double bytes = 0;
    for (auto loop : body.getOps<scf::ForOp>()) {
      if (!sameInterval(loop.getLowerBound(), h, quotient) ||
          !sameInterval(loop.getUpperBound(), h, quotient) ||
          !sameInterval(loop.getStep(), h, quotient)) {
        bytes = 0;
        break;
      }
      for (auto &read : loop.getBody()->without_terminator()) {
        if (read.getName().stripDialect() != "slice_read")
          continue;
        auto cs = read.getOperands().drop_front();
        if (quotient && !llvm::is_contained(cs, quotient))
          continue;
        if (!llvm::all_of(
                cs, [&](Value v) { return sameInterval(v, h, quotient); }))
          continue;
        auto *s = facts.shapeOf(read.getOperand(0));
        if (!s)
          continue;
        bytes += (*s)[read.getAttrOfType<IntegerAttr>("axis").getInt()]
                     .getAsInteger()
                     .value_or(0) *
                 cast<ShapedType>(read.getResult(0).getType())
                     .getElementTypeBitWidth() /
                 8.;
      }
    }
    if (bytes <= 0 && !quotient) {
      bool hasDivergentMemIf = false;
      for (auto loop : body.getOps<scf::ForOp>()) {
        if (!sameInterval(loop.getLowerBound(), h, Value{}) ||
            !sameInterval(loop.getUpperBound(), h, Value{}) ||
            !sameInterval(loop.getStep(), h, Value{})) {
          bytes = 0;
          break;
        }
        loop.walk([&](Operation *op) {
          if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            if (!sameInterval(ifOp.getCondition(), h, Value{})) {
              ifOp.walk([&](Operation *inner) {
                if (isa<memref::LoadOp, memref::StoreOp>(inner))
                  hasDivergentMemIf = true;
              });
            }
            return;
          }
          double weight = (isa<scf::ForOp>(op->getParentOp()) &&
                           op->getParentOp() != loop.getOperation())
                              ? 8.0
                              : 1.0;
          if (auto ld = dyn_cast<memref::LoadOp>(op)) {
            double elemB = ld.getType().getIntOrFloatBitWidth() / 8.0;
            if (!writeSites.contains(ld.getMemref()) &&
                llvm::all_of(ld.getIndices(), [&](Value v) {
                  return sameInterval(v, h, Value{});
                })) {
              bytes += elemB * weight;
            } else if (!ld.getIndices().empty() &&
                       ld.getIndices().back() == h &&
                       unitTrailingStride(ld.getMemref()) &&
                       llvm::all_of(ld.getIndices().drop_back(), [&](Value v) {
                         return sameInterval(v, h, Value{});
                       })) {
              bytes += elemB * weight;
            }
          } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
            if (!st.getIndices().empty() && st.getIndices().back() == h &&
                unitTrailingStride(st.getMemref()) &&
                llvm::all_of(st.getIndices().drop_back(), [&](Value v) {
                  return sameInterval(v, h, Value{});
                })) {
              bytes +=
                  (st.getValue().getType().getIntOrFloatBitWidth() / 8.0) *
                  weight;
            }
          }
        });
      }
      if (hasDivergentMemIf)
        bytes *= 0.25;
    }
    if (bytes <= 0)
      return {};
    return OwnerReuse{axis, groupVal, *extent, bytes, quotient};
  };

  auto scoreOf = [](const OwnerReuse &r) -> double {
    int64_t bestBatch = 1;
    for (int64_t w : {2, 4, 8, 16})
      if (r.group % w == 0 && r.extent % w == 0)
        bestBatch = w;
    return bestBatch > 1 ? double(bestBatch) * r.readBytes : 0.0;
  };

  std::optional<OwnerReuse> best;
  double bestScore = 0.0;
  for (auto d : body.getOps<arith::FloorDivSIOp>()) {
    auto h = dyn_cast<BlockArgument>(d.getLhs());
    auto group = d.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (!h || h.getOwner() != &body || !group || group.value() < 2)
      continue;
    if (auto cand = evaluateCandidate(h, group.value(), d.getResult())) {
      double sc = scoreOf(*cand);
      if (sc > bestScore) {
        bestScore = sc;
        best = cand;
      }
    }
  }
  for (BlockArgument h : body.getArguments()) {
    auto *shape = facts.shapeOf(space->getOperand(h.getArgNumber()));
    if (!shape)
      continue;
    auto ext = (*shape)[axes[h.getArgNumber()]].getAsInteger();
    if (!ext || *ext < 2)
      continue;
    if (auto cand = evaluateCandidate(h, *ext, Value{})) {
      double sc = scoreOf(*cand);
      if (sc > bestScore) {
        bestScore = sc;
        best = cand;
      }
    }
  }
  return best;
}
llvm::SmallVector<unsigned> ownerRadices(Operation &space) {
  auto group = space.getAttrOfType<IntegerAttr>("cpu.innermost_owner");
  llvm::SmallVector<unsigned> result;
  if (group)
    result.push_back(group.getInt());
  for (unsigned i = space.getNumOperands(); i-- > 0;)
    if (!group || group.getInt() != i)
      result.push_back(i);
  return result;
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
void sharedOwnerTraversal(compilerinfra::Unit &u) {
  using namespace compilerinfra;
  Strategy s;
  s.name = "shared-intervals";
  s.family = u.qualify("ordered-owners");
  s.enumerate = [](Operation *op, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> cs;
    const auto &facts = *p.services.get<intentdsl::CallFacts>();
    auto reuse = intentdsl::cpu::sharedInterval(op, facts);
    auto &machine = *p.services.get<intentdsl::cpu::RowMachine>();
    if (!reuse || (!machine.packedOwners && !machine.repeatedScoreOwners))
      return cs;
    Builder b(op->getContext());
    bool repeat = bool(intentdsl::cpu::repeatedScore(*op));
    auto maximum = repeat ? std::max(machine.packedOwners, machine.repeatedScoreOwners)
                          : machine.packedOwners;
    if (!maximum)
      return cs;
    auto score = repeat ? intentdsl::cpu::reusableScores(
                              op, p, intentdsl::cpu::scoreSlotLimit(machine))
                        : std::nullopt;
    auto panel =
        repeat ? intentdsl::cpu::selectOrderedDot(*op, p, reuse->axis, maximum)
               : std::nullopt;
    auto vectorDot = machine.repeatedScoreOwners
                         ? intentdsl::cpu::orderedDotPanel(*op, facts, reuse->axis, maximum, false)
                         : std::nullopt;
    for (int grain : {2, 4, 8, 16, 32, 128, 256}) {
      unsigned limit =
          panel ? std::max(maximum, machine.orderedMatrixOwners) : maximum;
      for (unsigned batch = 2; batch <= limit && batch <= unsigned(grain);
           batch *= 2) {
        if (grain % batch || reuse->group % batch || reuse->extent % batch)
          continue;
        bool matrixOnly = batch > maximum;
        auto selectedPanel =
            matrixOnly
                ? intentdsl::cpu::selectOrderedDot(*op, p, reuse->axis, batch)
                : panel;
        if (matrixOnly && !selectedPanel)
          continue;
        std::string id =
            "shared-" + std::to_string(grain) + "x" + std::to_string(batch);
        auto d = b.getDictionaryAttr(
            {b.getNamedAttr("grain", b.getI64IntegerAttr(grain)),
             b.getNamedAttr("inner", b.getI64IntegerAttr(reuse->axis)),
             b.getNamedAttr("batch", b.getI64IntegerAttr(batch))});
        if (batch <= machine.packedOwners)
          cs.push_back({id, "shared-interval-owners", d, {"exclusive-output"}});
        if (repeat) {
          NamedAttrList row(d);
          int64_t slots = intentdsl::cpu::batchScoreSlots / batch;
          if (score && score->keys > slots &&
              score->keys <= intentdsl::cpu::scoreSlotLimit(machine) / batch)
            slots = score->keys;
          if (batch > machine.packedOwners && !matrixOnly) {
            if (!vectorDot)
              continue;
            // The unfused FP32 product has one element and one validity byte
            // per owner, key and depth coordinate. The score window is live
            // alongside the product and the two widened input panels.
            double tile = machine.orderedDotTokens, depth = vectorDot->depth;
            double bytes = sizeof(float) * double(batch) * std::min(slots, vectorDot->tokens) +
                           (sizeof(float) + 1) * depth * (batch * tile + batch + tile);
            if (bytes > machine.ownerStackBytes)
              continue;
          }
          row.set("window", b.getI64IntegerAttr(slots));
          if (!matrixOnly)
            cs.push_back({id + "-scores",
                          "shared-interval-owners",
                          row.getDictionary(b.getContext()),
                          {"exclusive-output"}});
          if (selectedPanel && batch >= maximum) {
            row.set("engine", b.getStringAttr("onednn-brgemm-bf16"));
            auto base = id + "-scores-brgemm";
            int64_t width =
                p.services.get<intentdsl::cpu::RowMachine>()->orderedDotTokens;
            auto value = orderedValuePanel(*op, facts, *panel);
            for (int mult : {1, 2, 4}) {
              int64_t tile = width * mult;
              if ((!width && mult > 1) || tile > panel->tokens)
                continue;
              row.set("score_tile",
                      b.getI64IntegerAttr(tile ? tile : panel->tokens));
              auto candidate =
                  base + (mult == 1 ? "" : "-" + std::to_string(tile));
              if (!matrixOnly)
                cs.push_back({candidate,
                              "shared-interval-owners",
                              row.getDictionary(b.getContext()),
                              {"exclusive-output"},
                              {"exits/" + p.name + "::onednn-brgemm-bf16"}});
              if (value && value->supported(p) && slots >= panel->tokens) {
                NamedAttrList pv(row);
                pv.set("value_width", b.getI64IntegerAttr(value->columns));
                cs.push_back({candidate + "-pv",
                              "shared-interval-owners",
                              pv.getDictionary(b.getContext()),
                              {"exclusive-output"},
                              {"exits/" + p.name + "::onednn-brgemm-bf16"}});
              }
            }
          }
        }
      }
    }
    return cs;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
