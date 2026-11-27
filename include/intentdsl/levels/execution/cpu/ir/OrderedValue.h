#pragma once
#include "intentdsl/levels/algorithm/kir/ir/NativeDot.h"
#include "intentdsl/levels/execution/cpu/ir/OrderedDot.h"

namespace intentdsl::cpu {
// A hardware FTZ control applies to every instruction on the worker. Check
// the effective IR mode including nested overrides before changing MXCSR.
inline bool allowsWorkerOutputFtz(mlir::Operation &scope,
                                  const compilerinfra::TargetProfile &profile) {
  if (!profile.features.count("amx_bf16"))
    return false;
  return !scope
              .walk([&](mlir::Operation *op) {
                return compilerinfra::effectiveDenormalMode(op).Output ==
                               llvm::DenormalMode::PreserveSign
                           ? mlir::WalkResult::advance()
                           : mlir::WalkResult::interrupt();
              })
              .wasInterrupted();
}

struct OrderedValuePanel {
  mlir::Operation *add, *read;
  int64_t columns, tokens, elements;
  bool supported(const compilerinfra::TargetProfile &profile) const {
    auto &machine = *profile.services.get<RowMachine>();
    // orderedValuePanel already verified the effective FTZ mode of the
    // multiply and its accumulator, including author and per-op overrides.
    return columns >= machine.orderedValueMinColumns &&
           columns % machine.orderedValueColumnMultiple == 0;
  }
};

// The BF16 value contraction is legal only at the author's explicit round.
// Replay keeps every other use of the same weight (e.g. mass) in f32.
inline std::optional<OrderedValuePanel>
orderedValuePanel(mlir::Operation &space, const CallFacts &facts,
                  const OrderedDotPanel &qk, bool requiresFtz = true) {
  using namespace mlir;
  std::optional<OrderedValuePanel> result;
  auto repeated = repeatedScore(space);
  space.walk([&](scf::ForOp loop) {
    if (repeated) {
      if (loop != repeated->second)
        return;
    } else {
      auto branch = dyn_cast<scf::IfOp>(loop->getParentOp());
      if (!branch || !branch->hasAttr("cpu.window_branch") ||
          loop->getParentRegion() != &branch.getThenRegion())
        return;
    }
    if (result || loop.getNumRegionIterArgs() != 2 ||
        !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One()))
      return;
    auto *yield = loop.getBody()->getTerminator();
    auto add = yield->getOperand(1).getDefiningOp<arith::AddFOp>();
    auto carry = loop.getRegionIterArgs()[1];
    if (!add ||
        !compilerinfra::hasFastMath(add, arith::FastMathFlags::reassoc) ||
        (requiresFtz ? !intentdsl::kir::permitsBf16Dot(add)
                     : !compilerinfra::hasFastMath(
                           add, arith::FastMathFlags::contract)) ||
        !carry.hasOneUse())
      return;
    auto *zero = loop.getInitArgs()[1].getDefiningOp();
    if (!zero || zero->getName().stripDialect() != "lane_splat" ||
        !matchPattern(zero->getOperand(0), m_PosZeroFloat()))
      return;
    auto v = add.getLhs() == carry   ? add.getRhs()
             : add.getRhs() == carry ? add.getLhs()
                                     : Value{};
    auto mul = v ? v.getDefiningOp<arith::MulFOp>() : arith::MulFOp{};
    if (!mul || (requiresFtz ? !intentdsl::kir::permitsBf16Dot(mul)
                             : !compilerinfra::hasFastMath(
                                   mul, arith::FastMathFlags::contract)))
      return;
    for (unsigned i = 0; i < 2; ++i) {
      auto *splat = mul->getOperand(i).getDefiningOp();
      auto ext = mul->getOperand(1 - i).getDefiningOp<arith::ExtFOp>();
      auto *read = ext ? ext.getIn().getDefiningOp() : nullptr;
      if (!splat || splat->getName().stripDialect() != "lane_splat" || !read ||
          read->getName().stripDialect() != "slice_read" ||
          read->getBlock() != loop.getBody())
        continue;
      auto factor = splat->getOperand(0).getDefiningOp<arith::ExtFOp>();
      auto narrow = factor ? factor.getIn().getDefiningOp<arith::TruncFOp>()
                           : arith::TruncFOp{};
      if (!narrow || !narrow.getOut().getType().isBF16() ||
          !narrow.getIn().getType().isF32())
        continue;
      auto arg = dyn_cast<BlockArgument>(read->getOperand(0));
      auto ty = arg ? dyn_cast<MemRefType>(arg.getType()) : MemRefType{};
      if (!ty || ty.getRank() < 2 || !ty.getElementType().isBF16() ||
          !isa<func::FuncOp>(arg.getOwner()->getParentOp()) ||
          read->getAttrOfType<IntegerAttr>("axis").getInt() !=
              ty.getRank() - 1 ||
          read->getOperands().back() != loop.getInductionVar())
        continue;
      if (read->getOperands().drop_front().drop_back() !=
          qk.key->getOperands().drop_front().drop_back())
        continue;
      if (arg.getArgNumber() >= facts.arguments.size())
        continue;
      auto *record = facts.arguments[arg.getArgNumber()].getAsObject();
      auto *shape = record ? record->getArray("shape") : nullptr;
      auto *stride = record ? record->getArray("strides") : nullptr;
      if (!shape || !stride || stride->size() != shape->size() ||
          shape->size() != unsigned(ty.getRank()))
        continue;
      int64_t extent = 1;
      for (int j = ty.getRank() - 1; j >= 0; --j) {
        auto n = (*shape)[j].getAsInteger();
        if (!n || *n <= 0 || (*stride)[j].getAsInteger() != extent ||
            __builtin_mul_overflow(extent, *n, &extent)) {
          extent = 0;
          break;
        }
      }
      if (!extent || (*shape)[shape->size() - 2].getAsInteger() != qk.tokens ||
          qk.tokens % 2)
        continue;
      auto group = record->getInteger("alias_group");
      bool unsafe =
          space
              .walk([&](Operation *op) {
                auto code = op->getName().stripDialect();
                if (code != "slice_write" && !isa<memref::StoreOp>(op))
                  return WalkResult::advance();
                auto out = dyn_cast<BlockArgument>(op->getOperand(1));
                auto *f =
                    out && out.getArgNumber() < facts.arguments.size()
                        ? facts.arguments[out.getArgNumber()].getAsObject()
                        : nullptr;
                auto outGroup = f ? f->getInteger("alias_group") : std::nullopt;
                return !group || !outGroup || *group < 0 || *outGroup < 0 ||
                               *outGroup == *group
                           ? WalkResult::interrupt()
                           : WalkResult::advance();
              })
              .wasInterrupted();
      if (!unsafe && extent <= INT64_MAX / 2)
        result = OrderedValuePanel{
            add, read, *shape->back().getAsInteger(), qk.tokens, extent};
    }
  });
  return result;
}
} // namespace intentdsl::cpu
