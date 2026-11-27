#include "intentdsl/levels/execution/cpu/ir/FoldProducts.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/NativeDot.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace intentdsl::cpu {
// Match independent vector sums of weighted input rows. Each weight is a
// scalar expression over row dot products. No intermediate rounding is moved.
std::optional<FoldProducts> foldProducts(mlir::Operation &space,
                                         mlir::func::FuncOp fn,
                                         const CallFacts &facts) {
  using namespace mlir;
  Block &body = space.getRegion(0).front();
  auto loops = body.getOps<scf::ForOp>();
  if (!llvm::hasSingleElement(loops) || body.getNumArguments() == 0)
    return {};
  auto loop = *loops.begin();
  auto owner = body.getArguments().back();
  Value iv = loop.getInductionVar();
  if (!loop->hasAttr("kir.reassociate") || !loop.getNumResults() ||
      !matchPattern(loop.getStep(), m_One()))
    return {};
  // Interchanging owners must not expose a read to another owner's store.
  llvm::SmallDenseSet<int64_t> outputs, inputs;
  for (auto &op : body) {
    if (op.getName().stripDialect() != "slice_write")
      continue;
    Value view = op.getOperand(1);
    auto arg = dyn_cast<BlockArgument>(view);
    if (!arg || arg.getOwner() != &fn.front() ||
        arg.getArgNumber() >= facts.arguments.size() ||
        op.getAttrOfType<IntegerAttr>("axis").getInt() !=
            cast<MemRefType>(view.getType()).getRank() - 1 ||
        op.getNumOperands() != body.getNumArguments() + 2 ||
        !llvm::equal(op.getOperands().drop_front(2), body.getArguments()) ||
        op.getBlock() != loop->getBlock() || op.isBeforeInBlock(loop))
      return {};
    auto *record = facts.arguments[arg.getArgNumber()].getAsObject();
    auto group = record ? record->getInteger("alias_group") : std::nullopt;
    if (!group || !outputs.insert(*group).second)
      return {};
  }
  if (outputs.empty())
    return {};
  for (auto arg : fn.getArguments()) {
    if (!isa<MemRefType>(arg.getType()))
      continue;
    if (arg.getArgNumber() >= facts.arguments.size())
      return {};
    auto *record = facts.arguments[arg.getArgNumber()].getAsObject();
    auto group = record ? record->getInteger("alias_group") : std::nullopt;
    if (!group)
      return {};
    bool writes = false;
    for (auto *u : arg.getUsers()) {
      if (u->getName().stripDialect() == "slice_write")
        writes = true;
      else if (isa<memref::LoadOp>(u) ||
               u->getName().stripDialect() == "slice_read")
        inputs.insert(*group);
      else if (llvm::any_of(u->getResultTypes(),
                            [](Type t) { return isa<BaseMemRefType>(t); }))
        return {}; // A derived view can hide reads of a written buffer.
    }
    if (writes) {
      auto *dims = record->getArray("shape"), *st = record->getArray("strides");
      if (!dims || !st || st->size() != dims->size())
        return {};
      int64_t span = 1;
      for (int i = int(dims->size()) - 1; i >= 0; --i) {
        auto n = (*dims)[i].getAsInteger(), stride = (*st)[i].getAsInteger();
        if (!n || !stride || *n < 1 || (*n > 1 && *stride < span) ||
            __builtin_mul_overflow(std::max<int64_t>(span, *stride), *n, &span))
          return {};
      }
    }
  }
  for (auto g : inputs)
    if (outputs.contains(g))
      return {};
  for (auto &op : body)
    if (&op != loop.getOperation() &&
        op.getName().stripDialect() != "slice_write" &&
        op.getName().stripDialect() != "slice_read" &&
        op.getName().stripDialect() != "end" && !isa<memref::LoadOp>(op) &&
        !isMemoryEffectFree(&op))
      return {};
  bool prefix = matchPattern(loop.getLowerBound(), m_Zero());
  if (!prefix && loop.getLowerBound() != owner)
    return {};
  auto abiAttr = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abiAttr)
    return {};
  auto abi = llvm::json::parse(abiAttr.getValue());
  if (!abi) {
    llvm::consumeError(abi.takeError());
    return {};
  }
  auto *slots = abi->getAsArray();
  std::array<std::string, 3> logicalAxes;
  auto axisMatches = [&](unsigned role, BlockArgument arg, size_t pos) {
    if (!slots || arg.getArgNumber() >= slots->size())
      return false;
    auto *slot = (*slots)[arg.getArgNumber()].getAsObject();
    auto *dims = slot ? slot->getArray("dims") : nullptr;
    if (!dims || pos >= dims->size())
      return false;
    std::string label;
    llvm::raw_string_ostream stream(label);
    unsigned occurrence = 0;
    for (size_t i = 0; i < pos; ++i)
      occurrence += (*dims)[i] == (*dims)[pos];
    stream << (*dims)[pos] << "/" << occurrence;
    if (!logicalAxes[role].empty())
      return logicalAxes[role] == label;
    if (llvm::is_contained(logicalAxes, label))
      return false;
    logicalAxes[role] = label;
    return true;
  };
  int64_t depth = 0, steps = 0;
  Type inputType;
  SmallVector<Operation *> products;
  auto row = [&](Value v, bool member) {
    auto ext = v.getDefiningOp<arith::ExtFOp>();
    auto *read = ext ? ext.getIn().getDefiningOp() : nullptr;
    if (!read || read->getName().stripDialect() != "slice_read")
      return false;
    auto arg = dyn_cast<BlockArgument>(read->getOperand(0));
    if (!arg || arg.getOwner() != &fn.front())
      return false;
    auto *record = facts.arguments[arg.getArgNumber()].getAsObject();
    auto *shape = record ? record->getArray("shape") : nullptr;
    auto *strides = record ? record->getArray("strides") : nullptr;
    auto mt = cast<MemRefType>(arg.getType());
    if (!shape ||
        (!mt.getElementType().isF16() && !mt.getElementType().isBF16()) ||
        (inputType && mt.getElementType() != inputType) || mt.getRank() < 2 ||
        read->getAttrOfType<IntegerAttr>("axis").getInt() != mt.getRank() - 1 ||
        read->getOperands().back() != (member ? iv : owner) ||
        !llvm::all_of(read->getOperands().drop_front().drop_back(),
                      [&](Value i) {
                        return isa<BlockArgument>(i) &&
                               i.getParentBlock() == &body && i != owner;
                      }))
      return false;
    if (shape->size() != size_t(mt.getRank()) || !strides ||
        strides->size() != shape->size())
      return false;
    auto last = shape->size() - 1;
    int64_t d = (*shape)[last].getAsInteger().value_or(-1);
    if (d <= 0 || d % 16 || (depth && depth != d) ||
        (*strides)[last].getAsInteger() != 1 ||
        (*strides)[last - 1].getAsInteger() != d)
      return false;
    if (!getElementTypeOrSelf(v.getType()).isF32() ||
        !axisMatches(member ? 1 : 0, arg, last - 1) ||
        !axisMatches(2, arg, last))
      return false;
    depth = d;
    inputType = mt.getElementType();
    if (member) {
      auto extent = (*shape)[last - 1].getAsInteger().value_or(-1);
      if (extent <= 0 || (steps && steps != extent))
        return false;
      steps = extent;
    }
    return true;
  };
  bool usesDot = false;
  auto weight = [&](auto &self, Value v) -> bool {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg.getOwner() == &fn.front() && v.getType().isF32();
    auto *op = v.getDefiningOp();
    if (!op)
      return false;
    if (op->getName().stripDialect() == "lane_sum") {
      usesDot = true;
      if (!llvm::is_contained(products, op))
        products.push_back(op);
      auto mul = op->getOperand(0).getDefiningOp<arith::MulFOp>();
      return mul && kir::hasFastMath(op, arith::FastMathFlags::reassoc) &&
             ((row(mul.getLhs(), false) && row(mul.getRhs(), true)) ||
              (row(mul.getRhs(), false) && row(mul.getLhs(), true)));
    }
    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      auto ix = ld.getIndices();
      auto arg = dyn_cast<BlockArgument>(ld.getMemref());
      if (!arg || arg.getOwner() != &fn.front() || ix.empty())
        return false;
      auto *strides =
          facts.arguments[arg.getArgNumber()].getAsObject()->getArray(
              "strides");
      return strides && strides->size() == ix.size() &&
             (*strides)[ix.size() - 1].getAsInteger() == 1 &&
             axisMatches(ix.back() == iv ? 1 : 0, arg, ix.size() - 1) &&
             v.getType().isF32() && (ix.back() == iv || ix.back() == owner) &&
             llvm::all_of(ix.drop_back(), [&](Value i) {
               return isa<BlockArgument>(i) && i.getParentBlock() == &body &&
                      i != owner;
             });
    }
    if (!isa<arith::ConstantOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
             arith::TruncFOp, arith::ExtFOp, math::Exp2Op>(op) ||
        (!v.getType().isF32() && v.getType() != inputType))
      return false;
    if (isa<math::Exp2Op>(op) &&
        !kir::hasFastMath(op, arith::FastMathFlags::afn))
      return false;
    return llvm::all_of(op->getOperands(),
                        [&](Value x) { return self(self, x); });
  };
  for (auto [i, v] :
       llvm::enumerate(loop.getBody()->getTerminator()->getOperands())) {
    auto *init = loop.getInitArgs()[i].getDefiningOp();
    auto add = v.getDefiningOp<arith::AddFOp>();
    if (!init || init->getName().stripDialect() != "lane_splat" ||
        !matchPattern(init->getOperand(0), m_PosZeroFloat()) || !add ||
        add.getLhs() != loop.getRegionIterArgs()[i] ||
        !kir::hasFastMath(add, arith::FastMathFlags::reassoc))
      return {};
    auto mul = add.getRhs().getDefiningOp<arith::MulFOp>();
    if (!mul || !row(mul.getLhs(), true))
      return {};
    auto *factor = mul.getRhs().getDefiningOp();
    usesDot = false;
    if (!factor || factor->getName().stripDialect() != "lane_splat" ||
        !weight(weight, factor->getOperand(0)) || !usesDot)
      return {};
    // F16 x F16 is exact in F32, including the subnormal inputs.
    auto widen = factor->getOperand(0).getDefiningOp<arith::ExtFOp>();
    if (!widen || widen.getIn().getType() != inputType ||
        !widen.getIn().getDefiningOp<arith::TruncFOp>())
      return {};
    auto initView = dyn_cast<BlockArgument>(init->getOperand(1));
    auto initAxis = init->getAttrOfType<IntegerAttr>("axis").getInt();
    auto *initShape = facts.shapeOf(init->getOperand(1));
    if (!initView || initView.getOwner() != &fn.front() || !initShape ||
        initAxis < 0 || size_t(initAxis) >= initShape->size() ||
        (*initShape)[initAxis].getAsInteger() != depth ||
        !axisMatches(2, initView, initAxis))
      return {};
    if (inputType.isBF16() &&
        (!kir::permitsBf16Dot(mul) || !kir::permitsBf16Dot(add)))
      return {};
  }
  if (llvm::any_of(loop.getBody()->without_terminator(), [](Operation &op) {
        return !isMemoryEffectFree(&op) && !isa<memref::LoadOp>(op) &&
               op.getName().stripDialect() != "slice_read";
      }))
    return {};
  bool rectangular = false;
  if (prefix) {
    auto dim = loop.getUpperBound().getDefiningOp<memref::DimOp>();
    if (dim && dim.getConstantIndex()) {
      const auto *shape = facts.shapeOf(dim.getSource());
      rectangular =
          shape && (*shape)[*dim.getConstantIndex()].getAsInteger() == steps;
    } else
      rectangular = getConstantIntValue(loop.getUpperBound()) == steps;
  }
  if (inputType.isBF16() &&
      (!rectangular || llvm::any_of(products, [](Operation *dot) {
        return !kir::permitsBf16Dot(dot) ||
               !kir::permitsBf16Dot(dot->getOperand(0).getDefiningOp());
      })))
    return {};
  if (!prefix) {
    auto dim = loop.getUpperBound().getDefiningOp<memref::DimOp>();
    if (!dim || !dim.getConstantIndex())
      return {};
    auto *s = facts.shapeOf(dim.getSource());
    if (!s || (*s)[*dim.getConstantIndex()].getAsInteger() != steps)
      return {};
  }
  if (products.empty())
    return {};
  Value bound = prefix ? loop.getUpperBound() : loop.getLowerBound();
  llvm::SmallPtrSet<Operation *, 16> needed;
  auto collect = [&](auto &self, Value v) -> void {
    auto *op = v.getDefiningOp();
    if (!op || op->getBlock() != &body || !needed.insert(op).second)
      return;
    op->walk([&](Operation *inner) {
      for (Value x : inner->getOperands())
        self(self, x);
    });
  };
  collect(collect, bound);
  SmallVector<Operation *> boundOps;
  for (auto &op : body)
    if (needed.contains(&op))
      boundOps.push_back(&op);
  bool monotone =
      bound == owner ||
      isa_and_nonnull<memref::DimOp, arith::ConstantOp>(bound.getDefiningOp());
  if (auto select = bound.getDefiningOp<scf::IfOp>()) {
    auto cmp = select.getCondition().getDefiningOp<arith::CmpIOp>();
    auto add = select.thenYield().getOperand(0).getDefiningOp<arith::AddIOp>();
    Value cap = select.elseYield().getOperand(0);
    // min(owner+1, extent), with the same bound on both sides.
    if (cmp && add && cmp.getPredicate() == arith::CmpIPredicate::slt &&
        cmp.getLhs() == owner && add.getLhs() == owner &&
        matchPattern(add.getRhs(), m_One())) {
      auto a = cmp.getRhs().getDefiningOp<memref::DimOp>(),
           b = cap.getDefiningOp<memref::DimOp>();
      monotone =
          cmp.getRhs() == cap ||
          (a && b && a.getSource() == b.getSource() && a.getConstantIndex() &&
           a.getConstantIndex() == b.getConstantIndex());
    }
  }
  return FoldProducts{loop,
                      owner,
                      depth,
                      steps,
                      prefix,
                      std::move(products),
                      std::move(boundOps),
                      monotone,
                      inputType.isBF16(),
                      rectangular};
}
bool fitsProductPanel(const FoldProducts &p, int64_t rows, int64_t keys,
                      int64_t extent, const RowMachine &m) {
  if (rows < 8 || rows > 128 || keys < 16 || keys > 256 ||
      !llvm::isPowerOf2_64(rows) || !llvm::isPowerOf2_64(keys) ||
      extent < rows || extent % rows || p.steps % keys || p.steps < 2 * keys ||
      !llvm::isPowerOf2_64(p.depth))
    return false;
  double workingBytes =
      4. * rows *
          (keys * p.products.size() + p.depth * p.loop->getNumResults()) +
      2. * keys * (p.depth + rows * p.loop->getNumResults());
  return workingBytes <= m.l2CacheBytes / 2.;
}
} // namespace intentdsl::cpu
