#include "intentdsl/levels/execution/block/ir/RowProgram.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "../capabilities/Index.h"
#include "intentdsl/levels/execution/block/ir/ScaleDispatch.h"
#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/KeyReduction.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include <set>

#include "Dialect.h.inc"
namespace CI_UNIT_NAMESPACE {
mlir::Value scaledPanels(intentdsl::block::Tiles &,intentdsl::block::MatrixAccess &);
using namespace mlir;
using namespace compilerinfra;
void registerIR(DialectRegistry &);
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
llvm::Error describeArtifact(ModuleOp, llvm::raw_pwrite_stream &);
TerminalPrinter cutilePrinter();
void registerContracts(Unit &);
Value nativeMatrixRead(intentdsl::block::Tiles &,
                       intentdsl::block::MatrixAccess &, bool, Value, Value);
Value nativeWindowPanels(intentdsl::block::Tiles &,
                         intentdsl::block::MatrixAccess &, Value &, Value &,
                         Value &);
bool nativeMatrixWrite(intentdsl::block::Tiles &,
                       intentdsl::block::MatrixAccess &, Value, Value,
                       ArrayRef<Value>);
LogicalResult verifyArrayGeometry(func::FuncOp,const TargetProfile &);
intentdsl::block::OwnerAccess
nativeOwnerViews(const intentdsl::block::OwnerMachine *);
namespace {
std::optional<int64_t> ownerRectangleAxis(Operation *op, func::FuncOp fn) {
  auto kind = op->getName().stripDialect();
  if (kind != "slice_read" && kind != "slice_write")
    return std::nullopt;
  auto &space = fn.front().front();
  auto pair = space.getAttrOfType<IntegerAttr>("block.pair_axis");
  if (!pair || space.hasAttr("ct_src.matrix_rows") ||
      op->getParentOp() != &space)
    return std::nullopt;
  auto &bb = space.getRegion(0).front();
  auto coords = op->getOperands().drop_front(kind == "slice_write" ? 2 : 1);
  if (coords.empty() ||
      op->getAttrOfType<IntegerAttr>("axis").getInt() !=
          int64_t(coords.size()))
    return std::nullopt;
  Value pairArg = bb.getArgument(pair.getInt());
  int64_t pairPos = -1;
  llvm::SmallDenseSet<Value, 4> seen;
  for (auto [idx, v] : llvm::enumerate(coords)) {
    auto arg = dyn_cast<BlockArgument>(v);
    if (!arg || arg.getOwner() != &bb || !seen.insert(v).second)
      return std::nullopt;
    if (v == pairArg)
      pairPos = idx;
  }
  if (pairPos < 0)
    return std::nullopt;
  auto view =
      cast<BlockArgument>(op->getOperand(kind == "slice_write" ? 1 : 0));
  auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(view.getArgNumber(),
                                                      "block.shape");
  auto ownerTile =
      space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  if (!shape || shape.size() != coords.size() + 1 || ownerTile <= 0 ||
      shape[pairPos] < ownerTile || shape.asArrayRef().back() <= 0 ||
      !llvm::isPowerOf2_64(shape.asArrayRef().back()))
    return std::nullopt;
  auto tailProof = compilerinfra::proveTailAccess(op);
  if (!tailProof.proven ||
      tailProof.kind != compilerinfra::TailKind::ExactInBounds)
    return std::nullopt;
  return pairPos;
}
struct SignedSpan {
  int64_t lo, hi;
};
bool canNarrowCutileAddress(Operation &op) {
  auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
  if (!strides || llvm::any_of(strides.asArrayRef(), [](int64_t s) {
        return s < 0 || s > INT32_MAX;
      }))
    return false;
  Operation *scope = op.getParentOfType<func::FuncOp>();
  if (!scope)
    scope = op.getParentOp();
  compilerinfra::IntegerRangeContext intRanges(scope);
  if (!intRanges.valid())
    return false;
  llvm::DenseMap<Value, std::optional<SignedSpan>> memo;
  auto fit64 = [](__int128 lo, __int128 hi) -> std::optional<SignedSpan> {
    if (lo > hi || lo < INT64_MIN || hi > INT64_MAX)
      return std::nullopt;
    return SignedSpan{int64_t(lo), int64_t(hi)};
  };
  auto floorDiv = [](__int128 a, __int128 d) -> __int128 {
    __int128 q = a / d, r = a % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
  };
  std::function<std::optional<SignedSpan>(Value)> range =
      [&](Value v) -> std::optional<SignedSpan> {
    if (auto it = memo.find(v); it != memo.end())
      return it->second;
    memo[v] = std::nullopt;
    std::optional<SignedSpan> res;
    if (!isa<ShapedType>(v.getType())) {
      compilerinfra::AddressProofOptions opts;
      opts.targetBitwidth = 32;
      opts.isSigned = true;
      auto proof = compilerinfra::proveNarrowAddress(v, opts, &intRanges);
      if (proof.proven) {
        if (auto cr = intRanges.query(v))
          res = fit64(cr->smin().getSExtValue(), cr->smax().getSExtValue());
      }
      if (res) {
        memo[v] = res;
        return res;
      }
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;
    if (auto c = dyn_cast<arith::ConstantOp>(def)) {
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
          ia && (ia.getType().isIndex() ||
                 (ia.getType().isIntOrFloat() &&
                  ia.getType().getIntOrFloatBitWidth() <= 64))) {
        int64_t val = ia.getInt();
        res = SignedSpan{val, val};
      } else if (auto da = dyn_cast<DenseIntElementsAttr>(c.getValue());
                 da && da.isSplat() &&
                 da.getElementType().getIntOrFloatBitWidth() <= 64) {
        int64_t val = da.getSplatValue<APInt>().getSExtValue();
        res = SignedSpan{val, val};
      }
      memo[v] = res;
      return res;
    }
    StringRef n = def->getName().getStringRef();
    if (n == "ct_src.tile_axis") {
      int64_t sz = cast<RankedTensorType>(v.getType()).getDimSize(0);
      if (sz > 0)
        res = SignedSpan{0, sz - 1};
    } else if (n == "ct_src.tile_broadcast" || n == "ct_src.tile_prefix" ||
               n == "ct_src.drop_unit_owner" || n == "ct_src.reshape" ||
               isa<arith::IndexCastOp>(def)) {
      res = range(def->getOperand(0));
    } else if (auto es = dyn_cast<arith::ExtSIOp>(def)) {
      res = range(es.getIn());
      if (auto it =
              dyn_cast<IntegerType>(getElementTypeOrSelf(es.getIn().getType()));
          it && it.getWidth() > 0 && it.getWidth() <= 32) {
        int64_t lo = -(int64_t(1) << (it.getWidth() - 1));
        int64_t hi = (int64_t(1) << (it.getWidth() - 1)) - 1;
        res = res ? SignedSpan{std::max(res->lo, lo), std::min(res->hi, hi)}
                  : SignedSpan{lo, hi};
      }
    } else if (auto eu = dyn_cast<arith::ExtUIOp>(def)) {
      if (auto it =
              dyn_cast<IntegerType>(getElementTypeOrSelf(eu.getIn().getType()));
          it && it.getWidth() > 0 && it.getWidth() < 32) {
        res = SignedSpan{0, (int64_t(1) << it.getWidth()) - 1};
      } else if (auto rem = eu.getIn().getDefiningOp<arith::RemUIOp>()) {
        auto bR = range(rem.getRhs());
        if (bR && bR->lo > 0 && bR->hi <= INT32_MAX)
          res = SignedSpan{0, bR->hi - 1};
      }
    } else if (n == "ct_src.tile_reduce") {
      auto alg = def->getAttrOfType<StringAttr>("algebra");
      if (alg && (alg.getValue() == "min" || alg.getValue() == "max"))
        res = range(def->getOperand(0));
    } else if (isa<arith::RemSIOp>(def)) {
      auto a = range(def->getOperand(0)), b = range(def->getOperand(1));
      if (a && b && b->lo > 0 && b->lo == b->hi)
        res = a->lo >= 0 ? SignedSpan{0, std::min(a->hi, b->lo - 1)}
                         : SignedSpan{-(b->lo - 1), b->lo - 1};
    } else if (isa<arith::RemUIOp>(def)) {
      auto b = range(def->getOperand(1));
      if (b && b->lo > 0 && b->lo == b->hi && b->lo <= INT32_MAX) {
        auto a = range(def->getOperand(0));
        res = SignedSpan{0, (a && a->lo >= 0) ? std::min(a->hi, b->lo - 1)
                                              : (b->lo - 1)};
      }
    } else if (isa<arith::DivUIOp>(def)) {
      auto a = range(def->getOperand(0)), b = range(def->getOperand(1));
      if (a && b && a->lo >= 0 && b->lo > 0 && b->lo == b->hi)
        res = SignedSpan{a->lo / b->lo, a->hi / b->lo};
    } else if (isa<arith::FloorDivSIOp>(def)) {
      auto a = range(def->getOperand(0)), b = range(def->getOperand(1));
      if (a && b && b->lo > 0 && b->lo == b->hi)
        res = fit64(floorDiv(a->lo, b->lo), floorDiv(a->hi, b->lo));
    } else if (auto sel = dyn_cast<arith::SelectOp>(def)) {
      auto a = range(sel.getTrueValue()), b = range(sel.getFalseValue());
      if (a && b)
        res = SignedSpan{std::min(a->lo, b->lo), std::max(a->hi, b->hi)};
    } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::MinSIOp>(
                   def)) {
      auto a = range(def->getOperand(0)), b = range(def->getOperand(1));
      if (a && b) {
        __int128 al = a->lo, ah = a->hi, bl = b->lo, bh = b->hi;
        if (isa<arith::MinSIOp>(def))
          res = fit64(std::min(al, bl), std::min(ah, bh));
        else if (isa<arith::AddIOp>(def))
          res = fit64(al + bl, ah + bh);
        else if (isa<arith::SubIOp>(def))
          res = fit64(al - bh, ah - bl);
        else {
          auto ex = std::minmax({al * bl, al * bh, ah * bl, ah * bh});
          res = fit64(ex.first, ex.second);
        }
      }
    }
    if (!res) {
      Type elemTy = getElementTypeOrSelf(v.getType());
      if (auto it = dyn_cast<IntegerType>(elemTy);
          it && it.isSignless() && it.getWidth() > 0 && it.getWidth() <= 32) {
        unsigned w = it.getWidth();
        res = SignedSpan{-(int64_t(1) << (w - 1)),
                         (int64_t(1) << (w - 1)) - 1};
      }
    }
    memo[v] = res;
    return res;
  };
  compilerinfra::AddressProofOptions narrowOpts;
  narrowOpts.targetBitwidth = 32;
  narrowOpts.isSigned = true;
  if (llvm::none_of(op.getOperands().drop_front(),
                    [](Value v) { return isa<ShapedType>(v.getType()); }) &&
      compilerinfra::proveNarrowAddress(&op, narrowOpts, &intRanges).proven)
    return true;
  for (Value coord : op.getOperands().drop_front()) {
    if (!isa<ShapedType>(coord.getType()) &&
        compilerinfra::proveNarrowAddress(coord, narrowOpts, &intRanges).proven)
      continue;
    auto r = range(coord);
    if (!r || r->lo < INT32_MIN || r->hi > INT32_MAX)
      return false;
  }
  return true;
}
bool isBlockScaledFp4Quant(Operation *op) {
  if (op->getName().stripDialect() != "quant_contract")
    return false;
  const auto *spec = intentdsl::kir::quantFormatSpec(*op);
  return spec &&
         spec->roleLayout == intentdsl::kir::QuantRoleLayout::BlockScaledPair &&
         spec->valueCode == intentdsl::kir::QuantValueCode::Fp4E2M1 &&
         spec->scaleCode == intentdsl::kir::QuantScaleCode::E8M0;
}
DenseI64ArrayAttr adjacentScaleGroups(Operation *op,func::FuncOp fn){
  int64_t groupSpan = 32;
  if (const auto *spec = intentdsl::kir::quantFormatSpec(*op))
    if (spec->roleLayout == intentdsl::kir::QuantRoleLayout::BlockScaledPair)
      groupSpan = spec->recordBytes;
  SmallVector<int64_t> slots;
  for(unsigned pos:{0u,2u}){
    auto i=cast<BlockArgument>(op->getOperand(pos)).getArgNumber();
    auto s=fn.getArgAttrOfType<DenseI64ArrayAttr>(i,"block.strides");
    int axis=pos==0?1:0;
    if(s && s.size()==3 && s[axis]==groupSpan*s[axis+1])slots.push_back(i);
  }
  return Builder(fn.getContext()).getDenseI64ArrayAttr(slots);
}
bool rectangularSlab(Operation *op,func::FuncOp fn){
  auto arg=dyn_cast<BlockArgument>(op->getOperand(0));
  auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  if(!owner || owner.getInt()>1 || !arg || arg.getOwner()!=&fn.front())return false;
  if(op->getNumOperands()<3)return false;
  auto *r=op->getOperand(1).getDefiningOp(),*c=op->getOperand(2).getDefiningOp();
  if(!r || r->getName().stripDialect()!="rows" || !c || c->getName().stripDialect()!="columns")return false;
  auto shape=fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),"ct_src.shape");
  return shape && shape.size()==3 && llvm::isPowerOf2_64(shape[2]) &&
    llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),[](Attribute row){
      return llvm::isPowerOf2_64(cast<DictionaryAttr>(row).getAs<IntegerAttr>("width").getInt());
    });
}
bool usesCutileFtzAttr(Operation *op) {
  StringRef full = op->getName().getStringRef();
  return full == "ct_src.div" || full == "arith.divf" || full == "math.exp2" ||
         (full == "math.exp" &&
          intentdsl::kir::hasFastMath(op, arith::FastMathFlags::afn));
}
void computeCutileLoopAttrs(scf::ForOp loop, StringAttr &kindOut,
                            ArrayAttr &boundsOut) {
  Builder b(loop.getContext());
  auto constInt = [](Value v) -> std::optional<int64_t> {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return std::nullopt;
    auto ia = dyn_cast<IntegerAttr>(cst.getValue());
    if (!ia || ia.getValue().getSignificantBits() > 64)
      return std::nullopt;
    return ia.getInt();
  };
  auto constI32 = [](Value v) -> std::optional<int64_t> {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return std::nullopt;
    auto ia = dyn_cast<IntegerAttr>(cst.getValue());
    if (!ia || !ia.getValue().isSignedIntN(32))
      return std::nullopt;
    return ia.getInt();
  };
  Value ops[3] = {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
  bool hasNestedLoop =
      loop.getBody()
          ->walk([](scf::ForOp) { return WalkResult::interrupt(); })
          .wasInterrupted();
  int64_t carriedWords = 0;
  for (Type rt : loop.getResultTypes()) {
    if (auto st = dyn_cast<ShapedType>(rt); st && st.hasStaticShape())
      carriedWords +=
          st.getNumElements() *
          std::max<int64_t>(1, llvm::divideCeil(st.getElementTypeBitWidth(), 32));
  }
  bool dynamicBounds = !constInt(ops[0]) || !constInt(ops[1]);
  if (loop.getInductionVar().getType().isInteger(32) &&
      !(hasNestedLoop && dynamicBounds && carriedWords >= 16384)) {
    kindOut = b.getStringAttr("range_i32");
    SmallVector<Attribute, 3> bounds;
    for (Value v : ops) {
      if (auto c = constInt(v))
        bounds.push_back(b.getI64IntegerAttr(*c));
      else
        bounds.push_back(b.getUnitAttr());
    }
    boundsOut = b.getArrayAttr(bounds);
    return;
  }
  auto lb = constI32(ops[0]), ub = constI32(ops[1]), st = constI32(ops[2]);
  if (lb && ub && st) {
    kindOut = b.getStringAttr("range_const");
    boundsOut = b.getArrayAttr({b.getI64IntegerAttr(*lb),
                                b.getI64IntegerAttr(*ub),
                                b.getI64IntegerAttr(*st)});
    return;
  }
  kindOut = b.getStringAttr("while");
  boundsOut = {};
}
ArrayAttr computeCutileExtraParams(func::FuncOp fn) {
  Builder b(fn.getContext());
  SmallVector<Attribute> extra;
  auto convention =
      fn.front().front().getAttrOfType<StringAttr>("calling_convention");
  bool matrix = convention && convention.getValue() == "matrix";
  bool direct = convention || intentdsl::kir::orderedOwners(fn);
  if (!direct) {
    int rank = cast<MemRefType>(
                   fn.getArgument(intentdsl::domainSlot(fn)).getType())
                   .getRank();
    if (rank > 2)
      for (int i = 0; i < rank - 1; ++i)
        extra.push_back(
            b.getStringAttr("D" + std::to_string(i) + ":ct.Constant[int]"));
    if (fn->getAttr("block.grid") == intentdsl::block::grid(b, true))
      extra.push_back(b.getStringAttr("COLUMN_TILES:ct.Constant[int]"));
  }
  extra.push_back(b.getStringAttr(matrix   ? "BM:ct.Constant[int]"
                                  : direct ? "ROWS:ct.Constant[int]"
                                           : "R:ct.Constant[int]"));
  extra.push_back(b.getStringAttr(matrix   ? "BN:ct.Constant[int]"
                                  : direct ? "WIDTH:ct.Constant[int]"
                                           : "W:ct.Constant[int]"));
  return b.getArrayAttr(extra);
}
bool isCutileCarryOwner(Operation *op) {
  return isa<scf::ForOp, scf::IfOp>(op) ||
         op->getName().getStringRef() == "ct_src.tiles";
}
bool isCutileCarryYield(Operation *op) {
  return isa<scf::YieldOp>(op) ||
         op->getName().getStringRef() == "ct_src.yield";
}
void stampCutileCarryIds(func::FuncOp fn) {
  Builder b(fn.getContext());
  int64_t nextId = 0;
  fn.walk([&](Operation *op) {
    if (isCutileCarryOwner(op) && op->getNumResults() > 0) {
      auto idAttr = b.getI64IntegerAttr(nextId++);
      op->setAttr("ct_src.carry_id", idAttr);
      for (Region &reg : op->getRegions()) {
        for (Block &blk : reg) {
          if (blk.empty())
            continue;
          Operation &term = blk.back();
          if (isCutileCarryYield(&term) && term.getNumOperands() > 0)
            term.setAttr("ct_src.carry_id", idAttr);
        }
      }
    }
  });
}
LogicalResult verifyCutileCarryIds(func::FuncOp fn) {
  llvm::DenseSet<int64_t> seenIds;
  auto res = fn.walk([&](Operation *op) {
    if (isCutileCarryOwner(op) && op->getNumResults() > 0) {
      auto idAttr = op->getAttrOfType<IntegerAttr>("ct_src.carry_id");
      if (!idAttr || !seenIds.insert(idAttr.getInt()).second) {
        op->emitError("missing or duplicate ct_src.carry_id");
        return WalkResult::interrupt();
      }
    }
    if (isCutileCarryYield(op) && op->getNumOperands() > 0) {
      Operation *parent = op->getParentOp();
      auto idAttr = op->getAttrOfType<IntegerAttr>("ct_src.carry_id");
      if (!parent || !isCutileCarryOwner(parent) || !idAttr ||
          parent->getAttr("ct_src.carry_id") != idAttr ||
          parent->getNumResults() != op->getNumOperands()) {
        op->emitError("ct_src.carry_id on yield disagrees with enclosing op");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return success(!res.wasInterrupted());
}
bool broadcastInner(Value viewArg, func::FuncOp fn) {
  auto arg = dyn_cast<BlockArgument>(viewArg);
  if (!arg || arg.getOwner() != &fn.front())
    return false;
  auto strides =
      fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(), "ct_src.strides");
  auto shape =
      fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(), "ct_src.shape");
  return strides && shape && !strides.empty() &&
         strides.size() == shape.size() && strides.asArrayRef().back() == 0 &&
         shape.asArrayRef().back() > 0;
}
bool alignedProjectedView(Value viewArg, func::FuncOp fn,
                          const intentdsl::CallFacts &facts, bool permitTma) {
  if (!permitTma)
    return false;
  auto arg = dyn_cast<BlockArgument>(viewArg);
  if (!arg || arg.getOwner() != &fn.front())
    return false;
  auto view = cast<MemRefType>(arg.getType());
  int64_t elemBytes = llvm::divideCeil(view.getElementTypeBitWidth(), 8);
  unsigned slot = arg.getArgNumber();
  if (auto map = fn.getArgAttrOfType<AffineMapAttr>(slot, "kir.indexing")) {
    auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.strides");
    auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.shape");
    auto src = fn.getArgAttrOfType<IntegerAttr>(slot, "kir.source");
    if (!strides || !shape || strides.size() != size_t(view.getRank()) ||
        shape.size() != strides.size() || strides.empty() ||
        strides.asArrayRef().back() != 1 || !src || src.getInt() < 0 ||
        size_t(src.getInt()) >= facts.arguments.size())
      return false;
    for (int64_t st : strides.asArrayRef().drop_back())
      if (st <= 0 || (st * elemBytes) % 16 != 0)
        return false;
    auto *srcObj = facts.arguments[src.getInt()].getAsObject();
    auto *srcStrides = srcObj ? srcObj->getArray("strides") : nullptr;
    if (!srcStrides || srcStrides->size() != map.getValue().getNumResults())
      return false;
    int64_t offsetElems = 0;
    for (auto [i, expr] : llvm::enumerate(map.getValue().getResults())) {
      SmallVector<int64_t> factors(map.getValue().getNumDims() + 1, 0);
      auto st = (*srcStrides)[i].getAsInteger();
      if (!st || !intentdsl::coefficients(expr, factors))
        return false;
      int64_t term = 0;
      if (__builtin_mul_overflow(factors.back(), *st, &term) ||
          __builtin_add_overflow(offsetElems, term, &offsetElems))
        return false;
    }
    return offsetElems >= 0 && (offsetElems * elemBytes) % 16 == 0;
  }
  if (slot >= facts.arguments.size())
    return false;
  auto *obj = facts.arguments[slot].getAsObject();
  auto *strides = obj ? obj->getArray("strides") : nullptr;
  if (!strides || strides->size() != size_t(view.getRank()) || strides->empty())
    return false;
  if ((*strides)[strides->size() - 1].getAsInteger() != 1)
    return false;
  for (size_t i = 0; i + 1 < strides->size(); ++i) {
    auto st = (*strides)[i].getAsInteger();
    if (!st || *st <= 0 || (*st * elemBytes) % 16 != 0)
      return false;
  }
  return true;
}
bool panelPermitsTma(Operation *op, func::FuncOp fn,
                     const intentdsl::CallFacts &facts, bool permitTma) {
  if (!op->hasAttr("block.rectangular_tokens"))
    return false;
  auto *parent = op->getParentOp();
  auto span =
      parent ? parent->getAttrOfType<IntegerAttr>("block.contribution_width")
             : IntegerAttr();
  if (!span || span.getInt() <= 0 ||
      !alignedProjectedView(op->getOperand(0), fn, facts, permitTma))
    return false;
  auto view = cast<MemRefType>(op->getOperand(0).getType());
  int64_t elemBytes = llvm::divideCeil(view.getElementTypeBitWidth(), 8);
  auto arg = cast<BlockArgument>(op->getOperand(0));
  auto shape =
      fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(), "block.shape");
  if (!shape || shape.empty())
    return false;
  int64_t depth = shape.asArrayRef().back();
  return depth > 0 && llvm::isPowerOf2_64(depth) &&
         span.getInt() * depth * elemBytes <= 16384;
}
bool ownerRectPermitsTma(Operation *op, func::FuncOp fn,
                         const intentdsl::CallFacts &facts, bool permitTma) {
  if (op->getName().stripDialect() != "slice_read" ||
      !ownerRectangleAxis(op, fn) ||
      !alignedProjectedView(op->getOperand(0), fn, facts, permitTma))
    return false;
  auto rows = fn.front().front().getAttrOfType<IntegerAttr>("block.owner_tile");
  if (!rows || rows.getInt() < 32)
    return false;
  auto view = cast<MemRefType>(op->getOperand(0).getType());
  int64_t elemBytes = llvm::divideCeil(view.getElementTypeBitWidth(), 8);
  auto arg = cast<BlockArgument>(op->getOperand(0));
  auto shape =
      fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(), "block.shape");
  if (!shape || shape.empty())
    return false;
  int64_t depth = shape.asArrayRef().back();
  return depth > 0 && llvm::isPowerOf2_64(depth) &&
         rows.getInt() * depth * elemBytes <= 16384;
}
bool tileLoadStorePermitsTma(Value viewArg, func::FuncOp fn,
                             const intentdsl::CallFacts &facts,
                             bool permitTma) {
  if (!permitTma)
    return false;
  auto arg = dyn_cast<BlockArgument>(viewArg);
  if (!arg || arg.getOwner() != &fn.front())
    return false;
  int rank = cast<MemRefType>(arg.getType()).getRank();
  auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
  if (rank <= 2 && configs && !configs.empty() &&
      llvm::all_of(configs, [](Attribute row) {
        return cast<DictionaryAttr>(row).getAs<IntegerAttr>("rows").getInt() ==
               1;
      }))
    return false;
  unsigned slot = arg.getArgNumber();
  if (rank <= 2 && !fn.getArgAttr(slot, "kir.indexing"))
    return true;
  if (!alignedProjectedView(viewArg, fn, facts, permitTma))
    return false;
  if (rank > 2) {
    std::optional<int64_t> rowDim;
    if (auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.shape");
        shape && shape.size() == size_t(rank))
      rowDim = shape[rank - 2];
    else if (auto *sh = facts.shapeOf(arg); sh && sh->size() == size_t(rank))
      rowDim = (*sh)[rank - 2].getAsInteger();
    auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
    if (!rowDim || *rowDim <= 0 || !configs)
      return false;
    fn.getContext()
        ->getOrLoadDialect<arith::ArithDialect>();
    fn.getContext()
        ->getOrLoadDialect<memref::MemRefDialect>();
    fn.getContext()
        ->getOrLoadDialect<scf::SCFDialect>();
    OpBuilder b(&fn.front().front());
    Location loc = fn.getLoc();
    auto buf = b.create<memref::AllocOp>(
        loc, MemRefType::get({*rowDim}, b.getI8Type()));
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cHi = b.create<arith::ConstantIndexOp>(loc, *rowDim);
    bool allExact = true;
    for (Attribute row : configs) {
      int64_t r =
          cast<DictionaryAttr>(row).getAs<IntegerAttr>("rows").getInt();
      if (r <= 0 || *rowDim < r) {
        allExact = false;
        break;
      }
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointAfter(cHi.getDefiningOp());
      Value cStep = b.create<arith::ConstantIndexOp>(loc, r);
      auto loop = b.create<scf::ForOp>(loc, c0, cHi, cStep);
      b.setInsertionPointToStart(loop.getBody());
      auto sub = b.create<memref::SubViewOp>(
          loc, buf, ArrayRef<OpFoldResult>{loop.getInductionVar()},
          ArrayRef<OpFoldResult>{b.getIndexAttr(r)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      auto tp = compilerinfra::proveTailAccess(sub.getOperation());
      loop.erase();
      cStep.getDefiningOp()->erase();
      if (!tp.proven || tp.kind != compilerinfra::TailKind::ExactInBounds) {
        allExact = false;
        break;
      }
    }
    cHi.getDefiningOp()->erase();
    c0.getDefiningOp()->erase();
    buf.erase();
    if (!allExact)
      return false;
  }
  return true;
}
struct Lower : PassWrapper<Lower, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Lower)
  bool tma, scaledMma;
  intentdsl::CallFacts facts;
  Lower(const intentdsl::cutile::ArrayIndices &indices,const intentdsl::CallFacts &f) : tma(indices.permitTma),scaledMma(indices.scaledMma),facts(f) {}
  void runOnOperation() final {
    auto fn=*getOperation().getOps<func::FuncOp>().begin();
    if(tma)fn->setAttr("ct_src.tma",UnitAttr::get(fn.getContext()));
    if(failed(intentdsl::specializeProjectedLayout(fn,facts,"ct_src",true))){signalPassFailure();return;}
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "block.owner_space" &&
          op->hasAttr("block.pair_axis") &&
          op->getAttrOfType<IntegerAttr>("block.owner_tile").getInt() < 16)
        op->setAttr("ct_src.matrix_rows", Builder(&getContext()).getI64IntegerAttr(16));
    });
    getContext().getOrLoadDialect<intentdsl::cutile::TileSource>();
    getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->getName().getDialectNamespace() != "block")
        return;
      OpBuilder b(op);
      auto code = op->getName().stripDialect();
      if (code == "stripe_load" || code == "stripe_store") {
        auto view = op->getOperand(0);
        int rank = cast<MemRefType>(view.getType()).getRank();
        if (code == "stripe_load" && rank == 1) {
          OperationState so(op->getLoc(), "ct_src.stripe_owner_load");
          so.addOperands({view, op->getOperand(1), op->getOperand(2)});
          so.addAttribute("stride", op->getAttr("stride"));
          so.addTypes(op->getResultTypes());
          op->replaceAllUsesWith(b.create(so));
        } else {
          auto *r = op->getOperand(2).getDefiningOp(),
               *c = op->getOperand(3).getDefiningOp();
          OperationState st(op->getLoc(), code == "stripe_load"
                                              ? "ct_src.stripe_tile_load"
                                              : "ct_src.stripe_tile_store");
          st.addOperands(
              {view, op->getOperand(1), r->getOperand(0), c->getOperand(0)});
          if (code == "stripe_store")
            st.addOperands(op->getOperand(5));
          st.addAttribute("stride", op->getAttr("stride"));
          st.addAttribute("allow_tma", b.getBoolAttr(tma));
          st.addTypes(op->getResultTypes());
          op->replaceAllUsesWith(b.create(st));
        }
        op->erase();
        return;
      }
      bool access = code == "load" || code == "store";
      bool slab = access && op->getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis");
      OperationState s(op->getLoc(),
                       "ct_src." +
                           (slab ? (code=="load"?"gather_slab":"scatter_slab") : access ? ("tile_" + code).str() : code.str()));
      s.addAttributes(op->getAttrs());
      if (tma && op->hasAttr("block.page_extent"))
        s.addAttribute("ct_src.page_tma", b.getUnitAttr());
      if (panelPermitsTma(op, fn, facts, tma))
        s.addAttribute("ct_src.panel_tma", b.getUnitAttr());
      if (auto rectAxis = ownerRectangleAxis(op, fn)) {
        s.addAttribute("ct_src.owner_rectangle",
                       b.getI64IntegerAttr(*rectAxis));
        if (ownerRectPermitsTma(op, fn, facts, tma))
          s.addAttribute("ct_src.owner_tma", b.getUnitAttr());
      }
      if (code == "scaled_contract" || isBlockScaledFp4Quant(op)) {
        s.addAttribute("ct_src.flat_groups", adjacentScaleGroups(op, fn));
        if (code == "quant_contract" && scaledMma)
          s.addAttribute("ct_src.scaled_mma", b.getUnitAttr());
      }
      if (code == "sum_columns") {
        auto extent = op->getAttrOfType<IntegerAttr>("extent").getInt();
        s.addAttribute(
            "native_panel",
            b.getI64IntegerAttr(std::min<uint64_t>(
                256, llvm::PowerOf2Ceil(std::max<int64_t>(1, extent)))));
      }
      if (slab && rectangularSlab(op, fn)) {
        s.addAttribute("ct_src.rectangular", b.getUnitAttr());
        bool bcast = code == "load" && broadcastInner(op->getOperand(0), fn);
        if (bcast)
          s.addAttribute("ct_src.broadcast_inner", b.getUnitAttr());
        else if (alignedProjectedView(op->getOperand(0), fn, facts, tma))
          s.addAttribute("ct_src.slab_tma", b.getUnitAttr());
      }
      if (access && !slab) {
        auto view = op->getOperand(0);
        auto axes = intentdsl::block::projection(view.getType());
        auto *r = op->getOperand(1).getDefiningOp(),
             *c = op->getOperand(2).getDefiningOp();
        bool bcast = code == "load" && broadcastInner(view, fn);
        if (bcast)
          s.addAttribute("ct_src.broadcast_inner", b.getUnitAttr());
        // normalized verified a rectangular active domain and both coordinates.
        s.addOperands({view, r->getOperand(0), c->getOperand(0)});
        s.addAttribute(
            "allow_tma",
            b.getBoolAttr(!bcast &&
                          tileLoadStorePermitsTma(view, fn, facts, tma)));
        s.addAttribute(
            "matrix",
            b.getBoolAttr(*axes == intentdsl::block::RowProjection::Matrix));
        if (code == "store")
          s.addOperands(op->getOperand(4));
      } else
        s.addOperands(op->getOperands());
      s.addTypes(op->getResultTypes());
      for (auto &region : op->getRegions())
        s.addRegion()->takeBody(region);
      auto *lowered=b.create(s);
      if(code=="scaled_contract")intentdsl::block::formScaleDispatch(*lowered,scaledMma?"mx":"bf16");
      op->replaceAllUsesWith(lowered);
      op->erase();
    });
    fn.walk([&](Operation *store){
      if(store->getName().getStringRef()!="ct_src.tile_store")return;
      auto *value=store->getOperand(3).getDefiningOp();
      if(!value || value->getName().getStringRef()!="ct_src.key_counts" || !value->hasOneUse())return;
      if(value->getAttrOfType<StringAttr>("bin_mode")!="global-atomic")return;
      OpBuilder b(store);
      Location loc = store->getLoc();
      auto extent = value->getAttrOfType<IntegerAttr>("extent").getInt();
      auto span = value->getAttrOfType<IntegerAttr>("span").getInt();
      int64_t step = std::min<int64_t>(256, uint64_t(span) & -uint64_t(span));
      int64_t panels = span / step;
      auto axes = store->getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
      bool swap = axes && axes[0] == 1;
      Value src = value->getOperand(0), dst = store->getOperand(0);
      auto cInt = [&](int64_t v, Type ty) -> Value {
        OperationState cs(loc, "ct_src.constant");
        cs.addAttribute("value", b.getIntegerAttr(ty, v));
        cs.addTypes(ty);
        return b.create(cs)->getResult(0);
      };
      auto rowMap = [&](StringRef kind, ValueRange args, Type resTy) -> Value {
        OperationState ms(loc, "ct_src.map");
        ms.addOperands(args);
        ms.addAttribute("kind", b.getStringAttr(kind));
        ms.addTypes(resTy);
        return b.create(ms)->getResult(0);
      };
      OperationState dimSt(loc, "ct_src.array_dim");
      dimSt.addOperands(dst);
      dimSt.addAttribute("axis", b.getI64IntegerAttr(swap ? 0 : 1));
      dimSt.addTypes(b.getI64Type());
      Value bins = b.create(dimSt)->getResult(0);
      Value c0 = cInt(0, b.getI64Type());
      Value hasBins = rowMap("lt", {c0, bins}, b.getI1Type());

      OperationState s(loc, "ct_src.private_bin_store");
      s.addAttribute("transposed", b.getBoolAttr(swap));
      s.addAttribute("source_step", b.getI64IntegerAttr(step));
      s.addOperands(src);
      s.addOperands(dst);
      s.addOperands(value->getOperands().drop_front());
      s.addOperands(hasBins);
      s.addAttributes(value->getAttrs());
      s.addRegion();
      auto *guard = b.create(s);
      b.setInsertionPointToStart(b.createBlock(&guard->getRegion(0)));

      OperationState zb(loc, "ct_src.zero_bins");
      zb.addOperands(dst);
      zb.addAttribute("transposed", b.getBoolAttr(swap));
      b.create(zb);

      OperationState pidSt(loc, "ct_src.tile_pid");
      pidSt.addAttribute("axis", b.getI64IntegerAttr(0));
      pidSt.addTypes(b.getI64Type());
      Value pid = b.create(pidSt)->getResult(0);
      Value cPanels = cInt(panels, b.getI64Type());
      Value cSpan = cInt(span, b.getI64Type());
      Value cExt = cInt(extent, b.getI64Type());
      Value cStep = cInt(step, b.getI64Type());
      Value limit;
      if (extent % span == 0) {
        limit = cInt(panels, b.getI32Type());
      } else {
        Value partBase = rowMap("mul", {pid, cSpan}, b.getI64Type());
        Value rem = rowMap("sub", {cExt, partBase}, b.getI64Type());
        Value remPanels = rowMap("ceildiv", {rem, cStep}, b.getI64Type());
        Value clamped = rowMap("min", {cPanels, remPanels}, b.getI64Type());
        limit = rowMap("i32", {clamped}, b.getI32Type());
      }

      OperationState panelsSt(loc, "ct_src.bin_panels");
      panelsSt.addOperands(limit);
      panelsSt.addRegion();
      auto *panelsOp = b.create(panelsSt);
      auto *body = b.createBlock(&panelsOp->getRegion(0), {}, b.getI32Type(),
                                 {loc});
      b.setInsertionPointToStart(body);
      Value iv = body->getArgument(0);
      Value iv64 = rowMap("i64", {iv}, b.getI64Type());
      Value panelBase = rowMap("mul", {pid, cPanels}, b.getI64Type());
      Value panelIdx = rowMap("add", {panelBase, iv64}, b.getI64Type());

      Type srcElemTy = cast<MemRefType>(src.getType()).getElementType();
      OperationState read(loc, "ct_src.read_bin_keys");
      read.addOperands({src, panelIdx});
      read.addTypes(RankedTensorType::get({step}, srcElemTy));
      Value raw = b.create(read)->getResult(0);
      Type keyElemTy = srcElemTy.isInteger(64) ? b.getI64Type() : b.getI32Type();
      auto keyVecTy = RankedTensorType::get({step}, keyElemTy);
      auto i64VecTy = RankedTensorType::get({step}, b.getI64Type());
      auto i32VecTy = RankedTensorType::get({step}, b.getI32Type());
      auto i1VecTy = RankedTensorType::get({step}, b.getI1Type());
      OperationState convert(loc, "ct_src.bin_index");
      convert.addOperands(raw);
      convert.addTypes(keyVecTy);
      Value keys = b.create(convert)->getResult(0);

      Value partOff = rowMap("mul", {pid, cSpan}, b.getI64Type());
      Value stepOff = rowMap("mul", {iv64, cStep}, b.getI64Type());
      Value basePos = rowMap("add", {partOff, stepOff}, b.getI64Type());
      OperationState rangeSt(loc, "ct_src.range");
      rangeSt.addAttribute("size", b.getI64IntegerAttr(step));
      rangeSt.addTypes(i64VecTy);
      Value lanes = b.create(rangeSt)->getResult(0);
      Value pos = rowMap("add", {basePos, lanes}, i64VecTy);
      Value inExtent = rowMap("lt", {pos, cExt}, i1VecTy);
      Value cZeroKey = cInt(0, keyElemTy);
      Value ge0 = rowMap("ge", {keys, cZeroKey}, i1VecTy);
      Value binsKey =
          keyElemTy.isInteger(64) ? bins : rowMap("i32", {bins}, b.getI32Type());
      Value ltBins = rowMap("lt", {keys, binsKey}, i1VecTy);
      Value inBins = rowMap("and", {ge0, ltBins}, i1VecTy);
      Value valid = rowMap("and", {inExtent, inBins}, i1VecTy);
      Value cInvalidKey = cInt(-1, keyElemTy);
      Value safeKey = rowMap("select", {valid, keys, cInvalidKey}, keyVecTy);
      Value inc = rowMap("i32", {valid}, i32VecTy);

      OperationState update(loc, "ct_src.atomic_bins");
      update.addOperands({dst, safeKey, inc});
      update.addAttribute("transposed", b.getBoolAttr(swap));
      update.addAttribute("checked_bounds", b.getBoolAttr(true));
      update.addAttribute("scope", b.getStringAttr("workgroup"));
      b.create(update);

      store->erase();
      value->erase();
    });
    bool fnFtz = intentdsl::kir::allowsSubnormalFtz(fn);
    Builder fb(&getContext());
    fn.walk([&](Operation *op) {
      if (usesCutileFtzAttr(op)) {
        if (fnFtz)
          op->setAttr("ct_src.ftz", fb.getUnitAttr());
        else
          op->removeAttr("ct_src.ftz");
      } else if (op->hasAttr("ct_src.ftz")) {
        op->removeAttr("ct_src.ftz");
      }
    });
  }
};
struct Signature : PassWrapper<Signature, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Signature)
  unsigned bits;
  TargetProfile profile;
  Signature(unsigned bits, const TargetProfile &p) : bits(bits), profile(p) {}
  void runOnOperation() final {
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    Builder b(&getContext());
    getContext().loadDialect<arith::ArithDialect,scf::SCFDialect>();
    auto *cap = profile.services.get<intentdsl::cutile::ArrayIndices>();
    if (intentdsl::kir::orderedOwners(fn)) {
      auto *om = profile.services.get<intentdsl::block::OwnerMachine>();
      intentdsl::block::expandOwnerProgram(fn, "exits/cutile",
                                           nativeOwnerViews(om));
    }
    else if(intentdsl::kir::matrixDomain(fn)){
      intentdsl::block::MatrixAccess p(fn);
      if(auto flat=p.dot.getAttrOfType<DenseI64ArrayAttr>("ct_src.flat_groups")){
        bool canUseFlat = true;
        int64_t packedK = p.shape[2];
        if (p.dot.getName().stripDialect() == "quant_contract") {
          const auto *spec = intentdsl::kir::quantFormatSpec(p.dot);
          int64_t totalGroups = p.shape[2] / 32;
          canUseFlat = spec && spec->packFactor == 2 && p.tile[2] >= 32 &&
                       p.tile[2] % 32 == 0 && p.shape[0] > 0 &&
                       p.shape[0] % p.tile[0] == 0 && p.shape[1] > 0 &&
                       p.shape[1] % p.tile[1] == 0 && p.shape[2] > 0 &&
                       p.shape[2] % p.tile[2] == 0 && totalGroups > 0 &&
                       totalGroups <= 256 &&
                       llvm::isPowerOf2_64(totalGroups) && flat.size() == 2;
          if (canUseFlat)
            packedK = p.shape[2] / spec->packFactor;
        }
        if (canUseFlat) {
          llvm::json::Object views;
          for(auto slot:flat.asArrayRef())
            views[std::to_string(slot)]=slot==p.slot(p.dot.getOperand(0))?llvm::json::Array{p.shape[0],packedK}:llvm::json::Array{packedK,p.shape[1]};
          fn->setAttr("ct_src.native_views",b.getStringAttr(llvm::formatv("{0}",llvm::json::Value(std::move(views))).str()));
        }
      }
      auto mt = profile.services.get<intentdsl::block::MatrixTarget>();
      intentdsl::block::expandMatrixProgram(
          fn, "exits/cutile", scaledPanels, nativeMatrixRead,
          nativeWindowPanels, nativeMatrixWrite,
          mt && mt->flatRecordBitPlanes, mt && mt->flatSplitDequant,
          cap && cap->permitTma,
          mt ? StringRef(mt->libraryGemmImpl) : StringRef(""));
      if (cap && cap->preferRectRowReads) {
        // On SM90, aligned 1-row contiguous loads inside a TMA contraction
        // K-loop stall warpgroup scheduling when lowered as per-lane gathers.
        // Rewrite unmasked aligned 2D 1-row tile_reads to non-TMA rect_reads.
        auto extractAlignedBlock = [](Value coord, int64_t width) -> Value {
          auto add = coord.getDefiningOp<arith::AddIOp>();
          if (!add)
            return {};
          Value baseScalar;
          for (int side : {0, 1}) {
            Operation *bc = add->getOperand(side).getDefiningOp();
            Operation *ax = add->getOperand(1 - side).getDefiningOp();
            if (bc && ax &&
                bc->getName().getStringRef() == "ct_src.tile_broadcast" &&
                bc->getAttrOfType<IntegerAttr>("axis").getInt() == -1 &&
                !isa<ShapedType>(bc->getOperand(0).getType()) &&
                ax->getName().getStringRef() == "ct_src.tile_axis" &&
                cast<RankedTensorType>(ax->getResult(0).getType())
                        .getDimSize(0) == width) {
              baseScalar = bc->getOperand(0);
              break;
            }
          }
          if (!baseScalar)
            return {};
          if (auto mul = baseScalar.getDefiningOp<arith::MulIOp>()) {
            for (int side : {0, 1}) {
              if (auto c = mul->getOperand(side).getDefiningOp<arith::ConstantOp>())
                if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
                    ia && ia.getValue().getSignificantBits() <= 64 &&
                    ia.getInt() == width)
                  return mul->getOperand(1 - side);
            }
          }
          return {};
        };
        SmallVector<Operation *> rowReads;
        fn.walk([&](Operation *op) {
          if (op->getName().getStringRef() == "ct_src.tile_read" &&
              op->getNumOperands() == 2 &&
              op->getParentOfType<scf::ForOp>())
            rowReads.push_back(op);
        });
        for (Operation *rd : rowReads) {
          auto resTy = dyn_cast<RankedTensorType>(rd->getResult(0).getType());
          auto view = dyn_cast<BlockArgument>(rd->getOperand(0));
          Operation *addr = rd->getOperand(1).getDefiningOp();
          if (!resTy || resTy.getRank() != 1 || !view ||
              view.getOwner() != &fn.front() || !addr ||
              addr->getName().getStringRef() != "ct_src.tile_address" ||
              addr->getNumOperands() != 3)
            continue;
          auto mem = dyn_cast<MemRefType>(view.getType());
          auto proj = addr->getAttrOfType<DenseI64ArrayAttr>("projection");
          auto strides = addr->getAttrOfType<DenseI64ArrayAttr>("strides");
          int64_t width = resTy.getDimSize(0);
          if (!mem || mem.getRank() != 2 || !proj || proj.size() != 2 ||
              proj[0] != -1 || proj[1] != 0 || !strides ||
              strides.size() != 2 || strides[0] <= 0 || strides[1] != 1 ||
              width <= 0 || !llvm::isPowerOf2_64(width))
            continue;
          Value rowScalar = addr->getOperand(1);
          Value colBlock = extractAlignedBlock(addr->getOperand(2), width);
          if (isa<ShapedType>(rowScalar.getType()) || !colBlock)
            continue;
          OpBuilder rb(rd);
          Location rloc = rd->getLoc();
          auto rectTy = RankedTensorType::get({1, width}, resTy.getElementType());
          OperationState rSt(rloc, "ct_src.rect_read");
          rSt.addOperands({view, rowScalar, colBlock});
          rSt.addTypes(rectTy);
          rSt.addAttribute("allow_tma", rb.getBoolAttr(false));
          Value rect = rb.create(rSt)->getResult(0);
          OperationState shSt(rloc, "ct_src.reshape");
          shSt.addOperands(rect);
          shSt.addTypes(resTy);
          Value reshaped = rb.create(shSt)->getResult(0);
          rd->getResult(0).replaceAllUsesWith(reshaped);
          rd->erase();
          if (addr->use_empty())
            addr->erase();
        }
      }
    }
    else {
      fn.walk<WalkOrder::PostOrder>([](Operation *op){
        if(op->getName().getDialectNamespace()!="ct_src") return;
        auto n=op->getName().stripDialect();
        if(n=="key_counts")op->setAttr("native_panel",Builder(op->getContext()).getI64IntegerAttr(64));
        if(n=="indexed_tile_read" || n=="fill_ranges_at" || n=="segment_fold" || n=="segment_size" || n=="scatter_fragment" ||
           n=="sum_columns" || n=="key_counts" || n=="publish_offsets" || n=="stripe_loop")
          intentdsl::block::expandRowOperation(*op,"exits/cutile",false);
      });
      auto ownerAttr = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
      StringAttr idxTy = b.getStringAttr(bits == 64 ? "ct.int64" : "ct.int32");
      BoolAttr colTiles =
          b.getBoolAttr(fn->getAttr("block.grid") == intentdsl::block::grid(b, true));
      fn.walk([&](Operation *op) {
        if (op->getName().getDialectNamespace() != "ct_src")
          return;
        auto n = op->getName().stripDialect();
        if (n == "dimension") {
          OpBuilder db(op);
          Location loc = op->getLoc();
          Value base = fn.getArgument(intentdsl::domainSlot(fn));
          int rank = cast<MemRefType>(base.getType()).getRank();
          bool isM = op->getAttrOfType<StringAttr>("name") == "M";
          auto dimAt = [&](int axis) -> Value {
            OperationState ds(loc, "ct_src.array_dim");
            ds.addOperands(base);
            ds.addAttribute("axis", db.getI64IntegerAttr(axis));
            ds.addTypes(db.getI64Type());
            return db.create(ds)->getResult(0);
          };
          auto constOne = [&]() -> Value {
            OperationState cs(loc, "ct_src.constant");
            cs.addAttribute("value", db.getI64IntegerAttr(1));
            cs.addTypes(db.getI64Type());
            return db.create(cs)->getResult(0);
          };
          auto mulI64 = [&](Value lhs, Value rhs) -> Value {
            OperationState ms(loc, "ct_src.map");
            ms.addOperands({lhs, rhs});
            ms.addAttribute("kind", db.getStringAttr("mul"));
            ms.addTypes(db.getI64Type());
            return db.create(ms)->getResult(0);
          };
          Value val;
          if (ownerAttr && isM) {
            val = dimAt(ownerAttr.getInt());
          } else if (!ownerAttr && !isM) {
            val = dimAt(rank - 1);
          } else {
            val = constOne();
            for (int i = 0; i < (ownerAttr ? rank : rank - 1); ++i)
              if (!ownerAttr || i != ownerAttr.getInt())
                val = mulI64(val, dimAt(i));
          }
          op->getResult(0).replaceAllUsesWith(val);
          op->erase();
          return;
        }
        if (n == "gather_slab" || n == "scatter_slab") {
          bool store = n == "scatter_slab";
          if (op->hasAttr("ct_src.rectangular")) {
            OpBuilder sb(op);
            auto slot = cast<BlockArgument>(op->getOperand(0)).getArgNumber();
            int64_t extent =
                fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.shape")[2];
            Value rowBlk = op->getOperand(1).getDefiningOp()->getOperand(0);
            Value colBlk = op->getOperand(2).getDefiningOp()->getOperand(0);
            OperationState rs(op->getLoc(),
                              store ? "ct_src.rect_slab_store"
                                    : "ct_src.rect_slab_load");
            rs.addOperands({op->getOperand(0), rowBlk, colBlk});
            if (store)
              rs.addOperands(op->getOperand(4));
            rs.addAttribute("owner_axis",
                            sb.getI64IntegerAttr(ownerAttr.getInt()));
            rs.addAttribute("inner_extent", sb.getI64IntegerAttr(extent));
            if (!store)
              rs.addAttribute(
                  "broadcast_inner",
                  sb.getBoolAttr(op->hasAttr("ct_src.broadcast_inner")));
            rs.addAttribute("allow_tma",
                            sb.getBoolAttr(op->hasAttr("ct_src.slab_tma")));
            rs.addTypes(op->getResultTypes());
            auto *repl = sb.create(rs);
            op->replaceAllUsesWith(repl);
            op->erase();
            return;
          }
          op->setAttr("owner_axis", b.getI64IntegerAttr(ownerAttr.getInt()));
          return;
        }
        if (n == "rows" || n == "columns" || n == "stripe_owner_load" ||
            n == "stripe_tile_load" || n == "stripe_tile_store") {
          op->setAttr("index_type", idxTy);
        } else if (n == "pid") {
          op->setAttr("index_type", idxTy);
          op->setAttr("column_tiles", colTiles);
        } else if (n == "tiles" || n == "active" || n == "linear_index") {
          if (ownerAttr)
            op->setAttr("ct_src.owner_axis",
                        b.getI64IntegerAttr(ownerAttr.getInt()));
        }
      });
    }
    if (const auto *om = profile.services.get<intentdsl::block::OwnerMachine>();
        om && om->residentRegisters > 0) {
      std::function<std::optional<std::pair<int64_t, int64_t>>(Value, unsigned)>
          smallRange = [&](Value v, unsigned depth)
          -> std::optional<std::pair<int64_t, int64_t>> {
        if (!v || depth > 12)
          return std::nullopt;
        Operation *d = v.getDefiningOp();
        if (!d)
          return std::nullopt;
        if (auto c = dyn_cast<arith::ConstantOp>(d)) {
          if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
              ia && ia.getValue().getSignificantBits() <= 32) {
            int64_t x = ia.getInt();
            return std::pair{x, x};
          }
          if (auto da = dyn_cast<DenseIntElementsAttr>(c.getValue());
              da && da.isSplat() &&
              da.getElementType().getIntOrFloatBitWidth() <= 32) {
            int64_t x = da.getSplatValue<APInt>().getSExtValue();
            return std::pair{x, x};
          }
          return std::nullopt;
        }
        StringRef dn = d->getName().getStringRef();
        if (dn == "ct_src.tile_broadcast" || dn == "ct_src.reshape")
          return smallRange(d->getOperand(0), depth + 1);
        if (auto andOp = dyn_cast<arith::AndIOp>(d)) {
          if (auto r = smallRange(andOp.getRhs(), depth + 1);
              r && r->first >= 0 && r->first == r->second && r->second <= 4096)
            return std::pair{int64_t(0), r->second};
        }
        if (auto subOp = dyn_cast<arith::SubIOp>(d)) {
          auto l = smallRange(subOp.getLhs(), depth + 1),
               r = smallRange(subOp.getRhs(), depth + 1);
          if (l && r)
            return std::pair{l->first - r->second, l->second - r->first};
        }
        if (auto addOp = dyn_cast<arith::AddIOp>(d)) {
          auto l = smallRange(addOp.getLhs(), depth + 1),
               r = smallRange(addOp.getRhs(), depth + 1);
          if (l && r)
            return std::pair{l->first + r->first, l->second + r->second};
        }
        return std::nullopt;
      };
      SmallVector<arith::TruncFOp> truncs;
      fn.walk([&](arith::TruncFOp tr) { truncs.push_back(tr); });
      for (arith::TruncFOp tr : truncs) {
        auto outTy = dyn_cast<RankedTensorType>(tr.getType());
        if (!outTy ||
            (!outTy.getElementType().isF16() &&
             !outTy.getElementType().isBF16()) ||
            outTy.getNumElements() * 2 <= om->residentRegisters)
          continue;
        auto mul = tr.getIn().getDefiningOp<arith::MulFOp>();
        if (!mul || !mul->hasOneUse())
          continue;
        int64_t maxExact = outTy.getElementType().isF16() ? 2048 : 128;
        arith::SIToFPOp i2f;
        Value other;
        for (int side : {0, 1}) {
          if (auto cand = mul->getOperand(side).getDefiningOp<arith::SIToFPOp>();
              cand && cand->hasOneUse()) {
            if (auto r = smallRange(cand.getIn(), 0);
                r && r->first >= -maxExact && r->second <= maxExact) {
              i2f = cand;
              other = mul->getOperand(1 - side);
              break;
            }
          }
        }
        if (!i2f)
          continue;
        Value halfScale;
        OpBuilder nb(tr);
        if (auto ext = other.getDefiningOp<arith::ExtFOp>();
            ext && ext.getIn().getType() == outTy) {
          halfScale = ext.getIn();
        } else if (Operation *bc = other.getDefiningOp();
                   bc &&
                   bc->getName().getStringRef() == "ct_src.tile_broadcast") {
          if (auto ext = bc->getOperand(0).getDefiningOp<arith::ExtFOp>();
              ext && getElementTypeOrSelf(ext.getIn().getType()) ==
                         outTy.getElementType()) {
            OperationState bs(tr.getLoc(), "ct_src.tile_broadcast");
            bs.addOperands(ext.getIn());
            bs.addAttributes(bc->getAttrs());
            bs.addTypes(outTy);
            halfScale = nb.create(bs)->getResult(0);
          }
        }
        if (!halfScale)
          continue;
        Value wHalf =
            nb.create<arith::SIToFPOp>(tr.getLoc(), outTy, i2f.getIn());
        auto newMul = nb.create<arith::MulFOp>(tr.getLoc(), wHalf, halfScale,
                                               mul.getFastmathAttr());
        tr.replaceAllUsesWith(newMul.getResult());
        tr.erase();
        if (mul->use_empty())
          mul.erase();
        if (Operation *od = other.getDefiningOp();
            od && isMemoryEffectFree(od) && od->use_empty()) {
          Value inner = od->getNumOperands() ? od->getOperand(0) : Value();
          od->erase();
          if (inner)
            if (Operation *id = inner.getDefiningOp();
                id && isMemoryEffectFree(id) && id->use_empty())
              id->erase();
        }
        if (i2f->use_empty())
          i2f.erase();
      }
    }
    const auto *facts = profile.services.get<intentdsl::CallFacts>();
    auto finalizeFunc = [&](func::FuncOp f) {
      if (cap && cap->permitTma)
        f->setAttr("ct_src.tma", b.getUnitAttr());
      bool fFtz = intentdsl::kir::allowsSubnormalFtz(f);
      f.walk([&](Operation *op) {
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          StringAttr kind;
          ArrayAttr bounds;
          computeCutileLoopAttrs(loop, kind, bounds);
          loop->setAttr("ct_src.loop_kind", kind);
          if (bounds)
            loop->setAttr("ct_src.range_bounds", bounds);
          else
            loop->removeAttr("ct_src.range_bounds");
        }
        if (usesCutileFtzAttr(op)) {
          if (fFtz)
            op->setAttr("ct_src.ftz", b.getUnitAttr());
          else
            op->removeAttr("ct_src.ftz");
        } else if (op->hasAttr("ct_src.ftz")) {
          op->removeAttr("ct_src.ftz");
        }
      });
      auto entryAttr = f->getAttrOfType<StringAttr>("block.entry");
      f->setAttr("ct_src.entry",
                 entryAttr ? entryAttr : b.getStringAttr("kernel"));
      f->setAttr("ct_src.index_bits", b.getI32IntegerAttr(bits));
      f->setAttr("ct_src.extra_params", computeCutileExtraParams(f));
      stampCutileCarryIds(f);
      if (bits == 32) {
        f.walk([&](Operation *op) {
          if (op->getName().getStringRef() == "ct_src.tile_address" &&
              canNarrowCutileAddress(*op))
            op->setAttr("index_bits", b.getI32IntegerAttr(32));
        });
      }
      f.walk([&](Operation *op) {
        if (op->getName().getDialectNamespace() != "ct_src")
          return;
        auto n = op->getName().stripDialect();
        if (n == "uniform") {
          bool quad = intentdsl::cutile::quadAlignedUniform(op, f, facts);
          if (quad)
            op->setAttr("quad_aligned", b.getUnitAttr());
          op->setAttr("ci.implementation",
                      b.getStringAttr(quad ? "exits/cutile::random-quad"
                                           : "exits/cutile::random"));
          return;
        }
        auto id = llvm::StringSwitch<StringRef>(n)
                      .Case("summary_reduce", "moments-reduction")
                      .Case("summary_merge", "moments-merge")
                      .Case("scan_sum", "inclusive-scan")
                      .Case("reduce", "native-row-reduce")
                      .Default("");
        if (!id.empty())
          op->setAttr("ci.implementation",
                      b.getStringAttr(("exits/cutile::" + id).str()));
      });
    };
    for (auto f : getOperation().getOps<func::FuncOp>())
      finalizeFunc(f);
    if (bits == 32) {
      // Narrow proven-bounded i64 coordinate chains (rooted at i32 extui/extsi)
      // feeding narrowed tile_address/cmpi/trunci directly to i32 to cut 64-bit
      // register pressure in unrolled multi-axis indexing loops.
      if (fn.front().front().getName().getStringRef() ==
          "ct_src.tile_program") {
        auto &progBlock = fn.front().front().getRegion(0).front();
        llvm::DenseMap<Value, std::pair<int64_t, int64_t>> bounds;
        llvm::DenseMap<Value, Value> narrowed;
        llvm::DenseMap<Value, bool> constBool;
        constexpr int64_t kSafeLim = 1LL << 29;
        auto inSafe = [](int64_t lo, int64_t hi) {
          return lo >= -kSafeLim && hi <= kSafeLim && lo <= hi;
        };
        auto getConstI64 = [](Value v) -> std::optional<int64_t> {
          auto c = v.getDefiningOp<arith::ConstantOp>();
          if (!c)
            return std::nullopt;
          if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
              ia && ia.getType().isInteger(64))
            return ia.getInt();
          if (auto da = dyn_cast<DenseIntElementsAttr>(c.getValue());
              da && da.isSplat() && da.getElementType().isInteger(64))
            return da.getSplatValue<APInt>().getSExtValue();
          return std::nullopt;
        };
        auto i32Of = [&](Type t) -> Type {
          if (auto rt = dyn_cast<RankedTensorType>(t))
            return rt.clone(b.getI32Type());
          return b.getI32Type();
        };
        auto makeConstI32 = [&](OpBuilder &nb, Location loc, Type ty,
                                int64_t val) -> Value {
          if (auto rt = dyn_cast<RankedTensorType>(ty))
            return nb.create<arith::ConstantOp>(
                loc, rt, DenseElementsAttr::get(rt, nb.getI32IntegerAttr(val)));
          return nb.create<arith::ConstantIntOp>(loc, val, 32);
        };
        for (Operation &op : llvm::make_early_inc_range(progBlock)) {
          if (auto eu = dyn_cast<arith::ExtUIOp>(&op)) {
            if (getElementTypeOrSelf(eu.getType()).isInteger(64) &&
                getElementTypeOrSelf(eu.getIn().getType()).isInteger(32)) {
              if (auto rem = eu.getIn().getDefiningOp<arith::RemUIOp>()) {
                if (auto c = rem.getRhs().getDefiningOp<arith::ConstantOp>()) {
                  std::optional<int64_t> rv;
                  if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
                    rv = ia.getInt();
                  else if (auto da = dyn_cast<DenseIntElementsAttr>(c.getValue());
                           da && da.isSplat())
                    rv = da.getSplatValue<APInt>().getSExtValue();
                  if (rv && *rv > 0 && *rv <= kSafeLim) {
                    bounds[eu.getResult()] = {0, *rv - 1};
                    narrowed[eu.getResult()] = eu.getIn();
                  }
                } else if (auto bc = rem.getRhs().getDefiningOp();
                           bc && bc->getName().getStringRef() ==
                                     "ct_src.tile_broadcast") {
                  if (auto c = bc->getOperand(0)
                                   .getDefiningOp<arith::ConstantOp>()) {
                    if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
                        ia && ia.getInt() > 0 && ia.getInt() <= kSafeLim) {
                      bounds[eu.getResult()] = {0, ia.getInt() - 1};
                      narrowed[eu.getResult()] = eu.getIn();
                    }
                  }
                }
              }
            }
            continue;
          }
          if (op.getName().getStringRef() == "ct_src.tile_broadcast" &&
              getElementTypeOrSelf(op.getResult(0).getType()).isInteger(64)) {
            Value src = op.getOperand(0);
            if (auto it = narrowed.find(src); it != narrowed.end()) {
              OpBuilder nb(&op);
              OperationState st(op.getLoc(), "ct_src.tile_broadcast");
              st.addOperands(it->second);
              st.addAttributes(op.getAttrs());
              st.addTypes(i32Of(op.getResult(0).getType()));
              Value nv = nb.create(st)->getResult(0);
              narrowed[op.getResult(0)] = nv;
              bounds[op.getResult(0)] = bounds[src];
            } else if (auto cv = getConstI64(src); cv && inSafe(*cv, *cv)) {
              OpBuilder nb(&op);
              Value nv = makeConstI32(nb, op.getLoc(),
                                      i32Of(op.getResult(0).getType()), *cv);
              narrowed[op.getResult(0)] = nv;
              bounds[op.getResult(0)] = {*cv, *cv};
            }
            continue;
          }
          if (isa<arith::AddIOp, arith::SubIOp>(&op) &&
              getElementTypeOrSelf(op.getResult(0).getType()).isInteger(64)) {
            auto resolve = [&](Value v)
                -> std::optional<std::pair<Value, std::pair<int64_t, int64_t>>> {
              if (auto it = narrowed.find(v); it != narrowed.end())
                return std::pair{it->second, bounds[v]};
              if (auto cv = getConstI64(v); cv && inSafe(*cv, *cv)) {
                OpBuilder nb(&op);
                return std::pair{
                    makeConstI32(nb, op.getLoc(), i32Of(v.getType()), *cv),
                    std::pair{*cv, *cv}};
              }
              if (auto bc = v.getDefiningOp();
                  bc && bc->getName().getStringRef() == "ct_src.tile_broadcast") {
                if (auto cv = getConstI64(bc->getOperand(0));
                    cv && inSafe(*cv, *cv)) {
                  OpBuilder nb(&op);
                  return std::pair{
                      makeConstI32(nb, op.getLoc(), i32Of(v.getType()), *cv),
                      std::pair{*cv, *cv}};
                }
              }
              return std::nullopt;
            };
            auto l = resolve(op.getOperand(0)), r = resolve(op.getOperand(1));
            if (l && r) {
              int64_t lo = isa<arith::AddIOp>(&op)
                               ? (l->second.first + r->second.first)
                               : (l->second.first - r->second.second);
              int64_t hi = isa<arith::AddIOp>(&op)
                               ? (l->second.second + r->second.second)
                               : (l->second.second - r->second.first);
              if (inSafe(lo, hi)) {
                OpBuilder nb(&op);
                Value nv =
                    isa<arith::AddIOp>(&op)
                        ? Value(nb.create<arith::AddIOp>(op.getLoc(), l->first,
                                                         r->first))
                        : Value(nb.create<arith::SubIOp>(op.getLoc(), l->first,
                                                         r->first));
                narrowed[op.getResult(0)] = nv;
                bounds[op.getResult(0)] = {lo, hi};
              }
            }
            continue;
          }
          if (auto cmp = dyn_cast<arith::CmpIOp>(&op);
              cmp && getElementTypeOrSelf(cmp.getLhs().getType()).isInteger(64)) {
            auto resolveWithBounds = [&](Value v)
                -> std::optional<std::pair<Value, std::pair<int64_t, int64_t>>> {
              if (auto it = narrowed.find(v); it != narrowed.end())
                return std::pair{it->second, bounds[v]};
              if (auto cv = getConstI64(v); cv && inSafe(*cv, *cv)) {
                OpBuilder nb(cmp);
                return std::pair{
                    makeConstI32(nb, cmp.getLoc(), i32Of(v.getType()), *cv),
                    std::pair{*cv, *cv}};
              }
              if (auto bc = v.getDefiningOp();
                  bc && bc->getName().getStringRef() == "ct_src.tile_broadcast") {
                if (auto cv = getConstI64(bc->getOperand(0));
                    cv && inSafe(*cv, *cv)) {
                  OpBuilder nb(cmp);
                  return std::pair{
                      makeConstI32(nb, cmp.getLoc(), i32Of(v.getType()), *cv),
                      std::pair{*cv, *cv}};
                }
              }
              return std::nullopt;
            };
            if (narrowed.count(cmp.getLhs()) || narrowed.count(cmp.getRhs())) {
              auto l = resolveWithBounds(cmp.getLhs());
              auto r = resolveWithBounds(cmp.getRhs());
              if (l && r) {
                auto [lLo, lHi] = l->second;
                auto [rLo, rHi] = r->second;
                std::optional<bool> folded;
                switch (cmp.getPredicate()) {
                case arith::CmpIPredicate::sge:
                  if (lLo >= rHi)
                    folded = true;
                  else if (lHi < rLo)
                    folded = false;
                  break;
                case arith::CmpIPredicate::sle:
                  if (lHi <= rLo)
                    folded = true;
                  else if (lLo > rHi)
                    folded = false;
                  break;
                case arith::CmpIPredicate::slt:
                  if (lHi < rLo)
                    folded = true;
                  else if (lLo >= rHi)
                    folded = false;
                  break;
                case arith::CmpIPredicate::sgt:
                  if (lLo > rHi)
                    folded = true;
                  else if (lHi <= rLo)
                    folded = false;
                  break;
                default:
                  break;
                }
                OpBuilder nb(cmp);
                Value nc;
                if (folded) {
                  Type resTy = cmp.getType();
                  if (auto rt = dyn_cast<RankedTensorType>(resTy))
                    nc = nb.create<arith::ConstantOp>(
                        cmp.getLoc(), rt,
                        DenseElementsAttr::get(rt, nb.getBoolAttr(*folded)));
                  else
                    nc = nb.create<arith::ConstantIntOp>(cmp.getLoc(),
                                                         *folded ? 1 : 0, 1);
                  constBool[nc] = *folded;
                } else {
                  nc = nb.create<arith::CmpIOp>(cmp.getLoc(),
                                                cmp.getPredicate(), l->first,
                                                r->first);
                }
                cmp.replaceAllUsesWith(nc);
                cmp.erase();
              }
            }
            continue;
          }
          if (auto andOp = dyn_cast<arith::AndIOp>(&op);
              andOp &&
              getElementTypeOrSelf(andOp.getType()).isInteger(1)) {
            auto lb = constBool.find(andOp.getLhs());
            auto rb = constBool.find(andOp.getRhs());
            if (lb != constBool.end()) {
              if (lb->second) {
                andOp.replaceAllUsesWith(andOp.getRhs());
                andOp.erase();
              } else {
                andOp.replaceAllUsesWith(andOp.getLhs());
                andOp.erase();
              }
            } else if (rb != constBool.end()) {
              if (rb->second) {
                andOp.replaceAllUsesWith(andOp.getLhs());
                andOp.erase();
              } else {
                andOp.replaceAllUsesWith(andOp.getRhs());
                andOp.erase();
              }
            }
            continue;
          }
          if (op.getName().getStringRef() == "ct_src.tile_address") {
            auto ib = op.getAttrOfType<IntegerAttr>("index_bits");
            if (ib && ib.getInt() == 32) {
              for (unsigned i = 1; i < op.getNumOperands(); ++i) {
                if (auto it = narrowed.find(op.getOperand(i));
                    it != narrowed.end())
                  op.setOperand(i, it->second);
              }
            }
          }
        }
        bool changed = true;
        while (changed) {
          changed = false;
          for (Operation &op :
               llvm::make_early_inc_range(llvm::reverse(progBlock))) {
            if (op.use_empty() && isMemoryEffectFree(&op) &&
                op.getNumRegions() == 0 && !op.hasTrait<OpTrait::IsTerminator>()) {
              op.erase();
              changed = true;
            }
          }
        }
      }
    }
    std::set<std::string> seenResources;
    SmallVector<std::string> runtimeImpls;
    getOperation().walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<StringAttr>("ci.implementation")) {
        auto key = attr.getValue().str();
        auto it = profile.implementations.find(key);
        if (it != profile.implementations.end() &&
            it->second.source == ImplementationSource::Runtime &&
            seenResources.insert(it->second.resource).second)
          runtimeImpls.push_back(key);
      }
    });
    auto firstFn = *getOperation().getOps<func::FuncOp>().begin();
    OpBuilder mb(firstFn);
    for (auto &impl : runtimeImpls) {
      OperationState st(firstFn.getLoc(), "ct_src.runtime_source");
      st.addAttribute("implementation", mb.getStringAttr(impl));
      mb.create(st);
    }
    SmallVector<char> data;llvm::raw_svector_ostream os(data);
    if(auto err=describeArtifact(getOperation(),os)){fn.emitError()<<llvm::toString(std::move(err));signalPassFailure();return;}
    getOperation()->setAttr("ct_src.package",b.getStringAttr(os.str()));
  }
};
LogicalResult verify(ModuleOp module, const TargetProfile &profile,
                     bool callable) {
  SmallVector<func::FuncOp, 2> fns(module.getOps<func::FuncOp>());
  if (fns.empty() || fns.size() > (callable ? 2u : 1u))
    return failure();
  for (auto f : fns)
    if (f.empty() || f.front().empty())
      return failure();
  bool isSplitModule = fns.size() == 2;
  auto matrix = fns.back();
  if (isSplitModule) {
    auto dqFn = fns.front();
    if (dqFn->getAttrOfType<StringAttr>("ct_src.entry") != "dequant_rhs" ||
        dqFn.front().front().getName().getStringRef() !=
            "ct_src.tile_program" ||
        dqFn.front().front().getAttrOfType<StringAttr>("calling_convention") !=
            "split_dequant" ||
        !matrix->hasAttr("block.split_dequant"))
      return dqFn.emitError("invalid cuTile split dequantization module");
  }
  if (!callable) {
    if (!llvm::hasSingleElement(module.getBody()->getOperations()))
      return failure();
  } else {
    auto attached=module->getAttrOfType<StringAttr>("ct_src.package");
    if(!attached)return matrix.emitError("missing cuTile callable package");
    if (&module.getBody()->back() != matrix.getOperation())
      return matrix.emitError("expected function after runtime_source declarations");
    std::set<std::string> declaredResources;
    bool seenFunc = false;
    for (auto &top : module.getBody()->without_terminator()) {
      if (isa<func::FuncOp>(top)) {
        seenFunc = true;
        continue;
      }
      if (seenFunc || top.getName().getStringRef() != "ct_src.runtime_source")
        return top.emitError("unexpected module-level operation");
      auto impl = top.getAttrOfType<StringAttr>("implementation");
      if (!impl)
        return top.emitError("missing runtime_source implementation");
      if (auto err = profile.checkImplementation(impl.getValue()))
        return top.emitError() << llvm::toString(std::move(err));
      auto &c = profile.implementations.at(impl.getValue().str());
      if (c.source != ImplementationSource::Runtime ||
          !declaredResources.insert(c.resource).second)
        return top.emitError("invalid or duplicate runtime_source");
    }
    std::set<std::string> usedResources;
    module.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<StringAttr>("ci.implementation")) {
        auto it = profile.implementations.find(attr.getValue().str());
        if (it != profile.implementations.end() &&
            it->second.source == ImplementationSource::Runtime)
          usedResources.insert(it->second.resource);
      }
    });
    if (declaredResources != usedResources)
      return matrix.emitError("runtime_source declarations disagree with bound operations");
  }
  auto arrayCap=profile.services.get<intentdsl::cutile::ArrayIndices>();
  for (auto f : fns) {
    if(!arrayCap || f->hasAttr("ct_src.tma")!=arrayCap->permitTma)return f.emitError("invalid array transfer capability");
    bool expectedFtz = intentdsl::kir::allowsSubnormalFtz(f);
    auto commonWalk = f.walk([&](Operation *op) {
      if (op->hasAttr("ct_src.ftz") != (usesCutileFtzAttr(op) && expectedFtz)) {
        op->emitError("ct_src.ftz disagrees with function denormal-fp-math mode");
        return WalkResult::interrupt();
      }
      if (callable) {
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          StringAttr expectedKind;
          ArrayAttr expectedBounds;
          computeCutileLoopAttrs(loop, expectedKind, expectedBounds);
          if (loop->getAttr("ct_src.loop_kind") != expectedKind ||
              loop->getAttr("ct_src.range_bounds") != expectedBounds) {
            loop.emitError(
                "missing or mismatched cuTile scf.for lowering decision");
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (commonWalk.wasInterrupted())
      return failure();
    if (callable) {
      if (f->getAttr("ct_src.extra_params") != computeCutileExtraParams(f))
        return f.emitError("missing or mismatched ct_src.extra_params");
      if (failed(verifyCutileCarryIds(f)))
        return failure();
    }
  }
  if (isSplitModule) {
    for (auto f : fns) {
      if (failed(verifyArrayGeometry(f, profile)) ||
          failed(intentdsl::block::verifyTileProgram(
              f, profile, [&](Operation &op) {
                return arrayCap->bits == 32 && canNarrowCutileAddress(op);
              })))
        return failure();
    }
    return success();
  }
  if(matrix.front().front().getName().getStringRef()=="ct_src.tile_program"){
    if(!callable || failed(verifyArrayGeometry(matrix,profile)))return failure();
    return intentdsl::block::verifyTileProgram(
        matrix, profile, [&](Operation &op) {
          return arrayCap->bits == 32 && canNarrowCutileAddress(op);
        });
  }
  if (intentdsl::kir::orderedOwners(matrix)) {
    {
      auto &owner = matrix.front().front();
      auto rows = owner.getAttrOfType<IntegerAttr>("block.owner_tile");
      Attribute expected;
      if (owner.hasAttr("block.pair_axis") && rows && rows.getInt() < 16)
        expected = Builder(matrix.getContext()).getI64IntegerAttr(16);
      if (owner.getAttr("ct_src.matrix_rows") != expected)
        return matrix.emitError("invalid native matrix fragment height"),
               failure();
    }
    const auto *facts = profile.services.get<intentdsl::CallFacts>();
    if (!facts ||
        failed(intentdsl::specializeProjectedLayout(matrix, *facts, "ct_src",
                                                    false)) ||
        failed(intentdsl::block::verifyOwnerMap(matrix, profile, "ct_src",
                                                "realized")))
      return failure();
    auto result = matrix.walk([&](Operation *op) {
      auto cap = profile.services.get<intentdsl::cutile::ArrayIndices>();
      if (op->hasAttr("ct_src.panel_tma") !=
          panelPermitsTma(op, matrix, *facts, cap->permitTma)) {
        op->emitError(
            "rectangular transfer requires a compatible array layout");
        return WalkResult::interrupt();
      }
      if (op->hasAttr("ct_src.page_tma") !=
          (op->hasAttr("block.page_extent") && cap->permitTma)) {
        op->emitError("page transfer requires bounded page coordinates and a "
                      "compatible array layout");
        return WalkResult::interrupt();
      }
      Attribute expectedRect;
      if (auto rectAxis = ownerRectangleAxis(op, matrix))
        expectedRect =
            Builder(matrix.getContext()).getI64IntegerAttr(*rectAxis);
      if (op->getAttr("ct_src.owner_rectangle") != expectedRect ||
          op->hasAttr("ct_src.owner_tma") !=
              ownerRectPermitsTma(op, matrix, *facts, cap->permitTma)) {
        op->emitError("owner rectangle requires an aligned full group and "
                      "contiguous logical axes");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }
  if (intentdsl::kir::matrixDomain(matrix)) {
    if (failed(intentdsl::block::verifyScaleDispatch(
            matrix, profile, "ct_src",
            profile.services.get<intentdsl::cutile::ArrayIndices>()->scaledMma
                ? "mx"
                : "bf16")))
      return failure();
    if(callable && matrix->getAttr("ct_src.index_bits")!=Builder(matrix.getContext()).getI32IntegerAttr(profile.services.get<intentdsl::cutile::ArrayIndices>()->bits))
      return matrix.emitError("matrix index width disagrees with the target");
    return success(!matrix.walk([&](Operation *op){
      if (op->getName().getStringRef() == "ct_src.scaled_contract")
        return op->getAttr("ct_src.flat_groups") != adjacentScaleGroups(op, matrix)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      if (isBlockScaledFp4Quant(op))
        return (op->getAttr("ct_src.flat_groups") !=
                    adjacentScaleGroups(op, matrix) ||
                op->hasAttr("ct_src.scaled_mma") != arrayCap->scaledMma)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      return WalkResult::advance();}).wasInterrupted());
  }
  if (failed(intentdsl::verifyMapping(module, profile, "ct_src")))
    return failure();
  auto fn = *module.getOps<func::FuncOp>().begin();
  if(failed(intentdsl::specializeProjectedLayout(fn,*profile.services.get<intentdsl::CallFacts>(),"ct_src",false)))return failure();
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) || fn.getNumResults())
    return fn.emitError("expected a defined output-buffer kernel");
  auto abi = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abi)
    return fn.emitError("missing cuTile argument slots");
  auto signature = llvm::json::parse(abi.getValue());
  if (!signature) {
    llvm::consumeError(signature.takeError());
    return fn.emitError("malformed cuTile argument slots");
  }
  auto *slots = signature->getAsArray();
  if (!slots || slots->size() != fn.getNumArguments())
    return fn.emitError("cuTile argument count mismatch");
  for (auto [arg, slot] : llvm::zip(fn.getArguments(), *slots)) {
    auto *record = slot.getAsObject();
    auto view = dyn_cast<MemRefType>(arg.getType());
    Type scalar = view ? view.getElementType() : arg.getType();
    if (!record || !record->getString("name") || !record->getArray("dims") ||
        record->getArray("dims")->size() !=
            unsigned(view ? view.getRank() : 0) ||
        (!scalar.isF16() && !scalar.isF32() && !scalar.isBF16() &&
         !scalar.isInteger(32) && !(view && scalar.isSignlessInteger(64)) && (view || !scalar.isUnsignedInteger(64))) ||
        record->getString("dtype") != (scalar.isF16()         ? "f16"
                                       : scalar.isBF16()      ? "bf16"
                                       : scalar.isInteger(32) ? "i32"
                                       : scalar.isSignlessInteger(64) ? "i64"
                                       : scalar.isUnsignedInteger(64) ? "u64"
                                                              : "f32") ||
        (view ? (record->getString("direction") != "In" &&
                 record->getString("direction") != "Out" &&
                 record->getString("direction") != "InOut")
              : record->getString("direction") != "Scalar"))
      return fn.emitError(
          "cuTile array or scalar slot does not match function type");
  }
  auto indices = profile.services.get<intentdsl::cutile::ArrayIndices>();
  if (!indices)
    return fn.emitError("missing cuTile array index profile");
  if (callable &&
      (fn->getAttrOfType<StringAttr>("ct_src.entry") != "kernel" ||
       fn->getAttr("ct_src.index_bits") !=
           Builder(module.getContext()).getI32IntegerAttr(indices->bits)))
    return fn.emitError("missing cuTile callable ABI");
  for (auto row : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto v = llvm::json::parse(
        cast<DictionaryAttr>(row).getAs<StringAttr>("payload").getValue());
    if (!v) {
      llvm::consumeError(v.takeError());
      return failure();
    }
    auto *o = v->getAsObject();
    if (!o || o->size() != 3 || o->getInteger("num_ctas") != 1 ||
        o->getInteger("opt_level") != 3 ||
        (o->getInteger("occupancy") != 1 && o->getInteger("occupancy") != 2 &&
         o->getInteger("occupancy") != 4))
      return fn.emitError("invalid native hints for row");
  }
  if(callable && failed(intentdsl::block::verifyRowProgram(fn,"ct_src")))return failure();
  const auto *factsPtr = profile.services.get<intentdsl::CallFacts>();
  auto res = module.walk([&](Operation *op) {
    if(callable && isa<arith::ConstantOp,arith::IndexCastOp,scf::ForOp,scf::YieldOp>(op))return WalkResult::advance();
    if (op == module || op->getName().getDialectNamespace() == "func" ||
        (callable && op->getName().getStringRef() == "ct_src.runtime_source"))
      return WalkResult::advance();
    auto n = op->getName().stripDialect();
    bool ok = op->getName().getDialectNamespace() == "ct_src";
    if (ok && n == "uniform") {
      bool expectedQuad =
          callable && intentdsl::cutile::quadAlignedUniform(op, fn, factsPtr);
      if (op->hasAttr("quad_aligned") != expectedQuad)
        return op->emitError("uniform quad alignment disagrees with row layout"),
               WalkResult::interrupt();
      if (callable &&
          op->getAttrOfType<StringAttr>("ci.implementation") !=
              (expectedQuad ? "exits/cutile::random-quad"
                            : "exits/cutile::random"))
        return op->emitError("uniform implementation contract mismatch"),
               WalkResult::interrupt();
    }
    if (ok && n == "array_dim") {
      auto arr = dyn_cast<BlockArgument>(op->getOperand(0));
      auto ax = op->getAttrOfType<IntegerAttr>("axis");
      ok = arr && arr.getOwner() == &fn.front() && isa<MemRefType>(arr.getType()) &&
           ax && ax.getInt() >= 0 && ax.getInt() < cast<MemRefType>(arr.getType()).getRank();
      if (!ok)
        return op->emitError("invalid array_dim operand or axis"), WalkResult::interrupt();
    }
    if (ok && (n=="zero_bins" || n=="bin_panels" || n=="read_bin_keys" || n=="bin_index" || n=="atomic_bins")) {
      auto parent=op->getParentOp();
      StringRef required=n=="zero_bins" || n=="bin_panels"?"ct_src.private_bin_store":"ct_src.bin_panels";
      if(!parent || parent->getName().getStringRef()!=required)
        return op->emitError("histogram operation requires its enclosing count region"),WalkResult::interrupt();
    }
    if(ok && n=="private_bin_store"){
      auto src=dyn_cast<BlockArgument>(op->getOperand(0)),dst=dyn_cast<BlockArgument>(op->getOperand(1));
      auto *mask=op->getOperand(4).getDefiningOp();
      auto actualExtent = intentdsl::kir::keyExtent(
          op, *profile.services.get<intentdsl::CallFacts>());
      if (!actualExtent ||
          op->getAttrOfType<IntegerAttr>("extent").getInt() != *actualExtent)
        return op->emitError(
                   "private_bin_store extent disagrees with source view"),
               WalkResult::interrupt();
      int64_t extentVal = *actualExtent;
      int64_t spanVal = op->getAttrOfType<IntegerAttr>("span").getInt();
      bool swapVal = op->getAttrOfType<BoolAttr>("transposed").getValue();
      auto isMap = [](Value v, StringRef kind) -> Operation * {
        auto *d = v.getDefiningOp();
        return d && d->getName().getStringRef() == "ct_src.map" &&
                       d->getAttrOfType<StringAttr>("kind") == kind
                   ? d
                   : nullptr;
      };
      auto isConst = [](Value v, int64_t expected) -> bool {
        auto *d = v.getDefiningOp();
        if (!d || d->getName().getStringRef() != "ct_src.constant")
          return false;
        auto a = d->getAttrOfType<IntegerAttr>("value");
        return a && a.getValue().getSExtValue() == expected;
      };
      auto *nonEmpty = isMap(op->getOperand(5), "lt");
      auto *binsDef = nonEmpty ? nonEmpty->getOperand(1).getDefiningOp() : nullptr;
      ok=src && dst && src.getOwner()==&fn.front() && dst.getOwner()==&fn.front() &&
         op->getAttrOfType<StringAttr>("bin_mode")=="global-atomic" &&
         (*slots)[src.getArgNumber()].getAsObject()->getString("direction")=="In" &&
         (*slots)[dst.getArgNumber()].getAsObject()->getString("direction")=="Out" &&
         cast<MemRefType>(src.getType()).getRank()==1 && cast<MemRefType>(dst.getType()).getRank()==2 &&
         cast<MemRefType>(dst.getType()).getElementType().isInteger(32) && mask && mask->getName().stripDialect()=="active" &&
         mask->getOperand(1)==op->getOperand(2) && mask->getOperand(2)==op->getOperand(3) &&
         nonEmpty && isConst(nonEmpty->getOperand(0), 0) && binsDef &&
         binsDef->getName().getStringRef() == "ct_src.array_dim" &&
         binsDef->getOperand(0) == dst &&
         binsDef->getAttrOfType<IntegerAttr>("axis").getInt() == (swapVal ? 0 : 1) &&
         spanVal>0 && fn->getAttrOfType<StringAttr>("block.topology")=="resident" &&
         llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),[](Attribute row){return cast<DictionaryAttr>(row).getAs<IntegerAttr>("rows").getInt()==1;});
      auto step=op->getAttrOfType<IntegerAttr>("source_step");
      if(!step || step.getInt()<=0 || step.getInt()>2048 || !llvm::isPowerOf2_64(step.getInt()) || spanVal%step.getInt()!=0 || !llvm::hasSingleElement(op->getRegion(0)))
        return op->emitError("invalid private_bin_store step or region"),WalkResult::interrupt();
      int64_t stepVal = step.getInt(), panelsVal = spanVal / stepVal;
      auto &guard=op->getRegion(0).front();
      if(guard.empty() || guard.front().getName().getStringRef()!="ct_src.zero_bins" ||
         guard.front().getOperand(0)!=dst ||
         guard.front().getAttrOfType<BoolAttr>("transposed").getValue()!=swapVal ||
         guard.back().getName().getStringRef()!="ct_src.bin_panels" ||
         !llvm::hasSingleElement(guard.back().getRegion(0)))
        return op->emitError("expected guarded zero then count panels"),WalkResult::interrupt();
      Value limitVal = guard.back().getOperand(0);
      bool limitOk = false;
      if (extentVal % spanVal == 0) {
        limitOk = isConst(limitVal, panelsVal);
      } else if (auto *toI32 = isMap(limitVal, "i32")) {
        if (auto *mn = isMap(toI32->getOperand(0), "min"))
          if (isConst(mn->getOperand(0), panelsVal))
            if (auto *cd = isMap(mn->getOperand(1), "ceildiv"))
              if (isConst(cd->getOperand(1), stepVal))
                if (auto *sb = isMap(cd->getOperand(0), "sub"))
                  if (isConst(sb->getOperand(0), extentVal))
                    if (auto *ml = isMap(sb->getOperand(1), "mul"))
                      limitOk = ml->getOperand(0).getDefiningOp() &&
                                ml->getOperand(0).getDefiningOp()->getName().getStringRef() == "ct_src.tile_pid" &&
                                isConst(ml->getOperand(1), spanVal);
      }
      if (!limitOk)
        return guard.back().emitError("bin_panels upper bound disagrees with partition geometry"), WalkResult::interrupt();
      auto &loop=guard.back().getRegion(0).front();
      Operation *readOp = nullptr, *convOp = nullptr, *atomOp = &loop.back();
      for (auto &inst : loop) {
        if (inst.getName().getStringRef() == "ct_src.read_bin_keys")
          readOp = &inst;
        else if (inst.getName().getStringRef() == "ct_src.bin_index")
          convOp = &inst;
      }
      if (loop.getNumArguments() != 1 ||
          !loop.getArgument(0).getType().isInteger(32) || !readOp || !convOp ||
          atomOp->getName().getStringRef() != "ct_src.atomic_bins" ||
          readOp->getOperand(0) != src ||
          convOp->getOperand(0) != readOp->getResult(0) ||
          atomOp->getOperand(0) != dst ||
          atomOp->getAttrOfType<BoolAttr>("transposed").getValue() != swapVal ||
          !atomOp->getAttrOfType<BoolAttr>("checked_bounds") ||
          atomOp->getAttrOfType<StringAttr>("scope") != "workgroup" || !src ||
          convOp->getResult(0).getType() !=
              RankedTensorType::get(
                  {stepVal},
                  cast<MemRefType>(src.getType()).getElementType().isInteger(64)
                      ? IntegerType::get(fn.getContext(), 64)
                      : IntegerType::get(fn.getContext(), 32)) ||
          readOp->getResult(0).getType() !=
              RankedTensorType::get(
                  {stepVal}, cast<MemRefType>(src.getType()).getElementType()))
        return op->emitError("invalid count panel or atomic update"),WalkResult::interrupt();
      auto isPid0 = [](Value v) -> bool {
        auto *d = v.getDefiningOp();
        return d && d->getName().getStringRef() == "ct_src.tile_pid" &&
               d->getAttrOfType<IntegerAttr>("axis").getInt() == 0;
      };
      auto isIv64 = [&](Value v) -> bool {
        auto *d = isMap(v, "i64");
        return d && d->getOperand(0) == loop.getArgument(0);
      };
      auto isBinsDim = [&](Value v) -> bool {
        auto *d = v.getDefiningOp();
        return d && d->getName().getStringRef() == "ct_src.array_dim" &&
               d->getOperand(0) == dst &&
               d->getAttrOfType<IntegerAttr>("axis").getInt() ==
                   (swapVal ? 0 : 1);
      };
      auto *panelAdd = isMap(readOp->getOperand(1), "add");
      auto *panelMul =
          panelAdd ? isMap(panelAdd->getOperand(0), "mul") : nullptr;
      bool panelIdxOk = panelAdd && panelMul && isPid0(panelMul->getOperand(0)) &&
                        isConst(panelMul->getOperand(1), panelsVal) &&
                        isIv64(panelAdd->getOperand(1));
      auto *sel = isMap(atomOp->getOperand(1), "select");
      auto *incCast = isMap(atomOp->getOperand(2), "i32");
      auto *validOp = (sel && incCast && sel->getOperand(0) == incCast->getOperand(0))
                          ? isMap(sel->getOperand(0), "and")
                          : nullptr;
      auto *inExtOp = validOp ? isMap(validOp->getOperand(0), "lt") : nullptr;
      auto *posAdd = inExtOp ? isMap(inExtOp->getOperand(0), "add") : nullptr;
      auto *baseAdd = posAdd ? isMap(posAdd->getOperand(0), "add") : nullptr;
      auto *partMul = baseAdd ? isMap(baseAdd->getOperand(0), "mul") : nullptr;
      auto *stepMul = baseAdd ? isMap(baseAdd->getOperand(1), "mul") : nullptr;
      auto *lanesOp = posAdd ? posAdd->getOperand(1).getDefiningOp() : nullptr;
      bool posOk = posAdd && baseAdd && partMul && stepMul && lanesOp &&
                   isPid0(partMul->getOperand(0)) &&
                   isConst(partMul->getOperand(1), spanVal) &&
                   isIv64(stepMul->getOperand(0)) &&
                   isConst(stepMul->getOperand(1), stepVal) &&
                   lanesOp->getName().getStringRef() == "ct_src.range" &&
                   lanesOp->getAttrOfType<IntegerAttr>("size").getInt() == stepVal;
      auto *inBinsOp = validOp ? isMap(validOp->getOperand(1), "and") : nullptr;
      auto *ge0Op = inBinsOp ? isMap(inBinsOp->getOperand(0), "ge") : nullptr;
      auto *ltBinsOp = inBinsOp ? isMap(inBinsOp->getOperand(1), "lt") : nullptr;
      bool wideKey =
          cast<MemRefType>(src.getType()).getElementType().isInteger(64);
      bool binsBoundOk = false;
      if (ltBinsOp) {
        if (wideKey) {
          binsBoundOk = isBinsDim(ltBinsOp->getOperand(1));
        } else if (auto *binsI32 = isMap(ltBinsOp->getOperand(1), "i32")) {
          binsBoundOk = isBinsDim(binsI32->getOperand(0));
        }
      }
      bool inBinsOk =
          ge0Op && ltBinsOp && ge0Op->getOperand(0) == convOp->getResult(0) &&
          isConst(ge0Op->getOperand(1), 0) &&
          ltBinsOp->getOperand(0) == convOp->getResult(0) && binsBoundOk;
      int64_t invalidKey =
          atomOp->getAttrOfType<BoolAttr>("checked_bounds").getValue() ? -1 : 0;
      if (!panelIdxOk || !sel || sel->getOperand(1) != convOp->getResult(0) ||
          !isConst(sel->getOperand(2), invalidKey) || !inExtOp || !posOk ||
          !isConst(inExtOp->getOperand(1), extentVal) || !inBinsOk)
        return atomOp->emitError(
                   "atomic_bins requires verified extent and bin range masks"),
               WalkResult::interrupt();
    }
    if (ok && op->hasAttr("ct_src.slab_tma") && n != "gather_slab" &&
        n != "scatter_slab")
      ok = false;
    if (ok && op->hasAttr("ct_src.broadcast_inner") && n != "gather_slab" &&
        n != "tile_load")
      ok = false;
    if (ok && n == "dimension")
      ok = !callable;
    if (ok && callable &&
        (n == "rows" || n == "columns" || n == "stripe_owner_load" ||
         n == "stripe_tile_load" || n == "stripe_tile_store" || n == "pid")) {
      Builder ib(fn.getContext());
      StringAttr expectedIdxTy =
          ib.getStringAttr(indices->bits == 64 ? "ct.int64" : "ct.int32");
      ok = op->getAttr("index_type") == expectedIdxTy;
      if (ok && n == "pid") {
        BoolAttr expectedCols = ib.getBoolAttr(
            fn->getAttr("block.grid") == intentdsl::block::grid(ib, true));
        ok = op->getAttr("column_tiles") == expectedCols;
      }
    }
    if (ok && callable &&
        (n == "tiles" || n == "active" || n == "linear_index")) {
      auto ownerAttr = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
      Attribute expectedOwner =
          ownerAttr
              ? Attribute(Builder(fn.getContext())
                              .getI64IntegerAttr(ownerAttr.getInt()))
              : Attribute();
      ok = op->getAttr("ct_src.owner_axis") == expectedOwner;
    }
    if (ok && (n == "gather_slab" || n == "scatter_slab")) {
      bool rect = rectangularSlab(op, fn);
      if (callable) {
        auto ownerAttr = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
        ok = !rect && !op->hasAttr("ct_src.rectangular") &&
             !op->hasAttr("ct_src.broadcast_inner") &&
             !op->hasAttr("ct_src.slab_tma") && ownerAttr &&
             op->getAttr("owner_axis") ==
                 Builder(fn.getContext()).getI64IntegerAttr(ownerAttr.getInt());
      } else {
        bool bcast =
            n == "gather_slab" && rect && broadcastInner(op->getOperand(0), fn);
        bool slabTma = rect && !bcast && factsPtr &&
                       alignedProjectedView(op->getOperand(0), fn, *factsPtr,
                                            indices->permitTma);
        ok = op->hasAttr("ct_src.rectangular") == rect &&
             op->hasAttr("ct_src.broadcast_inner") == bcast &&
             op->hasAttr("ct_src.slab_tma") == slabTma;
      }
    }
    if (ok && (n == "rect_slab_load" || n == "rect_slab_store")) {
      bool store = n == "rect_slab_store";
      auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
      auto ownerAttr = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
      if (!callable || !arg || arg.getOwner() != &fn.front() || !ownerAttr ||
          ownerAttr.getInt() > 1 ||
          ((*slots)[arg.getArgNumber()].getAsObject()->getString("direction") !=
               "InOut" &&
           (*slots)[arg.getArgNumber()].getAsObject()->getString("direction") !=
               (store ? "Out" : "In"))) {
        op->emitError("invalid rectangular slab access");
        return WalkResult::interrupt();
      }
      auto shape =
          fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(), "ct_src.shape");
      auto *pid = op->getOperand(1).getDefiningOp(),
           *col = op->getOperand(2).getDefiningOp();
      auto ind = dyn_cast<BlockArgument>(op->getOperand(2));
      bool bcast = !store && broadcastInner(arg, fn);
      bool allowTma = !bcast && factsPtr &&
                      alignedProjectedView(arg, fn, *factsPtr,
                                           indices->permitTma);
      Builder sb(fn.getContext());
      ok = shape && shape.size() == 3 && llvm::isPowerOf2_64(shape[2]) &&
           llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),
                        [](Attribute row) {
                          return llvm::isPowerOf2_64(
                              cast<DictionaryAttr>(row)
                                  .getAs<IntegerAttr>("width")
                                  .getInt());
                        }) &&
           op->getAttr("owner_axis") ==
               sb.getI64IntegerAttr(ownerAttr.getInt()) &&
           op->getAttr("inner_extent") == sb.getI64IntegerAttr(shape[2]) &&
           (store || op->getAttr("broadcast_inner") == sb.getBoolAttr(bcast)) &&
           op->getAttr("allow_tma") == sb.getBoolAttr(allowTma) && pid &&
           pid->getName().getStringRef() == "ct_src.pid" &&
           ((col && col->getName().getStringRef() == "ct_src.zero") ||
            (col && col->getName().getStringRef() == "ct_src.pid" &&
             col->getAttrOfType<IntegerAttr>("axis").getInt() == 1 &&
             pid->getAttrOfType<IntegerAttr>("axis").getInt() == 0 &&
             isa<ArrayAttr>(fn->getAttrOfType<ArrayAttr>("block.grid")[1])) ||
            (ind && ind.getArgNumber() == 0 &&
             ind.getOwner()->getParentOp()->getName().getStringRef() ==
                 "ct_src.tiles"));
    }
    if (ok && n == "sum_columns") {
      auto panel = op->getAttrOfType<IntegerAttr>("native_panel");
      auto extent = op->getAttrOfType<IntegerAttr>("extent");
      int64_t expected = extent ? int64_t(std::min<uint64_t>(
                                      256, llvm::PowerOf2Ceil(std::max<int64_t>(
                                               1, extent.getInt()))))
                                : 0;
      ok = !callable && panel && panel.getInt() == expected;
    }
    if (ok && n == "stripe_loop")
      ok = !callable;
    if (ok && (n == "stripe_tile_load" || n == "stripe_tile_store" ||
               n == "stripe_owner_load")) {
      bool store = n == "stripe_tile_store";
      bool ownerLd = n == "stripe_owner_load";
      auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
      auto round = dyn_cast<BlockArgument>(op->getOperand(1));
      auto *parent = op->getParentOp();
      auto st = op->getAttrOfType<IntegerAttr>("stride");
      bool validRound =
          round && parent && round.getOwner() == &parent->getRegion(0).front() &&
          round.getArgNumber() == 0 &&
          (callable ? isa<scf::ForOp>(parent)
                    : parent->getName().getStringRef() == "ct_src.stripe_loop");
      bool validDir =
          arg && arg.getOwner() == &fn.front() &&
          ((*slots)[arg.getArgNumber()].getAsObject()->getString("direction") ==
               "InOut" ||
           (*slots)[arg.getArgNumber()].getAsObject()->getString("direction") ==
               (store ? "Out" : "In"));
      if (!validRound || !validDir || !st || st.getInt() <= 0) {
        op->emitError("invalid stripe access binding");
        return WalkResult::interrupt();
      }
      int rank = cast<MemRefType>(arg.getType()).getRank();
      if (ownerLd) {
        auto *rowsOp = op->getOperand(2).getDefiningOp();
        ok = rank == 1 && rowsOp &&
             rowsOp->getName().getStringRef() == "ct_src.rows";
      } else {
        auto *pid = op->getOperand(2).getDefiningOp(),
             *col = op->getOperand(3).getDefiningOp();
        ok = rank == 2 &&
             op->getAttrOfType<BoolAttr>("allow_tma").getValue() ==
                 indices->permitTma &&
             pid && pid->getName().getStringRef() == "ct_src.pid" && col &&
             col->getName().getStringRef() == "ct_src.zero" &&
             llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),
                          [&](Attribute row) {
                            return st.getInt() %
                                       cast<DictionaryAttr>(row)
                                           .getAs<IntegerAttr>("rows")
                                           .getInt() ==
                                   0;
                          });
      }
    }
    if (ok && (n == "tile_store" || n == "tile_load")) {
      auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
      if (!arg || arg.getOwner() != &fn.front() ||
          ((*slots)[arg.getArgNumber()].getAsObject()->getString("direction") !=
               "InOut" &&
           (*slots)[arg.getArgNumber()].getAsObject()->getString("direction") !=
               (n == "tile_load" ? "In" : "Out"))) {
        op->emitError("tile access conflicts with public buffer direction");
        return WalkResult::interrupt();
      }
      auto axes = intentdsl::block::projection(op->getOperand(0).getType());
      auto *pid = op->getOperand(1).getDefiningOp(),
           *col = op->getOperand(2).getDefiningOp();
      auto ind = dyn_cast<BlockArgument>(op->getOperand(2));
      bool bcast = n == "tile_load" && broadcastInner(arg, fn);
      bool allowTma =
          !bcast && factsPtr &&
          tileLoadStorePermitsTma(arg, fn, *factsPtr, indices->permitTma);
      ok = op->hasAttr("ct_src.broadcast_inner") == bcast &&
           op->getAttrOfType<BoolAttr>("allow_tma").getValue() == allowTma &&
           succeeded(axes) &&
           op->getAttrOfType<BoolAttr>("matrix").getValue() ==
               (*axes == intentdsl::block::RowProjection::Matrix) &&
           (n != "tile_store" ||
            *axes == intentdsl::block::RowProjection::Matrix ||
            (cast<MemRefType>(fn.getArgument(fn->getAttrOfType<IntegerAttr>(
                                                   "kir.domain_arg")
                                                 .getInt())
                                  .getType())
                     .getRank() == 1 &&
             op->getAttrOfType<DenseI64ArrayAttr>("block.member_axes") ==
                 DenseI64ArrayAttr::get(fn.getContext(), {0}))) &&
           pid && pid->getName().getStringRef() == "ct_src.pid" &&
           ((col && col->getName().getStringRef() == "ct_src.zero") ||
            (col && col->getName().getStringRef()=="ct_src.pid" &&
             col->getAttrOfType<IntegerAttr>("axis").getInt()==1 &&
             pid->getAttrOfType<IntegerAttr>("axis").getInt()==0 &&
             isa<ArrayAttr>(fn->getAttrOfType<ArrayAttr>("block.grid")[1])) ||
            (ind && ind.getArgNumber() == 0 &&
             ind.getOwner()->getParentOp()->getName().getStringRef() ==
                 "ct_src.tiles"));
    }
    if (!ok) {
      op->emitError("invalid cuTile native projection or tile index");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!res.wasInterrupted());
}
} // namespace
void matrixIssues(Unit &);
void bind(Unit &unit) {
  matrixIssues(unit);
  registerContracts(unit);
  auto addParity = [&](ParityDeclaration p) {
    unit.parity.push_back({std::move(p), COMPILERINFRA_SOURCE});
  };
  addParity({"levels/execution/block::matrix-contraction",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction", "exits/cutile::native-dot"},
             "Tensor Core ct.mma matrix contraction across SM90 and SM120"});
  addParity({"levels/execution/block::block-scaled-fp8-contraction",
             "sm120",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction", "exits/cutile::scaled-mma"},
             "Blackwell SM120 native ct.mma_scaled MX8/E8M0 block-scaled FP8 Tensor Core MMA"});
  addParity({"levels/execution/block::block-scaled-fp8-contraction",
             "sm90",
             ParityStatus::Unsupported,
             {},
             "Hopper SM90 lacks Blackwell block-scaled MMA hardware; legalizes to E8M0 scale decode + ct.mma"});
  addParity({"levels/execution/block::q4k-packed-contraction",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction", "exits/cutile::row-collective"},
             "Packed Q4_K x Q8_K super-block integer subgroup dot and FP32 scale accumulation"});
  addParity({"levels/execution/block::ordered-owner-pairing",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::ordered-owners", "exits/cutile::native-dot",
              "exits/cutile::row-collective"},
             "Paired QK/PV Tensor Core owner tiles and single-pass online rescaling"});
  addParity({"levels/execution/block::row-reduction-and-scan",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::structure",
              "exits/cutile::native-row-reduce", "exits/cutile::inclusive-scan",
              "exits/cutile::moments-reduction", "exits/cutile::moments-merge"},
             "Native ct.sum/ct.max, ct.cumsum, and Welford moments reduction"});
  addParity({"levels/execution/block::shared-private-histogram",
             "*",
             ParityStatus::Unsupported,
             {},
             "tileiras toolchain lacks on-chip shared-memory histogram/atomic intrinsics; lowers to global-atomic workspace bins"});
  addParity({"levels/execution/block::counter-philox-rng",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::structure", "exits/cutile::random",
              "exits/cutile::random-quad"},
             "Runtime Philox 4x32 counter-based uniform_at / uniform_quad"});
  addParity({"levels/execution/block::bulk-tma-transfer",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction",
              "levels/execution/block::ordered-owners",
              "levels/execution/block::structure"},
             "Hardware TMA 2D/ND rectangular tile loads via ct.load(..., allow_tma=True)"});
  addParity({"levels/execution/block::narrow-i32-addressing",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction",
              "levels/execution/block::ordered-owners"},
             "proveNarrowAddress-verified 32-bit array coordinate indexing"});
  unit.dialects = registerIR;
  for (bool callable : {false, true})
    unit.boundaries.push_back(
        {{callable ? "callable" : "legal",
          [callable](OpPassManager &pm, const TargetProfile &p) -> llvm::Error {
            if (callable) {
              auto indices = p.services.get<intentdsl::cutile::ArrayIndices>();
              if (!indices)
                return llvm::createStringError(
                    "cuTile needs bound device options");
              pm.addPass(std::make_unique<Signature>(indices->bits, p));
            } else {
              auto indices = p.services.get<intentdsl::cutile::ArrayIndices>();
              if (!indices)
                return llvm::createStringError("missing array profile");
              pm.addPass(std::make_unique<Lower>(*indices,*p.services.get<intentdsl::CallFacts>()));
            }
            return llvm::Error::success();
          },
          [callable](ModuleOp m, const TargetProfile &p) {
            return verify(m, p, callable);
          }},
         COMPILERINFRA_SOURCE});
  ExitImplementation exit;exit.capabilities=capabilities;exit.terminal=cutilePrinter();
  unit.exit=Located<ExitImplementation>{std::move(exit),COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
