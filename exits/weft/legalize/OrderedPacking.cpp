#include "Ordered.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"
#include <numeric>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
std::string list(ArrayRef<int64_t> xs) {
  SmallVector<std::string> s;
  for (auto x : xs)
    s.push_back(std::to_string(x));
  return llvm::join(s, ",");
}
std::string panelType(StringRef kind, ArrayRef<int64_t> shape,
                      ArrayRef<int64_t> axes) {
  return ("!weft_kernel." + kind + "<" +
          (kind == "value" ? "bf16"
                           : "!weft_kernel.encoding<\"bf16\",\"dense\",\"dense."
                             "bf16\",[]>") +
          ",[" + list(shape) + "],[" + list(axes) + "]>")
      .str();
}
Operation *emit(OpBuilder &b, Location loc, StringRef name, ValueRange args,
                ArrayRef<NamedAttribute> attrs, bool result = false,
                bool region = false) {
  OperationState st(loc, ("weft_host." + name).str());
  st.addAttributes(attrs);
  st.addOperands(args);
  if (result)
    st.addTypes(b.getIndexType());
  if (region)
    st.addRegion();
  return b.create(st);
}
bool constant(Value v, int64_t n) {
  auto *op = v.getDefiningOp();
  return op && op->getName().stripDialect() == "task_const" &&
         op->getAttrOfType<StringAttr>("lit") == std::to_string(n);
}

struct Use {
  Operation *slice, *sub, *dot;
};
struct Panel {
  int64_t slot, tile, width, rows, n, k, redAxis, freeAxis, ownerAxis;
  bool transpose;
  SmallVector<Use> uses;
};

// The replacement view occupies the same task argument. Check every use and
// both offsets before changing its ABI. Other slices can still use the dense
// layout.
std::optional<Panel> match(BlockArgument arg, ArrayRef<int64_t> dims,
                           ArrayRef<int64_t> axes) {
  Panel p{};
  p.slot = arg.getArgNumber();
  for (auto *slice : arg.getUsers()) {
    if (slice->getName().stripDialect() != "task_slice" ||
        slice->getNumOperands() + 1 != dims.size())
      return {};
    auto sel = slice->getAttrOfType<ArrayAttr>("selectors");
    if (sel.size() != dims.size())
      return {};
    for (auto [i, s] : llvm::enumerate(sel))
      if (cast<StringAttr>(s).getValue() !=
          (i + 2 < dims.size() ? "index" : "all"))
        return {};
    for (auto *sub : slice->getResult(0).getUsers()) {
      if (sub->getName().stripDialect() != "task_subview" ||
          sub->getNumOperands() != 3 || !sub->getResult(0).hasOneUse() ||
          !constant(sub->getOperand(2), 0))
        return {};
      auto size = sub->getAttrOfType<DenseI64ArrayAttr>("static_extents");
      if (!size || size.size() != 2 || size[0] <= 0 ||
          dims[dims.size() - 2] % size[0] || size[1] != dims.back())
        return {};
      auto iv = dyn_cast<BlockArgument>(sub->getOperand(1));
      auto *loop = iv ? iv.getOwner()->getParentOp() : nullptr;
      if (!loop || iv.getArgNumber() != 0 ||
          loop->getName().stripDialect() != "task_for" ||
          !constant(loop->getOperand(0), 0) ||
          !constant(loop->getOperand(2), size[0]))
        return {};
      auto *dot = *sub->getResult(0).getUsers().begin();
      if (dot->getName().stripDialect() != "task_contract" ||
          dot->getOperand(1) != sub->getResult(0) ||
          dot->getAttrOfType<StringAttr>("engine") != "onednn-brgemm-bf16")
        return {};
      auto over = dot->getAttrOfType<DenseI64ArrayAttr>("over");
      if (!over || over.size() != 1)
        return {};
      auto red = over[0];
      bool transpose = red == axes.back();
      if (!transpose && red != axes[axes.size() - 2])
        return {};
      if (!p.uses.empty() && (p.tile != size[0] || p.transpose != transpose))
        return {};
      p.tile = size[0];
      p.width = size[1];
      p.transpose = transpose;
      p.k = transpose ? p.width : p.tile;
      p.n = transpose ? p.tile : p.width;
      if (p.k % 2)
        return {};
      p.redAxis = red;
      p.freeAxis = axes[axes.size() - (transpose ? 2 : 1)];
      // Splitting the last (reduction) axis preserves the LHS physical order.
      auto out = dot->getAttrOfType<StringAttr>("val_type").getValue();
      auto parts = out.split(",[").second.split("],[");
      int64_t rows, owner;
      if (parts.first.split(',').first.getAsInteger(10, rows) ||
          parts.second.split(',').first.getAsInteger(10, owner) || rows <= 0 ||
          (!p.uses.empty() && (p.rows != rows || p.ownerAxis != owner)))
        return {};
      p.rows = rows;
      p.ownerAxis = owner;
      auto lhs = cast<StringAttr>(dot->getAttrOfType<ArrayAttr>("in_types")[0]);
      if (lhs != panelType("value", {rows, p.k}, {owner, red}))
        return {};
      p.uses.push_back({slice, sub, dot});
    }
  }
  return p.uses.empty() ? std::nullopt : std::optional<Panel>(std::move(p));
}

void packHost(OpBuilder &b, Operation &team, const Panel &p, int64_t elements,
              ArrayRef<Attribute> cleanup, StringRef name) {
  auto loc = team.getLoc();
  auto s = [&](StringRef k, const std::string &v) {
    return b.getNamedAttr(k, b.getStringAttr(v));
  };
  auto flag = [&](StringRef k, bool v) {
    return b.getNamedAttr(k, b.getBoolAttr(v));
  };
  auto bytes = b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(2));
  b.setInsertionPoint(&team);
  emit(b, loc, "host_alloc", {},
       {s("buf_name", name.str()), s("placement", "invocation"),
        s("c_type", "uint16_t"), s("count_expr", std::to_string(elements)),
        bytes, b.getNamedAttr("cleanup", b.getArrayAttr(cleanup))});
  b.setInsertionPointAfter(&team);
  emit(b, loc, "host_free", {},
       {s("buf_name", name.str()),
        b.getNamedAttr("invocation", b.getUnitAttr())});
  b.setInsertionPointToStart(&team.getRegion(0).front());
  auto body = [&](Operation *op) {
    auto &block = op->getRegion(0).emplaceBlock();
    b.setInsertionPointToStart(&block);
    auto *end = emit(b, loc, "end", {}, {});
    b.setInsertionPoint(end);
  };
  auto num = [](int64_t n) { return std::to_string(n); };
  body(emit(b, loc, "host_omp_for", {},
            {s("iv", "pn"), s("lb", "0"), s("ub", num(elements / (p.k * p.n))),
             s("step", "1"), flag("parallel", false), flag("simd", false),
             flag("x86_simd", false), s("schedule", "static"), s("if_cond", ""),
             s("num_threads", "")},
            false, true));
  for (auto [iv, extent] : {std::pair{"g", p.k / 2}, std::pair{"n", p.n}})
    body(emit(b, loc, "host_for", {},
              {s("iv", iv), s("lb", "0"), s("ub", num(extent)), s("step", "1")},
              false, true));
  std::string base = "pn*" + num(p.k * p.n);
  for (int lane : {0, 1}) {
    auto k = "(2*g+" + num(lane) + ")";
    auto src = p.transpose ? "n*" + num(p.width) + "+" + k
                           : k + "*" + num(p.width) + "+n";
    emit(b, loc, "host_elem_op", {},
         {s("op", "assign"),
          s("dst_expr", name.str() + "[" + base + "+(g*" + num(p.n) + "+n)*2+" +
                            num(lane) + "]"),
          s("src_expr", "((const uint16_t*)a" + num(p.slot) + ")[" + base +
                            "+" + src + "]"),
          s("aux_expr", ""), bytes});
  }
}
} // namespace

void lowerInvocationPanels(
    func::FuncOp fn, Operation &program, Operation &kernel, Operation &team,
    DenseI64ArrayAttr slots,
    const llvm::DenseMap<Value, SmallVector<int64_t>> &axes) {
  OpBuilder b(&kernel);
  auto loc = kernel.getLoc();
  auto &bb = kernel.getRegion(0).front();
  auto viewTypes =
      llvm::to_vector(kernel.getAttrOfType<ArrayAttr>("view_types").getValue());
  auto ranks = llvm::to_vector(
      program.getAttrOfType<DenseI64ArrayAttr>("task_ranks").asArrayRef());
  Operation *call = nullptr;
  team.walk([&](Operation *op) {
    if (op->getName().stripDialect() == "host_call_owner_task")
      call = op;
  });
  if (!call)
    return;
  auto args =
      llvm::to_vector(call->getAttrOfType<ArrayAttr>("slot_args").getValue());
  int64_t newAxis = 0;
  for (auto &it : axes)
    for (auto ax : it.second)
      newAxis = std::max(newAxis, ax);
  SmallVector<Attribute> cleanup;
  for (auto slot : slots.asArrayRef()) {
    auto arg = fn.getArgument(slot);
    ArrayRef<int64_t> ax = axes.find(arg)->second;
    auto dims =
        fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "cpu.shape").asArrayRef();
    auto p = match(bb.getArgument(slot), dims, ax);
    if (!p)
      continue;
    auto tileAxis = ++newAxis, group = ++newAxis, pair = ++newAxis;
    SmallVector<int64_t> rhsDims{p->k / 2, p->n, 2},
        rhsAxes{group, p->freeAxis, pair};
    SmallVector<int64_t> shape(dims.drop_back(2)), ids(ax.drop_back(2));
    shape.push_back(dims[dims.size() - 2] / p->tile);
    shape.append(rhsDims);
    ids.push_back(tileAxis);
    ids.append(rhsAxes);
    std::string view = panelType("view", shape, ids),
                rhs = panelType("value", rhsDims, rhsAxes);
    auto str = [&](StringRef k, StringRef v) {
      return b.getNamedAttr(k, b.getStringAttr(v));
    };
    auto val = [&](StringRef op, ValueRange v, ArrayRef<NamedAttribute> a) {
      return emit(b, loc, op, v, a, true)->getResult(0);
    };
    llvm::SmallPtrSet<Operation *, 4> oldSlices;
    for (auto [slice, sub, dot] : p->uses) {
      b.setInsertionPoint(dot);
      auto lhsType =
          cast<StringAttr>(dot->getAttrOfType<ArrayAttr>("in_types")[0])
              .getValue();
      auto m = p->rows, owner = p->ownerAxis;
      Value tile =
          val("task_const", {},
              {str("lit", std::to_string(p->tile)), str("val_type", "index")});
      Value index = val("task_binary", {sub->getOperand(1), tile},
                        {str("kind", "div"), str("lhs_type", "index"),
                         str("rhs_type", "index"), str("val_type", "index")});
      SmallVector<Value> coords(slice->getOperands());
      coords.push_back(index);
      SmallVector<Attribute> selectors(coords.size() - 1,
                                       b.getStringAttr("index"));
      selectors.append(3, b.getStringAttr("all"));
      Value packed =
          val("task_slice", coords,
              {str("view_type", view),
               str("slice_type", panelType("slice", rhsDims, rhsAxes)),
               str("val_type", rhs),
               b.getNamedAttr("selectors", b.getArrayAttr(selectors)),
               b.getNamedAttr("admit", b.getBoolAttr(true))});
      auto lhs = panelType("value", {m, p->k / 2, 2}, {owner, group, pair});
      auto reshaped = val("task_reshape", {dot->getOperand(0)},
                          {str("in_type", lhsType), str("val_type", lhs),
                           str("order", list({owner, p->redAxis}))});
      dot->setOperands({reshaped, packed});
      dot->setAttr("kind", b.getStringAttr("contract"));
      dot->setAttr("over", b.getDenseI64ArrayAttr({group, pair}));
      dot->setAttr("in_types", b.getArrayAttr({b.getStringAttr(lhs),
                                               b.getStringAttr(rhs)}));
      sub->erase();
      oldSlices.insert(slice);
    }
    for (auto *op : oldSlices)
      op->erase();
    // Unused slices may have been formed for an eliminated contraction tail.
    for (auto *op : llvm::make_early_inc_range(bb.getArgument(slot).getUsers()))
      if (op->use_empty() && op->getName().stripDialect() == "task_slice")
        op->erase();
    viewTypes[slot] = b.getStringAttr(view);
    ranks[slot] = shape.size();
    auto name = b.getStringAttr("packed_" + std::to_string(slot));
    args[slot] = name;
    packHost(b, team, *p,
             std::accumulate(dims.begin(), dims.end(), int64_t(1),
                             std::multiplies<>()),
             cleanup, name.getValue());
    cleanup.push_back(name);
  }
  kernel.setAttr("view_types", b.getArrayAttr(viewTypes));
  program.setAttr("task_ranks", b.getDenseI64ArrayAttr(ranks));
  call->setAttr("slot_args", b.getArrayAttr(args));
}
} // namespace CI_UNIT_NAMESPACE
