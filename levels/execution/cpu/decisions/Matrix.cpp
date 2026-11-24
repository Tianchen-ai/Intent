#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::cpu {
bool groupedI8(Operation &dot, const TargetProfile &p, int64_t width);
static ArrayRef<MatrixImplementation> available(Operation &region,
                                                const MatrixMachine &caps) {
  return calculation(region).getName().stripDialect() == "q4k_dot"
             ? caps.records
             : caps.choices;
}
static std::string inputDtype(Operation &dot, func::FuncOp fn) {
  auto abi = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abi)
    return "";
  auto parsed = llvm::json::parse(abi.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return "";
  }
  auto i = cast<BlockArgument>(dot.getOperand(0)).getArgNumber();
  return (*parsed->getAsArray())[i]
      .getAsObject()
      ->getString("dtype")
      .value_or("")
      .str();
}
static std::string candidateName(const MatrixImplementation &impl,
                                 int64_t packCode) {
  auto shortId = llvm::StringRef(impl.id).rsplit("::").second;
  if (shortId.empty())
    shortId = impl.id;
  std::string suffix = packCode == 1   ? "-pack"
                       : packCode == 3 ? "-f32pack"
                                       : "";
  return std::to_string(impl.rows) + "x" + std::to_string(impl.cols) + "x" +
         std::to_string(impl.depth) + "-" + shortId.str() + suffix;
}
static bool validContractPacking(StringRef kind, StringRef inType,
                                 const MatrixImplementation &impl,
                                 int64_t packCode) {
  if (kind == "q4k_dot")
    return impl.dtype == "q4k" && packCode == 0;
  if (kind == "scaled_contract")
    return (impl.dtype == "any" || impl.dtype == "f32") && packCode == 0;
  if (kind == "window_contract") {
    if (impl.dtype == "any")
      return packCode == 0 || packCode == 1;
    return impl.dtype == "f32" && packCode == 1;
  }
  if (kind != "contract")
    return false;
  if (impl.dtype == "any")
    return packCode == 0 || packCode == 1;
  if (impl.dtype == "vnni-dot4")
    return inType == "i8" && packCode == 2;
  if (impl.dtype == "f32")
    return inType == "f32" ? (packCode == 0 || packCode == 1)
                           : ((inType == "f16" || inType == "bf16") &&
                              packCode == 3);
  if (impl.dtype == "f16")
    return inType == "f16" && (packCode == 0 || packCode == 1);
  if (impl.dtype == "bf16")
    return inType == "bf16" && packCode == 4;
  if (impl.dtype == "s8")
    return inType == "i8" && packCode == 4;
  return false;
}
static std::string packingFor(StringRef kind, int64_t packCode,
                              const MatrixMachine &caps) {
  if (kind == "q4k_dot")
    return "shared-q8k-per-call";
  if (kind == "scaled_contract")
    return "decoded-f32-per-call";
  if (kind == "window_contract")
    return packCode ? "mapped-f32-panels" : "mapped-right-per-call";
  switch (packCode) {
  case 1:
    return "rhs-panels-per-call";
  case 2:
    return caps.integerPacking;
  case 3:
    return "f32-panels-per-call";
  case 4:
    return "vnni-pack32-panels";
  default:
    return "";
  }
}
static llvm::Error checkContractPreconditions(Operation &region,
                                              const MatrixImplementation &impl,
                                              int64_t packCode,
                                              const TargetProfile &p) {
  auto &dot = calculation(region);
  auto kind = dot.getName().stripDialect();
  auto *facts = p.services.get<CallFacts>();
  auto g = kir::contractionShape(dot, region.getOperand(0), *facts);
  if (!g)
    return g.takeError();
  if (packCode == 2 && !groupedI8(dot, p, impl.cols))
    return llvm::createStringError(
        "VNNI dot4 requires K>0, K%4==0, N%tile_n==0, unit LHS K stride, and "
        "direct i32 accumulation");
  bool isBrgemm = llvm::StringRef(impl.id).contains("onednn-brgemm");
  if (isBrgemm && g->k <= 0)
    return llvm::createStringError(
        "oneDNN BRGeMM microkernel requires positive reduction extent K > 0");
  if (impl.kAlign > 1 && g->k % impl.kAlign != 0)
    return llvm::createStringError(
        "reduction extent K is not a multiple of microkernel K alignment");
  bool allowSubnormalFtz =
      bool(dot.getAttrOfType<UnitAttr>("kir.allow_subnormal_ftz"));
  if (impl.flushesSubnormals && !allowSubnormalFtz)
    return llvm::createStringError(
        "numerical contract forbids implicit BF16 subnormal FTZ");
  if (kind == "contract" && impl.requiresContiguousLhsK && packCode != 3 &&
      !g->routed) {
    auto slot = cast<BlockArgument>(dot.getOperand(0)).getArgNumber();
    auto *strides = facts->arguments[slot].getAsObject()->getArray("strides");
    int ak = dot.getAttrOfType<IntegerAttr>("lhs_k").getInt();
    int ab = dot.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
    int am = 0;
    while (am == ak || am == ab)
      ++am;
    auto sk = (*strides)[ak].getAsInteger().value_or(0);
    auto sm = (*strides)[am].getAsInteger().value_or(0);
    if (sk != 1 || sm <= 0)
      return llvm::createStringError(
          "microkernel requires unit LHS K stride and positive row stride");
  }
  if (kind == "contract" && isBrgemm && packCode == 0) {
    auto slot = cast<BlockArgument>(dot.getOperand(1)).getArgNumber();
    auto *strides = facts->arguments[slot].getAsObject()->getArray("strides");
    int bk = dot.getAttrOfType<IntegerAttr>("rhs_k").getInt();
    int bb = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
    int bn = 0;
    while (bn == bk || bn == bb)
      ++bn;
    auto sn = (*strides)[bn].getAsInteger().value_or(0);
    auto sk = (*strides)[bk].getAsInteger().value_or(0);
    if (sn != 1 || sk <= 0)
      return llvm::createStringError(
          "unpacked BRGeMM requires unit RHS column stride and positive K "
          "stride");
  }
  return llvm::Error::success();
}

bool groupedI8(Operation &dot, const TargetProfile &p, int64_t width) {
  if (dot.getName().stripDialect() != "contract" ||
      !cast<MemRefType>(dot.getOperand(0).getType())
           .getElementType()
           .isInteger(8) ||
      p.services.get<MatrixMachine>()->integerPacking.empty())
    return false;
  auto &region = *dot.getParentOp();
  auto f = p.services.get<CallFacts>();
  auto g = kir::contractionShape(dot, region.getOperand(0), *f);
  if (!g) {
    llvm::consumeError(g.takeError());
    return false;
  }
  if (!cast<MemRefType>(region.getOperand(0).getType())
           .getElementType()
           .isInteger(32) ||
      llvm::any_of(region.getRegion(0).front(), [](Operation &op) {
        return op.getName().stripDialect() == "convert_matrix";
      }))
    return false;
  auto i = cast<BlockArgument>(dot.getOperand(0)).getArgNumber();
  auto k = dot.getAttrOfType<IntegerAttr>("lhs_k").getInt();
  return g->k > 0 && g->k % 4 == 0 && g->n % width == 0 &&
         (*f->arguments[i].getAsObject()->getArray("strides"))[k]
                 .getAsInteger() == 1;
}
Operation *findMatrixTask(func::FuncOp fn) {
  if (!fn || fn.empty())
    return nullptr;
  for (auto &op : fn.front())
    if (op.getName().getStringRef() == "cpu.matrix_task")
      return &op;
  return nullptr;
}
bool matrixTask(func::FuncOp fn) { return findMatrixTask(fn) != nullptr; }
Operation &calculation(Operation &domain) {
  auto &bb = domain.getRegion(0).front();
  return *llvm::find_if(bb, [](Operation &o) {
    auto n = o.getName().stripDialect();
    return kir::isContraction(n);
  });
}
LogicalResult verifyMatrixTask(func::FuncOp fn, const TargetProfile &p,
                               StringRef storage) {
  auto &task = fn.front().front();
  if (task.getName().getStringRef() != "cpu.matrix_task" ||
      !llvm::hasSingleElement(task.getRegion(0)) ||
      task.getRegion(0).front().getOperations().size() != 2)
    return task.emitError("expected a matrix region and task completion");
  auto &region = task.getRegion(0).front().front();
  if (fn.front().getOperations().size() != 3 ||
      task.getNextNode()->getName().getStringRef() != "cpu.join" ||
      task.getNextNode()->getOperand(0) != task.getResult(0) ||
      region.getName().getStringRef() != "cpu.matrix_domain" ||
      task.getOperands() != fn.getArguments() ||
      failed(kir::verifyMatrixBody(fn, region, "cpu", true)))
    return task.emitError("invalid matrix task captures, computation or join");
  if (failed(verifyMatrixPartition(fn, region, p)))
    return failure();
  Builder b(fn.getContext());
  if (storage == "pending")
    return success(task.getAttr("cpu.storage") ==
                   b.getStringAttr("private-values"));
  return success(task.getAttr("cpu.realization") == b.getStringAttr(storage) &&
                 !task.hasAttr("cpu.storage"));
}
LogicalResult verifyMatrixPartition(func::FuncOp fn, Operation &region,
                                    const TargetProfile &p) {
  auto tile = region.getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile");
  auto caps = p.services.get<MatrixMachine>();
  if (!tile || tile.size() != 4 || !caps)
    return region.emitError("invalid matrix implementation");
  auto &dot = calculation(region);
  auto kind = dot.getName().stripDialect();
  auto packing = region.getAttrOfType<StringAttr>("cpu.packing");
  auto inType = inputDtype(dot, fn);
  if (caps->contracts.empty()) {
    if (!llvm::any_of(available(region, *caps), [&](auto &c) {
          return tile.asArrayRef() ==
                 ArrayRef<int64_t>{c.rows, c.cols, c.depth, c.lanes};
        }))
      return region.emitError("invalid matrix implementation");
  } else {
    auto implAttr = region.getAttrOfType<StringAttr>("cpu.implementation");
    if (!implAttr)
      return region.emitError("missing bound matrix implementation");
    int64_t packCode = 0;
    if (packing) {
      if (packing == "rhs-panels-per-call" || packing == "mapped-f32-panels")
        packCode = 1;
      else if (packing == caps->integerPacking && !caps->integerPacking.empty())
        packCode = 2;
      else if (packing == "f32-panels-per-call")
        packCode = 3;
      else if (packing == "vnni-pack32-panels")
        packCode = 4;
    }
    const MatrixImplementation *matched = nullptr;
    for (auto &c : caps->contracts) {
      if (c.id == implAttr.getValue() &&
          tile.asArrayRef() ==
              ArrayRef<int64_t>{c.rows, c.cols, c.depth, c.lanes} &&
          validContractPacking(kind, inType, c, packCode)) {
        matched = &c;
        break;
      }
    }
    if (!matched)
      return region.emitError("invalid matrix implementation or packing");
    if (auto err = p.checkImplementation(matched->id))
      return region.emitError() << llvm::toString(std::move(err));
    if (auto err = checkContractPreconditions(region, *matched, packCode, p))
      return region.emitError() << llvm::toString(std::move(err));
  }
  auto facts = p.services.get<CallFacts>();
  auto s = kir::contractionShape(dot, region.getOperand(0), *facts);
  bool routed = kind == "contract" && dot.getNumOperands() == 4;
  auto routing = region.getAttrOfType<StringAttr>("cpu.route_storage");
  if (routed ? (!routing || routing != "gather-once-private-output")
             : bool(routing))
    return region.emitError("routed matrix requires member-ordered input and "
                            "disjoint output scatter");
  bool validPack = false;
  if (kind == "contract") {
    validPack = !packing || packing == "rhs-panels-per-call" ||
                (packing == caps->integerPacking &&
                 groupedI8(dot, p, tile[1])) ||
                (packing == "f32-panels-per-call" &&
                 (inType == "f16" || inType == "bf16")) ||
                (packing == "vnni-pack32-panels" &&
                 (inType == "i8" || inType == "bf16"));
  } else if (kind == "window_contract") {
    validPack =
        packing && (packing == "mapped-right-per-call" ||
                    packing == "mapped-f32-panels");
  } else if (kind == "q4k_dot") {
    validPack = packing && packing == "shared-q8k-per-call";
  } else {
    validPack = packing && packing == "decoded-f32-per-call";
  }
  if (!validPack)
    return region.emitError("invalid shared activation packing contract");
  if (!s)
    return region.emitError() << llvm::toString(s.takeError());
  auto placement = region.getAttrOfType<StringAttr>("cpu.record_storage");
  if (kind == "q4k_dot"
          ? (!placement || placement != (s->k / 256 > 32 ? "heap" : "stack"))
          : bool(placement))
    return region.emitError("invalid Q8_K record workspace placement");
  auto sh = region.getAttrOfType<DenseI64ArrayAttr>("cpu.extents");
  if (!sh || sh.asArrayRef() != ArrayRef<int64_t>{s->m, s->n, s->k, s->batches})
    return region.emitError("incorrect task output coverage");
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides"),
         shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.shape");
    auto *sv = facts->shapeOf(v);
    auto *tv = facts->arguments[i].getAsObject()->getArray("strides");
    if (!sv || !tv || !strides || !shape || shape.size() != sv->size() ||
        strides.size() != tv->size())
      return region.emitError("missing CPU matrix layout");
    for (auto [a, b] : llvm::zip(shape.asArrayRef(), *sv))
      if (a != b.getAsInteger())
        return region.emitError("incorrect matrix extent");
    for (auto [a, b] : llvm::zip(strides.asArrayRef(), *tv))
      if (a != b.getAsInteger())
        return region.emitError("incorrect matrix stride");
  }
  return success();
}
std::string matrixHostParams(func::FuncOp fn) {
  SmallVector<std::string> params;
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    auto a = "a" + std::to_string(i);
    if (!isa<MemRefType>(v.getType())) {
      params.push_back((v.getType().isF32() ? "float " : "int32_t ") + a);
      continue;
    }
    int rank = cast<MemRefType>(v.getType()).getRank();
    params.push_back("void *" + a);
    if (rank > 2)
      for (int j = 0; j < rank - 1; ++j)
        params.push_back("int64_t " + a + "d" + std::to_string(j));
    for (int j = 0; j < rank; ++j)
      params.push_back("int64_t " + a + "s" + std::to_string(j));
  }
  return llvm::join(params, ",") + ",int64_t rows,int64_t columns,int64_t "
                                   "grain,int64_t panel,int64_t workers";
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
using namespace intentdsl;
void matrixTasks(Unit &u) {
  DecisionFamily f;
  f.name = "matrix-partition";
  f.applies = [](Operation *op) {
    return op->getName().getStringRef() == "cpu.matrix_domain" &&
           !op->hasAttr("cpu.output_tile");
  };
  f.read = [](Operation *op) { return op->getAttr("cpu.matrix_choice"); };
  f.write = [](Operation *op, Attribute a) {
    op->setAttr("cpu.matrix_choice", a);
    return llvm::Error::success();
  };
  f.verify = [](Operation *op, Attribute a,
                const TargetProfile &p) -> llvm::Error {
    auto c = p.services.get<cpu::MatrixMachine>();
    auto t = dyn_cast<DenseI64ArrayAttr>(a);
    if (!c || !t)
      return llvm::createStringError("unsupported CPU contraction block");
    auto &dot = cpu::calculation(*op);
    auto kind = dot.getName().stripDialect();
    if (c->contracts.empty()) {
      if (t.size() != 5 || t[4] < 0 || t[4] > 2 ||
          (t[4] != 0 && kind != "contract" &&
           !(t[4] == 1 && kind == "window_contract")) ||
          (t[4] == 2 && !cpu::groupedI8(dot, p, t[1])) ||
          !llvm::any_of(cpu::available(*op, *c), [&](auto &r) {
            return t.asArrayRef().take_front(4) ==
                   ArrayRef<int64_t>{r.rows, r.cols, r.depth, r.lanes};
          }))
        return llvm::createStringError("unsupported CPU contraction block");
      return llvm::Error::success();
    }
    if (t.size() != 6 || t[5] < 0 || size_t(t[5]) >= c->contracts.size())
      return llvm::createStringError("unsupported CPU contraction contract");
    const auto &impl = c->contracts[t[5]];
    auto fn = op->getParentOfType<func::FuncOp>();
    if (t.asArrayRef().take_front(4) !=
            ArrayRef<int64_t>{impl.rows, impl.cols, impl.depth, impl.lanes} ||
        !cpu::validContractPacking(kind, cpu::inputDtype(dot, fn), impl, t[4]))
      return llvm::createStringError("unsupported CPU contraction block");
    return llvm::Error::success();
  };
  f.verifyCandidate = [](Operation *op, const StructuralCandidate &c,
                         const TargetProfile &p) -> llvm::Error {
    auto t = cast<DenseI64ArrayAttr>(c.value);
    auto *caps = p.services.get<cpu::MatrixMachine>();
    if (caps->contracts.empty()) {
      if (c.topology != (t[4] ? "packed-output-tasks" : "output-tasks") ||
          c.id != std::to_string(t[0]) + "x" + std::to_string(t[1]) + "x" +
                      std::to_string(t[2]) +
                      (t[4] == 2 ? "-dot4"
                       : t[4]    ? "-pack"
                                 : "") ||
          c.preconditions != std::vector<std::string>{"disjoint-output-blocks"})
        return llvm::createStringError("invalid matrix task candidate");
      return llvm::Error::success();
    }
    const auto &impl = caps->contracts[t[5]];
    if (c.topology != (t[4] ? "packed-output-tasks" : "output-tasks") ||
        c.id != cpu::candidateName(impl, t[4]) ||
        c.preconditions !=
            std::vector<std::string>{"disjoint-output-blocks"} ||
        c.implementations != std::vector<std::string>{impl.id})
      return llvm::createStringError("invalid matrix task candidate");
    return cpu::checkContractPreconditions(*op, impl, t[4], p);
  };
  u.families.push_back(
      {{std::move(f), "cpu@formed", "cpu@formed"}, COMPILERINFRA_SOURCE});
  Strategy strategy;
  strategy.name = "contract-blocks";
  strategy.family = u.qualify("matrix-partition");
  strategy.enumerate = [](Operation *op, const TargetProfile &p,
                          AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    auto caps = p.services.get<cpu::MatrixMachine>();
    if (!caps)
      return llvm::createStringError("missing native matrix capability");
    Builder b(op->getContext());
    std::vector<StructuralCandidate> cs;
    auto &dot = cpu::calculation(*op);
    auto kind = dot.getName().stripDialect();
    if (caps->contracts.empty()) {
      for (auto &r : cpu::available(*op, *caps))
        for (int pack : {0, 1, 2}) {
          if (pack && kind != "contract" &&
              !(pack == 1 && kind == "window_contract"))
            continue;
          if (pack == 2 && !cpu::groupedI8(dot, p, r.cols))
            continue;
          cs.push_back(
              {std::to_string(r.rows) + "x" + std::to_string(r.cols) + "x" +
                   std::to_string(r.depth) +
                   (pack == 2 ? "-dot4"
                    : pack    ? "-pack"
                              : ""),
               pack ? "packed-output-tasks" : "output-tasks",
               b.getDenseI64ArrayAttr({r.rows, r.cols, r.depth, r.lanes, pack}),
               {"disjoint-output-blocks"}});
        }
      return cs;
    }
    auto fn = op->getParentOfType<func::FuncOp>();
    auto inType = cpu::inputDtype(dot, fn);
    for (auto [idx, impl] : llvm::enumerate(caps->contracts)) {
      for (int64_t pack : {0, 1, 2, 3, 4}) {
        if (!cpu::validContractPacking(kind, inType, impl, pack))
          continue;
        StructuralCandidate cand;
        cand.id = cpu::candidateName(impl, pack);
        cand.topology = pack ? "packed-output-tasks" : "output-tasks";
        cand.value = b.getDenseI64ArrayAttr(
            {impl.rows, impl.cols, impl.depth, impl.lanes, pack,
             int64_t(idx)});
        cand.preconditions = {"disjoint-output-blocks"};
        cand.implementations = {impl.id};
        cs.push_back(std::move(cand));
      }
    }
    return cs;
  };
  u.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "own-output-blocks";
  r.family = u.qualify("matrix-partition");
  r.apply = [](Operation *op, Attribute t, const TargetProfile &p,
               RewriterBase &b) -> llvm::Error {
    auto facts = p.services.get<CallFacts>();
    auto &dot = cpu::calculation(*op);
    auto kind = dot.getName().stripDialect();
    auto g = kir::contractionShape(dot, op->getOperand(0), *facts);
    if (!g)
      return g.takeError();
    if (g->routed)
      op->setAttr("cpu.route_storage",
                  b.getStringAttr("gather-once-private-output"));
    auto row = cast<DenseI64ArrayAttr>(t);
    auto *caps = p.services.get<cpu::MatrixMachine>();
    op->setAttr("cpu.output_tile",
                b.getDenseI64ArrayAttr(row.asArrayRef().take_front(4)));
    auto pack = cpu::packingFor(kind, row[4], *caps);
    if (!pack.empty())
      op->setAttr("cpu.packing", b.getStringAttr(pack));
    if (row.size() == 6)
      op->setAttr("cpu.implementation",
                  b.getStringAttr(caps->contracts[row[5]].id));
    op->setAttr("cpu.extents",
                b.getDenseI64ArrayAttr({g->m, g->n, g->k, g->batches}));
    if (kind == "q4k_dot")
      op->setAttr("cpu.record_storage",
                  b.getStringAttr(g->k / 256 > 32 ? "heap" : "stack"));
    auto fn = op->getParentOfType<func::FuncOp>();
    for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
      for (auto key : {"shape", "strides"}) {
        SmallVector<int64_t> axes;
        for (auto &dim : *facts->arguments[i].getAsObject()->getArray(key))
          axes.push_back(*dim.getAsInteger());
        fn.setArgAttr(i, std::string("cpu.") + key,
                      b.getDenseI64ArrayAttr(axes));
      }
    }
    op->removeAttr("cpu.matrix_choice");
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
