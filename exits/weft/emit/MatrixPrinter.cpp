#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

namespace {
std::string irId(PrintContext &c, Value v) { return "%" + c.name(v); }

std::string formatRecordDecl(bool weight) {
  struct Field {
    std::string name, type, layout;
    int extent, offset, bits;
  };
  const std::string natural = "[{kind=\"natural\"}]";
  auto joined = [](int role) {
    return "[{kind=\"joined\",fields=2:i64,low_bits=4:i64,order=\"lo_first\","
           "size=4:i64,role=" +
           std::to_string(role) + ":i64}]";
  };
  const std::string nibble = "[{kind=\"grouped\",size=64:i64},{kind="
                             "\"layered\",size=32:i64,order=\"lo_first\"}]";
  std::vector<Field> fields =
      weight ? std::vector<Field>{{"d", "f16", natural, 1, 0, 16},
                                  {"dmin", "f16", natural, 1, 16, 16},
                                  {"sc", "ui6", joined(0), 8, 32, 96},
                                  {"m", "ui6", joined(1), 8, 32, 96},
                                  {"q", "ui4", nibble, 256, 128, 1024}}
             : std::vector<Field>{{"ds", "f32", natural, 1, 0, 32},
                                  {"q", "si8", natural, 256, 32, 2048},
                                  {"bsum", "si16", natural, 16, 2080, 256}};
  SmallVector<std::string> names, types, layouts, shapes, offsets, bits;
  for (auto &f : fields) {
    names.push_back("\"" + f.name + "\"");
    types.push_back(f.type);
    layouts.push_back(f.layout);
    shapes.push_back(f.extent == 1
                         ? "array<i64>"
                         : "array<i64:" + std::to_string(f.extent) + ">");
    offsets.push_back(std::to_string(f.offset));
    bits.push_back(std::to_string(f.bits));
  }
  std::string name = weight ? "Q4_K" : "Q8_K";
  return "\"weft_kernel.encoding_decl\"() <{sym_name=\"" + name +
         "\",kind=\"base\",layout_identity=\"" + name +
         "\",bit_order=\"lsb_first\",byte_order=\"little\",alignment=" +
         std::to_string(weight ? 2 : 4) + ":i64,elements=256:i64,storage_bits=" +
         std::to_string((weight ? 144 : 292) * 8) + ":i64,field_names=[" +
         llvm::join(names, ",") + "],field_types=[" + llvm::join(types, ",") +
         "],field_shapes=[" + llvm::join(shapes, ",") + "],field_layouts=[" +
         llvm::join(layouts, ",") + "],field_bit_offsets=array<i64:" +
         llvm::join(offsets, ",") + ">,field_storage_bits=array<i64:" +
         llvm::join(bits, ",") + ">,padding=array<i64>}>:()->()";
}
} // namespace

TerminalPrinter weftMatrixPrinter() {
  TerminalPrinter p;
  auto noop = [](Operation &, PrintPoint, PrintContext &) {
    return llvm::Error::success();
  };
  p.operations["builtin.module"] = noop;
  p.operations["func.func"] = noop;
  p.operations["func.return"] = noop;
  p.operations["weft_host.end"] = noop;

  p.operations["weft_host.matrix_program"] = [](Operation &, PrintPoint at,
                                                PrintContext &c) {
    if (at == PrintPoint::BeginRegion) {
      c.file = c.region == 0 ? "task.weftir" : "host.c";
      c.indent = 0;
      if (c.region == 0)
        c.line("module {");
    } else if (at == PrintPoint::EndRegion) {
      if (c.region == 0)
        c.line("}");
    }
    return llvm::Error::success();
  };

  // ---------------- Region 0 ("task.weftir") Ops ----------------
  p.operations["weft_host.encoding_decl"] = [](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(formatRecordDecl(op.getAttrOfType<BoolAttr>("is_weight").getValue()));
    return llvm::Error::success();
  };

  p.operations["weft_host.task_kernel"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto &bb = op.getRegion(0).front();
      auto vTypes = op.getAttrOfType<ArrayAttr>("view_types");
      auto writes = op.getAttrOfType<ArrayAttr>("arg_writes");
      auto strided = op.getAttrOfType<DenseI64ArrayAttr>("strided_slots");
      SmallVector<std::string> names, effects, aliases, params, stridedNames;
      for (size_t i = 0; i < vTypes.size(); ++i) {
        std::string aName = "a" + std::to_string(i);
        names.push_back("\"" + aName + "\"");
        effects.push_back(cast<BoolAttr>(writes[i]).getValue() ? "\"write\""
                                                               : "\"read\"");
        aliases.push_back("0");
        params.push_back(irId(c, bb.getArgument(i)) + ":" +
                         cast<StringAttr>(vTypes[i]).str());
      }
      for (int64_t idx : strided.asArrayRef())
        stridedNames.push_back("\"a" + std::to_string(idx) + "\"");

      c.line("\"weft_kernel.kernel\"() <{sym_name=\"" +
             op.getAttrOfType<StringAttr>("sym_name").str() + "\",source=\"" +
             op.getAttrOfType<StringAttr>("source").str() + "\",arg_names=[" +
             llvm::join(names, ",") + "],arg_access=[" +
             llvm::join(effects, ",") + "],arg_alias_sets=array<i64:" +
             llvm::join(aliases, ",") +
             ">,shape_symbols=[\"R\",\"C\"],strided_args=[" +
             llvm::join(stridedNames, ",") + "]}> ({");
      c.line("^bb0(" + llvm::join(params, ",") + "):");
      size_t nViews = vTypes.size();
      c.line(irId(c, bb.getArgument(nViews)) +
             "=\"weft_kernel.symbol\"() "
             "<{choices=array<i64>,kind=\"shape\",name=\"R\"}>:()->index");
      c.line(irId(c, bb.getArgument(nViews + 1)) +
             "=\"weft_kernel.symbol\"() "
             "<{choices=array<i64>,kind=\"shape\",name=\"C\"}>:()->index");
      c.line(irId(c, bb.getArgument(nViews + 2)) +
             "=\"weft_kernel.root_domain\"():()->!weft_kernel.domain<\"root\","
             "0,-1,0,\"root\",\"exact\">");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("\"weft_kernel.return\"():()->()");
      c.indent--;
      c.line("}):()->()");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_const"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto lit = op.getAttrOfType<StringAttr>("lit").str();
    auto ty = op.getAttrOfType<StringAttr>("val_type").str();
    if (ty == "index")
      c.line(irId(c, op.getResult(0)) + "=arith.constant " + lit + ":index");
    else
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.constant\"() <{value=" +
             lit + ":" + ty + "}>:()->" + ty);
    return llvm::Error::success();
  };

  p.operations["weft_host.task_new"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.new\"(" +
             irId(c, op.getOperand(0)) + ") <{initialized=true}>:(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_slice"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto vTy = op.getAttrOfType<StringAttr>("view_type").str();
    auto slTy = op.getAttrOfType<StringAttr>("slice_type").str();
    auto vlTy = op.getAttrOfType<StringAttr>("val_type").str();
    bool admit = op.getAttrOfType<BoolAttr>("admit").getValue();
    SmallVector<std::string> sels, argNames, argTys{vTy};
    for (Attribute a : op.getAttrOfType<ArrayAttr>("selectors"))
      sels.push_back("\"" + cast<StringAttr>(a).str() + "\"");
    for (auto [idx, v] : llvm::enumerate(op.getOperands())) {
      argNames.push_back(irId(c, v));
      if (idx > 0)
        argTys.push_back("index");
    }
    std::string resId = irId(c, op.getResult(0));
    std::string slId = admit ? (resId + "_s") : resId;
    c.line(slId + "=\"weft_kernel.slice\"(" + llvm::join(argNames, ",") +
           ") <{selectors=[" + llvm::join(sels, ",") + "]}>:(" +
           llvm::join(argTys, ",") + ")->" + slTy);
    if (admit)
      c.line(resId + "=\"weft_kernel.admit\"(" + slId + "):(" + slTy + ")->" +
             vlTy);
    return llvm::Error::success();
  };

  p.operations["weft_host.task_binary"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.binary\"(" +
             irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) +
             ") <{kind=\"" + op.getAttrOfType<StringAttr>("kind").str() +
             "\"}>:(" + op.getAttrOfType<StringAttr>("lhs_type").str() + "," +
             op.getAttrOfType<StringAttr>("rhs_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_compare"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.compare\"(" +
             irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) +
             ") <{predicate=\"" +
             op.getAttrOfType<StringAttr>("predicate").str() + "\"}>:(" +
             op.getAttrOfType<StringAttr>("lhs_type").str() + "," +
             op.getAttrOfType<StringAttr>("rhs_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_select"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.select\"(" +
             irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) + "," +
             irId(c, op.getOperand(2)) + "):(" +
             op.getAttrOfType<StringAttr>("cond_type").str() + "," + vTy + "," +
             vTy + ")->" + vTy);
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_cast"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel." +
             op.getAttrOfType<StringAttr>("op_kind").str() + "\"(" +
             irId(c, op.getOperand(0)) + "):(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_field"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.field\"(" +
             irId(c, op.getOperand(0)) + ") <{name=\"" +
             op.getAttrOfType<StringAttr>("field_name").str() + "\"}>:(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_fold2"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.fold2\"(" +
             irId(c, op.getOperand(0)) + "):(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_contract"] = [](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto kind = op.getAttrOfType<StringAttr>("kind").getValue();
    auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
    auto inTys = op.getAttrOfType<ArrayAttr>("in_types");
    std::string resId = irId(c, op.getResult(0));
    if (kind == "fma") {
      std::string prodId = resId + "_m";
      std::string tB = cast<StringAttr>(inTys[1]).str();
      std::string tA = cast<StringAttr>(inTys[2]).str();
      c.line(prodId + "=\"weft_kernel.binary\"(" + irId(c, op.getOperand(1)) +
             "," + irId(c, op.getOperand(2)) + ") <{kind=\"mul\"}>:(" + tB +
             "," + tA + ")->" + vTy);
      c.line(resId + "=\"weft_kernel.binary\"(" + irId(c, op.getOperand(0)) +
             "," + prodId + ") <{kind=\"add\"}>:(" + vTy + "," + vTy + ")->" +
             vTy);
      return llvm::Error::success();
    }
    SmallVector<std::string> ov;
    for (int64_t x : op.getAttrOfType<DenseI64ArrayAttr>("over").asArrayRef())
      ov.push_back(std::to_string(x));
    std::string attrs = "over=array<i64:" + llvm::join(ov, ",") + ">,acc_type=" +
                        op.getAttrOfType<StringAttr>("acc_type").str() +
                        ",engine=\"" +
                        op.getAttrOfType<StringAttr>("engine").str() + "\"";
    if (op.getAttrOfType<BoolAttr>("allow_subnormal_ftz").getValue())
      attrs += ",allow_subnormal_ftz=true";
    c.line(resId + "=\"weft_kernel." + kind.str() + "\"(" +
           irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) +
           ") <{" + attrs + "}>:(" + cast<StringAttr>(inTys[0]).str() + "," +
           cast<StringAttr>(inTys[1]).str() + ")->" + vTy);
    return llvm::Error::success();
  };

  p.operations["weft_host.task_for"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    auto &bb = op.getRegion(0).front();
    auto iterTys = op.getAttrOfType<ArrayAttr>("iter_types");
    if (at == PrintPoint::Enter) {
      std::string iv = irId(c, bb.getArgument(0));
      std::string lb = irId(c, op.getOperand(0));
      std::string ub = irId(c, op.getOperand(1));
      std::string st = irId(c, op.getOperand(2));
      if (op.getNumResults() == 0) {
        c.line("scf.for " + iv + "=" + lb + " to " + ub + " step " + st + " {");
      } else {
        SmallVector<std::string> carries, tys, resDefs;
        for (size_t i = 0; i < iterTys.size(); ++i) {
          carries.push_back(irId(c, bb.getArgument(1 + i)) + "=" +
                            irId(c, op.getOperand(3 + i)));
          tys.push_back(cast<StringAttr>(iterTys[i]).str());
        }
        if (op.getNumResults() == 1) {
          c.line(irId(c, op.getResult(0)) + "=scf.for " + iv + "=" + lb +
                 " to " + ub + " step " + st + " iter_args(" +
                 llvm::join(carries, ",") + ")->(" + llvm::join(tys, ",") +
                 ") {");
        } else {
          std::string tup = irId(c, op.getResult(0)) + "_t";
          c.line(tup + ":" + std::to_string(op.getNumResults()) + "=scf.for " +
                 iv + "=" + lb + " to " + ub + " step " + st + " iter_args(" +
                 llvm::join(carries, ",") + ")->(" + llvm::join(tys, ",") +
                 ") {");
        }
      }
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
      if (op.getNumResults() > 1) {
        std::string tup = irId(c, op.getResult(0)) + "_t";
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          std::string t = cast<StringAttr>(iterTys[i]).str();
          std::string zId = irId(c, op.getResult(i)) + "_z";
          std::string scTy =
              t.find("f32") != std::string::npos ? "f32" : "si32";
          std::string zLit = scTy == "f32" ? "0.0" : "0";
          c.line(zId + "=\"weft_kernel.constant\"() <{value=" + zLit + ":" +
                 scTy + "}>:()->" + scTy);
          c.line(irId(c, op.getResult(i)) + "=\"weft_kernel.binary\"(" + tup +
                 "#" + std::to_string(i) + "," + zId + ") <{kind=\"add\"}>:(" +
                 t + "," + scTy + ")->" + t);
        }
      }
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_if"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    auto resTys = op.getAttrOfType<ArrayAttr>("res_types");
    SmallVector<std::string> tys;
    for (Attribute a : resTys)
      tys.push_back(cast<StringAttr>(a).str());
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=scf.if " + irId(c, op.getOperand(0)) +
             " -> (" + llvm::join(tys, ",") + ") {");
    } else if (at == PrintPoint::BeginRegion) {
      if (c.region == 1)
        c.line("} else {");
      c.indent++;
    } else if (at == PrintPoint::EndRegion) {
      c.indent--;
      if (c.region == 1)
        c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_yield"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at != PrintPoint::Enter || op.getNumOperands() == 0)
      return llvm::Error::success();
    SmallVector<std::string> vals, tys;
    for (Value v : op.getOperands())
      vals.push_back(irId(c, v));
    for (Attribute a : op.getAttrOfType<ArrayAttr>("val_types"))
      tys.push_back(cast<StringAttr>(a).str());
    c.line("scf.yield " + llvm::join(vals, ",") + ":" + llvm::join(tys, ","));
    return llvm::Error::success();
  };

  p.operations["weft_host.task_subquant_level"] = [](Operation &op,
                                                     PrintPoint at,
                                                     PrintContext &c) {
    std::string spec = "\"subquant.1\",1,0,3,\"blocks\",\"exact\"";
    std::string domTy = "!weft_kernel.domain<" + spec + ">";
    std::string ptTy = "!weft_kernel.point<<" + spec + ">>";
    std::string base = irId(c, op.getResult(0));
    if (at == PrintPoint::Enter) {
      auto &bb = op.getRegion(0).front();
      c.line(base + "_e=arith.constant 256:index");
      c.line(base + "_s=arith.constant 32:index");
      c.line(base + "_c=arith.ceildivui " + base + "_e," + base + "_s:index");
      c.line(base + "_d=\"weft_kernel.domain\"(" + irId(c, op.getOperand(0)) +
             "," + base + "_e," + base + "_s," + base + "_c):(!"
             "weft_kernel.domain<\"root\",0,-1,0,\"root\",\"exact\">,index,"
             "index,index)->" +
             domTy);
      c.line(base + "_p:2=\"weft_kernel.level\"(" + base + "_d," +
             irId(c, op.getOperand(1)) + "," + irId(c, op.getOperand(2)) +
             ") ({^bb0(%b0:" + ptTy + "):\n\"weft_kernel.births_yield\"():()->()\n},"
             "{^bb0(%b1:" + ptTy + "):\n\"weft_kernel.births_yield\"():()->()\n},"
             "{^bb0(" +
             irId(c, bb.getArgument(0)) + ":" + ptTy + "," +
             irId(c, bb.getArgument(1)) + ":si32," +
             irId(c, bb.getArgument(2)) + ":si32):");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}):(" + domTy + ",si32,si32)->(si32,si32)");
      c.line(base + "_z=\"weft_kernel.constant\"() <{value=0:si32}>:()->si32");
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.binary\"(" + base +
             "_p#0," + base + "_z) <{kind=\"add\"}>:(si32,si32)->si32");
      c.line(irId(c, op.getResult(1)) + "=\"weft_kernel.binary\"(" + base +
             "_p#1," + base + "_z) <{kind=\"add\"}>:(si32,si32)->si32");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_extract"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.extract\"(" +
             irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) +
             ") <{selectors=[\"" +
             op.getAttrOfType<StringAttr>("selector").str() + "\"]}>:(" +
             op.getAttrOfType<StringAttr>("in_type").str() + "," +
             op.getAttrOfType<StringAttr>("point_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_handoff"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("\"weft_kernel.handoff\"(" + irId(c, op.getOperand(0)) + "," +
             irId(c, op.getOperand(1)) + "):(si32,si32)->()");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.task_commit"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("\"weft_kernel.commit\"(" + irId(c, op.getOperand(0)) + "," +
             irId(c, op.getOperand(1)) + "):(" +
             op.getAttrOfType<StringAttr>("val_type").str() + "," +
             op.getAttrOfType<StringAttr>("slice_type").str() + ")->()");
    }
    return llvm::Error::success();
  };

  // ---------------- Region 1 ("host.c") Ops ----------------
  p.operations["arith.constant"] = [](Operation &op, PrintPoint at,
                                      PrintContext &c) -> llvm::Error {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    if (auto fa = op.getAttrOfType<FloatAttr>("value")) {
      llvm::SmallString<24> num;
      fa.getValue().toString(num);
      std::string s = num.str().str();
      if (s.find_first_of(".Ee") == std::string::npos)
        s += ".0";
      c.line("const float " + c.name(op.getResult(0)) + " = " + s + "f;");
    } else {
      int64_t val = op.getAttrOfType<IntegerAttr>("value").getInt();
      c.line("const int64_t " + c.name(op.getResult(0)) + " = " +
             std::to_string(val) + "LL;");
    }
    return llvm::Error::success();
  };

  auto hostBin = [&](StringRef opName, StringRef cSym, bool isFloat) {
    p.operations[opName.str()] = [cSym, isFloat](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
      if (at == PrintPoint::Enter) {
        std::string cTy = isFloat ? "const float " : "const int64_t ";
        c.line(cTy + c.name(op.getResult(0)) + " = " +
               c.name(op.getOperand(0)) + " " + cSym.str() + " " +
               c.name(op.getOperand(1)) + ";");
      }
      return llvm::Error::success();
    };
  };
  hostBin("arith.addi", "+", false);
  hostBin("arith.subi", "-", false);
  hostBin("arith.muli", "*", false);
  hostBin("arith.divsi", "/", false);
  hostBin("arith.remsi", "%", false);
  hostBin("arith.addf", "+", true);
  hostBin("arith.subf", "-", true);
  hostBin("arith.mulf", "*", true);

  p.operations["arith.extf"] = [](Operation &op, PrintPoint at,
                                  PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = expand_bf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  };
  p.operations["arith.truncf"] = [](Operation &op, PrintPoint at,
                                    PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const uint16_t " + c.name(op.getResult(0)) + " = narrow_bf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  };
  p.operations["arith.minimumf"] = [](Operation &op, PrintPoint at,
                                      PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = min_ieee(" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ");");
    return llvm::Error::success();
  };
  p.operations["math.exp"] = [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = expf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  };

  p.operations["weft_host.address"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
    int32_t bits = op.getAttrOfType<IntegerAttr>("index_bits").getInt();
    std::string cTy = bits == 32 ? "int32_t" : "int64_t";
    SmallVector<std::string> terms{"(" + cTy + ")0"};
    for (auto [v, st] : llvm::zip(op.getOperands(), strides.asArrayRef())) {
      std::string stLit =
          bits == 32 ? ("(" + cTy + ")" + std::to_string(st))
                     : (std::to_string(st) + "LL");
      terms.push_back("(" + cTy + ")(" + c.name(v) + ") * " + stLit);
    }
    c.line("const " + cTy + " " + c.name(op.getResult(0)) + " = " +
           llvm::join(terms, " + ") + ";");
    return llvm::Error::success();
  };

  p.operations["weft_host.host_fn"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("#include <math.h>");
      c.line("#include <stddef.h>");
      c.line("#include <stdint.h>");
      c.line("#include <stdlib.h>");
      c.line("#include <string.h>");
      c.line("#if defined(__x86_64__) && (defined(__AVX512F__) || defined(__F16C__))");
      c.line("#include <immintrin.h>");
      c.line("#endif");
      c.line("#include \"task_bridge.h\"");

      c.line("struct IntentWsSlot { void *ptr; size_t cap; int busy; };");
      c.line("static _Thread_local struct IntentWsSlot g_intent_ws[4];");
      c.line("static inline void *intent_ws_acquire(int slot, size_t bytes) {");
      c.line("  size_t need = ((bytes ? bytes : 64u) + 63u) & ~(size_t)63u;");
      c.line("  if ((unsigned)slot < 4u && __atomic_exchange_n(&g_intent_ws[slot].busy, 1, __ATOMIC_ACQUIRE) == 0) {");
      c.line("    if (g_intent_ws[slot].cap < need) {");
      c.line("      void *np = aligned_alloc(64, need);");
      c.line("      if (!np) { __atomic_store_n(&g_intent_ws[slot].busy, 0, __ATOMIC_RELEASE); return NULL; }");
      c.line("      free(g_intent_ws[slot].ptr); g_intent_ws[slot].ptr = np; g_intent_ws[slot].cap = need;");
      c.line("    }");
      c.line("    return g_intent_ws[slot].ptr;");
      c.line("  }");
      c.line("  return aligned_alloc(64, need);");
      c.line("}");
      c.line("static inline void intent_ws_release(int slot, void *p) {");
      c.line("  if (!p) return;");
      c.line("  if ((unsigned)slot < 4u && p == g_intent_ws[slot].ptr) { __atomic_store_n(&g_intent_ws[slot].busy, 0, __ATOMIC_RELEASE); return; }");
      c.line("  free(p);");
      c.line("}");

      if (op.getAttrOfType<BoolAttr>("uses_q8k").getValue()) {
        using Q8 = intentdsl::kir::Q8KRecord;
        c.line("struct Record { float d; int8_t q[" +
               std::to_string(Q8::elements) + "]; int16_t sums[" +
               std::to_string(Q8::elements / Q8::sumGroup) + "]; };");
        c.line("_Static_assert(sizeof(struct Record)==" +
               std::to_string(Q8::bytes) + ", \"Q8_K record ABI\");");
        c.line("static inline void pack_q8_record(struct Record *out, const float *x, int64_t stride) {");
        c.line("  float extreme=0.0f;");
        c.line("  for (int64_t k=0; k<256; ++k) { float v=x[k*stride]; if (fabsf(v)>fabsf(extreme)) extreme=v; }");
        c.line("  float inv=extreme==0.0f ? 0.0f : -127.0f/extreme;");
        c.line("  out->d=extreme==0.0f ? 0.0f : 1.0f/inv;");
        c.line("  for (int group=0; group<16; ++group) {");
        c.line("    int sum=0;");
        c.line("#if defined(__x86_64__)");
        c.line("    #pragma omp simd reduction(+:sum)");
        c.line("#endif");
        c.line("    for (int64_t k=16*group; k<16*group+16; ++k) {");
        c.line("      float v=x[k*stride]*inv; int q=(int)v; float fraction=fabsf(v-(float)q);");
        c.line("      if (fraction>0.5f || (fraction==0.5f && (q&1))) q+=v<0 ? -1 : 1;");
        c.line("      out->q[k]=(int8_t)(q>127 ? 127 : q); sum+=out->q[k];");
        c.line("    }");
        c.line("    out->sums[group]=(int16_t)sum;");
        c.line("  }");
        c.line("}");
      }
      if (op.getAttrOfType<BoolAttr>("uses_bf16_math").getValue()) {
        c.line("static inline float expand_bf(uint16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}");
        c.line("static inline uint16_t narrow_bf(float f){uint32_t b;memcpy(&b,&f,4);if((b&0x7fffffff)>0x7f800000)return (b>>16)|64;return (b+32767+((b>>16)&1))>>16;}");
        c.line("static inline float min_ieee(float a,float b){return isnan(a)?a:isnan(b)?b:fminf(a,b);}");
        c.line("static inline void widen_bf16_span(float *out, const uint16_t *src, int64_t n){");
        c.line("  if (n <= 0) return; int64_t i=0;");
        c.line("#if defined(__AVX512F__)");
        c.line("  for (; i <= n - 16; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16))); }");
        c.line("#endif");
        c.line("  for (; i<n; ++i) out[i] = expand_bf(src[i]);");
        c.line("}");
      }
      if (op.getAttrOfType<BoolAttr>("uses_f16_math").getValue()) {
        c.line("static inline float expand_f16(uint16_t h){");
        c.line("#if defined(__F16C__)");
        c.line("  return _cvtsh_ss(h);");
        c.line("#else");
        c.line("  uint32_t sign=(uint32_t)(h&0x8000u)<<16, exp=(h>>10)&0x1fu, mant=h&0x03ffu, bits;");
        c.line("  if(exp==0){ if(mant==0){bits=sign;float z;memcpy(&z,&bits,4);return z;} return (h&0x8000u?-1.0f:1.0f)*(float)mant*(1.0f/16777216.0f); }");
        c.line("  if(exp==31) bits=sign|0x7f800000u|(mant<<13)|(mant?0x00400000u:0u);");
        c.line("  else bits=sign|((exp+112u)<<23)|(mant<<13);");
        c.line("  float f; memcpy(&f,&bits,4); return f;");
        c.line("#endif");
        c.line("}");
        c.line("static inline void widen_f16_span(float *out, const uint16_t *src, int64_t n){");
        c.line("  if (n <= 0) return; int64_t i=0;");
        c.line("#if defined(__AVX512F__) && defined(__F16C__)");
        c.line("  for (; i <= n - 16; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_cvtph_ps(h)); }");
        c.line("#endif");
        c.line("  for (; i<n; ++i) out[i] = expand_f16(src[i]);");
        c.line("}");
      }
      if (op.getAttrOfType<BoolAttr>("uses_mx8_math").getValue()) {
        using MX = intentdsl::kir::MX8Block;
        c.line("static float g_mx8_val[256], g_mx8_scale[256];");
        c.line("static int g_mx8_ready = 0;");
        c.line("static void ensure_mx8_tables(void) {");
        c.line("  if (__atomic_load_n(&g_mx8_ready, __ATOMIC_ACQUIRE)) return;");
        c.line("  for (unsigned x=0; x<256; ++x) {");
        c.line("    unsigned mag=x&127u, ex=mag>>3, m=mag&7u, sign=(x&128u)<<24;");
        c.line("    if (ex == 0) g_mx8_val[x] = (x&128u ? -1.0f : 1.0f) * (float)m * 0.001953125f;");
        c.line("    else { uint32_t bits = sign | (mag==127u ? 0x7fc00000u : ((ex+120u)<<23) | (m<<20)); memcpy(&g_mx8_val[x], &bits, 4); }");
        c.line("    uint32_t sbits = x==" + std::to_string(MX::nan) +
               "u ? 0x7fc00000u : x==0u ? 0x00400000u : (x<<23);");
        c.line("    memcpy(&g_mx8_scale[x], &sbits, 4);");
        c.line("  }");
        c.line("  __atomic_store_n(&g_mx8_ready, 1, __ATOMIC_RELEASE);");
        c.line("}");
      }

      auto ext = op.getAttrOfType<DenseI64ArrayAttr>("extents");
      auto tl = op.getAttrOfType<DenseI64ArrayAttr>("tile");
      c.line("int32_t intent_entry(" +
             op.getAttrOfType<StringAttr>("params").str() + ") {");
      c.indent++;
      c.line("const int64_t M=" + std::to_string(ext[0]) +
             ", N=" + std::to_string(ext[1]) + ", K=" + std::to_string(ext[2]) +
             ", G=" + std::to_string(ext[3]) + ";");
      c.line("const int64_t mr=" + std::to_string(tl[0]) +
             ", nr=" + std::to_string(tl[1]) + ";");
      c.line("const int64_t mb=mr>0?(M+mr-1)/mr:0, nc=nr>0?(N+nr-1)/nr:0, jobs=G*mb*nc;");
      c.line("(void)rows; (void)columns; (void)grain; (void)panel;");
      if (op.getAttrOfType<BoolAttr>("uses_mx8_math").getValue())
        c.line("ensure_mx8_tables();");
      c.indent--;
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("return 0;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.host_alloc"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto name = op.getAttrOfType<StringAttr>("buf_name").str();
    auto place = op.getAttrOfType<StringAttr>("placement").getValue();
    auto cType = op.getAttrOfType<StringAttr>("c_type").str();
    auto count = op.getAttrOfType<StringAttr>("count_expr").str();
    int wsSlot = name == "left" ? 0 : name == "rhs" ? 1 : name == "members" ? 2 : 3;
    if (place == "stack") {
      c.line("_Alignas(64) " + cType + " " + name + "[" + count + "];");
    } else {
      std::string frees;
      for (Attribute a : op.getAttrOfType<ArrayAttr>("cleanup")) {
        std::string prev = cast<StringAttr>(a).str();
        int pSlot = prev == "left" ? 0 : prev == "rhs" ? 1 : prev == "members" ? 2 : 3;
        frees += "intent_ws_release(" + std::to_string(pSlot) + ", " + prev + ");";
      }
      c.line(cType + " *" + name + " = (" + cType + "*)intent_ws_acquire(" +
             std::to_string(wsSlot) + ", (size_t)(" + count + ") * sizeof(" +
             cType + "));");
      c.line("if (!" + name + ") { " + frees + "return 2; }");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.host_col_loop"] = [](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    auto *prep = op.getParentOp();
    auto slots = prep->getAttrOfType<DenseI64ArrayAttr>("src_slots");
    auto st = prep->getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
    if (at == PrintPoint::Enter) {
      auto &varBb = op.getRegion(0).front();
      c.line("#pragma omp simd");
      c.line("for (int64_t n=0; n<N; ++n) {");
      c.indent++;
      c.line("const uint16_t " + c.name(varBb.getArgument(0)) +
             " = ((const uint16_t*)a" + std::to_string(slots[0]) + ")[batch*" +
             std::to_string(st[0]) + "LL+(chunk*K+k)*" +
             std::to_string(st[1]) + "LL+group*" + std::to_string(st[2]) +
             "LL+n*" + std::to_string(st[3]) + "LL];");
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.host_panel_yield"] = [](Operation &op, PrintPoint at,
                                                  PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto *prep = op.getParentOp()->getParentOp();
    bool wide = prep->getAttrOfType<StringAttr>("kind").getValue() ==
                "map-window-rhs-f32";
    std::string dst = prep->getAttrOfType<StringAttr>("dst_buf").str();
    std::string v = c.name(op.getOperand(0));
    c.line(dst + "[p*N+n] = " + (wide ? "expand_bf(" + v + ")" : v) + ";");
    return llvm::Error::success();
  };

  p.operations["weft_host.host_prep"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) -> llvm::Error {
    auto pk = op.getAttrOfType<StringAttr>("kind").getValue();
    bool isMap = pk == "map-window-rhs-f32" || pk == "map-window-rhs-bf16";
    if (isMap) {
      auto slots = op.getAttrOfType<DenseI64ArrayAttr>("src_slots");
      auto access = [&](int idx, StringRef cTy, ArrayRef<std::string> coords) {
        auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_" +
                                                      std::to_string(idx));
        std::string s =
            "((" + cTy.str() + "*)a" + std::to_string(slots[idx]) + ")[0";
        for (auto [ax, e] : llvm::enumerate(coords))
          s += "+(" + e + ")*" + std::to_string(st[ax]) + "LL";
        return s + "]";
      };
      if (at == PrintPoint::Enter) {
        int64_t H = op.getAttrOfType<IntegerAttr>("head_extent").getInt();
        int64_t chunks = op.getAttrOfType<IntegerAttr>("chunk_extent").getInt();
        int64_t hg = op.getAttrOfType<IntegerAttr>("head_group").getInt();
        c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(G*K>1)");
        c.line("for (int64_t p=0; p<G*K; ++p) {");
        c.indent++;
        c.line("const int64_t g=p/K, k=p%K, head=g%" + std::to_string(H) +
               ", chunk=g/" + std::to_string(H) + "%" + std::to_string(chunks) +
               ", batch=g/" + std::to_string(H) + "/" + std::to_string(chunks) +
               ", group=head/" + std::to_string(hg) + ";");
        auto &invBb = op.getRegion(0).front();
        c.line("const float " + c.name(invBb.getArgument(0)) + " = " +
               access(1, "const float", {"batch", "head", "chunk", "k"}) + ";");
        c.line("const float " + c.name(invBb.getArgument(1)) + " = " +
               access(2, "const float", {"batch", "head", "chunk", "k"}) + ";");
        c.line("const float " + c.name(invBb.getArgument(2)) + " = " +
               access(2, "const float", {"batch", "head", "chunk", "K-1"}) + ";");
      } else if (at == PrintPoint::EndBlock) {
        c.indent--;
        c.line("}");
      }
      return llvm::Error::success();
    }

    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto slots = op.getAttrOfType<DenseI64ArrayAttr>("src_slots");
    auto dst = op.getAttrOfType<StringAttr>("dst_buf").str();

    if (pk == "widen-window-lhs") {
      int64_t H = op.getAttrOfType<IntegerAttr>("head_extent").getInt();
      int64_t chunks = op.getAttrOfType<IntegerAttr>("chunk_extent").getInt();
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(G>1)");
      c.line("for (int64_t g=0; g<G; ++g) {");
      c.line("  const int64_t batch=g/" + std::to_string(H) + "/" +
             std::to_string(chunks) + ", chunk=g/" + std::to_string(H) + "%" +
             std::to_string(chunks) + ", head=g%" + std::to_string(H) + ";");
      c.line("  for (int64_t m=0; m<M; ++m) for (int64_t k=0; k<K; ++k)");
      c.line("    " + dst + "[(g*M+m)*K+k] = expand_bf(((const uint16_t*)a" +
             std::to_string(slots[0]) + ")[batch*" + std::to_string(st[0]) +
             "LL+(chunk*K+k)*" + std::to_string(st[1]) + "LL+head*" +
             std::to_string(st[2]) + "LL+m*" + std::to_string(st[3]) + "LL]);");
      c.line("}");
    } else if (pk == "widen-lhs-f32") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t ak = op.getAttrOfType<IntegerAttr>("lhs_k").getInt();
      int64_t ab = op.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
      int64_t am = 0;
      while (am == ak || am == ab)
        ++am;
      std::string banks = ab < 0 ? "1LL" : "G";
      std::string fnWiden = dt == "bf16" ? "expand_bf" : "expand_f16";
      std::string fnSpan = dt == "bf16" ? "widen_bf16_span" : "widen_f16_span";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*M*K > 131072)");
      c.line("for (int64_t row=0; row<(" + banks + ")*M; ++row) {");
      c.line("  const int64_t g = M ? row / M : 0, m = M ? row % M : 0;");
      c.line("  const uint16_t *src = (const uint16_t*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(ab < 0 ? 0 : st[ab]) + "LL + m*" +
             std::to_string(st[am]) + "LL;");
      c.line("  float *out = " + dst + " + row*K;");
      if (st[ak] == 1)
        c.line("  " + fnSpan + "(out, src, K);");
      else
        c.line("  for (int64_t k=0; k<K; ++k) out[k] = " + fnWiden + "(src[k*" +
               std::to_string(st[ak]) + "LL]);");
      c.line("}");
    } else if (pk == "widen-pack-rhs-f32") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t bk = op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      std::string fnWiden = dt == "bf16" ? "expand_bf" : "expand_f16";
      std::string fnSpan = dt == "bf16" ? "widen_bf16_span" : "widen_f16_span";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*nc*K*nr > 131072 && (" + banks + ")*nc >= 4)");
      c.line("for (int64_t p=0; p<(" + banks + ")*nc; ++p) {");
      c.line("  const int64_t g = nc ? p / nc : 0, c = (nc ? p % nc : 0)*nr, valid = N-c < nr ? N-c : nr;");
      c.line("  for (int64_t k=0; k<K; ++k) {");
      c.line("    const uint16_t *src = (const uint16_t*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + k*" +
             std::to_string(st[bk]) + "LL + c*" + std::to_string(st[bn]) +
             "LL;");
      c.line("    float *out = " + dst + " + (p*K + k)*nr;");
      if (st[bn] == 1)
        c.line("    " + fnSpan + "(out, src, valid);");
      else
        c.line("    for (int64_t j=0; j<valid; ++j) out[j] = " + fnWiden +
               "(src[j*" + std::to_string(st[bn]) + "LL]);");
      c.line("    for (int64_t j=valid; j<nr; ++j) out[j] = 0.0f;");
      c.line("  }");
      c.line("}");
    } else if (pk == "pack-rhs-panels") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t bytes = dt == "f32" || dt == "i32" ? 4 : dt == "i8" || dt == "u8" ? 1 : 2;
      int64_t bk = op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*nc*K*nr > 131072 && (" + banks + ")*nc >= 4)");
      c.line("for (int64_t p=0; p<(" + banks + ")*nc; ++p) {");
      c.line("  const int64_t g = nc ? p / nc : 0, c = (nc ? p % nc : 0)*nr, valid = N-c < nr ? N-c : nr;");
      c.line("  for (int64_t k=0; k<K; ++k) {");
      c.line("    uint8_t *out = (uint8_t*)" + dst + " + (p*K + k)*nr*" +
             std::to_string(bytes) + "LL;");
      c.line("    const uint8_t *src = (const uint8_t*)a" +
             std::to_string(slots[0]) + " + (g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + k*" +
             std::to_string(st[bk]) + "LL + c*" + std::to_string(st[bn]) +
             "LL)*" + std::to_string(bytes) + "LL;");
      if (st[bn] == 1)
        c.line("    memcpy(out, src, (size_t)(valid*" + std::to_string(bytes) +
               "LL));");
      else
        c.line("    for (int64_t j=0; j<valid; ++j) memcpy(out+j*" +
               std::to_string(bytes) + "LL, src+j*" +
               std::to_string(st[bn] * bytes) + "LL, " + std::to_string(bytes) +
               ");");
      c.line("    if (valid < nr) memset(out + valid*" + std::to_string(bytes) +
             "LL, 0, (size_t)((nr - valid)*" + std::to_string(bytes) + "LL));");
      c.line("  }");
      c.line("}");
    } else if (pk == "pack-vnni") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      int64_t kBlk = op.getAttrOfType<IntegerAttr>("k_block").getInt();
      int64_t bytes = kBlk == 4 ? 1 : 2;
      std::string cTy = bytes == 1 ? "uint8_t" : "uint16_t";
      int64_t bk = op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*nc*K*nr > 131072 && (" + banks + ")*nc >= 4)");
      c.line("for (int64_t p=0; p<(" + banks + ")*nc; ++p) {");
      c.line("  const int64_t g = nc ? p / nc : 0, c = (nc ? p % nc : 0)*nr, valid = N-c < nr ? N-c : nr;");
      c.line("  const int64_t kPad = ((K+" + std::to_string(kBlk - 1) + ")/" +
             std::to_string(kBlk) + ")*" + std::to_string(kBlk) + ";");
      c.line("  " + cTy + " *out = (" + cTy + "*)" + dst + " + p*kPad*nr;");
      c.line("  if (valid < nr || kPad > K) memset(out, 0, (size_t)(kPad*nr)*sizeof(" + cTy + "));");
      c.line("  int64_t kb = 0;");
      c.line("  for (; kb + " + std::to_string(kBlk) + " <= K; kb += " +
             std::to_string(kBlk) + ") {");
      c.line("    const " + cTy + " *base = (const " + cTy + "*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
             std::to_string(st[bn]) + "LL + kb*" + std::to_string(st[bk]) + "LL;");
      c.line("    " + cTy + " *dst_k = out + (kb/" + std::to_string(kBlk) +
             ")*nr*" + std::to_string(kBlk) + ";");
      c.line("    #pragma omp simd");
      c.line("    for (int64_t j=0; j<valid; ++j)");
      c.line("      for (int lane=0; lane<" + std::to_string(kBlk) + "; ++lane)");
      c.line("        dst_k[j*" + std::to_string(kBlk) +
             " + lane] = base[j*" + std::to_string(st[bn]) + "LL + lane*" +
             std::to_string(st[bk]) + "LL];");
      c.line("  }");
      c.line("  for (; kb < K; kb += " + std::to_string(kBlk) + ") {");
      c.line("    const " + cTy + " *base = (const " + cTy + "*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
             std::to_string(st[bn]) + "LL + kb*" + std::to_string(st[bk]) + "LL;");
      c.line("    for (int64_t j=0; j<valid; ++j)");
      c.line("      for (int lane=0; lane<" + std::to_string(kBlk) + "; ++lane)");
      c.line("        if (kb+lane < K) out[(kb/" + std::to_string(kBlk) +
             ")*nr*" + std::to_string(kBlk) + " + j*" + std::to_string(kBlk) +
             " + lane] = base[j*" + std::to_string(st[bn]) + "LL + lane*" +
             std::to_string(st[bk]) + "LL];");
      c.line("  }");
      c.line("}");
    } else if (pk == "decode-mx8-lhs") {
      auto stA = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto stS = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_1");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(M*K>16384 && workers>1)");
      c.line("for (int64_t r=0; r<M; ++r) {");
      c.line("  const uint8_t *src_r = (const uint8_t*)a" +
             std::to_string(slots[0]) + " + r*" + std::to_string(stA[0]) + "LL;");
      c.line("  const uint8_t *sc_r = (const uint8_t*)a" +
             std::to_string(slots[1]) + " + r*" + std::to_string(stS[0]) + "LL;");
      c.line("  float *dst_r = " + dst + " + r*K;");
      c.line("  for (int64_t g=0; g<K/32; ++g) {");
      c.line("    const float factor = g_mx8_scale[sc_r[g*" +
             std::to_string(stS[1]) + "LL]];");
      c.line("    const uint8_t *src_g = src_r + g*" + std::to_string(stA[1]) + "LL;");
      c.line("    float *dst_g = dst_r + g*32;");
      c.line("#if defined(__x86_64__)");
      c.line("    #pragma omp simd");
      c.line("#endif");
      c.line("    for (int t=0; t<32; ++t) dst_g[t] = g_mx8_val[src_g[t*" +
             std::to_string(stA[2]) + "LL]] * factor;");
      c.line("  }");
      c.line("}");
    } else if (pk == "decode-mx8-rhs") {
      auto stB = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto stS = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_1");
      int64_t colAx =
          op.getAttrOfType<IntegerAttr>("scale_column_axis").getInt();
      std::string scExpr =
          colAx == 0 ? ("(c0+jc)*" + std::to_string(stS[0]) + "LL + g*" +
                        std::to_string(stS[1]) + "LL")
                     : ("g*" + std::to_string(stS[0]) + "LL + (c0+jc)*" +
                        std::to_string(stS[1]) + "LL");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(N*K>16384 && workers>1)");
      c.line("for (int64_t g=0; g<K/32; ++g) {");
      c.line("  for (int64_t c0=0; c0<N; c0+=64) {");
      c.line("    const int64_t blk = N - c0 < 64 ? N - c0 : 64;");
      c.line("    float sc_blk[64];");
      c.line("    const uint8_t *sc_base = (const uint8_t*)a" +
             std::to_string(slots[1]) + ";");
      c.line("    for (int64_t jc=0; jc<blk; ++jc) sc_blk[jc] = g_mx8_scale[sc_base[" +
             scExpr + "]];");
      c.line("    for (int t=0; t<32; ++t) {");
      c.line("      const uint8_t *src_gt = (const uint8_t*)a" +
             std::to_string(slots[0]) + " + g*" + std::to_string(stB[0]) +
             "LL + t*" + std::to_string(stB[1]) + "LL + c0*" +
             std::to_string(stB[2]) + "LL;");
      c.line("      float *dst_gt = " + dst + " + (g*32 + t)*N + c0;");
      c.line("#if defined(__x86_64__)");
      c.line("      #pragma omp simd");
      c.line("#endif");
      c.line("      for (int64_t jc=0; jc<blk; ++jc) dst_gt[jc] = g_mx8_val[src_gt[jc*" +
             std::to_string(stB[2]) + "LL]] * sc_blk[jc];");
      c.line("    }");
      c.line("  }");
      c.line("}");
    } else if (pk == "encode-q8k") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      c.line("for (int64_t g=0; g<K/256; ++g)");
      c.line("  pack_q8_record(&" + dst + "[g], (const float*)a" +
             std::to_string(slots[0]) + " + g*" + std::to_string(st[0]) +
             "LL, " + std::to_string(st[1]) + "LL);");
    } else if (pk == "gather-routed-lhs" || pk == "gather-widen-routed-lhs-f32") {
      auto stA = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto stO = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_1");
      auto stR = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_2");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t ak = op.getAttrOfType<IntegerAttr>("lhs_k").getInt();
      int64_t mult = op.getAttrOfType<IntegerAttr>("multiplicity").getInt();
      bool wide = pk == "gather-widen-routed-lhs-f32";
      c.line("const int64_t active_begin = ((const int32_t*)a" +
             std::to_string(slots[1]) + ")[0], active_end = ((const int32_t*)a" +
             std::to_string(slots[1]) + ")[G*" + std::to_string(stO[0]) + "LL];");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(active_end > active_begin + 1)");
      c.line("for (int64_t r=active_begin; r<active_end; ++r) {");
      c.line("  const int64_t src = ((const int32_t*)a" +
             std::to_string(slots[2]) + ")[r*" + std::to_string(stR[0]) +
             "LL] / " + std::to_string(mult) + "LL;");
      if (wide) {
        std::string fnWiden = dt == "bf16" ? "expand_bf" : "expand_f16";
        c.line("  const uint16_t *from = (const uint16_t*)a" +
               std::to_string(slots[0]) + " + src*" +
               std::to_string(stA[1 - ak]) + "LL;");
        c.line("  float *to = " + dst + " + r*K;");
        c.line("  for (int64_t k=0; k<K; ++k) to[k] = " + fnWiden + "(from[k*" +
               std::to_string(stA[ak]) + "LL]);");
      } else {
        int64_t bytes = dt == "f32" || dt == "i32" ? 4 : dt == "i8" || dt == "u8" ? 1 : 2;
        c.line("  const uint8_t *from = (const uint8_t*)a" +
               std::to_string(slots[0]) + " + src*" +
               std::to_string(stA[1 - ak] * bytes) + "LL;");
        c.line("  uint8_t *to = (uint8_t*)" + dst + " + r*K*" +
               std::to_string(bytes) + "LL;");
        if (stA[ak] == 1)
          c.line("  memcpy(to, from, (size_t)(K*" + std::to_string(bytes) + "LL));");
        else
          c.line("  for (int64_t k=0; k<K; ++k) memcpy(to+k*" +
                 std::to_string(bytes) + "LL, from+k*" +
                 std::to_string(stA[ak] * bytes) + "LL, " +
                 std::to_string(bytes) + ");");
      }
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.host_parallel_tiles"] = [](Operation &op,
                                                     PrintPoint at,
                                                     PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto &bb = op.getRegion(0).front();
      bool grouped = op.getAttrOfType<BoolAttr>("grouped").getValue();
      bool routed = op.getAttrOfType<BoolAttr>("routed").getValue();
      bool windowed = op.getAttrOfType<BoolAttr>("windowed").getValue();
      int64_t H = op.getAttrOfType<IntegerAttr>("head_extent").getInt();
      int64_t chunks = op.getAttrOfType<IntegerAttr>("chunk_extent").getInt();

      c.line("if (jobs > 0) {");
      c.indent++;
      c.line("const int nt = jobs < workers ? (int)jobs : (int)workers;");
      c.line("#pragma omp parallel for num_threads(nt) schedule(static) if(nt > 1)");
      c.line("for (int64_t job=0; job<jobs; ++job) {");
      c.indent++;
      c.line("const int64_t " + c.name(bb.getArgument(0)) +
             " = job / (nc * mb), rem = job % (nc * mb);");
      c.line("const int64_t " + c.name(bb.getArgument(1)) +
             " = (rem / mb) * nr;");
      c.line("int64_t " + c.name(bb.getArgument(2)) +
             " = (rem % mb) * mr, end = M;");
      if (windowed) {
        std::string gN = c.name(bb.getArgument(0));
        c.line("const int64_t " + c.name(bb.getArgument(7)) + " = " + gN + "%" +
               std::to_string(H) + ", " + c.name(bb.getArgument(6)) + " = " +
               gN + "/" + std::to_string(H) + "%" + std::to_string(chunks) +
               ", " + c.name(bb.getArgument(5)) + " = " + gN + "/" +
               std::to_string(H) + "/" + std::to_string(chunks) + ";");
      } else {
        c.line("const int64_t " + c.name(bb.getArgument(5)) + " = 0, " +
               c.name(bb.getArgument(6)) + " = 0, " +
               c.name(bb.getArgument(7)) + " = 0;");
        c.line("(void)" + c.name(bb.getArgument(5)) + "; (void)" +
               c.name(bb.getArgument(6)) + "; (void)" +
               c.name(bb.getArgument(7)) + ";");
      }
      if (grouped) {
        int64_t oSlot = op.getAttrOfType<IntegerAttr>("offsets_slot").getInt();
        int64_t oSt = op.getAttrOfType<IntegerAttr>("offsets_stride").getInt();
        std::string gN = c.name(bb.getArgument(0));
        std::string rN = c.name(bb.getArgument(2));
        c.line(rN + " += ((const int32_t*)a" + std::to_string(oSlot) + ")[" +
               gN + "*" + std::to_string(oSt) + "LL];");
        c.line("end = ((const int32_t*)a" + std::to_string(oSlot) + ")[(" + gN +
               "+1)*" + std::to_string(oSt) + "LL];");
        c.line("if (" + rN + " >= end) continue;");
      }
      if (routed) {
        int64_t rSlot = op.getAttrOfType<IntegerAttr>("routes_slot").getInt();
        int64_t rSt = op.getAttrOfType<IntegerAttr>("routes_stride").getInt();
        int64_t mult = op.getAttrOfType<IntegerAttr>("multiplicity").getInt();
        std::string rN = c.name(bb.getArgument(2));
        std::string destN = c.name(bb.getArgument(4));
        c.line("const int64_t " + destN + " = ((const int32_t*)a" +
               std::to_string(rSlot) + ")[" + rN + "*" + std::to_string(rSt) +
               "LL], " + c.name(bb.getArgument(3)) + " = " + destN + " / " +
               std::to_string(mult) + "LL;");
      } else {
        std::string rN = c.name(bb.getArgument(2));
        c.line("const int64_t " + c.name(bb.getArgument(4)) + " = " + rN +
               ", " + c.name(bb.getArgument(3)) + " = " + rN + ";");
      }
      std::string rN = c.name(bb.getArgument(2));
      std::string cN = c.name(bb.getArgument(1));
      c.line("const int64_t " + c.name(bb.getArgument(8)) + " = end - " + rN +
             " < mr ? end - " + rN + " : mr, " + c.name(bb.getArgument(9)) +
             " = N - " + cN + " < nr ? N - " + cN + " : nr;");
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["weft_host.host_view_ptr"] = [](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t idx = op.getAttrOfType<IntegerAttr>("index").getInt();
    auto ptrTy = op.getAttrOfType<StringAttr>("c_ptr_type").str();
    auto base = op.getAttrOfType<StringAttr>("base_expr").str();
    int64_t byteOff = op.getAttrOfType<IntegerAttr>("byte_offset").getInt();
    if (op.hasAttr("routed_temp")) {
      c.line("_Alignas(64) uint8_t routed_out[" + std::to_string(byteOff) +
             "]; " + ptrTy + "p" + std::to_string(idx) + " = (" + ptrTy +
             ")routed_out;");
      return llvm::Error::success();
    }
    std::string rawBase =
        byteOff == 0
            ? ("(" + ptrTy + ")(" + base + ")")
            : ("(" + ptrTy + ")((const char*)(" + base + ") + " +
               std::to_string(byteOff) + "LL)");
    if (op.getNumOperands() > 0)
      c.line(ptrTy + "p" + std::to_string(idx) + " = " + rawBase +
             " + (ptrdiff_t)" + c.name(op.getOperand(0)) + ";");
    else
      c.line(ptrTy + "p" + std::to_string(idx) + " = " + rawBase + ";");
    return llvm::Error::success();
  };

  p.operations["weft_host.host_call_task"] = [](Operation &op, PrintPoint at,
                                                PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t arity = op.getAttrOfType<IntegerAttr>("arity").getInt();
    SmallVector<std::string> args;
    for (int64_t i = 0; i < arity; ++i)
      args.push_back("p" + std::to_string(i));
    args.push_back("(size_t)" + c.name(op.getOperand(0)));
    args.push_back("(size_t)" + c.name(op.getOperand(1)));
    for (int64_t st :
         op.getAttrOfType<DenseI64ArrayAttr>("strides").asArrayRef())
      args.push_back("(size_t)" + std::to_string(st) + "ULL");
    c.line("matrix_panel_instance(" + llvm::join(args, ", ") + ");");
    return llvm::Error::success();
  };

  p.operations["weft_host.host_scatter_routed"] = [](Operation &op,
                                                     PrintPoint at,
                                                     PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto *pt = op.getParentOp();
    auto &ptBb = pt->getRegion(0).front();
    std::string cVar = c.name(ptBb.getArgument(1));
    std::string rVar = c.name(ptBb.getArgument(2));
    std::string rowsVar = c.name(ptBb.getArgument(8));
    std::string colsVar = c.name(ptBb.getArgument(9));

    int64_t dstSlot = op.getAttrOfType<IntegerAttr>("dst_slot").getInt();
    int64_t mult = op.getAttrOfType<IntegerAttr>("multiplicity").getInt();
    auto st = op.getAttrOfType<DenseI64ArrayAttr>("dst_strides");
    int64_t bytes = op.getAttrOfType<IntegerAttr>("elem_bytes").getInt();
    int64_t rSlot = op.getAttrOfType<IntegerAttr>("routes_slot").getInt();
    int64_t rSt = op.getAttrOfType<IntegerAttr>("routes_stride").getInt();
    c.line("for (int64_t i=0; i<" + rowsVar + "; ++i) {");
    c.line("  const int64_t d_i = ((const int32_t*)a" + std::to_string(rSlot) +
           ")[(" + rVar + "+i)*" + std::to_string(rSt) + "LL];");
    c.line("  uint8_t *out = (uint8_t*)a" + std::to_string(dstSlot) +
           " + ((d_i/" + std::to_string(mult) + "LL)*" + std::to_string(st[0]) +
           "LL + (d_i%" + std::to_string(mult) + "LL)*" +
           std::to_string(st[1]) + "LL + " + cVar + "*" +
           std::to_string(st[2]) + "LL)*" + std::to_string(bytes) + "LL;");
    c.line("  const uint8_t *from = routed_out + i*nr*" +
           std::to_string(bytes) + "LL;");
    if (st[2] == 1)
      c.line("  memcpy(out, from, (size_t)(" + colsVar + "*" +
             std::to_string(bytes) + "LL));");
    else
      c.line("  for (int64_t j=0; j<" + colsVar + "; ++j) memcpy(out+j*" +
             std::to_string(st[2] * bytes) + "LL, from+j*" +
             std::to_string(bytes) + "LL, " + std::to_string(bytes) + ");");
    c.line("}");
    return llvm::Error::success();
  };

  p.operations["weft_host.host_free"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string name = op.getAttrOfType<StringAttr>("buf_name").str();
      int wsSlot = name == "left" ? 0 : name == "rhs" ? 1 : name == "members" ? 2 : 3;
      c.line("intent_ws_release(" + std::to_string(wsSlot) + ", " + name + ");");
    }
    return llvm::Error::success();
  };

  p.package = [](DictionaryAttr attrs,
                 const std::map<std::string, std::string> &files,
                 llvm::raw_pwrite_stream &out) -> llvm::Error {
    auto pkgAttr = attrs.getAs<StringAttr>("weft_host.package");
    if (!pkgAttr)
      return llvm::createStringError("missing weft_host.package attribute");
    auto parsed = llvm::json::parse(pkgAttr.getValue());
    if (!parsed)
      return parsed.takeError();
    auto *obj = parsed->getAsObject();
    (*obj)["task"] = files.at("task.weftir");
    (*obj)["host"] = files.at("host.c");
    out << llvm::json::Value(std::move(*obj));
    return llvm::Error::success();
  };

  return p;
}
} // namespace CI_UNIT_NAMESPACE
