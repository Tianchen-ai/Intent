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

void emitTupleUnpack(PrintContext &c, Operation &op, StringRef tup,
                     ArrayAttr tys) {
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    std::string t = cast<StringAttr>(tys[i]).str();
    std::string zId = irId(c, op.getResult(i)) + "_z";
    if (t == "index") {
      c.line(zId + "=arith.constant 0:index");
      c.line(irId(c, op.getResult(i)) + "=\"weft_kernel.binary\"(" + tup.str() +
             "#" + std::to_string(i) + "," + zId +
             ") <{kind=\"add\"}>:(index,index)->index");
      continue;
    }
    std::string scTy = t.find("f32") != std::string::npos    ? "f32"
                       : t.find("f16") != std::string::npos  ? "f16"
                       : t.find("bf16") != std::string::npos ? "bf16"
                       : t.find("si64") != std::string::npos ? "si64"
                       : t.find("ui64") != std::string::npos ? "ui64"
                                                             : "si32";
    std::string zLit =
        (scTy == "f32" || scTy == "f16" || scTy == "bf16") ? "0.0" : "0";
    c.line(zId + "=\"weft_kernel.constant\"() <{value=" + zLit + ":" + scTy +
           "}>:()->" + scTy);
    c.line(irId(c, op.getResult(i)) + "=\"weft_kernel.binary\"(" + tup.str() +
           "#" + std::to_string(i) + "," + zId + ") <{kind=\"add\"}>:(" + t +
           "," + scTy + ")->" + t);
  }
}

std::string formatEncodingDecl(Operation &op) {
  std::string name = op.getAttrOfType<StringAttr>("sym_name").str();
  int64_t align = op.getAttrOfType<IntegerAttr>("alignment").getInt();
  int64_t elems = op.getAttrOfType<IntegerAttr>("elements").getInt();
  int64_t storageBits = op.getAttrOfType<IntegerAttr>("storage_bits").getInt();
  auto fNames = op.getAttrOfType<ArrayAttr>("field_names");
  auto fTypes = op.getAttrOfType<ArrayAttr>("field_types");
  auto fExts = op.getAttrOfType<DenseI64ArrayAttr>("field_extents");
  auto fLayouts = op.getAttrOfType<ArrayAttr>("field_layouts");
  auto fOffsets = op.getAttrOfType<DenseI64ArrayAttr>("field_bit_offsets");
  auto fBits = op.getAttrOfType<DenseI64ArrayAttr>("field_storage_bits");

  SmallVector<std::string> names, types, shapes, layouts, offsets, bits;
  for (size_t i = 0; i < fNames.size(); ++i) {
    names.push_back("\"" + cast<StringAttr>(fNames[i]).str() + "\"");
    types.push_back(cast<StringAttr>(fTypes[i]).str());
    int64_t ext = fExts[i];
    shapes.push_back(ext == 1 ? "array<i64>"
                              : "array<i64:" + std::to_string(ext) + ">");
    layouts.push_back(cast<StringAttr>(fLayouts[i]).str());
    offsets.push_back(std::to_string(fOffsets[i]));
    bits.push_back(std::to_string(fBits[i]));
  }
  return "\"weft_kernel.encoding_decl\"() <{sym_name=\"" + name +
         "\",kind=\"base\",layout_identity=\"" + name +
         "\",bit_order=\"lsb_first\",byte_order=\"little\",alignment=" +
         std::to_string(align) + ":i64,elements=" + std::to_string(elems) +
         ":i64,storage_bits=" + std::to_string(storageBits) +
         ":i64,field_names=[" + llvm::join(names, ",") + "],field_types=[" +
         llvm::join(types, ",") + "],field_shapes=[" + llvm::join(shapes, ",") +
         "],field_layouts=[" + llvm::join(layouts, ",") +
         "],field_bit_offsets=array<i64:" + llvm::join(offsets, ",") +
         ">,field_storage_bits=array<i64:" + llvm::join(bits, ",") +
         ">,padding=array<i64>}>:()->()";
}
} // namespace

TerminalPrinter weftTerminalPrinter() {
  TerminalPrinter p;
  auto noop = [](Operation &, PrintPoint, PrintContext &) {
    return llvm::Error::success();
  };
  p.operations["builtin.module"] = PrintRule::custom(noop, "legacy_operation_rule");
  p.operations["func.func"] = PrintRule::custom(noop, "legacy_operation_rule");
  p.operations["func.return"] = PrintRule::custom(noop, "legacy_operation_rule");
  p.operations["weft_host.end"] = PrintRule::custom(noop, "legacy_operation_rule");

  auto programRule = [](Operation &, PrintPoint at, PrintContext &c) {
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
  p.operations["weft_host.matrix_program"] = PrintRule::custom(programRule, "legacy_operation_rule");
  p.operations["weft_host.row_program"] = PrintRule::custom(programRule, "legacy_operation_rule");
  p.operations["weft_host.owner_program"] = PrintRule::custom(programRule, "legacy_operation_rule");

  // ---------------- Region 0 ("task.weftir") Ops ----------------
  p.operations["weft_host.encoding_decl"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(formatEncodingDecl(op));
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_kernel"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto &bb = op.getRegion(0).front();
      auto vTypes = op.getAttrOfType<ArrayAttr>("view_types");
      auto writes = op.getAttrOfType<ArrayAttr>("arg_writes");
      auto accessModes = op.getAttrOfType<ArrayAttr>("arg_access");
      auto aliasModes = op.getAttrOfType<DenseI64ArrayAttr>("arg_aliases");
      auto shapeSymsAttr = op.getAttrOfType<ArrayAttr>("shape_symbols");
      auto strided = op.getAttrOfType<DenseI64ArrayAttr>("strided_slots");
      SmallVector<std::string> names, effects, aliases, params, stridedNames,
          symLiterals, symNames;
      if (shapeSymsAttr) {
        for (Attribute a : shapeSymsAttr) {
          std::string s = cast<StringAttr>(a).str();
          symNames.push_back(s);
          symLiterals.push_back("\"" + s + "\"");
        }
      } else {
        symNames = {"R", "C"};
        symLiterals = {"\"R\"", "\"C\""};
      }
      for (size_t i = 0; i < vTypes.size(); ++i) {
        std::string aName = "a" + std::to_string(i);
        names.push_back("\"" + aName + "\"");
        if (accessModes) {
          effects.push_back("\"" + cast<StringAttr>(accessModes[i]).str() +
                            "\"");
        } else {
          effects.push_back(cast<BoolAttr>(writes[i]).getValue() ? "\"write\""
                                                                 : "\"read\"");
        }
        aliases.push_back(aliasModes ? std::to_string(aliasModes[i]) : "0");
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
             llvm::join(aliases, ",") + ">,shape_symbols=[" +
             llvm::join(symLiterals, ",") + "],strided_args=[" +
             llvm::join(stridedNames, ",") + "]}> ({");
      c.line("^bb0(" + llvm::join(params, ",") + "):");
      size_t nViews = vTypes.size();
      for (size_t k = 0; k < symNames.size(); ++k) {
        c.line(irId(c, bb.getArgument(nViews + k)) +
               "=\"weft_kernel.symbol\"() "
               "<{choices=array<i64>,kind=\"shape\",name=\"" +
               symNames[k] + "\"}>:()->index");
      }
      c.line(irId(c, bb.getArgument(nViews + symNames.size())) +
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_const"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto lit = op.getAttrOfType<StringAttr>("lit").str();
    auto ty = op.getAttrOfType<StringAttr>("val_type").str();
    if (ty == "index")
      c.line(irId(c, op.getResult(0)) + "=arith.constant " + lit + ":index");
    else if (ty == "i1" && (lit == "true" || lit == "false"))
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.constant\"() <{value=" +
             lit + "}>:()->i1");
    else
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.constant\"() <{value=" +
             lit + ":" + ty + "}>:()->" + ty);
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_new"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.new\"(" +
             irId(c, op.getOperand(0)) + ") <{initialized=true}>:(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_slice"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_subview"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto slTy = op.getAttrOfType<StringAttr>("slice_type").str();
    auto vlTy = op.getAttrOfType<StringAttr>("val_type").str();
    bool admit = op.getAttrOfType<BoolAttr>("admit").getValue();
    std::string resId = irId(c, op.getResult(0));
    std::string subId = admit ? (resId + "_s") : resId;
    if (auto exts = op.getAttrOfType<DenseI64ArrayAttr>("static_extents")) {
      SmallVector<std::string, 4> extStrs, offStrs, opArgs{irId(c, op.getOperand(0))},
          opTys{slTy};
      for (auto [idx, e] : llvm::enumerate(exts.asArrayRef())) {
        extStrs.push_back(std::to_string(e));
        offStrs.push_back("-1");
        opArgs.push_back(irId(c, op.getOperand(1 + idx)));
        opTys.push_back("index");
      }
      std::string outSlTy = slTy;
      auto bracket = outSlTy.rfind(">,[");
      if (bracket != std::string::npos) {
        auto close = outSlTy.find(']', bracket + 2);
        outSlTy.replace(bracket + 2, close - (bracket + 2) + 1,
                        "[" + llvm::join(extStrs, ",") + "]");
      }
      c.line(subId + "=\"weft_kernel.subview\"(" + llvm::join(opArgs, ",") +
             ") <{extents=array<i64:" + llvm::join(extStrs, ",") +
             ">,offsets=array<i64:" + llvm::join(offStrs, ",") + ">}>:(" +
             llvm::join(opTys, ",") + ")->" + outSlTy);
      if (admit)
        c.line(resId + "=\"weft_kernel.admit\"(" + subId + "):(" + outSlTy +
               ")->" + vlTy);
      return llvm::Error::success();
    }
    int64_t staticExt = op.getAttrOfType<IntegerAttr>("static_extent").getInt();
    std::string outSlTy = slTy;
    auto bracket = outSlTy.rfind(">,[");
    if (bracket != std::string::npos) {
      auto close = outSlTy.find(']', bracket + 2);
      outSlTy.replace(bracket + 2, close - (bracket + 2) + 1,
                      "[" + std::to_string(staticExt) + "]");
    }
    if (staticExt > 0) {
      c.line(subId + "=\"weft_kernel.subview\"(" + irId(c, op.getOperand(0)) +
             "," + irId(c, op.getOperand(1)) +
             ") <{extents=array<i64:" + std::to_string(staticExt) +
             ">,offsets=array<i64:-1>}>:(" + slTy + ",index)->" + outSlTy);
    } else {
      c.line(subId + "=\"weft_kernel.subview\"(" + irId(c, op.getOperand(0)) +
             "," + irId(c, op.getOperand(1)) + "," + irId(c, op.getOperand(2)) +
             ") <{extents=array<i64:-1>,offsets=array<i64:-1>}>:(" + slTy +
             ",index,index)->" + outSlTy);
    }
    if (admit)
      c.line(resId + "=\"weft_kernel.admit\"(" + subId + "):(" + outSlTy +
             ")->" + vlTy);
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_unary"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.unary\"(" +
             irId(c, op.getOperand(0)) + ") <{kind=\"" +
             op.getAttrOfType<StringAttr>("kind").str() + "\"}>:(" + vTy +
             ")->" + vTy);
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_reduce"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.reduce\"(" +
             irId(c, op.getOperand(0)) + ") <{axis=" +
             std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) +
             ":i64,kind=\"" + op.getAttrOfType<StringAttr>("kind").str() +
             "\"}>:(" + op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_scan"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    auto &bb = op.getRegion(0).front();
    auto eTy = op.getAttrOfType<StringAttr>("elem_type").str();
    auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
    if (at == PrintPoint::Enter) {
      int64_t axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
      bool rev = op.getAttrOfType<BoolAttr>("reverse").getValue();
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.scan\"(" +
             irId(c, op.getOperand(0)) + ") <{axis=" + std::to_string(axis) +
             ":i64,reverse=" + (rev ? "true" : "false") + "}> ({");
      c.line("^bb0(" + irId(c, bb.getArgument(0)) + ":" + eTy + "," +
             irId(c, bb.getArgument(1)) + ":" + eTy + "):");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}):(" + vTy + ")->" + vTy);
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_scan_yield"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("\"weft_kernel.scan_yield\"(" + irId(c, op.getOperand(0)) + "):(" +
             op.getAttrOfType<StringAttr>("val_type").str() + ")->()");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_iota"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.iota\"() <{end=" +
             std::to_string(op.getAttrOfType<IntegerAttr>("end").getInt()) +
             ":i64,start=" +
             std::to_string(op.getAttrOfType<IntegerAttr>("start").getInt()) +
             ":i64}>:()->" + op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_local_alloc"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                                  PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) +
             "=\"weft_kernel.local_alloc\"() <{alignment=" +
             std::to_string(op.getAttrOfType<IntegerAttr>("alignment").getInt()) +
             ":i64,elements=" +
             std::to_string(op.getAttrOfType<IntegerAttr>("elements").getInt()) +
             ":i64}>:()->" + op.getAttrOfType<StringAttr>("ptr_type").str());
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  auto ptrIdxMaskTypes =
      [](StringRef vTy) -> std::pair<std::string, std::string> {
    if (vTy.starts_with("!weft_kernel.value<")) {
      auto comma = vTy.find(',');
      if (comma != StringRef::npos) {
        StringRef rest = vTy.substr(comma);
        return {("!weft_kernel.value<si64" + rest).str(),
                ("!weft_kernel.value<i1" + rest).str()};
      }
    }
    return {"si64", "i1"};
  };

  p.operations["weft_host.task_ptr_load"] = PrintRule::custom([ptrIdxMaskTypes](Operation &op,
                                                              PrintPoint at,
                                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto pTy = op.getAttrOfType<StringAttr>("ptr_type").str();
      auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
      auto [offTy, mskTy] = ptrIdxMaskTypes(vTy);
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.pointer_load\"(" +
             irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) + "," +
             irId(c, op.getOperand(2)) + "," + irId(c, op.getOperand(3)) +
             "):(" + pTy + "," + offTy + "," + mskTy + "," + vTy + ")->" + vTy);
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_ptr_store"] =
      PrintRule::custom([ptrIdxMaskTypes](Operation &op, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter) {
          auto pTy = op.getAttrOfType<StringAttr>("ptr_type").str();
          auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
          auto [offTy, mskTy] = ptrIdxMaskTypes(vTy);
          c.line("\"weft_kernel.pointer_store\"(" + irId(c, op.getOperand(0)) +
                 "," + irId(c, op.getOperand(1)) + "," +
                 irId(c, op.getOperand(2)) + "," + irId(c, op.getOperand(3)) +
                 "):(" + pTy + "," + offTy + "," + mskTy + "," + vTy + ")->()");
        }
        return llvm::Error::success();
      }, "legacy_operation_rule");

  p.operations["weft_host.task_atomic_rmw"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
      auto sTy = op.getAttrOfType<StringAttr>("slice_type").str();
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.atomic_rmw\"(" +
             irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) +
             ") <{kind=\"" + op.getAttrOfType<StringAttr>("kind").str() +
             "\",ordering=\"" +
             op.getAttrOfType<StringAttr>("ordering").str() + "\"}>:(" + vTy +
             "," + sTy + ")->" + vTy);
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_binary"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_compare"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_select"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto vTy = op.getAttrOfType<StringAttr>("val_type").str();
      auto tAttr = op.getAttrOfType<StringAttr>("true_type");
      auto fAttr = op.getAttrOfType<StringAttr>("false_type");
      std::string tTy = tAttr ? tAttr.str() : vTy;
      std::string fTy = fAttr ? fAttr.str() : vTy;
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.select\"(" +
             irId(c, op.getOperand(0)) + "," + irId(c, op.getOperand(1)) + "," +
             irId(c, op.getOperand(2)) + "):(" +
             op.getAttrOfType<StringAttr>("cond_type").str() + "," + tTy + "," +
             fTy + ")->" + vTy);
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_cast"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel." +
             op.getAttrOfType<StringAttr>("op_kind").str() + "\"(" +
             irId(c, op.getOperand(0)) + "):(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_field"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.field\"(" +
             irId(c, op.getOperand(0)) + ") <{name=\"" +
             op.getAttrOfType<StringAttr>("field_name").str() + "\"}>:(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_fold2"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line(irId(c, op.getResult(0)) + "=\"weft_kernel.fold2\"(" +
             irId(c, op.getOperand(0)) + "):(" +
             op.getAttrOfType<StringAttr>("in_type").str() + ")->" +
             op.getAttrOfType<StringAttr>("val_type").str());
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_contract"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_for"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
        emitTupleUnpack(c, op, tup, iterTys);
      }
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_if"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    auto resTys = op.getAttrOfType<ArrayAttr>("res_types");
    SmallVector<std::string> tys;
    for (Attribute a : resTys)
      tys.push_back(cast<StringAttr>(a).str());
    bool hasElse = !op.getRegion(1).empty();
    if (at == PrintPoint::Enter) {
      if (op.getNumResults() == 0) {
        c.line("scf.if " + irId(c, op.getOperand(0)) + " {");
      } else if (op.getNumResults() == 1) {
        c.line(irId(c, op.getResult(0)) + "=scf.if " + irId(c, op.getOperand(0)) +
               " -> (" + llvm::join(tys, ",") + ") {");
      } else {
        std::string tup = irId(c, op.getResult(0)) + "_t";
        c.line(tup + ":" + std::to_string(op.getNumResults()) + "=scf.if " +
               irId(c, op.getOperand(0)) + " -> (" + llvm::join(tys, ",") +
               ") {");
      }
    } else if (at == PrintPoint::BeginRegion) {
      if (c.region == 0) {
        c.indent++;
      } else if (hasElse) {
        c.line("} else {");
        c.indent++;
      }
    } else if (at == PrintPoint::EndRegion) {
      if (c.region == 0) {
        c.indent--;
        if (!hasElse)
          c.line("}");
      } else if (hasElse) {
        c.indent--;
        c.line("}");
        if (op.getNumResults() > 1) {
          std::string tup = irId(c, op.getResult(0)) + "_t";
          emitTupleUnpack(c, op, tup, resTys);
        }
      }
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_yield"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_subquant_level"] = PrintRule::custom([](Operation &op,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_extract"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.task_handoff"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("\"weft_kernel.handoff\"(" + irId(c, op.getOperand(0)) + "," +
             irId(c, op.getOperand(1)) + "):(si32,si32)->()");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.task_commit"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("\"weft_kernel.commit\"(" + irId(c, op.getOperand(0)) + "," +
             irId(c, op.getOperand(1)) + "):(" +
             op.getAttrOfType<StringAttr>("val_type").str() + "," +
             op.getAttrOfType<StringAttr>("slice_type").str() + ")->()");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  // ---------------- Region 1 ("host.c") Ops ----------------
  p.operations["arith.constant"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  auto hostBin = [&](StringRef opName, StringRef cSym, bool isFloat) {
    p.operations[opName.str()] = PrintRule::custom([cSym, isFloat](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
      if (at == PrintPoint::Enter) {
        std::string cTy = isFloat ? "const float " : "const int64_t ";
        c.line(cTy + c.name(op.getResult(0)) + " = " +
               c.name(op.getOperand(0)) + " " + cSym.str() + " " +
               c.name(op.getOperand(1)) + ";");
      }
      return llvm::Error::success();
    }, "legacy_operation_rule");
  };
  hostBin("arith.addi", "+", false);
  hostBin("arith.subi", "-", false);
  hostBin("arith.muli", "*", false);
  hostBin("arith.divsi", "/", false);
  hostBin("arith.remsi", "%", false);
  hostBin("arith.addf", "+", true);
  hostBin("arith.subf", "-", true);
  hostBin("arith.mulf", "*", true);

  p.operations["arith.extf"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                  PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = expand_bf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  }, "legacy_operation_rule");
  p.operations["arith.truncf"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                    PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const uint16_t " + c.name(op.getResult(0)) + " = narrow_bf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  }, "legacy_operation_rule");
  p.operations["arith.minimumf"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                      PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = min_ieee(" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ");");
    return llvm::Error::success();
  }, "legacy_operation_rule");
  p.operations["math.exp"] = PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = expf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.address"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.host_fn"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
        const std::string q8Elems = std::to_string(Q8::elements);
        const std::string q8Groups = std::to_string(Q8::elements / Q8::sumGroup);
        const std::string q8GrpSz = std::to_string(Q8::sumGroup);
        c.line("struct Record { float d; int8_t q[" + q8Elems +
               "]; int16_t sums[" + q8Groups + "]; };");
        c.line("_Static_assert(sizeof(struct Record)==" +
               std::to_string(Q8::bytes) + ", \"Q8_K record ABI\");");
        c.line("static inline void pack_q8_record(struct Record *out, const float *x, int64_t stride) {");
        c.line("  float extreme=0.0f;");
        c.line("  for (int64_t k=0; k<" + q8Elems +
               "; ++k) { float v=x[k*stride]; if (fabsf(v)>fabsf(extreme)) extreme=v; }");
        c.line("  float inv=extreme==0.0f ? 0.0f : -127.0f/extreme;");
        c.line("  out->d=extreme==0.0f ? 0.0f : 1.0f/inv;");
        c.line("  for (int group=0; group<" + q8Groups + "; ++group) {");
        c.line("    int sum=0;");
        c.line("#if defined(__x86_64__)");
        c.line("    #pragma omp simd reduction(+:sum)");
        c.line("#endif");
        c.line("    for (int64_t k=" + q8GrpSz + "*group; k<" + q8GrpSz +
               "*group+" + q8GrpSz + "; ++k) {");
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
      c.line("static _Thread_local char g_intent_last_error[512];");
      c.line("const char *intent_last_error(void) { return g_intent_last_error; }");
      c.line("int32_t intent_entry(" +
             op.getAttrOfType<StringAttr>("params").str() + ") {");
      c.indent++;
      c.line("const int64_t M=" + std::to_string(ext[0]) +
             ", N=" + std::to_string(ext[1]) + ", K=" + std::to_string(ext[2]) +
             ", G=" + std::to_string(ext[3]) + ";");
      c.line("const int64_t mr=" + std::to_string(tl[0]) +
             ", nr=" + std::to_string(tl[1]) + ";");
      c.line("const int64_t mb=mr>0?(M+mr-1)/mr:0, nc=nr>0?(N+nr-1)/nr:0, jobs=G*mb*nc;");
      c.line("int32_t intent_status = 0;");
      c.line("char intent_err_buf[512]; intent_err_buf[0] = 0;");
      c.line("g_intent_last_error[0] = 0;");
      c.line("(void)rows; (void)columns; (void)grain; (void)panel;");
      if (op.getAttrOfType<BoolAttr>("uses_mx8_math").getValue())
        c.line("ensure_mx8_tables();");
      c.indent--;
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("if (intent_status != 0) { memcpy(g_intent_last_error, intent_err_buf, sizeof(g_intent_last_error)); g_intent_last_error[sizeof(g_intent_last_error) - 1] = 0; }");
      c.line("return intent_status;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_alloc"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.host_if"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("if (" + op.getAttrOfType<StringAttr>("cond").str() + ") {");
    else if (at == PrintPoint::BeginBlock)
      c.indent++;
    else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_omp_for"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      bool par = op.getAttrOfType<BoolAttr>("parallel").getValue();
      bool simd = op.getAttrOfType<BoolAttr>("simd").getValue();
      bool x86Simd = op.getAttrOfType<BoolAttr>("x86_simd").getValue();
      std::string ic = op.getAttrOfType<StringAttr>("if_cond").str();
      std::string iv = op.getAttrOfType<StringAttr>("iv").str();
      std::string lb = op.getAttrOfType<StringAttr>("lb").str();
      std::string ub = op.getAttrOfType<StringAttr>("ub").str();
      std::string st = op.getAttrOfType<StringAttr>("step").str();
      if (simd) {
        if (x86Simd) {
          c.line("#if defined(__x86_64__)");
          c.line("#pragma omp simd");
          c.line("#endif");
        } else {
          c.line("#pragma omp simd");
        }
      } else if (par) {
        c.line("#pragma omp parallel for num_threads(workers) schedule(static)" +
               (ic.empty() ? "" : " if(" + ic + ")"));
      } else {
        c.line("#pragma omp for schedule(static)");
      }
      c.line("for (int64_t " + iv + "=" + lb + "; " + iv + "<" + ub + "; " +
             iv + "+=" + st + ") {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_for"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string iv = op.getAttrOfType<StringAttr>("iv").str();
      std::string lb = op.getAttrOfType<StringAttr>("lb").str();
      std::string ub = op.getAttrOfType<StringAttr>("ub").str();
      std::string st = op.getAttrOfType<StringAttr>("step").str();
      c.line("for (int64_t " + iv + "=" + lb + "; " + iv + "<" + ub + "; " +
             iv + "+=" + st + ") {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_let_i64"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const int64_t " + op.getAttrOfType<StringAttr>("name").str() +
             " = " + op.getAttrOfType<StringAttr>("expr").str() + ";");
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_load_scalar"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                                  PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t s = op.getAttrOfType<IntegerAttr>("slot").getInt();
      std::string cTy = op.getAttrOfType<StringAttr>("c_type").str();
      std::string idx = op.getAttrOfType<StringAttr>("index_expr").str();
      std::string base =
          s >= 0 ? ("a" + std::to_string(s)) : ("wk_" + std::to_string(-1 - s));
      c.line("const " + cTy + " " + c.name(op.getResult(0)) + " = ((const " +
             cTy + "*)" + base + ")[" + idx + "];");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_ptr_offset"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string cTy = op.getAttrOfType<StringAttr>("c_type").str();
      std::string nm = op.getAttrOfType<StringAttr>("name").str();
      std::string bs = op.getAttrOfType<StringAttr>("base_expr").str();
      std::string of = op.getAttrOfType<StringAttr>("offset_expr").str();
      c.line(cTy + " *" + nm + " = (" + cTy + "*)(" + bs + ") + (" + of + ");");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_span_op"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto kind = op.getAttrOfType<StringAttr>("op").getValue();
    std::string dst = op.getAttrOfType<StringAttr>("dst_expr").str();
    std::string src = op.getAttrOfType<StringAttr>("src_expr").str();
    std::string cnt = op.getAttrOfType<StringAttr>("count_expr").str();
    int64_t eb = op.getAttrOfType<IntegerAttr>("elem_bytes").getInt();
    if (kind == "widen_bf16")
      c.line("widen_bf16_span(" + dst + ", " + src + ", " + cnt + ");");
    else if (kind == "widen_f16")
      c.line("widen_f16_span(" + dst + ", " + src + ", " + cnt + ");");
    else if (kind == "memcpy")
      c.line("memcpy(" + dst + ", " + src + ", (size_t)(" + cnt + ") * " +
             std::to_string(eb) + "ULL);");
    else if (kind == "zero")
      c.line("memset(" + dst + ", 0, (size_t)(" + cnt + ") * " +
             std::to_string(eb) + "ULL);");
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_elem_op"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto kind = op.getAttrOfType<StringAttr>("op").getValue();
    std::string dst = op.getAttrOfType<StringAttr>("dst_expr").str();
    std::string src = op.getAttrOfType<StringAttr>("src_expr").str();
    std::string aux = op.getAttrOfType<StringAttr>("aux_expr").str();
    int64_t eb = op.getAttrOfType<IntegerAttr>("elem_bytes").getInt();
    if (kind == "expand_bf")
      c.line(dst + " = expand_bf(" + src + ");");
    else if (kind == "expand_f16")
      c.line(dst + " = expand_f16(" + src + ");");
    else if (kind == "assign")
      c.line(dst + " = " + src + ";");
    else if (kind == "memcpy")
      c.line("memcpy(" + dst + ", " + src + ", " + std::to_string(eb) + ");");
    else if (kind == "pack_q8k")
      c.line("pack_q8_record(" + dst + ", " + src + ", " + aux + ");");
    else if (kind == "let_mx8_scale")
      c.line("const float " + dst + " = g_mx8_scale[" + src + "];");
    else if (kind == "mx8_scale")
      c.line(dst + " = g_mx8_scale[" + src + "];");
    else if (kind == "mx8_mul")
      c.line(dst + " = g_mx8_val[" + src + "] * " + aux + ";");
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_panel_yield"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                                  PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    bool wide = op.getAttrOfType<BoolAttr>("widen_bf16").getValue();
    std::string dst = op.getAttrOfType<StringAttr>("dst_expr").str();
    std::string v = c.name(op.getOperand(0));
    c.line(dst + " = " + (wide ? "expand_bf(" + v + ")" : v) + ";");
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_parallel_tiles"] = PrintRule::custom([](Operation &op,
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
      c.line("if (__atomic_load_n(&intent_status, __ATOMIC_RELAXED) != 0) continue;");
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
  }, "legacy_operation_rule");

  p.operations["weft_host.host_view_ptr"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.host_call_task"] = PrintRule::custom([](Operation &op, PrintPoint at,
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
    c.line("{ char task_err[512]; int32_t task_st = (int32_t)weft_kernel_take_status(task_err, sizeof(task_err));");
    c.line("  if (task_st != 0) { int32_t expected_st = 0;");
    c.line("    if (__atomic_compare_exchange_n(&intent_status, &expected_st, task_st, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {");
    c.line("      memcpy(intent_err_buf, task_err, sizeof(intent_err_buf)); intent_err_buf[sizeof(intent_err_buf) - 1] = 0; }");
    c.line("    continue; } }");
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_scatter_routed"] = PrintRule::custom([](Operation &op,
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
  }, "legacy_operation_rule");

  p.operations["weft_host.host_free"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string name = op.getAttrOfType<StringAttr>("buf_name").str();
      int wsSlot = name == "left" ? 0 : name == "rhs" ? 1 : name == "members" ? 2 : 3;
      c.line("intent_ws_release(" + std::to_string(wsSlot) + ", " + name + ");");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_row_fn"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("#include <stddef.h>");
      c.line("#include <stdint.h>");
      c.line("#include \"task_bridge.h\"");
      c.line("int32_t intent_entry(" +
             op.getAttrOfType<StringAttr>("params").str() + ") {");
      c.indent++;
      c.line("if (rows < 0 || cols < 0 || grain <= 0 || panel < 0 || workers "
             "<= 0) return 1;");
      c.line("if (!rows) return 0;");
      auto walk = op.getAttrOfType<StringAttr>("walk").getValue();
      if (walk == "destination") {
        c.line("{ int64_t t = rows; rows = cols; cols = t; t = a0s0; a0s0 = "
               "a0s1; a0s1 = t; t = a1s0; a1s0 = a1s1; a1s1 = t; }");
      }
      if (op.getAttrOfType<BoolAttr>("member_tasks").getValue())
        c.line("rows = cols;");
      if (op.getAttrOfType<BoolAttr>("single_dispatcher").getValue())
        c.line("workers = 1;");
      c.indent--;
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("return 0;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_parallel_transpose_tiles"] =
      PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter) {
          auto &bb = op.getRegion(0).front();
          std::string st =
              std::to_string(op.getAttrOfType<IntegerAttr>("tile").getInt());
          std::string rN = c.name(bb.getArgument(0));
          std::string cN = c.name(bb.getArgument(1));
          std::string nrN = c.name(bb.getArgument(2));
          std::string ncN = c.name(bb.getArgument(3));
          c.line("const int64_t nt = (cols - 1) / " + st +
                 " + 1, mt = (rows - 1) / " + st + " + 1;");
          c.line("#pragma omp parallel for schedule(static) "
                 "num_threads(workers) if(workers > 1)");
          c.line("for (int64_t g = 0; g < mt * nt; ++g) {");
          c.indent++;
          c.line("const int64_t " + rN + " = (g / nt) * " + st + ", " + cN +
                 " = (g % nt) * " + st + ";");
          c.line("const int64_t " + nrN + " = rows - " + rN + " < " + st +
                 " ? rows - " + rN + " : " + st + ", " + ncN + " = cols - " +
                 cN + " < " + st + " ? cols - " + cN + " : " + st + ";");
          c.indent--;
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      }, "legacy_operation_rule");

  p.operations["weft_host.host_parallel_key_ranges"] =
      PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter) {
          auto &bb = op.getRegion(0).front();
          int64_t keys = op.getAttrOfType<IntegerAttr>("key_extent").getInt();
          c.line("const int64_t keys = " + std::to_string(keys) +
                 ", lines = (keys + 15) / 16, n_thr = workers < lines ? "
                 "workers : lines;");
          c.line("#pragma omp parallel for num_threads(n_thr) schedule(static) "
                 "if(n_thr > 1)");
          c.line("for (int64_t b = 0; b < n_thr; ++b) {");
          c.indent++;
          c.line("const int64_t " + c.name(bb.getArgument(0)) + " = 0, " +
                 c.name(bb.getArgument(1)) + " = rows;");
          c.line("const int64_t " + c.name(bb.getArgument(2)) +
                 " = 16 * (lines * b / n_thr), " + c.name(bb.getArgument(3)) +
                 " = 16 * (lines * (b + 1) / n_thr);");
          c.indent--;
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      }, "legacy_operation_rule");

  p.operations["weft_host.host_parallel_row_intervals"] =
      PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter) {
          auto &bb = op.getRegion(0).front();
          std::string chunk =
              op.getAttrOfType<StringAttr>("chunk_expr").str();
          std::string stN = c.name(bb.getArgument(0));
          std::string ctN = c.name(bb.getArgument(1));
          c.line("const int64_t groups = 1 + (rows - 1) / grain;");
          c.line("const int64_t bins = groups < workers ? groups : workers;");
          c.line("#pragma omp parallel for num_threads(bins) schedule(static) "
                 "if(bins > 1)");
          c.line("for (int64_t b = 0; b < bins; ++b) {");
          c.indent++;
          c.line("const int64_t group = (groups / bins * b + (groups % bins * "
                 "b) / bins) * grain;");
          c.line("const int64_t end = b + 1 == bins ? rows : (groups / bins * "
                 "(b + 1) + (groups % bins * (b + 1)) / bins) * grain;");
          c.line("for (int64_t cur = group; cur < end;) {");
          c.indent++;
          c.line("const int64_t rem = " + chunk + " - cur % " + chunk + ";");
          c.line("const int64_t " + stN + " = cur, " + ctN +
                 " = end - cur < rem ? end - cur : rem;");
          c.line("cur += " + ctN + ";");
          c.indent -= 2;
        } else if (at == PrintPoint::BeginBlock) {
          c.indent += 2;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      }, "legacy_operation_rule");

  p.operations["weft_host.host_row_offset"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto mode = op.getAttrOfType<StringAttr>("mode").getValue();
    auto dims = op.getAttrOfType<ArrayAttr>("dim_vars");
    auto strides = op.getAttrOfType<ArrayAttr>("stride_vars");
    std::string outN = c.name(op.getResult(0));
    if (mode == "linear") {
      c.line("const int64_t " + outN + " = (" + c.name(op.getOperand(0)) +
             ") * " + cast<StringAttr>(strides[0]).str() + ";");
    } else if (mode == "tile2d") {
      c.line("const int64_t " + outN + " = (" + c.name(op.getOperand(0)) +
             ") * " + cast<StringAttr>(strides[0]).str() + " + (" +
             c.name(op.getOperand(1)) + ") * " +
             cast<StringAttr>(strides[1]).str() + ";");
    } else {
      std::string row = c.name(op.getOperand(0));
      SmallVector<std::string> terms{"0LL"};
      for (int j = int(dims.size()) - 1; j >= 0; --j) {
        std::string dm = cast<StringAttr>(dims[j]).str();
        std::string st = cast<StringAttr>(strides[j]).str();
        terms.push_back("((" + row + ") % " + dm + ") * " + st);
        row = "(" + row + " / " + dm + ")";
      }
      c.line("const int64_t " + outN + " = " + llvm::join(terms, " + ") + ";");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_call_row_task"] =
      PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &c) {
        if (at != PrintPoint::Enter)
          return llvm::Error::success();
        SmallVector<std::string> args;
        for (Attribute a : op.getAttrOfType<ArrayAttr>("slot_args"))
          args.push_back(cast<StringAttr>(a).str());
        for (Attribute a : op.getAttrOfType<ArrayAttr>("shape_exprs")) {
          std::string raw = cast<StringAttr>(a).str();
          std::string expr;
          for (size_t i = 0; i < raw.size();) {
            if (raw[i] == '$' && i + 1 < raw.size() &&
                std::isdigit(static_cast<unsigned char>(raw[i + 1]))) {
              size_t j = i + 1;
              while (j < raw.size() &&
                     std::isdigit(static_cast<unsigned char>(raw[j])))
                ++j;
              unsigned idx = std::stoul(raw.substr(i + 1, j - (i + 1)));
              expr += c.name(op.getOperand(idx));
              i = j;
            } else {
              expr.push_back(raw[i++]);
            }
          }
          args.push_back("(size_t)(" + expr + ")");
        }
        for (Attribute a : op.getAttrOfType<ArrayAttr>("stride_exprs"))
          args.push_back("(size_t)(" + cast<StringAttr>(a).str() + ")");
        c.line(op.getAttrOfType<StringAttr>("sym_name").str() + "_instance(" +
               llvm::join(args, ", ") + ");");
        return llvm::Error::success();
      }, "legacy_operation_rule");

  p.operations["weft_host.host_owner_fn"] = PrintRule::custom([](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("#include <stddef.h>");
      c.line("#include <stdint.h>");
      c.line("#include \"task_bridge.h\"");
      c.line("int32_t intent_entry(" +
             op.getAttrOfType<StringAttr>("params").str() + ") {");
      c.indent++;
      c.line("const int64_t count = " +
             std::to_string(op.getAttrOfType<IntegerAttr>("owners").getInt()) +
             "LL, tile = " +
             std::to_string(op.getAttrOfType<IntegerAttr>("grain").getInt()) +
             "LL;");
      c.line("(void)rows; (void)columns; (void)grain; (void)panel;");
      c.indent--;
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("return 0;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  }, "legacy_operation_rule");

  p.operations["weft_host.host_parallel_owners"] =
      PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter) {
          auto &bb = op.getRegion(0).front();
          std::string loN = c.name(bb.getArgument(0));
          std::string hiN = c.name(bb.getArgument(1));
          auto dynAttr = op.getAttrOfType<BoolAttr>("dynamic");
          if (dynAttr && dynAttr.getValue()) {
            c.line("const int64_t chunks = (count + tile - 1) / tile;");
            c.line("#pragma omp parallel for num_threads(workers) "
                   "schedule(dynamic, 1) if(workers > 1 && chunks > 1)");
            c.line("for (int64_t ch_i = 0; ch_i < chunks; ++ch_i) {");
            c.indent++;
            c.line("const int64_t lo_i = ch_i * tile;");
            c.line("const int64_t " + loN + " = lo_i, " + hiN +
                   " = count - lo_i < tile ? count : lo_i + tile;");
            c.indent--;
          } else {
            c.line("#pragma omp parallel for num_threads(workers) "
                   "schedule(static, 1) if(workers > 1 && count > tile)");
            c.line("for (int64_t lo_i = 0; lo_i < count; lo_i += tile) {");
            c.indent++;
            c.line("const int64_t " + loN + " = lo_i, " + hiN +
                   " = count - lo_i < tile ? count : lo_i + tile;");
            c.indent--;
          }
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      }, "legacy_operation_rule");

  p.operations["weft_host.host_call_owner_task"] =
      PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &c) {
        if (at != PrintPoint::Enter)
          return llvm::Error::success();
        SmallVector<std::string> args;
        for (Attribute a : op.getAttrOfType<ArrayAttr>("slot_args"))
          args.push_back(cast<StringAttr>(a).str());
        args.push_back("(size_t)(" + c.name(op.getOperand(0)) + ")");
        args.push_back("(size_t)(" + c.name(op.getOperand(1)) + ")");
        if (op.getAttrOfType<BoolAttr>("empty_extent").getValue())
          args.push_back("(size_t)0ULL");
        for (Attribute a : op.getAttrOfType<ArrayAttr>("stride_exprs"))
          args.push_back("(size_t)(" + cast<StringAttr>(a).str() + ")");
        c.line(op.getAttrOfType<StringAttr>("sym_name").str() + "_instance(" +
               llvm::join(args, ", ") + ");");
        return llvm::Error::success();
      }, "legacy_operation_rule");

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

TerminalPrinter weftMatrixPrinter() { return weftTerminalPrinter(); }
} // namespace CI_UNIT_NAMESPACE
