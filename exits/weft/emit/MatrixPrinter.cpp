#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/cpu/ir/QuantDecode.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include <cctype>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

namespace {
std::string irId(StringRef name) { return ("%" + name).str(); }

void emitTupleUnpack(PrintContext &c, const LocalOpView &view, StringRef tup,
                     ArrayAttr tys) {
  for (unsigned i = 0; i < view.numResults(); ++i) {
    std::string t = cast<StringAttr>(tys[i]).str();
    std::string rId = irId(view.result(i));
    std::string zId = rId + "_z";
    if (t == "index") {
      c.line(zId + "=arith.constant 0:index");
      c.line(rId + "=\"weft_kernel.binary\"(" + tup.str() + "#" +
             std::to_string(i) + "," + zId +
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
    c.line(rId + "=\"weft_kernel.binary\"(" + tup.str() + "#" +
           std::to_string(i) + "," + zId + ") <{kind=\"add\"}>:(" + t + "," +
           scTy + ")->" + t);
  }
}

std::string formatEncodingDecl(const LocalOpView &view) {
  std::string name = view.attrOfType<StringAttr>("sym_name").str();
  int64_t align = view.attrOfType<IntegerAttr>("alignment").getInt();
  int64_t elems = view.attrOfType<IntegerAttr>("elements").getInt();
  int64_t storageBits = view.attrOfType<IntegerAttr>("storage_bits").getInt();
  auto fNames = view.attrOfType<ArrayAttr>("field_names");
  auto fTypes = view.attrOfType<ArrayAttr>("field_types");
  auto fExts = view.attrOfType<DenseI64ArrayAttr>("field_extents");
  auto fLayouts = view.attrOfType<ArrayAttr>("field_layouts");
  auto fOffsets = view.attrOfType<DenseI64ArrayAttr>("field_bit_offsets");
  auto fBits = view.attrOfType<DenseI64ArrayAttr>("field_storage_bits");

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
  auto &rules = p.operations;

  auto enterLocal =
      [&](StringRef name,
          std::function<llvm::Error(const LocalOpView &, PrintContext &)> fn) {
        rules[name.str()] = PrintRule::local(
            [fn = std::move(fn)](const LocalOpView &view, PrintPoint pt,
                                 PrintContext &c) -> llvm::Error {
              if (pt != PrintPoint::Enter)
                return llvm::Error::success();
              return fn(view, c);
            });
      };

  // Declarative format and regionFormat rules.
  rules["builtin.module"] = PrintRule::format("");
  rules["func.func"] = PrintRule::format("");
  rules["func.return"] = PrintRule::format("");
  rules["weft_host.end"] = PrintRule::format("");

  rules["weft_host.task_new"] = PrintRule::format(
      "%$r0=\"weft_kernel.new\"(%$0) <{{initialized=true}}>:({in_type})->{val_type}");
  rules["weft_host.task_unary"] = PrintRule::format(
      "%$r0=\"weft_kernel.unary\"(%$0) <{{kind=\"{kind}\"}}>:({val_type})->{val_type}");
  rules["weft_host.task_reduce"] = PrintRule::format(
      "%$r0=\"weft_kernel.reduce\"(%$0) <{{axis={axis}:i64,kind=\"{kind}\"}}>:({in_type})->{val_type}");
  rules["weft_host.task_scan_yield"] =
      PrintRule::format("\"weft_kernel.scan_yield\"(%$0):({val_type})->()");
  rules["weft_host.task_iota"] = PrintRule::format(
      "%$r0=\"weft_kernel.iota\"() <{{end={end}:i64,start={start}:i64}}>:()->{val_type}");
  rules["weft_host.task_local_alloc"] = PrintRule::format(
      "%$r0=\"weft_kernel.local_alloc\"() <{{alignment={alignment}:i64,elements={elements}:i64}}>:()->{ptr_type}");
  rules["weft_host.task_atomic_rmw"] = PrintRule::format(
      "%$r0=\"weft_kernel.atomic_rmw\"(%$0,%$1) <{{kind=\"{kind}\",ordering=\"{ordering}\"}}>:({val_type},{slice_type})->{val_type}");
  enterLocal("weft_host.task_binary", [](const LocalOpView &view, PrintContext &c) {
    std::string flags;
    if (auto attr = view.attrOfType<Attribute>("fastmath")) {
      llvm::raw_string_ostream os(flags);
      os << " {fastmath = " << attr << "}";
    }
    c.line(irId(view.result(0)) + "=\"weft_kernel.binary\"(" + irId(view.operand(0)) +
           "," + irId(view.operand(1)) + ") <{kind=\"" + view.attrOfType<StringAttr>("kind").str() +
           "\"}>" + flags + ":(" + view.attrOfType<StringAttr>("lhs_type").str() + "," +
           view.attrOfType<StringAttr>("rhs_type").str() + ")->" + view.attrOfType<StringAttr>("val_type").str());
    return llvm::Error::success();
  });
  rules["weft_host.task_compare"] = PrintRule::format(
      "%$r0=\"weft_kernel.compare\"(%$0,%$1) <{{predicate=\"{predicate}\"}}>:({lhs_type},{rhs_type})->{val_type}");
  rules["weft_host.task_assume"] =
      PrintRule::format("\"weft_kernel.assume\"(%$0):(i1)->()");
  rules["weft_host.task_reshape"] = PrintRule::format(
      "%$r0=\"weft_kernel.reshape\"(%$0) "
      "<{{order=array<i64:{order}>}}>:({in_type})->{val_type}");
  rules["weft_host.task_cast"] = PrintRule::format(
      "%$r0=\"weft_kernel.{op_kind}\"(%$0):({in_type})->{val_type}");
  enterLocal("weft_host.task_narrow",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               std::string resId = irId(view.result(0));
               std::string inId = irId(view.operand(0));
               std::string rounding =
                   view.attrOfType<StringAttr>("rounding").str();
               bool sat = view.attrOfType<BoolAttr>("saturate").getValue();
               std::string inTy = view.attrOfType<StringAttr>("in_type").str();
               std::string valTy = view.attrOfType<StringAttr>("val_type").str();
               c.line(resId + "=\"weft_kernel.narrow\"(" + inId +
                      ") <{rounding=\"" + rounding + "\",saturate=" +
                      (sat ? "true" : "false") + "}>:(" + inTy + ")->" + valTy);
               return llvm::Error::success();
             });
  rules["weft_host.task_field"] = PrintRule::format(
      "%$r0=\"weft_kernel.field\"(%$0) <{{name=\"{field_name}\"}}>:({in_type})->{val_type}");
  rules["weft_host.task_fold2"] =
      PrintRule::format("%$r0=\"weft_kernel.fold2\"(%$0):({in_type})->{val_type}");
  enterLocal("weft_host.task_extract",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               std::string sels =
                   "\"" + view.attrOfType<StringAttr>("selector").str() + "\"";
               if (view.attr("keep_columns"))
                 sels += ",\"all\"";
               c.line(irId(view.result(0)) + "=\"weft_kernel.extract\"(" +
                      irId(view.operand(0)) + "," + irId(view.operand(1)) +
                      ") <{selectors=[" + sels + "]}>:(" +
                      view.attrOfType<StringAttr>("in_type").str() + "," +
                      view.attrOfType<StringAttr>("point_type").str() + ")->" +
                      view.attrOfType<StringAttr>("val_type").str());
               return llvm::Error::success();
             });
  rules["weft_host.task_lookup"] = PrintRule::format(
      "%$r0=\"weft_kernel.lookup\"(%$0,%$1) <{{bounds=\"{bounds}\"}}>:({table_type},{idx_type})->{val_type}");
  enterLocal("weft_host.task_handoff",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               SmallVector<std::string, 4> ops, tys;
               for (unsigned i = 0; i < view.numOperands(); ++i) {
                 ops.push_back(irId(view.operand(i)));
                 tys.push_back("si32");
               }
               c.line("\"weft_kernel.handoff\"(" + llvm::join(ops, ",") +
                      "):(" + llvm::join(tys, ",") + ")->()");
               return llvm::Error::success();
             });
  rules["weft_host.task_commit"] = PrintRule::format(
      "\"weft_kernel.commit\"(%$0,%$1):({val_type},{slice_type})->()");

  rules["arith.addi"] = PrintRule::format("const int64_t $r0 = $0 + $1;");
  rules["arith.subi"] = PrintRule::format("const int64_t $r0 = $0 - $1;");
  rules["arith.muli"] = PrintRule::format("const int64_t $r0 = $0 * $1;");
  rules["arith.divsi"] = PrintRule::format("const int64_t $r0 = $0 / $1;");
  rules["arith.remsi"] = PrintRule::format("const int64_t $r0 = $0 % $1;");
  rules["arith.addf"] = PrintRule::format("const float $r0 = $0 + $1;");
  rules["arith.subf"] = PrintRule::format("const float $r0 = $0 - $1;");
  rules["arith.mulf"] = PrintRule::format("const float $r0 = $0 * $1;");
  rules["arith.extf"] = PrintRule::format("const float $r0 = expand_bf($0);");
  rules["arith.truncf"] =
      PrintRule::format("const uint16_t $r0 = narrow_bf($0);");
  rules["arith.minimumf"] =
      PrintRule::format("const float $r0 = min_ieee($0, $1);");
  rules["math.exp"] = PrintRule::format("const float $r0 = expf($0);");

  auto substVals = [](StringRef raw, const LocalOpView &view) -> std::string {
    if (view.numOperands() == 0 || raw.find('$') == StringRef::npos)
      return raw.str();
    std::string out;
    out.reserve(raw.size() + 16);
    for (size_t i = 0; i < raw.size();) {
      if (raw[i] == '$' && i + 1 < raw.size() &&
          std::isdigit(static_cast<unsigned char>(raw[i + 1]))) {
        size_t j = i + 1;
        while (j < raw.size() &&
               std::isdigit(static_cast<unsigned char>(raw[j])))
          ++j;
        unsigned idx = std::stoul(raw.substr(i + 1, j - (i + 1)).str());
        out += view.operand(idx);
        i = j;
      } else {
        out.push_back(raw[i++]);
      }
    }
    return out;
  };

  rules["weft_host.host_guard_return"] =
      PrintRule::format("if ({cond}) return {code};");

  enterLocal("weft_host.host_guard_continue",
             [substVals](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               c.line("if (" +
                      substVals(view.attrOfType<StringAttr>("cond").getValue(),
                                view) +
                      ") continue;");
               return llvm::Error::success();
             });

  rules["weft_host.host_if"] = PrintRule::local(
      [substVals](const LocalOpView &view, PrintPoint at,
                  PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter)
          c.line("if (" +
                 substVals(view.attrOfType<StringAttr>("cond").getValue(),
                           view) +
                 ") {");
        else if (at == PrintPoint::BeginBlock)
          c.indent++;
        else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["weft_host.host_for"] = PrintRule::local(
      [substVals](const LocalOpView &view, PrintPoint at,
                  PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          std::string iv = view.attrOfType<StringAttr>("iv").str();
          std::string lb =
              substVals(view.attrOfType<StringAttr>("lb").getValue(), view);
          std::string ub =
              substVals(view.attrOfType<StringAttr>("ub").getValue(), view);
          std::string st =
              substVals(view.attrOfType<StringAttr>("step").getValue(), view);
          c.line("for (int64_t " + iv + "=" + lb + "; " + iv + "<" + ub + "; " +
                 iv + "+=" + st + ") {");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["weft_host.host_while"] = PrintRule::local(
      [substVals](const LocalOpView &view, PrintPoint at,
                  PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter)
          c.line("while (" +
                 substVals(view.attrOfType<StringAttr>("cond").getValue(),
                           view) +
                 ") {");
        else if (at == PrintPoint::BeginBlock)
          c.indent++;
        else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  enterLocal("weft_host.host_let_i64",
             [substVals](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mut = view.attrOfType<BoolAttr>("is_mut");
               std::string prefix =
                   (mut && mut.getValue()) ? "int64_t " : "const int64_t ";
               c.line(prefix + view.attrOfType<StringAttr>("name").str() +
                      " = " +
                      substVals(view.attrOfType<StringAttr>("expr").getValue(),
                                view) +
                      ";");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_def_i64",
             [substVals](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mut = view.attrOfType<BoolAttr>("is_mut");
               std::string prefix =
                   (mut && mut.getValue()) ? "int64_t " : "const int64_t ";
               c.line(prefix + view.result(0).str() + " = " +
                      substVals(view.attrOfType<StringAttr>("expr").getValue(),
                                view) +
                      ";");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_ptr_offset",
             [substVals](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               std::string cTy = view.attrOfType<StringAttr>("c_type").str();
               std::string name = view.attrOfType<StringAttr>("name").str();
               std::string base = substVals(
                   view.attrOfType<StringAttr>("base_expr").getValue(), view);
               std::string off = substVals(
                   view.attrOfType<StringAttr>("offset_expr").getValue(), view);
               c.line(cTy + " *" + name + " = (" + cTy + "*)(" + base +
                      ") + (" + off + ");");
               return llvm::Error::success();
             });

  // Strictly local rules (only LocalOpView, zero Operation& access).
  auto programRule = PrintRule::local(
      [](const LocalOpView &, PrintPoint at, PrintContext &c) -> llvm::Error {
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
      });
  rules["weft_host.matrix_program"] = programRule;
  rules["weft_host.row_program"] = programRule;
  rules["weft_host.owner_program"] = programRule;

  enterLocal("weft_host.encoding_decl",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               c.line(formatEncodingDecl(view));
               return llvm::Error::success();
             });

  rules["weft_host.task_kernel"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginBlock) {
          auto bArgs = view.blockArguments();
          auto vTypes = view.attrOfType<ArrayAttr>("view_types");
          auto writes = view.attrOfType<ArrayAttr>("arg_writes");
          auto accessModes = view.attrOfType<ArrayAttr>("arg_access");
          auto aliasModes = view.attrOfType<DenseI64ArrayAttr>("arg_aliases");
          auto shapeSymsAttr = view.attrOfType<ArrayAttr>("shape_symbols");
          auto strided = view.attrOfType<DenseI64ArrayAttr>("strided_slots");
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
              effects.push_back(cast<BoolAttr>(writes[i]).getValue()
                                    ? "\"write\""
                                    : "\"read\"");
            }
            aliases.push_back(aliasModes ? std::to_string(aliasModes[i]) : "0");
            params.push_back(irId(bArgs[i]) + ":" +
                             cast<StringAttr>(vTypes[i]).str());
          }
          for (int64_t idx : strided.asArrayRef())
            stridedNames.push_back("\"a" + std::to_string(idx) + "\"");

          c.line("\"weft_kernel.kernel\"() <{sym_name=\"" +
                 view.attrOfType<StringAttr>("sym_name").str() + "\",source=\"" +
                 view.attrOfType<StringAttr>("source").str() + "\",arg_names=[" +
                 llvm::join(names, ",") + "],arg_access=[" +
                 llvm::join(effects, ",") + "],arg_alias_sets=array<i64:" +
                 llvm::join(aliases, ",") + ">,shape_symbols=[" +
                 llvm::join(symLiterals, ",") + "],strided_args=[" +
                 llvm::join(stridedNames, ",") + "]}> ({");
          c.line("^bb0(" + llvm::join(params, ",") + "):");
          size_t nViews = vTypes.size();
          for (size_t k = 0; k < symNames.size(); ++k) {
            c.line(irId(bArgs[nViews + k]) +
                   "=\"weft_kernel.symbol\"() "
                   "<{choices=array<i64>,kind=\"shape\",name=\"" +
                   symNames[k] + "\"}>:()->index");
          }
          c.line(irId(bArgs[nViews + symNames.size()]) +
                 "=\"weft_kernel.root_domain\"():()->!weft_kernel.domain<\"root\","
                 "0,-1,0,\"root\",\"exact\">");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.line("\"weft_kernel.return\"():()->()");
          c.indent--;
          c.line("}):()->()");
        }
        return llvm::Error::success();
      });

  enterLocal("weft_host.task_const",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto lit = view.attrOfType<StringAttr>("lit").str();
               auto ty = view.attrOfType<StringAttr>("val_type").str();
               std::string r0 = irId(view.result(0));
               if (ty == "index")
                 c.line(r0 + "=arith.constant " + lit + ":index");
               else if (ty == "i1" && (lit == "true" || lit == "false"))
                 c.line(r0 + "=\"weft_kernel.constant\"() <{value=" + lit +
                        "}>:()->i1");
               else
                 c.line(r0 + "=\"weft_kernel.constant\"() <{value=" + lit + ":" +
                        ty + "}>:()->" + ty);
               return llvm::Error::success();
             });

  enterLocal("weft_host.task_slice",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto vTy = view.attrOfType<StringAttr>("view_type").str();
               auto slTy = view.attrOfType<StringAttr>("slice_type").str();
               auto vlTy = view.attrOfType<StringAttr>("val_type").str();
               bool admit = view.attrOfType<BoolAttr>("admit").getValue();
               SmallVector<std::string> sels, argNames, argTys{vTy};
               for (Attribute a : view.attrOfType<ArrayAttr>("selectors"))
                 sels.push_back("\"" + cast<StringAttr>(a).str() + "\"");
               for (unsigned idx = 0; idx < view.numOperands(); ++idx) {
                 argNames.push_back(irId(view.operand(idx)));
                 if (idx > 0)
                   argTys.push_back("index");
               }
               std::string resId = irId(view.result(0));
               std::string slId = admit ? (resId + "_s") : resId;
               c.line(slId + "=\"weft_kernel.slice\"(" +
                      llvm::join(argNames, ",") + ") <{selectors=[" +
                      llvm::join(sels, ",") + "]}>:(" +
                      llvm::join(argTys, ",") + ")->" + slTy);
               if (admit)
                 c.line(resId + "=\"weft_kernel.admit\"(" + slId + "):(" + slTy +
                        ")->" + vlTy);
               return llvm::Error::success();
             });

  enterLocal("weft_host.task_subview",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto slTy = view.attrOfType<StringAttr>("slice_type").str();
               auto vlTy = view.attrOfType<StringAttr>("val_type").str();
               bool admit = view.attrOfType<BoolAttr>("admit").getValue();
               std::string resId = irId(view.result(0));
               std::string subId = admit ? (resId + "_s") : resId;
               if (auto exts =
                       view.attrOfType<DenseI64ArrayAttr>("static_extents")) {
                 SmallVector<std::string, 4> extStrs, offStrs,
                     opArgs{irId(view.operand(0))}, opTys{slTy};
                 for (auto [idx, e] : llvm::enumerate(exts.asArrayRef())) {
                   extStrs.push_back(std::to_string(e));
                   offStrs.push_back("-1");
                   opArgs.push_back(irId(view.operand(1 + idx)));
                   opTys.push_back("index");
                 }
                 std::string outSlTy = slTy;
                 auto bracket = outSlTy.rfind(">,[");
                 if (bracket != std::string::npos) {
                   auto close = outSlTy.find(']', bracket + 2);
                   outSlTy.replace(bracket + 2, close - (bracket + 2) + 1,
                                   "[" + llvm::join(extStrs, ",") + "]");
                 }
                 c.line(subId + "=\"weft_kernel.subview\"(" +
                        llvm::join(opArgs, ",") + ") <{extents=array<i64:" +
                        llvm::join(extStrs, ",") + ">,offsets=array<i64:" +
                        llvm::join(offStrs, ",") + ">}>:(" +
                        llvm::join(opTys, ",") + ")->" + outSlTy);
                 if (admit)
                   c.line(resId + "=\"weft_kernel.admit\"(" + subId + "):(" +
                          outSlTy + ")->" + vlTy);
                 return llvm::Error::success();
               }
               int64_t staticExt =
                   view.attrOfType<IntegerAttr>("static_extent").getInt();
               std::string outSlTy = slTy;
               auto bracket = outSlTy.rfind(">,[");
               if (bracket != std::string::npos) {
                 auto close = outSlTy.find(']', bracket + 2);
                 outSlTy.replace(bracket + 2, close - (bracket + 2) + 1,
                                 "[" + std::to_string(staticExt) + "]");
               }
               if (staticExt > 0) {
                 c.line(subId + "=\"weft_kernel.subview\"(" +
                        irId(view.operand(0)) + "," + irId(view.operand(1)) +
                        ") <{extents=array<i64:" + std::to_string(staticExt) +
                        ">,offsets=array<i64:-1>}>:(" + slTy + ",index)->" +
                        outSlTy);
               } else {
                 c.line(subId + "=\"weft_kernel.subview\"(" +
                        irId(view.operand(0)) + "," + irId(view.operand(1)) +
                        "," + irId(view.operand(2)) +
                        ") <{extents=array<i64:-1>,offsets=array<i64:-1>}>:(" +
                        slTy + ",index,index)->" + outSlTy);
               }
               if (admit)
                 c.line(resId + "=\"weft_kernel.admit\"(" + subId + "):(" +
                        outSlTy + ")->" + vlTy);
               return llvm::Error::success();
             });

  rules["weft_host.task_scan"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        auto eTy = view.attrOfType<StringAttr>("elem_type").str();
        auto vTy = view.attrOfType<StringAttr>("val_type").str();
        if (at == PrintPoint::BeginBlock) {
          auto bArgs = view.blockArguments();
          int64_t axis = view.attrOfType<IntegerAttr>("axis").getInt();
          bool rev = view.attrOfType<BoolAttr>("reverse").getValue();
          c.line(irId(view.result(0)) + "=\"weft_kernel.scan\"(" +
                 irId(view.operand(0)) + ") <{axis=" + std::to_string(axis) +
                 ":i64,reverse=" + (rev ? "true" : "false") + "}> ({");
          c.line("^bb0(" + irId(bArgs[0]) + ":" + eTy + "," + irId(bArgs[1]) +
                 ":" + eTy + "):");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}):(" + vTy + ")->" + vTy);
        }
        return llvm::Error::success();
      });

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

  enterLocal("weft_host.task_ptr_load",
             [ptrIdxMaskTypes](const LocalOpView &view,
                               PrintContext &c) -> llvm::Error {
               auto pTy = view.attrOfType<StringAttr>("ptr_type").str();
               auto vTy = view.attrOfType<StringAttr>("val_type").str();
               auto [offTy, mskTy] = ptrIdxMaskTypes(vTy);
               c.line(irId(view.result(0)) + "=\"weft_kernel.pointer_load\"(" +
                      irId(view.operand(0)) + "," + irId(view.operand(1)) +
                      "," + irId(view.operand(2)) + "," +
                      irId(view.operand(3)) + "):(" + pTy + "," + offTy + "," +
                      mskTy + "," + vTy + ")->" + vTy);
               return llvm::Error::success();
             });

  enterLocal("weft_host.task_ptr_store",
             [ptrIdxMaskTypes](const LocalOpView &view,
                               PrintContext &c) -> llvm::Error {
               auto pTy = view.attrOfType<StringAttr>("ptr_type").str();
               auto vTy = view.attrOfType<StringAttr>("val_type").str();
               auto [offTy, mskTy] = ptrIdxMaskTypes(vTy);
               c.line("\"weft_kernel.pointer_store\"(" + irId(view.operand(0)) +
                      "," + irId(view.operand(1)) + "," +
                      irId(view.operand(2)) + "," + irId(view.operand(3)) +
                      "):(" + pTy + "," + offTy + "," + mskTy + "," + vTy +
                      ")->()");
               return llvm::Error::success();
             });

  enterLocal("weft_host.task_select",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto vTy = view.attrOfType<StringAttr>("val_type").str();
               auto tAttr = view.attrOfType<StringAttr>("true_type");
               auto fAttr = view.attrOfType<StringAttr>("false_type");
               std::string tTy = tAttr ? tAttr.str() : vTy;
               std::string fTy = fAttr ? fAttr.str() : vTy;
               c.line(irId(view.result(0)) + "=\"weft_kernel.select\"(" +
                      irId(view.operand(0)) + "," + irId(view.operand(1)) +
                      "," + irId(view.operand(2)) + "):(" +
                      view.attrOfType<StringAttr>("cond_type").str() + "," +
                      tTy + "," + fTy + ")->" + vTy);
               return llvm::Error::success();
             });

  enterLocal("weft_host.task_contract",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto kind = view.attrOfType<StringAttr>("kind").getValue();
               auto vTy = view.attrOfType<StringAttr>("val_type").str();
               auto inTys = view.attrOfType<ArrayAttr>("in_types");
               std::string resId = irId(view.result(0));
               if (kind == "fma") {
                 std::string prodId = resId + "_m";
                 std::string tB = cast<StringAttr>(inTys[1]).str();
                 std::string tA = cast<StringAttr>(inTys[2]).str();
                 c.line(prodId + "=\"weft_kernel.binary\"(" +
                        irId(view.operand(1)) + "," + irId(view.operand(2)) +
                        ") <{kind=\"mul\"}>:(" + tB + "," + tA + ")->" + vTy);
                 c.line(resId + "=\"weft_kernel.binary\"(" +
                        irId(view.operand(0)) + "," + prodId +
                        ") <{kind=\"add\"}>:(" + vTy + "," + vTy + ")->" + vTy);
                 return llvm::Error::success();
               }
               SmallVector<std::string> ov;
               for (int64_t x :
                    view.attrOfType<DenseI64ArrayAttr>("over").asArrayRef())
                 ov.push_back(std::to_string(x));
               std::string attrs =
                   "over=array<i64:" + llvm::join(ov, ",") + ">,acc_type=" +
                   view.attrOfType<StringAttr>("acc_type").str() + ",engine=\"" +
                   view.attrOfType<StringAttr>("engine").str() + "\"";
               if (view.attrOfType<BoolAttr>("allow_subnormal_ftz").getValue())
                 attrs += ",allow_subnormal_ftz=true";
               c.line(resId + "=\"weft_kernel." + kind.str() + "\"(" +
                      irId(view.operand(0)) + "," + irId(view.operand(1)) +
                      ") <{" + attrs + "}>:(" +
                      cast<StringAttr>(inTys[0]).str() + "," +
                      cast<StringAttr>(inTys[1]).str() + ")->" + vTy);
               return llvm::Error::success();
             });

  rules["weft_host.task_for"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        auto iterTys = view.attrOfType<ArrayAttr>("iter_types");
        if (at == PrintPoint::BeginBlock) {
          auto bArgs = view.blockArguments();
          std::string iv = irId(bArgs[0]);
          std::string lb = irId(view.operand(0));
          std::string ub = irId(view.operand(1));
          std::string st = irId(view.operand(2));
          if (view.numResults() == 0) {
            c.line("scf.for " + iv + "=" + lb + " to " + ub + " step " + st +
                   " {");
          } else {
            SmallVector<std::string> carries, tys;
            for (size_t i = 0; i < iterTys.size(); ++i) {
              carries.push_back(irId(bArgs[1 + i]) + "=" +
                                irId(view.operand(3 + i)));
              tys.push_back(cast<StringAttr>(iterTys[i]).str());
            }
            if (view.numResults() == 1) {
              c.line(irId(view.result(0)) + "=scf.for " + iv + "=" + lb +
                     " to " + ub + " step " + st + " iter_args(" +
                     llvm::join(carries, ",") + ")->(" + llvm::join(tys, ",") +
                     ") {");
            } else {
              std::string tup = irId(view.result(0)) + "_t";
              c.line(tup + ":" + std::to_string(view.numResults()) +
                     "=scf.for " + iv + "=" + lb + " to " + ub + " step " + st +
                     " iter_args(" + llvm::join(carries, ",") + ")->(" +
                     llvm::join(tys, ",") + ") {");
            }
          }
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
          if (view.numResults() > 1) {
            std::string tup = irId(view.result(0)) + "_t";
            emitTupleUnpack(c, view, tup, iterTys);
          }
        }
        return llvm::Error::success();
      });

  rules["weft_host.task_if"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        auto resTys = view.attrOfType<ArrayAttr>("res_types");
        SmallVector<std::string> tys;
        for (Attribute a : resTys)
          tys.push_back(cast<StringAttr>(a).str());
        bool hasElse = view.numRegions() > 1 && view.numBlocks(1) > 0;
        if (at == PrintPoint::Enter) {
          if (view.numResults() == 0) {
            c.line("scf.if " + irId(view.operand(0)) + " {");
          } else if (view.numResults() == 1) {
            c.line(irId(view.result(0)) + "=scf.if " + irId(view.operand(0)) +
                   " -> (" + llvm::join(tys, ",") + ") {");
          } else {
            std::string tup = irId(view.result(0)) + "_t";
            c.line(tup + ":" + std::to_string(view.numResults()) + "=scf.if " +
                   irId(view.operand(0)) + " -> (" + llvm::join(tys, ",") +
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
            if (view.numResults() > 1) {
              std::string tup = irId(view.result(0)) + "_t";
              emitTupleUnpack(c, view, tup, resTys);
            }
          }
        }
        return llvm::Error::success();
      });

  enterLocal("weft_host.task_yield",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (view.numOperands() == 0)
                 return llvm::Error::success();
               SmallVector<std::string> vals, tys;
               for (unsigned i = 0; i < view.numOperands(); ++i)
                 vals.push_back(irId(view.operand(i)));
               for (Attribute a : view.attrOfType<ArrayAttr>("val_types"))
                 tys.push_back(cast<StringAttr>(a).str());
               c.line("scf.yield " + llvm::join(vals, ",") + ":" +
                      llvm::join(tys, ","));
               return llvm::Error::success();
             });

  rules["weft_host.task_subquant_level"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        auto domNumAttr = view.attrOfType<IntegerAttr>("domain_num");
        int64_t domNum = domNumAttr ? domNumAttr.getInt() : 1;
        std::string spec = "\"subquant." + std::to_string(domNum) + "\"," +
                           std::to_string(domNum) + ",0,3,\"blocks\",\"exact\"";
        std::string domTy = "!weft_kernel.domain<" + spec + ">";
        std::string ptTy = "!weft_kernel.point<<" + spec + ">>";
        std::string base = irId(view.result(0));
        unsigned nCarry = view.numResults();
        if (at == PrintPoint::BeginBlock) {
          auto bArgs = view.blockArguments();
          c.line(base + "_e=arith.constant 256:index");
          c.line(base + "_s=arith.constant 32:index");
          c.line(base + "_c=arith.ceildivui " + base + "_e," + base + "_s:index");
          c.line(base + "_d=\"weft_kernel.domain\"(" + irId(view.operand(0)) +
                 "," + base + "_e," + base + "_s," + base +
                 "_c):(!weft_kernel.domain<\"root\",0,-1,0,\"root\",\"exact\">,"
                 "index,index,index)->" +
                 domTy);
          SmallVector<std::string, 4> inOps, blkHdr{irId(bArgs[0]) + ":" + ptTy};
          for (unsigned i = 0; i < nCarry; ++i) {
            inOps.push_back(irId(view.operand(1 + i)));
            blkHdr.push_back(irId(bArgs[1 + i]) + ":si32");
          }
          std::string resPack =
              nCarry > 1 ? (base + "_p:" + std::to_string(nCarry))
                         : (base + "_p");
          c.line(resPack + "=\"weft_kernel.level\"(" + base + "_d," +
                 llvm::join(inOps, ",") + ") ({^bb0(%b0:" + ptTy +
                 "):\n\"weft_kernel.births_yield\"():()->()\n},"
                 "{^bb0(%b1:" +
                 ptTy + "):\n\"weft_kernel.births_yield\"():()->()\n},"
                 "{^bb0(" +
                 llvm::join(blkHdr, ",") + "):");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          SmallVector<std::string, 5> inTys{domTy}, outTys;
          for (unsigned i = 0; i < nCarry; ++i) {
            inTys.push_back("si32");
            outTys.push_back("si32");
          }
          std::string outSig =
              nCarry > 1 ? ("(" + llvm::join(outTys, ",") + ")") : "si32";
          c.line("}):(" + llvm::join(inTys, ",") + ")->" + outSig);
          c.line(base + "_z=\"weft_kernel.constant\"() <{value=0:si32}>:()->si32");
          for (unsigned i = 0; i < nCarry; ++i) {
            std::string src =
                nCarry > 1 ? (base + "_p#" + std::to_string(i)) : (base + "_p");
            c.line(irId(view.result(i)) + "=\"weft_kernel.binary\"(" + src +
                   "," + base + "_z) <{kind=\"add\"}>:(si32,si32)->si32");
          }
        }
        return llvm::Error::success();
      });

  // ---------------- Region 1 ("host.c") Local Ops ----------------
  enterLocal("arith.constant",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (auto fa = view.attrOfType<FloatAttr>("value")) {
                 llvm::SmallString<24> num;
                 fa.getValue().toString(num);
                 std::string s = num.str().str();
                 if (s.find_first_of(".Ee") == std::string::npos)
                   s += ".0";
                 c.line("const float " + view.result(0).str() + " = " + s + "f;");
               } else {
                 int64_t val = view.attrOfType<IntegerAttr>("value").getInt();
                 c.line("const int64_t " + view.result(0).str() + " = " +
                        std::to_string(val) + "LL;");
               }
               return llvm::Error::success();
             });

  enterLocal("weft_host.address",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto strides = view.attrOfType<DenseI64ArrayAttr>("strides");
               int32_t bits =
                   view.attrOfType<IntegerAttr>("index_bits").getInt();
               std::string cTy = bits == 32 ? "int32_t" : "int64_t";
               SmallVector<std::string> terms{"(" + cTy + ")0"};
               for (auto [vName, st] :
                    llvm::zip(view.operands(), strides.asArrayRef())) {
                 std::string stLit =
                     bits == 32 ? ("(" + cTy + ")" + std::to_string(st))
                                : (std::to_string(st) + "LL");
                 terms.push_back("(" + cTy + ")(" + vName + ") * " + stLit);
               }
               c.line("const " + cTy + " " + view.result(0).str() + " = " +
                      llvm::join(terms, " + ") + ";");
               return llvm::Error::success();
             });

  rules["weft_host.host_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("#include <math.h>");
          c.line("#include <stddef.h>");
          c.line("#include <stdint.h>");
          c.line("#include <stdlib.h>");
          c.line("#include <string.h>");
          c.line("#if defined(__x86_64__)");
          c.line("#include <immintrin.h>");
          c.line("#elif defined(__riscv_vector)");
          c.line("#include <riscv_vector.h>");
          c.line("#endif");
          c.line("#if defined(_OPENMP)");
          c.line("#include <omp.h>");
          c.line("#endif");
          c.line("#include \"task_bridge.h\"");
          c.line("static inline int64_t intent_omp_tid(void) {");
          c.line("#if defined(_OPENMP)");
          c.line("  return (int64_t)omp_get_thread_num();");
          c.line("#else");
          c.line("  return 0;");
          c.line("#endif");
          c.line("}");
          c.line("static inline int64_t intent_omp_nth(void) {");
          c.line("#if defined(_OPENMP)");
          c.line("  return (int64_t)omp_get_num_threads();");
          c.line("#else");
          c.line("  return 1;");
          c.line("#endif");
          c.line("}");
          c.line("__attribute__((weak)) void weft_kernel_release_hw_context(void);");

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

          if (view.attrOfType<BoolAttr>("uses_q8k").getValue()) {
            intentdsl::cpu::emitPackQ8KHelperC(c);
          }
          if (view.attrOfType<BoolAttr>("uses_bf16_math").getValue() ||
              view.attrOfType<BoolAttr>("uses_f16_math").getValue()) {
            c.line("#if defined(__x86_64__)");
            c.line("__attribute__((target(\"avx2\")))");
            c.line("static inline void transpose8x8_ps(__m256 r0, __m256 r1, __m256 r2, __m256 r3, __m256 r4, __m256 r5, __m256 r6, __m256 r7, float *out, int64_t ldo){");
            c.line("  __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);");
            c.line("  __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);");
            c.line("  __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);");
            c.line("  __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);");
            c.line("  __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE);");
            c.line("  __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);");
            c.line("  __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE);");
            c.line("  __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);");
            c.line("  _mm256_storeu_ps(out + 0*ldo, _mm256_permute2f128_ps(s0, s4, 0x20));");
            c.line("  _mm256_storeu_ps(out + 1*ldo, _mm256_permute2f128_ps(s1, s5, 0x20));");
            c.line("  _mm256_storeu_ps(out + 2*ldo, _mm256_permute2f128_ps(s2, s6, 0x20));");
            c.line("  _mm256_storeu_ps(out + 3*ldo, _mm256_permute2f128_ps(s3, s7, 0x20));");
            c.line("  _mm256_storeu_ps(out + 4*ldo, _mm256_permute2f128_ps(s0, s4, 0x31));");
            c.line("  _mm256_storeu_ps(out + 5*ldo, _mm256_permute2f128_ps(s1, s5, 0x31));");
            c.line("  _mm256_storeu_ps(out + 6*ldo, _mm256_permute2f128_ps(s2, s6, 0x31));");
            c.line("  _mm256_storeu_ps(out + 7*ldo, _mm256_permute2f128_ps(s3, s7, 0x31));");
            c.line("}");
            c.line("#endif");
          }
          if (view.attrOfType<BoolAttr>("uses_bf16_math").getValue()) {
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
            c.line("#if defined(__x86_64__)");
            c.line("__attribute__((target(\"avx2\")))");
            c.line("static void widen_transpose_bf16_avx2(float *out, const uint16_t *src, int64_t K, int64_t valid, int64_t nr, int64_t ldn, int64_t *jb_out){");
            c.line("  int64_t jb = 0;");
            c.line("  for (; jb + 8 <= valid; jb += 8) {");
            c.line("    const uint16_t *s0 = src + (jb+0)*ldn, *s1 = src + (jb+1)*ldn, *s2 = src + (jb+2)*ldn, *s3 = src + (jb+3)*ldn;");
            c.line("    const uint16_t *s4 = src + (jb+4)*ldn, *s5 = src + (jb+5)*ldn, *s6 = src + (jb+6)*ldn, *s7 = src + (jb+7)*ldn;");
            c.line("    int64_t kb = 0;");
            c.line("    for (; kb <= K - 8; kb += 8) {");
            c.line("      __m256 r0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s0 + kb))), 16));");
            c.line("      __m256 r1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s1 + kb))), 16));");
            c.line("      __m256 r2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s2 + kb))), 16));");
            c.line("      __m256 r3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s3 + kb))), 16));");
            c.line("      __m256 r4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s4 + kb))), 16));");
            c.line("      __m256 r5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s5 + kb))), 16));");
            c.line("      __m256 r6 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s6 + kb))), 16));");
            c.line("      __m256 r7 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s7 + kb))), 16));");
            c.line("      transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7, out + (size_t)kb * (size_t)nr + (size_t)jb, nr);");
            c.line("    }");
            c.line("    for (; kb < K; ++kb) { float *dst = out + (size_t)kb * (size_t)nr + (size_t)jb; dst[0]=expand_bf(s0[kb]); dst[1]=expand_bf(s1[kb]); dst[2]=expand_bf(s2[kb]); dst[3]=expand_bf(s3[kb]); dst[4]=expand_bf(s4[kb]); dst[5]=expand_bf(s5[kb]); dst[6]=expand_bf(s6[kb]); dst[7]=expand_bf(s7[kb]); }");
            c.line("  }");
            c.line("  *jb_out = jb;");
            c.line("}");
            c.line("#endif");
            c.line("static inline void widen_transpose_bf16_panel(float *out, const uint16_t *src, int64_t K, int64_t valid, int64_t nr, int64_t ldn){");
            c.line("  int64_t jb = 0;");
            c.line("#if defined(__x86_64__)");
            c.line("  if (__builtin_cpu_supports(\"avx2\")) widen_transpose_bf16_avx2(out, src, K, valid, nr, ldn, &jb);");
            c.line("#endif");
            c.line("  if (jb < valid || valid < nr) { for (int64_t k = 0; k < K; ++k) { float *dst = out + (size_t)k * (size_t)nr; for (int64_t j = jb; j < valid; ++j) dst[j] = expand_bf(src[j*ldn + k]); if (valid < nr) memset(dst + valid, 0, (size_t)(nr - valid) * sizeof(float)); } }");
            c.line("}");
          }
          if (view.attrOfType<BoolAttr>("uses_f16_math").getValue()) {
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
            c.line("#if defined(__x86_64__)");
            c.line("__attribute__((target(\"avx512f,f16c\")))");
            c.line("static void widen_f16_span_avx512(float *out, const uint16_t *src, int64_t n){");
            c.line("  int64_t i = 0;");
            c.line("  for (; i <= n - 16; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_cvtph_ps(h)); }");
            c.line("  for (; i < n; ++i) out[i] = _cvtsh_ss(src[i]);");
            c.line("}");
            c.line("__attribute__((target(\"avx2,f16c\")))");
            c.line("static void widen_transpose_f16_avx2(float *out, const uint16_t *src, int64_t K, int64_t valid, int64_t nr, int64_t ldn, int64_t *jb_out){");
            c.line("  int64_t jb = 0;");
            c.line("  for (; jb + 8 <= valid; jb += 8) {");
            c.line("    const uint16_t *s0 = src + (jb+0)*ldn, *s1 = src + (jb+1)*ldn, *s2 = src + (jb+2)*ldn, *s3 = src + (jb+3)*ldn;");
            c.line("    const uint16_t *s4 = src + (jb+4)*ldn, *s5 = src + (jb+5)*ldn, *s6 = src + (jb+6)*ldn, *s7 = src + (jb+7)*ldn;");
            c.line("    int64_t kb = 0;");
            c.line("    for (; kb <= K - 8; kb += 8) {");
            c.line("      __m256 r0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s0 + kb)));");
            c.line("      __m256 r1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s1 + kb)));");
            c.line("      __m256 r2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s2 + kb)));");
            c.line("      __m256 r3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s3 + kb)));");
            c.line("      __m256 r4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s4 + kb)));");
            c.line("      __m256 r5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s5 + kb)));");
            c.line("      __m256 r6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s6 + kb)));");
            c.line("      __m256 r7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s7 + kb)));");
            c.line("      transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7, out + (size_t)kb * (size_t)nr + (size_t)jb, nr);");
            c.line("    }");
            c.line("    for (; kb < K; ++kb) { float *dst = out + (size_t)kb * (size_t)nr + (size_t)jb; dst[0]=_cvtsh_ss(s0[kb]); dst[1]=_cvtsh_ss(s1[kb]); dst[2]=_cvtsh_ss(s2[kb]); dst[3]=_cvtsh_ss(s3[kb]); dst[4]=_cvtsh_ss(s4[kb]); dst[5]=_cvtsh_ss(s5[kb]); dst[6]=_cvtsh_ss(s6[kb]); dst[7]=_cvtsh_ss(s7[kb]); }");
            c.line("  }");
            c.line("  *jb_out = jb;");
            c.line("}");
            c.line("#endif");
            c.line("static inline void widen_f16_span(float *out, const uint16_t *src, int64_t n){");
            c.line("  if (n <= 0) return;");
            c.line("#if defined(__x86_64__)");
            c.line("  if (__builtin_cpu_supports(\"avx512f\") && __builtin_cpu_supports(\"f16c\")) { widen_f16_span_avx512(out, src, n); return; }");
            c.line("#endif");
            c.line("  for (int64_t i=0; i<n; ++i) out[i] = expand_f16(src[i]);");
            c.line("}");
            c.line("static inline void widen_transpose_f16_panel(float *out, const uint16_t *src, int64_t K, int64_t valid, int64_t nr, int64_t ldn){");
            c.line("  int64_t jb = 0;");
            c.line("#if defined(__x86_64__)");
            c.line("  if (__builtin_cpu_supports(\"avx2\") && __builtin_cpu_supports(\"f16c\")) widen_transpose_f16_avx2(out, src, K, valid, nr, ldn, &jb);");
            c.line("#endif");
            c.line("  if (jb < valid || valid < nr) { for (int64_t k = 0; k < K; ++k) { float *dst = out + (size_t)k * (size_t)nr; for (int64_t j = jb; j < valid; ++j) dst[j] = expand_f16(src[j*ldn + k]); if (valid < nr) memset(dst + valid, 0, (size_t)(nr - valid) * sizeof(float)); } }");
            c.line("}");
          }
          if (view.attrOfType<BoolAttr>("uses_mx8_math").getValue()) {
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
          if (auto u16Tr = view.attrOfType<BoolAttr>("uses_u16_pack_transpose");
              u16Tr && u16Tr.getValue()) {
            intentdsl::cpu::emitPackTransposeU16HelperC(c);
          }
          SmallVector<const intentdsl::kir::QuantFormatSpec *, 4> qSpecs;
          if (auto qfs = view.attrOfType<ArrayAttr>("quant_formats")) {
            for (Attribute a : qfs)
              if (const auto *sp = intentdsl::kir::lookupQuantFormat(
                      cast<StringAttr>(a).getValue()))
                qSpecs.push_back(sp);
            intentdsl::cpu::emitQuantHostHelpersC(qSpecs, c);
          }

          auto ext = view.attrOfType<DenseI64ArrayAttr>("extents");
          auto tl = view.attrOfType<DenseI64ArrayAttr>("tile");
          c.line("static _Thread_local char g_intent_last_error[512];");
          c.line("const char *intent_last_error(void) { return g_intent_last_error; }");
          c.line("int32_t intent_entry(" +
                 view.attrOfType<StringAttr>("params").str() + ") {");
          c.indent++;
          c.line("const int64_t M=" + std::to_string(ext[0]) +
                 ", N=" + std::to_string(ext[1]) + ", K=" +
                 std::to_string(ext[2]) + ", G=" + std::to_string(ext[3]) + ";");
          c.line("const int64_t mr=" + std::to_string(tl[0]) +
                 ", nr=" + std::to_string(tl[1]) + ";");
          c.line("const int64_t mb=mr>0?(M+mr-1)/mr:0, nc=nr>0?(N+nr-1)/nr:0, jobs=G*mb*nc;");
          c.line("int32_t intent_status = 0;");
          c.line("char intent_err_buf[512]; intent_err_buf[0] = 0;");
          c.line("g_intent_last_error[0] = 0;");
          c.line("(void)rows; (void)columns; (void)grain; (void)panel;");
          if (view.attrOfType<BoolAttr>("uses_mx8_math").getValue())
            c.line("ensure_mx8_tables();");
          for (const auto *sp : qSpecs)
            c.line("ensure_quant_tables_" +
                   intentdsl::cpu::sanitizeQuantIdent(sp->name) + "();");
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
      });

  enterLocal("weft_host.host_alloc",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto name = view.attrOfType<StringAttr>("buf_name").str();
               auto place = view.attrOfType<StringAttr>("placement").getValue();
               auto cType = view.attrOfType<StringAttr>("c_type").str();
               auto count = view.attrOfType<StringAttr>("count_expr").str();
               int wsSlot = (name == "left" || name == "a_tile") ? 0
                            : (name == "rhs" || name == "b_grp") ? 1
                            : name == "members"                  ? 2
                                                                 : 3;
               if (place == "invocation") {
                 c.line(cType + " *" + name + " = (" + cType +
                        "*)malloc((size_t)(" + count + ") * sizeof(" + cType +
                        "));");
                 std::string cleanup;
                 for (auto prior : view.attrOfType<ArrayAttr>("cleanup"))
                   cleanup += "free(" + cast<StringAttr>(prior).str() + "); ";
                 c.line("if (!" + name + ") { " + cleanup + "return 2; }");
               } else if (place == "stack") {
                 c.line("_Alignas(64) " + cType + " " + name + "[" + count +
                        "];");
               } else if (place == "tls") {
                 c.line(cType + " *" + name + " = (" + cType +
                        "*)intent_ws_acquire(" + std::to_string(wsSlot) +
                        ", (size_t)(" + count + ") * sizeof(" + cType + "));");
                 c.line("if (!" + name + ") { int32_t _ws_e = 0; __atomic_compare_exchange_n(&intent_status, &_ws_e, 2, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED); }");
               } else {
                 std::string frees;
                 for (Attribute a : view.attrOfType<ArrayAttr>("cleanup")) {
                   std::string prev = cast<StringAttr>(a).str();
                   int pSlot = prev == "left"      ? 0
                               : prev == "rhs"     ? 1
                               : prev == "members" ? 2
                                                   : 3;
                   frees += "intent_ws_release(" + std::to_string(pSlot) +
                            ", " + prev + ");";
                 }
                 c.line(cType + " *" + name + " = (" + cType +
                        "*)intent_ws_acquire(" + std::to_string(wsSlot) +
                        ", (size_t)(" + count + ") * sizeof(" + cType + "));");
                 c.line("if (!" + name + ") { " + frees + "return 2; }");
               }
               return llvm::Error::success();
             });

  rules["weft_host.host_omp_parallel"] = PrintRule::local(
      [substVals](const LocalOpView &view, PrintPoint at,
                  PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          std::string nt = view.attrOfType<StringAttr>("num_threads").str();
          std::string ic = substVals(
              view.attrOfType<StringAttr>("if_cond").getValue(), view);
          std::string pragma = "#pragma omp parallel";
          if (!nt.empty())
            pragma += " num_threads(" + nt + ")";
          if (!ic.empty())
            pragma += " if(" + ic + ")";
          c.line(pragma);
          c.line("{");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          if (view.attrOfType<BoolAttr>("release_hw_ctx").getValue())
            c.line("if (weft_kernel_release_hw_context) weft_kernel_release_hw_context();");
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["weft_host.host_omp_for"] = PrintRule::local(
      [substVals](const LocalOpView &view, PrintPoint at,
                  PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          bool par = view.attrOfType<BoolAttr>("parallel").getValue();
          bool simd = view.attrOfType<BoolAttr>("simd").getValue();
          bool x86Simd = view.attrOfType<BoolAttr>("x86_simd").getValue();
          auto nwAttr = view.attrOfType<BoolAttr>("nowait");
          bool nowait = nwAttr && nwAttr.getValue();
          std::string sched = view.attrOfType<StringAttr>("schedule").str();
          std::string nt = view.attrOfType<StringAttr>("num_threads").str();
          std::string ic = substVals(
              view.attrOfType<StringAttr>("if_cond").getValue(), view);
          std::string iv = view.attrOfType<StringAttr>("iv").str();
          std::string lb =
              substVals(view.attrOfType<StringAttr>("lb").getValue(), view);
          std::string ub =
              substVals(view.attrOfType<StringAttr>("ub").getValue(), view);
          std::string st =
              substVals(view.attrOfType<StringAttr>("step").getValue(), view);
          if (simd) {
            if (x86Simd) {
              c.line("#if defined(__x86_64__)");
              c.line("#pragma omp simd");
              c.line("#endif");
            } else {
              c.line("#pragma omp simd");
            }
          } else if (par) {
            std::string pragma = "#pragma omp parallel for";
            if (!nt.empty())
              pragma += " num_threads(" + nt + ")";
            if (!sched.empty())
              pragma += " schedule(" + sched + ")";
            if (!ic.empty())
              pragma += " if(" + ic + ")";
            c.line(pragma);
          } else {
            std::string pragma =
                "#pragma omp for schedule(" +
                (sched.empty() ? std::string("static") : sched) + ")";
            if (nowait)
              pragma += " nowait";
            c.line(pragma);
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
      });

  enterLocal("weft_host.host_load_scalar",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t s = view.attrOfType<IntegerAttr>("slot").getInt();
               std::string cTy = view.attrOfType<StringAttr>("c_type").str();
               std::string idx =
                   view.attrOfType<StringAttr>("index_expr").str();
               std::string base = s >= 0 ? ("a" + std::to_string(s))
                                         : ("wk_" + std::to_string(-1 - s));
               c.line("const " + cTy + " " + view.result(0).str() +
                      " = ((const " + cTy + "*)" + base + ")[" + idx + "];");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_span_op",
             [substVals](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto kind = view.attrOfType<StringAttr>("op").getValue();
               std::string dst = substVals(
                   view.attrOfType<StringAttr>("dst_expr").getValue(), view);
               std::string src = substVals(
                   view.attrOfType<StringAttr>("src_expr").getValue(), view);
               std::string cnt = substVals(
                   view.attrOfType<StringAttr>("count_expr").getValue(), view);
               int64_t eb = view.attrOfType<IntegerAttr>("elem_bytes").getInt();
               if (kind == "widen_bf16")
                 c.line("widen_bf16_span(" + dst + ", " + src + ", " + cnt +
                        ");");
               else if (kind == "widen_f16")
                 c.line("widen_f16_span(" + dst + ", " + src + ", " + cnt +
                        ");");
               else if (kind == "widen_transpose_bf16")
                 c.line("widen_transpose_bf16_panel(" + dst + ", " + src +
                        ", K, valid, nr, " + cnt + ");");
               else if (kind == "widen_transpose_f16")
                 c.line("widen_transpose_f16_panel(" + dst + ", " + src +
                        ", K, valid, nr, " + cnt + ");");
               else if (kind == "pack_transpose_u16")
                 c.line("pack_transpose_u16_panel(" + dst + ", " + src +
                        ", K, valid, nr, " + cnt + ");");
               else if (kind == "decode_quant_block_lhs") {
                 auto id = intentdsl::cpu::sanitizeQuantIdent(
                     view.attrOfType<StringAttr>("format").getValue());
                 c.line("intent_decode_block_lhs_row_" + id + "(" + dst +
                        ", " + src + ", " + cnt + ");");
               } else if (kind == "decode_quant_block_rhs") {
                 auto id = intentdsl::cpu::sanitizeQuantIdent(
                     view.attrOfType<StringAttr>("format").getValue());
                 c.line("intent_decode_block_rhs_panel_" + id + "(" + dst +
                        ", " + src + ", " + cnt + ");");
               } else if (kind == "decode_quant_grouped_rhs") {
                 auto id = intentdsl::cpu::sanitizeQuantIdent(
                     view.attrOfType<StringAttr>("format").getValue());
                 c.line("intent_decode_grouped_rhs_panel_" + id + "(" + dst +
                        ", " + src + ", " + cnt + ");");
               } else if (kind == "memcpy")
                 c.line("memcpy(" + dst + ", " + src + ", (size_t)(" + cnt +
                        ") * " + std::to_string(eb) + "ULL);");
               else if (kind == "zero")
                 c.line("memset(" + dst + ", 0, (size_t)(" + cnt + ") * " +
                        std::to_string(eb) + "ULL);");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_elem_op",
             [substVals](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto kind = view.attrOfType<StringAttr>("op").getValue();
               std::string dst = substVals(
                   view.attrOfType<StringAttr>("dst_expr").getValue(), view);
               std::string src = substVals(
                   view.attrOfType<StringAttr>("src_expr").getValue(), view);
               std::string aux = substVals(
                   view.attrOfType<StringAttr>("aux_expr").getValue(), view);
               int64_t eb = view.attrOfType<IntegerAttr>("elem_bytes").getInt();
               if (kind == "expand_bf")
                 c.line(dst + " = expand_bf(" + src + ");");
               else if (kind == "expand_f16")
                 c.line(dst + " = expand_f16(" + src + ");");
               else if (kind == "assign")
                 c.line(dst + " = " + src + ";");
               else if (kind == "memcpy")
                 c.line("memcpy(" + dst + ", " + src + ", " +
                        std::to_string(eb) + ");");
               else if (kind == "pack_q8k")
                 c.line("pack_q8_record(" + dst + ", " + src + ", " + aux +
                        ");");
               else if (kind == "let_mx8_scale")
                 c.line("const float " + dst + " = g_mx8_scale[" + src + "];");
               else if (kind == "mx8_scale")
                 c.line(dst + " = g_mx8_scale[" + src + "];");
               else if (kind == "mx8_mul")
                 c.line(dst + " = g_mx8_val[" + src + "] * " + aux + ";");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_panel_yield",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               bool wide = view.attrOfType<BoolAttr>("widen_bf16").getValue();
               std::string dst = view.attrOfType<StringAttr>("dst_expr").str();
               std::string v = view.operand(0).str();
               c.line(dst + " = " + (wide ? "expand_bf(" + v + ")" : v) + ";");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_view_ptr",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t idx = view.attrOfType<IntegerAttr>("index").getInt();
               auto ptrTy = view.attrOfType<StringAttr>("c_ptr_type").str();
               auto base = view.attrOfType<StringAttr>("base_expr").str();
               int64_t byteOff =
                   view.attrOfType<IntegerAttr>("byte_offset").getInt();
               if (view.attr("routed_temp")) {
                 c.line("_Alignas(64) uint8_t routed_out[" +
                        std::to_string(byteOff) + "]; " + ptrTy + "p" +
                        std::to_string(idx) + " = (" + ptrTy + ")routed_out;");
                 return llvm::Error::success();
               }
               std::string rawBase =
                   byteOff == 0
                       ? ("(" + ptrTy + ")(" + base + ")")
                       : ("(" + ptrTy + ")((const char*)(" + base + ") + " +
                          std::to_string(byteOff) + "LL)");
               if (view.numOperands() > 0)
                 c.line(ptrTy + "p" + std::to_string(idx) + " = " + rawBase +
                        " + (ptrdiff_t)" + view.operand(0).str() + ";");
               else
                 c.line(ptrTy + "p" + std::to_string(idx) + " = " + rawBase +
                        ";");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_call_task",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t arity = view.attrOfType<IntegerAttr>("arity").getInt();
               SmallVector<std::string> args;
               for (int64_t i = 0; i < arity; ++i)
                 args.push_back("p" + std::to_string(i));
               args.push_back("(size_t)" + view.operand(0).str());
               args.push_back("(size_t)" + view.operand(1).str());
               for (int64_t st :
                    view.attrOfType<DenseI64ArrayAttr>("strides").asArrayRef())
                 args.push_back("(size_t)" + std::to_string(st) + "ULL");
               c.line("matrix_panel_instance(" + llvm::join(args, ", ") + ");");
               c.line("{ char task_err[512]; int32_t task_st = (int32_t)weft_kernel_take_status(task_err, sizeof(task_err));");
               c.line("  if (task_st != 0) { int32_t expected_st = 0;");
               c.line("    if (__atomic_compare_exchange_n(&intent_status, &expected_st, task_st, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {");
               c.line("      memcpy(intent_err_buf, task_err, sizeof(intent_err_buf)); intent_err_buf[sizeof(intent_err_buf) - 1] = 0; }");
               c.line("    continue; } }");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_free",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               std::string name = view.attrOfType<StringAttr>("buf_name").str();
               if (view.attr("invocation")) {
                 c.line("free(" + name + ");");
                 return llvm::Error::success();
               }
               int wsSlot = (name == "left" || name == "a_tile") ? 0
                            : (name == "rhs" || name == "b_grp") ? 1
                            : name == "members"                  ? 2
                                                                 : 3;
               c.line("intent_ws_release(" + std::to_string(wsSlot) + ", " +
                      name + ");");
               return llvm::Error::success();
             });

  rules["weft_host.host_row_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("#include <stddef.h>");
          c.line("#include <stdint.h>");
          c.line("#include \"task_bridge.h\"");
          c.line("int32_t intent_entry(" +
                 view.attrOfType<StringAttr>("params").str() + ") {");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.line("return 0;");
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  enterLocal("weft_host.host_row_offset",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mode = view.attrOfType<StringAttr>("mode").getValue();
               auto dims = view.attrOfType<ArrayAttr>("dim_vars");
               auto strides = view.attrOfType<ArrayAttr>("stride_vars");
               std::string outN = view.result(0).str();
               if (mode == "linear") {
                 c.line("const int64_t " + outN + " = (" +
                        view.operand(0).str() + ") * " +
                        cast<StringAttr>(strides[0]).str() + ";");
               } else if (mode == "tile2d") {
                 c.line("const int64_t " + outN + " = (" +
                        view.operand(0).str() + ") * " +
                        cast<StringAttr>(strides[0]).str() + " + (" +
                        view.operand(1).str() + ") * " +
                        cast<StringAttr>(strides[1]).str() + ";");
               } else {
                 std::string row = view.operand(0).str();
                 SmallVector<std::string> terms{"0LL"};
                 for (int j = int(dims.size()) - 1; j >= 0; --j) {
                   std::string dm = cast<StringAttr>(dims[j]).str();
                   std::string st = cast<StringAttr>(strides[j]).str();
                   terms.push_back("((" + row + ") % " + dm + ") * " + st);
                   row = "(" + row + " / " + dm + ")";
                 }
                 c.line("const int64_t " + outN + " = " +
                        llvm::join(terms, " + ") + ";");
               }
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_call_row_task",
             [substVals](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               SmallVector<std::string> args;
               for (Attribute a : view.attrOfType<ArrayAttr>("slot_args"))
                 args.push_back(cast<StringAttr>(a).str());
               for (Attribute a : view.attrOfType<ArrayAttr>("shape_exprs"))
                 args.push_back(
                     "(size_t)(" +
                     substVals(cast<StringAttr>(a).getValue(), view) + ")");
               for (Attribute a : view.attrOfType<ArrayAttr>("stride_exprs"))
                 args.push_back("(size_t)(" + cast<StringAttr>(a).str() + ")");
               c.line(view.attrOfType<StringAttr>("sym_name").str() +
                      "_instance(" + llvm::join(args, ", ") + ");");
               return llvm::Error::success();
             });

  enterLocal("weft_host.host_collect_task_status",
             [](const LocalOpView &, PrintContext &c) -> llvm::Error {
               c.line("char message[512];");
               c.line("int32_t status = weft_kernel_take_status(message, sizeof(message));");
               c.line("int32_t expected = 0;");
               c.line("if (status && __atomic_compare_exchange_n(&intent_status, &expected, status, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))");
               c.line("  memcpy(intent_error, message, sizeof(intent_error));");
               return llvm::Error::success();
             });
  rules["weft_host.host_owner_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        bool checked = bool(view.attr("checked_tasks"));
        if (at == PrintPoint::Enter) {
          c.line("#include <stdlib.h>");
          c.line("#include <stddef.h>");
          c.line("#include <stdint.h>");
          c.line("#include \"task_bridge.h\"");
          if (checked) {
            c.line("#include <string.h>");
            c.line("void weft_kernel_release_hw_context(void);");
            c.line("static _Thread_local char g_intent_last_error[512];");
            c.line("const char *intent_last_error(void) { return g_intent_last_error; }");
          }
          c.line("int32_t intent_entry(" +
                 view.attrOfType<StringAttr>("params").str() + ") {");
          c.indent++;
          if (checked) {
            c.line("int32_t intent_status = 0;");
            c.line("char intent_error[512] = {0};");
          }
          c.line("const int64_t count = " +
                 std::to_string(view.attrOfType<IntegerAttr>("owners").getInt()) +
                 "LL, tile = " +
                 std::to_string(view.attrOfType<IntegerAttr>("grain").getInt()) +
                 "LL;");
          c.line("(void)rows; (void)columns; (void)grain; (void)panel;");
          c.indent--;
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          if (checked) {
            c.line("memcpy(g_intent_last_error, intent_error, sizeof(intent_error));");
            c.line("return intent_status;");
          } else
            c.line("return 0;");
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  enterLocal("weft_host.host_call_owner_task",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               SmallVector<std::string> args;
               for (Attribute a : view.attrOfType<ArrayAttr>("slot_args"))
                 args.push_back(cast<StringAttr>(a).str());
               args.push_back("(size_t)(" + view.operand(0).str() + ")");
               args.push_back("(size_t)(" + view.operand(1).str() + ")");
               if (view.attrOfType<BoolAttr>("empty_extent").getValue())
                 args.push_back("(size_t)0ULL");
               for (Attribute a : view.attrOfType<ArrayAttr>("stride_exprs"))
                 args.push_back("(size_t)(" + cast<StringAttr>(a).str() + ")");
               c.line(view.attrOfType<StringAttr>("sym_name").str() +
                      "_instance(" + llvm::join(args, ", ") + ");");
               return llvm::Error::success();
             });

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
