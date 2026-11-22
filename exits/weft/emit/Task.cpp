#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
#include "intentdsl/levels/execution/cpu/ir/Pack.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
std::string printType(Type t) {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << t;
  return s;
}
std::string regionType(StringRef kind, StringRef element, int rank,
                       int columnDim = -2) {
  std::string shape =
      rank == 2 ? "[-1,-2],[1,2]" : "[" + std::to_string(columnDim) + "],[2]";
  if (kind == "value")
    return "!weft_kernel.value<" + element.str() + "," + shape + ">";
  std::string family=element=="si64"?"i64":element=="si32"?"i32":element.str();
  return "!weft_kernel." + kind.str() + "<!weft_kernel.encoding<\"" +
         family + "\",\"dense\",\"dense." + family + "\",[]>," +
         shape + ">";
}
struct Term {
  std::string id, type;
};
class CanonicalTask {
  llvm::raw_ostream &s;
  llvm::DenseMap<Value, Term> terms;
  unsigned serial = 0;
  Term origin, columns;
  Term indexLiteral(int64_t value) {
    Term result{"%index"+std::to_string(serial++),"index"};
    s<<result.id<<"=arith.constant "<<value<<" : index\n";
    return result;
  }
  std::optional<int64_t> ownerAxis;
  SmallVector<Term> extents;
  llvm::DenseMap<Value,SmallVector<Term>> records;
  std::string spelling(Type t) {
    return t.isInteger(32) ? "si32" : t.isSignlessInteger(64) ? "si64" : printType(t);
  }

  Term inst(StringRef op, ArrayRef<Term> args, std::string type,
            std::string props = "") {
    std::string id = "%v" + std::to_string(serial++);
    if (type != "()")
      s << id << " = ";
    s << "\"weft_kernel." << op << "\"(";
    llvm::interleaveComma(args, s, [&](auto &a) { s << a.id; });
    s << ") ";
    if (!props.empty())
      s << "<{" << props << "}> ";
    s << ": (";
    llvm::interleaveComma(args, s, [&](auto &a) { s << a.type; });
    s << ") -> " << type << "\n";
    return {id, type};
  }

  Term uniform(Term seed, Term position, int dim) {
    const auto vector = regionType("value", "ui64", 1, dim);
    auto constant = [&](uint64_t value) {
      return inst("constant", {}, "ui64", "value="+std::to_string(value)+" : ui64");
    };
    auto binary = [&](StringRef kind, Term lhs, Term rhs) {
      return inst("binary", {lhs,rhs}, lhs.type, "kind=\""+kind.str()+"\"");
    };
    auto mask=constant(UINT32_MAX), bits=constant(32), zero=constant(0);
    auto base=binary("shr",position,constant(2));
    auto a=binary("and",base,mask), b=binary("shr",base,bits);
    auto c=binary("and",base,zero), d=c;
    auto lo=binary("and",seed,mask), hi=binary("shr",seed,bits);
    const auto multiplierA=constant(0xd2511f53), multiplierB=constant(0xcd9e8d57);
    const auto incrementA=constant(0x9e3779b9), incrementB=constant(0xbb67ae85);
    for(unsigned round=0; round<10; ++round) {
      auto left=binary("mul",a,multiplierA), right=binary("mul",c,multiplierB);
      a=binary("xor",binary("xor",binary("shr",right,bits),b),lo);
      c=binary("xor",binary("xor",binary("shr",left,bits),d),hi);
      b=binary("and",right,mask);
      d=binary("and",left,mask);
      lo=binary("and",binary("add",lo,incrementA),mask);
      hi=binary("and",binary("add",hi,incrementB),mask);
    }
    auto word=binary("and",position,constant(3));
    for(auto [index,value]: {std::pair{2,c},std::pair{1,b},std::pair{0,a}}){
      auto active=inst("compare",{word,constant(index)},regionType("value","i1",1,dim),"predicate=\"eq\"");
      d=inst("select",{active,value,d},vector);
    }
    auto f=inst("cast",{binary("shr",d,constant(8))},regionType("value","f32",1,dim));
    return binary("mul",f,inst("constant",{},"f32","value=0x33800000 : f32"));
  }

  // A projected task owns an interval of one tensor axis. Its scalar member
  // coordinate ranges over the product of all remaining extents.
  Term memberSlice(Operation &op, Term member, bool owned) {
    auto view=op.getOperand(0);
    auto rank=cast<MemRefType>(view.getType()).getRank();
    SmallVector<Term> coords(rank);
    for(int i=rank-1;i>=0;--i){
      if(!ownerAxis){coords[i]=i==rank-1?member:Term{"%r","index"};continue;}
      if(owned || i==*ownerAxis){coords[i]={"%r","index"};continue;}
      auto size=extents[i];
      coords[i]=inst("binary",{member,size},"index","kind=\"mod\"");
      member=inst("binary",{member,size},"index","kind=\"div\"");
    }
    SmallVector<Term> args{terms.lookup(view)};
    args.append(coords);
    SmallVector<std::string> selectors(rank,"\"index\"");
    auto elem=printType(cast<MemRefType>(view.getType()).getElementType());
    return inst("slice",args,"!weft_kernel.slice<!weft_kernel.encoding<\""+elem+"\",\"dense\",\"dense."+elem+"\",[]>,[],[]>",
                "selectors=["+llvm::join(selectors,",")+"]");
  }

  llvm::Error ownerSweep(Operation &sweep) {
    auto kind=sweep.getAttrOfType<StringAttr>("combine");
    bool publish=kind=="publish", moments=kind=="welford";
    bool scalarReduction=kind=="sum" || kind=="max_num";
    if(!publish && !moments && !scalarReduction && kind!="store")
      return llvm::createStringError("unsupported member sweep algebra");
    std::string prefix="%owner"+std::to_string(serial++);
    Term index{prefix+"_i","index"};
    SmallVector<Term> carries;
    auto &body=sweep.getRegion(0).front();
    for(auto &op:body)if(op.getName().stripDialect()=="scan_sum"){
      auto t=spelling(cast<ShapedType>(op.getResult(0).getType()).getElementType());
      carries.push_back(inst("constant",{},t,"value="+std::string(t=="f32"?"0.0":"0")+" : "+t));
    }
    if(!carries.empty() && kind!="store")return llvm::createStringError("prefix sweep requires a store consumer");
    if(!publish) {
      if(moments) s<<prefix<<":3 = ";
      else if(scalarReduction)s<<prefix<<" = ";
      else if(!carries.empty())s<<prefix<<":"<<carries.size()<<" = ";
      s<<"scf.for "<<index.id<<" = %lo to "<<columns.id<<" step %step";
      if(moments)s<<" iter_args("<<prefix<<"_n = %zero_count, "<<prefix<<"_m = %zero_f, "<<prefix<<"_s = %zero_f) -> (si32,f32,f32)";
      if(scalarReduction)s<<" iter_args("<<prefix<<"_acc = "<<(kind=="sum"?"%zero_f":"%minus_inf")<<") -> (f32)";
      if(!carries.empty()){
        SmallVector<std::string> bindings,types;
        for(auto [i,c]:llvm::enumerate(carries)){bindings.push_back(prefix+"_p"+std::to_string(i)+"="+c.id);types.push_back(c.type);}
        s<<" iter_args("<<llvm::join(bindings,",")<<") -> ("<<llvm::join(types,",")<<")";
      }
      s<<" {\n";
    }
    unsigned cursor=0;
    for(auto &op:body.without_terminator()){
      auto n=op.getName().stripDialect();
      if(n=="write_panel" || n=="write_owner"){
        auto addr=memberSlice(op,index,n=="write_owner");
        inst("commit",{terms.lookup(op.getOperands().back()),addr},"()");
        continue;
      }
      auto v=op.getResult(0);
      Type t=v.getType();if(auto vector=dyn_cast<ShapedType>(t))t=vector.getElementType();
      auto ty=spelling(t);
      if(n=="read_panel" || n=="read_owner"){
        terms[v]=inst("admit",{memberSlice(op,index,n=="read_owner")},ty);
      } else if(n=="ordered_sum" || n=="segment_length"){
        auto *task=sweep.getParentOp();
        auto flat=cast<MemRefType>(task->getOperand(task->getAttrOfType<IntegerAttr>("domain_arg").getInt()).getType()).getRank()==1;
        Term owner=inst("binary",{origin,flat?index:Term{"%r","index"}},"index","kind=\"add\"");
        auto read=[&](Value source,ArrayRef<Term> coords){
          auto view=cast<MemRefType>(source.getType());auto elem=printType(view.getElementType());
          SmallVector<Term> args{terms.lookup(source)};
          args.append(coords.begin(),coords.end());
          SmallVector<std::string> selectors(coords.size(),"\"index\"");
          auto address=inst("slice",args,"!weft_kernel.slice<!weft_kernel.encoding<\""+elem+"\",\"dense\",\"dense."+elem+"\",[]>,[],[]>",
                            "selectors=["+llvm::join(selectors,",")+"]");
          return inst("admit",{address},spelling(view.getElementType()));
        };
        Term lo=inst("cast",{read(op.getOperand(0),{owner})},"index");
        Term next=inst("binary",{owner,{"%step","index"}},"index","kind=\"add\"");
        Term hi=inst("cast",{read(op.getOperand(0),{next})},"index");
        if(n=="segment_length"){
          terms[v]=inst("cast",{inst("binary",{hi,lo},"index","kind=\"sub\"")},"f32");
        } else {
          auto &bb=op.getRegion(0).front();
          std::string tag="%seg"+std::to_string(serial++);
          Term member{tag+"_i","index"},carry{tag+"_a","f32"};
          terms[bb.getArgument(0)]=member;terms[bb.getArgument(1)]=index;
          auto zero=inst("constant",{},"f32","value=0.0 : f32");
          s<<tag<<"=scf.for "<<member.id<<" = "<<lo.id<<" to "<<hi.id<<" step %step iter_args("<<carry.id<<"="<<zero.id<<")->(f32){\n";
          for(auto &scalar:bb){
            auto code=scalar.getName().stripDialect();
            if(code=="member_load"){
              SmallVector<Term> coords;
              for(auto i:scalar.getOperands().drop_front()){auto t=terms.lookup(i);coords.push_back(t.type=="index"?t:inst("cast",{t},"index"));}
              terms[scalar.getResult(0)]=read(scalar.getOperand(0),coords);
            }else if(code=="term"){
              auto total=inst("binary",{carry,terms.lookup(scalar.getOperand(0))},"f32","kind=\"add\"");
              s<<"scf.yield "<<total.id<<" : f32\n}\n";
            }else if(code=="mul" || code=="add")
              terms[scalar.getResult(0)]=inst("binary",{terms.lookup(scalar.getOperand(0)),terms.lookup(scalar.getOperand(1))},"f32","kind=\""+code.str()+"\"");
            else return llvm::createStringError("illegal ordered member operation "+code);
          }
          terms[v]={tag,"f32"};
        }
      } else if(n=="scan_sum"){
        auto &carry=carries[cursor];carry=inst("binary",{{prefix+"_p"+std::to_string(cursor),ty},terms.lookup(op.getOperand(0))},ty,"kind=\"add\"");
        terms[v]=carry;++cursor;
      } else if(scalarReduction && n==kind.getValue()){
        terms[v]=terms.lookup(op.getOperand(0));
      } else if(n=="field"){
        auto saved=records.find(op.getOperand(0));
        if(saved==records.end())return llvm::createStringError("unbound Weft tuple field");
        terms[v]=saved->second[op.getAttrOfType<IntegerAttr>("position").getInt()];
      } else if(n=="constant"){
        std::string lit;llvm::raw_string_ostream(lit)<<op.getAttr("value");
        terms[v]=inst("constant",{},ty,"value="+lit);
      } else {
        SmallVector<Term> args;
        for(auto a:op.getOperands())args.push_back(terms.lookup(a));
        if(n=="cast")terms[v]=inst("cast",args,ty);
        else if(n=="rsqrt" || n=="exp" || n=="sqrt")terms[v]=inst("unary",args,ty,"kind=\""+n.str()+"\"");
        else if(n=="mul" || n=="add" || n=="div" || n=="sub")terms[v]=inst("binary",args,ty,"kind=\""+n.str()+"\"");
        else return llvm::createStringError("unsupported scalar member calculation "+n.str());
      }
    }
    if(!carries.empty()){
      SmallVector<std::string> ids,types;for(auto &v:carries){ids.push_back(v.id);types.push_back(v.type);}
      s<<"scf.yield "<<llvm::join(ids,",")<<" : "<<llvm::join(types,",")<<"\n";
    }
    if(moments){
      Term n{prefix+"_n","si32"},mean{prefix+"_m","f32"},m2{prefix+"_s","f32"},v=terms.lookup(body.back().getOperand(0));
      auto bin=[&](StringRef k,Term a,Term b){return inst("binary",{a,b},a.type,"kind=\""+k.str()+"\"");};
      auto count=bin("add",n,{"%one_count","si32"});
      auto delta=bin("sub",v,mean), change=bin("div",delta,inst("cast",{count},"f32"));
      auto m=bin("add",mean,change);
      auto variance=bin("add",m2,bin("mul",bin("mul",delta,inst("cast",{n},"f32")),change));
      auto first=inst("compare",{n,{"%zero_count","si32"}},"i1","predicate=\"eq\"");
      m=inst("select",{first,v,m},"f32");
      variance=inst("select",{first,{"%zero_f","f32"},variance},"f32");
      s<<"scf.yield "<<count.id<<","<<m.id<<","<<variance.id<<" : si32,f32,f32\n";
      records[sweep.getResult(0)]={{prefix+"#0","si32"},{prefix+"#1","f32"},{prefix+"#2","f32"}};
    } else if(scalarReduction){
      Term acc{prefix+"_acc","f32"},value=terms.lookup(body.back().getOperand(0));
      auto next=inst("binary",{acc,value},"f32",kind=="sum"?"kind=\"add\"":"kind=\"max\"");
      s<<"scf.yield "<<next.id<<" : f32\n";
      terms[sweep.getResult(0)]={prefix,"f32"};
    }
    if(!publish)s<<"}\n";
    return llvm::Error::success();
  }

  llvm::Error writeBody(Block &body, StringRef start, Term active,
                        int dim = -2) {
    for (auto &op : body.without_terminator()) {
      auto code = op.getName().stripDialect();
      SmallVector<Term> args;
      if(code=="gather_row"){
        auto keys=terms.lookup(op.getOperand(1)),table=terms.lookup(op.getOperand(0));
        auto elem=printType(cast<MemRefType>(op.getOperand(1).getType()).getElementType());
        elem=elem=="i64"?"si64":"si32";
        auto family=elem=="si64"?std::string("i64"):std::string("i32");
        auto at=inst("slice",{keys,{"%r","index"}},"!weft_kernel.slice<!weft_kernel.encoding<\""+family+"\",\"dense\",\"dense."+family+"\",[]>,[],[]>","selectors=[\"index\"]");
        auto source=inst("cast",{inst("admit",{at},elem)},"index");
        auto dtype=spelling(cast<MemRefType>(op.getOperand(0).getType()).getElementType());
        auto s=inst("slice",{table,source},regionType("slice",dtype,1),"selectors=[\"index\",\"all\"]");
        terms[op.getResult(0)]=inst("admit",{s},regionType("value",dtype,1));continue;
      }
      if (code == "read" || code == "write" || code == "read_panel" ||
          code == "write_panel") {
        auto view = op.getOperand(0);
        auto vt = cast<MemRefType>(view.getType());
        std::string e = printType(vt.getElementType());
        args.push_back(terms.lookup(view));
        if (vt.getRank() >= 2)
          args.push_back({"%r", "index"});
        auto slice = inst("slice", args, regionType("slice", e, 1),
                          vt.getRank() >= 2 ? "selectors=[\"index\",\"all\"]"
                                            : "selectors=[\"all\"]");
        auto window =
            start.empty()
                ? slice
                : (dim > 0
                       ? inst("subview", {slice, {start.str(), "index"}},
                              regionType("slice", e, 1, dim),
                              "offsets=array<i64:-1>,extents=array<i64:" +
                                  std::to_string(dim) + ">")
                       : inst("subview",
                              {slice, {start.str(), "index"}, active},
                              regionType("slice", e, 1, dim),
                              "offsets=array<i64:-1>,extents=array<i64:-1>"));
        if (code == "write_panel" || code == "write")
          inst("commit",
               {terms.lookup(op.getOperand(op.getNumOperands() - 1)), window},
               "()");
        else
          terms[op.getResult(0)] =
              inst("admit", {window}, regionType("value", e, 1, dim));
        continue;
      }
      if (code == "ordinal") {
        bool wide=cast<RankedTensorType>(op.getResult(0).getType())
                     .getElementType().isUnsignedInteger(64);
        if (dim <= 0 || start.empty())
          return llvm::createStringError(
              "absolute lane coordinates require static panels");
        auto row =
            inst("binary", {{"%r", "index"}, origin}, "index", "kind=\"add\"");
        auto offset = inst("binary", {row, columns}, "index", "kind=\"mul\"");
        offset = inst("binary", {offset, {start.str(), "index"}}, "index",
                      "kind=\"add\"");
        auto elem=wide?"ui64":"ui32";
        auto ix = regionType("value", elem, 1, dim);
        offset = inst("cast", {offset}, elem);
        auto lanes = inst("iota", {}, ix,
                          "start=0:i64,end=" + std::to_string(dim) + ":i64");
        auto idx = inst("binary", {lanes, offset}, ix, "kind=\"add\"");
        terms[op.getResult(0)] =
            wide ? idx : inst("cast", {idx}, regionType("value", "si32", 1, dim));
        continue;
      }
      if (code=="scan_sum"){
        auto v=terms.lookup(op.getOperand(0));
        auto ty=spelling(cast<ShapedType>(op.getResult(0).getType()).getElementType());
        auto id="%prefix"+std::to_string(serial++);
        s<<id<<" = \"weft_kernel.scan\"("<<v.id<<") <{axis=0 : i64,reverse=false}> ({\n^bb0(%left:"<<ty<<",%right:"<<ty<<"):\n";
        auto x=inst("binary",{{"%left",ty},{"%right",ty}},ty,"kind=\"add\"");
        s<<"\"weft_kernel.scan_yield\"("<<x.id<<"):("<<ty<<")->()\n}):("<<v.type<<")->"<<v.type<<"\n";
        terms[op.getResult(0)]={id,v.type};continue;
      }
      if (code=="uniform") {
        terms[op.getResult(0)]=uniform(terms.lookup(op.getOperand(0)),terms.lookup(op.getOperand(1)),dim);
        continue;
      }
      if (code == "constant") {
        if (auto integer = op.getAttrOfType<IntegerAttr>("value")) {
          auto t = spelling(integer.getType());
          terms[op.getResult(0)] =
              inst("constant", {}, t,
                   "value=" + std::to_string(integer.getInt()) + " : " + t);
          continue;
        }
        auto val = op.getAttrOfType<FloatAttr>("value");
        auto t = printType(val.getType());
        std::string str;
        llvm::raw_string_ostream stream(str);
        stream << val;
        terms[op.getResult(0)] = inst("constant", {}, t, "value=" + str);
        continue;
      }
      for (auto operand : op.getOperands())
        args.push_back(terms.lookup(operand));
      auto v = op.getResult(0);
      auto t = v.getType();
      std::string result =
          isa<RankedTensorType>(t)
              ? regionType("value",
                           spelling(cast<RankedTensorType>(t).getElementType()),
                           1, dim)
              : spelling(t);
      std::string kind = code.str(), operation;
      if (code == "max_num" || code == "sum") {
        operation = "reduce";
        kind = code == "max_num" ? "max" : "add";
      } else if (code == "cast")
        operation = "cast";
      else if (code == "sqrt" || code == "rsqrt" || code == "exp")
        operation = "unary";
      else if (code == "select")
        operation = "select";
      else if (code == "gt")
        operation = "compare";
      else if (code == "xor" || code == "and" || code == "shl" ||
               code == "shr" || code == "maximum" || code == "mul" ||
               code == "add" || code == "div" || code == "sub")
        operation = "binary";
      else
        return llvm::createStringError("unsupported Weft task operation " +
                                       kind);
      auto p = operation == "compare" ? std::string("predicate=\"gt\"")
               : (operation == "cast" || operation == "select")
                   ? ""
                   : "kind=\"" + kind + "\"";
      if (operation == "reduce")
        p += ",axis=0:i64";
      terms[v] = inst(operation, args, result, p);
    }
    return llvm::Error::success();
  }

public:
  explicit CanonicalTask(llvm::raw_ostream &s) : s(s) {}
  llvm::Error write(Operation *task, const llvm::json::Array &abi) {
    const bool absolute =
        task->getAttrOfType<BoolAttr>("weft_host.absolute_origin").getValue();
    auto fn=task->getParentOfType<func::FuncOp>();
    int rankDomain=cast<MemRefType>(fn.getArgument(fn->getAttrOfType<IntegerAttr>("kir.domain_arg").getInt()).getType()).getRank();
    if(auto a=fn->getAttrOfType<IntegerAttr>("kir.owner_axis"))ownerAxis=a.getInt();
    std::string extraSymbols;
    bool ownedKeys=task->hasAttr("cpu.key_extent");
    if(ownedKeys)extraSymbols=",\"KL\",\"KH\"";
    if(ownerAxis)for(int j=0;j<rankDomain;++j)extraSymbols+=",\"D"+std::to_string(j)+"\"";
    SmallVector<std::string> names, effects, aliases, strided, params;
    auto &bb = task->getRegion(0).front();
    for (auto [i, arg] : llvm::enumerate(bb.getArguments().drop_front())) {
      auto a = abi[i].getAsObject();
      auto n = "a" + std::to_string(i);
      auto elem = a->getString("dtype")->str();
      int rank = std::min<size_t>(2,a->getArray("dims")->size());
      if (elem == "i32")
        elem = "si32";
      if(elem=="i64")elem="si64";
      if (elem == "u64")
        elem = "ui64";
      auto t = rank ? regionType("view", elem, rank) : elem;
      auto fn=task->getParentOfType<func::FuncOp>();
      if(auto count=fn.getArgAttrOfType<IntegerAttr>(i,"weft_host.table_rows"))
        t.replace(t.find("[-1,-2]"),7,"["+std::to_string(count.getInt())+",-2]");
      if(fn.getArgAttr(i,"weft_host.task_indices"))
        t.replace(t.find("[-2],[2]"),8,"[-1],[1]");
      if(auto sizes=fn.getArgAttrOfType<DenseI64ArrayAttr>(i,"weft_host.indirect_extents")){
        SmallVector<std::string> bound;for(auto d:sizes.asArrayRef())bound.push_back(std::to_string(d));
        auto at=t.rfind(">,[")+2;
        t.replace(at,t.find(']',at)-at+1,"["+llvm::join(bound,",")+"]");
      }
      if(ownerAxis && rank){
        rank=a->getArray("dims")->size();
        SmallVector<std::string> shape,axes;
        for(int j=0;j<rank;++j){
          shape.push_back(j==*ownerAxis || rank==1?"-1":std::to_string(-4-absolute-j));
          axes.push_back(std::to_string(j+1));
        }
        t="!weft_kernel.view<!weft_kernel.encoding<\""+elem+"\",\"dense\",\"dense."+elem+"\",[]>,["+llvm::join(shape,",")+"],["+llvm::join(axes,",")+"]>";
      }
      terms[arg] = {"%" + n, t};
      names.push_back("\"" + n + "\"");
      params.push_back("%" + n + ": " + t);
      bool initializedBins=task->walk([&](Operation *s){
        if(!s->hasAttr("weft_host.private_bins"))return WalkResult::advance();
        auto &body=s->getRegion(0).front();
        return body.front().getNextNode()->getOperand(0)==arg?WalkResult::interrupt():WalkResult::advance();
      }).wasInterrupted();
      effects.push_back(rank ? (a->getString("direction") == "Out" ? (initializedBins?"\"readwrite\"":"\"write\"")
                                : a->getString("direction") == "InOut"
                                    ? "\"readwrite\""
                                    : "\"read\"")
                             : "\"none\"");
      aliases.push_back(rank ? "0" : "-1");
      if (rank && !fn.getArgAttr(i,"weft_host.contiguous"))
        strided.push_back(names.back());
    }
    s << "module {\n\"weft_kernel.kernel\"() <{sym_name=\""
      << task->getAttrOfType<StringAttr>("task_symbol").getValue()
      << "\",source=\"CPU row interval\",arg_names=[" << llvm::join(names, ",")
      << "],arg_access=[" << llvm::join(effects, ",")
      << "],arg_alias_sets=array<i64:" << llvm::join(aliases, ",")
      << ">,shape_symbols=[\"T\",\"N\",\"C\"" << (absolute ? ",\"O\"" : "") << extraSymbols
      << "],strided_args=[" << llvm::join(strided, ",") << "]}> ({\n^bb0("
      << llvm::join(params, ",") << "):\n";
    auto extent = inst("symbol", {}, "index",
                       "choices=array<i64>,kind=\"shape\",name=\"T\"");
    columns = inst("symbol", {}, "index",
                   "choices=array<i64>,kind=\"shape\",name=\"N\"");
    auto panelWidth = inst("symbol", {}, "index",
                           "choices=array<i64>,kind=\"shape\",name=\"C\"");
    if (absolute)
      origin = inst("symbol", {}, "index",
                    "choices=array<i64>,kind=\"shape\",name=\"O\"");
    Term keyLo,keyHi;
    if(ownedKeys){
      keyLo=inst("symbol",{},"index","choices=array<i64>,kind=\"shape\",name=\"KL\"");
      keyHi=inst("symbol",{},"index","choices=array<i64>,kind=\"shape\",name=\"KH\"");
    }
    if(ownerAxis){
      for(int j=0;j<rankDomain;++j)
        extents.push_back(inst("symbol",{},"index","choices=array<i64>,kind=\"shape\",name=\"D"+std::to_string(j)+"\""));
    }
    // Entry descriptors are materialized at the root. All ABI shape symbols
    // used by their types must dominate that boundary.
    inst("root_domain", {},
         "!weft_kernel.domain<\"root\",0,-1,0,\"root\",\"exact\">");
    s << "%lo=arith.constant 0 : index\n%step=arith.constant 1 : index\n";
    if(task->hasAttr("weft_host.scalar_members")){
      for(auto [name,literal,type]:{std::tuple{"zero_count","0","si32"},std::tuple{"one_count","1","si32"},std::tuple{"zero_f","0.0","f32"}})
        s<<"%"<<name<<" = \"weft_kernel.constant\"() <{value="<<literal<<" : "<<type<<"}> : ()->"<<type<<"\n";
      s<<"%minus_inf=arith.constant 0xFF800000 : f32\n";
    }
    bool whole = cast<DictionaryAttr>(task->getAttrOfType<ArrayAttr>("rows")[0])
                     .getAs<IntegerAttr>("columns")
                     .getInt() == 0;
    auto configurations = task->getAttrOfType<ArrayAttr>("rows");
    int64_t fixed = cast<DictionaryAttr>(configurations[0])
                        .getAs<IntegerAttr>("columns")
                        .getInt();
    if (!llvm::all_of(configurations, [&](Attribute row) {
          return cast<DictionaryAttr>(row)
                     .getAs<IntegerAttr>("columns")
                     .getInt() == fixed;
        }))
      fixed = 0;
    Term panelCount, tailStart;
    if (!whole) {
      panelCount =
          inst("binary", {columns, panelWidth}, "index", "kind=\"div\"");
      tailStart =
          inst("binary", {panelCount, panelWidth}, "index", "kind=\"mul\"");
    }
    if(intentdsl::cpu::reservations(bb)){
      auto bins=task->getAttrOfType<IntegerAttr>("cpu.batch_extent").getInt();
      auto size=indexLiteral(bins),one=inst("constant",{},"si32","value=1:si32"),zero=inst("constant",{},"si32","value=0:si32");
      auto mask=inst("compare",{one,one},"i1","predicate=\"eq\"");
      auto ptr=inst("local_alloc",{},"!weft_kernel.pointer<si32>","elements="+std::to_string(bins)+":i64,alignment=64:i64");
      auto read=[&](Term k){return inst("pointer_load",{ptr,inst("cast",{k},"si64"),mask,zero},"si32");};
      auto store=[&](Term k,Term value){inst("pointer_store",{ptr,inst("cast",{k},"si64"),mask,value},"()");};
      auto addr=[&](Value v,SmallVector<Term> coords){
        SmallVector<Term> args{terms.lookup(v)};args.append(coords);SmallVector<std::string> sel(coords.size(),"\"index\"");
        auto elem=printType(cast<MemRefType>(v.getType()).getElementType());
        return inst("slice",args,"!weft_kernel.slice<!weft_kernel.encoding<\""+elem+"\",\"dense\",\"dense."+elem+"\",[]>,[],[]>","selectors=["+llvm::join(sel,",")+"]");
      };
      auto add=[&](Term a,Term b){return inst("binary",{a,b},a.type,"kind=\"add\"");};
      for(auto &stage:bb.without_terminator()){
        auto n=stage.getName().stripDialect();
        if(n=="tally_keys"){
          s<<"scf.for %k0=%lo to "<<size.id<<" step %step {\n";
          store(inst("cast",{{"%k0","index"}},"si32"),zero);s<<"}\n";
        }
        if(n=="reserve_tickets"){
          s<<"scf.for %k1=%lo to "<<size.id<<" step %step {\n";Term k{"%k1","index"};
          auto offset=inst("cast",{k},"si32"),count=read(offset),used=inst("compare",{count,zero},"i1","predicate=\"ne\"");
          s<<"scf.if "<<used.id<<" {\n";auto base=inst("atomic_rmw",{count,addr(stage.getOperand(0),{k})},"si32","kind=\"add\",ordering=\"relaxed\"");
          store(offset,base);s<<"}\n}\n";continue;
        }
        std::string suffix=n=="tally_keys"?"1":"2";Term r{"%r"+suffix,"index"},j{"%j"+suffix,"index"};
        s<<"scf.for "<<r.id<<"=%lo to "<<extent.id<<" step %step {\nscf.for "<<j.id<<"=%lo to "<<columns.id<<" step %step {\n";
        auto key=inst("admit",{addr(stage.getOperand(0),{r,j})},"si32"),old=read(key);
        if(n=="distribute_tickets"){
          auto offset=inst("admit",{addr(stage.getOperand(1),{inst("cast",{key},"index")})},"si32");
          auto pos=inst("cast",{add(old,offset)},"index");
          auto global=inst("binary",{add(r,origin),columns},"index","kind=\"mul\"");
          inst("commit",{inst("cast",{add(global,j)},"si32"),addr(stage.getOperand(3),{pos})},"()");
        }
        store(key,add(old,one));s<<"}\n}\n";
      }
      s<<"\"weft_kernel.return\"():()->()\n}):()->()\n}\n";return llvm::Error::success();
    }
    s << "scf.for %r = %lo to " << extent.id << " step %step {\n";
    if(intentdsl::cpu::atomicMembers(bb)){
      auto &effects=bb.front().getRegion(0).front();
      auto found=llvm::find_if(effects,[](Operation &v){return v.getName().stripDialect()=="write_offsets";});
      bool prefix=found!=effects.end();
      bool guarded=false;
      Operation *ends=prefix?&*found:nullptr;
      auto zero=inst("constant",{},"si32","value=0:si32");
      SmallVector<Value> scans;
      for(auto &v:effects)if(v.getName().stripDialect()=="scan_sum")scans.push_back(v.getResult(0));
      auto slot=[&](Value v){return std::to_string(llvm::find(scans,v)-scans.begin());};
      SmallVector<std::string> binds,types;
      for(auto v:scans){binds.push_back("%carry"+slot(v)+"="+zero.id);types.push_back("si32");}
      if(!scans.empty())s<<"%scan:"<<scans.size()<<" = ";
      s<<"scf.for %j=%lo to "<<columns.id<<" step %step";
      if(!scans.empty())s<<" iter_args("<<llvm::join(binds,",")<<")->("<<llvm::join(types,",")<<")";
      s<<" {\n";
      auto address=[&](Value source,SmallVector<Term> keys){
        auto base=terms.lookup(source);SmallVector<Term> args{base};
        args.append(keys);SmallVector<std::string> sel(keys.size(),"\"index\"");
        auto elem=printType(cast<MemRefType>(source.getType()).getElementType());
        return inst("slice",args,"!weft_kernel.slice<!weft_kernel.encoding<\""+elem+"\",\"dense\",\"dense."+elem+"\",[]>,[],[]>","selectors=["+llvm::join(sel,",")+"]");
      };
      for(auto &v:bb.front().getRegion(0).front().without_terminator()){
        auto n=v.getName().stripDialect();auto a=[&](unsigned i){return terms.lookup(v.getOperand(i));};Term rhs;
        if(n=="read"){
          SmallVector<Term> coords;if(cast<MemRefType>(v.getOperand(0).getType()).getRank()==2)coords.push_back({"%r","index"});
          coords.push_back({"%j","index"});rhs=inst("admit",{address(v.getOperand(0),coords)},spelling(cast<MemRefType>(v.getOperand(0).getType()).getElementType()));
        }
        else if(n=="fill_ranges"){
          auto j=Term{"%j","index"};auto hi=inst("binary",{j,{"%step","index"}},"index","kind=\"add\"");
          auto in=inst("compare",{hi,columns},"i1","predicate=\"lt\"");s<<"scf.if "<<in.id<<" {\n";
          auto step=indexLiteral(v.getAttrOfType<IntegerAttr>("unit").getInt());
          auto block=[&](Term pos){
            auto n=inst("admit",{address(v.getOperand(1),{pos})},"si32");
            return inst("binary",{inst("cast",{n},"index"),step},"index","kind=\"div\"");
          };
          auto first=block(j),end=block(hi);s<<"scf.for %b = "<<first.id<<" to "<<end.id<<" step %step {\n";
          inst("commit",{inst("cast",{j},"si32"),address(v.getOperand(0),{{"%b","index"}})},"()");s<<"}\n}\n";continue;
        }
        else if(n=="constant"){
          if(auto x=v.getAttrOfType<IntegerAttr>("value"))rhs=inst("constant",{},"si32","value="+std::to_string(x.getInt())+":si32");
          else {std::string repr;llvm::raw_string_ostream(repr)<<v.getAttr("value");rhs=inst("constant",{},"f32","value="+repr);}
        }else if(n=="member_index")rhs=inst("cast",{{"%j","index"}},"si32");
        else if(n=="ne")rhs=inst("compare",{a(0),a(1)},"i1","predicate=\"ne\"");
        else if(n=="cast")rhs=inst("cast",{a(0)},"si32");
        else if(n=="prefix_scatter"){
          s<<"scf.if "<<a(3).id<<" {\n";
          auto ix=inst("binary",{inst("cast",{a(4)},"index"),{"%step","index"}},"index","kind=\"sub\"");
          inst("commit",{a(5),address(v.getOperand(0),{{"%r","index"},ix})},"()");
          s<<"}\n";continue;
        }
        else if(n=="ordinal"){
          auto r=inst("binary",{origin,{"%r","index"}},"index","kind=\"add\"");
          auto row=inst("binary",{r,columns},"index","kind=\"mul\"");
          rhs=inst("cast",{inst("binary",{row,{"%j","index"}},"index","kind=\"add\"")},"si32");
        }else if(n=="add" || n=="shr" || n=="shl")rhs=inst("binary",{a(0),a(1)},"si32","kind=\""+n.str()+"\"");
        else if(n=="scan_sum")rhs=inst("binary",{{"%carry"+slot(v.getResult(0)),"si32"},a(0)},"si32","kind=\"add\"");
        else if(n=="write_offsets"){
          auto dst=inst("binary",{{"%j","index"},{"%step","index"}},"index","kind=\"add\"");
          inst("commit",{a(3),address(v.getOperand(0),{dst})},"()");continue;
        }
        else if(n=="atomic_add" || n=="index_read" || n=="unique_put"){
          auto index=inst("cast",{a(1)},"index");
          if(ownedKeys && n=="atomic_add"){
            auto lower=inst("compare",{index,keyLo},"i1","predicate=\"ge\"");
            auto upper=inst("compare",{index,keyHi},"i1","predicate=\"lt\"");
            auto live=inst("binary",{lower,upper},"i1","kind=\"and\"");
            s<<"scf.if "<<live.id<<" {\n";guarded=true;
          }
          auto dst=address(v.getOperand(0),{index});
          if(n=="atomic_add")rhs=inst("atomic_rmw",{a(2),dst},"si32","kind=\"add\",ordering=\"relaxed\"");
          else if(n=="unique_put"){inst("commit",{a(2),dst},"()");continue;}
          else rhs=inst("admit",{dst},"si32");
        }else return llvm::createStringError("unsupported atomic task expression "+n);
        terms[v.getResult(0)]=rhs;
      }
      if(guarded)s<<"}\n";
      if(!scans.empty()){
        SmallVector<std::string> outputs;
        for(auto v:scans)outputs.push_back(terms.lookup(v).id);
        s<<"scf.yield "<<llvm::join(outputs,",")<<":"<<llvm::join(types,",")<<"\n";
      }
      s<<"}\n";
      if(prefix){
        inst("commit",{zero,address(ends->getOperand(0),{{"%lo","index"}})},"()");
        inst("commit",{{"%scan#"+slot(ends->getOperand(3)),"si32"},address(ends->getOperand(1),{{"%lo","index"}})},"()");
      }
      for(auto &v:effects)if(v.getName().stripDialect()=="prefix_scatter"){
        Term total{"%scan#"+slot(v.getOperand(4)),"si32"},row{"%r","index"};
        inst("commit",{total,address(v.getOperand(1),{row})},"()");
        auto first=inst("cast",{total},"index");
        auto pad=inst("constant",{},"si32","value="+std::to_string(v.getAttrOfType<IntegerAttr>("fill").getInt())+":si32");
        s<<"scf.for %tail="<<first.id<<" to "<<columns.id<<" step %step {\n";
        inst("commit",{pad,address(v.getOperand(0),{row,{"%tail","index"}})},"()");s<<"}\n";
      }
    }else if(intentdsl::cpu::packed(bb)){
      auto &pack=bb.front();Term row{"%r","index"},col{"%j","index"};
      auto address=[&](Value v,SmallVector<Term> at){
        auto elem=printType(cast<MemRefType>(v.getType()).getElementType());
        SmallVector<Term> args{terms.lookup(v)};args.append(at);
        SmallVector<std::string> selector(at.size(),"\"index\"");
        return inst("slice",args,"!weft_kernel.slice<!weft_kernel.encoding<\""+elem+"\",\"dense\",\"dense."+elem+"\",[]>,[],[]>","selectors=["+llvm::join(selector,",")+"]");
      };
      std::string threshold;llvm::raw_string_ostream(threshold)<<pack.getAttr("threshold");
      auto rhs=inst("constant",{},"f32","value="+threshold);
      auto pad=inst("constant",{},"si32","value="+std::to_string(pack.getAttrOfType<IntegerAttr>("fill").getInt())+":si32");
      s<<"%count=scf.for %j=%lo to "<<columns.id<<" step %step iter_args(%pos=%lo)->(index){\n";
      auto x=inst("admit",{address(pack.getOperand(0),{row,col})},"f32");
      auto keep=inst("compare",{x,rhs},"i1","predicate=\"ne\"");
      s<<"scf.if "<<keep.id<<" {\n";
      inst("commit",{inst("cast",{col},"si32"),address(pack.getOperand(1),{row,{"%pos","index"}})},"()");s<<"}\n";
      auto next=inst("binary",{{"%pos","index"},inst("select",{keep,{"%step","index"},{"%lo","index"}},"index")},"index","kind=\"add\"");
      s<<"scf.yield "<<next.id<<":index\n}\nscf.for %tail=%count to "<<columns.id<<" step %step {\n";
      inst("commit",{pad,address(pack.getOperand(1),{row,{"%tail","index"}})},"()");s<<"}\n";
      inst("commit",{inst("cast",{{"%count","index"}},"si32"),address(pack.getOperand(2),{row})},"()");
    }else if(task->hasAttr("weft_host.scalar_members")){
      for(auto &op:bb.without_terminator()){
        if(op.getName().stripDialect()=="capture_owner")
          terms[op.getResult(0)]=inst("admit",{memberSlice(op,{},true)},spelling(op.getResult(0).getType()));
        else if(auto error=ownerSweep(op))return error;
      }
    } else if (whole) {
      if (auto e = writeBody(bb, {}, {}))
        return e;
    } else
      for (auto &sweep : bb.without_terminator()) {
        if(sweep.hasAttr("weft_host.private_bins") || sweep.hasAttr("weft_host.vertical_sum")){
          auto &body=sweep.getRegion(0).front();auto &key=body.front();auto src=key.getOperand(0),dst=body.back().getPrevNode()->getOperand(0);
          auto *fnOp=task->getParentOfType<func::FuncOp>().getOperation();
          auto shape=cast<func::FuncOp>(fnOp).getArgAttrOfType<DenseI64ArrayAttr>(cast<BlockArgument>(src).getArgNumber()-1,"weft_host.indirect_extents");
          auto size=indexLiteral(shape[0]);
          auto scalarAddress=[&](Value buffer,SmallVector<Term> coords){
            auto e=printType(cast<MemRefType>(buffer.getType()).getElementType());
            SmallVector<Term> args{terms.lookup(buffer)};args.append(coords);
            SmallVector<std::string> selectors(coords.size(),"\"index\"");
            return inst("slice",args,"!weft_kernel.slice<!weft_kernel.encoding<\""+e+"\",\"dense\",\"dense."+e+"\",[]>,[],[]>","selectors=["+llvm::join(selectors,",")+"]");
          };
          auto zero=inst("constant",{},"si32","value=0:si32");
          if(sweep.hasAttr("weft_host.private_bins")){
            s<<"scf.for %bin = %lo to "<<columns.id<<" step %step {\n";
            inst("commit",{zero,scalarAddress(dst,{{"%r","index"},{"%bin","index"}})},"()");s<<"}\n";
            auto span=indexLiteral(key.getAttrOfType<IntegerAttr>("span").getInt());
            auto row=inst("binary",{origin,{"%r","index"}},"index","kind=\"add\"");
            auto start=inst("binary",{row,span},"index","kind=\"mul\"");
            auto limit=inst("binary",{start,span},"index","kind=\"add\"");
            auto shorter=inst("compare",{limit,size},"i1","predicate=\"lt\"");
            auto end=inst("select",{shorter,limit,size},"index");
            s<<"scf.for %sample = "<<start.id<<" to "<<end.id<<" step %step {\n";
            auto val=inst("admit",{scalarAddress(src,{{"%sample","index"}})},spelling(cast<MemRefType>(src.getType()).getElementType()));
            auto bin=val.type=="f32"?inst("cast",{val},"si32"):val;
            auto k=inst("cast",{bin},"index");
            auto lo=inst("compare",{k,{"%lo","index"}},"i1","predicate=\"ge\"");
            auto hi=inst("compare",{k,columns},"i1","predicate=\"lt\"");
            auto keep=inst("binary",{lo,hi},"i1","kind=\"and\"");
            s<<"scf.if "<<keep.id<<" {\n";
            auto ptr=scalarAddress(dst,{{"%r","index"},k}),old=inst("admit",{ptr},"si32");
            auto one=inst("constant",{},"si32","value=1:si32");
            inst("commit",{inst("binary",{old,one},"si32","kind=\"add\""),ptr},"()");s<<"}\n}\n";
          }else{
            s<<"scf.for %bin = %lo to "<<columns.id<<" step %step {\n%sum=scf.for %part=%lo to "<<size.id<<" step %step iter_args(%count="<<zero.id<<")->(si32){\n";
            auto x=inst("admit",{scalarAddress(src,{{"%part","index"},{"%bin","index"}})},"si32");
            auto sum=inst("binary",{{"%count","si32"},x},"si32","kind=\"add\"");
            s<<"scf.yield "<<sum.id<<" : si32\n}\n";
            Term total{"%sum","si32"};auto type=spelling(cast<MemRefType>(dst.getType()).getElementType());
            if(type!=total.type)total=inst("cast",{total},type);
            inst("commit",{total,scalarAddress(dst,{{"%bin","index"}})},"()");s<<"}\n";
          }
          continue;
        }
        if(sweep.hasAttr("weft_host.prefix_carry") || sweep.hasAttr("weft_host.segment_loop")){
          if(auto error=ownerSweep(sweep))return error;
          continue;
        }
        auto combine = sweep.getAttrOfType<StringAttr>("combine");
        bool store = combine == "store";
        std::string previous;
        if (!store) {
          previous = "%neutral" + std::to_string(serial++);
          s << previous << "=arith.constant "
            << (combine == "sum" ? "0.0" : "0xFF800000") << " : f32\n";
        }
        auto &body = sweep.getRegion(0).front();
        for (bool tail : {false, true}) {
          auto id = std::to_string(serial++), induction = "%index" + id,
               acc = "%acc" + id, summary = "%summary" + id;
          if (!store)
            s << summary << " = ";
          s << "scf.for " << induction << " = " << (tail ? tailStart.id : "%lo")
            << " to " << (tail ? columns.id : panelCount.id) << " step %step";
          if (!store)
            s << " iter_args(" << acc << "=" << previous << ") -> (f32)";
          s << " {\n";
          auto offset = tail
                            ? Term{induction, "index"}
                            : inst("binary", {{induction, "index"}, panelWidth},
                                   "index", "kind=\"mul\"");
          if (auto e = writeBody(body, offset.id, panelWidth,
                                 tail    ? 1
                                 : fixed ? fixed
                                         : -3))
            return e;
          if (!store) {
            auto partial = terms.lookup(body.back().getOperand(0));
            auto next =
                inst("binary", {{acc, "f32"}, partial}, "f32",
                     combine == "sum" ? "kind=\"add\"" : "kind=\"max\"");
            s << "scf.yield " << next.id << " : f32\n";
            previous = summary;
          }
          s << "}\n";
        }
        if (!store)
          terms[sweep.getResult(0)] = {previous, "f32"};
      }
    s << "}\n\"weft_kernel.return\"():()->()\n}):()->()\n}\n";
    return llvm::Error::success();
  }
};
} // namespace
llvm::Expected<std::string> taskSource(Operation *op,
                                       const llvm::json::Array &abi) {
  std::string s;
  llvm::raw_string_ostream out(s);
  if (auto e = CanonicalTask(out).write(op, abi))
    return e;
  return s;
}

std::string hostSource(Operation *dispatch, const llvm::json::Array &abi) {
  bool members=dispatch->hasAttr("cpu.member_tasks");
  auto slots =
      dispatch->getAttrOfType<DenseI64ArrayAttr>("weft_host.slot_ranks");
  auto symbol = dispatch->getAttrOfType<StringAttr>("task_symbol").str();
  SmallVector<std::string> params, args, strides;
  bool projected=llvm::any_of(slots.asArrayRef(),[](auto n){return n>2;});
  auto owner=dispatch->getParentOfType<func::FuncOp>()->getAttrOfType<IntegerAttr>("kir.owner_axis");
  if(owner)projected=false;
  for (auto [i, entry] : llvm::enumerate(abi)) {
    auto a = entry.getAsObject();
    auto n = "a" + std::to_string(i);
    auto rank = slots[i];
    auto type = a->getString("dtype") == "f16"    ? "_Float16"
                : a->getString("dtype") == "bf16" ? "uint16_t"
                : a->getString("dtype") == "i32"  ? "int32_t"
                : a->getString("dtype") == "i64"  ? "int64_t"
                : a->getString("dtype") == "u64"  ? "uint64_t"
                                                  : "float";
    params.push_back(std::string(type) + (rank ? " *" : " ") + n);
    bool swap = false;
    dispatch->walk([&](Operation *op) {
      if (op->hasAttr("axes") &&
          op->getOperand(0) ==
              dispatch->getRegion(0).front().getArgument(i + 1))
        swap = true;
    });
    if(rank>2)for(int j=0;j<rank-1;++j)params.push_back("int64_t "+n+"d"+std::to_string(j));
    for (size_t axis = 0; axis < rank; ++axis) {
      auto stride = n + "s" + std::to_string(axis);
      params.push_back("int64_t " + stride);
      if(!dispatch->getParentOfType<func::FuncOp>().getArgAttr(i,"weft_host.contiguous") &&
         (owner || rank<=2 || axis>=rank-2))strides.push_back(swap ? n + "s" + std::to_string(1 - axis) : stride);
    }
    auto fn=dispatch->getParentOfType<func::FuncOp>();
    if(fn.getArgAttr(i,"weft_host.table_rows") || fn.getArgAttr(i,"weft_host.indirect_extents"))args.push_back(n);
    else if(members && i==dispatch->getAttrOfType<IntegerAttr>("domain_arg").getInt())args.push_back(n+"+start*"+n+"s0");
    else if(fn.getArgAttr(i,"weft_host.task_indices"))args.push_back(n+"+start*"+n+"s0");
    else if(owner && rank)args.push_back(n+"+start*"+n+"s"+std::to_string(rank==1?0:owner.getInt()));
    else if(rank>2){
      std::string row="start",offset=n;
      for(int j=rank-2;j>=0;--j){auto dim=n+"d"+std::to_string(j);offset+="+(("+row+")%"+dim+")*"+n+"s"+std::to_string(j);row="("+row+"/"+dim+")";}
      args.push_back(offset);
    }else args.push_back(n +
                   (rank == 2 ? " + start*" + n + (swap ? "s1" : "s0") : ""));
  }
  args.push_back(members?"1":"count");
  args.push_back(members?"count":"cols");
  args.push_back("panel ? panel : cols");
  if (dispatch->getAttrOfType<BoolAttr>("weft_host.absolute_origin").getValue())
    args.push_back("start");
  if(dispatch->hasAttr("cpu.key_extent")){args.push_back("lo");args.push_back("hi");}
  if(owner){
    auto domain=dispatch->getAttrOfType<IntegerAttr>("domain_arg").getInt();
    auto rank=slots[domain]; auto base="a"+std::to_string(domain);
    std::string last="cols";
    for(int j=0;j<rank-1;++j)if(j!=owner.getInt())last+="/"+base+"d"+std::to_string(j);
    for(int j=0;j<rank;++j)args.push_back(j==owner.getInt()?"count":rank==2?"cols":j+1==rank?last:base+"d"+std::to_string(j));
  }
  args.append(strides);
  auto domain=dispatch->getAttrOfType<IntegerAttr>("domain_arg").getInt();
  std::string chunk=projected?"a"+std::to_string(domain)+"d"+std::to_string(slots[domain]-2):"rows";
  auto walk=dispatch->getAttrOfType<StringAttr>("cpu.walk");
  std::string prefix=std::string("#include <stdint.h>\n#include \"task_bridge.h\"\nint32_t intent_entry(") +
         llvm::join(params, ",") +
         ", int64_t rows, int64_t cols, int64_t grain, int64_t panel, int64_t "
         "workers){\nif(rows<0 || cols<0 || grain<=0 || panel<0 || workers<=0) return 1;\n"
         "if(!rows) return 0;\n";
  if(walk=="tile")return prefix+intentdsl::cpu::tileLoop(dispatch->getAttrOfType<IntegerAttr>("cpu.tile").getInt())+symbol+"_instance(a0+r*a0s0+c*a0s1,a1+c*a1s0+r*a1s1,nr,nc,panel?panel:nc,a0s0,a0s1,a1s1,a1s0);}return 0;}\n";
  if(walk=="destination")prefix+="int64_t t=rows;rows=cols;cols=t;t=a0s0;a0s0=a0s1;a0s1=t;t=a1s0;a1s0=a1s1;a1s1=t;\n";
  // A worker owns a contiguous range of complete grains. Give its native
  // kernel the full interval, splitting only at a projected view boundary.
  if(members)prefix+="rows=cols;\n";
  if(auto keys=dispatch->getAttrOfType<IntegerAttr>("cpu.key_extent")){
    prefix+="int64_t keys="+std::to_string(keys.getInt())+",lines=(keys+15)/16,n=workers<lines?workers:lines;\n";
    return prefix+"#pragma omp parallel for num_threads(n) schedule(static) if(n>1)\nfor(int64_t b=0;b<n;++b){int64_t start=0,count=rows,lo=16*(lines*b/n),hi=16*(lines*(b+1)/n);"+
      symbol+"_instance("+llvm::join(args,",")+");}return 0;}\n";
  }
  if(dispatch->hasAttr("cpu.single_dispatcher"))prefix+="workers=1;\n";
  prefix += "int64_t groups=1+(rows-1)/grain; int64_t bins=groups<workers?groups:workers;\n";
  return prefix+"#pragma omp parallel for num_threads(bins) "
         "schedule(static) if(workers>1)\n"
         "for(int64_t b=0;b<bins;++b){int64_t group=(groups/bins*b+(groups%bins*b)/bins)*grain;"
         "int64_t end=b+1==bins?rows:(groups/bins*(b+1)+(groups%bins*(b+1))/bins)*grain;"
         "for(int64_t start=group;start<end;){int64_t remaining=" + chunk + "-start%" + chunk + ";int64_t count=end-start<remaining?end-start:remaining; " +
         symbol + "_instance(" + llvm::join(args, ",") + "); start+=count;}}\nreturn 0;\n}\n";
}
} // namespace CI_UNIT_NAMESPACE
