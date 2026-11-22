#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
using Text = std::string;
Text ctype(Type t) {
  if (auto s = dyn_cast<ShapedType>(t))
    t = s.getElementType();
  if (t.isBF16())
    return "unsigned short";
  if (t.isInteger(1))
    return "unsigned char";
  if (t.isInteger(32))
    return "int";
  if(t.isSignlessInteger(64))return "long long";
  if (t.isUnsignedInteger(64)) return "unsigned long long";
  return t.isF16() ? "half" : "float";
}
bool tile(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.getRank() == 2;
}
class Source {
  llvm::raw_ostream &o;
  func::FuncOp fn;
  bool resident;
  unsigned depth;
  llvm::DenseMap<Value, Text> names;
  unsigned seq = 0;
  Text fresh() { return "v" + std::to_string(seq++); }
  Text name(Value v) { return names.lookup(v); }
  void line(const Text &s) { o << s << "\n"; }
  Text args(bool decl) {
    Text s;
    for (auto a : fn.getArguments()) {
      auto t = dyn_cast<MemRefType>(a.getType());
      auto n = "a" + std::to_string(a.getArgNumber());
      names[a] = n;
      s += (decl ? ctype(a.getType()) + (t ? " *" : " ") : "") + n + ", ";
      if (t)
        for (int j = 0; j < t.getRank(); ++j)
          s += (decl ? "long long " : "") + n + "d" + std::to_string(j) + ", ";
      if (t)
        for (int j = 0; j < t.getRank(); ++j)
          s += (decl ? "long long " : "") + n + "s" + std::to_string(j) + ", ";
    }
    s += decl ? "long long M,long long N,long long grid" : "M,N,grid";
    return s;
  }
  // A submitted group owns complete, pitched slots, even for partial edge
  // groups.
  Text address(Operation &op, StringRef key = "bangc.offset") {
    auto parts = op.getAttrOfType<DenseI64ArrayAttr>(key);
    return "(storage+P*(" + std::to_string(parts[0] * depth) + "*R+" +
           std::to_string(parts[1]) + (fn->getAttrOfType<DictionaryAttr>("dsa.layout").getAs<BoolAttr>("group_temporaries").getValue() ? "*R" : "") + "))";
  }
  std::pair<Text, Text> strides(Operation &op, unsigned start,
                                const Text &view) {
    Text offset="0",step="0";
    for(auto [i,c]:llvm::enumerate(op.getOperands().drop_front(start))){
      auto value=cast<OpResult>(c);Text stride=view+"s"+std::to_string(i);
      if(c.getDefiningOp()->getName().getStringRef()=="dsa.lookup_axis"){
        auto ids=name(c.getDefiningOp()->getOperand(0));
        offset+="+"+ids+"[(row0+j)*"+ids+"s0]*"+stride;
      }else if(c.getDefiningOp()->getName().getStringRef()=="dsa.member_axis"){
        auto *axis=c.getDefiningOp();auto domain=axis->getOperand(1);int k=axis->getAttrOfType<IntegerAttr>("axis").getInt();
        Text coord="(row0+j)";
        for(int j=k+1;j<cast<MemRefType>(domain.getType()).getRank()-1;++j)coord+="/"+name(domain)+"d"+std::to_string(j);
        offset+="+(("+coord+")%"+name(domain)+"d"+std::to_string(k)+")*"+stride;
      }else if(value.getResultNumber()==0)offset+="+(row0+j)*"+stride;
      else step=stride;
    }return {offset,step};
  }
  Text slabOffset(Operation &op,unsigned first,const Text &view){
    Text offset="0";
    const auto owner = fn->getAttrOfType<IntegerAttr>("kir.owner_axis").getInt();
    for (auto [pos, value] : llvm::enumerate(op.getOperands().drop_front(first))) {
      auto *coord = value.getDefiningOp();
      Text index = "(row0+j)";
      if (coord->getName().getStringRef() == "dsa.slab_axis") {
        auto domain = coord->getOperand(1);
        auto axis = coord->getAttrOfType<IntegerAttr>("axis").getInt();
        auto rank = cast<MemRefType>(domain.getType()).getRank();
        index = "(col+k)";
        for (auto dim = axis + 1; dim < rank; ++dim)
          if (dim != owner) index += "/" + name(domain) + "d" + std::to_string(dim);
        index = "(" + index + "%" + name(domain) + "d" + std::to_string(axis) + ")";
      }
      offset += "+" + index + "*" + view + "s" + std::to_string(pos);
    }
    return offset;
  }
  // A flattened slab stays affine until the fastest non-owner axis wraps.
  // Projected views retain their bound strides, including zero for broadcast.
  void slabTransfer(Operation &op,unsigned first,const Text &view,
                    const Text &local,bool load) {
    auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis").getInt();
    Text extent="1", stride="0", outerStride;
    for(auto [pos,v]:llvm::enumerate(op.getOperands().drop_front(first))){
      auto *axis=v.getDefiningOp();
      if(axis->getName().getStringRef()!="dsa.slab_axis") continue;
      auto domain=axis->getOperand(1);
      auto rank=cast<MemRefType>(domain.getType()).getRank();
      auto inner=rank-1-(owner==rank-1);
      extent=name(domain)+"d"+std::to_string(inner);
      if(axis->getAttrOfType<IntegerAttr>("axis").getInt()==inner)
        stride=view+"s"+std::to_string(pos);
      else if(rank==3) outerStride=view+"s"+std::to_string(pos);
    }
    line("{auto *buffer="+local+";for(int k=0;k<live;){long long avail="+extent+"-(col+k)%"+extent+";int run=avail<live-k?(int)avail:live-k;");
    line("auto *global="+view+"+("+slabOffset(op,first,view)+");");
    if(!outerStride.empty()) {
      // In a two-axis reduction all complete inner runs have the same pitch.
      line("if("+stride+"==1 && run=="+extent+" && live-k>=2*run && "+outerStride+">=run && "+outerStride+"<=2147483647/sizeof(*global)){");
      line("int count=(live-k)/run,bytes=run*sizeof(*global);");
      if(load)
        line("__memcpy(buffer+k,global,bytes,GDRAM2NRAM,bytes,"+outerStride+"*sizeof(*global),count-1);");
      else
        line("__memcpy(global,buffer+k,bytes,NRAM2GDRAM,"+outerStride+"*sizeof(*global),bytes,count-1);");
      line("k+=count*run;continue;}");
    }
    if(load) {
      line("if("+stride+"==1)__memcpy(buffer+k,global,run*sizeof(*global),GDRAM2NRAM);");
      line("else if("+stride+"==0){auto scalar=*global;__bang_write_value(buffer+k,run,scalar);}");
      line("else for(int t=0;t<run;++t)buffer[k+t]=global[t*"+stride+"];");
    } else{
      line("if("+stride+"==1)__memcpy(global,buffer+k,run*sizeof(*global),NRAM2GDRAM);");
      line("else for(int t=0;t<run;++t)global[t*"+stride+"]=buffer[k+t];");
    }
    line("k+=run;}}");
  }
  void issue(Block &fetch, const Text &iteration, const Text &slot) {
    line("{ long long incoming=" + iteration + ";");
    line(resident ? "long long row0=incoming*R, col=0;"
                  : "long long row0=group*R, col=incoming*W;");
    line("int live=(int)((N-col)<W?(N-col):W);");
    for (auto &op : fetch.without_terminator()) {
      if (op.getName().getStringRef() == "dsa.window" || op.getName().getStringRef()=="dsa.member_axis" || op.getName().getStringRef()=="dsa.slab_axis" || op.getName().getStringRef()=="dsa.lookup_axis")
        continue;
      if(op.hasAttr("dsa.shared"))continue;
      auto src = op.getOperand(0);
      auto n = name(src), dst = name(op.getResult(0)),
           type = ctype(src.getType());
      auto [rowStride, stride] = strides(op, 2, n);
      auto primitive=op.getAttrOfType<StringAttr>("bangc.primitive").getValue();
      if(primitive=="indexed_dma"){
        auto keys=name(op.getOperand(2).getDefiningOp()->getOperand(0));
        auto keyType=ctype(op.getOperand(2).getDefiningOp()->getOperand(0).getType());
        line("{int count=(M-row0<R?M-row0:R);");
        line("if(R>1 && "+keys+"s0==1 && "+n+"s1==1 && "+n+"d0>0 && (unsigned long long)"+n+"s0<=0xffffffffULL/sizeof("+type+")/"+n+"d0){");
        line("auto *ids=("+keyType+"*)(storage+BYTES-((R*12+127)/128*128));");
        line("__memcpy(ids,"+keys+"+row0,count*sizeof(*ids),GDRAM2NRAM);");
        line("for(int j=0;j<count;++j)__memcpy_async("+dst+"+(("+slot+")*R+j)*P,"+n+"+col+ids[j]*"+n+"s0,live*sizeof("+type+"),GDRAM2NRAM);");
        if(depth==1)line("__sync_io();");
        line("}else{for(int j=0;j<count;++j){auto *src="+n+"+"+rowStride+"+col*"+stride+";auto *out="+dst+"+(("+slot+")*R+j)*P;");
        line("if("+stride+"==1)__memcpy(out,src,live*sizeof("+type+"),GDRAM2NRAM);else for(int k=0;k<live;++k)out[k]=src[k*"+stride+"];}}}");
        continue;
      }
      if(primitive=="segmented_slab_read"){
        line("for(int j=0;j<R && row0+j<M;++j)");
        slabTransfer(op,2,n,dst+"+("+slot+"*R+j)*P",true);
        continue;
      }
      bool grouped=primitive.starts_with("group_dma");
      Text move=primitive.ends_with("_async") ? "__memcpy_async" : "__memcpy";
      if(grouped) {
        int axis=cast<MemRefType>(src.getType()).getRank()-2;
        auto dim=n+"d"+std::to_string(axis), outer=n+"s"+std::to_string(axis);
        line("{int j=0, count=(int)((M-row0)<R?(M-row0):R);");
        line("if(count>1 && "+stride+"==1 && "+outer+">=live && "+outer+"<=2147483647/sizeof("+type+") && row0%"+dim+"+count<="+dim+"){");
        line(move+"("+dst+"+("+slot+")*R*P,"+n+"+"+rowStride+"+col,live*sizeof("+type+"),GDRAM2NRAM,P*sizeof("+type+"),"+outer+"*sizeof("+type+"),count-1);");
        line("}else");
      }
      line("for(int j=0;j<R && row0+j<M;++j){");
      line("auto *src=" + n + "+" + rowStride + "+" + "col*" +
           stride + ";");
      line("auto *dst=" + dst + "+((" + slot + ")*R+j)*P;");
      line("if(" + stride + "==1){" +
           move +
           "(dst,src,live*sizeof(" + type +
           "),GDRAM2NRAM);}else if("+stride+"==0){auto scalar=*src;__bang_write_value(dst,live,scalar);}else{for(int k=0;k<live;++k) dst[k]=src[k*" +
           stride + "];}}");
      if(grouped) line("}");
    }
    line("}");
  }
  llvm::Error phase(Operation &op) {
    auto &fetch = op.getRegion(0).front(), &body = op.getRegion(1).front();
    Text carry;
    if (op.getNumResults()) {
      carry = name(op.getOperand(0));
      names[op.getResult(0)] = carry + "[j]";
    }
    line("{");
    unsigned index = 0;
    for (auto &copy : fetch.without_terminator()) {
      if (copy.getName().getStringRef() == "dsa.window" || copy.getName().getStringRef()=="dsa.member_axis" || copy.getName().getStringRef()=="dsa.slab_axis" || copy.getName().getStringRef()=="dsa.lookup_axis")
        continue;
      auto n = fresh();
      auto t = ctype(copy.getResult(0).getType());
      names[copy.getResult(0)] = n;
      names[body.getArgument(index++)] = copy.hasAttr("dsa.shared") ? n : n + "+(slot*R+j)*P";
      line("auto *" + n + "=(" + t + "*)" + address(copy) + ";");
    }
    if (!carry.empty())
      names[body.getArgument(index)] = carry + "[j]";
    auto status = emitSchedule(op.getRegion(2).front(), fetch, body, carry);
    line("}");
    return status;
  }
  llvm::Error emitSchedule(Block &steps, Block &fetch, Block &body,
                           const Text &carry) {
    for (auto &op : steps) {
      auto n = op.getName().stripDialect();
      auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
      if (n == "range") {
        bool tasks =
            op.getAttrOfType<StringAttr>("source") == "participant_stride";
        for (auto [i, expr] : llvm::enumerate(SmallVector<Text>{
                 tasks ? "taskId" : "0", tasks ? "grid" : "((N+W-1)/W)",
                 tasks ? "taskDim" : "1"}))
          names[op.getResult(i)] = expr;
      } else if(n=="load_shared"){
        line("if("+arg(0)+"<"+arg(1)+"){");
        for(auto &copy:fetch)if(copy.hasAttr("dsa.shared")){
          auto src=name(copy.getOperand(0)),dst=name(copy.getResult(0)),type=ctype(copy.getResult(0).getType());
          line("if("+src+"s0==1)__memcpy("+dst+","+src+",N*sizeof("+type+"),GDRAM2NRAM);else for(long long k=0;k<N;++k)"+dst+"[k]="+src+"[k*"+src+"s0];");
        }
        line("}");
      } else if (n == "slot_zero")
        names[op.getResult(0)] = "0";
      else if (n == "iterate") {
        auto &bb = op.getRegion(0).front();
        names[bb.getArgument(0)] = "pos";
        names[bb.getArgument(1)] = "slot";
        line("int slot=" + arg(3) + ";for(long long pos=" + arg(0) + ";pos<" +
             arg(1) + ";pos+=" + arg(2) + "){");
        if (auto e = emitSchedule(bb, fetch, body, carry))
          return e;
        line("}");
      } else if (n == "issue_if_in_bounds") {
        line("if(" + arg(0) + "<" + arg(1) + ")");
        issue(fetch, arg(0), arg(2));
      } else if (n == "next")
        names[op.getResult(0)] = "(" + arg(0) + "+" + arg(1) + ")";
      else if (n == "other_slot")
        names[op.getResult(0)] = "(1-" + arg(0) + ")";
      else if (n == "wait_slot")
        line("__sync();");
      else if (n == "advance")
        line("slot=" + arg(0) + ";");
      else if (n == "compute_members") {
        if (auto e = compute(body, carry, op.getAttrOfType<StringAttr>("bangc.primitive")=="packed_pointwise"))
          return e;
      } else if (n != "end")
        return llvm::createStringError("unlowered supply schedule operation " +
                                       n);
    }
    return llvm::Error::success();
  }
  llvm::Error compute(Block &body, const Text &carry, bool packed) {
    line(resident ? "long long row0=pos*R,col=0;"
                  : "long long row0=group*R,col=pos*W;");
    line("int live=(int)((N-col)<W?(N-col):W);");
    if(packed) {
      line("int members=live==P?(int)((M-row0)<R?(M-row0):R):1;");
      line("for(int j=0;j<R && row0+j<M;j+=members){int lanes=members*((live+63)/64*64);");
    } else line("int lanes=(live+63)/64*64;for(int j=0;j<R && row0+j<M;++j){");
    for (auto &inst : body) {
      auto n = inst.getName().stripDialect();
      auto prim = inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
      auto in = [&](unsigned i) { return name(inst.getOperand(i)); };
      if (n == "await") {
        names[inst.getResult(0)] = "(" + in(0) + ")";
        continue;
      }
      if (n == "window" || n=="member_axis" || n=="slab_axis")
        continue;
      if(n=="owner_read"){
        auto v=fresh(); names[inst.getResult(0)]=v;line("auto "+v+"="+in(0)+"[(row0+j)*"+in(0)+"s0];");
        continue;
      }
      if(n=="owner_write"){
        line(in(0)+"[(row0+j)*"+in(0)+"s0]="+in(2)+";");continue;
      }
      if (n == "complete") {
        if (inst.getNumOperands() > 1)
          line(carry + "[j]=" + in(1) + ";");
        continue;
      }
      if (n == "drain") {
        if (llvm::any_of(inst.getOperands(), [](Value v) {
              auto mode=v.getDefiningOp()->getAttrOfType<BoolAttr>("async"); return mode && mode.getValue();
            }))
          line("__sync();");
        continue;
      }
      if (n == "write_back") {
        auto ptr = in(0), src = in(1), t = ctype(inst.getOperand(0).getType());
        if(prim=="segmented_slab_write"){
          slabTransfer(inst,3,ptr,src,false);continue;
        }
        auto [rowStride, columnStride] = strides(inst, 3, ptr);
        if(packed) {
          auto axis=std::to_string(cast<MemRefType>(inst.getOperand(0).getType()).getRank()-2);
          auto stride=ptr+"s"+axis, dim=ptr+"d"+axis;
          Text write=inst.getAttrOfType<BoolAttr>("async").getValue()?"__memcpy_async":"__memcpy";
          line("{int first=j;auto *first_value="+src+";");
          if (prim=="group_store" || prim=="group_store_async") {
            line("if(members>1 && "+columnStride+"==1 && "+stride+">=live && "+stride+"<=2147483647/sizeof("+t+") && (row0+j)%"+dim+"+members<="+dim+"){");
            line(write+"("+ptr+"+"+rowStride+"+col,first_value,live*sizeof("+t+"),NRAM2GDRAM,"+stride+"*sizeof("+t+"),P*sizeof("+t+"),members-1); }else");
          }
          line("for(int h=0;h<members;++h){int j=first+h;auto *out="+ptr+"+"+rowStride+"+col*"+columnStride+";auto *v=first_value+h*P;");
          line("if("+columnStride+"==1)"+write+"(out,v,live*sizeof("+t+"),NRAM2GDRAM);else for(int k=0;k<live;++k)out[k*"+columnStride+"]=v[k];}}");
          continue;
        }
        line("{auto *out=" + ptr + "+" + rowStride + "+col*" +
             columnStride + ";");
        Text move =
            (prim == "async_store_extent" || prim == "group_store_async") ? "__memcpy_async" : "__memcpy";
        line("if(" + columnStride + "==1)" + move + "(out," + src +
             ",live*sizeof(" + t +
             "),NRAM2GDRAM);else for(int k=0;k<live;++k)out[k*" + columnStride +
             "]=" + src + "[k];}");
        continue;
      }
      auto val = inst.getResult(0);
      Text id = fresh();
      names[val] = id;
      if(prim=="welford_f32"){
        line("Moment "+id+"={0,0,0};for(int k=0;k<live;++k)"+id+"=merge_moments("+id+",Moment{1,"+in(0)+"[k],0});");continue;
      }
      if(prim=="summary_field"){
        auto pos=inst.getAttrOfType<IntegerAttr>("position").getInt();
        if(pos<0 || pos>2)return llvm::createStringError("invalid moment field");
        line("auto "+id+"="+in(0)+(pos==0?".n;":pos==1?".mean;":".m2;"));continue;
      }
      if(prim=="summary_merge"){line("auto "+id+"=merge_moments("+in(0)+","+in(1)+");");continue;}
      if (n == "reduce") {
        Text fn = prim == "row_sum" ? "row_sum" : "row_max_num";
        line("float " + id + "=" + fn + "((float*)" + address(inst) + "," +
             in(0) + ",live);");
        continue;
      }
      bool vec = tile(val);
      auto type = ctype(val.getType());
      if (vec)
        line("auto *" + id + "=(" + type + "*)" + address(inst) + ";");
      if(prim=="row_inclusive_sum"){
        line("tile_prefix("+id+","+in(0)+",("+type+"*)"+address(inst,"bangc.scan_scratch")+",lanes,live);");
        continue;
      }
      if (prim == "positive_half_or_nan") {
        // Signed binary16 encodings -1023..-1 are negative NaNs. Preserve
        // those and every positive encoding; all other negative values map to
        // +0.
        line("__bang_ge_scalar((short*)" + id + ",(const short*)" + in(0) +
             ",(short)-1023,lanes);");
        line("__bang_mul((short*)" + id + ",(const short*)" + in(0) +
             ",(const short*)" + id + ",lanes);");
        continue;
      }
      if (prim == "quotient_for_narrow")
        continue;
      if (prim == "certified_half_quotient") {
        auto *div = inst.getOperand(0).getDefiningOp();
        line("row_narrow_quotient(" + id + "," + in(0) + ",(float*)" +
             address(inst, "bangc.scratch") + "," + name(div->getOperand(0)) +
             "," + name(div->getOperand(1)) + ",live,lanes);");
        continue;
      }
      auto operand = [&](unsigned i) {
        return in(i) + (tile(inst.getOperand(i)) ? "[k]" : "");
      };
      if (prim == "bf16_product" || prim == "bf16_round" ||
          prim == "bf16_widen") {
        if (vec && tile(inst.getOperand(0)) && prim != "bf16_product") {
          if (prim == "bf16_round")
            line("narrow_bf16_tile("+id+","+in(0)+",live,lanes);");
          else
            line("widen_bf16_tile("+id+","+in(0)+",lanes);");
          continue;
        }
        auto a = operand(0);
        auto value = prim == "bf16_product"
                         ? "round_bf16(widen_bf16(" + a + ")*widen_bf16(" +
                               operand(1) + "))"
                     : prim == "bf16_round" ? "round_bf16(" + a + ")"
                                            : "widen_bf16(" + a + ")";
        line((vec ? "for(int k=0;k<live;++k)" + id + "[k]" : type + " " + id) +
             "=" + value + ";");
        continue;
      }
      if (n == "linear_index") {
        line("for(int k=0;k<live;++k)" + id + "[k]=("+type+")((unsigned long long)(row0+j)*N+col+k);");
        continue;
      }
      if (prim=="philox4x32_uniform") {
        line("for(int k=0;k<live;++k)"+id+"[k]=uniform_at("+operand(0)+","+operand(1)+");");
        continue;
      }
      if (n == "constant") {
        auto literal = inst.getAttrOfType<TypedAttr>("value");
        line(type + " " + id + "=(" + type + ")" +
             intentdsl::kir::literal(literal) + ";");
        continue;
      }
      auto a = operand(0);
      Text expr;
      if (n == "cast")
        expr = "(" + type + ")" + a;
      else if (n == "sqrt")
        expr = "sqrtf(" + a + ")";
      else if (n == "rsqrt")
        expr = "(1.0f/sqrtf(" + a + "))";
      else if (n == "exp")
        expr = "expf(" + a + ")";
      else if (n == "select")
        expr = "(" + a + "?" + operand(1) + ":" + operand(2) + ")";
      else if (n == "gt" || n == "and" || n == "xor" || n == "shr" ||
               n == "shl") {
        auto token = n == "gt"    ? ">"
                     : n == "and" ? "&"
                     : n == "xor" ? "^"
                     : n == "shr" ? ">>"
                                  : "<<";
        expr = (n == "shl" ? "(unsigned)" : "") + a + token + operand(1);
      } else if (ctype(val.getType()) == "int" &&
                 (n == "mul" || n == "add" || n == "sub"))
        expr = "(unsigned)" + a +
               (n == "mul"   ? "*"
                : n == "add" ? "+"
                             : "-") +
               "(unsigned)" + operand(1);
      else if (n == "maximum")
        expr = "(isnan((float)" + a + ") || isnan((float)" + operand(1) +
               ") ? NAN : fmaxf(" + a + "," + operand(1) + "))";
      else if (n == "combine")
        expr = prim == "sum" ? a + "+" + operand(1)
                             : "fmaxf(" + a + "," + operand(1) + ")";
      else if (n == "mul" || n == "add" || n == "sub" || n == "div")
        expr = a +
               (n == "mul"   ? "*"
                : n == "add" ? "+"
                : n == "sub" ? "-"
                             : "/") +
               operand(1);
      else
        return llvm::createStringError("unsupported BANG row instruction " + n);
      if (vec && n == "cast" &&
          ((type == "half" && ctype(inst.getOperand(0).getType()) == "float") ||
           (type == "float" &&
            ctype(inst.getOperand(0).getType()) == "half")) &&
          tile(inst.getOperand(0)))
        line(Text(type == "half" ? "__bang_float2half_rn"
                                 : "__bang_half2float") +
             "(" + id + "," + in(0) + ",lanes);");
      else if (vec && (n == "mul" || n == "add" || n == "sub") &&
               tile(inst.getOperand(0)) && tile(inst.getOperand(1)))
        line("__bang_" + n.str() + "(" + id + "," + in(0) + "," + in(1) +
             ",lanes);");
      else if (vec && (n == "mul" || n == "sub" || n == "add") &&
               tile(inst.getOperand(0)) && !tile(inst.getOperand(1)))
        line("__bang_" + Text(n == "sub" ? "add" : n.str()) + "_scalar(" + id +
             "," + in(0) + "," + (n == "sub" ? "-(" : "(") + in(1) +
             "),lanes);");
      else if (vec && n == "exp")
        line("row_exp(" + id + "," + in(0) + ",live,lanes);");
      else
        line((vec ? "for(int k=0;k<live;++k)" + id + "[k]" : type + " " + id) +
             "=" + expr + ";");
    }
    line("}");
    return llvm::Error::success();
  }

public:
  Source(llvm::raw_ostream &os, func::FuncOp function, unsigned d)
      : o(os), fn(function),
        resident(fn->getAttrOfType<StringAttr>("block.topology") == "resident"),
        depth(d) {}
  llvm::Error write() {
    line("#include <bang.h>\n#include <cnrt.h>\n#include <math.h>");
    bool moments=false;
    fn.walk([&](Operation *op){moments |= op->getName().stripDialect()=="summary_reduce";});
    if(moments) o <<
#include "Moments.inc"
    ;
    bool random=false;
    fn.walk([&](Operation *op){random |= op->getName().getStringRef()=="block.uniform";});
    if (random) o <<
#include "Philox.inc"
      ;
    o <<
#include "RowMath.inc"
        ;
    line("constexpr int R=INTENT_TILE_R, W=INTENT_TILE_W, P=INTENT_TILE_P, "
         "BYTES=INTENT_TILE_BYTES;");
    line("__mlu_global__ void tile(" + args(true) +
         "){ __nram__ unsigned char storage[BYTES] "
         "__attribute__((aligned(128)));");
    return emitKernel();
  }
  llvm::Error outer(Block &block) {
    for (auto &op : block) {
      auto code = op.getName().stripDialect();
      if (code == "range") {
        names[op.getResult(0)] = "taskId";
        names[op.getResult(1)] = "grid";
        names[op.getResult(2)] = "taskDim";
      } else if (code == "task_groups") {
        line("for(long long group=" + name(op.getOperand(0)) + ";group<" +
             name(op.getOperand(1)) + ";group+=" + name(op.getOperand(2)) +
             "){");
        if (auto e = outer(op.getRegion(0).front()))
          return e;
        line("}");
      } else if (code == "phase") {
        if (auto e = phase(op))
          return e;
      } else if(code=="owner_rows"){
        names[op.getResult(0)]="(group*R+j)";
      } else if(code=="owner_read" || code=="owner_write"){
        auto base=name(op.getOperand(0)), row=name(op.getOperand(1));
        auto address=base+"["+row+"*"+base+"s0]";
        Text lanes="for(int j=0;j<R && group*R+j<M;++j)";
        if(code=="owner_write")line(lanes+address+"="+name(op.getOperand(2))+";");
        else{
          auto v=fresh();names[op.getResult(0)]=v+"[j]";
          line(ctype(op.getResult(0).getType())+" "+v+"[R];"+lanes+v+"[j]="+address+";");
        }
      } else if(code=="drain"){
        // Owner stores are blocking scalar writes. Their completions need no native wait.
        continue;
      } else if(code=="summary_identity"){
        auto n=fresh();names[op.getResult(0)]=n;
        line("Moment "+n+"[R]={};");
      } else if (code == "identity") {
        auto n = fresh();
        names[op.getResult(0)] = n;
        line("float " + n + "[R];for(int j=0;j<R;++j)" + n + "[j]=" +
             (op.getAttrOfType<StringAttr>("algebra") == "sum" ? "0.0f"
                                                               : "-INFINITY") +
             ";");
      } else if (code=="cast" || code=="summary_field" || code=="div" || code=="mul" ||
                 code=="add" || code=="sub" || code=="rsqrt" || code=="constant"){
        auto n=fresh();auto t=ctype(op.getResult(0).getType());
        auto input=[&](unsigned i){return name(op.getOperand(i));};
        bool perOwner=isa<RankedTensorType>(op.getResult(0).getType());
        Text expr;
        if(code=="constant")expr=intentdsl::kir::literal(op.getAttrOfType<TypedAttr>("value"));
        else if(code=="cast")expr="("+t+")("+input(0)+")";
        else if(code=="rsqrt")expr="1.0f/sqrtf("+input(0)+")";
        else if(code=="summary_field"){
          auto pos=op.getAttrOfType<IntegerAttr>("position").getInt();
          if(pos<0 || pos>=3)return llvm::createStringError("invalid streamed summary field");
          expr=input(0)+(pos==0?".n":pos==1?".mean":".m2");
        } else expr="("+input(0)+")"+(code=="add"?"+":code=="sub"?"-":code=="div"?"/":"*")+"("+input(1)+")";
        names[op.getResult(0)]=perOwner?n+"[j]":n;
        line(perOwner?t+" "+n+"[R];for(int j=0;j<R && group*R+j<M;++j)"+n+"[j]="+expr+";":t+" "+n+"="+expr+";");
      } else if (code != "return" && code != "end")
        return llvm::createStringError("unexpected native outer op " + code);
    }
    return llvm::Error::success();
  }
  llvm::Error emitKernel() {
    if (auto e = outer(fn.front()))
      return e;
    line("}");
    line("extern \"C\" int intent_enqueue(" + args(true) +
         ",int cores,void *queue){");
    line("if(grid==0 || N==0)return 0; cnrtDim3_t "
         "dim={(unsigned)(grid<cores?grid:cores),1,1};");
    line("tile<<<dim,cnrtFuncTypeBlock,(cnrtQueue_t)queue>>>(" + args(false) +
         ");return (int)cnrtGetLastError();}");
    return llvm::Error::success();
  }
};
} // namespace
llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
  std::string program;
  llvm::raw_string_ostream text(program);
  auto depth = layout.getAs<IntegerAttr>("depth").getInt();
  if (auto e = Source(text, fn, depth).write())
    return e;
  llvm::json::Array rows;
  for (auto a : layout.getAs<ArrayAttr>("rows")) {
    auto allocation = cast<DictionaryAttr>(a);
    auto id = allocation.getAs<IntegerAttr>("id");
    for (auto attr : fn->getAttrOfType<ArrayAttr>("block.configs")) {
      auto row = cast<DictionaryAttr>(attr);
      if (row.get("id") != id)
        continue;
      unsigned tasks;
      if (row.getAs<StringAttr>("payload").getValue().getAsInteger(10, tasks))
        return llvm::createStringError("invalid physical task count");
      rows.push_back(llvm::json::Object{
          {"id", id.getInt()},
          {"rows", row.getAs<IntegerAttr>("rows").getInt()},
          {"width", row.getAs<IntegerAttr>("width").getInt()},
          {"bytes", allocation.getAs<IntegerAttr>("bytes").getInt()},
          {"pitch", allocation.getAs<IntegerAttr>("pitch").getInt()},
          {"tasks", layout.getAs<IntegerAttr>("participants").getInt()}});
    }
  }
  auto abi =
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if (!abi)
    return abi.takeError();
  auto guard = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
  llvm::json::Array axes;
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.grid"))
    axes.push_back(cast<StringAttr>(a).str());
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", program},
      {"domain",intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"rows", std::move(rows)},
      {"depth", depth},
      {"grid", llvm::json::Array{std::move(axes), 1, 1}},
      {"guard",
       llvm::json::Object{{"min_width", guard[0]},
                          {"max_width", guard[1] < 0
                                            ? llvm::json::Value(nullptr)
                                            : llvm::json::Value(guard[1])}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
