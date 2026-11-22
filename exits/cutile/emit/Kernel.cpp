#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "llvm/ADT/StringExtras.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
class Kernel {
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> names;
  unsigned sequence = 0;
  std::string indexType;
  std::string local() { return "v" + std::to_string(sequence++); }
  void line(int depth, const std::string &s) {
    out.indent(depth * 4) << s << "\n";
  }
  llvm::Error body(Block &block, int depth) {
    for (auto &op : block) {
      auto code = op.getName().stripDialect();
      auto arg = [&](unsigned n) { return names.lookup(op.getOperand(n)); };
      auto algebra = op.getAttrOfType<StringAttr>("algebra");
      auto owner=op.getParentOfType<func::FuncOp>()->getAttrOfType<IntegerAttr>("kir.owner_axis");
      auto memberCount=[&](Value view,const std::string &name){
        std::string span="1";
        for(int i=0;i<cast<MemRefType>(view.getType()).getRank();++i)
          if(i!=owner.getInt())span+="*"+name+".shape["+std::to_string(i)+"]";
        return span;
      };
      bool sum = algebra && algebra == "sum";
      if(code=="output_partition"){
        line(depth,"if ct.bid(1) == "+std::to_string(op.getAttrOfType<IntegerAttr>("index").getInt())+":");
        if(auto err=body(op.getRegion(0).front(),depth+1))return err;
        continue;
      }
      if (code == "return" || code=="partition_end")
        continue;
      if (code == "yield") {
        if (op.getNumOperands())
          line(depth, names.lookup(block.getArgument(1)) + " = " + arg(0));
        continue;
      }
      if (code == "tiles") {
        auto &inner = op.getRegion(0).front();
        auto i = local();
        names[inner.getArgument(0)] = i;
        if (op.getNumResults()) {
          auto carry = local();
          names[inner.getArgument(1)] = names[op.getResult(0)] = carry;
          line(depth, carry + " = " + arg(1));
        }
        line(depth,
             "for " + i + " in range(ct.cdiv(" + (owner?memberCount(op.getOperand(0),arg(0)):arg(0)+".shape[-1]") + ", W)):");
        if (auto err = body(inner, depth + 1))
          return err;
        continue;
      }
      std::string expr;
      if(code=="owner_load") expr="ct.gather("+arg(0)+","+arg(1)+")";
      else if(code=="owner_store"){
        line(depth,"ct.scatter("+arg(0)+","+arg(1)+","+arg(2)+")");
        continue;
      } else if(code=="gather_slab" || code=="scatter_slab"){
        auto base=arg(0), col="ct.expand_dims("+arg(2)+",0)";
        if(op.hasAttr("ct_src.rectangular")){
          auto fn=op.getParentOfType<func::FuncOp>();
          auto slot=cast<BlockArgument>(op.getOperand(0)).getArgNumber();
          int64_t extent=fn.getArgAttrOfType<DenseI64ArrayAttr>(slot,"ct_src.shape")[2];
          auto row=names.lookup(op.getOperand(1).getDefiningOp()->getOperand(0));
          auto tile=names.lookup(op.getOperand(2).getDefiningOp()->getOperand(0));
          auto inner="min(W,"+std::to_string(extent)+")";
          auto segments="(W//"+inner+")";
          auto major="("+tile+"//("+std::to_string(extent)+"//"+inner+"))";
          auto minor="("+tile+"%("+std::to_string(extent)+"//"+inner+"))";
          auto shape=owner.getInt()==0?"(R,"+segments+","+inner+")":"("+segments+",R,"+inner+")";
          auto coord=owner.getInt()==0?"("+row+","+major+","+minor+")":"("+major+","+row+","+minor+")";
          if(code=="scatter_slab"){
            auto v=arg(4);
            auto shaped="ct.reshape("+v+",(R,"+segments+","+inner+"))";
            if(owner.getInt()!=0)shaped="ct.permute("+shaped+",(1,0,2))";
            line(depth,"ct.store("+base+","+coord+","+shaped+",allow_tma=False)");
          }else{
            auto v="ct.load("+base+","+coord+",shape="+shape+",padding_mode=ct.PaddingMode.ZERO,allow_tma=False)";
            if(owner.getInt()!=0)v="ct.permute("+v+",(1,0,2))";
            auto id=local();line(depth,id+" = ct.reshape("+v+",(R,W))");names[op.getResult(0)]=id;
          }
          continue;
        }
        int rank=cast<MemRefType>(op.getOperand(0).getType()).getRank();
        SmallVector<std::string> coords(rank);
        for(int i=rank-1;i>=0;--i){
          auto size=base+".shape["+std::to_string(i)+"]";
          if(i==owner.getInt()) coords[i]="ct.expand_dims("+arg(1)+",1)";
          else {coords[i]="("+col+")%"+size;col="(("+col+")//"+size+")";}
        }
        auto address=base+",("+llvm::join(coords,",")+")";
        if(code=="scatter_slab"){line(depth,"ct.scatter("+address+","+arg(4)+",mask="+arg(3)+")");continue;}
        expr="ct.gather("+address+",mask="+arg(3)+")";
      } else if(code=="active" && owner){
        expr="(ct.expand_dims("+arg(1)+",1)<"+arg(0)+".shape["+std::to_string(owner.getInt())+"]) & (ct.expand_dims("+arg(2)+",0)<"+memberCount(op.getOperand(0),arg(0))+")";
      } else if (code=="summary_reduce")
        expr="ct.reduce((ct.astype("+arg(1)+",ct.int32),ct.where("+arg(1)+","+arg(0)+",0.),ct.full((R,W),0.,ct.float32)),1,merge_moments,(0,0.,0.))";
      else if (code=="summary_identity")
        expr="(ct.full((R,),0,ct.int32),ct.full((R,),0.,ct.float32),ct.full((R,),0.,ct.float32))";
      else if (code=="summary_merge")
        expr="merge_moments("+arg(0)+"[0],"+arg(0)+"[1],"+arg(0)+"[2],"+arg(1)+"[0],"+arg(1)+"[1],"+arg(1)+"[2])";
      else if (code=="summary_field") expr=arg(0)+"["+std::to_string(op.getAttrOfType<IntegerAttr>("position").getInt())+"]";
      else if(code=="indexed_tile_read"){
        auto key="ct.gather("+arg(1)+","+arg(2)+",mask="+arg(2)+"<"+arg(1)+".shape[0])";
        expr="ct.gather("+arg(0)+",(ct.expand_dims("+key+",1),ct.expand_dims("+arg(3)+",0)),mask="+arg(4)+")";
      }
      else if(code=="scan_sum")expr="ct.cumsum("+arg(0)+",axis=1)";
      else if (code == "uniform")
        expr = "uniform_at("+arg(0)+","+arg(1)+")";
      else if (code == "linear_index") {
        bool wide = cast<RankedTensorType>(op.getResult(0).getType())
                        .getElementType()
                        .isUnsignedInteger(64);
        std::string row = wide ? "ct.astype(" + arg(1) + ",ct.uint64)" : arg(1);
        auto width=arg(0)+".shape[1]", column=arg(2);
        if (wide) {
          width="ct.astype("+width+",ct.uint64)";
          column="ct.astype("+column+",ct.uint64)";
        }
        expr = "ct.astype(ct.expand_dims(" + row + ",1)*" + width +
               "+ct.expand_dims(" + column + ",0)," + (wide ? "ct.uint64)" : "ct.int32)");
      } else if (code == "select")
        expr = "ct.where(" + arg(0) + "," + arg(1) + "," + arg(2) + ")";
      else if (code == "xor" || code == "and" || code == "shr" ||
               code == "shl" || code == "gt") {
        auto token = code == "xor"   ? " ^ "
                     : code == "and" ? " & "
                     : code == "shl" ? " << "
                     : code == "shr" ? " >> "
                                     : " > ";
        expr = arg(0) + token + arg(1);
      } else if (code == "constant")
        expr = intentdsl::kir::literal(op.getAttrOfType<TypedAttr>("value"));
      else if (code == "maximum")
        expr = "ct.maximum(" + arg(0) + "," + arg(1) + ",propagate_nan=True)";
      else if (code == "pid")
        { expr = "ct.astype(ct.bid(0), " + indexType + ")";
          auto fn=op.getParentOfType<func::FuncOp>(); Builder b(fn.getContext());
          if(fn->getAttr("block.grid")==intentdsl::block::grid(b,true)) {
            expr += op.getAttrOfType<IntegerAttr>("axis").getInt()==0 ? " // COLUMN_TILES" : " % COLUMN_TILES";
          }
        }
      else if (code == "zero")
        expr = "0";
      else if (code == "rows")
        expr = arg(0) + " * R + ct.arange(R, dtype=" + indexType + ")";
      else if (code == "columns")
        expr = "ct.astype(" + arg(0) + ", " + indexType +
               ") * W + ct.arange(W, dtype=" + indexType + ")";
      else if (code == "active" &&
               cast<MemRefType>(op.getOperand(0).getType()).getRank() == 1)
        expr = "(ct.expand_dims(" + arg(1) + ",1)<1) & (ct.expand_dims(" +
               arg(2) + ",0)<" + arg(0) + ".shape[0])";
      else if (code=="active" && cast<MemRefType>(op.getOperand(0).getType()).getRank()>2){
        auto a=arg(0);auto rank=cast<MemRefType>(op.getOperand(0).getType()).getRank();
        std::string count="1";for(int i=0;i<rank-1;++i)count+="*"+a+".shape["+std::to_string(i)+"]";
        expr="(ct.expand_dims("+arg(1)+",1)<"+count+") & (ct.expand_dims("+arg(2)+",0)<"+a+".shape[-1])";
      } else if (code == "active")
        expr = "(ct.expand_dims(" + arg(1) + ", 1) < " + arg(0) +
               ".shape[0]) & (ct.expand_dims(" + arg(2) + ", 0) < " + arg(0) +
               ".shape[1])";
      else if (code == "tile_load" || code == "tile_store") {
        int rank=cast<MemRefType>(op.getOperand(0).getType()).getRank();
        std::string hint = op.getAttrOfType<BoolAttr>("allow_tma").getValue()
                               ? ""
                               : ", allow_tma=False";
        if(rank>2){
          auto array=arg(0), row=arg(1);
          SmallVector<std::string> indices(rank),shape(rank,"1");
          indices.back()=arg(2);shape.back()="W";shape[rank-2]="R";
          for(int j=rank-2;j>=0;--j){
            auto span="D"+std::to_string(j);
            if(j==rank-2) span="("+span+"//R)";
            indices[j]="("+row+" % "+span+")";
            row="("+row+" // "+span+")";
          }
          auto coord="("+llvm::join(indices,",")+")", tile="("+llvm::join(shape,",")+")";
          if(code=="tile_store")line(depth,"ct.store("+array+","+coord+",ct.reshape("+arg(3)+","+tile+")"+hint+")");
          else {
            auto id=local();names[op.getResult(0)]=id;
            line(depth,id+" = ct.reshape(ct.load("+array+","+coord+",shape="+tile+",padding_mode=ct.PaddingMode.ZERO"+hint+"),(R,W))");
          }
          continue;
        }
        bool matrix = op.getAttrOfType<BoolAttr>("matrix").getValue();
        std::string index =
            matrix ? "(" + arg(1) + ", " + arg(2) + ")" : "(" + arg(2) + ",)";
        if (code == "tile_store") {
          auto axes = op.getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
          bool swap = axes && axes.size() == 2 && axes[0] == 1;
          if (swap)
            index = "(" + arg(2) + "," + arg(1) + ")";
          line(depth, "ct.store(" + arg(0) + ", " + index + ", " +
                          (swap      ? "ct.transpose(" + arg(3) + ")"
                           : !matrix ? "ct.reshape(" + arg(3) + ",(W,))"
                                     : arg(3)) +
                          hint + ")");
          continue;
        }
        expr = "ct.load(" + arg(0) + ", " + index +
               ", shape=" + (matrix ? "(R,W)" : "(W,)") +
               ", padding_mode=ct.PaddingMode.ZERO" + hint + ")";
        if (!matrix)
          expr = "ct.broadcast_to(ct.expand_dims(" + expr + ",0),(R,W))";
      } else if (code == "reduce")
        expr = std::string(sum ? "ct.sum" : "ct.max") + "(ct.where(" + arg(1) +
               ", " + arg(0) + ", " + (sum ? "0.0" : "-float('inf')") +
               "), axis=1)";
      else if (code == "identity")
        expr = std::string("ct.full((R,), ") + (sum ? "0.0" : "-float('inf')") +
               ", ct.float32)";
      else if (code == "combine")
        expr = sum ? arg(0) + " + " + arg(1)
                   : "ct.maximum(" + arg(0) + ", " + arg(1) + ")";
      else if (code == "broadcast_rows")
        expr = "ct.expand_dims(" + arg(0) + ",1)";
      else if (code == "mul" || code == "sub" || code == "add")
        expr = arg(0) +
               (code == "mul"   ? " * "
                : code == "sub" ? " - "
                                : " + ") +
               arg(1);
      else if (code == "div")
        expr = "ct.truediv(" + arg(0) + ", " + arg(1) +
               ", rounding_mode=ct.RoundingMode.RN)";
      else if (code == "sqrt" || code == "exp" || code == "rsqrt")
        expr = "ct." + code.str() + "(" + arg(0) + ")";
      else if (code == "cast") {
        auto t = op.getResult(0).getType();
        if (auto f = dyn_cast<ShapedType>(t))
          t = f.getElementType();
        if (!t.isF32() && !t.isF16() && !t.isBF16() && !t.isInteger(32))
          return llvm::createStringError("unsupported cuTile conversion");
        expr = "ct.astype(" + arg(0) + ", ct." +
               std::string(t.isInteger(32) ? "int32"
                           : t.isBF16()    ? "bfloat16"
                           : t.isF16()     ? "float16"
                                           : "float32") +
               ")";
      } else
        return llvm::createStringError("unexpected cuTile operation " + code);
      auto id = local();
      names[op.getResult(0)] = id;
      line(depth, id + " = " + expr);
    }
    return llvm::Error::success();
  }

public:
  explicit Kernel(llvm::raw_ostream &out) : out(out) {}
  llvm::Error write(func::FuncOp fn) {
    bool wide =
        fn->getAttrOfType<IntegerAttr>("ct_src.index_bits").getInt() == 64;
    indexType = wide ? "ct.int64" : "ct.int32";
    out << "import cuda.tile as ct\n";
    bool moments=false;
    fn.walk([&](Operation *op){moments |= op->getName().stripDialect()=="summary_reduce";});
    if (moments) out <<
#include "Moments.inc"
    ;
    bool random = false;
    fn.walk([&](Operation *op){random |= op->getName().getStringRef()=="ct_src.uniform";});
    if (random) out <<
#include "Philox.inc"
    ;
    out << "\n@ct.kernel\ndef kernel(";
    for (auto a : fn.getArguments()) {
      auto n = "a" + std::to_string(a.getArgNumber());
      names[a] = n;
      out << n;
      if (isa<MemRefType>(a.getType()))
        out << (wide ? ": ct.IndexedWithInt64[ct.Array]" : ": ct.Array");
      out << ", ";
    }
    auto domain=fn.getArgument(intentdsl::domainSlot(fn));
    int rank=cast<MemRefType>(domain.getType()).getRank();
    if(rank>2)
      for(int axis=0;axis<rank-1;++axis) out<<"D"<<axis<<": ct.Constant[int], ";
    Builder b(fn.getContext());
    if(fn->getAttr("block.grid")==intentdsl::block::grid(b,true))
      out << "COLUMN_TILES: ct.Constant[int], ";
    out << "R: ct.Constant[int], W: ct.Constant[int]):\n";
    return body(fn.front(), 1);
  }
};
} // namespace
llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  llvm::json::Array rows;
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto d = cast<DictionaryAttr>(a);
    auto hints = llvm::json::parse(d.getAs<StringAttr>("payload").getValue());
    if (!hints)
      return hints.takeError();
    rows.push_back(
        llvm::json::Object{{"id", d.getAs<IntegerAttr>("id").getInt()},
                           {"rows", d.getAs<IntegerAttr>("rows").getInt()},
                           {"width", d.getAs<IntegerAttr>("width").getInt()},
                           {"native", std::move(*hints)}});
  }
  Builder b(fn.getContext());
  auto formula=intentdsl::block::gridExpression(fn->getAttr("block.grid"));
  auto guard = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
  std::string src;
  llvm::raw_string_ostream stream(src);
  if (auto err = Kernel(stream).write(fn))
    return err;
  out << llvm::json::Value(llvm::json::Object{
      {"domain",intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"version", 3},
      {"column_tiles_parameter",fn->getAttr("block.grid")==intentdsl::block::grid(b,true)},
      {"outer_shape_slot",cast<MemRefType>(fn.getArgument(intentdsl::domainSlot(fn)).getType()).getRank()>2 ? intentdsl::domainSlot(fn) : -1},
      {"source", src},
      {"entry", fn->getAttrOfType<StringAttr>("ct_src.entry").str()},
      {"grid", llvm::json::Array{std::move(formula),std::max<int64_t>(1,llvm::count_if(fn.front(),[](Operation &op){return op.getName().getStringRef()=="ct_src.output_partition";})),1}},
      {"rows", std::move(rows)},
      {"guard",
       llvm::json::Object{{"min_width", guard[0]},
                          {"max_width", guard[1] < 0
                                            ? llvm::json::Value(nullptr)
                                            : llvm::json::Value(guard[1])}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
