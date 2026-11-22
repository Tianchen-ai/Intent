#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
namespace intentdsl::kir {
bool matrixDomain(func::FuncOp fn) {
  return fn && !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().stripDialect()=="matrix_domain";
}

LogicalResult verifyMatrix(func::FuncOp fn,StringRef dialect,bool canonical){
  auto &domain=fn.front().front();
  if(!llvm::hasSingleElement(fn.getBody()) || fn.getNumResults() || fn.front().getOperations().size()!=2 ||
     domain.getName().getDialectNamespace()!=dialect || !llvm::hasSingleElement(domain.getRegion(0)))
    return fn.emitError("expected a single destination contraction region");
  auto abi=fn->getAttrOfType<StringAttr>("kir.abi");
  if(!abi)return fn.emitError("missing contraction signature");
  auto parsed=llvm::json::parse(abi.getValue());
  if(!parsed){llvm::consumeError(parsed.takeError());return fn.emitError("bad contraction ABI");}
  auto *args=parsed->getAsArray();
  if(!args || args->size()!=fn.getNumArguments())return fn.emitError("matrix ABI arity");
  using Shape=std::vector<llvm::json::Value>;
  llvm::DenseMap<Value,Shape> shapes;
  Value output=domain.getOperand(0); unsigned outputs=0;
  for(auto [i,v]:llvm::enumerate(fn.getArguments())){
    auto *s=(*args)[i].getAsObject();auto type=dyn_cast<MemRefType>(v.getType());
    if(!s || !type || !s->getArray("dims") || s->getArray("dims")->size()!=size_t(type.getRank()))
      return fn.emitError("contraction slot rank mismatch");
    auto et=type.getElementType();auto dtype=s->getString("dtype");
    if(!(et.isBF16() && dtype=="bf16") && !(et.isF16() && dtype=="f16") &&
       !(et.isF32() && dtype=="f32") && !(et.isInteger(8) && (dtype=="u8" || dtype=="i8")) &&
       !(isa<Float8E4M3FNType>(et) && dtype=="f8e4m3fn") && !(et.isInteger(32) && dtype=="i32"))
      return fn.emitError("invalid contraction carrier");
    if(s->getString("direction")!=(v==output?"Out":"In"))return fn.emitError("invalid matrix access direction");
    if(v==output){++outputs;if(canonical && fn->getAttr("kir.domain_arg")!=IntegerAttr::get(IntegerType::get(fn.getContext(),64),i))
      return fn.emitError("matrix domain must be its output");}
    auto *dims=s->getArray("dims");
    for(auto &axis:*dims){
      if(!axis.getAsString() && (!axis.getAsInteger() || *axis.getAsInteger()<0))return fn.emitError("invalid matrix dimension");
      shapes[v].push_back(axis);
    }
  }
  if(outputs!=1)return fn.emitError("missing output buffer");
  auto &body=domain.getRegion(0).front();unsigned dots=0,stores=0;
  auto dtype=[&](Value v){return (*args)[cast<BlockArgument>(v).getArgNumber()].getAsObject()->getString("dtype");};
  auto elem=[](Value v){return cast<ShapedType>(v.getType()).getElementType();};
  for(auto &op:body){
    if(op.getName().getDialectNamespace()!=dialect)return op.emitError("foreign matrix expression");
    auto name=op.getName().stripDialect();
    if(name=="end")continue;
    if(stores)return op.emitError("operation follows matrix publication");
    for(auto v:op.getOperands())if(!shapes.count(v) || (v==output && name!="publish_matrix"))
      return op.emitError("unexpected matrix value or read from output");
    auto a=op.getOperand(0);auto A=shapes.lookup(a);Shape shape;Type type;
    if(name=="contract"){
      auto b=op.getOperand(1);auto B=shapes.lookup(b);
      auto index=[&](StringRef key){return op.getAttrOfType<IntegerAttr>(key).getInt();};
      auto ak=index("lhs_k"),bk=index("rhs_k"),ab=index("lhs_batch"),bb=index("rhs_batch");
      auto valid=[](int64_t k,int64_t batch,size_t rank){
        return rank==(batch<0?2u:3u) && k>=0 && k<int64_t(rank) &&
               batch>=-1 && batch<int64_t(rank) && k!=batch;
      };
      if(!valid(ak,ab,A.size()) || !valid(bk,bb,B.size()) || A[ak]!=B[bk] ||
         (ab>=0 && bb>=0 && A[ab]!=B[bb]) || dtype(a)!=dtype(b) ||
         (dtype(a)!="i8" && dtype(a)!="bf16" && dtype(a)!="f16" && dtype(a)!="f32"))
        return op.emitError("incompatible contraction axes or input types");
      auto free=[](const Shape &s,int64_t k,int64_t batch){for(size_t i=0;i<s.size();++i)
        if(int64_t(i)!=k && int64_t(i)!=batch)return s[i];llvm_unreachable("verified matrix rank");};
      if(op.getNumOperands()==2){
        if(ab>=0 || bb>=0)shape.push_back(ab>=0?A[ab]:B[bb]);
      }else{
        if(ab!=-1 || bb<0 || op.getNumOperands()>4)return op.emitError("segments require a bank of right matrices");
        for(auto v:op.getOperands().drop_front(2))if(shapes.lookup(v).size()!=1 || dtype(v)!="i32")
          return op.emitError("offsets and routes must be i32 vectors");
      }
      shape.push_back(free(A,ak,ab));
      if(op.getNumOperands()==4){
        auto declared=shapes.lookup(output);
        if(declared.size()!=3 || declared[1].getAsInteger().value_or(0)<=0)return op.emitError("routed output needs a positive route multiplicity");
        shape.push_back(declared[1]);
      }
      shape.push_back(free(B,bk,bb));type=dtype(a)=="i8"?Type(IntegerType::get(fn.getContext(),32)):Type(Float32Type::get(fn.getContext()));++dots;
    }else if(name=="transpose_scale"){
      if(A.size()!=2 || dtype(a)!="u8")return op.emitError("expected a two-axis E8M0 scale view");
      shape={A[1],A[0]};type=elem(a);
    }else if(name=="scaled_contract"){
      auto B=shapes.lookup(op.getOperand(2));
      if(A.size()!=3 || B.size()!=3 || A[2].getAsInteger()!=32 || B[1]!=A[2] || A[1]!=B[0] ||
         dtype(a)!="f8e4m3fn" || dtype(op.getOperand(2))!="f8e4m3fn" || dtype(op.getOperand(1))!="u8" ||
         shapes.lookup(op.getOperand(1))!=Shape{A[0],A[1]} || shapes.lookup(op.getOperand(3))!=Shape{B[2],A[1]} ||
         !op.getOperand(3).getDefiningOp() || op.getOperand(3).getDefiningOp()->getName().stripDialect()!="transpose_scale")
        return op.emitError("expected E4M3FN [M,G,32] x [G,32,N] and E8M0 [M,G], [N,G]");
      shape={A[0],B[2]};type=Float32Type::get(fn.getContext());++dots;
    }else if(name=="quantize_q8k"){
      if(A.size()!=2 || A[1].getAsInteger()!=256 || !elem(a).isF32())return op.emitError("Q8_K input must be f32[G,256]");
      shape={A[0],292};type=IntegerType::get(fn.getContext(),8);
    }else if(name=="q4k_dot"){
      auto *encoder=op.getOperand(1).getDefiningOp();
      if(A.size()!=3 || A[2].getAsInteger()!=144 || dtype(a)!="u8" || !encoder ||
         encoder->getName().stripDialect()!="quantize_q8k" || shapes.lookup(op.getOperand(1))!=Shape{A[1],292})
        return op.emitError("expected Q4_K[N,G,144] and Q8_K[G,292]");
      shape={A[0]};type=Float32Type::get(fn.getContext());++dots;
    }else if(name=="add_columns"){
      if(A.empty() || shapes.lookup(op.getOperand(1))!=Shape{A.back()} || elem(a)!=elem(op.getOperand(1)))return op.emitError("incompatible column bias");
      shape=A;type=elem(a);
    }else if(name=="convert_matrix"){shape=A;type=elem(op.getResult(0));}
    else if(name=="publish_matrix"){
      if(a!=output || shapes.lookup(op.getOperand(1))!=A || elem(op.getOperand(1))!=elem(a))
        return op.emitError("result shape or element type disagrees with destination");
      ++stores;continue;
    }else return op.emitError("unsupported matrix operation");
    auto result=op.getResult(0);auto rt=cast<RankedTensorType>(result.getType());
    if(rt.getRank()!=int64_t(shape.size()) || rt.getElementType()!=type)return op.emitError("matrix result type mismatch");
    for(auto [i,d]:llvm::enumerate(rt.getShape()))
      if(!ShapedType::isDynamic(d) && shape[i].getAsInteger()!=d)return op.emitError("static matrix extent mismatch");
    shapes[result]=std::move(shape);
  }
  if(dots!=1 || stores!=1 || body.empty() || body.back().getName().stripDialect()!="end")
    return domain.emitError("expected one complete contraction and publication");
  return success();
}
}
