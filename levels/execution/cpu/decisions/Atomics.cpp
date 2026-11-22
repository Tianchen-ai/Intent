#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerAtomics(Unit &unit){
  Strategy s;s.name="keyed-atomics";s.family=unit.qualify("partition");
  s.enumerate=[](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<std::vector<StructuralCandidate>>{
    std::vector<StructuralCandidate> candidates;
    if(op->walk([](Operation *v){auto n=v->getName().stripDialect();return n=="atomic_add" || n=="fill_ranges"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted()){
      auto *m=p.services.get<intentdsl::cpu::RowMachine>();if(!m)return llvm::createStringError("missing CPU task resources");
      Builder b(op->getContext());auto base=intentdsl::cpu::partition(b,"source",m->configurations(op->getContext(),false));
      candidates.push_back({"source","source",base,{"row-independent"}});
      if(op->walk([](Operation *v){return v->getName().stripDialect()=="atomic_add"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted()){
        NamedAttrList d(base);d.set("single_dispatcher",b.getUnitAttr());
        candidates.push_back({"serial-tickets","serial-tickets",d.getDictionary(op->getContext()),{"relaxed-keyed-updates"}});
        Operation *atomic=nullptr;unsigned updates=0;
        op->walk([&](Operation *v){if(v->getName().stripDialect()=="atomic_add"){atomic=v;++updates;}});
        auto *facts=p.services.get<intentdsl::CallFacts>();
        if(updates==1 && facts && llvm::all_of(op->getRegion(0).front(),[](Operation &v){
          return llvm::is_contained(ArrayRef<StringRef>{"read","atomic_add","unique_put","index_read","add","ordinal","constant","end"},v.getName().stripDialect());})){
          auto slot=cast<BlockArgument>(atomic->getOperand(0)).getArgNumber();
          auto *shape=facts->arguments[slot].getAsObject()->getArray("shape");
          if(!shape || shape->size()!=1 || !shape->front().getAsInteger())return candidates;
          // Distinct logical shapes rule out the identical-view exception of
          // the product's InOut alias contract. Partial overlap is rejected.
          bool distinct=llvm::all_of(llvm::enumerate(facts->arguments),[&](auto item){
            auto *other=item.value().getAsObject();return item.index()==slot || !other || *other->getArray("shape")!=*shape;});
          auto pattern=intentdsl::cpu::matchReservations(op->getRegion(0).front());
          if(*shape->front().getAsInteger()>16 && distinct && pattern.update){
            NamedAttrList byKey(base);byKey.set("key_extent",b.getI64IntegerAttr(*shape->front().getAsInteger()));
            candidates.push_back({"key-owners","key-owners",byKey.getDictionary(op->getContext()),{"disjoint-key-ranges"}});
            auto count=*shape->front().getAsInteger();
            bool aliasFree=true;
            for(size_t i=0;aliasFree && i<facts->arguments.size();++i)for(size_t j=0;j<i;++j){
              auto a=facts->arguments[i].getAsObject(),b=facts->arguments[j].getAsObject();
              if(a && b && *a->getArray("shape")==*b->getArray("shape"))aliasFree=false;
            }
            if(count<=1024 && aliasFree){
              NamedAttrList batch(base);batch.set("batch_extent",b.getI64IntegerAttr(count));
              candidates.push_back({"range-reservations","range-reservations",batch.getDictionary(op->getContext()),{"independent-relaxed-unit-tickets"}});
            }
          }
        }
      }
    }
    return candidates;
  };
  unit.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
}
