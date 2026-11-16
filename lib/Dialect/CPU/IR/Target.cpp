#include "Intent/Dialect/CPU/IR/Target.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace intent::cpu {
llvm::json::Object serializeCapabilities(CapabilitiesAttr target) {
  auto spelling = [](Type type) {
    std::string value;
    llvm::raw_string_ostream stream(value);
    type.print(stream);
    return value;
  };
  llvm::json::Array contractions;
  for (auto item : target.getContractions()) {
    auto capability = cast<ContractionCapabilityAttr>(item);
    contractions.push_back(llvm::json::Object{{"lhs", spelling(capability.getLhs())},
        {"rhs", spelling(capability.getRhs())}, {"accumulator", spelling(capability.getAccumulator())},
        {"flush_subnormals", capability.getFlushSubnormals()}});
  }
  return llvm::json::Object{{"family", "cpu"},
      {"vector", llvm::json::Object{{"minimum_bits", target.getVectorBits()}, {"scalable", target.getVector().getScalable()}}},
      {"contractions", std::move(contractions)},
      {"execution", llvm::json::Object{{"workers", target.getWorkers()}, {"private_bytes", target.getPrivateBytes()}}}};
}
}
