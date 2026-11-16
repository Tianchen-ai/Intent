#ifndef INTENT_CPU_TARGET_H
#define INTENT_CPU_TARGET_H
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "llvm/Support/JSON.h"
namespace intent::cpu {
llvm::json::Object serializeCapabilities(CapabilitiesAttr);
}
#endif
