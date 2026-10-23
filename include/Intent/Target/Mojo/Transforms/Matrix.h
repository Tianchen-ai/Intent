#ifndef INTENT_TARGET_MOJO_TRANSFORMS_MATRIX_H
#define INTENT_TARGET_MOJO_TRANSFORMS_MATRIX_H

#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"

namespace intent::mojo {
void registerMatrixImplementations(cpu::ImplementationRegistry &registry);
}
#endif
