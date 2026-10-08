#ifndef INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATION_TUNINGPROFILES_H
#define INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATION_TUNINGPROFILES_H

#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"

namespace intent::triton {

const gpu::TuningProfileSchema &tuningProfileSchema();

} // namespace intent::triton

#endif
