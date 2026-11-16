add_subdirectory(Serialization)
add_subdirectory(Transforms)
intent_backend(Mojo SOURCE Backend.cpp PROFILE Transforms/TuningProfiles.json PROFILE_NAME mojo.json
  LIBRARIES MLIRIntentMojoTransforms MLIRIntentMojoSerialization)
