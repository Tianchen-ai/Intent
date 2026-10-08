add_llvm_executable(intent-opt
  intent-opt.cpp
)

llvm_update_compile_flags(intent-opt)
add_dependencies(intent-opt intent-pass-reference)
if(SKBUILD)
  set_target_properties(intent-opt PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
else()
  set_target_properties(intent-opt PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
install(TARGETS intent-opt
  RUNTIME DESTINATION "${INTENT_RUNTIME_INSTALL_DIR}"
  COMPONENT IntentRuntime)

# tools/ installs intent-compile first; collect one runtime closure only after
# both tools are staged beside the shared profiles directory.
if(SKBUILD)
  configure_file("${PROJECT_SOURCE_DIR}/cmake/BundleRuntime.cmake.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/BundleRuntime.cmake.in" @ONLY)
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BundleRuntime-$<CONFIG>.cmake"
       INPUT "${CMAKE_CURRENT_BINARY_DIR}/BundleRuntime.cmake.in")
  install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/BundleRuntime-$<CONFIG>.cmake"
          COMPONENT IntentRuntime)
endif()

target_link_libraries(intent-opt PRIVATE
  MLIRIntentCompiler
  MLIROptLib
)

mlir_check_all_link_libraries(intent-opt)
