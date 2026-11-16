add_llvm_executable(intent-compile
  intent-compile.cpp
)

llvm_update_compile_flags(intent-compile)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/profiles")
set(intent_profile_files)
get_property(profile_names GLOBAL PROPERTY INTENT_PROFILE_NAMES)
get_property(profile_sources GLOBAL PROPERTY INTENT_PROFILE_SOURCES)
foreach(name source IN ZIP_LISTS profile_names profile_sources)
  configure_file("${source}"
                 "${CMAKE_CURRENT_BINARY_DIR}/profiles/${name}" COPYONLY)
  list(APPEND intent_profile_files
       "${CMAKE_CURRENT_BINARY_DIR}/profiles/${name}")
endforeach()
if(SKBUILD)
  set_target_properties(intent-compile PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
else()
  set_target_properties(intent-compile PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
install(TARGETS intent-compile
  RUNTIME DESTINATION "${INTENT_RUNTIME_INSTALL_DIR}"
  COMPONENT IntentRuntime)
install(FILES ${intent_profile_files}
  DESTINATION "${INTENT_RUNTIME_INSTALL_DIR}/profiles"
  COMPONENT IntentRuntime)
target_link_libraries(intent-compile PRIVATE
  MLIRIntentCompiler
  MLIRParser
)

mlir_check_all_link_libraries(intent-compile)
