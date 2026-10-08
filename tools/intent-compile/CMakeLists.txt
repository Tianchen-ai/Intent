add_llvm_executable(intent-compile
  intent-compile.cpp
)

llvm_update_compile_flags(intent-compile)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/profiles")
set(intent_profile_files)
foreach(provider Triton CuTile Mojo Weft)
  string(TOLOWER "${provider}" provider_name)
  set(profile_directory "${PROJECT_SOURCE_DIR}/lib/Target/${provider}/Transforms")
  if(provider STREQUAL "Triton" OR provider STREQUAL "CuTile")
    string(APPEND profile_directory "/Configuration")
  endif()
  configure_file("${profile_directory}/TuningProfiles.json"
                 "${CMAKE_CURRENT_BINARY_DIR}/profiles/${provider_name}.json" COPYONLY)
  list(APPEND intent_profile_files
       "${CMAKE_CURRENT_BINARY_DIR}/profiles/${provider_name}.json")
endforeach()
configure_file("${PROJECT_SOURCE_DIR}/lib/Target/BangC/Transforms/TuningProfiles.json"
               "${CMAKE_CURRENT_BINARY_DIR}/profiles/bangc.json" COPYONLY)
list(APPEND intent_profile_files "${CMAKE_CURRENT_BINARY_DIR}/profiles/bangc.json")
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
