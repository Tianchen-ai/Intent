# The pass declaration is also the installed developer reference. Keep generated
# files beside intent-opt, outside the source tree and the language-only manual.
set(INTENT_PASS_REFERENCE_DIR "${PROJECT_BINARY_DIR}/tools/intent-opt/passes")
add_custom_target(intent-pass-reference)

function(intent_pass_reference group)
  mlir_tablegen(Passes.md -gen-pass-doc)
  set(reference "${INTENT_PASS_REFERENCE_DIR}/${group}.md")
  add_custom_command(OUTPUT "${reference}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${INTENT_PASS_REFERENCE_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${CMAKE_CURRENT_BINARY_DIR}/Passes.md" "${reference}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/Passes.md"
    VERBATIM)
  add_custom_target(intent-${group}-pass-reference DEPENDS "${reference}")
  add_dependencies(intent-pass-reference intent-${group}-pass-reference)
  install(FILES "${reference}"
    DESTINATION "${INTENT_RUNTIME_INSTALL_DIR}/passes"
    COMPONENT IntentRuntime)
endfunction()
