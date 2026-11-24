cmake_minimum_required(VERSION 3.22)
option(INTENTDSL_PORTABLE_BINARY "Hide static LLVM/MLIR symbols and avoid build-host paths" OFF)
find_package(CompilerInfra 0.5.1 EXACT REQUIRED CONFIG COMPONENTS Project)
compilerinfra_project_name(product_name)
project(${product_name} LANGUAGES C CXX)
find_package(CompilerInfra 0.5.1 EXACT REQUIRED CONFIG)
compilerinfra_product()
if(INTENTDSL_PORTABLE_BINARY AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(${product_name} PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,--as-needed")
  set_target_properties(${product_name} PROPERTIES INSTALL_RPATH "" INSTALL_RPATH_USE_LINK_PATH OFF)
endif()
install(DIRECTORY frontend/python/intentdsl
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        FILES_MATCHING PATTERN "*.py")
install(FILES doc/manual.zh.md doc/manual.en.md
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/intentdsl/_manual")
