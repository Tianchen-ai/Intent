cmake_minimum_required(VERSION 3.22)
find_package(CompilerInfra 0.3.6 EXACT REQUIRED CONFIG COMPONENTS Project)
compilerinfra_project_name(product_name)
project(${product_name} LANGUAGES C CXX)
find_package(CompilerInfra 0.3.6 EXACT REQUIRED CONFIG)
compilerinfra_product()
install(DIRECTORY frontend/python/intentdsl
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        FILES_MATCHING PATTERN "*.py")
