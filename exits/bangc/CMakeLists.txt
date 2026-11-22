compilerinfra_target(dsa levels/storage/dsa)
compilerinfra_unit(SOURCES cost/Transfers.cpp cost/Matrix.cpp capabilities/Rows.cpp legalize/Bind.cpp emit/Kernel.cpp emit/Matrix.cpp emit/Ordered.cpp LINK_LIBRARIES ${dsa})
