compilerinfra_target(dsa levels/storage/dsa)
compilerinfra_unit(SOURCES capabilities/Rows.cpp legalize/Bind.cpp emit/Kernel.cpp LINK_LIBRARIES ${dsa})
