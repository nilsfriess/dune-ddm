# Spectra - header-only eigenvalue solver library
# Add include directories directly to duneddm to avoid export issues
if(TARGET Eigen3::Eigen)
  target_include_directories(duneddm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/spectra/include>
    $<INSTALL_INTERFACE:extern/spectra/include>
  )
  target_compile_definitions(duneddm PUBLIC DUNE_DDM_HAVE_SPECTRA=1)
  target_link_libraries(duneddm PUBLIC Eigen3::Eigen)

  # Install the header tree so the $<INSTALL_INTERFACE:...> path above actually
  # exists in the install tree (otherwise consumers of the exported Dune::DDM
  # target hit a "non-existent path" error).
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/spectra/include/
    DESTINATION extern/spectra/include)
else()
  message(STATUS "Eigen not found, Spectra eigensolver will not be available")
endif()

# dune-pdelab does not install the headers under dune/pdelab/test (they are
# treated as test-only), but downstream code (e.g. dune-heart's rwdata.hh)
# includes some of them, such as l2difference.hh. Install that header tree into
# the same include destination as the rest of pdelab so the includes resolve
# against the installed module. Only relevant for the in-tree superbuild.
if(DUNE_DDM_SUPERBUILD)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dune-pdelab/dune/pdelab/test/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/pdelab/test
    FILES_MATCHING PATTERN "*.hh")
endif()
