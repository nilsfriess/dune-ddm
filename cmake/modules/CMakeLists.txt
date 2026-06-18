# Install dune-ddm's module-specific cmake files so that downstream consumers of
# the installed Dune::DDM target pick them up: Dune adds this directory to their
# CMAKE_MODULE_PATH and auto-includes DuneDdmMacros.cmake.
install(FILES DuneDdmMacros.cmake
        DESTINATION ${DUNE_INSTALL_MODULEDIR})
