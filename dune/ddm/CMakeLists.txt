# Install all headers under dune/ddm, preserving the subdirectory layout
# (coarsespaces/, eigensolvers/, ...) and excluding the test tree. Using a
# recursive glob rather than a hand-maintained file list avoids headers silently
# going uninstalled when new ones are added.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/ddm
        FILES_MATCHING
        PATTERN "*.hh")
