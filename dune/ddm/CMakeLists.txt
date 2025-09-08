#install headers
install(FILES
        combined_preconditioner.hh
        datahandles.hh
        galerkin_preconditioner.hh
        helpers.hh
        logger.hh
        nonoverlapping_operator.hh
        overlap_extension.hh
        pdelab_helper.hh
        pou.hh
        schwarz.hh
        strumpack.hh
        coarsespaces/coarse_spaces.hh
        coarsespaces/eigensolvers.hh
        coarsespaces/energy_minimal_extension.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/ddm)
