# Unit tests of dune-ddm. Every test gets the label ddm, which creates the target build_ddm_tests: build_tests would
# also build the tests of the DUNE modules in the superbuild. Build and run with
#
#     cmake --build build --target build_ddm_tests && ctest --test-dir build -L ddm
dune_add_test(SOURCES test_multivec.cc LINK_LIBRARIES duneddm LABELS ddm)
dune_add_test(SOURCES test_vec.cc LINK_LIBRARIES duneddm LABELS ddm)
