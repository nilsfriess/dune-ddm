#include "dune/ddm/backend/host/backend.hh"
#include "dune/ddm/multivector.hh"
#include "tests/test_utils.hh"

#include <cstdint>
#include <dune/common/test/testsuite.hh>

namespace {
template <class Scalar, class Backend>
void run(Dune::TestSuite& t, typename Backend::context_type ctx)
{
  std::uint_least32_t rows = 1000;
  std::uint_least32_t cols = 10;

  ddm::MultiVector<Scalar, Backend> v(ctx, rows, cols);
  t.check(v.rows() == rows, "multivector-rows") << "Multivector didn't return the correct number of rows";
  t.check(v.cols() == cols, "multivector-columns") << "Multivector didn't return the correct number of columns";
  t.check(v.data() != nullptr, "multivector-data") << "Multivector data() is nullptr";

  // Test the const overloads
  ([&](const ddm::MultiVector<Scalar, Backend>& vc) {
    t.check(v.rows() == rows, "multivector-rows") << "Multivector didn't return the correct number of rows";
    t.check(v.cols() == cols, "multivector-columns") << "Multivector didn't return the correct number of columns";
    t.check(v.data() != nullptr, "multivector-data") << "Multivector data() is nullptr";
  })(v);
}
} // namespace

int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);
  return ddmtest::runParallelTest("test_multivector", [&](Dune::TestSuite& t) {
    run<double, ddm::backend::HostBackend>(t, {});
    run<float, ddm::backend::HostBackend>(t, {});
  });
}
