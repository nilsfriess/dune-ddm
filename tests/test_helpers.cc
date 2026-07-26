#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_utils.hh"

#include <dune/common/fmatrix.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/ddm/helpers.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <string>
#include <vector>

/** @file
 *
 *  Sequential tests for the pure helpers in helpers.hh: detect_dirichlet_dofs and
 *  get_parameter_tree_prefix.
 */

namespace {

/** Five rows, one per case detect_dirichlet_dofs has to distinguish:
 *    0: identity row                        -> Dirichlet
 *    1: diagonal 1, nonzero off-diagonals   -> not Dirichlet
 *    2: diagonal != 1, no off-diagonals     -> not Dirichlet
 *    3: scaled identity (diagonal 2)        -> not Dirichlet
 *    4: no stored diagonal at all           -> not Dirichlet (must not throw)
 */
template <class Mat>
Mat buildDirichletTestMatrix()
{
  Mat A;
  A.setBuildMode(Mat::implicit);
  A.setImplicitBuildModeParameters(3, 1);
  A.setSize(5, 5);

  A.entry(0, 0) = 1.0;

  A.entry(1, 1) = 1.0;
  A.entry(1, 2) = -1.0;

  A.entry(2, 2) = 4.0;

  A.entry(3, 3) = 2.0;

  A.entry(4, 0) = -1.0;
  A.entry(4, 3) = -1.0;

  A.compress();
  return A;
}

template <class Mat>
void checkDirichletDetection(Dune::TestSuite& t, const std::string& what)
{
  const auto A = buildDirichletTestMatrix<Mat>();
  const auto dirichlet = detect_dirichlet_dofs(A);

  const std::vector<bool> expected = {true, false, false, false, false};
  const std::vector<std::string> row_kind = {"identity row", "diagonal 1 with off-diagonals", "diagonal != 1", "scaled identity", "no stored diagonal"};

  const bool size_ok = dirichlet.size() == expected.size();
  t.check(size_ok, what + ": result size") << "expected " << expected.size() << ", got " << dirichlet.size();
  if (!size_ok) return;

  for (std::size_t i = 0; i < expected.size(); ++i)
    t.check(dirichlet[i] == expected[i], what + ": " + row_kind[i]) << "row " << i << " expected " << expected[i] << ", got " << dirichlet[i];
}

void checkParameterTreePrefix(Dune::TestSuite& t)
{
  Dune::ParameterTree ptree;
  ptree["solver"] = "cg";
  ptree.sub("coarse")["type"] = "galerkin";
  ptree.sub("coarse").sub("eigen")["tol"] = "1e-8";

  const auto check_prefix = [&t](const Dune::ParameterTree& tree, const std::string& expected, const std::string& name) {
    const auto got = get_parameter_tree_prefix(tree);
    t.check(got == expected, name) << "expected '" << expected << "', got '" << got << "'";
  };

  check_prefix(ptree, "", "root prefix is empty");
  check_prefix(ptree.sub("coarse"), "coarse.", "sub prefix");
  check_prefix(ptree.sub("coarse").sub("eigen"), "coarse.eigen.", "nested sub prefix");
  check_prefix(ptree.sub("coarse.eigen"), "coarse.eigen.", "dotted key matches nesting");

  // A missing sub-tree resolves to ParameterTree::empty_, whose prefix is the literal "<unknown>"
  // rather than an error. Pinned because callers build parameter keys from this.
  const Dune::ParameterTree& const_ptree = ptree;
  check_prefix(const_ptree.sub("does_not_exist", false), "<unknown>", "missing sub-tree reports '<unknown>'");
}

} // namespace

int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);

  return ddmtest::runParallelTest("test_helpers", [](Dune::TestSuite& t) {
    checkDirichletDetection<Dune::BCRSMatrix<double>>(t, "BCRSMatrix<double>");
    checkDirichletDetection<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>>(t, "BCRSMatrix<FieldMatrix<double,1,1>>");
    checkParameterTreePrefix(t);
  });
}
