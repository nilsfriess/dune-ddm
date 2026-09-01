#include "dune/ddm/helpers.hh"
#include "dune/ddm/sycl/mat.hh"
#include "dune/ddm/sycl/vec.hh"

#include <cstdint>
#include <dune/common/exceptions.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>
#include <iostream>
#include <memory>
#include <span>
#include <sycl/sycl.hpp>
#include <utility>

template <class Vec>
class Jacobi : public Dune::Preconditioner<Vec, Vec> {
public:
  template <class Mat>
  Jacobi(const Mat& A)
      : diag(A.getdiag())
  {
    auto* d = diag.data();
    A.queue().parallel_for(sycl::range<1>(A.N()), [=](auto idx) { d[idx] = 1. / d[idx]; });
  }

  void pre(Vec&, Vec&) override {}
  void post(Vec&) override {}

  void apply(Vec& x, const Vec& d) override
  {
    DDM_CHECK(x.size() == d.size(), "x and d must have the same size");
    DDM_CHECK(x.size() == diag.size(), "x and diag must have the same size");

    auto* dd = diag.data();
    auto* bd = d.data();
    auto* xd = x.data();

    x.queue().parallel_for(sycl::range<1>(diag.size()), [=](auto idx) { xd[idx] = bd[idx] * dd[idx]; });
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::sequential; }

private:
  Vec diag;
};

int main(int argc, char** argv)
{
  sycl::queue q({sycl::property::queue::in_order{}, sycl::property::queue::AdaptiveCpp_coarse_grained_events{}});

  Dune::ParameterTree ptree;
  ptree["reduction"] = "1e-8";
  ptree["restart"] = "30";
  ptree["maxit"] = "10000";
  ptree["verbose"] = "4";
  Dune::ParameterTreeParser parser;
  parser.readOptions(argc, argv, ptree);

  using Scalar = float;
  using Vec = ddm::Sycl::Vec<Scalar>;
  using Mat = ddm::Sycl::Mat<Scalar>;
  using Index = typename Mat::index_type;

  using Op = Dune::MatrixAdapter<Mat, Vec, Vec>;
  Index gridsize = 1024;
  auto n = gridsize * gridsize;
  auto nnz = 5 * n;

  std::shared_ptr<Mat> A;
  {
    std::vector<Index> r;
    std::vector<Index> c;
    std::vector<Scalar> a;
    r.reserve(n + 1);
    c.reserve(nnz);
    a.reserve(nnz);
    r.push_back(0);

    const auto to_linear = [&](auto i, auto j) { return i * gridsize + j; };

    for (Index i = 0; i < gridsize; ++i) {
      for (Index j = 0; j < gridsize; ++j) {
        if (i > 0) {
          c.push_back(to_linear(i - 1, j));
          a.push_back(-1);
        }

        if (i < gridsize - 1) {
          c.push_back(to_linear(i + 1, j));
          a.push_back(-1);
        }

        if (j > 0) {
          c.push_back(to_linear(i, j - 1));
          a.push_back(-1);
        }

        if (j < gridsize - 1) {
          c.push_back(to_linear(i, j + 1));
          a.push_back(-1);
        }

        c.push_back(to_linear(i, j));
        a.push_back(4);

        r.push_back(c.size());
      }
    }

    A = std::make_shared<Mat>(q, n, n, std::span{r.data(), r.size()}, std::span{c.data(), c.size()}, std::span{a.data(), a.size()});
  }

  auto op = std::make_shared<Op>(A);
  auto prec = std::make_shared<Jacobi<Vec>>(*A);

  // malloc_device does not zero-initialise, so both vectors must be set explicitly
  Vec x(q, n);
  x = 0.0;
  Vec b(q, n);
  b = 1.0;

  // Dune::RestartedGMResSolver<Vec> solver(op, prec, ptree);
  Dune::CGSolver<Vec> solver(op, prec, ptree);
  Dune::InverseOperatorResult res;
  solver.apply(x, b, res);
}
