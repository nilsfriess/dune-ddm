#include "dune/ddm/communication.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/sycl/vec.hh"
#include "tests/test_utils.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char** argv)
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    sycl::queue q({sycl::property::queue::in_order{}});

    const int dim = 2;
    using Grid = Dune::YaspGrid<dim>;
    Grid grid({1., 1.}, {64, 64}, 0ULL, 2);
    auto gv = grid.leafGridView();
    auto oocc = ddmtest::create_communication_for_grid(gv);
    auto comm = ddm::make_communication_from_dune(*oocc);

    std::vector<int> v_host(gv.size(dim), 0);
    for (const auto& idx : oocc->indexSet())
      if (idx.local().attribute() == Dune::OwnerOverlapCopyAttributeSet::owner) v_host[idx.local().local()] = 1;

    auto v = ddm::Sycl::Vec<int>::from_host_vector(q, v_host);
    auto h = comm.reduce(v);
    h.wait();
    v_host = v.to_host_vector();

    for (auto e : v_host)
      if (e != 1) std::cout << "err\n";
  }
  catch (const Dune::Exception& e) {
    std::cout << "Dune exception thrown: " << e.what() << "\n";
    return 1;
  }
}
