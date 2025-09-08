#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/common/parallel/mpihelper.hh>
#include <dune/ddm/logger.hh>
#include <dune/ddm/nonoverlapping_operator.hh>
#include <dune/ddm/twolevel_schwarz.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#include <dune/pdelab.hh>
#pragma GCC diagnostic pop

#include "pdelab_example.hh"
#include "pdelab_helper.hh"

constexpr bool use_UGGrid = false;

using Dune::PDELab::Backend::native;
using Dune::PDELab::Backend::Native;

int main(int argc, char *argv[])
{
  try {
    const auto &helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    Dune::ParameterTree ptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("pdelab_example.ini", ptree);
    ptreeparser.readOptions(argc, argv, ptree);

    spdlog::info("MPI ranks: {}", helper.size());

    // Create grid
    using Grid = std::conditional_t<use_UGGrid, Dune::UGGrid<2>, Dune::YaspGrid<2>>;
    auto grid = make_grid<Grid>(helper, ptree);
    auto gv = grid->leafGridView();

    // Assemble the problem
    auto [A, gfs, cc, x, d] = assemble_problem(gv);

    using Mat = std::remove_cvref_t<decltype(*A)>;
    using Vec = std::remove_cvref_t<decltype(x)>;

    // Set up the preconditioner and solver
    auto remoteids = make_remote_indices(gfs, helper);
    using Prec = TwoLevelSchwarz<Mat, std::remove_cvref_t<decltype(*remoteids)>, Native<Vec>>;
    using Op = NonOverlappingOperator<Mat, Native<Vec>>;
    Dune::initSolverFactories<Op>(); // register all DUNE solvers so we can choose them via the command line

    auto solver_subtree = ptree.sub("solver");
    solver_subtree["verbose"] = helper.rank() == 0 ? solver_subtree["verbose"] : "0";

    // Create the basis vectors for the patition of unity coarse space
    std::vector<Vec> template_vecs(3, gfs);
    std::vector<Native<Vec>> native_template_vecs(3, Native<Vec>(template_vecs[0].N()));
    Dune::PDELab::interpolate([](auto &&) { return 1; }, gfs, template_vecs[0]);
    Dune::PDELab::interpolate([](auto &&x) { return x[0]; }, gfs, template_vecs[1]);
    Dune::PDELab::interpolate([](auto &&x) { return x[1]; }, gfs, template_vecs[2]);
    std::for_each(template_vecs.begin(), template_vecs.end(), [&](auto &&v) { Dune::PDELab::set_constrained_dofs(cc, 0., v); }); // ensure they're zero on the Dirichlet boundary
    std::transform(template_vecs.begin(), template_vecs.end(), native_template_vecs.begin(), [](auto &&v) { return native(v); });

    auto prec = std::make_shared<Prec>(A, *remoteids, native_template_vecs, ptree);
    auto op = std::make_shared<Op>(A, *remoteids);
    auto solver = Dune::getSolverFromFactory(op, solver_subtree, prec);

    Dune::InverseOperatorResult res;
    Native<Vec> v(x);
    v = 0;
    solver->apply(v, native(d), res);
    native(x) -= v;

    // Visualise the solution
    Dune::SubsamplingVTKWriter writer(gv, Dune::refinementLevels(0));
    Dune::PDELab::addSolutionToVTKWriter(writer, gfs, x);

    // Also visualise the  MPI partitioning
    std::vector<int> rankVec(gv.size(0), helper.rank());
    Dune::P0VTKFunction rankFunc(gv, rankVec, "Rank");
    writer.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

    writer.write("pdelab_example_output");
  }
  catch (Dune::Exception &e) {
    std::cout << "Error in DUNE: " << e.what() << "\n";
  }
  catch (std::exception &e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}
