#pragma once

/**
 * @file ddm_utilities.hh
 * @brief Common utility functions for DDM examples
 * 
 * This file provides helper functions that are shared across multiple
 * domain decomposition method examples, including:
 * - Grid creation and partitioning
 * - Visualization helpers
 * - Common lambda functions for boundary conditions
 * 
 * Part of the unified PDELab example framework.
 */

#include <cmath>
#include <memory>
#include <string>

#include <dune/common/parametertree.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#if USE_UGGRID
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/uggrid.hh>
#endif

#include <dune/grid/yaspgrid.hh>

#include "dune/ddm/logger.hh"

namespace DDMUtilities {

/**
 * @brief Check if a grid type is YaspGrid
 * 
 * Uses concept-based detection to check if a grid has the torus() method,
 * which is specific to YaspGrid.
 */
template <class Grid>
constexpr bool isYaspGrid()
{
  return requires(Grid g) { g.torus(); };
}

/**
 * @brief Create and partition a grid based on configuration
 * 
 * Creates either a structured grid (UGGrid/YaspGrid) from the parameter tree.
 * Supports:
 * - Loading from mesh file (UGGrid with meshfile key)
 * - Creating structured cube grids
 * - Automatic grid partitioning with ParMETIS (for UGGrid)
 * - Global refinement
 * 
 * @tparam Grid The grid type (Dune::UGGrid or Dune::YaspGrid)
 * @param ptree Parameter tree with grid configuration
 * @param helper MPI helper for parallel partitioning
 * @param subtree_name Name of the subtree containing grid parameters (default: "")
 * @return Unique pointer to the created and partitioned grid
 * 
 * Parameter tree keys:
 * - meshfile: Path to mesh file (UGGrid only)
 * - gridsize: Number of elements per dimension
 * - gridsize_per_rank: Alternative to gridsize, scales with MPI ranks
 * - refine: Number of global refinement steps
 * - grid_overlap: Grid overlap for YaspGrid (default: 0)
 * - verbose: Verbosity level for mesh loading
 */
template <class Grid>
std::unique_ptr<Grid> make_grid(const Dune::ParameterTree& ptree, const Dune::MPIHelper& helper, const std::string& subtree_name = "")
{
  auto* event = Logger::get().registerEvent("Grid", "create");
  Logger::ScopedLog sl(event);

  // Get the parameter tree (either root or subtree)
  const Dune::ParameterTree& grid_ptree = subtree_name.empty() ? ptree : ptree.sub(subtree_name);

  constexpr int dim = Grid::dimension;
  std::unique_ptr<Grid> grid;

  if constexpr (isYaspGrid<Grid>()) {
    // YaspGrid path
    auto gridsize = grid_ptree.get("gridsize", 32);
    if (grid_ptree.hasKey("gridsize_per_rank")) {
      auto grid_sqrt = static_cast<int>(std::pow(helper.size(), 1.0 / dim));
      gridsize = grid_ptree.get<int>("gridsize_per_rank") * grid_sqrt;
    }

    const int grid_overlap = grid_ptree.get("grid_overlap", 0);
    
    // Create YaspGrid based on dimension
    if constexpr (dim == 2) {
      Dune::Yasp::PowerDPartitioning<dim> partitioner;
      grid = std::make_unique<Grid>(
          Dune::FieldVector<typename Grid::ctype, dim>{1.0, 1.0},
          std::array<int, dim>{gridsize, gridsize},
          std::bitset<dim>(0ULL),
          grid_overlap,
          typename Grid::Communication(),
          &partitioner);
    } else if constexpr (dim == 3) {
      Dune::Yasp::PowerDPartitioning<dim> partitioner;
      grid = std::make_unique<Grid>(
          Dune::FieldVector<typename Grid::ctype, dim>{1.0, 1.0, 1.0},
          std::array<int, dim>{gridsize, gridsize, gridsize},
          std::bitset<dim>(0ULL),
          grid_overlap,
          typename Grid::Communication(),
          &partitioner);
    }
    
    grid->loadBalance();
  } else {
    // UGGrid or other unstructured grid path
#if USE_UGGRID
    if (grid_ptree.hasKey("meshfile")) {
      logger::info("Loading mesh from file");
      const auto meshfile = grid_ptree.get("meshfile", "../data/unitsquare.msh");
      const auto verbose = grid_ptree.get("verbose", 0);
      grid = Dune::GmshReader<Grid>::read(meshfile, verbose > 2);
    } else
#endif
    {
      auto gridsize = static_cast<unsigned int>(grid_ptree.get("gridsize", 32));
      if (grid_ptree.hasKey("gridsize_per_rank")) {
        auto grid_sqrt = static_cast<int>(std::pow(helper.size(), 1.0 / dim));
        gridsize = grid_ptree.get<int>("gridsize_per_rank") * grid_sqrt;
      }
      
      // Create structured grid based on dimension
      if constexpr (dim == 2) {
        grid = Dune::StructuredGridFactory<Grid>::createCubeGrid(
            {0.0, 0.0}, {1.0, 1.0}, {gridsize, gridsize});
      } else if constexpr (dim == 3) {
        grid = Dune::StructuredGridFactory<Grid>::createCubeGrid(
            {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {gridsize, gridsize, gridsize});
      }
    }

    // Partition with ParMETIS and load balance
    auto gv = grid->leafGridView();
    auto part = Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, helper);
    grid->loadBalance(part, 0);
  }

  // Global refinement
  const int refine = grid_ptree.get("refine", 0);
  if (refine > 0) {
    logger::info("Refining grid {} times", refine);
    grid->globalRefine(refine);
  }

  logger::info("Created grid with {} elements on rank {}", 
               grid->leafGridView().size(0), helper.rank());

  return grid;
}

// Forward declaration - actual implementation needs full communication headers
// This function is defined in the original poisson.cc and should remain there
// or be moved to a separate implementation file with proper includes

/**
 * @brief Lambda helper to zero out values at Dirichlet DOFs
 * 
 * Returns a lambda function that sets all entries to zero where
 * the Dirichlet mask is positive.
 * 
 * @tparam DirichletMask Type of the Dirichlet mask vector
 * @param dirichlet_mask Reference to the Dirichlet constraint mask
 * @return Lambda function that zeros Dirichlet DOFs
 */
template <class DirichletMask>
auto make_zero_at_dirichlet(const DirichletMask& dirichlet_mask)
{
  return [&dirichlet_mask](auto&& x) {
    for (std::size_t i = 0; i < x.size(); ++i)
      if (dirichlet_mask[i] > 0) 
        x[i] = 0;
  };
}

} // namespace DDMUtilities
