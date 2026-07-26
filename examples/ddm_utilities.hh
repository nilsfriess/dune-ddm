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

#include <array>
#include <bitset>
#include <cmath>
#include <memory>
#include <string>

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/type.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
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

enum class ElementType { Simplex, Cube };

inline std::string to_string(ElementType element_type)
{
  return element_type == ElementType::Cube ? "cube" : "simplex";
}

/// @brief Element type matching a finite element space built with @p qk_elements
///
/// The grid and the finite element space have to agree on the element type, so derive one from the
/// other rather than configuring both by hand.
constexpr ElementType element_type_for(bool qk_elements)
{
  return qk_elements ? ElementType::Cube : ElementType::Simplex;
}

namespace Impl {

/// @brief Geometry type that the elements of a grid of @p element_type have
inline Dune::GeometryType geometry_type(ElementType element_type, int dim)
{
  return element_type == ElementType::Cube ? Dune::GeometryTypes::cube(dim) : Dune::GeometryTypes::simplex(dim);
}

inline std::string describe(const Dune::GeometryType& gt)
{
  if (gt.isCube()) return "cube";
  if (gt.isSimplex()) return "simplex";
  return "unsupported";
}

/// @brief Number of elements per coordinate direction
///
/// "gridsize_per_rank" takes precedence over "gridsize" and scales with the number of MPI ranks.
inline int resolve_gridsize(const Dune::ParameterTree& ptree, const Dune::MPIHelper& helper, int dim)
{
  if (not ptree.hasKey("gridsize_per_rank")) return ptree.get("gridsize", 32);

  const auto ranks_per_dim = static_cast<int>(std::llround(std::pow(helper.size(), 1.0 / dim)));
  return ptree.get<int>("gridsize_per_rank") * ranks_per_dim;
}

/// @brief Throw unless every element of @p grid has the geometry type the caller asked for
///
/// A mesh file carries its own element type, so it can disagree with the finite element space the
/// caller intends to build on the grid. Without this check the mismatch stays invisible until it
/// surfaces much later as a crash inside the assembler.
template <class Grid>
void check_element_type(const Grid& grid, ElementType element_type, const std::string& meshfile, const Dune::MPIHelper& helper)
{
  const auto expected = geometry_type(element_type, Grid::dimension);

  int mismatch = 0;
  for (const auto& gt : grid.leafGridView().indexSet().types(0)) {
    if (gt == expected) continue;
    mismatch = 1;
    logger::error("Mesh file '{}' contains {} elements, but {} elements were requested", meshfile, describe(gt), to_string(element_type));
  }

  // Only ranks that actually hold elements can spot the mismatch, but every rank has to throw.
  if (helper.getCommunication().max(mismatch) == 0) return;

  DUNE_THROW(Dune::IOError, "make_grid(): mesh file '" << meshfile << "' does not consist of " << to_string(element_type)
                                                       << " elements. Either load a matching mesh or build a finite element space that matches the mesh.");
}

} // namespace Impl

/**
 * @brief Create and partition a grid based on configuration
 *
 * Supports:
 * - Loading from a Gmsh mesh file (any grid with a GridFactory, e.g. UGGrid)
 * - Creating structured cube or simplex grids
 * - Automatic grid partitioning with ParMETIS (for everything but YaspGrid)
 * - Global refinement
 *
 * @tparam Grid The grid type (e.g. Dune::UGGrid or Dune::YaspGrid)
 * @param ptree Parameter tree with grid configuration
 * @param helper MPI helper for parallel partitioning
 * @param element_type Element type the grid must consist of; this has to match the finite element
 *                     space that will be built on the grid. YaspGrid only supports cubes.
 * @param subtree_name Name of the subtree containing grid parameters (default: "")
 * @return Unique pointer to the created and partitioned grid
 *
 * Parameter tree keys:
 * - meshfile: Path to a Gmsh mesh file; leave empty (or omit) to build a structured grid
 * - gridsize: Number of elements per dimension
 * - gridsize_per_rank: Alternative to gridsize, scales with MPI ranks
 * - refine: Number of global refinement steps
 * - grid_overlap: Grid overlap for YaspGrid (default: 0)
 * - verbose: Verbosity level for mesh loading
 */
template <class Grid>
std::unique_ptr<Grid> make_grid(const Dune::ParameterTree& ptree, const Dune::MPIHelper& helper, ElementType element_type, const std::string& subtree_name = "")
{
  auto* event = Logger::get().registerEvent("Grid", "create");
  Logger::ScopedLog sl(event);

  // Get the parameter tree (either root or subtree)
  const Dune::ParameterTree& grid_ptree = subtree_name.empty() ? ptree : ptree.sub(subtree_name);

  constexpr int dim = Grid::dimension;
  using Corner = Dune::FieldVector<typename Grid::ctype, Grid::dimensionworld>;

  std::unique_ptr<Grid> grid;

  if constexpr (isYaspGrid<Grid>()) {
    if (element_type != ElementType::Cube) logger::warn("YaspGrid only supports cube elements, ignoring the requested {} elements", to_string(element_type));

    std::array<int, dim> cells;
    cells.fill(Impl::resolve_gridsize(grid_ptree, helper, dim));

    const Dune::Yasp::PowerDPartitioning<dim> partitioner;
    grid = std::make_unique<Grid>(Corner(1.0), cells, std::bitset<dim>(0ULL), grid_ptree.get("grid_overlap", 0), typename Grid::Communication(), &partitioner);

    grid->loadBalance();
  }
  else {
    // Unstructured grid path (UGGrid and friends)
    const auto meshfile = grid_ptree.get("meshfile", std::string{});

    if (not meshfile.empty()) {
      logger::debug("Loading mesh from {}", meshfile);
      grid = Dune::GmshReader<Grid>::read(meshfile, grid_ptree.get("verbose", 0) > 2);
      Impl::check_element_type(*grid, element_type, meshfile, helper);
    }
    else {
      std::array<unsigned int, dim> cells;
      cells.fill(static_cast<unsigned int>(Impl::resolve_gridsize(grid_ptree, helper, dim)));

      grid = element_type == ElementType::Cube ? Dune::StructuredGridFactory<Grid>::createCubeGrid(Corner(0.0), Corner(1.0), cells)
                                               : Dune::StructuredGridFactory<Grid>::createSimplexGrid(Corner(0.0), Corner(1.0), cells);
    }

    // Partition with ParMETIS and load balance
    auto gv = grid->leafGridView();
    grid->loadBalance(Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, helper), 0);
  }

  // Global refinement
  const int refine = grid_ptree.get("refine", 0);
  if (refine > 0) {
    logger::debug("Refining grid {} times", refine);
    grid->globalRefine(refine);
  }

  logger::debug("Created grid with {} elements on rank {}", grid->leafGridView().size(0), helper.rank());

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
      if (dirichlet_mask[i] > 0) x[i] = 0;
  };
}

} // namespace DDMUtilities
