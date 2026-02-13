#pragma once

#include "../logger.hh"

#include <cstddef>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <memory>
#include <unordered_map>
#include <vector>

/**
 * @brief Energy-minimizing extension from boundary to interior of a subdomain region.
 *
 * This class implements energy-minimizing extension that takes values defined on the
 * boundary of a region within a subdomain and extends them harmonically to the interior
 * of that region. The extension minimizes the energy in the sense that it solves:
 *
 *   A_interior * u_interior = -A_interior_boundary * u_boundary
 *
 * where A is the discretized operator (e.g., stiffness matrix), u_boundary contains
 * the known values on the boundary, and u_interior are the unknown interior values
 * to be computed.
 *
 * @tparam Mat Matrix type (typically a DUNE matrix)
 * @tparam Vec Vector type (typically a DUNE vector)
 */
template <class Mat, class Vec>
class EnergyMinimalExtension {
public:
  /**
   * @brief Constructor for energy-minimizing extension operator.
   *
   * Creates the necessary data structures to compute an energy-minimal extension from
   * the dofs in boundary_indices to the dofs in interior indices.
   *
   * @note It is not necessary that the union of interior and boundary dofs is equal to
   *       the set of all dofs. The boundary and interior dofs may correspond to the
   *       boundary and interior of a sub-region of the domain that A lives on.
   *
   * @param A The full subdomain matrix
   * @param interior_indices Mapping from interior DOF index to subdomain DOF index
   * @param boundary_indices Mapping from boundary DOF index to subdomain DOF index
   */
  EnergyMinimalExtension(const Mat& A, const std::vector<std::size_t>& interior_indices, const std::vector<std::size_t>& boundary_indices)
      : n_interior_indices(interior_indices.size())
      , n_boundary_indices(boundary_indices.size())
      , b(n_interior_indices + n_boundary_indices) // The right-hand side of the harmonic extension (will be set in extend())
  {
    // We need the matrix A_{IB, IB} that contains both the dofs from the interior_indices and the boundary_indices.
    // We then eliminate the A_BB block by inserting trivial rows. Solving a linear system with this matrix then
    // corresponds to computing an energy-minimising (w.r.t. the energy-norm induced by A_{IB, IB}) extension from
    // the given boundary to the interior.

    // Let's start by creating a map from subdomain indices -> interior and boundary indices.
    // This is necessary because what we call interior and boundary here, is not necessarily the
    // interior + boundary of the whole subdomain, it might be interior + boundary of only part
    // of the subdomain.
    std::unordered_map<std::size_t, std::size_t> subdomain_to_IB;
    auto n = interior_indices.size() + boundary_indices.size();
    subdomain_to_IB.reserve(n);
    for (std::size_t i = 0; i < interior_indices.size(); ++i) subdomain_to_IB[interior_indices[i]] = i;
    auto cnt = subdomain_to_IB.size();
    for (std::size_t i = 0; i < boundary_indices.size(); ++i) subdomain_to_IB[boundary_indices[i]] = cnt + i;

    // Now we can extract the IB part from A
    Mat A_IB(n, n, A.nonzeroes() / A.N() + 2, 0.2, Mat::implicit);
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci)
        if (subdomain_to_IB.contains(ri.index()) and subdomain_to_IB.contains(ci.index())) A_IB.entry(subdomain_to_IB[ri.index()], subdomain_to_IB[ci.index()]) = *ci;
    }
    // Ensure diagonal entries exist for boundary rows (needed for elimination below)
    for (std::size_t i = interior_indices.size(); i < n; ++i) A_IB.entry(i, i);
    A_IB.compress();

    // Eliminate the rows corresponding to boundary dofs
    for (std::size_t i = interior_indices.size(); i < n; ++i)
      for (auto ci = A_IB[i].begin(); ci != A_IB[i].end(); ++ci) *ci = (ci.index() == i) ? 1. : 0.;

    // Initialise the solver
    solver = std::make_unique<Dune::UMFPack<Mat>>();
    solver->setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
    solver->setOption(UMFPACK_IRSTEP, 0); // Disable iterative refinement for performance
    solver->setMatrix(A_IB);
  }

  /**
   * @brief Perform energy-minimizing extension from boundary to interior.
   *
   * Given values on the boundary of the region, this function computes the
   * energy-minimizing (harmonic) extension to the interior by solving:
   *   A_ii * u_i = -A_ib * u_b
   * where u_b are the given boundary values and u_i are the computed interior values.
   *
   * @param boundary_values Vector containing values on the boundary DOFs
   * @return Vector containing the computed interior values (without the prescribed boundary values!)
   */
  Vec extend(const Vec& boundary_values)
  {
    if (boundary_values.size() != n_boundary_indices) throw std::invalid_argument("Boundary values vector has incorrect size");

    // Put the boundary values into b. The DOFs are ordered as: first all interior, then all boundary
    b = 0;
    for (std::size_t i = 0; i < boundary_values.size(); ++i) b[n_interior_indices + i] = boundary_values[i];

    // Solve the linear system
    Dune::InverseOperatorResult res;
    Vec x(b.size());
    solver->apply(x, b, res);

    // Resize the x value to remove the values corresponding to boundary dofs
    x.resize(n_interior_indices);
    return x;
  }

private:
  std::size_t n_interior_indices;
  std::size_t n_boundary_indices;

  Vec b;

  std::unique_ptr<Dune::UMFPack<Mat>> solver;
};
