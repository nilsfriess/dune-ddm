#pragma once

#include "../backend/backend.hh"
#include "../helpers.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/solverregistry.hh>
#include <memory>

namespace ddm {
class CuDSSSolver {};
} // namespace ddm

namespace Dune {
DUNE_REGISTER_SOLVER("cudss", [](auto op_traits, const auto& op, const Dune::ParameterTree& config) -> std::shared_ptr<typename decltype(op_traits)::solver_type> {
  using OpTraits = decltype(op_traits);

  if constexpr (OpTraits::isAssembled                                                                     // direct solver, so not matrix-free
                && ddm::backend::IsGpuResident<typename OpTraits::matrix_type>                            // matrix must live on GPU
                && ddm::backend::IsGpuResident<typename OpTraits::domain_type>                            // vector must live on GPU
                && std::is_same_v<typename OpTraits::domain_type, typename OpTraits::range_type>          // rhs and solution vector must be of the same type
                && (std::is_same_v<ddm::backend::element_of_t<typename OpTraits::matrix_type>, double> || //
                    std::is_same_v<ddm::backend::element_of_t<typename OpTraits::matrix_type>, float>)) {
    TODO("cudss");
  }
  else {
    DUNE_THROW(Dune::UnsupportedType, "cudss requires a GPU-resident matrix/vector pair");
  }
  return nullptr;
});
} // namespace Dune
