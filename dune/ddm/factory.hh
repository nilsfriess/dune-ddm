#pragma once

#include "backend/backend.hh"
#include "backend/host/backend.hh"
#include "backend/sycl/backend.hh"
#include "logger.hh"
#include "solvers/cudss.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/solverfactory.hh>

namespace ddm {
/// Name of the direct solver to use when the config does not name one.
/// First match wins:
///  - GPU-resident matrix:  "cudss"        (requires DUNE_DDM_HAVE_CUDSS)
///  - host, double:         "umfpack"      (requires HAVE_SUITESPARSE_UMFPACK)
///  - host, float:          "strumpack"    (requires DUNE_DDM_HAVE_STRUMPACK)
template <class Op>
std::string defaultDirectSolverName()
{
  using OpTraits = Dune::OperatorTraits<Op>;
  using Matrix = typename OpTraits::matrix_type;
  using Scalar = typename Matrix::field_type; // valid only if assembled, guarded below

  if constexpr (OpTraits::isAssembled && backend::IsGpuResident<Matrix>) {
#if defined(DUNE_DDM_HAVE_CUDSS)
    return "cudss";
#endif
  }
  else if constexpr (OpTraits::isAssembled) {
#if HAVE_SUITESPARSE_UMFPACK
    if constexpr (std::is_same_v<Scalar, double> || std::is_same_v<Scalar, std::complex<double>>) return "umfpack";
#endif
#if DUNE_DDM_HAVE_STRUMPACK
    if constexpr (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>) return "strumpack";
#endif
  }
  DUNE_THROW(Dune::InvalidStateException, "No direct solver available for this operator type ("
                                              << Dune::className<Matrix>() << "). Build with UMFPack/cuDSS or set the solver explicitly.");
}

template <class Op>
auto getDirectSolverFromFactory(std::shared_ptr<Op> op, const Dune::ParameterTree& config, const std::string& key = "type")
{
  const bool defaulted = !config.hasKey(key);
  std::string name = config.get(key, defaultDirectSolverName<Op>());

  // Fail with a precise message instead of Dune's generic "unknown type"
  if (!Dune::SolverFactory<Op>::instance().contains(name))
    DUNE_THROW(Dune::InvalidStateException, "Solver '" << name
                                                       << "' is not registered for this operator type. "
                                                          "Did you include the registration header / build with the required solver?");

  Dune::ParameterTree tmp = config;
  if (defaulted) {
    logger::info("No '{}' given, using default direct solver '{}'", key, name);
    tmp[key] = name;
  }

  if (key != "type") tmp["type"] = config.get<std::string>(key); // the solverfactory hardcodes 'type' as the key for the solver name
  return Dune::getSolverFromFactory(op, tmp);
}
} // namespace ddm
