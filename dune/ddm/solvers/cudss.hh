#pragma once

#if defined(DUNE_DDM_HAVE_CUDSS)

#include "../backend/backend.hh"
#include "../sycl/mat.hh"
#include "../sycl/vec.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/vector_factory.hh"

#include <cuda_runtime.h>
#include <cudss.h>
#include <dune/common/parametertree.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solverregistry.hh>
#include <exception>
#include <memory>
#include <sycl/sycl.hpp>

namespace ddm {
#define CUDSS_CHECK(call, name)                                                                                                            \
  do {                                                                                                                                     \
    auto status = call;                                                                                                                    \
    if (status != CUDSS_STATUS_SUCCESS) {                                                                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": Error in cuDss (" << (name) << ")" << std::endl;                                      \
      throw std::exception();                                                                                                              \
    }                                                                                                                                      \
  } while (0);

template <class Scalar, class Index = std::uint_least32_t>
class CuDSSSolver : public Dune::InverseOperator<ddm::Sycl::Vec<Scalar, Index>, ddm::Sycl::Vec<Scalar, Index>> {
public:
  using Vec = ddm::Sycl::Vec<Scalar, Index>;

  static_assert(std::is_same_v<Scalar, double> or std::is_same_v<Scalar, float>, "Only double and float for now");
  static constexpr cudssDataType_t cudss_value_type = std::is_same_v<Scalar, double> ? CUDSS_R_64F : CUDSS_R_32F;

  explicit CuDSSSolver(const ddm::Sycl::Mat<Scalar, Index>& A)
      : q(A.queue())
  {
    CUDSS_CHECK(cudssCreate(&handle), "cudssCreate");
    CUDSS_CHECK(cudssConfigCreate(&solver_config), "cudssConfigCreate");
    CUDSS_CHECK(cudssDataCreate(handle, &solver_data), "cudssDataCreate");

    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;

    int n = A.N();
    int m = A.M();
    int nnz = A.nonzeros();

    CUDSS_CHECK(cudssMatrixCreateCsr(&a, n, m, nnz, A.row_offsets(), nullptr, A.column_indices(), A.values(), CUDSS_R_32I, CUDSS_R_32I,
                                     cudss_value_type, mtype, mview, base),
                "cudssMatrixCreateCsr");

    // cuDSS needs vectors for the factorisation, so create some here
    auto x = create_vector_for_matrix(A);
    auto b = create_vector_for_matrix(A);
    auto xx = vec_to_cudss(x);
    auto bb = vec_to_cudss(b);

    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config, solver_data, a, xx, bb), "cudssExecute for analysis");
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config, solver_data, a, xx, bb), "cudssExecute for factor");

    CUDSS_CHECK(cudssMatrixDestroy(xx), "cudssMatrixDestroy for xx");
    CUDSS_CHECK(cudssMatrixDestroy(bb), "cudssMatrixDestroy for bb");
  }

  void apply(Vec& x, Vec& b, Dune::InverseOperatorResult& res) override
  {
    auto xx = vec_to_cudss(x);
    auto bb = vec_to_cudss(b);

    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, a, xx, bb), "cudssExecute for solve");

    res.iterations = 1;
    res.converged = true;
  }

  void apply(Vec& x, Vec& b, [[maybe_unused]] double reduction, Dune::InverseOperatorResult& res) override { apply(x, b, res); }

private:
  cudssMatrix_t vec_to_cudss(const Vec& v)
  {
    cudssMatrix_t vv;
    CUDSS_CHECK(cudssMatrixCreateDn(&vv, v.size(), 1, v.size(), v.data(), cudss_value_type, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn");
    return vv;
  }

  cudssHandle_t handle{};
  cudssConfig_t solver_config{};
  cudssData_t solver_data{};

  cudssMatrix_t a{};

  sycl::queue q;
};
} // namespace ddm

namespace Dune {
DUNE_REGISTER_SOLVER("cudss",
                     [](auto op_traits, const auto& op,
                        const Dune::ParameterTree&) -> std::shared_ptr<typename decltype(op_traits)::solver_type> {
                       using OpTraits = decltype(op_traits);
                       using Scalar = typename OpTraits::domain_type::field_type;

                       if constexpr (OpTraits::isAssembled                                          // direct solver, so not matrix-free
                                     && ddm::backend::IsGpuResident<typename OpTraits::matrix_type> // matrix must live on GPU
                                     && ddm::backend::IsGpuResident<typename OpTraits::domain_type> // vector must live on GPU
                                     && std::is_same_v<typename OpTraits::domain_type,
                                                       typename OpTraits::range_type> // rhs and solution vector must be of the
                                                                                      // same type
                                     && (std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>)) {
                         const auto& A = op_traits.getAssembledOpOrThrow(op);
                         const auto& mat = A->getmat();

                         return std::make_shared<ddm::CuDSSSolver<Scalar>>(mat);
                       }
                       else {
                         DUNE_THROW(Dune::UnsupportedType, "cudss requires a GPU-resident matrix/vector pair");
                       }
                       return nullptr;
                     });
} // namespace Dune

#endif
