#pragma once

#if defined(DUNE_DDM_HAVE_CUDSS)

#include "../backend/backend.hh"
#include "../sycl/mat.hh"
#include "../sycl/vec.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/vector_factory.hh"

#include <cstdint>
#include <cuda_runtime.h>
#include <cudss.h>
#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solverregistry.hh>
#include <memory>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace ddm {
namespace detail {
inline const char* cudss_status_name(cudssStatus_t status)
{
  switch (status) {
    case CUDSS_STATUS_SUCCESS: return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED: return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED: return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE: return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED: return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED: return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR: return "CUDSS_STATUS_INTERNAL_ERROR";
    case CUDSS_STATUS_IR_FAILED: return "CUDSS_STATUS_IR_FAILED";
    default: return "unknown cudssStatus_t";
  }
}
} // namespace detail

/// Checks a cuDSS call made on the host and throws if it failed. The status is part of the message: a solve that trips over
/// corrupted memory reports CUDSS_STATUS_EXECUTION_FAILED, which is worth telling apart from an allocation failure.
#define CUDSS_CHECK(call, name)                                                                                                            \
  do {                                                                                                                                     \
    const cudssStatus_t ddm_cudss_status = (call);                                                                                         \
    if (ddm_cudss_status != CUDSS_STATUS_SUCCESS) [[unlikely]]                                                                             \
      DUNE_THROW(Dune::Exception, "cuDSS call " << (name) << " failed with " << ddm::detail::cudss_status_name(ddm_cudss_status) << " ("   \
                                                << static_cast<int>(ddm_cudss_status) << ")");                                             \
  } while (0)

/// Checks a cuDSS call that must not throw and aborts if it failed. Used inside the SYCL custom operation, which runs on an
/// AdaptiveCpp worker thread where an escaping exception would hit std::terminate with no useful message.
#define CUDSS_CHECK_FATAL(call, name)                                                                                                      \
  do {                                                                                                                                     \
    const cudssStatus_t ddm_cudss_status = (call);                                                                                         \
    if (ddm_cudss_status != CUDSS_STATUS_SUCCESS) [[unlikely]] {                                                                           \
      logger::error_all("cuDSS call {} failed with {} ({})", (name), ddm::detail::cudss_status_name(ddm_cudss_status),                     \
                        static_cast<int>(ddm_cudss_status));                                                                               \
      MPI_Abort(MPI_COMM_WORLD, 17);                                                                                                       \
    }                                                                                                                                      \
  } while (0)

/// Checks a cuDSS cleanup call and only reports a failure. Used in the destructor, where there is nothing left to salvage and
/// throwing would terminate.
#define CUDSS_CHECK_CLEANUP(call, name)                                                                                                    \
  do {                                                                                                                                     \
    const cudssStatus_t ddm_cudss_status = (call);                                                                                         \
    if (ddm_cudss_status != CUDSS_STATUS_SUCCESS) [[unlikely]]                                                                             \
      logger::error("cuDSS cleanup call {} failed with {} ({})", (name), ddm::detail::cudss_status_name(ddm_cudss_status),                 \
                    static_cast<int>(ddm_cudss_status));                                                                                   \
  } while (0)

/** @brief Direct solver for a GPU-resident matrix, backed by NVIDIA's cuDSS.
 *
 *  The matrix is factorised once, in the constructor; every apply() is one triangular solve against that factorisation.
 *
 *  All cuDSS work is enqueued into the CUDA stream that backs the SYCL queue the matrix lives on, see enqueue_phase(). That
 *  makes a solve an ordinary node of the SYCL dependency graph: it is ordered against the kernels that fill its right hand side
 *  and read its solution without any host synchronisation, and waiting on the queue waits for the factorisation and the solve
 *  as well. The queue must be in-order, which is the assumption the rest of the SYCL backend makes anyway.
 */
template <class Scalar, class Index = std::uint_least32_t>
class CuDSSSolver : public Dune::InverseOperator<ddm::Sycl::Vec<Scalar, Index>, ddm::Sycl::Vec<Scalar, Index>> {
public:
  using Vec = ddm::Sycl::Vec<Scalar, Index>;

  static_assert(std::is_same_v<Scalar, double> or std::is_same_v<Scalar, float>, "Only double and float for now");
  static_assert(sizeof(Index) == 4 or sizeof(Index) == 8, "cuDSS only knows 32 and 64 bit indices");

  static constexpr cudssDataType_t cudss_value_type = std::is_same_v<Scalar, double> ? CUDSS_R_64F : CUDSS_R_32F;
  static constexpr cudssDataType_t cudss_index_type = sizeof(Index) == 4 ? CUDSS_R_32I : CUDSS_R_64I;

  explicit CuDSSSolver(const ddm::Sycl::Mat<Scalar, Index>& A)
      : q(A.queue())
      , n(A.N())
      , x_scratch(create_vector_for_matrix(A))
      , b_scratch(create_vector_for_matrix(A))
  {
    // Everything here hands raw USM pointers to the CUDA runtime and enqueues work into the queue's CUDA stream, so the queue
    // has to be on the CUDA backend. The default device selector will happily place it elsewhere when no CUDA device is
    // visible, and we would then feed host pointers to cuDSS and get silent garbage instead of an error.
    DDM_CHECK(q.get_device().get_backend() == sycl::backend::cuda, "cuDSS needs a queue on the CUDA backend, but this one runs on '{}'",
              q.get_device().get_info<sycl::info::device::name>());
    DDM_CHECK(q.is_in_order(), "cuDSS solver needs an in-order queue to be ordered against the kernels producing its right hand side");

    CUDSS_CHECK(cudssCreate(&handle), "cudssCreate");
    CUDSS_CHECK(cudssConfigCreate(&solver_config), "cudssConfigCreate");
    CUDSS_CHECK(cudssDataCreate(handle, &solver_data), "cudssDataCreate");

    const cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    const cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    const cudssIndexBase_t base = CUDSS_BASE_ZERO;

    CUDSS_CHECK(cudssMatrixCreateCsr(&a, A.N(), A.M(), A.nonzeros(), A.row_offsets(), nullptr, A.column_indices(), A.values(),
                                     cudss_index_type, cudss_index_type, cudss_value_type, mtype, mview, base),
                "cudssMatrixCreateCsr");

    // The two dense descriptors are created once here and retargeted at the caller's vectors in apply(); creating a pair per
    // solve leaks one descriptor per Krylov iteration. cudssMatrixCreateDn wants memory to describe, and the setup phases below
    // want a right hand side and a solution, so they start out on scratch vectors of our own.
    CUDSS_CHECK(cudssMatrixCreateDn(&x_desc, n, 1, n, x_scratch.data(), cudss_value_type, CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn (x)");
    CUDSS_CHECK(cudssMatrixCreateDn(&b_desc, n, 1, n, b_scratch.data(), cudss_value_type, CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn (b)");
    x_values = x_scratch.data();
    b_values = b_scratch.data();

    enqueue_phase(CUDSS_PHASE_ANALYSIS, "cudssExecute (analysis)", x_scratch.data(), b_scratch.data());
    enqueue_phase(CUDSS_PHASE_FACTORIZATION, "cudssExecute (factorization)", x_scratch.data(), b_scratch.data());

    // The caller expects a solver that is ready to use once the constructor returns. Unlike before, this wait covers the cuDSS
    // work too, because it runs on this queue's stream.
    q.wait();
  }

  CuDSSSolver(const CuDSSSolver&) = delete;
  CuDSSSolver& operator=(const CuDSSSolver&) = delete;
  CuDSSSolver(CuDSSSolver&&) = delete;
  CuDSSSolver& operator=(CuDSSSolver&&) = delete;

  ~CuDSSSolver() override
  {
    // No cuDSS work may still be in flight when the factorisation and the descriptors go away. The work is on the queue's
    // stream, so waiting on the queue is enough; it also covers the scratch vectors, which are freed right after this body.
    q.wait();

    if (x_desc) CUDSS_CHECK_CLEANUP(cudssMatrixDestroy(x_desc), "cudssMatrixDestroy (x)");
    if (b_desc) CUDSS_CHECK_CLEANUP(cudssMatrixDestroy(b_desc), "cudssMatrixDestroy (b)");
    if (a) CUDSS_CHECK_CLEANUP(cudssMatrixDestroy(a), "cudssMatrixDestroy (A)");
    if (solver_data) CUDSS_CHECK_CLEANUP(cudssDataDestroy(handle, solver_data), "cudssDataDestroy");
    if (solver_config) CUDSS_CHECK_CLEANUP(cudssConfigDestroy(solver_config), "cudssConfigDestroy");
    if (handle) CUDSS_CHECK_CLEANUP(cudssDestroy(handle), "cudssDestroy");
  }

  /** Solves A x = b for the factorisation built in the constructor.
   *
   *  Asynchronous: this enqueues the solve on the matrix's queue and returns. Anything that reads x through that queue is
   *  ordered after it automatically; anything that reaches around the queue -- MPI, a host read -- has to wait for the queue
   *  first, which is what Backend::sync() does.
   */
  void apply(Vec& x, Vec& b, Dune::InverseOperatorResult& res) override
  {
    DDM_CHECK(x.size() == n and b.size() == n, "cuDSS solver was factorised for {} unknowns but got x of size {} and b of size {}", n,
              x.size(), b.size());

    enqueue_phase(CUDSS_PHASE_SOLVE, "cudssExecute (solve)", x.data(), b.data());

    res.iterations = 1;
    res.converged = true;
  }

  void apply(Vec& x, Vec& b, [[maybe_unused]] double reduction, Dune::InverseOperatorResult& res) override { apply(x, b, res); }

private:
  /** Enqueues one cuDSS phase into the CUDA stream backing our SYCL queue.
   *
   *  cuDSS drives the GPU through a CUDA stream of its own. Left alone it uses the legacy default stream, which AdaptiveCpp's
   *  streams do not synchronise with -- they are created with cudaStreamNonBlocking, which opts out of exactly that. A solve
   *  issued that way races with the kernel that writes its right hand side and with the gather that reads its solution, which
   *  is what made the results non-deterministic. Running it inside a custom operation puts it on the queue's own stream, where
   *  the in-order queue orders it against everything before and after it.
   *
   *  The callable runs on an AdaptiveCpp worker thread, in submission order, so it must not let an exception escape -- hence
   *  CUDSS_CHECK_FATAL. It runs on the host while earlier stream work may still be executing, which is why the descriptors are
   *  only retargeted when the caller actually handed us different memory: in the Schwarz preconditioner the same two vectors
   *  come back every iteration, so in the steady state the descriptors are never touched again.
   */
  void enqueue_phase(cudssPhase_t phase, const char* name, const Scalar* new_x, const Scalar* new_b)
  {
    q.AdaptiveCpp_enqueue_custom_operation([this, phase, name, new_x, new_b](sycl::interop_handle& h) {
      CUDSS_CHECK_FATAL(cudssSetStream(handle, h.get_native_queue<sycl::backend::cuda>()), "cudssSetStream");

      if (new_x != x_values) {
        CUDSS_CHECK_FATAL(cudssMatrixSetValues(x_desc, new_x), "cudssMatrixSetValues (x)");
        x_values = new_x;
      }
      if (new_b != b_values) {
        CUDSS_CHECK_FATAL(cudssMatrixSetValues(b_desc, new_b), "cudssMatrixSetValues (b)");
        b_values = new_b;
      }

      CUDSS_CHECK_FATAL(cudssExecute(handle, phase, solver_config, solver_data, a, x_desc, b_desc), name);
    });
  }

  sycl::queue q;
  Index n;

  // Memory for the dense descriptors to point at until apply() retargets them; cudssMatrixCreateDn and the setup phases need
  // a solution and a right hand side that exist.
  Vec x_scratch;
  Vec b_scratch;

  cudssHandle_t handle{};
  cudssConfig_t solver_config{};
  cudssData_t solver_data{};

  cudssMatrix_t a{};      ///< CSR description of the matrix we were built from
  cudssMatrix_t x_desc{}; ///< dense description of the solution vector of the current solve
  cudssMatrix_t b_desc{}; ///< dense description of the right hand side of the current solve

  // What x_desc and b_desc currently point at, so that a repeated solve on the same vectors does not touch them. Only read and
  // written from inside the custom operation, i.e. on one thread and in submission order.
  const Scalar* x_values{nullptr};
  const Scalar* b_values{nullptr};
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
