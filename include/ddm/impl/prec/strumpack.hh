#pragma once

#ifdef DUNE_DDM_HAVE_STRUMPACK

#include "ddm/check.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/impl/multivec/istlmultivec.hh"
#include "ddm/index.hh"
#include "ddm/logger.hh"
#include "ddm/mat/mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wextra-semi"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <StrumpackSparseSolver.hpp>
#pragma GCC diagnostic pop

#include <dune/common/parametertree.hh>
#include <memory>
#include <mpi.h>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddm {
namespace detail {
// getopt, which STRUMPACK uses to parse its options, keeps its state in globals
inline std::mutex strumpack_getopt_mutex;

inline void check_strumpack(strumpack::ReturnCode code, std::string_view what)
{
  if (code == strumpack::ReturnCode::SUCCESS) return;
  std::ostringstream name;
  name << code;
  DDM_CHECK(false, "STRUMPACK: {} failed with {}", what, name.str());
}

// Passes the STRUMPACK options in config to STRUMPACK's command line parser (see StrumpackPrec)
template <class T>
void set_strumpack_options(strumpack::SPOptions<T>& options, const Dune::ParameterTree& config, bool root)
{
  // getopt skips argv[0], the program name, so a placeholder has to come first
  std::vector<std::string> args{"strumpack"};
  for (const auto& key : config.getValueKeys()) {
    const bool is_option = key.starts_with("sp_") || key.starts_with("hss_") || key.starts_with("blr_") || key.starts_with("hodlr_") || key == "help";
    if (!is_option || (key == "help" && !root)) continue;
    args.push_back("--" + key);
    args.push_back(config[key]);
  }
  if (args.size() == 1) return;

  std::vector<const char*> argv;
  for (const auto& arg : args) argv.push_back(arg.c_str());
  std::lock_guard lock(strumpack_getopt_mutex);
  options.set_from_command_line(static_cast<int>(argv.size()), argv.data());
}

inline bool is_mpi_root()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0;
}
} // namespace detail

/** Sparse LU factorization with STRUMPACK: exact by default, approximate with compression (e.g. sp_compression = blr).
 *  Sequential matrices only, apply_transposed() is not supported. STRUMPACK uses OpenMP, so with one MPI rank per core
 *  set OMP_NUM_THREADS=1.
 *
 *  Config: STRUMPACK's command line options, written as keys without the leading "--". Every key starting with sp_,
 *  hss_, blr_ or hodlr_, and the key help, is passed on as "--key value", in config order. Other keys are ignored.
 *   - Options without an argument still need a value in the config, but STRUMPACK ignores it: sp_verbose = 0 also
 *     turns the output on (use sp_quiet = 1 to turn it off).
 *   - help = 1 lists all options (printed on rank 0 only).
 *   - STRUMPACK silently ignores unknown options, so a misspelled key has no effect.
 *  Two defaults differ from STRUMPACK's:
 *   - sp_Krylov_solver = direct: one solve per apply, so the preconditioner is a fixed linear operator. STRUMPACK's
 *     default refines iteratively, or runs GMRES with compression.
 *   - sp_quiet: verbose output only when requested, and then only from rank 0.
 */
template <class T>
class StrumpackPrec final : public Prec<T> {
public:
  StrumpackPrec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , factorize_event_(Logger::get().registerOrGetEvent("STRUMPACK", "factorize"))
      , solve_event_(Logger::get().registerOrGetEvent("STRUMPACK", "solve"))
      , solve_multi_event_(Logger::get().registerOrGetEvent("STRUMPACK", "solve_multi"))
      , solver_(false, detail::is_mpi_root())
  {
    // TODO(20261006-192425): It should be possible to mark a preconditioner or solver as "sequential" on a higher level
    DDM_CHECK(this->mat()->sequential(), "STRUMPACK is a sequential solver");
    solver_.options().set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
    detail::set_strumpack_options(solver_.options(), config, detail::is_mpi_root());
    do_update();
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{solve_event_};
    if (r.size() == 0) return;
    detail::check_strumpack(solver_.solve(as_istl(r).data().data(), as_istl(z).data().data()), "solve");
  }

  void do_apply_multi(MultiVec<T>& Z, const MultiVec<T>& R) override
  {
    Logger::ScopedLog sl{solve_multi_event_};
    if (R.rows() == 0 || R.cols() == 0) return;
    const auto n = static_cast<int>(R.rows());
    detail::check_strumpack(solver_.solve(static_cast<int>(R.cols()), as_istl(R).data().data(), n, as_istl(Z).data().data(), n), "solve");
  }

  // The first call sets the matrix, later ones only update its values, so STRUMPACK reuses the reordering
  void do_update() override
  {
    Logger::ScopedLog sl{factorize_event_};
    const auto& A = this->mat()->local();
    const auto n = A.rows();
    if (n == 0) return;

    const auto csr = A.host_csr();
    if (matrix_set_) solver_.update_matrix_values(n, csr.row_ptr.data(), csr.cols.data(), csr.values.data());
    else solver_.set_csr_matrix(n, csr.row_ptr.data(), csr.cols.data(), csr.values.data());
    matrix_set_ = true;
    detail::check_strumpack(solver_.factor(), "factorization");
  }

  void do_info() const override
  {
    const auto& options = solver_.options();
    logger::info("STRUMPACK solver (reordering {}, compression {})", strumpack::get_name(options.reordering_method()), strumpack::get_name(options.compression()));
  }

  Logger::Event* factorize_event_;
  Logger::Event* solve_event_;
  Logger::Event* solve_multi_event_;
  strumpack::SparseSolver<T, Index> solver_;
  bool matrix_set_ = false;
};
} // namespace ddm

#endif
