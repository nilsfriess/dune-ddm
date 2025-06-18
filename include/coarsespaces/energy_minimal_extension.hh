#pragma once

#include <cstddef>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solvers.hh>
#include <memory>
#include <unordered_map>
#include <vector>

#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#if DUNE_DDM_HAVE_UMFPACK_SIMD
#include <experimental/simd>
#endif

#include "strumpack.hh"

template <class Mat, class Vec>
class EnergyMinimalExtension {
public:
  EnergyMinimalExtension(const Mat &A, const std::vector<std::size_t> &interior_to_subdomain, const std::vector<std::size_t> &ring_to_subdomain, bool inexact = false)
      : A(A), interior_to_subdomain(interior_to_subdomain), ring_to_subdomain(ring_to_subdomain), inexact(inexact)
  {
    std::unordered_map<std::size_t, std::size_t> subdomain_to_interior;
    subdomain_to_interior.reserve(interior_to_subdomain.size());
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      subdomain_to_interior[interior_to_subdomain[i]] = i;
    }

    if (not inexact) {
      const auto N = interior_to_subdomain.size();
      Aint = std::make_shared<Mat>();
      auto avg = A.nonzeroes() / A.N() + 2;
      Aint->setBuildMode(Mat::implicit);
      Aint->setImplicitBuildModeParameters(avg, 0.2);
      Aint->setSize(N, N);
      for (auto ri = A.begin(); ri != A.end(); ++ri) {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (subdomain_to_interior.contains(ri.index()) and subdomain_to_interior.contains(ci.index())) {
            Aint->entry(subdomain_to_interior[ri.index()], subdomain_to_interior[ci.index()]) = *ci;
          }
        }
      }
      Aint->compress();

#ifndef DUNE_DDM_HAVE_STRUMPACK
      solver = std::make_unique<Dune::UMFPack<Mat>>(*Aint);
      solver->setOption(UMFPACK_IRSTEP, 0);
#else
      solver = std::make_unique<Dune::STRUMPACK<Mat>>(*Aint);
#endif
    }
    else {
      // Here we just copy A, but only those entries corresponding to interior dofs
      const auto N = A.N();
      Aint = std::make_shared<Mat>();
      auto avg = A.nonzeroes() / A.N() + 2;
      Aint->setBuildMode(Mat::implicit);
      Aint->setImplicitBuildModeParameters(avg, 0.2);
      Aint->setSize(N, N);
      for (auto ri = A.begin(); ri != A.end(); ++ri) {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (subdomain_to_interior.contains(ri.index()) and subdomain_to_interior.contains(ci.index())) {
            Aint->entry(ri.index(), ci.index()) = *ci;
          }
        }
      }
      Aint->compress();

      Ainv = std::make_unique<Dune::UMFPack<Mat>>(A);
      Ainv->setOption(UMFPACK_IRSTEP, 0);

      subdomain_prec = std::make_unique<Dune::InverseOperator2Preconditioner<Dune::UMFPack<Mat>>>(*Ainv);
      Aintop = std::make_unique<Dune::MatrixAdapter<Mat, Vec, Vec>>(Aint);
      subdomain_solver = std::make_unique<Dune::LoopSolver<Vec>>(*Aintop, *subdomain_prec, 1e-5, 1, 0);
    }
  }

  EnergyMinimalExtension(const EnergyMinimalExtension &) = delete;
  EnergyMinimalExtension(const EnergyMinimalExtension &&) = delete;
  EnergyMinimalExtension &operator=(const EnergyMinimalExtension &) = delete;
  EnergyMinimalExtension &operator=(const EnergyMinimalExtension &&) = delete;
  ~EnergyMinimalExtension() = default;

  Vec extend(const Vec &v_ring)
  {
    if (inexact) {
      return extend_inexact(v_ring);
    }
    else {
      return extend_exact(v_ring);
    }
  }

  Vec extend_inexact(const Vec &v_ring)
  {
    // Copy the values on the ring to a vector that lives on the whole subdomain
    Vec v_full(A.N());
    v_full = 0;
    for (std::size_t i = 0; i < v_ring.N(); ++i) {
      v_full[ring_to_subdomain[i]] = v_ring[i];
    }

    // Multiply by the whole subdomain matrix
    Vec A_vfull(A.N());
    A.mv(v_full, A_vfull);

    // Zero the values outside the interior
    v_full = 0;
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      v_full[interior_to_subdomain[i]] = A_vfull[interior_to_subdomain[i]];
    }

    Vec v_res(A.N());
    v_res = 0;
    Dune::InverseOperatorResult res;
    subdomain_solver->apply(v_res, v_full, res);
    spdlog::get("all_ranks")->debug("Inexact harmonic extension took {} iterations", res.iterations);

    v_res *= -1.;

    // Extract values in interior
    Vec v_res_int(interior_to_subdomain.size());
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      v_res_int[i] = v_res[interior_to_subdomain[i]];
    }
    return v_res_int;
  }

  Vec extend_exact(const Vec &v_ring)
  {
    // Copy the values on the ring to a vector that lives on the whole subdomain
    Vec v_full(A.N());
    v_full = 0;
    for (std::size_t i = 0; i < v_ring.N(); ++i) {
      v_full[ring_to_subdomain[i]] = v_ring[i];
    }

    // Multiply by the whole subdomain matrix
    Vec A_vfull(A.N());
    A.mv(v_full, A_vfull);

    // Extract the values in the interior
    Vec v_int(interior_to_subdomain.size());
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      v_int[i] = A_vfull[interior_to_subdomain[i]];
    }

    // Apply inverse of interior matrix
    Vec v_res(interior_to_subdomain.size());
    v_res = 0;
    Dune::InverseOperatorResult res;
    solver->apply(v_res, v_int, res);
    v_res *= -1.;

    return v_res;
  }

#if DUNE_DDM_HAVE_UMFPACK_SIMD
  std::vector<Vec> extend(const std::vector<Vec> &vs)
  {
    namespace stdx = std::experimental;

    using Scalar = double;
    using ScalarV = stdx::native_simd<Scalar>;

    if (vs.size() % ScalarV::size() != 0) {
      DUNE_THROW(Dune::Exception, "Number of vectors must be divisible by SIMD width " + std::to_string(ScalarV::size()) + "\n");
    }

    Vec zero(interior_to_subdomain.size());
    zero = 0;
    std::vector<Vec> v_res(vs.size(), zero);

    auto numeric_data = solver->get_numeric_data();

    std::vector<ScalarV> vs_v(interior_to_subdomain.size());
    for (std::size_t k = 0; k < vs.size() / ScalarV::size(); ++k) {
      auto block_start = k * ScalarV::size();

      for (std::size_t i = block_start; i < block_start + ScalarV::size(); ++i) {
        Vec v_full(A.N());
        v_full = 0;
        for (std::size_t j = 0; j < vs[i].N(); ++j) {
          v_full[ring_to_subdomain[j]] = vs[i][j];
        }

        // Multiply by the whole subdomain matrix
        Vec A_vfull(A.N());
        A.mv(v_full, A_vfull);

        // Extract the values in the interior
        for (std::size_t j = 0; j < interior_to_subdomain.size(); ++j) {
          vs_v[j][i % ScalarV::size()] = A_vfull[interior_to_subdomain[j]];
        }
      }

      // Apply inverse of interior matrix to all vectors of the current block
      std::vector<ScalarV> vres_v(interior_to_subdomain.size());
      solver->apply_simd(vres_v, vs_v, numeric_data);

      for (std::size_t i = block_start; i < block_start + ScalarV::size(); ++i) {
        for (std::size_t j = 0; j < interior_to_subdomain.size(); ++j) {
          v_res[i][j] = -vres_v[j][i % ScalarV::size()];
        }
      }
    }

    return v_res;
  }
#endif

private:
  const Mat &A;
  const std::vector<std::size_t> &interior_to_subdomain;
  const std::vector<std::size_t> &ring_to_subdomain;

#ifndef DUNE_DDM_HAVE_STRUMPACK
  std::unique_ptr<Dune::UMFPack<Mat>> solver;
#else
  std::unique_ptr<Dune::STRUMPACK<Mat>> solver;
#endif

  bool inexact;
  std::unique_ptr<Dune::UMFPack<Mat>> Ainv;
  std::unique_ptr<Dune::InverseOperator2Preconditioner<Dune::UMFPack<Mat>>> subdomain_prec;
  std::unique_ptr<Dune::IterativeSolver<Vec, Vec>> subdomain_solver;
  std::unique_ptr<Dune::Preconditioner<Vec, Vec>> prec;
  std::shared_ptr<Mat> Aint;
  std::unique_ptr<Dune::MatrixAdapter<Mat, Vec, Vec>> Aintop;
};
