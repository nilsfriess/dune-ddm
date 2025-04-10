#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

template <class Mat, class Vec>
class EnergyMinimalExtension {
public:
  EnergyMinimalExtension(const Mat &A, const std::vector<std::size_t> &interior_to_subdomain, const std::vector<std::size_t> &ring_to_subdomain, bool inexact = false)
      : A(A), interior_to_subdomain(interior_to_subdomain), ring_to_subdomain(ring_to_subdomain)
  {
    std::unordered_map<std::size_t, std::size_t> subdomain_to_interior;
    subdomain_to_interior.reserve(interior_to_subdomain.size());
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      subdomain_to_interior[interior_to_subdomain[i]] = i;
    }

    const auto N = interior_to_subdomain.size();
    auto Aint = std::make_shared<Mat>();
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

    if (not inexact) {
      solver = std::make_unique<Dune::UMFPack<Mat>>(*Aint);
    }
    else {
      auto prec = std::make_shared<Dune::SeqILU<Mat, Vec, Vec>>(*Aint, 1.0);
      auto op = std::make_shared<Dune::MatrixAdapter<Mat, Vec, Vec>>(Aint);
      auto sp = std::make_shared<Dune::SeqScalarProduct<Vec>>();
      solver = std::make_unique<Dune::RestartedGMResSolver<Vec>>(op, sp, prec, 1e-8, 50, 100, 0);
    }
  }

  EnergyMinimalExtension(const EnergyMinimalExtension &) = delete;
  EnergyMinimalExtension(const EnergyMinimalExtension &&) = delete;
  EnergyMinimalExtension &operator=(const EnergyMinimalExtension &) = delete;
  EnergyMinimalExtension &operator=(const EnergyMinimalExtension &&) = delete;
  ~EnergyMinimalExtension() = default;

  Vec extend(const Vec &v_ring)
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

private:
  const Mat &A;
  const std::vector<std::size_t> &interior_to_subdomain;
  const std::vector<std::size_t> &ring_to_subdomain;

  std::unique_ptr<Dune::InverseOperator<Vec, Vec>> solver;
};
