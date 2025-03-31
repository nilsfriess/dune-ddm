#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <dune/istl/umfpack.hh>

template <class Mat>
class EnergyMinimalExtension {
  using Solver = Dune::UMFPack<Mat>;

public:
  EnergyMinimalExtension(const Mat &A, const std::vector<std::size_t> &interior_to_subdomain, const std::vector<std::size_t> &ring_to_subdomain)
      : A(A), interior_to_subdomain(interior_to_subdomain), ring_to_subdomain(ring_to_subdomain)
  {
    std::unordered_map<std::size_t, std::size_t> subdomain_to_interior;
    subdomain_to_interior.reserve(interior_to_subdomain.size());
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      subdomain_to_interior[interior_to_subdomain[i]] = i;
    }

    const auto N = interior_to_subdomain.size();
    Mat Aint;
    auto avg = A.nonzeroes() / A.N() + 2;
    Aint.setBuildMode(Mat::implicit);
    Aint.setImplicitBuildModeParameters(avg, 0.2);
    Aint.setSize(N, N);
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        if (subdomain_to_interior.contains(ri.index()) and subdomain_to_interior.contains(ci.index())) {
          Aint.entry(subdomain_to_interior[ri.index()], subdomain_to_interior[ci.index()]) = *ci;
        }
      }
    }
    Aint.compress();

    solver = std::make_unique<Solver>(Aint);
  }

  EnergyMinimalExtension(const EnergyMinimalExtension &) = delete;
  EnergyMinimalExtension(const EnergyMinimalExtension &&) = delete;
  EnergyMinimalExtension &operator=(const EnergyMinimalExtension &) = delete;
  EnergyMinimalExtension &operator=(const EnergyMinimalExtension &&) = delete;
  ~EnergyMinimalExtension() = default;

  template <class Vec>
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

    // Combine with the original vector; reuse vfull for that
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      v_full[interior_to_subdomain[i]] = -1 * v_res[i];
    }

    return v_full;
  }

private:
  const Mat &A;
  const std::vector<std::size_t> &interior_to_subdomain;
  const std::vector<std::size_t> &ring_to_subdomain;

  std::unique_ptr<Solver> solver;
};
