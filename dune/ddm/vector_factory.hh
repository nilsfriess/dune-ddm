#pragma once

/** @file vector_factory.hh
 *
 *  @brief Creates the vector type that is canonically paired with a matrix type.
 *
 *  A GPU-resident ddm::Sycl::Mat pairs with a ddm::Sycl::Vec living on the
 *  matrix's queue; a host Dune::BCRSMatrix pairs with a Dune::BlockVector.
 *  Centralizing this pairing in one function keeps the invariant that a
 *  vector created for a matrix is compatible with it (same memory backend,
 *  same queue, matching block types) true by construction, at every call
 *  site that needs a working vector for a given matrix.
 */

#if __has_include(<sycl/sycl.hpp>)
#define DDM_HAVE_SYCL
#endif

#include "backend/backend.hh"
#include "sycl/mat.hh"
#include "sycl/vec.hh"

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

namespace ddm {

/// Host pairing: BCRSMatrix with FieldMatrix blocks -> BlockVector with
/// FieldVector blocks of matching field type and size.
// TODO: also use the allocator of the parent matrix for the vector
template <class F, class Allocator>
auto create_vector_for_matrix(const Dune::BCRSMatrix<F, Allocator>& A)
{
  return Dune::BlockVector<Dune::FieldVector<typename F::field_type, F::rows>>(A.N());
}

#ifdef DDM_HAVE_SYCL
/// SYCL pairing: a GPU-resident matrix yields a GPU-resident vector on the
/// same queue as the matrix.
template <class Scalar, class Index>
auto create_vector_for_matrix(const ddm::Sycl::Mat<Scalar, Index>& A)
{
  return ddm::Sycl::Vec<Scalar, Index>(A.queue(), A.N());
}
#endif

// TODO: Come up with some way to not copy here
template <class Container, class F, class Allocator>
auto create_vector_like_from_host([[maybe_unused]] const Dune::BlockVector<F, Allocator>& template_vector, const Container& host_vector)
{
  Dune::BlockVector<F, Allocator> v(host_vector.size());
  std::copy_n(host_vector.data(), host_vector.size(), v.data());
  return v;
}

#ifdef DDM_HAVE_SYCL
template <class Container, class Scalar, class Index>
auto create_vector_like_from_host(const Sycl::Vec<Scalar, Index>& template_vector, const Container& host_vector)
{
  return Sycl::Vec<Scalar, Index>::from_host_vector(template_vector.queue(), host_vector);
}
#endif

/** @brief Creates a zero vector with the same type, size and backend as \p v.
 *
 *  Useful wherever generic code needs scratch storage shaped like an incoming vector, e.g. in
 *  CombinedPreconditioner.
 */
template <class V>
auto create_zero_vector_like(const V& v)
{
  if constexpr (backend::IsGpuResident<V>) {
    V w(v.queue(), v.size());
    w = 0;
    return w;
  } else {
    V w(v.size());
    w = 0;
    return w;
  }
}

} // namespace ddm

#ifdef DDM_HAVE_SYCL
#undef DDM_HAVE_SYCL
#endif
