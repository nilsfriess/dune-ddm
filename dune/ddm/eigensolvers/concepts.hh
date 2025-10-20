#pragma once

#include <concepts>
#include <memory>

template <class EVP>
concept Eigenproblem = requires(EVP& evp, const typename EVP::BlockMultiVec& x, typename EVP::BlockMultiVec& y) {
  typename EVP::Real;
  typename EVP::BlockMultiVec;
  typename EVP::InnerProduct;

  /** @brief Application of the operator that defines the eigenvalue problem

      In a standard eigenvalue problem Ax = λx this usually implements y = Ax.
      For a generalised eigenvalue problem Ax = λBx this could, for instance,
      implement the shift-invert transformed operator y = (A - σB)^-1 B x.
   */
  { evp.apply(x, y) } -> std::same_as<void>;

  /** @brief Returns true if the problem is real symmetric */
  { evp.is_symmetric() } -> std::same_as<bool>;

  /** @brief Gets the inner product associated with the problem */
  { evp.get_inner_product() } -> std::convertible_to<std::shared_ptr<typename EVP::InnerProduct>>;
};

/** @brief Concept for inner product operations used in orthogonalisation
 *
 * An inner product must provide:
 * - A way to compute the Gram matrix between two vector blocks using the inner product
 * - Type information for the scalar and vector types
 *
 * Examples:
 * - Standard Euclidean inner product: <x,y> = x^T * y
 * - Matrix-induced inner product: <x,y>_M = x^T * M * y
 * - Mass matrix inner product for finite elements
 */
template <class IP>
concept InnerProduct = requires(IP& ip, const typename IP::BlockMultiVec& x, const typename IP::BlockMultiVec& y, typename IP::BlockMatrix& R, typename IP::ConstBlockMultiVecView xv,
                                typename IP::ConstBlockMultiVecView yv, typename IP::BlockMatrixBlockView Rv) {
  // Type requirements
  typename IP::Real;
  typename IP::BlockMultiVec;
  typename IP::BlockMultiVecView;
  typename IP::ConstBlockMultiVecView;
  typename IP::BlockMatrix;
  typename IP::BlockMatrixBlockView;

  /** @brief Dot product for a whole block vector */
  { ip.dot(x, y, R) } -> std::same_as<void>;

  /** @brief Dot product for a indivdual blocks */
  { ip.dot(xv, yv, Rv) } -> std::same_as<void>;

  // Query if this is the standard Euclidean inner product
  { ip.is_euclidean() } -> std::same_as<bool>;
};
