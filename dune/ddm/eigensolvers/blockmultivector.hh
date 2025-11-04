#pragma once

#include "blas.hh"
#include "blockmatrix.hh"
#include "dune/ddm/helpers.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <experimental/simd>
#include <memory>
#include <omp.h>
#include <random>
#include <span>
#include <stdexcept>
#include <type_traits>

/**
 * @brief Check if a pointer is properly aligned for the specified alignment.
 * @param ptr Pointer to check
 * @param required_alignment Required alignment in bytes
 * @return true if pointer is aligned
 */
inline bool is_aligned(const void* ptr, std::size_t required_alignment) noexcept
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<std::uintptr_t>(ptr) % required_alignment == 0;
}

/**
 * @brief Create an aligned array using std::aligned_alloc with RAII semantics.
 * @tparam T Element type
 * @param size Number of elements to allocate
 * @param alignment Required alignment in bytes
 * @return unique_ptr managing the aligned memory
 * @throws std::bad_alloc if allocation fails
 */
template <typename T>
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays)
std::unique_ptr<T[], decltype(&std::free)> make_aligned_array(std::size_t size, std::size_t alignment)
{
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  T* raw_ptr = static_cast<T*>(std::aligned_alloc(alignment, sizeof(T) * size));
  if (!raw_ptr) throw std::bad_alloc{};
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays)
  return std::unique_ptr<T[], decltype(&std::free)>(raw_ptr, &std::free);
}

template <class RealT, std::size_t blocksize_, std::size_t alignment_>
class BlockView {
public:
  using Real = std::remove_cv_t<RealT>;
  static constexpr std::size_t alignment = alignment_;
  static constexpr std::size_t blocksize = blocksize_;

  using BlockMatView = typename DenseSquareBlockMatrix<Real, blocksize, alignment>::BlockView;
  using ConstBlockMatView = typename DenseSquareBlockMatrix<Real, blocksize, alignment>::ConstBlockView;

  BlockView(RealT* first, std::size_t rows)
      : data_(first)
      , rows_(rows)
  {
  }

  // Allow conversion from compatible types
  template <class OtherReal, std::enable_if_t<std::is_convertible_v<OtherReal, Real>, bool> = false>
  BlockView(const BlockView<OtherReal, blocksize_, alignment_>& other)
      : data_(other.data_)
      , rows_(other.rows_)
  {
  }

  BlockView(const BlockView&) = default;
  BlockView(BlockView&&) noexcept = default;
  ~BlockView() = default;

  BlockView& operator=(const BlockView& other)
  {
    if (this == &other) { return *this; } // Self-assignment check
    if (rows_ != other.rows_ or cols() != other.cols()) throw std::invalid_argument("BlockView copy assignment: The two block views are not compatible");

    std::copy_n(other.data_, other.size(), data_);
    return *this;
  }

  template <class T, std::size_t E, std::size_t A>
  BlockView& operator=(const BlockView<const T, E, A>& other)
  {
    static_assert(std::is_same_v<std::remove_cv_t<RealT>, std::remove_cv_t<T>>, "BlockView assignment requires compatible element types");
    static_assert(!std::is_const_v<RealT> || std::is_const_v<T>, "Cannot assign from non-const to const BlockView");

    if (rows_ != other.rows_ or cols() != other.cols()) throw std::invalid_argument("BlockView copy assignment: The two block views are not compatible");

    std::copy_n(other.data_, other.size(), data_);
    return *this;
  }

  // Move assignment operator
  BlockView& operator=(BlockView&&) noexcept = default;

  template <class Mat, class Real2>
  void apply_to_mat(const Mat& A, BlockView<Real2, blocksize, alignment> v) const
  {
    using VReal = std::experimental::fixed_size_simd<Real, blocksize>;
    constexpr std::size_t simd_size = VReal::size();
    const std::size_t cols_simd = (cols() / simd_size) * simd_size;

    if (cols() >= simd_size) {
// SIMD-optimized path for larger block sizes
#pragma omp parallel for schedule(static) default(none) shared(A, v) firstprivate(cols_simd, simd_size)
      for (auto ri = A.begin(); ri < A.end(); ++ri) {
        const std::size_t row_idx = ri.index();

        // Process SIMD-aligned chunks
        for (std::size_t j = 0; j < cols_simd; j += simd_size) {
          VReal vi{};

          for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
            const Real coeff = *ci;
            const std::size_t col_idx = ci.index();

            VReal xj;
            const Real* from = this->data() + col_idx * cols() + j;

            xj.copy_from(from, std::experimental::vector_aligned);

            vi += coeff * xj;
          }

          Real* to = v.data() + row_idx * cols() + j;
          vi.copy_to(to, std::experimental::vector_aligned);
        }

        // Handle remaining elements (scalar)
        for (std::size_t j = cols_simd; j < cols(); ++j) {
          Real sum = 0;
          for (auto ci = ri->begin(); ci != ri->end(); ++ci) sum += *ci * (*this)(ci.index(), j);
          v(row_idx, j) = sum;
        }
      }
    }
    else { // Very small block size (likely <8), just do scalar multiplications (but still parallel over the matrix rows)
#pragma omp parallel for schedule(static) default(none) shared(A, v)
      for (auto ri = A.begin(); ri < A.end(); ++ri) {
        const std::size_t row_idx = ri.index();

        // Initialize output row to zero
        for (std::size_t j = 0; j < cols(); ++j) v(row_idx, j) = 0;

        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          const Real coeff = *ci;
          const std::size_t col_idx = ci.index();

          for (std::size_t j = 0; j < cols(); ++j) v(row_idx, j) += coeff * (*this)(col_idx, j);
        }
      }
    }
  }

  template <class OtherBlockView>
  void dot(OtherBlockView V, BlockMatView R) const
  {
    // Compute R = this^T * V
    // this is rows_ x cols(), V is rows_ x cols(), R is cols() x cols()
    // For CblasRowMajor with CblasTrans: leading dimension is the stride in the original (non-transposed) matrix
    blas::gemm(CblasRowMajor, CblasTrans, CblasNoTrans, cols(), cols(), rows_, 1., data_, cols(), V.data(), cols(), 0., R.data(), cols());
  }

// Include the codegen'd dot product implementation
#include "dot.hh"

  template <class OtherBlockView>
  void mult_manual(BlockMatView R, OtherBlockView V) const
  {
    blas::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_, cols(), cols(), 1., data_, cols(), R.data(), cols(), 0., V.data(), cols());
  }

  template <class OtherBlockView>
  void mult_add(ConstBlockMatView R, OtherBlockView V) const
  {
    blas::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_, cols(), cols(), 1., data_, cols(), R.data(), cols(), 1., V.data(), cols());
  }

  template <class OtherBlockView>
  void mult_transpose(BlockMatView R, OtherBlockView V) const
  {
    blas::gemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows_, cols(), cols(), 1., data_, cols(), R.data(), cols(), 0., V.data(), cols());
  }

  template <class OtherBlockView>
  void mult(BlockMatView R, OtherBlockView V) const
  {
    const auto m = this->rows(); // rows of A = rows_
    const auto k = this->cols(); // cols of A = inner dimension
    const auto n = R.cols();     // cols of B

    using VReal = std::experimental::fixed_size_simd<Real, blocksize>;
    constexpr std::size_t simd_size = VReal::size();

#pragma omp parallel for schedule(static) default(none) shared(V, R) firstprivate(m, n, k, simd_size)
    for (std::size_t i = 0; i < m; ++i) {
      std::size_t n_simd = (n / simd_size) * simd_size;

      for (std::size_t j = 0; j < n_simd; j += simd_size) {
        VReal sum_vec{};

        for (std::size_t p = 0; p < k; ++p) {
          Real a_ip = (*this)(i, p);
          VReal r_vec;
          r_vec.copy_from(&R(p, j), std::experimental::vector_aligned);

          if (p == 0) sum_vec = a_ip * r_vec;
          else sum_vec += a_ip * r_vec;
        }

        sum_vec.copy_to(&V(i, j), std::experimental::vector_aligned);
      }

      // Handle remaining elements
      for (std::size_t j = n_simd; j < n; ++j) {
        for (std::size_t p = 0; p < k; ++p) {
          Real a_ip = (*this)(i, p);
          if (p == 0) V(i, j) = a_ip * R(p, j);
          else V(i, j) += a_ip * R(p, j);
        }
      }
    }
  }

  void set_random(std::size_t seed = 1)
  {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> dist(-0.5, 0.5);
    for (std::size_t i = 0; i < rows_ * blocksize; ++i) data_[i] = dist(rng);
  }

  RealT& operator()(std::size_t row, std::size_t col)
  {
    assert(row < rows_);
    assert(col < cols());
    return data_[row * cols() + col];
  }

  const RealT& operator()(std::size_t row, std::size_t col) const
  {
    assert(row < rows_);
    assert(col < cols());
    return data_[row * cols() + col];
  }

  RealT* row_start(std::size_t row)
  {
    assert(row < rows_);
    return &data_[row * cols()];
  }

  const RealT* row_start(std::size_t row) const
  {
    assert(row < rows_);
    return &data_[row * cols()];
  }

  [[deprecated("Use frobenius_norm instead")]]
  RealT two_norm() const noexcept
  {
    RealT norm{0};
    for (std::size_t i = 0; i < size(); ++i) norm += data_[i] * data_[i];
    return std::sqrt(norm);
  }

  RealT frobenius_norm() const noexcept
  {
    RealT norm{0};
    for (std::size_t i = 0; i < size(); ++i) norm += data_[i] * data_[i];
    return std::sqrt(norm);
  }

  BlockView& operator-=(const BlockView& v)
  {
    // TODO: Check that the two views actually match in size
    for (std::size_t i = 0; i < size(); ++i) data_[i] -= v.data_[i];
    return *this;
  }

  BlockView& operator+=(const BlockView& v)
  {
    // TODO: Check that the two views actually match in size
    for (std::size_t i = 0; i < size(); ++i) data_[i] += v.data_[i];
    return *this;
  }

  BlockView& operator*=(Real r)
  {
    for (std::size_t i = 0; i < size(); ++i) data_[i] *= r;
    return *this;
  }

  static constexpr std::size_t cols() { return blocksize; }
  std::size_t rows() const { return rows_; }
  std::size_t size() const { return cols() * rows_; }

  RealT* data() { return data_; }
  const RealT* data() const { return data_; }

private:
  template <class, std::size_t, std::size_t>
  friend class BlockView;

  RealT* data_;
  std::size_t rows_;
};

/**
 * @brief A memory-aligned multi-vector container organized in blocks for efficient SIMD operations.
 *
 * This class provides a storage layout where vectors are grouped into blocks of a specified size.
 * Within each block, data is stored in row-major order. Each block is aligned to SIMD boundaries
 * for optimal vectorized operations. The class supports both compile-time and runtime block sizes.
 *
 * @tparam RealT The scalar type (e.g., double, float)
 * @tparam blocksize_ Block size
 * @tparam alignment_ Memory alignment in bytes (defaults to SIMD alignment for RealT)
 */
template <class RealT, std::size_t blocksize_, std::size_t alignment_ = std::experimental::memory_alignment_v<std::experimental::fixed_size_simd<RealT, blocksize_>>>
class BlockMultiVector {
public:
  using Real = RealT;                                  ///< Scalar type
  static constexpr std::size_t alignment = alignment_; ///< Memory alignment in bytes
  static constexpr std::size_t blocksize = blocksize_; ///< Blocksize

  using BlockViewType = BlockView<Real, blocksize, alignment>;
  using ConstBlockViewType = BlockView<const Real, blocksize, alignment>;

  // Matching dense block matrix
  using BlockMatrix = DenseSquareBlockMatrix<Real, blocksize, alignment>;
  using BlockMatrixBlockView = typename BlockMatrix::BlockView;
  using DenseBlockMatrixBlockView = BlockMatrixBlockView; // Alias for compatibility

  /**
   * @brief Construct a BlockMultiVector with specified dimensions.
   *
   * In this case, the blocksize is defined via the template parameter.
   *
   * @param rows Number of rows in each block
   * @param cols Total number of columns (must be divisible by blocksize)
   * @param blocksize Number of vectors per block
   * @throws std::invalid_argument if cols is not divisible by blocksize
   * @throws std::bad_alloc if memory allocation fails
   */
  BlockMultiVector(std::size_t rows, std::size_t cols)
      : rows_(rows)
      , cols_(cols)
      , data_ptr(nullptr, &std::free)
  {
    // Validate alignment is power of 2 (required by std::aligned_alloc)
    static_assert((alignment & (alignment - 1)) == 0, "Alignment must be a power of 2");

    // Validate dimensions
    if (rows == 0) throw std::invalid_argument("BlockMultiVector: rows (" + std::to_string(rows) + ") must be > 0");
    if (cols == 0) throw std::invalid_argument("BlockMultiVector: cols (" + std::to_string(cols) + ") must be > 0");
    if (blocksize == 0) throw std::invalid_argument("BlockMultiVector: blocksize (" + std::to_string(blocksize) + ") must be > 0");
    if (cols_ % blocksize != 0) throw std::invalid_argument("BlockMultiVector: cols (" + std::to_string(cols_) + ") must be divisible by blocksize (" + std::to_string(blocksize) + ")");

    // Allocate aligned memory
    data_ptr = make_aligned_array<RealT>(rows_ * cols_, alignment);
    assert(is_aligned(data_ptr.get(), alignment));

    // Sanity check: Ensure every block start is aligned
    for (std::size_t b = 0; b < blocks(); ++b) {
      RealT* ptr = data_ptr.get() + (b * rows_ * blocksize);
      assert(is_aligned(ptr, alignment));
    }
  }

  /**
   * @brief Copy constructor - creates a deep copy of another BlockMultiVector.
   * @param other The BlockMultiVector to copy from
   * @throws std::bad_alloc if memory allocation fails
   */
  BlockMultiVector(const BlockMultiVector& other)
      : rows_(other.rows_)
      , cols_(other.cols_)
      , data_ptr(make_aligned_array<RealT>(rows_ * cols_, alignment))
  {
    assert(is_aligned(data_ptr.get(), alignment));
    std::copy_n(other.data_ptr.get(), rows_ * cols_, data_ptr.get());
  }

  /**
   * @brief Move constructor - transfers ownership from another BlockMultiVector.
   * @param other The BlockMultiVector to move from (will be left in valid but unspecified state)
   */
  BlockMultiVector(BlockMultiVector&& other) noexcept
      : rows_(other.rows_)
      , cols_(other.cols_)
      , data_ptr(std::move(other.data_ptr))
  {
  }

  /**
   * @brief Move assignment operator - transfers ownership from another BlockMultiVector.
   * @param other The BlockMultiVector to move from
   * @return Reference to this object
   */
  BlockMultiVector& operator=(BlockMultiVector&& other) noexcept
  {
    if (this != &other) {
      data_ptr = std::move(other.data_ptr);
      rows_ = other.rows_;
      cols_ = other.cols_;
    }
    return *this;
  }

  /**
   * @brief Copy assignment operator - assigns data from another compatible BlockMultiVector.
   * @param other The BlockMultiVector to copy from (must have same dimensions and alignment)
   * @return Reference to this object
   * @throws std::invalid_argument if the two BlockMultiVectors are not compatible
   */
  BlockMultiVector& operator=(const BlockMultiVector& other)
  {
    if (this != &other) {
      if (not matches(other)) throw std::invalid_argument("BlockMultiVector copy assignment: The two block multivectors are not compatible");
      std::copy_n(other.data_ptr.get(), rows_ * cols_, data_ptr.get());
    }
    return *this;
  }

  /** @brief Destructor - unique_ptr automatically handles memory cleanup. */
  ~BlockMultiVector() = default;

  /**
   * @brief Get a mutable view of a specific block.
   * @param block Block index (must be < blocks())
   * @return BlockView providing access to the specified block
   */
  BlockViewType block_view(std::size_t block)
  {
    assert(block < blocks());
    RealT* ptr = data_ptr.get() + (block * rows_ * blocksize);
    assert(is_aligned(ptr, alignment));
    return {ptr, rows_};
  }

  /**
   * @brief Get a const view of a specific block.
   * @param block Block index (must be < blocks())
   * @return Const BlockView providing read-only access to the specified block
   */
  ConstBlockViewType block_view(std::size_t block) const
  {
    assert(block < blocks());
    const RealT* ptr = data_ptr.get() + (block * rows_ * blocksize);
    assert(is_aligned(ptr, alignment));
    return {ptr, rows_};
  }

  /**
   * @brief Check if this BlockMultiVector is compatible with another for operations.
   * @tparam RealT2 Scalar type of the other BlockMultiVector
   * @tparam alignment2 Alignment of the other BlockMultiVector
   * @tparam blocksize_extent2 Block size extent of the other BlockMultiVector
   * @param other The other BlockMultiVector to check compatibility with
   * @return true if compatible (same type, alignment, and dimensions)
   */
  template <class RealT2, std::size_t blocksize_extent2, std::size_t alignment2>
  bool matches(const BlockMultiVector<RealT2, blocksize_extent2, alignment2>& other) const
  {
    // clang-format off
    return std::is_same_v<RealT, RealT2>     
           && alignment == alignment2
           && rows_ == other.rows_
           && cols_ == other.cols_
           && blocksize == other.blocksize;
    // clang-format on
  }

  /**
   * @brief Check if one block of this BlockMultiVector is compatible with another for operations.
   *
   * This does essentially the same thing as matches() except for checking that both Multivecs
   * have the same number of columns.
   *
   * @tparam RealT2 Scalar type of the other BlockMultiVector
   * @tparam alignment2 Alignment of the other BlockMultiVector
   * @tparam blocksize_extent2 Block size extent of the other BlockMultiVector
   * @param other The other BlockMultiVector to check compatibility with
   * @return true if compatible (same type, alignment, and dimensions)
   */
  template <class RealT2, std::size_t blocksize_extent2, std::size_t alignment2>
  bool matches_blocks(const BlockMultiVector<RealT2, blocksize_extent2, alignment2>& other) const
  {
    // clang-format off
    return std::is_same_v<RealT, RealT2>
           && alignment == alignment2
           && rows_ == other.rows_
           && blocksize == other.blocksize;
    // clang-format on
  }

  /**
   * @brief Fill all elements with random values from a normal distribution.
   * @param seed Random seed for reproducible results
   */
  void set_random(std::size_t seed = 1)
  {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> dist(-0.5, 0.5);
    for (std::size_t i = 0; i < rows_ * cols_; ++i) data_ptr[i] = dist(rng);
  }

  /** @brief Multiply this vector blockwise by matrix and store the result in \p v */
  template <class Mat>
  void apply_to_mat(const Mat& A, BlockMultiVector& v) const
  {
    for (std::size_t i = 0; i < blocks(); ++i) {
      auto xb = block_view(i);
      auto vb = v.block_view(i);
      xb.apply_to_mat(A, vb);
    }
  }

  /** @brief Block-wise dot product */
  void dot(const BlockMultiVector& v, BlockMatrix& R) const
  {
    for (std::size_t i = 0; i < blocks(); ++i) {
      auto xi = block_view(i);
      for (std::size_t j = 0; j < v.blocks(); ++j) {
        auto bj = v.block_view(j);
        auto rij = R.block_view(i, j);
        xi.dot(bj, rij);
      }
    }
  }

  /** @brief Block-wise dot product but only up to block \p block_to */
  void dot(const BlockMultiVector& v, BlockMatrix& R, std::size_t block_to) const
  {
    for (std::size_t i = 0; i < block_to; ++i) {
      auto xi = block_view(i);
      for (std::size_t j = 0; j < block_to; ++j) {
        auto bj = v.block_view(j);
        auto rij = R.block_view(i, j);
        xi.dot(bj, rij);
      }
    }
  }

  /** @brief Blockvector times square block matrix */
  void mult(const BlockMatrix& R, BlockMultiVector& v)
  {
    v.set_zero();
    for (std::size_t i = 0; i < blocks(); ++i) {
      auto vi = v.block_view(i);

      for (std::size_t k = 0; k < blocks(); ++k) {
        auto rki = R.block_view(k, i);
        auto xk = block_view(k);

        xk.mult_add(rki, vi);
      }
    }
  }

  /** @brief Blockvector times part of a square block matrix.
   *
   *  This method computes V[0 : block_to - 1] = Q * R[:, 0 : block_to - 1]
   *
   *  @note All blocks of \p V are zeroed.
   */
  void mult(const BlockMatrix& R, BlockMultiVector& V, std::size_t block_to)
  {
    V.set_zero();
    for (std::size_t i = 0; i < block_to; ++i) {
      auto vi = V.block_view(i);

      for (std::size_t k = 0; k < blocks() - 1; ++k) {
        auto rki = R.block_view(k, i);
        auto xk = block_view(k);

        xk.mult_add(rki, vi);
      }
    }
  }

  BlockMultiVector& operator-=(const BlockMultiVector& v)
  {
    if (!this->matches(v)) throw std::invalid_argument("BlockMultiVector operator-=(): The multivectors are not compatible");

    for (std::size_t i = 0; i < rows_ * cols_; ++i) data_ptr[i] -= v.data_ptr[i];

    return *this;
  }

  BlockMultiVector& operator+=(const BlockMultiVector& v)
  {
    if (!this->matches(v)) throw std::invalid_argument("BlockMultiVector operator+=(): The multivectors are not compatible");

    for (std::size_t i = 0; i < rows_ * cols_; ++i) data_ptr[i] += v.data_ptr[i];

    return *this;
  }

  BlockMultiVector& operator*=(Real scalar)
  {
    for (std::size_t i = 0; i < rows_ * cols_; ++i) data_ptr[i] *= scalar;

    return *this;
  }

  Real frobenius_norm() const
  {
    Real res{0};
    for (std::size_t i = 0; i < rows_ * cols_; ++i) res += data_ptr[i] * data_ptr[i];
    return std::sqrt(res);
  }

  Real max_norm() const { return *std::max_element(data_ptr.get(), data_ptr.get() + rows_ * cols_); }

  void scale_columns(const std::vector<Real>& scale)
  {
    assert(scale.size() == cols());

    for (std::size_t block = 0; block < blocks(); ++block) {
      auto bv = block_view(block);
      for (std::size_t i = 0; i < bv.rows(); ++i)
        for (std::size_t j = 0; j < bv.cols(); ++j) bv(i, j) *= scale.at(block * blocksize + j);
    }
  }

  /** @brief Set all elements to zero. */
  void set_zero() { std::fill_n(data_ptr.get(), rows_ * cols_, 0); }

  /** @brief Get the number of blocks. */
  std::size_t blocks() const noexcept { return cols_ / blocksize; }

  /** @brief Get the number of rows per block. */
  std::size_t rows() const noexcept { return rows_; }

  /** @brief Get the total number of columns. */
  std::size_t cols() const noexcept { return cols_; }

  void lock_leading_blocks(std::size_t k) { locked_leading_blocks = k; }

private:
  template <class>
  friend class UMFPackMultivecSolver;

  /** @brief Get the number of alignment elements (alignment in bytes / sizeof(RealT)). */
  static constexpr std::size_t align_elements() noexcept { return alignment / sizeof(RealT); }

  /**
   * @brief Round up a size to the next alignment boundary.
   * @param size Size in elements to align
   * @return Size rounded up to alignment boundary
   */
  static std::size_t align_up(std::size_t size) noexcept
  {
    std::size_t align_elems = align_elements();
    std::size_t misalignment = size % align_elems;
    return (misalignment == 0) ? size : size + (align_elems - misalignment);
  }

  std::size_t locked_leading_blocks{};
  std::size_t rows_; ///< Number of rows per block
  std::size_t cols_; ///< Total number of columns across all blocks
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays)
  std::unique_ptr<Real[], decltype(&std::free)> data_ptr; ///< Smart pointer to aligned memory containing all block data
};
