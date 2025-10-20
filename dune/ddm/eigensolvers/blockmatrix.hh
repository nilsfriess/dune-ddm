#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

template <class RealT, std::size_t order>
class DenseMatrixBlockView {
public:
  using Real = RealT;

  explicit DenseMatrixBlockView(Real* it)
      : data_(it)
  {
  }

  // Explicit copy constructor to avoid deprecated warnings
  DenseMatrixBlockView(const DenseMatrixBlockView& other) = default;

  // Destructor (default since this is a view, doesn't own memory)
  ~DenseMatrixBlockView() = default;

  // Move operations
  DenseMatrixBlockView(DenseMatrixBlockView&& other) = default;
  DenseMatrixBlockView& operator=(DenseMatrixBlockView&& other) = default;

  Real* data() { return data_; }

  Real& operator()(std::size_t i, std::size_t j) { return data_[(i * order) + j]; }
  const Real& operator()(std::size_t i, std::size_t j) const { return data_[(i * order) + j]; }

  void copy_to_flat(std::vector<RealT>& flat) const
  {
    flat.resize(order * order);
    std::copy_n(data_, data_ + order * order, flat.begin());
  }

  void transpose()
  {
    for (std::size_t i = 0; i < order; ++i)
      for (std::size_t j = i + 1; j < order; ++j) std::swap(data_[i * order + j], data_[j * order + i]);
  }

  void set_zero()
  {
    for (std::size_t i = 0; i < order * order; ++i) data_[i] = 0.0;
  }

  std::size_t cols() const { return order; }
  std::size_t rows() const { return order; }

  /** @brief Computes the Frobenius norm of the matrix block */
  Real frobenius_norm() const
  {
    Real norm_sq = 0;
    for (std::size_t i = 0; i < rows(); ++i)
      for (std::size_t j = 0; j < cols(); ++j) norm_sq += (*this)(i, j) * (*this)(i, j);
    return std::sqrt(norm_sq);
  }

  // Copy assignment operator
  DenseMatrixBlockView& operator=(const DenseMatrixBlockView& other)
  {
    if (this != &other && rows() == other.rows() && cols() == other.cols()) std::copy_n(other.data_, rows() * cols(), data_);
    return *this;
  }

  DenseMatrixBlockView& operator-=(const DenseMatrixBlockView& other)
  {
    if (this != &other && rows() == other.rows() && cols() == other.cols())
      for (std::size_t i = 0; i < rows() * cols(); ++i) data_[i] -= other.data_[i];
    return *this;
  }

  DenseMatrixBlockView& operator+=(const DenseMatrixBlockView& other)
  {
    if (this != &other && rows() == other.rows() && cols() == other.cols())
      for (std::size_t i = 0; i < rows() * cols(); ++i) data_[i] += other.data_[i];
    return *this;
  }

  /** @brief In-place matrix multiplication: this = this * other */
  DenseMatrixBlockView& operator*=(const DenseMatrixBlockView& other)
  {
    // Create temporary storage for the result on the stack
    std::array<Real, order * order> temp;

    // Perform matrix multiplication: temp = this * other
    for (std::size_t i = 0; i < order; ++i) {
      for (std::size_t j = 0; j < order; ++j) {
        Real sum = 0;
        for (std::size_t k = 0; k < order; ++k) sum += (*this)(i, k) * other(k, j);
        temp[i * order + j] = sum;
      }
    }

    // Copy result back to this matrix
    std::copy(temp.begin(), temp.end(), data_);
    return *this;
  }

  template <class Vec>
  void mult(const Vec& x, Vec& y) const
  {
    for (std::size_t i = 0; i < order; ++i) {
      y[i] = 0;
      for (std::size_t j = 0; j < order; ++j) y[i] += (*this)(i, j) * x[j];
    }
  }

  void print() const
  {
    for (std::size_t i = 0; i < rows(); ++i) {
      for (std::size_t j = 0; j < cols(); ++j) {
        auto entry = (*this)(i, j);
        if (entry == 0.0) std::cout << std::setw(12) << "*";
        else std::cout << std::setw(12) << std::fixed << std::setprecision(7) << entry;
        std::cout << " ";
      }
      std::cout << "\n";
    }
  }

private:
  RealT* data_;
};

template <class RealT, std::size_t blocksize_, std::size_t alignment>
class DenseSquareBlockMatrix {
public:
  using Real = RealT;
  constexpr static auto blocksize = blocksize_;

  using BlockView = DenseMatrixBlockView<Real, blocksize>;
  using ConstBlockView = DenseMatrixBlockView<const Real, blocksize>;

  explicit DenseSquareBlockMatrix(std::size_t size)
      : size(size)
  {
    auto s = size * size * blocksize * blocksize;
    data = static_cast<RealT*>(std::aligned_alloc(alignment, s * sizeof(RealT)));
    std::fill_n(data, s, 0);
  }

  DenseSquareBlockMatrix(const DenseSquareBlockMatrix& other)
      : size(other.size)
  {
    auto s = size * size * blocksize * blocksize;
    data = static_cast<RealT*>(std::aligned_alloc(alignment, s * sizeof(RealT)));
    std::copy_n(other.data, s, data);
  }

  DenseSquareBlockMatrix& operator=(const DenseSquareBlockMatrix& other)
  {
    if (this != &other) {
      if (other.size != size) throw std::invalid_argument("DenseSquareBlockMatrix copy assignment: The two matrices are not compatible");
      auto s = size * size * blocksize * blocksize;
      std::copy_n(other.data, size, data);
    }
    return *this;
  }

  DenseSquareBlockMatrix(DenseSquareBlockMatrix&&) = default;
  DenseSquareBlockMatrix& operator=(DenseSquareBlockMatrix&& other) = default;

  ~DenseSquareBlockMatrix() { std::free(data); }

  BlockView block_view(std::size_t row, std::size_t col)
  {
    if (row >= size || col >= size)
      throw std::out_of_range("DenseSquareBlockMatrix::block_view(): Block indices out of range, have " + std::to_string(row) + "," + std::to_string(col) + " but size is " + std::to_string(size));
    std::size_t block_index = row * size + col;
    return BlockView(data + block_index * blocksize * blocksize);
  }

  ConstBlockView block_view(std::size_t row, std::size_t col) const
  {
    if (row >= size || col >= size)
      throw std::out_of_range("DenseSquareBlockMatrix::block_view(): Block indices out of range, have " + std::to_string(row) + "," + std::to_string(col) + " but size is " + std::to_string(size));
    std::size_t block_index = row * size + col;
    return ConstBlockView(data + block_index * blocksize * blocksize);
  }

  std::vector<Real> to_flat_column_major() const
  {
    std::vector<Real> flatdata(size * size * blocksize * blocksize);
    std::size_t total_size = size * blocksize;

    for (std::size_t global_col = 0; global_col < total_size; ++global_col) {
      for (std::size_t global_row = 0; global_row < total_size; ++global_row) {
        // Map global coordinates to block coordinates
        std::size_t block_row = global_row / blocksize;
        std::size_t block_col = global_col / blocksize;
        std::size_t local_row = global_row % blocksize;
        std::size_t local_col = global_col % blocksize;

        // Get the block and extract the element
        auto block = block_view(block_row, block_col);
        Real element = block(local_row, local_col);

        // Store in column-major order: column index determines the base position
        std::size_t flat_index = global_col * total_size + global_row;
        flatdata[flat_index] = element;
      }
    }

    return flatdata;
  }

  void from_flat_column_major(const std::vector<Real>& flatdata)
  {
    std::size_t total_size = size * blocksize;
    if (flatdata.size() != total_size * total_size) throw std::invalid_argument("Input flat data size does not match matrix dimensions");

    for (std::size_t global_col = 0; global_col < total_size; ++global_col) {
      for (std::size_t global_row = 0; global_row < total_size; ++global_row) {
        // Map global coordinates to block coordinates
        std::size_t block_row = global_row / blocksize;
        std::size_t block_col = global_col / blocksize;
        std::size_t local_row = global_row % blocksize;
        std::size_t local_col = global_col % blocksize;

        // Get the element from column-major flat data
        std::size_t flat_index = global_col * total_size + global_row;
        Real element = flatdata[flat_index];

        // Store in the block matrix
        auto block = block_view(block_row, block_col);
        block(local_row, local_col) = element;
      }
    }
  }

  void print(bool with_separator = true, bool scientific = true, int precision = 8, Real tolerance = 1e-10) const
  {
    if (with_separator) {
      std::cout << "--------------------------------------------------\n";
      std::cout << "Block matrix of order " << size << "\n";
    }

    // Set output format for high precision
    std::ios_base::fmtflags original_flags = std::cout.flags();
    std::streamsize original_precision = std::cout.precision();

    if (scientific) std::cout << std::scientific << std::setprecision(precision);
    else std::cout << std::fixed << std::setprecision(precision);

    // Calculate field width based on precision and format
    int field_width = scientific ? precision + 7 : precision + 4; // Adjust for format overhead

    for (std::size_t i = 0; i < size; ++i) {
      // Print each row of blocks
      for (std::size_t bi = 0; bi < blocksize; ++bi) {
        for (std::size_t j = 0; j < size; ++j) {
          auto block = block_view(i, j);

          // Print vertical separator before block (except for first block)
          if (j > 0 && with_separator) std::cout << "|";

          for (std::size_t bj = 0; bj < blocksize; ++bj) {
            Real entry = block(bi, bj);
            if (std::abs(entry) < tolerance) std::cout << std::setw(field_width) << "*";
            else std::cout << std::setw(field_width) << entry;
            if (bj < blocksize - 1) std::cout << " ";
          }
        }
        std::cout << "\n";
      }

      // Add separator line between block rows
      if (i < size - 1 && with_separator) {
        // Print horizontal separator line
        for (std::size_t j = 0; j < size; ++j) {
          if (j > 0) std::cout << "+";
          for (std::size_t bj = 0; bj < blocksize; ++bj) {
            std::cout << std::string(field_width, '-');
            if (bj < blocksize - 1) std::cout << "-";
          }
        }
        std::cout << "\n";
      }
    }

    if (with_separator) std::cout << "--------------------------------------------------\n";

    // Restore original formatting
    std::cout.flags(original_flags);
    std::cout.precision(original_precision);
  }

private:
  Real* data;
  std::size_t size; // The number of block rows/ block columns
};

/** @brief A symmetric block tridiagonal matrix with reduced storage requirements
 *
 *  This class implements a symmetric block tridiagonal matrix where the blocks are stored
 *  in row-major storage format and the blocks are stored contiguously in memory. The order
 *  of the stored blocks is: first the diagonal blocks, then the blocks of the upper off-
 *  diagonal (the lower off-diagonal blocks are not stored due to symmetry).
 */
template <class RealT, std::size_t blocksize_, std::size_t alignment>
class BlockTridiagonalMatrix {
public:
  using Real = RealT;
  constexpr static std::size_t blocksize = blocksize_;

  using BlockView = DenseMatrixBlockView<Real, blocksize>;
  using ConstBlockView = DenseMatrixBlockView<const Real, blocksize>;

  explicit BlockTridiagonalMatrix(std::size_t size_in_blocks)
      : size(size_in_blocks)
  {
    auto s = 2 * size * blocksize * blocksize;
    data = static_cast<RealT*>(std::aligned_alloc(alignment, s * sizeof(RealT)));
    std::fill_n(data, size, 0);
  }

  BlockTridiagonalMatrix(const BlockTridiagonalMatrix& other)
      : size(other.size)
  {
    auto s = size * size * blocksize * blocksize;
    data = static_cast<RealT*>(std::aligned_alloc(alignment, s * sizeof(RealT)));
    std::copy_n(other.data, size, data);
  }

  BlockTridiagonalMatrix& operator=(const BlockTridiagonalMatrix& other)
  {
    if (this != &other) {
      if (other.size != size) throw std::invalid_argument("DenseSquareBlockMatrix copy assignment: The two matrices are not compatible");
      auto s = size * size * blocksize * blocksize;
      std::copy_n(other.data, size, data);
    }
    return *this;
  }

  BlockTridiagonalMatrix(BlockTridiagonalMatrix&&) = default;
  BlockTridiagonalMatrix& operator=(BlockTridiagonalMatrix&& other) = default;

  ~BlockTridiagonalMatrix() { std::free(data); }

  BlockView block_view(std::size_t row, std::size_t col)
  {
    if ((row != col) and (col == row + 1)) throw std::invalid_argument("Can only access diagonal or upper off-diagonal blocks, not (" + std::to_string(row) + ", " + std::to_string(col) + ")");

    if (row == col) return BlockView(data + row * blocksize * blocksize);
    else
      return BlockView(data + size * blocksize * blocksize // skip the diagonal blocks
                       + row * blocksize * blocksize);     // and then skip the first 'row' off-diagonal blocks
  }

  ConstBlockView block_view(std::size_t row, std::size_t col) const
  {
    if ((row != col) and (col == row + 1)) throw std::invalid_argument("Can only access diagonal or upper off-diagonal blocks, not (" + std::to_string(row) + ", " + std::to_string(col) + ")");

    if (row == col) return ConstBlockView(data + row * blocksize * blocksize);
    else
      return ConstBlockView(data + size * blocksize * blocksize // skip the diagonal blocks
                            + row * blocksize * blocksize);     // and then skip the first 'row' off-diagonal blocks
  }

private:
  Real* data;
  std::size_t size; // The size of the matrix in blocks
};
