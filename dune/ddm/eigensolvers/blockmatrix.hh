#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <vector>

template <class RealT>
class DenseMatrixBlockView {
public:
  using Real = RealT;

  DenseMatrixBlockView(Real *it, std::size_t size) : data_(it), order(static_cast<std::size_t>(std::sqrt(size))) {}

  Real *data() { return data_; }

  Real &operator()(std::size_t i, std::size_t j) { return data_[(i * order) + j]; }

  std::size_t cols() const { return order; }
  std::size_t rows() const { return order; }

private:
  // friend std::ostream &operator<<(std::ostream &out, const DenseMatrixBlockView &R)
  // {
  //   out << "block matrix of size " << R.order << " x " << R.order << "\n";

  //   // formatting settings
  //   out.setf(std::ios::fixed, std::ios::floatfield);
  //   out.precision(3);

  //   for (std::size_t i = 0; i < R.view.size(); ++i) {
  //     Real val = R.view[i];

  //     // pretty print with width, replace tiny values with 0
  //     if (std::abs(val) < 1e-10) { out << std::setw(10) << "."; }
  //     else {
  //       out << std::setw(10) << val;
  //     }

  //     // new row if at the end of the current row
  //     if ((i + 1) % R.order == 0) { out << "\n"; }
  //   }

  //   return out;
  // }

  RealT *data_;
  std::size_t order;
};

template <class RealT, std::size_t alignment>
class DenseSquareBlockMatrix {
public:
  using Real = RealT;

  DenseSquareBlockMatrix(std::size_t order, std::size_t block_order) : order(order), block_order(block_order)
  {
    auto size = order * order * block_order * block_order;
    data = static_cast<RealT *>(std::aligned_alloc(alignment, size * sizeof(RealT)));
    std::fill_n(data, size, 0);
  }

  ~DenseSquareBlockMatrix() { std::free(data); }

  DenseMatrixBlockView<Real> block_view(std::size_t row, std::size_t col)
  {
    std::size_t block_size = block_order * block_order;
    std::size_t block_index = row * order + col;
    return {data + block_index * block_size, block_size};
  }

  DenseMatrixBlockView<const Real> block_view(std::size_t row, std::size_t col) const
  {
    std::size_t block_size = block_order * block_order;
    std::size_t block_index = row * order + col;
    return {data + block_index * block_size, block_size};
  }

  std::vector<Real> to_flat_column_major() const
  {
    std::vector<Real> flatdata(order * order * block_order * block_order);

    // Total matrix size is (order * block_order) x (order * block_order)
    std::size_t total_size = order * block_order;

    // Iterate column by column in the global matrix
    for (std::size_t global_col = 0; global_col < total_size; ++global_col) {
      for (std::size_t global_row = 0; global_row < total_size; ++global_row) {
        // Map global coordinates to block coordinates
        std::size_t block_row = global_row / block_order;
        std::size_t block_col = global_col / block_order;
        std::size_t local_row = global_row % block_order;
        std::size_t local_col = global_col % block_order;

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

  void from_flat_column_major(const std::vector<Real> &flatdata)
  {
    // Total matrix size is (order * block_order) x (order * block_order)
    std::size_t total_size = order * block_order;

    // Verify the input size matches expected size
    if (flatdata.size() != total_size * total_size) { throw std::invalid_argument("Input flat data size does not match matrix dimensions"); }

    // Iterate column by column in the global matrix
    for (std::size_t global_col = 0; global_col < total_size; ++global_col) {
      for (std::size_t global_row = 0; global_row < total_size; ++global_row) {
        // Map global coordinates to block coordinates
        std::size_t block_row = global_row / block_order;
        std::size_t block_col = global_col / block_order;
        std::size_t local_row = global_row % block_order;
        std::size_t local_col = global_col % block_order;

        // Get the element from column-major flat data
        std::size_t flat_index = global_col * total_size + global_row;
        Real element = flatdata[flat_index];

        // Store in the block matrix
        auto block = block_view(block_row, block_col);
        block(local_row, local_col) = element;
      }
    }
  }

private:
  friend std::ostream &operator<<(std::ostream &out, const DenseSquareBlockMatrix &R)
  {
    out << "--------------------------------------------------\n";
    out << "Block matrix of order " << R.order << "\n";
    for (std::size_t i = 0; i < R.order; ++i) {
      for (std::size_t j = 0; j < R.order; ++j) {
        out << R.block_view(i, j) << "  ";
      }
      out << "\n";
    }
    out << "--------------------------------------------------\n";
    return out;
  }

  Real *data;
  std::size_t order;
  std::size_t block_order;
};
