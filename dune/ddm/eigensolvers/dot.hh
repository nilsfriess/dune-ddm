template <class OtherBlockView, class DenseMatView>
void dot_(OtherBlockView V, DenseMatView R) const
{
  const auto m = this->cols();
  const auto n = V.cols();
  const auto k = this->rows();

  for (std::size_t bcol = 0; bcol < n; ++bcol)
    for (std::size_t acol = 0; acol < m; ++acol) R(acol, bcol) = 0;

#pragma omp parallel
  {
    // Compute size as smallest number >= m*n that is a multiple of the alignment size
    const std::size_t size = ((m * n + 63) / 64) * 64;
    auto c_tmp = make_aligned_array<Real>(size, 64);
    std::fill_n(c_tmp.get(), m * n, 0);

#pragma omp for
    for (std::size_t row = 0; row < k - 7; row += 8) {
      for (std::size_t bcol = 0; bcol < n; ++bcol) {
#pragma omp simd
        for (std::size_t acol = 0; acol < m; ++acol) {
          c_tmp[acol * n + bcol] += this->data()[(row + 0) * m + acol] * V.data()[(row + 0) * n + bcol] + this->data()[(row + 1) * m + acol] * V.data()[(row + 1) * n + bcol] +
                                    this->data()[(row + 2) * m + acol] * V.data()[(row + 2) * n + bcol] + this->data()[(row + 3) * m + acol] * V.data()[(row + 3) * n + bcol] +
                                    this->data()[(row + 4) * m + acol] * V.data()[(row + 4) * n + bcol] + this->data()[(row + 5) * m + acol] * V.data()[(row + 5) * n + bcol] +
                                    this->data()[(row + 6) * m + acol] * V.data()[(row + 6) * n + bcol] + this->data()[(row + 7) * m + acol] * V.data()[(row + 7) * n + bcol];
        }
      }
    }
#pragma omp critical
    for (std::size_t bcol = 0; bcol < n; ++bcol) {
#pragma omp simd
      for (std::size_t acol = 0; acol < m; ++acol) R(acol, bcol) += c_tmp[acol * n + bcol];
    }
  }

  // Epilogue: handle remaining rows when k is not divisible by 8
  const std::size_t remaining_start = (k / 8) * 8;
  for (std::size_t row = remaining_start; row < k; ++row) {
    for (std::size_t bcol = 0; bcol < n; ++bcol) {
#pragma omp simd
      for (std::size_t acol = 0; acol < m; ++acol) R(acol, bcol) += this->data()[row * m + acol] * V.data()[row * n + bcol];
    }
  }
}
