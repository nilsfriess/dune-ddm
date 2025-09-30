  template <class OtherBlockView, class DenseMatView>
  void dot_manual(OtherBlockView V, DenseMatView R) const
  {
    const auto m = this->cols();
    const auto n = V.cols();
    const auto k = this->rows();

    for (std::size_t bcol = 0; bcol < n; ++bcol) {
      for (std::size_t acol = 0; acol < m; ++acol) {
        R(bcol, acol) = 0;
      }
    }

    auto c_tmp = make_aligned_array<Real>(m * n, 64);
    std::fill_n(c_tmp.get(), m * n, 0);
  }
