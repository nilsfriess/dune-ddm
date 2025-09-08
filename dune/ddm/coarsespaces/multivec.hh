#pragma once

#include <cassert>
#include <iostream>

#include <lobpcg.h>

#include <numeric>
#include <random>

// A simple multivector that allows for masked operations (operations carried out on only a subset of the vectors).
// For usage with BLOPEX.
// Adapted from https://github.com/lobpcg/blopex/blob/master/blopex_serial_double/multivector/multi_vector.c
struct Multivec {
  Multivec(std::size_t N, std::size_t M) : N(N), M(M), entries(N * M), active_indices(M)
  {
    std::iota(active_indices.begin(), active_indices.end(), 0); // Initially all indices are active
  }

  Multivec(const Multivec &) = default;
  Multivec(const Multivec &&) = delete;
  Multivec &operator=(const Multivec &) = default;
  Multivec &operator=(const Multivec &&) = delete;
  ~Multivec() = default;

  std::size_t N; // The size of one vector
  std::size_t M; // The number of vectors

  std::vector<double> entries;
  std::vector<std::size_t> active_indices;
};

inline void *MultiVectorCreateCopy(void *src_, BlopexInt copyvalues)
{
  auto *src = static_cast<Multivec *>(src_);

  Multivec *dst{};
  if (copyvalues > 0) {
    dst = new Multivec(*src);
  }
  else {
    dst = new Multivec(src->N, src->M);
  }

  return dst;
}

inline void MultiVectorDestroy(void *vec_)
{
  auto *vec = static_cast<Multivec *>(vec_);
  delete vec;
}

inline BlopexInt MultiVectorWidth(void *vec_)
{
  auto *vec = static_cast<Multivec *>(vec_);
  return static_cast<BlopexInt>(vec->M);
}

inline void MultiVectorSetMask(void *vec_, BlopexInt *mask)
{
  auto *vec = static_cast<Multivec *>(vec_);

  // It seems like mask is allowed to be nullptr in which case it should be interpreted as "set all indices as active"
  if (mask == nullptr) {
    vec->active_indices.resize(vec->M);
    std::iota(vec->active_indices.begin(), vec->active_indices.end(), 0);
  }
  else {
    vec->active_indices.clear();
    for (std::size_t i = 0; i < vec->M; ++i) {
      if (mask[i] > 0) {
        vec->active_indices.push_back(i);
      }
    }
  }
}

// Copy x's active vectors to y's active vectors (assuming both Multivecs are of the same size and have the same number of active vectors)
inline void MultiVectorCopy(void *x_, void *y_)
{
  auto *x = static_cast<Multivec *>(x_);
  auto *y = static_cast<Multivec *>(y_);

  assert(x->N == y->N && x->active_indices.size() == y->active_indices.size());

  for (std::size_t i = 0; i < x->active_indices.size(); ++i) {
    auto xstart = x->active_indices[i] * x->N;
    auto ystart = y->active_indices[i] * y->N;

    for (std::size_t j = 0; j < x->N; ++j) {
      y->entries[ystart + j] = x->entries[xstart + j];
    }
  }
}

inline void MultiVectorSetConstantValues(void *vec_, double value)
{
  auto *vec = static_cast<Multivec *>(vec_);

  for (auto active_index : vec->active_indices) {
    auto start = active_index * vec->N;
    auto end = start + vec->N;
    for (auto j = start; j < end; ++j) {
      vec->entries[j] = value;
    }
  }
}

inline void MultiVectorClear(void *vec) { MultiVectorSetConstantValues(vec, 0.0); }

inline void MultiVectorSetRandomValues(void *vec_, BlopexInt seed)
{
  auto *vec = static_cast<Multivec *>(vec_);

  static std::mt19937 rng(seed);
  static std::normal_distribution dist;
  rng.seed(seed);

  for (auto active_index : vec->active_indices) {
    auto start = active_index * vec->N;
    auto end = start + vec->N;
    for (auto j = start; j < end; ++j) {
      vec->entries[j] = dist(rng);
    }
  }
}

inline void MultiVectorInnerProd(void *x_, void *y_, BlopexInt gh, BlopexInt h, [[maybe_unused]] BlopexInt w, void *v)
{
  auto *x = static_cast<Multivec *>(x_);
  auto *y = static_cast<Multivec *>(y_);

  assert(x->N == y->N && x->active_indices.size() == static_cast<std::size_t>(h) && y->active_indices.size() == static_cast<std::size_t>(w));

  auto *res = static_cast<double *>(v);
  auto curr = 0;
  const auto gap = gh - h;

  for (std::size_t j = 0; j < y->active_indices.size(); ++j) {
    auto ystart = y->active_indices[j] * y->N;
    for (std::size_t i = 0; i < x->active_indices.size(); ++i) {
      auto xstart = x->active_indices[i] * x->N;

      double current_prod = 0.0;
      for (std::size_t k = 0; k < x->N; ++k) {
        current_prod += x->entries[xstart + k] * y->entries[ystart + k];
      }

      res[curr] = current_prod;
      curr++;
    }
    curr += gap;
  }
}

inline void MultiVectorInnerProdDiag(void *x_, void *y_, BlopexInt *mask, BlopexInt n, void *diag_)
{
  auto *x = static_cast<Multivec *>(x_);
  auto *y = static_cast<Multivec *>(y_);
  auto *diag = static_cast<double *>(diag_);

  assert(x->N == y->N && x->active_indices.size() == y->active_indices.size());

  // Convert the given mask to a set of active indices or set all indices to active if mask is null
  std::vector<std::size_t> active_indices;
  if (mask == nullptr) {
    active_indices.resize(n);
    std::iota(active_indices.begin(), active_indices.end(), 0);
  }
  else {
    for (BlopexInt i = 0; i < n; ++i) {
      if (mask[i] > 0) {
        active_indices.push_back(i);
      }
    }
  }
  assert(x->active_indices.size() == active_indices.size()); // Check that we set the correct amount

  for (std::size_t i = 0; i < x->active_indices.size(); ++i) {
    auto xstart = x->active_indices[i] * x->N;
    auto ystart = y->active_indices[i] * y->N;

    double current_prod = 0.0;
    for (std::size_t k = 0; k < x->N; ++k) {
      current_prod += x->entries[xstart + k] * y->entries[ystart + k];
    }
    diag[active_indices[i]] = current_prod;
  }
}

inline void MultiVectorByMatrix(void *x_, BlopexInt gh, BlopexInt h, BlopexInt w, void *v_, void *y_)
{
  auto *x = static_cast<Multivec *>(x_);
  auto *y = static_cast<Multivec *>(y_);
  auto *v = static_cast<double *>(v_);

  assert(static_cast<std::size_t>(h) == x->active_indices.size() && static_cast<std::size_t>(w) == y->active_indices.size());

  double curr_coeff = 0.0;
  auto gap = gh - h;
  for (BlopexInt j = 0; j < w; ++j) {
    auto ystart = y->active_indices[j] * y->N;

    auto xstart = x->active_indices[0] * x->N;
    curr_coeff = *v++;

    for (std::size_t k = 0; k < y->N; ++k) {
      y->entries[ystart + k] = curr_coeff * x->entries[xstart + k];
    }

    for (BlopexInt i = 1; i < h; ++i) {
      xstart = x->active_indices[i] * x->N;
      curr_coeff = *v++;

      for (std::size_t k = 0; k < y->N; ++k) {
        y->entries[ystart + k] += curr_coeff * x->entries[xstart + k];
      }
    }

    v += gap;
  }
}

inline void MultiVectorByDiagonal(void *x_, BlopexInt *mask, BlopexInt n, void *diag_, void *y_)
{
  auto *x = static_cast<Multivec *>(x_);
  auto *y = static_cast<Multivec *>(y_);
  auto *diag = static_cast<double *>(diag_);

  assert(x->N == y->N && x->active_indices.size() == y->active_indices.size());

  // Convert the given mask to a set of active indices or set all indices to active if mask is null
  std::vector<std::size_t> active_indices;
  if (mask == nullptr) {
    active_indices.resize(n);
    std::iota(active_indices.begin(), active_indices.end(), 0);
  }
  else {
    for (BlopexInt i = 0; i < n; ++i) {
      if (mask[i] > 0) {
        active_indices.push_back(i);
      }
    }
  }
  assert(x->active_indices.size() == active_indices.size()); // Check that we set the correct amount

  for (std::size_t i = 0; i < active_indices.size(); ++i) {
    auto xstart = x->active_indices[i] * x->N;
    auto ystart = y->active_indices[i] * y->N;
    auto curr_diag = diag[active_indices[i]];

    for (std::size_t j = 0; j < x->N; ++j) {
      y->entries[ystart + j] = curr_diag * x->entries[xstart + j];
    }
  }
}

inline void MultiVectorAXPY(double alpha, void *x_, void *y_)
{
  auto *x = static_cast<Multivec *>(x_);
  auto *y = static_cast<Multivec *>(y_);

  assert(x->N == y->N && x->active_indices.size() == y->active_indices.size());

  for (std::size_t i = 0; i < x->active_indices.size(); ++i) {
    auto xstart = x->active_indices[i] * x->N;
    auto ystart = y->active_indices[i] * y->N;

    for (std::size_t j = 0; j < y->N; ++j) {
      y->entries[ystart + j] += alpha * x->entries[xstart + j];
    }
  }
}

inline void MultiVectorPrint(void *x_, char *tag, BlopexInt limit)
{
  auto *x = static_cast<Multivec *>(x_);

  std::cout << "======= " << tag << " =======\n";
  std::cout << "size: " << x->N << "\n";
  std::cout << "num vectors: " << x->M << "\n";
  std::cout << "num active vectors: " << x->active_indices.size() << "\n";

  auto rows = std::min(x->N, static_cast<std::size_t>(limit));
  auto cols = std::min(x->active_indices.size(), static_cast<std::size_t>(limit));

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      std::cout << i << " " << j << " " << x->entries[x->active_indices[i] * x->N + j] << "\n";
    }
  }
}

template <class Mat>
void MatMultiVec(void *A_, void *x_, void *Ax_)
{
  auto *A = static_cast<Mat *>(A_);
  auto *x = static_cast<Multivec *>(x_);
  auto *Ax = static_cast<Multivec *>(Ax_);

  assert(x->active_indices.size() == Ax->active_indices.size() && x->N == Ax->N);

  for (std::size_t i = 0; i < x->active_indices.size(); ++i) {
    auto xstart = x->active_indices[i] * x->N;
    auto zstart = Ax->active_indices[i] * Ax->N;

    // Initialise vector in result with zeros
    for (std::size_t j = zstart; j < zstart + Ax->N; ++j) {
      Ax->entries[j] = 0.0;
    }

    // Compute dot product of all rows of A with the current active vector
    for (auto ri = A->begin(); ri != A->end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        Ax->entries[zstart + ri.index()] += *ci * x->entries[xstart + ci.index()];
      }
    }
  }
}

inline void SetUpMultiVectorInterfaceInterpreter(mv_InterfaceInterpreter *i)
{
  // Vector part
  i->CreateVector = nullptr;
  i->DestroyVector = nullptr;
  i->InnerProd = nullptr;
  i->CopyVector = nullptr;
  i->ClearVector = nullptr;
  i->SetRandomValues = nullptr;
  i->ScaleVector = nullptr;
  i->Axpy = nullptr;

  // Multivector part
  i->CreateMultiVector = nullptr;
  i->CopyCreateMultiVector = MultiVectorCreateCopy;
  i->DestroyMultiVector = MultiVectorDestroy;

  i->Width = MultiVectorWidth;
  i->Height = nullptr;
  i->SetMask = MultiVectorSetMask;
  i->CopyMultiVector = MultiVectorCopy;
  i->ClearMultiVector = MultiVectorClear;
  i->SetRandomVectors = MultiVectorSetRandomValues;
  i->MultiInnerProd = MultiVectorInnerProd;
  i->MultiInnerProdDiag = MultiVectorInnerProdDiag;
  i->MultiVecMat = MultiVectorByMatrix;
  i->MultiVecMatDiag = MultiVectorByDiagonal;
  i->MultiAxpy = MultiVectorAXPY;
  i->MultiXapy = nullptr;
  i->Eval = nullptr;
  i->MultiPrint = MultiVectorPrint;
}
