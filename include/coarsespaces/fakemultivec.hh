#pragma once

#include <algorithm>
#include <dune/common/dynvector.hh>

#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <lobpcg.h>
#include <random>
#include <temp_multivector.h>

using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

inline void *createVector(void *v_)
{
  auto *v = static_cast<Vector *>(v_);

  auto w = new Vector(*v);
  return (void *)w;
}

inline int destroyVector(void *v_)
{
  auto *v = static_cast<Vector *>(v_);
  delete v;
  return 0;
}

inline int innerProd(void *x_, void *y_, void *p_)
{
  auto *p = static_cast<double *>(p_);
  auto *x = static_cast<Vector *>(x_);
  auto *y = static_cast<Vector *>(y_);

  *p = x->dot(*y);

  return 0;
}

inline int copyVector(void *x_, void *y_)
{
  auto *x = static_cast<Vector *>(x_);
  auto *y = static_cast<Vector *>(y_);

  *y = *x;

  return 0;
}

inline int clearVector(void *x_)
{
  auto *x = static_cast<Vector *>(x_);
  *x = 0.0;

  return 0;
}

inline int setRandomValues(void *x_, int seed)
{
  auto *x = static_cast<Vector *>(x_);

  std::mt19937 rng(seed);
  std::normal_distribution dist;

  std::generate(x->begin(), x->end(), [&]() { return dist(rng); });

  return 0;
}

inline int scaleVector(double alpha, void *x_)
{
  auto *x = static_cast<Vector *>(x_);

  *x *= alpha;

  return 0;
}

inline int axpyVector(void *a_, void *x_, void *y_)
{
  auto *a = static_cast<double *>(a_);
  auto *x = static_cast<Vector *>(x_);
  auto *y = static_cast<Vector *>(y_);

  y->axpy(*a, *x);

  return 0;
}

template <class Mat>
void MatFakeMultiVec(void *A_, void *x_, void *Ax_)
{
  auto *A = static_cast<Mat *>(A_);
  auto *x = static_cast<mv_TempMultiVector *>(x_);
  auto *Ax = static_cast<mv_TempMultiVector *>(Ax_);

  for (std::size_t i = 0; i < x->numVectors; ++i) {
    if (x->mask && x->mask[i] == 0) {
      continue;
    }

    auto *v = static_cast<Vector *>(x->vector[i]);
    auto *res = static_cast<Vector *>(Ax->vector[i]);

    A->mv(*v, *res);
  }
}

inline void SetUpFakeMultiVectorInterfaceInterpreter(mv_InterfaceInterpreter *i)
{
  // Vector part
  i->CreateVector = createVector;
  i->DestroyVector = destroyVector;
  i->InnerProd = innerProd;
  i->CopyVector = copyVector;
  i->ClearVector = clearVector;
  i->SetRandomValues = setRandomValues;
  i->ScaleVector = scaleVector;
  i->Axpy = axpyVector;

  // Multivector part
  i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
  i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = mv_TempMultiVectorDestroy;

  i->Width = mv_TempMultiVectorWidth;
  i->Height = mv_TempMultiVectorHeight;
  i->SetMask = mv_TempMultiVectorSetMask;
  i->CopyMultiVector = mv_TempMultiVectorCopy;
  i->ClearMultiVector = mv_TempMultiVectorClear;
  i->SetRandomVectors = mv_TempMultiVectorSetRandom;
  i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = mv_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
  i->MultiAxpy = mv_TempMultiVectorAxpy;
  i->MultiXapy = mv_TempMultiVectorXapy;
  i->Eval = mv_TempMultiVectorEval;
}
