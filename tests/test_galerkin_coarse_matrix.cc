#include "datahandles.hh"
#include "galerkin_preconditioner.hh"
#include "helpers.hh"
#include "overlap_extension.hh"

#include <dune/common/fvector.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parallel/plocalindex.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/matrixmarket.hh>

#include <iostream>
#include <mpi.h>
#include <sstream>

// The test matrix in MatrixMarket format
const std::string test_matrix_mm = R"(%%MatrixMarket matrix coordinate real general
% ISTL_STRUCT blocked 1 1
9 9 25
1 1 1
1 2 18
2 1 10
2 2 2
2 3 19
3 2 11
3 3 3
3 4 20
4 3 12
4 4 4
4 5 21
5 4 13
5 5 5
5 6 22
6 5 14
6 6 6
6 7 23
7 6 15
7 7 7
7 8 24
8 7 16
8 8 8
8 9 25
9 8 17
9 9 9)";

// The coarse matrix in MatrixMarket format
const std::string coarse_matrix_mm = R"(%%MatrixMarket matrix coordinate real general
% ISTL_STRUCT blocked 1 1
4 4 14
1 1 29.52777777777778
1 2 27.02777777777778
1 3 7.277777777777778
2 1 21.69444444444445
2 2 28.11111111111111
2 3 21.19444444444444
2 4 8.166666666666666
3 1 4.611111111111111
3 2 18.52777777777778
3 3 34.11111111111111
3 4 36.91666666666666
4 2 5.499999999999999
4 3 31.58333333333333
4 4 50.75)";

int main(int argc, char *argv[])
{
  const auto &helper = Dune::MPIHelper::instance(argc, argv);
  if (helper.size() != 4) {
    std::cerr << "This test must be run with exactly 4 MPI ranks.\n";
    return 77; // 77 indicates skipped test
  }

  /* First we build a test matrix. The non-zero pattern corresponds to
     a finite element discretisation of the Laplacian on a 1D domain.
     More precisely, the global stiffness matrix is given by

         /    1   18                                      \
        /    10    2   19                                  \
        |         11    3   20                             |
        |              12    4    21                       |
  A =   |                   13    5    22                  |
        |                        14    6    23             |
        |                             15    7    24        |
        \                                  16    8    25   /
         \                                      17    9   /

  This matrix will be distributed over four MPI ranks, each owning
  a 3x3 block of the matrix. The matrix will be distributed in overllapping
  additive form, so that rank 0, e.g., will own the following block

         /  1   18   0  \
   A0 =  | 10   2  19   | ,
         \  0  11  1.5  /

  rank 1 will own

         /  1.5  20  0    \
   A1 =  |  12    4   21  | ,
         \  0    13   2.5 /

  etc. In other words, the diagonal entries in the overlap are always divided
  by 2.
  */

  // Build test matrix in additive form
  Dune::BCRSMatrix<double> A(3, 3, 2, 0.2, Dune::BCRSMatrix<double>::implicit);
  if (helper.rank() == 0) {
    A.entry(0, 0) = 1;
    A.entry(1, 1) = 2;
    A.entry(2, 2) = 1.5;

    A.entry(0, 1) = 18;
    A.entry(1, 0) = 10;
    A.entry(1, 2) = 19;
    A.entry(2, 1) = 11;
  }
  else if (helper.rank() == 1) {
    A.entry(0, 0) = 1.5;
    A.entry(1, 1) = 4;
    A.entry(2, 2) = 2.5;

    A.entry(0, 1) = 20;
    A.entry(1, 0) = 12;
    A.entry(1, 2) = 21;
    A.entry(2, 1) = 13;
  }
  else if (helper.rank() == 2) {
    A.entry(0, 0) = 2.5;
    A.entry(1, 1) = 6;
    A.entry(2, 2) = 3.5;

    A.entry(0, 1) = 22;
    A.entry(1, 0) = 14;
    A.entry(1, 2) = 23;
    A.entry(2, 1) = 15;
  }
  else {
    A.entry(0, 0) = 3.5;
    A.entry(1, 1) = 8;
    A.entry(2, 2) = 9;

    A.entry(0, 1) = 24;
    A.entry(1, 0) = 16;
    A.entry(1, 2) = 25;
    A.entry(2, 1) = 17;
  }
  A.compress();

  // Now build parallel index set and remote indices
  using LocalIndex = Dune::ParallelLocalIndex<Attribute>;
  Dune::ParallelIndexSet<int, LocalIndex> paridxs;
  paridxs.beginResize();
  if (helper.rank() == 0) {
    paridxs.add(0, {0, Attribute::owner, false});
    paridxs.add(1, {1, Attribute::owner, false});
    paridxs.add(2, {2, Attribute::owner, true});
  }
  else if (helper.rank() == 1) {
    paridxs.add(2, {0, Attribute::copy, true});
    paridxs.add(3, {1, Attribute::owner, false});
    paridxs.add(4, {2, Attribute::owner, true});
  }
  else if (helper.rank() == 2) {
    paridxs.add(4, {0, Attribute::copy, true});
    paridxs.add(5, {1, Attribute::owner, false});
    paridxs.add(6, {2, Attribute::owner, true});
  }
  else {
    paridxs.add(6, {0, Attribute::copy, true});
    paridxs.add(7, {1, Attribute::owner, false});
    paridxs.add(8, {2, Attribute::owner, false});
  }
  paridxs.endResize();

  std::vector<int> nbs;
  if (helper.rank() == 0) {
    nbs = {1};
  }
  else if (helper.rank() == 1) {
    nbs = {0, 2};
  }
  else if (helper.rank() == 2) {
    nbs = {1, 3};
  }
  else {
    nbs = {2};
  }

  Dune::RemoteIndices remoteids(paridxs, paridxs, helper.getCommunicator(), nbs);
  remoteids.rebuild<false>();

  // Now that everything is set up, we can test the extension of the overlap.
  // With 6 layers of overlap, we should get the full matrix on rank 0.
  auto remoteparids = extendOverlap(remoteids, A, 6);
  auto Aovlp = createOverlappingMatrix(A, *remoteparids.first);

  if (helper.rank() == 0) {
    std::stringstream ss(test_matrix_mm);

    Dune::BCRSMatrix<double> Aexpected;
    Dune::readMatrixMarket(Aexpected, ss);

    Aexpected -= Aovlp;
    if (Aexpected.frobenius_norm() > 1e-16) {
      std::cerr << "Matrices are not equal\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // Now test if the Galerkin preconditioner computes the correct coarse matrix.
  // Here we use smaller overlap.
  remoteparids = extendOverlap(remoteids, A, 1);
  Aovlp = createOverlappingMatrix(A, *remoteparids.first);

  Dune::BlockVector<double> pou(remoteparids.second->size());
  Dune::BlockVector<double> t(4); // The 'template' vector
  t = 1;
  if (helper.rank() == 0) {
    pou[0] = 1;
    pou[1] = 0.5;
    pou[2] = 0.5;
    pou[3] = 1. / 3;
  }
  else if (helper.rank() == 1) {
    pou[0] = 0.5;
    pou[1] = 1. / 3;
    pou[2] = 0.5;
    pou[3] = 0.5;
    pou[4] = 1. / 3;
  }
  else if (helper.rank() == 2) {
    pou[0] = 0.5;
    pou[1] = 1. / 3;
    pou[2] = 0.5;
    pou[3] = 1. / 3;
    pou[4] = 0.5;
  }
  else {
    pou[0] = 0.5;
    pou[1] = 0.5;
    pou[2] = 1;
    pou[3] = 1. / 3;
  }

  // To make sure that we initialised the partition of unity correctly, we sum
  // it up and check that it is 1 everywhere.
  auto pou_sum = pou;
  Dune::Interface all_all_interface;
  all_all_interface.build(*remoteparids.first, AttributeSet{Attribute::owner, Attribute::copy}, AttributeSet{Attribute::owner, Attribute::copy});
  Dune::VariableSizeCommunicator all_all_comm(all_all_interface);
  AddVectorDataHandle<Dune::BlockVector<double>> advdh;
  advdh.setVec(pou_sum);
  all_all_comm.forward(advdh);
  for (auto &v : pou_sum) {
    if (v != 1) {
      std::cerr << "Partition of unity does not add up to 1\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  GalerkinPreconditioner prec(Aovlp, pou, t, remoteparids);
  const auto &A0 = prec.getCoarseMatrix();

  if (helper.rank() == 0) {
    std::stringstream ss(coarse_matrix_mm);

    Dune::BCRSMatrix<double> A0_expected;
    Dune::readMatrixMarket(A0_expected, ss);

    auto A0_expected_copy = A0_expected;
    A0_expected -= A0;
    if (A0_expected.frobenius_norm() > 1e-12) {
      std::cerr << "Coarse matrix is not correct. Expected: \n";
      Dune::printmatrix(std::cerr, A0_expected_copy, "", "");
      std::cerr << "Got: \n";
      Dune::printmatrix(std::cerr, A0, "", "");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  else {
    if (A0.N() != 0 or A0.M() != 0) {
      std::cerr << "Coarse matrix should be empty on all ranks except 0\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  return 0;
}