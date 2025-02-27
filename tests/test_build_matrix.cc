#include <numeric>

#include <mpi.h>

#include <dune/common/exceptions.hh>
#include <dune/istl/bvector.hh>

#include "helpers.hh"

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    // Test with a single row per rank
    Dune::BlockVector<double> v(10);
    std::iota(v.begin(), v.end(), rank);

    auto A = gatherMatrixFromRows(v, MPI_COMM_WORLD, 0, 0);

    if (rank == 0) {
      if (A.N() != static_cast<std::size_t>(size)) {
        DUNE_THROW(Dune::Exception, "Matrix has wrong number of rows");
      }
      if (A.M() != v.N()) {
        DUNE_THROW(Dune::Exception, "Matrix has wrong number of columns");
      }

      for (std::size_t i = 0; i < A.N(); ++i) {
        for (std::size_t j = 0; j < A.M(); ++j) {
          if (A[i][j] != static_cast<double>(i) + static_cast<double>(j)) {
            DUNE_THROW(Dune::Exception, "Matrix has wrong entry");
          }
        }
      }
    }
    else {
      if (A.N() != 0 or A.M() != 0) {
        DUNE_THROW(Dune::Exception, "Matrix should be empty on all ranks except rank 0");
      }
    }

    // Test with multiple rows per rank
    std::vector<Dune::BlockVector<double>> rows(rank % 2 == 0 ? 2 : 3, v);
    rows[0] = rank;
    rows[1] = rank;
    if (rank % 2 != 0) {
      rows[2] = rank;
    }

    auto B = gatherMatrixFromRows(rows, MPI_COMM_WORLD, 0, 0);

    if (rank == 0) {
      std::size_t expected_rows = 0;
      for (int i = 0; i < size; ++i) {
        expected_rows += i % 2 == 0 ? 2 : 3;
      }
      if (B.N() != expected_rows) {
        DUNE_THROW(Dune::Exception, "Matrix has wrong number of rows");
      }
      if (B.M() != v.N()) {
        DUNE_THROW(Dune::Exception, "Matrix has wrong number of columns");
      }

      int current_row = 0;
      for (int i = 0; i < size; ++i) {
        for (int r = 0; r < (i % 2 == 0 ? 2 : 3); ++r) {
          for (std::size_t j = 0; j < B.M(); ++j) {
            if (B[current_row][j] != i) {
              DUNE_THROW(Dune::Exception, "Matrix has wrong entry");
            }
          }
          ++current_row;
        }
      }
    }
  }
  catch (Dune::Exception &e) {
    std::cerr << "Caught exception: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
}