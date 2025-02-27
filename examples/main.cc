#include <iostream>

#include "helpers.hh"

#include <dune/pdelab.hh>
#include <mpi.h>

int main(int argc, char **argv) {
    
    MPI_Init(&argc, &argv);

    hello();

    MPI_Finalize();
} 