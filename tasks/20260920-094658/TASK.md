# Validate the input of begin_columns to get rid of the exception that might be thrown after receives have been posted

- STATUS: CLOSED
- PRIORITY: 50
- TAGS:

Comment form AI code review:
    ExchangeState::begin_columns can throw std::out_of_range (blocks.at(peer)) after some MPI_Irecvs are
    already posted, leaving the communicator broken — validate upfront; exchange_all_holders also doesn't 
    check that block widths sum to received.cols().

Closed on 2026-11-26: begin_columns and exchange_all_holders were removed in dfcf2a9; their successor neighbour_basis() does all lookups before posting receives and after MPI_Waitall.
