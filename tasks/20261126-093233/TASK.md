# Mark preconditioners and solvers as sequential on a higher level

- STATUS: OPEN
- PRIORITY: 50
- TAGS: solver

PARDISO, UMFPACK and STRUMPACK each check `DDM_CHECK(this->mat()->sequential(), ...)` in their constructor
(`include/ddm/impl/prec/{pardiso,umfpack,strumpack}.hh`). It should be possible to declare a preconditioner or solver
as sequential on a higher level instead, e.g. when it is registered, so that the factories can check it (or pick a
parallel alternative) before constructing anything, and the check isn't repeated in every wrapper.

The sequential `PetscPrec` from 20261126-092821 would be another such wrapper.
