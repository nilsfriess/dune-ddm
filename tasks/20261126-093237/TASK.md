# Allocate MultiVec data with an aligned allocator

- STATUS: OPEN
- PRIORITY: 40
- TAGS: performance

`IstlMultiVec` (`include/ddm/impl/multivec/istlmultivec.hh`) stores its data column by column in a `std::vector<T>`,
which only guarantees the alignment of `T`. Use an aligned allocator (e.g. 64 bytes) so the BLAS calls and vectorised
loops on the columns get aligned data. Note that with a column-major layout only the first column is aligned unless
the leading dimension is padded.
