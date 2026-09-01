#include "dune/ddm/backend/backend.hh"
#include "dune/ddm/backend/host/backend.hh"
#include "dune/ddm/backend/sycl/backend.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/sycl/vec.hh"
#include "test_utils.hh"

#include <algorithm>
#include <cstddef>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/bvector.hh>
#include <numeric>
#include <string>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

/** @file
 *
 *  Tests for the dune/ddm/backend headers: the generic backend machinery (backend_traits,
 *  HasBackend, backend_of/backend_of_t, Buffer) and the per-backend implementation
 *  (context, make_buffer/make_buffer_from_host, malloc/free, gather, scatter, scatter_reduce,
 *  copy_n).
 *
 *  The compile-time contract is pinned with static_asserts at namespace scope; every runtime
 *  test is a template over a TestHelper. Adding a new backend (e.g. SYCL) therefore only means
 *  providing a helper with the members documented below and a backend_traits specialisation;
 *  no test in this file knows the concrete backend.
 */

namespace {

// ---------------------------------------------------------------------------
// TestHelper concept
// ---------------------------------------------------------------------------
//
// There is no base class: tests are templates that work on any type with these members.
//
//   using scalar_type = ...;                                   // element data type
//   using vector_t    = ...;                                   // a backend-traited container
//   using backend_t   = ddm::backend::backend_of_t<vector_t>;  // resolved through backend_traits
//
//   vector_t make_vector(std::size_t n);                      // n == 0 must be allowed: contexts
//                                                             // are obtained via make_vector(0)
//   std::vector<scalar_type> to_host_vector(const vector_t&); // device -> host copy
//   void fill(vector_t&, scalar_type);                        // v = x everywhere
//   void fill_with(vector_t&, const std::vector<scalar_type>&); // v[i] = vals[i]
//   void fill_iota(vector_t&);                                // v[i] = i + 1
//
// A context is obtained generically as Backend::context(h.make_vector(0)) (for ddm::Sycl::Vec
// this is its queue()). Note that read-back verification is done through to_host_vector(), so
// the tests never dereference raw device pointers directly.
//
// The tests derive the backend's element type as what vector_t::data() points at (double for
// std::vector, Dune::FieldVector<double, 1> for Dune::BlockVector, field_type for
// ddm::Sycl::Vec). Buffers and naked malloc memory use exactly that element type, so the test
// matches what gather/scatter see in real use (e.g. in Communication). This is why
// make_buffer_from_host is only exercised with int index buffers: int buffers are the only
// kind the production code creates from host data.
//
// For a SYCL helper on top of ddm::Sycl::Vec<Scalar>:
//   * make_vector(n)        -> Vec(queue, n), where queue is owned by the helper
//   * to_host_vector(v)     -> q.memcpy(host.data(), v.data(), ...).wait()
//   * fill(v, x)            -> v = x (q.fill)
//   * fill_with/fill_iota   -> host loop into a std::vector + memcpy, since Vec has no operator[]

// ---------------------------------------------------------------------------
// Compile-time contract of the generic machinery
// ---------------------------------------------------------------------------

// backend_traits / HasBackend: only specialised containers have a backend
static_assert(ddm::backend::HasBackend<std::vector<double>>);
static_assert(ddm::backend::HasBackend<Dune::BlockVector<Dune::FieldVector<double, 1>>>);
static_assert(!ddm::backend::HasBackend<int>);
static_assert(!ddm::backend::HasBackend<double*>);
static_assert(!ddm::backend::HasBackend<std::string>);

// backend_of / backend_of_t resolve through the specialisation, including the allocator
static_assert(std::is_same_v<ddm::backend::backend_of_t<std::vector<double>>, ddm::backend::HostBackend>);
static_assert(std::is_same_v<ddm::backend::backend_of<std::vector<double>>::type, ddm::backend::HostBackend>);
static_assert(std::is_same_v<ddm::backend::backend_of_t<std::vector<double, std::allocator<double>>>, ddm::backend::HostBackend>);
static_assert(std::is_same_v<ddm::backend::backend_of_t<Dune::BlockVector<Dune::FieldVector<double, 1>>>, ddm::backend::HostBackend>);

// Buffer is move-only, and its moves are noexcept (backend-agnostic)
using HostBuffer = ddm::backend::Buffer<double, ddm::backend::HostBackend>;
static_assert(!std::is_copy_constructible_v<HostBuffer>);
static_assert(!std::is_copy_assignable_v<HostBuffer>);
static_assert(std::is_move_constructible_v<HostBuffer>);
static_assert(std::is_move_assignable_v<HostBuffer>);
static_assert(std::is_nothrow_move_constructible_v<HostBuffer>);
static_assert(std::is_nothrow_move_assignable_v<HostBuffer>);

// ---------------------------------------------------------------------------
// Runtime contract of Buffer
// ---------------------------------------------------------------------------

/** @brief Buffer life-cycle: default state, size/empty/data, ownership transfer on moves.
 *
 *  Contents are verified by scattering data in and gathering it out through a helper vector, so
 *  the test is backend-agnostic; only size/empty/data-pointer invariants are checked directly on
 *  the wrapper (they never dereference device memory).
 */
template <class TestHelper>
void check_buffer(TestHelper& h, Dune::TestSuite& t)
{
  using Vector = typename TestHelper::vector_t;
  using Backend = typename TestHelper::backend_t;
  using scalar_type = typename TestHelper::scalar_type;
  using element_type = std::remove_pointer_t<decltype(std::declval<Vector&>().data())>;
  using Buffer = ddm::backend::Buffer<element_type, Backend>;

  auto ctx = Backend::context(h.make_vector(0));

  // identity index vector for scatter-in / gather-out
  const auto identity = [](std::size_t n) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    return idx;
  };
  const auto make_idx = [&](const std::vector<int>& host) { return Backend::make_buffer_from_host(ctx, host); };

  // Read the whole buffer back through a helper vector (device-safe for any backend)
  const auto read_back = [&](const Buffer& b) {
    auto dst = h.make_vector(b.size());
    Backend::gather(ctx, b.data(), make_idx(identity(b.size())), dst.data());
    return h.to_host_vector(dst); // std::vector<scalar_type>
  };
  // Write host values into the whole buffer through a helper vector (non-const buffer: the
  // destination pointer of scatter is T*, while Buffer::data() const gives const T*)
  const auto write_into = [&](Buffer& b, const std::vector<scalar_type>& vals) {
    auto src = h.make_vector(vals.size());
    h.fill_with(src, vals);
    Backend::scatter(ctx, src.data(), make_idx(identity(vals.size())), b.data());
  };

  // Default construction: empty, owns no memory; destroying it must not call free()
  {
    Buffer b;
    t.check(b.empty(), "default-constructed buffer is empty");
    t.check(b.size() == 0, "default-constructed buffer has size 0");
    t.check(b.data() == nullptr, "default-constructed buffer owns no memory");
  }

  // Construction from (context, size)
  {
    Buffer b(ctx, 10);
    t.check(!b.empty(), "10-element buffer is not empty");
    t.check(b.size() == 10, "10-element buffer reports size 10");
    t.check(b.data() != nullptr, "10-element buffer owns memory");

    const Buffer& cb = b;
    t.check(cb.data() == b.data(), "const data() agrees with data()");

    const std::vector<scalar_type> values = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    write_into(b, values);
    t.check(read_back(b) == values, "values survive a write/read round trip through data()");
  }

  // Zero-size construction
  {
    Buffer b(ctx, 0);
    t.check(b.empty(), "0-element buffer is empty");
    t.check(b.size() == 0, "0-element buffer reports size 0");
  }

  // Move construction transfers ownership and fully empties the source
  {
    const std::vector<scalar_type> values = {10, 11, 12, 13};
    Buffer src(ctx, values.size());
    write_into(src, values);

    Buffer dst(std::move(src));
    t.check(!dst.empty() && dst.size() == 4, "move-constructed buffer keeps the size");
    t.check(read_back(dst) == values, "move-constructed buffer keeps the values");

    t.check(src.empty(), "move-constructed-from buffer is empty");
    t.check(src.size() == 0, "move-constructed-from buffer reports size 0");
    t.check(src.data() == nullptr, "move-constructed-from buffer owns no memory");
  } // both buffers destroyed here; the moved-from one must not double-free

  // Move assignment into a default-constructed buffer
  {
    const std::vector<scalar_type> values = {100, 101, 102};
    Buffer dst;
    Buffer src(ctx, values.size());
    write_into(src, values);

    dst = std::move(src);
    t.check(!dst.empty() && dst.size() == 3, "move-assigned buffer keeps the size");
    t.check(read_back(dst) == values, "move-assigned buffer keeps the values");
    t.check(src.empty() && src.data() == nullptr, "move-assigned-from buffer owns no memory");
  }

  // Move assignment into a buffer that already owns memory (must release it, not leak)
  {
    Buffer dst(ctx, 2);
    write_into(dst, {-1, -2});
    Buffer src(ctx, 3);
    write_into(src, {100, 101, 102});

    dst = std::move(src);
    t.check(!dst.empty() && dst.size() == 3, "move-assign over a live buffer keeps the size");
    t.check(read_back(dst) == std::vector<scalar_type>({100, 101, 102}), "move-assign over a live buffer keeps the values");
    t.check(src.empty() && src.data() == nullptr, "move-assign over a live buffer empties the source");
  }

  // Self-move must leave a valid, unchanged buffer (routed through a pointer to avoid -Wself-move)
  {
    const std::vector<scalar_type> values = {42, 43, 44, 45, 46};
    Buffer b(ctx, values.size());
    write_into(b, values);
    Buffer* self = &b;
    b = std::move(*self);
    t.check(!b.empty() && b.size() == 5, "self-move leaves the buffer unchanged");
    t.check(b.data() != nullptr && read_back(b) == values, "self-move keeps the data");
  }
}

// ---------------------------------------------------------------------------
// Runtime contract of the backend primitives
// ---------------------------------------------------------------------------

/** @brief Data-movement primitives, exercised on backend memory through helper vectors. */
template <class TestHelper>
void check_backend_primitives(TestHelper& h, Dune::TestSuite& t)
{
  using Vector = typename TestHelper::vector_t;
  using Backend = typename TestHelper::backend_t;
  using scalar_type = typename TestHelper::scalar_type;
  using element_type = std::remove_pointer_t<decltype(std::declval<Vector&>().data())>;

  auto ctx = Backend::context(h.make_vector(0));
  const std::size_t n = 8;
  const std::vector<int> perm = {3, 1, 7, 0, 5, 2, 6, 4}; // a permutation of 0..7
  const auto idx_buf = Backend::make_buffer_from_host(ctx, perm);

  // make_buffer_from_host copies, it does not alias the host data (exercised with an int index
  // buffer, the only kind of buffer the production code builds from host data; its contents are
  // read back by using it as a gather permutation)
  {
    std::vector<int> host = {5, 6, 7};
    auto buf = Backend::make_buffer_from_host(ctx, host);

    auto src = h.make_vector(8);
    h.fill_iota(src); // src[i] = i+1

    auto dst = h.make_vector(3);
    Backend::gather(ctx, src.data(), buf, dst.data()); // dst[i] = src[buf[i]] = buf[i]+1
    const auto got = h.to_host_vector(dst);
    t.check(got == std::vector<scalar_type>({6, 7, 8}), "make_buffer_from_host copies the values");

    host[0] = 99;
    auto after = h.make_vector(3);
    Backend::gather(ctx, src.data(), buf, after.data());
    const auto got_after = h.to_host_vector(after);
    t.check(got_after == std::vector<scalar_type>({6, 7, 8}), "make_buffer_from_host takes a copy, not a reference");
  }

  // Plain malloc/free, written and read via scatter/gather (device-safe)
  {
    auto* raw = Backend::template malloc<element_type>(ctx, n);
    t.check(raw != nullptr, "malloc returns a non-null pointer");

    auto src = h.make_vector(n);
    h.fill_iota(src);
    Backend::scatter(ctx, src.data(), idx_buf, raw); // raw[perm[i]] = i+1

    auto dst = h.make_vector(n);
    Backend::gather(ctx, raw, idx_buf, dst.data()); // dst[i] = raw[perm[i]]
    const auto got = h.to_host_vector(dst);
    for (std::size_t i = 0; i < n; ++i) t.check(got[i] == static_cast<scalar_type>(i + 1), "values survive a write/read round trip through malloc'd memory") << "i=" << i;

    Backend::free(ctx, raw);
  }

  // gather: dst[i] = src[perm[i]]
  {
    auto src = h.make_vector(n);
    h.fill_iota(src); // src[i] = i+1
    auto dst = h.make_vector(n);
    h.fill(dst, -1);

    Backend::gather(ctx, src.data(), idx_buf, dst.data());
    const auto got = h.to_host_vector(dst);
    for (std::size_t i = 0; i < n; ++i) t.check(got[i] == static_cast<scalar_type>(perm[i] + 1), "gather picks src[perm[i]]") << "i=" << i << " dst=" << got[i] << " expected=" << perm[i] + 1;
  }

  // scatter: dst[perm[i]] = src[i]
  {
    auto src = h.make_vector(n);
    h.fill_iota(src);
    auto dst = h.make_vector(n);
    h.fill(dst, -1);

    Backend::scatter(ctx, src.data(), idx_buf, dst.data());
    const auto got = h.to_host_vector(dst);
    for (std::size_t i = 0; i < n; ++i) t.check(got[perm[i]] == static_cast<scalar_type>(i + 1), "scatter writes src[i] to dst[perm[i]]") << "i=" << i;
  }

  // scatter followed by gather with the same permutation recovers the source exactly
  {
    auto src = h.make_vector(n);
    h.fill_iota(src);
    auto scratch = h.make_vector(n);
    auto back = h.make_vector(n);

    Backend::scatter(ctx, src.data(), idx_buf, scratch.data());
    Backend::gather(ctx, scratch.data(), idx_buf, back.data());
    const auto src_host = h.to_host_vector(src);
    const auto back_host = h.to_host_vector(back);
    t.check(back_host == src_host, "scatter-gather round trip recovers the source");
  }

  // Duplicate indices: scatter keeps the last writer, scatter_reduce accumulates
  {
    const std::vector<int> dup = {2, 0, 2, 1}; // index 2 appears twice
    const auto dup_buf = Backend::make_buffer_from_host(ctx, dup);

    auto src = h.make_vector(4);
    h.fill_with(src, {10, 20, 30, 40});

    auto dst = h.make_vector(4);
    h.fill(dst, -1);
    Backend::scatter(ctx, src.data(), dup_buf, dst.data());
    {
      const auto got = h.to_host_vector(dst);
      t.check(got[0] == 20, "scatter writes index 0 once");
      t.check(got[1] == 40, "scatter writes index 1 once");
      t.check(got[2] == 30, "scatter keeps the last writer for duplicate indices");
      t.check(got[3] == -1, "scatter leaves untouched entries alone");
    }

    auto acc = h.make_vector(4);
    h.fill(acc, 0);
    Backend::template scatter_reduce<ddm::ReductionOperation::Addition>(ctx, src.data(), dup_buf, acc.data());
    {
      const auto got = h.to_host_vector(acc);
      t.check(got[0] == 20, "scatter_reduce adds at index 0");
      t.check(got[1] == 40, "scatter_reduce adds at index 1");
      t.check(got[2] == 40, "scatter_reduce accumulates duplicates at index 2") << "got " << got[2];
      t.check(got[3] == 0, "scatter_reduce leaves untouched entries alone");
    }
  }
}

// ---------------------------------------------------------------------------
// End-to-end tests through a backend vector
// ---------------------------------------------------------------------------

/** @brief copy_n on the backend vector type, the way SchwarzPreconditioner uses it. */
template <class TestHelper>
void check_vector_copy_n(TestHelper& h, Dune::TestSuite& t)
{
  using Backend = typename TestHelper::backend_t;
  using scalar_type = typename TestHelper::scalar_type;

  auto src = h.make_vector(6);
  auto dst = h.make_vector(6);
  h.fill_iota(src);
  h.fill(dst, 0);

  Backend::copy_n(src, 4, dst);

  const auto got = h.to_host_vector(dst);
  for (std::size_t i = 0; i < 4; ++i) t.check(got[i] == static_cast<scalar_type>(i + 1), "copy_n copies the first entries") << "i=" << i;
  for (std::size_t i = 4; i < 6; ++i) t.check(got[i] == 0, "copy_n leaves the rest untouched") << "i=" << i;
}

/** @brief End-to-end: scatter a vector into backend memory, zero it, gather it back. */
template <class TestHelper>
void check_vector_roundtrip(TestHelper& h, Dune::TestSuite& t)
{
  using Vector = typename TestHelper::vector_t;
  using Backend = typename TestHelper::backend_t;
  using scalar_type = typename TestHelper::scalar_type;
  using element_type = std::remove_pointer_t<decltype(std::declval<Vector&>().data())>;

  const std::size_t n = 1024;
  auto v = h.make_vector(n);
  h.fill_iota(v); // v[i] = i + 1

  // A permutation, so that entry i is routed through a different slot on the way out and back.
  // 37 is coprime to 1024, so i -> (37*i + 13) mod n is a permutation of 0..n-1.
  std::vector<int> perm(n);
  for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<int>((37 * i + 13) % n);
  auto perm_buf = Backend::make_buffer_from_host(Backend::context(v), perm);

  auto ctx = Backend::context(v);
  auto* scratch = Backend::template malloc<element_type>(ctx, n);

  // scatter v -> scratch, zero v, gather scratch -> v
  Backend::scatter(ctx, v.data(), perm_buf, scratch);
  h.fill(v, 0);
  Backend::gather(ctx, scratch, perm_buf, v.data());
  Backend::free(ctx, scratch);

  const auto got = h.to_host_vector(v);
  t.check(got.size() == n, "round trip keeps the vector size");
  for (std::size_t i = 0; i < n; ++i) t.check(got[i] == static_cast<scalar_type>(i + 1), "round trip recovers the values") << "i=" << i << " got=" << got[i];
}

// ---------------------------------------------------------------------------
// Test helpers: one per backend-traited container type
// ---------------------------------------------------------------------------

template <class Scalar = double>
class ISTLTestHelper {
public:
  using scalar_type = Scalar;
  using vector_t = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;
  using backend_t = ddm::backend::backend_of_t<vector_t>;

  vector_t make_vector(std::size_t n) { return vector_t(n); }

  std::vector<Scalar> to_host_vector(const vector_t& v) { return std::vector<Scalar>(v.begin(), v.end()); }

  void fill(vector_t& v, Scalar x) { v = x; }

  void fill_with(vector_t& v, const std::vector<Scalar>& vals)
  {
    for (std::size_t i = 0; i < vals.size(); ++i) v[i] = vals[i];
  }

  void fill_iota(vector_t& v)
  {
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<Scalar>(i + 1);
  }
};

template <class Scalar = double>
class StdVectorTestHelper {
public:
  using scalar_type = Scalar;
  using vector_t = std::vector<Scalar>;
  using backend_t = ddm::backend::backend_of_t<vector_t>;

  vector_t make_vector(std::size_t n) { return vector_t(n); }

  std::vector<Scalar> to_host_vector(const vector_t& v) { return v; }

  void fill(vector_t& v, Scalar x) { std::fill(v.begin(), v.end(), x); }

  void fill_with(vector_t& v, const std::vector<Scalar>& vals)
  {
    for (std::size_t i = 0; i < vals.size(); ++i) v[i] = vals[i];
  }

  void fill_iota(vector_t& v)
  {
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<Scalar>(i + 1);
  }
};

template <class Scalar = double>
class SyclTestHelper {
public:
  using scalar_type = Scalar;
  using vector_t = ddm::Sycl::Vec<Scalar>;
  using backend_t = ddm::backend::backend_of_t<vector_t>;

  explicit SyclTestHelper(sycl::queue q_)
      : q(std::move(q_))
  {
  }

  vector_t make_vector(std::size_t n) { return ddm::Sycl::Vec<Scalar>(q, n); }

  std::vector<Scalar> to_host_vector(const vector_t& v)
  {
    std::vector<Scalar> vh(v.size());
    q.memcpy(vh.data(), v.data(), v.size() * sizeof(Scalar)).wait();
    return vh;
  }

  void fill(vector_t& v, Scalar x) { q.fill(v.data(), x, v.size()); }

  void fill_with(vector_t& v, const std::vector<Scalar>& vals) { q.memcpy(v.data(), vals.data(), vals.size() * sizeof(Scalar)).wait(); }

  void fill_iota(vector_t& v)
  {
    auto* ptr = v.data();
    q.parallel_for(sycl::range<1>(v.size()), [=](auto id) { ptr[id] = static_cast<Scalar>(id + 1); });
  }

private:
  sycl::queue q;
};

// ---------------------------------------------------------------------------

template <class TestHelper>
int test_backend(TestHelper& h)
{
  return ddmtest::runParallelTest("test_backend", [&](Dune::TestSuite& t) {
    using Vector = typename TestHelper::vector_t;
    using Backend = typename TestHelper::backend_t;
    static_assert(ddm::backend::HasBackend<Vector>);
    static_assert(std::is_same_v<Backend, ddm::backend::backend_of_t<Vector>>);

    // Buffer life-cycle in the context of a vector, the way Communication uses it
    auto ctx = Backend::context(h.make_vector(0));
    auto buf = Backend::template make_buffer<int>(ctx, 123);
    {
      auto buf2 = std::move(buf); // Buffers must be move-constructible
      buf = std::move(buf2);      // and move-assignable
    } // trigger destruction of buf2

    t.check(buf.size() == 123 && !buf.empty(), "buffer created from a vector context");

    check_buffer(h, t);
    check_backend_primitives(h, t);
    check_vector_copy_n(h, t);
    check_vector_roundtrip(h, t);
  });
}
} // namespace

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  int failed = 0;
  ISTLTestHelper istl_helper;
  failed |= test_backend(istl_helper);
  StdVectorTestHelper std_helper;
  failed |= test_backend(std_helper);

  sycl::queue q{sycl::property::queue::in_order{}};
  SyclTestHelper<float> sycl_helper(q);
  failed |= test_backend(sycl_helper);

  return failed;
}
