#pragma once

#include "ddm/vec/vec.hh"

#include <functional>
#include <string_view>

namespace ddm::debug {
/** Lets library code pass vectors to main for debugging, e.g. to visualise vectors that are set up in a preconditioner
 *  and as such not accesible from main(). Since this library supports runtime configuration of solvers/preconditioners,
 *  it might be inconvenient (or impossible) to writer getters to obtain these vectors, and this provides an easy way
 *  (that incurs essentially no runtime overhead when not used) to still access those debug vectors.
 *
 *  In main() one has to create a sink of type VecSink. Library code calls add_vec_to_sink(). Without a sink, this does
 *  nothing. Use has_vec_sink() to check if main has setup a VecSink (this can be used to skip certain preparations that
 *  might be necessary to visualise the vector if it won't be visualised anyway).
 *
 *  Usually, vectors passed to the sink have the same layout as the solution vector, so main can write them the same way.
 *  For a vector on the overlapping subdomains, one would typically create a copy of that vector, zero it on all but
 *  one rank (use vec_sink_debug_rank() to query the debug rank number that was set in main), then added globally using
 *  the overlapping communication object, and finally copied to the first n_original entries of a non-overlapping vector.
 *  See schwarz.hh for an example. Note that this preparation step for an overlapping vector is usually collective,
 *  so the sink must exist either on all ranks or on none.
 *
 *  Example in main:
 *  \code{.cpp}
 *    std::deque<std::vector<double>> data; // the VTKWriter keeps references, a deque does not move its elements
 *    Dune::VTKWriter writer(gv);
 *    ddm::debug::VecSink<double> sink(0, [&](std::string_view name, const ddm::Vec<double>& v) {
 *      const auto view = v.host_view(ddm::read);
 *      writer.addVertexData(data.emplace_back(view.begin(), view.end()), std::string(name));
 *    });
 *    auto P = ddm::create_prec<double>(config, A); // passes its vectors to the sink during setup
 *    writer.write("debug");
 *  \endcode
 *
 *  Example in library code:
 *  \code{.cpp}
 *    if (debug::has_vec_sink<T>()) debug::add_vec_to_sink("pou", pou);
 *  \endcode
 */
template <class T>
struct VecSinkImpl {
  int debug_rank = 0;
  std::function<void(std::string_view name, const Vec<T>& v)> f = nullptr;
};

// The installed sink. It is a static variable in a function template, so it is the same object in all translation
// units.
template <class T>
VecSinkImpl<T>& vec_sink_impl()
{
  static VecSinkImpl<T> sink;
  return sink;
}

// The rank whose subdomain vectors are passed to the sink, or -1 if no sink is installed
template <class T>
int vec_sink_debug_rank()
{
  auto& sink = vec_sink_impl<T>();
  if (sink.f) return sink.debug_rank;
  else return -1;
}

// Passes v to the sink. Does nothing if no sink is installed.
template <class T>
void add_vec_to_sink(std::string_view name, const Vec<T>& v)
{
  auto& sink = vec_sink_impl<T>();
  if (sink.f) sink.f(name, v);
}

// Whether a sink is installed
template <class T>
bool has_vec_sink()
{
  return static_cast<bool>(vec_sink_impl<T>().f);
}

// Installs f as the sink until the VecSink is destroyed
template <class T>
class VecSink {
public:
  explicit VecSink(int debug_rank, std::function<void(std::string_view name, const Vec<T>& v)>&& f)
  {
    vec_sink_impl<T>().f = std::move(f);
    vec_sink_impl<T>().debug_rank = debug_rank;
  }
  ~VecSink() { vec_sink_impl<T>().f = nullptr; }
  VecSink(const VecSink&) = delete;
  VecSink& operator=(const VecSink&) = delete;
};
} // namespace ddm::debug
