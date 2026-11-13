#pragma once

// Configuration and results of the examples that are used for experiments (ex2, ex6):
//  - read_config() reads the options from ini files and the command line,
//  - TimedPhase times one phase of a run (e.g. the setup of the preconditioner or the solve) with barriers around it,
//  - write_results() writes the configuration, the result of the solver and the timings of all ranks to a JSON file.

#include <ddm/logger.hh>
#include <ddm/solver/solver.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <format>
#include <mpi.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ddm::examples {
/** Reads the configuration of a run: first the ini files in the option `-config` (separated by spaces), in this order, so
 *  that a later file overrides an earlier one, then all options `-key value` on the command line, which override the
 *  files. For example
 *
 *    mpirun -np 64 ./ex2 -config "ex2.ini msgfem_ring.ini" -grid.refine 2
 *
 *  Also applies the timing options:
 *  - timing.barriers: barriers between the phases of a run and of the setup of the preconditioner (see
 *                     Logger::set_phase_barriers()), so that every phase is timed from a common start (default false)
 */
inline Dune::ParameterTree read_config(int argc, char** argv)
{
  Dune::ParameterTree command_line;
  Dune::ParameterTreeParser::readOptions(argc, argv, command_line);

  Dune::ParameterTree config;
  if (command_line.hasKey("config")) {
    std::istringstream files(command_line["config"]);
    for (std::string file; files >> file;) Dune::ParameterTreeParser::readINITree(file, config, true);
  }
  Dune::ParameterTreeParser::readOptions(argc, argv, config);

  Logger::get().set_phase_barriers(config.get("timing.barriers", false));
  return config;
}

/** Times a phase of a run as the event "Run"/name, from a phase barrier before it to a phase barrier after it (see
 *  Logger::phase_barrier()). With barriers, the time is the same on all ranks: the wall-clock time of the phase, until
 *  the slowest rank has finished it. Collective.
 */
class TimedPhase {
public:
  TimedPhase(const std::string& name, MPI_Comm comm = MPI_COMM_WORLD)
      : event_(Logger::get().registerOrGetEvent("Run", name))
      , comm_(comm)
  {
    Logger::get().phase_barrier(comm_);
    Logger::get().startEvent(event_);
  }

  ~TimedPhase()
  {
    Logger::get().phase_barrier(comm_);
    Logger::get().endEvent(event_);
  }

  TimedPhase(const TimedPhase&) = delete;
  TimedPhase& operator=(const TimedPhase&) = delete;

private:
  Logger::Event* event_;
  MPI_Comm comm_;
};

namespace detail {
inline void flatten(const Dune::ParameterTree& tree, const std::string& prefix, std::vector<std::pair<std::string, std::string>>& out)
{
  for (const auto& key : tree.getValueKeys()) out.emplace_back(prefix + key, tree[key]);
  for (const auto& sub : tree.getSubKeys()) flatten(tree.sub(sub), prefix + sub + ".", out);
}
} // namespace detail

/** Writes the run to the JSON file in the option output.json, nothing if it is not set. The file contains the timings and
 *  stats of all ranks (see Logger::write_json()), and as "meta"
 *  - "config":  all options, as strings (keys with dots, e.g. "prec.coarse_space.method"),
 *  - "results": the statistics of the last solve of the solver (iterations, converged, reduction, rate and solve
 *               time), and the given extra results, each a name and a JSON value (e.g. {"dofs", "123456"}).
 *  Collective.
 */
template <class T>
void write_results(const Dune::ParameterTree& config, const Solver<T>& solver, const std::vector<std::pair<std::string, std::string>>& extra = {}, MPI_Comm comm = MPI_COMM_WORLD)
{
  if (!config.hasKey("output.json")) return;

  std::vector<std::pair<std::string, std::string>> options;
  detail::flatten(config, "", options);
  std::string meta = "{\n    \"config\": {";
  for (std::size_t i = 0; i < options.size(); ++i)
    meta += std::format("{}\n      {}: {}", i == 0 ? "" : ",", Logger::json_string(options[i].first), Logger::json_string(options[i].second));
  meta += "\n    },\n    \"results\": {";

  const auto& r = solver.last_result();
  meta += std::format("\n      \"iterations\": {},\n      \"converged\": {},\n      \"reduction\": {:.6e},\n      \"rate\": {:.6e},\n      \"solve time\": {:.6e}", r.iterations,
                      r.converged ? "true" : "false", r.reduction, r.rate, r.elapsed);
  for (const auto& [name, value] : extra) meta += std::format(",\n      {}: {}", Logger::json_string(name), value);
  meta += "\n    }\n  }";

  Logger::get().write_json(comm, config["output.json"], meta);
}
} // namespace ddm::examples
