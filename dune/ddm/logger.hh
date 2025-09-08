#pragma once

/** @file logger.hh
    @brief Performance logging utilities for MPI parallel programs.

    This file provides a singleton Logger class for timing events across MPI processes
    and a setup function for configuring spdlog loggers in MPI environments.
*/

/**
 * @defgroup Logging Performance Logging
 * @brief Utilities for performance monitoring and logging in MPI parallel programs.
 *
 * This module provides:
 * - A singleton Logger class for timing events and generating performance reports
 * - MPI-aware logging setup utilities
 * - Scoped timing helpers for automatic event timing
 *
 * @{
 */

#include <chrono>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <ostream>
#include <ratio>
#include <string>
#include <vector>

#include <dune/common/parallel/mpitraits.hh>

#include <spdlog/cfg/argv.h>
#include <spdlog/details/registry.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

/**
 * @brief A simple logger to log timings for different events in an MPI parallel program.
 *
 * The logger itself is implemented as a singleton. Consider a simple class:
 * \code{.cpp}
 * class SomeOperation {
 * private:
 *   Logger::Event *apply_event{};
 *
 * public:
 *   SomeOperation() : apply_event{Logger::get().registerEvent("SomeOperation", "apply")} {}
 *
 *   void apply() {
 *     Logger::get().startEvent(apply_event);
 *     ... // some computations that we want to log
 *     Logger::get().endEvent(apply_event);
 *   }
 * };
 * \endcode
 * Here we register an `apply` event for some operation that we want to log.
 * In the main program we could then use this class and output a log report:
 * \code{.cpp}
 * int main(int argc, char *argv) {
 *   MPI_Init(&argc, &argv);
 *
 *   SomeOperation op;
 *   for (int i=0; i<3; ++i) op.apply();
 *
 *   Logger::get().report(MPI_COMM_WORLD);
 *   MPI_Finalize();
 * }
 * \endcode
 * This would create output of the following form:
 * \code
 *  ==========================================================================================
 *  #                                      Logger report                                     #
 *  ==========================================================================================
 *  Event                 |   Mean time [s] |    Min time [s] |    Max time [s] | Times called
 *  ------------------------------------------------------------------------------------------
 *  SomeOperation         |
 *    apply               |         3.00035 |         3.00029 |         3.00038 |            3
 *  ------------------------------------------------------------------------------------------
 * \endcode
 * The timings are the total times that the event took and the average/min/max are computed
 * over all MPI in the communicator that is passed to the `report` method.
 */
class Logger {
public:
  using Duration = std::chrono::duration<std::size_t, std::nano>;

  /**
   * Get the Logger singleton which can be used to register logging events and create reports.
   */
  static Logger &get()
  {
    static Logger instance;
    return instance;
  }

  Logger() = default;
  Logger(const Logger &) = delete;
  Logger(Logger &&) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger &operator=(Logger &&) = delete;
  ~Logger() = default;

  struct Event {
    std::string name;

    std::size_t times_called = 0;
    std::chrono::steady_clock::time_point last_start;
    Duration total_time{0};
    bool is_running = false;
  };

  struct Family {
    std::string name;
    std::vector<Event> events;
  };

  /**
   * A convenience class to log an event that should end when the lifetime of the ScopedLog object ends.
   *
   * Using this class, the example code given in the Logger class could equivalently be written as follows:
   * \code{.cpp}
   * class SomeOperation {
   * private:
   *   Logger::Event *apply_event{};
   *
   * public:
   *   SomeOperation() : apply_event{Logger::get().registerEvent("SomeOperation", "apply")} {}
   *
   *   void apply() {
   *     ScopedLog sl(apply_event);
   *     ... // some computations that we want to log
   *   }
   * };
   * \endcode
   */
  struct ScopedLog {
    explicit ScopedLog(Event *event) : event(event) { Logger::get().startEvent(event); }
    ~ScopedLog() { Logger::get().endEvent(event); }

    ScopedLog(const ScopedLog &) = delete;
    ScopedLog(ScopedLog &&) = delete;
    ScopedLog &operator=(const ScopedLog &) = delete;
    ScopedLog &operator=(ScopedLog &&) = delete;

  private:
    Event *event;
  };

  /**
   * Register a new family for the logger
   */
  Family *registerFamily(const std::string &name)
  {
    auto &new_family = families.emplace_back();
    new_family.events.reserve(100);
    new_family.name = name;
    return &new_family;
  }

  /**
   * Register a new family or return a pointer to one with the given name if it already exists.
   */
  Family *registerOrGetFamily(const std::string &name)
  {
    for (auto &family : families) {
      if (family.name == name) {
        return &family;
      }
    }
    return registerFamily(name);
  }

  /**
   * Register a new event for a given family that has been created with registerFamily() or registerOrGetFamily().
   */
  Event *registerEvent(Family *family, const std::string &event)
  {
    auto &new_event = family->events.emplace_back();
    new_event.name = event;
    return &new_event;
  }

  /**
   * Register a new event for the family with the given \p family_name.
   *
   * If the family does not yet exist, it is registered as a new family. In other words
   * this function is just a wrapper around
   * \code{.cpp}
   *   registerEvent(registerOrGetFamily(family_name), event_name);
   * \endcode
   */
  Event *registerEvent(const std::string &family_name, const std::string &event_name) { return registerEvent(registerOrGetFamily(family_name), event_name); }

  /**
   * Register a new event or get it if it already exists.
   *
   * If the family does not yet exist, it is registered as a new family. Same for the event.
   * Note that this performs a linear search over the families and a linear search over the events in the family.
   */
  Event *registerOrGetEvent(const std::string &family_name, const std::string &event_name)
  {
    auto *family = registerOrGetFamily(family_name);
    Event *found_event = nullptr;
    for (auto &event : family->events) {
      if (event.name == event_name) {
        found_event = &event;
        break;
      }
    }

    if (found_event != nullptr) {
      return found_event;
    }
    else {
      return registerEvent(family_name, event_name);
    }
  }

  void startEvent(Event *event)
  {
    if (event->is_running) {
      spdlog::error("Event '{}' was already started", event->name);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    event->last_start = std::chrono::steady_clock::now();
    event->is_running = true;
  }

  void endEvent(Event *event)
  {
    if (not event->is_running) {
      spdlog::error("Event '{}' was not started yet", event->name);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    event->total_time += std::chrono::steady_clock::now() - event->last_start;
    event->times_called++;
    event->is_running = false;
  }

  /**
   * Create the logging report
   * @param comm An MPI communicator
   * @param &out An output stream (by default is std::cout, could also be a filestream)
   *
   * Only output on rank 0 of the communicator is created. Since this computes means/minima/maxima
   * over all ranks in the communicator, all events that were logged /must/ be created on all MPI
   * ranks in \p comm.
   */
  void report(MPI_Comm comm, std::ostream &out = std::cout) const
  {
    using namespace std::literals;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int size = 0;
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
      out << "\n==========================================================================================\n";
      out << "#                                 Logger report (";
      out << size << " Ranks) " << std::setw(34 - num_digits(size)) << "#\n";
      out << "==========================================================================================\n\n";
    }

    if (rank == 0) {
      out << "------------------------------------------------------------------------------------------\n";
      out << "User Event            |   Mean time [s] |    Min time [s] |    Max time [s] | Times called\n";
      out << "------------------------------------------------------------------------------------------\n";
    }

    for (const auto &family : families) {
      if (rank == 0) {
        out << std::left << std::setw(22) << family.name << '|' << '\n';
      }
      for (const auto &event : family.events) {
        const auto [mean, min, max] = meanMinMaxTime(comm, event);
        if (rank == 0) {
          out << "  " << std::left << std::setw(20) << event.name;
          out << "| " << std::right << std::setw(15) << mean / 1.0s << ' ';
          out << "| " << std::right << std::setw(15) << min / 1.0s << ' ';
          out << "| " << std::right << std::setw(15) << max / 1.0s << ' ';
          out << "| " << std::right << std::setw(12) << event.times_called;

          out << '\n';
        }
      }
    }
    if (rank == 0) {
      out << "------------------------------------------------------------------------------------------\n";
    }
  }

private:
  std::tuple<Duration, Duration, Duration> meanMinMaxTime(MPI_Comm comm, const Event &event) const
  {
    auto val = event.total_time.count();
    std::size_t mean{};
    std::size_t min{};
    std::size_t max{};

    const auto type = Dune::MPITraits<std::size_t>::getType();
    MPI_Allreduce(&val, &mean, 1, type, MPI_SUM, comm);
    MPI_Allreduce(&val, &min, 1, type, MPI_MIN, comm);
    MPI_Allreduce(&val, &max, 1, type, MPI_MAX, comm);

    int size = 0;
    MPI_Comm_size(comm, &size);
    mean /= size;

    return {Duration(mean), Duration(min), Duration(max)};
  }

  // @brief Returns the number of digits of an integer-type number.
  template <class T>
  int num_digits(T number) const
  {
    int digits = 0;
    if (number < 0) {
      digits = 1;
    }
    while (number) {
      number /= 10;
      digits++;
    }
    return digits;
  }

  // Storage for families with stable pointers (deque never invalidates pointers to existing elements)
  std::deque<Family> families;
};

/**
 * @brief Setup spdlog loggers for MPI parallel programs.
 *
 * This function configures logging in an MPI environment by setting up two loggers:
 * 1. **Default logger**: Only active on rank 0, uses format "[HH:MM:SS.mmm] [level:0] message"
 * 2. **All ranks logger**: Active on all ranks, uses format "[HH:MM:SS.mmm] [level:rank] message"
 *
 * The function also processes command-line arguments to set log levels dynamically.
 * On ranks other than 0, the default logger is silenced to avoid duplicate output.
 *
 * @param rank The MPI rank of the current process
 * @param argc Reference to argument count (may be modified by spdlog argument parsing)
 * @param argv Reference to argument array (may be modified by spdlog argument parsing)
 *
 * @note This function should be called early in main() after MPI_Init() but before
 *       any logging operations.
 *
 * ### Usage Example:
 * @code{.cpp}
 * int main(int argc, char** argv) {
 *   MPI_Init(&argc, &argv);
 *
 *   int rank;
 *   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 *
 *   setup_loggers(rank, argc, argv);
 *
 *   // Now you can use logging
 *   spdlog::info("This only appears on rank 0");
 *   spdlog::get("all_ranks")->info("This appears on all ranks with rank number");
 *
 *   MPI_Finalize();
 * }
 * @endcode
 *
 * ### Command Line Log Level Control:
 * You can control log levels via command line arguments using the SPDLOG_LEVEL format:
 * @code{.bash}
 * # Set all loggers to debug level
 * ./myprogram "SPDLOG_LEVEL=debug"
 *
 * # Set all loggers to info level
 * ./myprogram "SPDLOG_LEVEL=info"
 *
 * # Turn off default logger but keep all_ranks logger at debug level
 * ./myprogram "SPDLOG_LEVEL=off,all_ranks=debug"
 * @endcode
 */
inline void setup_loggers(int rank, int &argc, char **&argv)
{
  // Default logger (only active on rank 0)
  if (rank == 0) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$:0] %v"); // Default format
  }

  // Logger for all ranks
  auto all_ranks_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto all_ranks_logger = std::make_shared<spdlog::logger>("all_ranks", all_ranks_sink);

  all_ranks_logger->set_level(spdlog::level::debug);
  all_ranks_logger->set_pattern("[%H:%M:%S.%e] [%^%l:%$" + std::to_string(rank) + "] %v");
  spdlog::register_logger(all_ranks_logger);

  spdlog::cfg::load_argv_levels(argc, argv);
  if (rank != 0) {
    spdlog::default_logger()->set_level(spdlog::level::off); // Silence logs on all ranks except zero for the default logger
  }
}

/** @} */ // End of Logging group
