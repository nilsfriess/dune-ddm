#pragma once

/** @file logger.hh
    @brief Performance logging utilities for MPI parallel programs.

    This file provides a singleton Logger class for timing events across MPI processes
    and a simple logging library for message logging in MPI environments.
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
#include <cstdlib>
#include <deque>
#include <dune/common/parallel/mpitraits.hh>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <ostream>
#include <ratio>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Simple logging namespace to replace spdlog dependency
 *
 * Provides basic logging functionality with MPI rank awareness.
 * Supports format strings and different log levels.
 */
namespace logger {

enum class Level { trace = 0, debug = 1, info = 2, warn = 3, error = 4, off = 5 };

namespace detail {
// Global logging state
static Level current_level = Level::info;
static int mpi_rank = 0;
static bool initialized = false;
static std::mutex log_mutex;

inline void ensure_initialized()
{
  if (!initialized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    const char* env_level = std::getenv("LOG_LEVEL");
    if (env_level) {
      std::string level_str(env_level);
      if (level_str == "trace") current_level = Level::trace;
      else if (level_str == "debug") current_level = Level::debug;
      else if (level_str == "info") current_level = Level::info;
      else if (level_str == "warn") current_level = Level::warn;
      else if (level_str == "error") current_level = Level::error;
      else if (level_str == "off") current_level = Level::off;
    }
    initialized = true;
  }
}

inline const char* level_name(Level level)
{
  switch (level) {
    case Level::trace: return "trace";
    case Level::debug: return "debug";
    case Level::info: return "info";
    case Level::warn: return "warn";
    case Level::error: return "error";
    case Level::off: return "off";
  }
  return "unknown";
}

// Overload for std::vector types
template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& data)
{
  out << "[ ";
  for (std::size_t i = 0; i < data.size(); ++i) {
    out << data[i];
    if (i + 1 != data.size()) out << ", ";
  }
  out << " ]";
  return out;
}

// Simple format string replacement for {} placeholders
template <typename T>
std::string format_string(const std::string& fmt, T&& value)
{
  std::ostringstream oss;
  oss << value;
  std::string result = fmt;
  size_t pos = result.find("{}");
  if (pos != std::string::npos) result.replace(pos, 2, oss.str());
  return result;
}

template <typename T, typename... Args>
std::string format_string(const std::string& fmt, T&& first, Args&&... args)
{
  std::ostringstream oss;
  oss << first;
  std::string result = fmt;
  size_t pos = result.find("{}");
  if (pos != std::string::npos) {
    result.replace(pos, 2, oss.str());
    return format_string(result, std::forward<Args>(args)...);
  }
  return result;
}

inline std::string format_string(const std::string& fmt) { return fmt; }

template <typename... Args>
void log_impl(Level level, bool all_ranks, const std::string& message, Args&&... args)
{
  ensure_initialized();

  if (level < current_level) return;
  if (!all_ranks && mpi_rank != 0) return;

  std::lock_guard<std::mutex> lock(log_mutex);

  std::string formatted_msg;
  if (sizeof...(args) > 0) formatted_msg = format_string(message, std::forward<Args>(args)...);
  else formatted_msg = message;

  std::cout << "[" << level_name(level) << ":" << mpi_rank << "] " << formatted_msg << std::endl;
}
} // namespace detail

// Set log level
inline void set_level(Level level) { detail::current_level = level; }

// Get current log level
inline Level get_level()
{
  detail::ensure_initialized();
  return detail::current_level;
}

// Initialize with MPI rank (optional, auto-initialized on first use)
inline void init(int rank)
{
  detail::mpi_rank = rank;
  detail::initialized = true;
}

// Logging functions for rank 0 only
template <typename... Args>
void trace(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::trace, false, message, std::forward<Args>(args)...);
}

template <typename... Args>
void debug(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::debug, false, message, std::forward<Args>(args)...);
}

template <typename... Args>
void info(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::info, false, message, std::forward<Args>(args)...);
}

template <typename... Args>
void warn(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::warn, false, message, std::forward<Args>(args)...);
}

template <typename... Args>
void error(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::error, false, message, std::forward<Args>(args)...);
}

// Logging functions for all ranks
template <typename... Args>
void trace_all(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::trace, true, message, std::forward<Args>(args)...);
}

template <typename... Args>
void debug_all(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::debug, true, message, std::forward<Args>(args)...);
}

template <typename... Args>
void info_all(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::info, true, message, std::forward<Args>(args)...);
}

template <typename... Args>
void warn_all(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::warn, true, message, std::forward<Args>(args)...);
}

template <typename... Args>
void error_all(const std::string& message, Args&&... args)
{
  detail::log_impl(Level::error, true, message, std::forward<Args>(args)...);
}

} // namespace logger

#include <cstdlib>
#include <mutex>
#include <regex>
#include <sstream>

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
  static Logger& get()
  {
    static Logger instance;
    return instance;
  }

  Logger() = default;
  Logger(const Logger&) = delete;
  Logger(Logger&&) = delete;
  Logger& operator=(const Logger&) = delete;
  Logger& operator=(Logger&&) = delete;
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
    explicit ScopedLog(Event* event)
        : event(event)
    {
      Logger::get().startEvent(event);
    }
    ~ScopedLog() { Logger::get().endEvent(event); }

    ScopedLog(const ScopedLog&) = delete;
    ScopedLog(ScopedLog&&) = delete;
    ScopedLog& operator=(const ScopedLog&) = delete;
    ScopedLog& operator=(ScopedLog&&) = delete;

  private:
    Event* event;
  };

  /**
   * Register a new family for the logger
   */
  Family* registerFamily(const std::string& name)
  {
    auto& new_family = families.emplace_back();
    new_family.events.reserve(100);
    new_family.name = name;
    return &new_family;
  }

  /**
   * Register a new family or return a pointer to one with the given name if it already exists.
   */
  Family* registerOrGetFamily(const std::string& name)
  {
    for (auto& family : families)
      if (family.name == name) return &family;
    return registerFamily(name);
  }

  /**
   * Register a new event for a given family that has been created with registerFamily() or registerOrGetFamily().
   */
  Event* registerEvent(Family* family, const std::string& event)
  {
    auto& new_event = family->events.emplace_back();
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
  Event* registerEvent(const std::string& family_name, const std::string& event_name) { return registerEvent(registerOrGetFamily(family_name), event_name); }

  /**
   * Register a new event or get it if it already exists.
   *
   * If the family does not yet exist, it is registered as a new family. Same for the event.
   * Note that this performs a linear search over the families and a linear search over the events in the family.
   */
  Event* registerOrGetEvent(const std::string& family_name, const std::string& event_name)
  {
    auto* family = registerOrGetFamily(family_name);
    Event* found_event = nullptr;
    for (auto& event : family->events) {
      if (event.name == event_name) {
        found_event = &event;
        break;
      }
    }

    if (found_event != nullptr) return found_event;
    else return registerEvent(family_name, event_name);
  }

  void startEvent(Event* event)
  {
    if (event->is_running) {
      logger::error("Event '{}' was already started", event->name);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    event->last_start = std::chrono::steady_clock::now();
    event->is_running = true;
  }

  void endEvent(Event* event)
  {
    if (not event->is_running) {
      logger::error("Event '{}' was not started yet", event->name);
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
  void report(MPI_Comm comm, std::ostream& out = std::cout) const
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

    for (const auto& family : families) {
      if (rank == 0) out << std::left << std::setw(22) << family.name << '|' << '\n';
      for (const auto& event : family.events) {
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
    if (rank == 0) out << "------------------------------------------------------------------------------------------\n";
  }

private:
  std::tuple<Duration, Duration, Duration> meanMinMaxTime(MPI_Comm comm, const Event& event) const
  {
    auto val = event.total_time.count();
    std::size_t mean{};
    std::size_t min{};
    std::size_t max{};

    // Use the standard MPI types for size_t
    static_assert(sizeof(std::size_t) == sizeof(unsigned long), "size_t must be unsigned long");
    MPI_Allreduce(&val, &mean, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    MPI_Allreduce(&val, &min, 1, MPI_UNSIGNED_LONG, MPI_MIN, comm);
    MPI_Allreduce(&val, &max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

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
    if (number < 0) digits = 1;
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
 * @brief Setup loggers for MPI parallel programs.
 *
 * This function configures logging in an MPI environment by initializing the logger namespace
 * with the current MPI rank and processing command-line arguments for log level configuration.
 *
 * @param rank The MPI rank of the current process
 * @param argc Reference to argument count (may be modified by argument parsing)
 * @param argv Reference to argument array (may be modified by argument parsing)
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
 *   logger::info("This only appears on rank 0");
 *   logger::info_all("This appears on all ranks with rank number");
 *
 *   MPI_Finalize();
 * }
 * @endcode
 *
 * ### Command Line Log Level Control:
 * You can control log levels via command line arguments:
 * @code{.bash}
 * # Set log level via environment variable
 * LOG_LEVEL=debug ./myprogram
 *
 * # Or via command line argument
 * ./myprogram --log-level=debug
 * @endcode
 */
inline void setup_loggers(int rank, int& argc, char**& argv)
{
  // Initialize the logger with the MPI rank
  logger::init(rank);

  // Process command line arguments for log level
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.find("--log-level=") == 0) {
      std::string level_str = arg.substr(12);
      if (level_str == "trace") logger::set_level(logger::Level::trace);
      else if (level_str == "debug") logger::set_level(logger::Level::debug);
      else if (level_str == "info") logger::set_level(logger::Level::info);
      else if (level_str == "warn") logger::set_level(logger::Level::warn);
      else if (level_str == "error") logger::set_level(logger::Level::error);
      else if (level_str == "off") logger::set_level(logger::Level::off);

      // Remove this argument from argv
      for (int j = i; j < argc - 1; ++j) argv[j] = argv[j + 1];
      argc--;
      i--; // Check this position again
    }
  }
}

/** @} */ // End of Logging group
