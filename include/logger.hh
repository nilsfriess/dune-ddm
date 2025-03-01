#pragma once

#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <ostream>
#include <string>
#include <vector>

// TODO: This is a bit sketchy because we preallocate memory for the families and events;
//       as soon as either of the vectors has to be resized, all pointers to the events
//       become invalid. For now it works well enough.
/**
 * A simple logger to log timings for different events in an MPI parallel program.
 *
 * Consider a simple class:
 *
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
 *
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
  static Logger &get()
  {
    static Logger instance;
    return instance;
  }

  Logger() { families.reserve(100); }
  Logger(const Logger &) = delete;
  Logger(Logger &&) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger &operator=(Logger &&) = delete;
  ~Logger() = default;

  struct Event {
    std::string name;

    std::size_t times_called = 0;
    double last_start = 0;
    double total_time = 0;
  };

  struct Family {
    std::string name;
    std::vector<Event> events;
  };

  /**
   * A convenience class to log an event that should end when the lifetime of the ScopedLog object ends
   *
   * Using this class, the example code given above could equivalently be written as follows:
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
   * Register a new family or return a pointer to one with the given name if it already exists
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
   * Register a new event for a given family that has been created with registerFamily() or registerOrGetFamily()
   */
  Event *registerEvent(Family *family, const std::string &event)
  {
    auto &new_event = family->events.emplace_back();
    new_event.name = event;
    return &new_event;
  }

  /**
   * Register a new event for the family with the given \p family_name
   *
   * If the family does not yet exist, it is registered as a new family. In other words
   * this function is just a wrapper around
   * \code{.cpp}
   *   registerEvent(registerOrGetFamily(family_name), event_name);
   * \endcode
   */
  Event *registerEvent(const std::string &family_name, const std::string &event_name) { return registerEvent(registerOrGetFamily(family_name), event_name); }

  void startEvent(Event *event) { event->last_start = MPI_Wtime(); }

  void endEvent(Event *event)
  {
    event->total_time += MPI_Wtime() - event->last_start;
    event->times_called++;
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
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
      out << "\n==========================================================================================\n";
      out << "#                                      Logger report                                     #\n";
      out << "==========================================================================================\n";

      out << "Event                 |   Mean time [s] |    Min time [s] |    Max time [s] | Times called\n";
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
          out << "| " << std::right << std::setw(15) << mean << ' ';
          out << "| " << std::right << std::setw(15) << min << ' ';
          out << "| " << std::right << std::setw(15) << max << ' ';
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
  std::tuple<double, double, double> meanMinMaxTime(MPI_Comm comm, const Event &event) const
  {
    double mean = event.total_time;
    double min = event.total_time;
    double max = event.total_time;

    MPI_Allreduce(MPI_IN_PLACE, &mean, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, comm);

    int size = 0;
    MPI_Comm_size(comm, &size);
    mean /= size;

    return {mean, min, max};
  }

  std::vector<Family> families;
};
