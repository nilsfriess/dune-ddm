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

  Family *registerFamily(const std::string &name)
  {
    auto &new_family = families.emplace_back();
    new_family.events.reserve(100);
    new_family.name = name;
    return &new_family;
  }

  Event *registerEvent(Family *family, const std::string &event)
  {
    auto &new_event = family->events.emplace_back();
    new_event.name = event;
    return &new_event;
  }

  void startEvent(Event *event) { event->last_start = MPI_Wtime(); }

  void endEvent(Event *event)
  {
    event->total_time += MPI_Wtime() - event->last_start;
    event->times_called++;
  }

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