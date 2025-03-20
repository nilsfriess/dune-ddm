#include "logger.hh"

#include "spdlog/cfg/argv.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

void setup_loggers(int rank, int &argc, char **&argv)
{
  // Default logger (only active on rank 0)
  if (rank == 0) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$:0] %v"); // Default format
    spdlog::cfg::load_argv_levels(argc, argv);
  }
  else {
    spdlog::set_level(spdlog::level::off); // Silence logs on other ranks
  }

  // Logger for all ranks
  auto all_ranks_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto all_ranks_logger = std::make_shared<spdlog::logger>("all_ranks", all_ranks_sink);
  all_ranks_logger->set_level(spdlog::level::info);
  all_ranks_logger->set_pattern("[%H:%M:%S.%e] [%^%l:%$" + std::to_string(rank) + "] %v");
  spdlog::register_logger(all_ranks_logger);
}
