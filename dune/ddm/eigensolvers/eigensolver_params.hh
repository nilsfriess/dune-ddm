#pragma once

#include "dune/ddm/logger.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>

struct EigensolverParams {
  enum class Type { Spectra };

  EigensolverParams()
      : ncv(2 * nev)
  {
    logger::info("Eigensolver of type '{}' set up with nev = {}, ncv = {}, maxit = {}, seed = {}, blocksize = {}, tolerance = {}, shift = {}", type_to_string(), nev, ncv, maxit, seed, blocksize,
                 tolerance, shift);
  }

  explicit EigensolverParams(const Dune::ParameterTree& ptree)
  {
    if (ptree.hasKey("nev")) nev = ptree.get<std::size_t>("nev");
    if (ptree.hasKey("ncv")) ncv = ptree.get<std::size_t>("ncv");
    else ncv = 2 * nev;
    if (ptree.hasKey("nev_max")) ncv = ptree.get<std::size_t>("nev_max");
    else nev_max = 2 * nev;
    if (ptree.hasKey("maxit")) maxit = ptree.get<std::size_t>("maxit");
    if (ptree.hasKey("tolerance")) tolerance = ptree.get<double>("tolerance");
    if (ptree.hasKey("shift")) shift = ptree.get<double>("shift");
    if (ptree.hasKey("seed")) seed = ptree.get<std::size_t>("seed");
    if (ptree.hasKey("blocksize")) blocksize = ptree.get<std::size_t>("blocksize");
    if (ptree.hasKey("threshold")) threshold = ptree.get<double>("threshold");

    if (ptree.hasKey("type")) {
      const auto& typestr = ptree.get<std::string>("type");
      if (typestr == "Spectra") type = Type::Spectra;
      else DUNE_THROW(Dune::NotImplemented, "Unknown eigensolver type '" + typestr + "'");
    }

    logger::debug("Eigensolver of type '{}' set up with nev = {}, ncv = {}, maxit = {}, seed = {}, blocksize = {}, tolerance = {}, shift = {}", type_to_string(), nev, ncv, maxit, seed, blocksize,
                  tolerance, shift);
  }

  Type type = Type::Spectra;
  std::size_t nev = 16;
  std::size_t nev_max; // Will be set to 2 * nev in the constructor if not given by user. Only used if threshold is positive
  std::size_t ncv;     // Will be set to 2 * nev in the constructor if not given by user
  std::size_t maxit = 1000;
  std::size_t seed = 1;
  std::size_t blocksize = 8;
  double tolerance = 1e-5;
  double shift = 1e-3;
  double threshold = -0.5;

private:
  std::string type_to_string() const
  {
    switch (type) {
      case Type::Spectra: return "Spectra";
    }
    assert(false && "Unreachable");
    return "";
  }
};
