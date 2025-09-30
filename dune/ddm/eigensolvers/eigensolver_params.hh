#pragma once

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>

struct EigensolverParams {
  enum class Type { Spectra, SubspaceIteration, /* RAES*/ };

  explicit EigensolverParams(const Dune::ParameterTree& ptree)
  {
    if (ptree.hasKey("nev")) nev = ptree.get<std::size_t>("nev");
    if (ptree.hasKey("maxit")) maxit = ptree.get<std::size_t>("maxit");
    if (ptree.hasKey("tolerance")) tolerance = ptree.get<double>("tolerance");
    if (ptree.hasKey("shift")) shift = ptree.get<double>("shift");
    if (ptree.hasKey("seed")) seed = ptree.get<std::size_t>("seed");
    if (ptree.hasKey("blocksize")) blocksize = ptree.get<std::size_t>("blocksize");
    const auto& typestr = ptree.get<std::string>("type");
    if (typestr == "Spectra") type = Type::Spectra;
    else if (typestr == "SubspaceIteration") type = Type::SubspaceIteration;
    // else if (typestr == "RAES") {
    //   type = Type::RAES;
    // }
    else DUNE_THROW(Dune::NotImplemented, "Unknown eigensolver type '" + typestr + "'");
  }

  Type type;
  std::size_t nev = 16;
  std::size_t maxit = 1000;
  std::size_t seed = 1;
  std::size_t blocksize = 8;
  double tolerance = 1e-5;
  double shift = 1e-3;
};
