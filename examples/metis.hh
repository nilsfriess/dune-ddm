#pragma once

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>
extern "C" {
#include <metis.h>
}

template <class GridView>
std::vector<unsigned int> partitionMETIS(const GridView &gv, const Dune::MPIHelper &helper)
{
  if (helper.rank() == 0) {
    constexpr auto dimension = GridView::dimension;
    // setup METIS parameters
    idx_t constraints = 1;
    idx_t parts = helper.size();
    idx_t ncommonnodes = dimension; // number of nodes elements must have in common to be considered adjacent to each other
                                    // In 2d we want 2 vertices, in 3d we want 3
    idx_t objval{};
    idx_t options[METIS_NOPTIONS]; // use default values for random seed, output and coupling
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    std::vector<idx_t> cells, nodes, element_part, node_part;
    cells.reserve(gv.size(0));
    nodes.reserve(gv.size(dimension));

    // create graph of elements and vertices
    int vertices = 0;
    cells.push_back(vertices);
    for (const auto &element : elements(gv)) {
      int cnt = 0;
      for (const auto &is : intersections(gv, element)) {
        if (is.neighbor()) {
          auto other = is.outside();
          nodes.push_back(gv.indexSet().index(other));
          cnt++;
        }
      }
      vertices += cnt;
      cells.push_back(vertices);
    }

    idx_t element_count = cells.size() - 1;
    idx_t node_count = nodes.size();
    element_part.assign(element_count, 0);

    // actual partition of elements
    auto result = METIS_PartGraphKway(&element_count, &constraints, cells.data(), nodes.data(), nullptr, nullptr, nullptr, &parts, nullptr, nullptr, options, &objval, element_part.data());

    if (result != METIS_OK) {
      DUNE_THROW(Dune::Exception, "Metis could not partition the grid");
    }

    std::vector<unsigned int> part(element_part.size());
    for (std::size_t i = 0; i < part.size(); ++i) {
      part[i] = static_cast<unsigned int>(element_part[i]);
    }

    return part;
  }
  else {
    return {};
  }
}
