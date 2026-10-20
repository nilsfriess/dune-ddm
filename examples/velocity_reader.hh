#pragma once

// Reads velocity fields for the Helmholtz examples. They are expected to be written with
// the script in data/segy_to_bin.py (or have compatible format)

#include "ddm/check.hh"

#include <algorithm>
#include <cstddef>
#include <dune/common/fvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace ddm::examples {
struct VelocityField {

  // Evaluates the field at coordinate x, where x[0] is the horizontal position and x[1] the height above the bottom of
  // the model, i.e. the model covers [0, nx * dx] x [0, nz * dz] with the surface at x[1] = nz * dz. Each sample is
  // constant on its cell [ix * dx, (ix + 1) * dx) x [iz * dz, (iz + 1) * dz) (in depth). Points up to one sample spacing
  // outside the model get the value of the nearest sample.
  // Throws if x is further outside the model.
  float operator()(const Dune::FieldVector<double, 2>& x) const
  {
    const double width = nx * dx;
    const double height = nz * dz;
    DDM_CHECK(x[0] >= -dx and x[0] <= width + dx, "VelocityField: operator() x[0]={} out of bounds (0, {})", x[0], width);
    DDM_CHECK(x[1] >= -dz and x[1] <= height + dz, "VelocityField: operator() x[1]={} out of bounds (0, {})", x[1], height);

    const auto depth = height - x[1];
    const auto ix = std::clamp(static_cast<int>(x[0] / dx), 0, nx - 1);
    const auto iz = std::clamp(static_cast<int>(depth / dz), 0, nz - 1);
    const auto index = static_cast<std::size_t>(ix) * nz + iz;
    DDM_ASSERT(index < data.size(), "VelocityField: operator() index out of bounds, got {}, size is {}", index, data.size());
    return data[index];
  }

  std::vector<float> data;
  int nx;
  int nz;
  double dx;
  double dz;
};

// Reads the field described by an .ini file written by data/segy_to_bin.py. The binary file named in it is looked up
// relative to the directory of the .ini file.
inline VelocityField read_velocity_field(const std::string& ini_filename)
{
  Dune::ParameterTree config;
  Dune::ParameterTreeParser::readINITree(ini_filename, config);

  VelocityField field;
  field.nx = config.get<int>("nx");
  field.nz = config.get<int>("nz");
  field.dx = config.get<double>("dx");
  field.dz = config.get<double>("dz");

  const auto filename = std::filesystem::path(ini_filename).parent_path() / config.get<std::string>("file");
  std::ifstream fin(filename, std::ios::binary);
  DDM_CHECK(fin.good(), "read_velocity_field: Could not open file {}", filename.string());

  fin.seekg(0, std::ios::end);
  const auto num_bytes = static_cast<std::size_t>(fin.tellg());
  fin.seekg(0, std::ios::beg);
  const auto num_elements = static_cast<std::size_t>(field.nx) * field.nz;
  DDM_CHECK(num_bytes == num_elements * sizeof(float), "read_velocity_field: {} has {} bytes, expected {} for nx = {}, nz = {}", filename.string(), num_bytes,
            num_elements * sizeof(float), field.nx, field.nz);

  field.data.resize(num_elements);
  fin.read(reinterpret_cast<char*>(field.data.data()), num_elements * sizeof(float));
  DDM_CHECK(fin.good(), "read_velocity_field: Reading {} failed", filename.string());

  return field;
}
} // namespace ddm::examples
