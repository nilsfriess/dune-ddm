#pragma once

#include <cmath>
#include <dune/common/exceptions.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>
#include <lua.hpp>

/**
 * @brief A problem class to define the PDE coefficients via a Lua script
 *
 * The script must contain the following functions:
 * - `function alpha(x, y)` (2D) or `function alpha(x, y, z)` (3D).
 *   Must return a scalar. It will be used to define the permeability tensor as A(x) = alpha(x) * I.
 * - `function f(x, y)` or `function f(x, y, z)`.
 *   Must return a scalar. Defines the source term of the PDE.
 * - `function g(x, y)` or `function g(x, y, z)`.
 *   Must return a scalar. Defines the value at the Dirichlet boundary.
 * - `function is_dirichlet(x, y)` or `function is_dirichlet(x, y, z)`.
 *   Must return a boolean. If true, the boundary at that position is Dirichlet, otherwise Outflow.
 */
template <class GridView, class RF, bool make_elliptic = false>
class LuaConvectionDiffusionProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  explicit LuaConvectionDiffusionProblem(const std::string& filename)
      : lua_state(luaL_newstate(), [](auto state) { lua_close(state); })
  {
    luaL_openlibs(lua_state.get());
    if (luaL_loadfile(lua_state.get(), filename.c_str()) != 0) DUNE_THROW(Dune::Exception, "Could not open Lua script file: " + filename);

    // Execute the script to load the functions into the global environment
    if (lua_pcall(lua_state.get(), 0, 0, 0) != 0) {
      std::string error_msg = lua_tostring(lua_state.get(), -1);
      lua_pop(lua_state.get(), 1);
      DUNE_THROW(Dune::Exception, "Error executing Lua script: " + error_msg);
    }
  }

  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& xlocal) const
  {
    auto xglobal = e.geometry().global(xlocal);
    double alpha = call_lua_scalar_function("alpha", xglobal);

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) I[i][j] = (i == j) ? alpha : 0.0;
    return I;
  }

  //! Velocity field (convection)
  auto b(const typename Traits::ElementType& e, const typename Traits::DomainType& xlocal) const
  {
    typename Traits::RangeType v(0.0);
    if constexpr (make_elliptic) return v;
    else {
      auto xglobal = e.geometry().global(xlocal);
      double b1 = call_lua_scalar_function("b1", xglobal);
      double b2 = call_lua_scalar_function("b2", xglobal);
      v[0] = b1;
      v[1] = b2;
    }
    return v;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType& e, const typename Traits::DomainType& xlocal) const
  {
    auto xglobal = e.geometry().global(xlocal);
    return call_lua_scalar_function("f", xglobal);
  }

  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, const typename Traits::DomainType& xlocal) const
  {
    auto xglobal = e.geometry().global(xlocal);
    return call_lua_scalar_function("g", xglobal);
  }

  BC bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    auto xglobal = is.geometry().global(x);
    bool is_dirichlet = call_lua_boolean_function("is_dirichlet", xglobal);
    return is_dirichlet ? Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet : Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow;
  }

private:
  std::shared_ptr<lua_State> lua_state;

  // Helper function to call a Lua function that returns a scalar
  template <typename DomainType>
  double call_lua_scalar_function(const char* func_name, const DomainType& x) const
  {
    lua_getglobal(lua_state.get(), func_name);
    if (!lua_isfunction(lua_state.get(), -1)) {
      lua_pop(lua_state.get(), 1);
      DUNE_THROW(Dune::Exception, std::string("Lua function '") + func_name + "' not found");
    }

    // Push individual coordinates as separate arguments
    constexpr int dim = GridView::dimension;
    for (int i = 0; i < dim; i++) lua_pushnumber(lua_state.get(), x[i]);

    // Call the function with dim arguments, expecting 1 result
    if (lua_pcall(lua_state.get(), dim, 1, 0) != 0) {
      std::string error_msg = lua_tostring(lua_state.get(), -1);
      lua_pop(lua_state.get(), 1);
      DUNE_THROW(Dune::Exception, std::string("Error calling Lua function '") + func_name + "': " + error_msg);
    }

    if (!lua_isnumber(lua_state.get(), -1)) {
      lua_pop(lua_state.get(), 1);
      DUNE_THROW(Dune::Exception, std::string("Lua function '") + func_name + "' did not return a number");
    }

    double result = lua_tonumber(lua_state.get(), -1);
    lua_pop(lua_state.get(), 1);
    return result;
  }

  // Helper function to call a Lua function that returns a boolean
  template <typename DomainType>
  bool call_lua_boolean_function(const char* func_name, const DomainType& x) const
  {
    lua_getglobal(lua_state.get(), func_name);
    if (!lua_isfunction(lua_state.get(), -1)) {
      lua_pop(lua_state.get(), 1);
      DUNE_THROW(Dune::Exception, std::string("Lua function '") + func_name + "' not found");
    }

    // Push individual coordinates as separate arguments
    constexpr int dim = GridView::dimension;
    for (int i = 0; i < dim; i++) lua_pushnumber(lua_state.get(), x[i]);

    // Call the function with dim arguments, expecting 1 result
    if (lua_pcall(lua_state.get(), dim, 1, 0) != 0) {
      std::string error_msg = lua_tostring(lua_state.get(), -1);
      lua_pop(lua_state.get(), 1);
      DUNE_THROW(Dune::Exception, std::string("Error calling Lua function '") + func_name + "': " + error_msg);
    }

    if (!lua_isboolean(lua_state.get(), -1)) {
      lua_pop(lua_state.get(), 1);
      DUNE_THROW(Dune::Exception, std::string("Lua function '") + func_name + "' did not return a boolean");
    }

    bool result = lua_toboolean(lua_state.get(), -1);
    lua_pop(lua_state.get(), 1);
    return result;
  }
};
