#pragma once

#include "assemblewrapper.hh"
#include "pdelab_helper.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/pdelab.hh>
#include <memory>

#ifdef DUNE_DDM_HAVE_LUA
#include <lua.hpp>
#endif

template <typename GV>
class LinearElasticityParameters : public Dune::PDELab::LinearElasticityParameterInterface<Dune::PDELab::LinearElasticityParameterTraits<GV, double>, LinearElasticityParameters<GV>> {
public:
  using Traits = Dune::PDELab::LinearElasticityParameterTraits<GV, double>;

  void f(auto&, auto&, auto& y) const { y = G_; }

  bool isDirichlet(auto& ig, const auto& coord) const
  {
    auto xg = ig.geometry().global(coord);
    return std::abs(xg[0]) < 1e-9;
  }

  void u(const auto&, const auto&, auto& y) const { y = 0.0; }

  auto lambda(const auto&, const auto&) const { return lambda_; }
  auto mu(const auto&, const auto&) const { return mu_; }

  // With this function being present, we can use this class in a VTKGridFunctionAdapter to visualise.
  // Here we visualise the value of lambda on the given element.
  void evaluate(const auto& e, const auto& x, auto& y) const { y = lambda(e, x); }

private:
  typename Traits::RangeType G_{0, 0, -10};
  typename Traits::RangeFieldType lambda_{100};
  typename Traits::RangeFieldType mu_{10000};
};

#ifdef DUNE_DDM_HAVE_LUA
template <typename GridView>
class LinearElasticityParametersLua
    : public Dune::PDELab::LinearElasticityParameterInterface<Dune::PDELab::LinearElasticityParameterTraits<GridView, double>, LinearElasticityParametersLua<GridView>> {
public:
  LinearElasticityParametersLua()
      : L(luaL_newstate())
  {
    luaL_openlibs(L);

    if (luaL_dofile(L, "coefficient.lua") != 0) DUNE_THROW(Dune::Exception, lua_tostring(L, -1));

    lua_getglobal(L, "is_dirichlet");
    is_dirichlet_ref = luaL_ref(L, LUA_REGISTRYINDEX);

    lua_getglobal(L, "lambda");
    lambda_ref = luaL_ref(L, LUA_REGISTRYINDEX);

    lua_getglobal(L, "mu");
    mu_ref = luaL_ref(L, LUA_REGISTRYINDEX);
  }

  ~LinearElasticityParametersLua() { lua_close(L); }

  // f is fixed to be gravity in z-direction
  void f(auto&, auto&, auto& y) const
  {
    y[0] = 0;
    y[1] = 0;
    y[2] = -20000;
  }

  bool isDirichlet(auto& ig, const auto& coord) const
  {
    auto xg = ig.geometry().global(coord);
    return lua_call_return_bool(is_dirichlet_ref, xg);
  }

  // Dirichlet condition fixed for now
  void u(const auto&, const auto&, auto& y) const { y = 0.0; }

  auto lambda(const auto& e, const auto& x) const
  {
    auto xg = e.geometry().global(x);
    return lua_call_return_double(lambda_ref, xg);
  }

  auto mu(const auto& e, const auto& x) const
  {
    auto xg = e.geometry().global(x);
    return lua_call_return_double(mu_ref, xg);
  }

  // With this function being present, we can use this class in a VTKGridFunctionAdapter to visualise.
  // Here we visualise the value of lambda on the given element.
  void evaluate(const auto& e, const auto& x, auto& y) const { y = lambda(e, x); }

private:
  double lua_call_return_double(int ref, const auto& x) const
  {
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
    lua_pushnumber(L, x[0]);
    lua_pushnumber(L, x[1]);
    lua_pushnumber(L, x[2]);
    lua_pcall(L, 3, 1, 0);
    auto res = static_cast<double>(lua_tonumber(L, -1));
    lua_pop(L, 1);
    return res;
  }

  bool lua_call_return_bool(int ref, const auto& x) const
  {
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
    lua_pushnumber(L, x[0]);
    lua_pushnumber(L, x[1]);
    lua_pushnumber(L, x[2]);
    lua_pcall(L, 3, 1, 0);
    auto res = static_cast<bool>(lua_toboolean(L, -1));
    lua_pop(L, 1);
    return res;
  }

  lua_State* L;

  int is_dirichlet_ref;
  int lambda_ref;
  int mu_ref;
};
#endif

template <class GridView>
class LinearElasticityProblem {
public:
  static constexpr auto dim = GridView::dimension;

  using DF = typename GridView::Grid::ctype;

  // Setup types for entity set, finite element map and problem parameters
  using ES = Dune::PDELab::AllEntitySet<GridView>;
  using FEM = Dune::PDELab::PkLocalFiniteElementMap<ES, DF, double, 1>;

  template <typename T>
#ifdef DUNE_DDM_HAVE_LUA
  using Parameters = LinearElasticityParametersLua<T>;
#else
  using Parameters = LinearElasticityParameters<T>;
#endif
  using ModelProblem = Parameters<ES>;

  using Constraints = Dune::PDELab::ConformingDirichletConstraints;
  using ComponentVectorBackend = Dune::PDELab::ISTL::VectorBackend<>;
  using Mapper = Dune::PDELab::DefaultLeafOrderingTag;
  using OrderingTag = Dune::PDELab::LexicographicOrderingTag;
  using GFS = Dune::PDELab::VectorGridFunctionSpace<ES, FEM, dim, Dune::PDELab::ISTL::VectorBackend<>, ComponentVectorBackend, Constraints, OrderingTag, Mapper>;
  using CC = typename GFS::template ConstraintsContainer<double>::Type;

  // Local and global operator
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using LOP = Dune::PDELab::LinearElasticity<ModelProblem>;
  using GOP = Dune::PDELab::GridOperator<GFS, GFS, AssembleWrapper<LOP>, MBE, double, double, double, CC, CC>;

  // Matrix and vector types
  using Vec = Dune::PDELab::Backend::Vector<GFS, double>;
  using Mat = typename GOP::Jacobian;
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;
  using NativeVec = Dune::PDELab::Backend::Native<Vec>;

  LinearElasticityProblem(GridView gv, const Dune::MPIHelper& helper)
      : // modelproblem_()
        // ,
      es_(gv)
      , fem_(es_)
      , gfs_(std::make_shared<GFS>(es_, fem_))
      , lop_(modelproblem_)
      , asw_(&lop_)
      , x_(std::make_unique<Vec>(*gfs_))
      , r_(std::make_unique<Vec>(*gfs_))
      , dirichlet_mask_(std::make_unique<Vec>(*gfs_))
  {
    cc_.clear();
    Dune::PDELab::constraints(modelproblem_, *gfs_, cc_);

    // Set boundary conditions in x
    Dune::PDELab::LinearElasticityDirichletExtensionAdapter g(es_, modelproblem_);
    Dune::PDELab::interpolate(g, *gfs_, *x_);
    Dune::PDELab::set_nonconstrained_dofs(cc_, 0., *x_);

    // Set up global operator
    MBE mbe(27);
    gop_ = std::make_unique<GOP>(*gfs_, cc_, *gfs_, cc_, asw_, mbe);

    // Set up stiffness matrix (this creates the sparsity pattern but does not yet assemble)
    A_ = std::make_unique<Mat>(*gop_);

    // Initialiase Dirichlet mask
    *dirichlet_mask_ = 0;
    Dune::PDELab::set_constrained_dofs(cc_, 1., *dirichlet_mask_);

    // Assemble residual
    *x_ = 0;
    gop_->residual(*x_, *r_);
    if (helper.size() > 1) {
      Dune::PDELab::AddDataHandle adddhd(*gfs_, *r_);
      gv.communicate(adddhd, Dune::All_All_Interface, Dune::ForwardCommunication);
    }
  }

  template <class Communication>
  std::tuple<std::shared_ptr<NativeMat>, std::shared_ptr<NativeMat>, std::shared_ptr<NativeMat>, std::shared_ptr<NativeVec>> assemble_overlapping_matrices(Communication& comm, int overlap = 1)
  {
    auto [matrices, dirichlet_mask_ovlp, neumann_region_to_subdomain] = ::assemble_overlapping_matrices(*A_, *x_, *gop_, dirichlet_mask(), comm, NeumannRegion::All, NeumannRegion::All, overlap, true);

    return {matrices.A_dir, matrices.A_neu, matrices.B_neu, Dune::PDELab::Backend::native(dirichlet_mask_ovlp)};
  }

  std::shared_ptr<GFS> gfs() const { return gfs_; }

  const auto& A() const { return Dune::PDELab::Backend::native(*A_); }
  std::shared_ptr<NativeMat> A_ptr() const { return A_->storage(); }
  auto& x() const { return Dune::PDELab::Backend::native(*x_); }
  auto& x_pdelab() const { return *x_; }
  auto& r() const { return Dune::PDELab::Backend::native(*r_); }
  const auto& dirichlet_mask() const { return Dune::PDELab::Backend::native(*dirichlet_mask_); }

private:
  ModelProblem modelproblem_;

  // Grid/finite element data structures
  ES es_;
  FEM fem_;
  std::shared_ptr<GFS> gfs_;
  CC cc_;

  LOP lop_;
  AssembleWrapper<LOP> asw_;

  std::unique_ptr<GOP> gop_;
  std::unique_ptr<Mat> A_;
  std::unique_ptr<Vec> x_;
  std::unique_ptr<Vec> r_;
  std::unique_ptr<Vec> dirichlet_mask_;
};
