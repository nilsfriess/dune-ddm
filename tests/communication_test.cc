#include "dune/ddm/logger.hh"
#include "tests/test_utils.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/ddm/communication.hh>
#include <dune/grid/yaspgrid.hh>
#include <iostream>
#include <utility>
#include <vector>

int basic_test(const Dune::MPIHelper& helper)
{
  if (helper.size() != 3) {
    std::cout << "This test requires exactly 3 ranks\n";
    return 77;
  }

  return ddmtest::runParallelTest("test_communication_basic", [&](Dune::TestSuite& t) {
    // Test broadcast
    {
      std::vector<ddm::CommunicationNodes> roots;
      if (helper.rank() == 0) {
        roots.push_back({0, 0});
        roots.push_back({1, 1});
        roots.push_back({1, 2});
      }
      else if (helper.rank() == 1) {
        roots.push_back({1, 1});
        roots.push_back({1, 2});
        roots.push_back({1, 3});
        roots.push_back({1, 4});
      }
      else if (helper.rank() == 2) {
        roots.push_back({1, 3});
        roots.push_back({1, 4});
        roots.push_back({2, 5});
        roots.push_back({2, 6});
      }

      ddm::Communication comm(helper.getCommunicator(), roots);

      std::vector<int> expected;
      if (helper.rank() == 0) expected = {1, 2, 2};
      else if (helper.rank() == 1) expected = {2, 2, 2, 2};
      else if (helper.rank() == 2) expected = {2, 2, 3, 3};

      // Blocking interface
      std::vector<int> v(roots.size(), helper.rank() + 1);
      comm.broadcast(v);
      t.check(v == expected, "Communicated vector has correct entries");

      // Non-blocking interface: wait() explicitly ...
      std::fill(v.begin(), v.end(), helper.rank() + 1);
      {
        auto exchange = comm.broadcast(v);
        exchange.wait();
        exchange.wait(); // wait() is idempotent
      }
      t.check(v == expected, "Communicated vector has correct entries (explicit wait)");

      // ... or let the handle go out of scope, which waits for us
      std::fill(v.begin(), v.end(), helper.rank() + 1);
      {
        auto exchange = comm.broadcast(v);
      }
      t.check(v == expected, "A discarded Exchange completes the broadcast in its destructor");

      // Moving hands the exchange over; the moved-from handle must have nothing left to wait for,
      // which the broadcast right after would notice (it would throw if the plan were still busy).
      std::fill(v.begin(), v.end(), helper.rank() + 1);
      {
        auto exchange = comm.broadcast(v);
        auto moved = std::move(exchange);
        moved.wait();
      }
      comm.broadcast(v);
      t.check(v == expected, "A moved-from Exchange leaves the plan free");

      // The same Communication object serves a different element type without rebuilding the
      // pattern: the plans are vector-agnostic, only the message buffers are typed.
      std::vector<double> w(roots.size(), helper.rank() + 1.);
      std::vector<double> expected_w(expected.begin(), expected.end());
      comm.broadcast(w);
      t.check(w == expected_w, "A second element type on the same Communication is broadcast correctly");

      for (auto e : v) std::cout << "[" << helper.rank() << "] " << e << "\n";
      std::cout << std::endl;
    }

    // Test reduction
    {
      std::vector<ddm::CommunicationNodes> roots;
      if (helper.rank() == 0) {
        roots.push_back({0, 0});
        roots.push_back({0, 1});
        roots.push_back({1, 2});
        roots.push_back({1, 3});
      }
      else if (helper.rank() == 1) {
        roots.push_back({1, 2});
        roots.push_back({1, 3});
        roots.push_back({1, 4});
      }
      else if (helper.rank() == 2) {
        roots.push_back({1, 3});
        roots.push_back({1, 4});
        roots.push_back({2, 5});
        roots.push_back({2, 6});
      }

      ddm::Communication comm(helper.getCommunicator(), roots);

      std::vector<int> expected;
      if (helper.rank() == 0) expected = {1, 1, 3, 6};
      else if (helper.rank() == 1) expected = {3, 6, 5};
      else if (helper.rank() == 2) expected = {6, 5, 3, 3};

      std::vector<int> v(roots.size(), helper.rank() + 1);
      comm.reduce(v);
      t.check(v == expected, "Communicated vector has correct entries");

      // A reduction and a broadcast of a different element type can be in flight at the same
      // time: they use separate plans and separate exchangers, so their state cannot collide.
      std::vector<int> expected_bcast;
      if (helper.rank() == 0) expected_bcast = {1, 1, 2, 2};
      else if (helper.rank() == 1) expected_bcast = {2, 2, 2};
      else if (helper.rank() == 2) expected_bcast = {2, 2, 3, 3};

      std::fill(v.begin(), v.end(), helper.rank() + 1);
      std::vector<double> w(roots.size(), helper.rank() + 1.);

      {
        auto reduction = comm.reduce(v);
        auto bcast = comm.broadcast(w);

        // A second exchange on a plan that is already in flight has nowhere to put its data
        bool threw = false;
        try {
          auto again = comm.reduce(v);
        }
        catch (const Dune::InvalidStateException&) {
          threw = true;
        }
        t.check(threw, "Starting a second reduction while one is in flight throws");

        bcast.wait();
        reduction.wait();
      }

      std::vector<double> expected_bcast_w(expected_bcast.begin(), expected_bcast.end());
      t.check(v == expected, "Overlapping reduction completes correctly");
      t.check(w == expected_bcast_w, "Broadcast of another type completes correctly while a reduction is in flight");
    }
  });
}

/* Checks make_communication_from_dune() against the ISTL communication it was built from:
 * a broadcast must reproduce copyOwnerToAll(), a reduction addOwnerCopyToOwnerCopy().
 */
int istl_test(const Dune::MPIHelper& helper)
{
  return ddmtest::runParallelTest("test_communication_istl", [&](Dune::TestSuite& t) {
    constexpr int dim = 2;
    using Grid = Dune::YaspGrid<dim>;
    Grid grid({1., 1.}, {64, 64}, 0ULL, 4);
    auto gv = grid.leafGridView();

    auto oocc = ddmtest::create_communication_for_grid(gv);
    auto comm = ddm::make_communication_from_dune(*oocc);

    const auto n = oocc->indexSet().size();
    const double own_value = helper.rank() + 1.;

    // Broadcast: every copy must end up with the value its owner had.
    {
      std::vector<double> v(n, own_value);
      std::vector<double> expected(n, own_value);
      oocc->copyOwnerToAll(expected, expected);

      comm.broadcast(v);
      t.check(v == expected, "broadcast matches copyOwnerToAll");
    }

    // Reduction: every holder of a shared index must end up with the sum over all holders.
    {
      std::vector<double> v(n, own_value);
      std::vector<double> expected(n, own_value);
      std::vector<double> source(n, own_value);
      oocc->addOwnerCopyToOwnerCopy(source, expected);

      comm.reduce(v);
      t.check(v == expected, "reduction matches addOwnerCopyToOwnerCopy");
    }
  });
}

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  auto ret_basic_test = basic_test(helper);
  auto ret_istl_test = istl_test(helper);

  if (ret_basic_test != 0 and ret_basic_test != 77) return ret_basic_test;
  return ret_istl_test;
}
