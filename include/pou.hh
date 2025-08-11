#pragma once

/** @file pou.hh
 *
 *  Provides functions to compute partitions of unity on overlapping domains.
 *  
 *  Partition of unity functions are essential for overlapping domain decomposition
 *  methods, ensuring that the sum of all partition functions equals 1 across the
 *  entire computational domain while providing smooth transitions between subdomains.
 */

#include <limits>
#include <string>

#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bvector.hh>

#include "datahandles.hh"
#include "helpers.hh"

/** Types of partition of unity functions available */
enum class PartitionOfUnityType : std::uint8_t {
  Trivial,  ///< Trivial partition: 1 on owned DOFs, 0 on copied DOFs
  Standard, ///< Standard weighting: 1 divided by the number of subdomains sharing each DOF
  Distance  ///< Distance-based weighting: weighted by distance from subdomain boundary (Toselli & Widlund, p. 84)
};

/** @brief Create a partition of unity vector on an overlapping subdomain

    Creates a discrete partition of unity function for domain decomposition methods.
    The partition ensures that the sum across all subdomains equals 1 at each DOF.

    @param A           Matrix defined on the overlapping index set (used for connectivity)
    @param extids      Extended remote indices defining the overlapping subdomain structure  
    @param pou_type    Type of partition of unity to create
    @param shrink      Shrink factor for oversampling simulation (only affects Distance type).
                       A shrink of n means the partition stops n layers from the boundary.
    @return            Vector representing the discrete partition of unity
 */
template <class Mat, class ExtendedRemoteIndices>
Dune::BlockVector<Dune::FieldVector<double, 1>> create_pou(const Mat &A, const ExtendedRemoteIndices &extids, PartitionOfUnityType pou_type, unsigned int shrink = 0)
{
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;

  // Helper struct for MPI communication: adds values from different processors
  struct AddGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] += v; }
  };

  Vec pou(A.N()); // Initialize partition of unity vector

  // Set up communication interfaces for all-to-all communication between subdomains
  Dune::BufferedCommunicator all_all_comm;
  Dune::Interface all_all_interface;
  AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  all_all_interface.build(extids.get_remote_indices(), allAttributes, allAttributes);
  all_all_comm.build<Vec>(all_all_interface);

  // Identify boundary DOFs (except for trivial partition which doesn't need them)
  std::vector<bool> boundary_mask;
  if (pou_type != PartitionOfUnityType::Trivial) {
    IdentifyBoundaryDataHandle ibdh(A, extids.get_parallel_index_set());
    Dune::VariableSizeCommunicator<> var_comm(all_all_interface);
    var_comm.forward(ibdh);
    boundary_mask = ibdh.get_boundary_mask();
  }

  switch (pou_type) {
  case PartitionOfUnityType::Standard: {
    // Standard partition: weight by inverse of subdomain count per DOF
    pou = 1.0;
    // Set boundary DOFs to zero initially
    for (std::size_t i = 0; i < pou.N(); ++i) {
      if (boundary_mask[i]) {
        pou[i] = 0.0;
      }
    }

    // Sum contributions from all subdomains to count how many share each DOF
    all_all_comm.forward<AddGatherScatter>(pou);

    // Create final partition by taking reciprocal (1/count) for interior DOFs
    for (std::size_t i = 0; i < pou.N(); ++i) {
      if (!boundary_mask[i]) {
        pou[i] = 1. / pou[i]; // Inverse of subdomain count
      }
      else {
        pou[i] = 0.0; // Boundary DOFs remain zero
      }
    }
  } break;

  case PartitionOfUnityType::Distance: {
    // Distance-based partition: weight by distance from subdomain boundary
    
    // Initialize distance array - boundary DOFs have distance 0
    std::vector<int> boundary_dst(extids.size(), std::numeric_limits<int>::max() - 1);
    for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
      if (boundary_mask[i]) {
        boundary_dst[i] = 0;
      }
    }

    // Compute distances using the distance induced by the matrix graph.
    // TODO: The factor 4*overlap might be larger than necessary, iirc 2*overlap sometimes didn't produce the correct results.
    int overlap = extids.get_overlap();
    for (int round = 0; round <= 4 * overlap; ++round) {
      for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
        // Update distance based on neighboring DOFs in the matrix graph
        for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) {
          boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cIt.index()] + 1);
        }
      }
    }

    // Create initial partition weights based on distance (accounting for shrink)
    pou = 1;
    for (std::size_t i = 0; i < pou.N(); ++i) {
      if (boundary_dst[i] <= 4 * overlap) {
        if (boundary_dst[i] <= shrink) {
          pou[i] = 0; // DOFs within shrink distance are set to zero
        }
        else {
          pou[i] = boundary_dst[i] - shrink; // Linear distance weighting
        }
      }
    }

    // Sum weights across all subdomains to normalize
    auto pou_sum = pou;
    all_all_comm.forward<AddGatherScatter>(pou_sum);

    // Normalize to create proper partition of unity
    for (std::size_t i = 0; i < pou.N(); ++i) {
      if (!boundary_mask[i]) {
        pou[i] /= pou_sum[i]; // Normalize by total weight
      }
      else {
        pou[i] = 0.0; // Boundary DOFs remain zero
      }
    }
  } break;

  case PartitionOfUnityType::Trivial: {
    // Trivial partition: 1 on owned DOFs, 0 on copied DOFs
    pou = 1.0;

    // Set non-owner DOFs to zero
    for (const auto &idx : extids.get_parallel_index_set()) {
      if (idx.local().attribute() != Attribute::owner) {
        pou[idx.local()] = 0;
      }
    }
  } break;
  }

  return pou;
}

/** @brief Create a partition of unity vector from parameter configuration

    Convenience function that creates a partition of unity from configuration parameters.
    This allows runtime selection of partition type and parameters.

    @param A             Matrix defined on the overlapping index set  
    @param extids        Extended remote indices defining the overlapping subdomain structure
    @param ptree         Parameter tree containing configuration
    @param subtree_name  Name of subtree containing POU parameters (default: "pou")
    
    Configuration parameters in subtree:
    - `type`: Partition type - "trivial", "standard", or "distance" (default: "distance")
    - `shrink`: Shrinkage factor for oversampling, must be < overlap (default: 0)

    @return              Vector representing the discrete partition of unity
    @throws Dune::Exception If partition type is unknown or shrink parameter is invalid
 */
template <class Mat, class ExtendedRemoteIndices>
Dune::BlockVector<Dune::FieldVector<double, 1>> create_pou(const Mat &A, const ExtendedRemoteIndices &extids, const Dune::ParameterTree &ptree, const std::string &subtree_name = "pou")
{
  const auto &subtree = ptree.sub(subtree_name);

  // Parse partition type from configuration (default: distance-based)
  const auto &type_string = subtree.get("type", "distance");
  PartitionOfUnityType type;
  if (type_string == "trivial") {
    type = PartitionOfUnityType::Trivial;
  }
  else if (type_string == "standard") {
    type = PartitionOfUnityType::Standard;
  }
  else if (type_string == "distance") {
    type = PartitionOfUnityType::Distance;
  }
  else {
    DUNE_THROW(Dune::Exception, "Unknown partition of unity type: " + type_string);
  }

  // Parse and validate shrink parameter
  auto shrink = subtree.get("shrink", 0);
  if (shrink < 0 or shrink >= extids.get_overlap()) {
    DUNE_THROW(Dune::Exception, "Invalid value for shrink: " + std::to_string(shrink) + 
               " (must be >= 0 and < overlap size " + std::to_string(extids.get_overlap()) + ")");
  }

  return create_pou(A, extids, type, shrink);
}
