#pragma once

/** @file pou.hh
 *
 *  Provides functions to compute partitions of unity on overlapping domains.
 *
 *  Partition of unity functions are essential for overlapping domain decomposition
 *  methods, ensuring that the sum of all partition functions equals 1 across the
 *  entire computational domain while providing smooth transitions between subdomains.
 */

#include "datahandles.hh"
#include "helpers.hh"

#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <limits>
#include <string>

/** Types of partition of unity functions available */
enum class PartitionOfUnityType : std::uint8_t {
  Trivial,  ///< Trivial partition: 1 on owned DOFs, 0 on copied DOFs
  Standard, ///< Standard weighting: 1 divided by the number of subdomains sharing each DOF
  Distance  ///< Distance-based weighting: weighted by distance from subdomain boundary (Toselli & Widlund, p. 84)
};

/** @brief Partition of unity class for overlapping domain decomposition

    This class encapsulates a discrete partition of unity function for domain decomposition methods.
    The partition ensures that the sum across all subdomains equals 1 at each DOF.
    It acts as a vector through operator[] while providing access to configuration parameters.
 */
class PartitionOfUnity {
private:
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;

  Vec pou_vector_;            ///< The actual partition of unity vector
  int shrink_;                ///< Shrink factor used in construction
  PartitionOfUnityType type_; ///< Type of partition of unity

public:
  /** @brief Constructor that creates a partition of unity vector on an overlapping subdomain

      Creates a discrete partition of unity function for domain decomposition methods.
      The partition ensures that the sum across all subdomains equals 1 at each DOF.

      @param A           Matrix defined on the overlapping index set (used for connectivity)
      @param comm        Communication object defining the overlapping subdomain structure
      @param pou_type    Type of partition of unity to create
      @param shrink      Shrink factor for oversampling simulation (only affects Distance type).
                         A shrink of n means the partition stops n layers from the boundary.
      @param overlap     The overlap size (only needed for Distance type)
   */
  template <class Mat, class Communication>
  PartitionOfUnity(const Mat& A, const Communication& comm, PartitionOfUnityType pou_type, int shrink = 0, int overlap = 0)
      : shrink_(shrink)
      , type_(pou_type)
  {
    pou_vector_.resize(A.N()); // Initialize partition of unity vector

    // Identify boundary DOFs (except for trivial partition which doesn't need them)
    std::vector<bool> boundary_mask;
    if (pou_type != PartitionOfUnityType::Trivial) {
      IdentifyBoundaryDataHandle ibdh(A, comm.indexSet());

      // Set up communication interface for boundary identification
      Dune::Interface all_all_interface;
      using AllSet = Communication::AllSet;
      AllSet all_att;
      all_all_interface.build(comm.remoteIndices(), all_att, all_att);
      Dune::VariableSizeCommunicator<> var_comm(all_all_interface);
      var_comm.forward(ibdh);
      boundary_mask = ibdh.get_boundary_mask();
    }

    switch (pou_type) {
      case PartitionOfUnityType::Standard: {
        // Standard partition: weight by inverse of subdomain count per DOF
        pou_vector_ = 1.0;
        // Set boundary DOFs to zero initially
        for (std::size_t i = 0; i < pou_vector_.N(); ++i)
          if (boundary_mask[i]) pou_vector_[i] = 0.0;

        // Sum contributions from all subdomains to count how many share each DOF
        comm.addOwnerCopyToAll(pou_vector_, pou_vector_);

        // Create final partition by taking reciprocal (1/count) for interior DOFs
        for (std::size_t i = 0; i < pou_vector_.N(); ++i)
          if (!boundary_mask[i]) pou_vector_[i] = 1. / pou_vector_[i]; // Inverse of subdomain count
          else pou_vector_[i] = 0.0;                                   // Boundary DOFs remain zero
      } break;

      case PartitionOfUnityType::Distance: {
        // Distance-based partition: weight by distance from subdomain boundary

        // Initialize distance array - boundary DOFs have distance 0
        std::vector<int> boundary_dst(comm.indexSet().size(), std::numeric_limits<int>::max() - 1);
        for (std::size_t i = 0; i < boundary_dst.size(); ++i)
          if (boundary_mask[i]) boundary_dst[i] = 0;

        // Compute distances using the distance induced by the matrix graph.
        // TODO: The factor 4*overlap might be larger than necessary, iirc 2*overlap sometimes didn't produce the correct results.
        for (int round = 0; round <= 4 * overlap; ++round) {
          for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
            // Update distance based on neighboring DOFs in the matrix graph
            for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cIt.index()] + 1);
          }
        }

        // Create initial partition weights based on distance (accounting for shrink)
        pou_vector_ = 1;
        for (std::size_t i = 0; i < pou_vector_.N(); ++i) {
          if (boundary_dst[i] <= 4 * overlap) {
            if (boundary_dst[i] <= shrink) pou_vector_[i] = 0; // DOFs within shrink distance are set to zero
            else pou_vector_[i] = boundary_dst[i] - shrink;    // Linear distance weighting
          }
        }

        // Sum weights across all subdomains to normalize
        auto pou_sum = pou_vector_;
        comm.addOwnerCopyToAll(pou_sum, pou_sum);

        // Normalize to create proper partition of unity
        for (std::size_t i = 0; i < pou_vector_.N(); ++i)
          if (!boundary_mask[i]) pou_vector_[i] /= pou_sum[i]; // Normalize by total weight
          else pou_vector_[i] = 0.0;                           // Boundary DOFs remain zero
      } break;

      case PartitionOfUnityType::Trivial: {
        // Trivial partition: 1 on owned DOFs, 0 on copied DOFs
        pou_vector_ = 1.0;

        // Set non-owner DOFs to zero
        for (const auto& idx : comm.indexSet())
          if (idx.local().attribute() != Dune::OwnerOverlapCopyAttributeSet::owner) pou_vector_[idx.local()] = 0;
      } break;
    }
  }

  /** @brief Constructor that creates a partition of unity from parameter configuration

      Convenience constructor that creates a partition of unity from configuration parameters.
      This allows runtime selection of partition type and parameters.

      @param A             Matrix defined on the overlapping index set
      @param comm          Communication object defining the overlapping subdomain structure
      @param ptree         Parameter tree containing configuration
      @param overlap       The overlap size
      @param subtree_name  Name of subtree containing POU parameters (default: "pou")

      Configuration parameters in subtree:
      - `type`: Partition type - "trivial", "standard", or "distance" (default: "distance")
      - `shrink`: Shrinkage factor for oversampling, must be < overlap (default: 0)

      @throws Dune::Exception If partition type is unknown or shrink parameter is invalid
   */
  template <class Mat, class Communication>
  PartitionOfUnity(const Mat& A, const Communication& comm, const Dune::ParameterTree& ptree, int overlap, const std::string& subtree_name = "pou")
      : PartitionOfUnity(A, comm, parse_type(ptree, subtree_name), parse_shrink(ptree, subtree_name, overlap), overlap)
  {
  }

private:
  /** @brief Helper function to parse partition type from parameter tree */
  static PartitionOfUnityType parse_type(const Dune::ParameterTree& ptree, const std::string& subtree_name)
  {
    const auto& subtree = ptree.sub(subtree_name);
    const auto& type_string = subtree.get("type", "distance");

    if (type_string == "trivial") return PartitionOfUnityType::Trivial;
    else if (type_string == "standard") return PartitionOfUnityType::Standard;
    else if (type_string == "distance") return PartitionOfUnityType::Distance;
    else DUNE_THROW(Dune::Exception, "Unknown partition of unity type: " + type_string);
  }

  /** @brief Helper function to parse and validate shrink parameter */
  static unsigned int parse_shrink(const Dune::ParameterTree& ptree, const std::string& subtree_name, int overlap)
  {
    const auto& subtree = ptree.sub(subtree_name);
    auto shrink = subtree.get("shrink", 0);
    if (shrink < 0 or shrink >= overlap) DUNE_THROW(Dune::Exception, "Invalid value for shrink: " + std::to_string(shrink) + " (must be >= 0 and < overlap size " + std::to_string(overlap) + ")");
    return shrink;
  }

public:
  /** @brief Get the shrink parameter used in construction */
  int get_shrink() const { return shrink_; }

  /** @brief Get the type of partition of unity */
  PartitionOfUnityType get_type() const { return type_; }

  /** @brief Get the size of the partition vector */
  std::size_t size() const { return pou_vector_.size(); }

  /** @brief Access vector element (const version) */
  const Dune::FieldVector<double, 1>& operator[](std::size_t i) const { return pou_vector_[i]; }

  /** @brief Access vector element (non-const version) */
  Dune::FieldVector<double, 1>& operator[](std::size_t i) { return pou_vector_[i]; }

  /** @brief Get reference to underlying vector */
  const Vec& vector() const { return pou_vector_; }

  /** @brief Get reference to underlying vector (non-const) */
  Vec& vector() { return pou_vector_; }
};
