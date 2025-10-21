"""Graph generation utilities for SBM (Stochastic Block Model)."""

import numpy as np
import networkx as nx
from scipy import sparse
from typing import Tuple, Optional
from loguru import logger


def compute_mixing_matrix(
    K: int,
    avg_degree: int,
    intra_strength: float = 0.95,
    inter_floor: float = 0.05,
) -> np.ndarray:
    """
    Compute the mixing matrix P for SBM.

    Args:
        K: Number of clusters
        avg_degree: Target average degree
        intra_strength: Probability of intra-cluster edges (diagonal)
        inter_floor: Minimum inter-cluster edge probability

    Returns:
        K x K mixing matrix where P[i,j] = probability of edge between clusters i and j
    """
    # Initialize matrix
    P = np.zeros((K, K))

    # Intra-cluster (diagonal)
    p_intra = intra_strength

    # Inter-cluster: distribute remaining probability
    # Total expected degree = avg_degree
    # If uniform cluster sizes: each node has ~(N/K) nodes in its cluster
    # and ~(N - N/K) nodes outside
    # Expected degree from intra: p_intra * (N/K - 1) ≈ p_intra * N/K
    # Expected degree from inter: p_inter * (N - N/K) ≈ p_inter * N
    # We want: p_intra * N/K + p_inter * N ≈ avg_degree
    # So: p_inter ≈ (avg_degree - p_intra * N/K) / N

    # Simplified: distribute inter_floor uniformly across off-diagonal
    p_inter = inter_floor / (K - 1) if K > 1 else 0

    # Fill matrix
    for i in range(K):
        for j in range(K):
            if i == j:
                P[i, j] = p_intra
            else:
                P[i, j] = p_inter

    return P


def generate_sbm_graph(
    N: int,
    K: int,
    avg_degree: int,
    cluster_sizes: str = "powerlaw",
    intra_strength: float = 0.95,
    inter_floor: float = 0.05,
    seed: int = 42,
) -> Tuple[sparse.csr_matrix, np.ndarray, dict]:
    """
    Generate a Stochastic Block Model graph.

    Args:
        N: Total number of nodes
        K: Number of clusters
        avg_degree: Target average degree
        cluster_sizes: "uniform", "powerlaw", or "custom"
        intra_strength: Intra-cluster connection probability
        inter_floor: Inter-cluster connection probability
        seed: Random seed

    Returns:
        Tuple of (adjacency_matrix_csr, node_to_cluster, metadata)
    """
    rng = np.random.RandomState(seed)
    logger.info(f"Generating SBM: N={N}, K={K}, avg_degree={avg_degree}")

    # Generate cluster sizes
    if cluster_sizes == "uniform":
        sizes = np.full(K, N // K)
        sizes[: N % K] += 1
    elif cluster_sizes == "powerlaw":
        # Power-law distribution: size ~ k^(-alpha)
        alpha = 2.0
        sizes = np.random.pareto(alpha - 1, K) + 1
        sizes = (sizes / sizes.sum() * N).astype(int)
        sizes[-1] += N - sizes.sum()  # Adjust last to match N
    else:
        raise ValueError(f"Unknown cluster_sizes: {cluster_sizes}")

    sizes = np.maximum(sizes, 1)  # Ensure at least 1 node per cluster
    logger.info(f"Cluster sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")

    # Compute mixing matrix
    P = compute_mixing_matrix(K, avg_degree, intra_strength, inter_floor)

    # Create node-to-cluster mapping
    node_to_cluster = np.repeat(np.arange(K), sizes)
    assert len(node_to_cluster) == N

    # Generate edges using SBM
    rows, cols = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ci, cj = node_to_cluster[i], node_to_cluster[j]
            p_edge = P[ci, cj]
            if rng.rand() < p_edge:
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)

    # Create sparse adjacency matrix
    data = np.ones(len(rows))
    adj = sparse.csr_matrix(
        (data, (rows, cols)), shape=(N, N), dtype=np.uint8
    )

    # Compute actual average degree
    actual_avg_degree = adj.sum() / N
    logger.info(f"Generated graph: {adj.nnz} edges, avg_degree={actual_avg_degree:.2f}")

    metadata = {
        "N": N,
        "K": K,
        "cluster_sizes": sizes.tolist(),
        "avg_degree_target": avg_degree,
        "avg_degree_actual": float(actual_avg_degree),
        "num_edges": int(adj.nnz),
        "intra_strength": intra_strength,
        "inter_floor": inter_floor,
    }

    return adj, node_to_cluster, metadata


def get_neighbors(adj: sparse.csr_matrix, node: int) -> np.ndarray:
    """Get neighbors of a node from sparse adjacency matrix."""
    return adj.getrow(node).nonzero()[1]


def sample_neighbors(
    adj: sparse.csr_matrix, node: int, k: int, rng: np.random.RandomState
) -> np.ndarray:
    """
    Sample k distinct neighbors of a node without replacement.

    Args:
        adj: Sparse adjacency matrix
        node: Node index
        k: Number of neighbors to sample
        rng: Random state

    Returns:
        Array of sampled neighbor indices
    """
    neighbors = get_neighbors(adj, node)
    if len(neighbors) == 0:
        return np.array([], dtype=int)
    if len(neighbors) <= k:
        return neighbors
    return rng.choice(neighbors, size=k, replace=False)



# Route baseline generator to fast O(E) implementation
from .graph_generation_optimized import generate_sbm_graph_fast as _fast_sbm
generate_sbm_graph = _fast_sbm  # backward-compatible name