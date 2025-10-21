"""Numba-accelerated kernels for hot loops in simulation.

This module provides JIT-compiled functions for performance-critical operations.
Numba compilation provides 10-100x speedup for tight loops.
"""

import numpy as np
from numba import jit, prange, njit
from typing import Tuple


@njit
def sample_neighbors_numba(
    neighbors_list: np.ndarray,
    neighbor_counts: np.ndarray,
    node_id: int,
    k: int,
    rng_state: np.ndarray,
) -> np.ndarray:
    """Sample k distinct neighbors without replacement (Numba-optimized).
    
    Args:
        neighbors_list: Flattened neighbor list
        neighbor_counts: Count of neighbors per node
        node_id: Node to sample from
        k: Number of neighbors to sample
        rng_state: Random state for reproducibility
    
    Returns:
        Array of sampled neighbor indices
    """
    # Find start and end indices for this node's neighbors
    start_idx = 0
    for i in range(node_id):
        start_idx += neighbor_counts[i]
    
    end_idx = start_idx + neighbor_counts[node_id]
    num_neighbors = end_idx - start_idx
    
    if num_neighbors == 0:
        return np.array([], dtype=np.int64)
    
    k = min(k, num_neighbors)
    
    # Fisher-Yates shuffle for sampling
    neighbors = neighbors_list[start_idx:end_idx].copy()
    
    for i in range(k):
        # Random index from i to end
        j = i + np.random.randint(0, num_neighbors - i)
        neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
    
    return neighbors[:k]


@njit(parallel=True)
def compute_transmission_parallel(
    I_nodes: np.ndarray,
    neighbors_list: np.ndarray,
    neighbor_counts: np.ndarray,
    fanout: int,
    p_dropout: float,
    rng_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transmissions for all infected nodes in parallel.
    
    Args:
        I_nodes: Array of infected node indices
        neighbors_list: Flattened neighbor list
        neighbor_counts: Count of neighbors per node
        fanout: Number of neighbors to contact
        p_dropout: Dropout probability
        rng_seed: Random seed
    
    Returns:
        Tuple of (target_nodes, source_nodes)
    """
    np.random.seed(rng_seed)
    
    targets = []
    sources = []
    
    for idx in prange(len(I_nodes)):
        node_id = I_nodes[idx]
        
        # Check dropout
        if np.random.random() < p_dropout:
            continue
        
        # Sample neighbors
        start_idx = 0
        for i in range(node_id):
            start_idx += neighbor_counts[i]
        
        end_idx = start_idx + neighbor_counts[node_id]
        num_neighbors = end_idx - start_idx
        
        if num_neighbors == 0:
            continue
        
        k = min(fanout, num_neighbors)
        
        # Sample k neighbors
        neighbors = neighbors_list[start_idx:end_idx].copy()
        for i in range(k):
            j = i + np.random.randint(0, num_neighbors - i)
            neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
        
        for i in range(k):
            targets.append(neighbors[i])
            sources.append(node_id)
    
    return np.array(targets, dtype=np.int64), np.array(sources, dtype=np.int64)


@njit
def update_states_numba(
    S: np.ndarray,
    I: np.ndarray,
    R: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update SIR states based on transmissions (Numba-optimized).
    
    Args:
        S: Susceptible set
        I: Infected set
        R: Recovered set
        targets: Target nodes for transmission
        sources: Source nodes for transmission
    
    Returns:
        Updated (S, I, R) sets
    """
    new_S = S.copy()
    new_I = I.copy()
    new_R = R.copy()
    
    # Mark all current I as R
    for node in I:
        new_R = np.append(new_R, node)
    
    # Remove I from S and I
    new_S_list = []
    for node in S:
        is_target = False
        for target in targets:
            if node == target:
                is_target = True
                break
        if not is_target:
            new_S_list.append(node)
    
    new_S = np.array(new_S_list, dtype=np.int64)
    
    # New I are targets that were in S
    new_I_list = []
    for target in targets:
        is_in_S = False
        for node in S:
            if node == target:
                is_in_S = True
                break
        if is_in_S:
            new_I_list.append(target)
    
    new_I = np.array(new_I_list, dtype=np.int64)
    
    return new_S, new_I, new_R


@njit
def count_neighbors_numba(
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Count neighbors for each node from COO format (Numba-optimized).
    
    Args:
        row_indices: Row indices of edges
        col_indices: Column indices of edges
        n_nodes: Total number of nodes
    
    Returns:
        Array of neighbor counts per node
    """
    counts = np.zeros(n_nodes, dtype=np.int64)
    
    for i in range(len(row_indices)):
        row = row_indices[i]
        counts[row] += 1
    
    return counts


@njit(parallel=True)
def generate_sbm_edges_parallel(
    cluster_sizes: np.ndarray,
    p_intra: float,
    p_inter: float,
    rng_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate SBM edges in parallel (Numba-optimized).
    
    Args:
        cluster_sizes: Size of each cluster
        p_intra: Intra-cluster connection probability
        p_inter: Inter-cluster connection probability
        rng_seed: Random seed
    
    Returns:
        Tuple of (row_indices, col_indices)
    """
    np.random.seed(rng_seed)
    
    n_nodes = np.sum(cluster_sizes)
    K = len(cluster_sizes)
    
    # Compute cluster boundaries
    cluster_starts = np.zeros(K, dtype=np.int64)
    for i in range(1, K):
        cluster_starts[i] = cluster_starts[i-1] + cluster_sizes[i-1]
    
    rows = []
    cols = []
    
    # Generate edges within and between clusters
    for i in prange(n_nodes):
        # Find cluster of node i
        cluster_i = 0
        for k in range(K):
            if i < cluster_starts[k] + cluster_sizes[k]:
                cluster_i = k
                break
        
        for j in range(i + 1, n_nodes):
            # Find cluster of node j
            cluster_j = 0
            for k in range(K):
                if j < cluster_starts[k] + cluster_sizes[k]:
                    cluster_j = k
                    break
            
            # Determine connection probability
            if cluster_i == cluster_j:
                p = p_intra
            else:
                p = p_inter
            
            # Add edge with probability p
            if np.random.random() < p:
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
    
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


@njit
def check_target_reached_numba(
    I: np.ndarray,
    R: np.ndarray,
    target_node: int,
) -> bool:
    """Check if target node has been reached (Numba-optimized).
    
    Args:
        I: Infected set
        R: Recovered set
        target_node: Target node to check
    
    Returns:
        True if target is in I or R
    """
    for node in I:
        if node == target_node:
            return True
    
    for node in R:
        if node == target_node:
            return True
    
    return False

