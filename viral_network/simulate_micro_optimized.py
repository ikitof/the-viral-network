"""Optimized micro-scale simulation with Numba and multiprocessing.

This module provides high-performance simulation for N up to 1 billion nodes.
Uses Numba JIT compilation, sparse matrices, and parallel processing.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import sparse
from loguru import logger
import multiprocessing as mp
from functools import partial

try:
    from viral_network.numba_kernels import (
        step_push_frontier,
        compact_true_indices,
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, using fallback implementations")


@dataclass
class SimulationStateOptimized:
    """Compact state snapshot for optimized micro simulation."""
    t: int
    I_count: int
    R_count: int


class SimulatorMicroOptimized:
    """Optimized micro-scale simulator for large N (up to 1B).

    Optimizations:
    - Numba JIT compilation for hot loops
    - Sparse matrix representation
    - Set-based state tracking (O(1) lookups)
    - Multiprocessing for graph generation
    - Vectorized operations where possible
    """

    def __init__(
        self,
        adj: sparse.csr_matrix,
        node_to_cluster: np.ndarray,
        config,
        n_workers: Optional[int] = None,
    ):
        """Initialize optimized simulator.

        Args:
            adj: Sparse adjacency matrix (CSR format)
            node_to_cluster: Cluster assignment per node
            config: Configuration object
            n_workers: Number of worker processes (default: CPU count)
        """
        self.adj = adj
        self.node_to_cluster = node_to_cluster
        self.config = config
        self.N = adj.shape[0]
        self.K = len(np.unique(node_to_cluster))

        self.n_workers = n_workers or mp.cpu_count()
        self.rng = np.random.RandomState(config.seed)

        # Extract compact CSR arrays (int32) and drop adjacency to reduce RAM
        self.indptr = adj.indptr.astype(np.int32, copy=True)
        self.indices = adj.indices.astype(np.int32, copy=True)
        # Allow GC to free CSR data if not otherwise referenced
        try:
            self.adj = None
        except Exception:
            pass

        logger.info(
            f"Initialized SimulatorMicroOptimized: N={self.N}, K={self.K}, "
            f"workers={self.n_workers}, numba={'enabled' if NUMBA_AVAILABLE else 'disabled'}"
        )

    def _build_neighbor_dict(self) -> Dict[int, np.ndarray]:
        """Build neighbor dictionary from sparse matrix."""
        neighbors = {}
        for i in range(self.N):
            row_start = self.adj.indptr[i]
            row_end = self.adj.indptr[i + 1]
            neighbors[i] = self.adj.indices[row_start:row_end]
        return neighbors

    def run(
        self,
        initial_seeds: List[int],
        target_node: int,
    ) -> Tuple[List[SimulationStateOptimized], Dict]:
        """Run optimized simulation using frontier + Numba kernels and compact memory."""
        logger.info(
            f"Starting optimized simulation: seeds={initial_seeds}, "
            f"target={target_node} (cluster {self.node_to_cluster[target_node]})"
        )

        N = self.N
        fanout = int(self.config.fanout)
        p_dropout = float(self.config.p_dropout)

        # State as boolean masks for memory efficiency
        infected = np.zeros(N, dtype=np.bool_)
        recovered = np.zeros(N, dtype=np.bool_)
        susceptible = np.ones(N, dtype=np.bool_)

        # Initialize frontier
        for s in initial_seeds:
            infected[s] = True
            susceptible[s] = False
        frontier = np.array(initial_seeds, dtype=np.int64)

        next_mask = np.zeros(N, dtype=np.bool_)

        # Per-cluster tracking for visualization/diagnostics
        K = self.K
        cur_I_by_cluster = np.zeros(K, dtype=np.int64)
        cur_R_by_cluster = np.zeros(K, dtype=np.int64)
        seed_clusters = self.node_to_cluster[frontier]
        if seed_clusters.size > 0:
            cur_I_by_cluster += np.bincount(seed_clusters, minlength=K)
        first_hit_time_by_cluster = np.full(K, -1, dtype=np.int64)
        if seed_clusters.size > 0:
            for c in np.unique(seed_clusters):
                first_hit_time_by_cluster[int(c)] = 0
        I_by_cluster_series: list = []
        seed_cluster = int(self.node_to_cluster[frontier[0]]) if frontier.size > 0 else None
        target_cluster = int(self.node_to_cluster[target_node])
        states: List[SimulationStateOptimized] = []
        target_reached = False
        target_reach_time = None


        for t in range(self.config.max_steps):
            # Check if target reached
            if infected[target_node] or recovered[target_node]:
                if not target_reached:
                    target_reached = True
                    target_reach_time = t
                    logger.info(f"Target reached at t={t}")

            # Stop if no more infected
            if frontier.size == 0:
                logger.info(f"Simulation ended at t={t}: no more infected nodes")
                break

            # Snapshot counts (compact)
            # Snapshot per-cluster infected before update (aligns with state at time t)
            I_by_cluster_series.append(cur_I_by_cluster.copy())
            states.append(
                SimulationStateOptimized(
                    t=t,
                    I_count=int(infected.sum()),
                    R_count=int(recovered.sum()),
                )
            )

            # Kernel: mark candidate next infections in next_mask
            if NUMBA_AVAILABLE:
                seed = (self.config.seed or 0) + t
                step_push_frontier(self.indptr, self.indices, frontier, fanout, p_dropout, next_mask, seed & 0xFFFFFFFF)
            else:
                # Fallback (Python): similar semantics
                for u in frontier:
                    start, end = int(self.indptr[u]), int(self.indptr[u + 1])
                    deg = end - start
                    if deg <= 0:
                        continue
                    if np.random.random() < p_dropout:
                        continue
                    k = fanout if fanout < deg else deg
                    chosen = set()
                    for _ in range(k):
                        tries = 0
                        while True:
                            v = self.indices[start + np.random.randint(0, deg)]

                            if v not in chosen or tries > 8:
                                chosen.add(int(v))
                                break
                            tries += 1
                    for v in chosen:
                        next_mask[v] = True

            # Only susceptibles can become newly infected
            newly_mask = np.logical_and(next_mask, susceptible)
            next_mask[:] = False  # clear for next round

            # Build next frontier and update masks
            if NUMBA_AVAILABLE:
                next_frontier = compact_true_indices(newly_mask)
            else:
                next_frontier = np.where(newly_mask)[0].astype(np.int64)
            # Per-cluster updates for next step (based on next_frontier)
            if next_frontier.size > 0:
                next_counts = np.bincount(self.node_to_cluster[next_frontier], minlength=K)
            else:
                next_counts = np.zeros(K, dtype=np.int64)

            # Record first-hit times for clusters newly infected at t+1
            new_hit_mask = (first_hit_time_by_cluster < 0) & (next_counts > 0)
            if np.any(new_hit_mask):
                first_hit_time_by_cluster[new_hit_mask] = t + 1

            # Move current infected to recovered, then set new infected
            cur_R_by_cluster += cur_I_by_cluster
            cur_I_by_cluster = next_counts

            # Update S/I/R
            recovered |= infected
            infected[:] = False
            infected[next_frontier] = True
            susceptible[next_frontier] = False

            frontier = next_frontier

        # Final snapshot
        states.append(
            SimulationStateOptimized(
                t=len(states),
                I_count=int(infected.sum()),
                R_count=int(recovered.sum()),
            )
        )

        # Append final per-cluster infected snapshot to align with final state
        I_by_cluster_series.append(cur_I_by_cluster.copy())

        # Prepare arrays for metrics serialization
        times = [s.t for s in states]
        I_counts = [s.I_count for s in states]
        R_counts = [s.R_count for s in states]
        cumulative = [i + r for i, r in zip(I_counts, R_counts)]
        I_by_cluster_arr = np.stack(I_by_cluster_series, axis=1) if I_by_cluster_series else np.zeros((K, 0), dtype=int)
        first_hits_list = [None if int(v) < 0 else int(v) for v in first_hit_time_by_cluster.tolist()]


        # Metrics
        metrics = {
            "target_reached": target_reached,
            "target_reach_time": target_reach_time,
            "seed_cluster": seed_cluster,
            "target_cluster": target_cluster,
            "final_infected_count": int((infected | recovered).sum()),
            "max_concurrent_I": max(I_counts) if I_counts else 0,
            "times": times,
            "I_counts": I_counts,
            "R_counts": R_counts,
            "cumulative_infected": cumulative,
            "I_by_cluster": I_by_cluster_arr.tolist(),
            "first_hit_time_by_cluster": first_hits_list,
        }

        logger.info(f"Simulation complete: target_reached={target_reached}")
        return states, metrics

    def _compute_transmissions_optimized(
        self,
        I: set,
        S: set,
    ) -> set:
        """Compute transmissions with optimization.

        Args:
            I: Infected nodes
            S: Susceptible nodes

        Returns:
            Set of newly infected nodes
        """
        new_infections = set()

        for node in I:
            # Check dropout
            if self.rng.random() < self.config.p_dropout:
                continue

            # Get neighbors
            neighbors = self.neighbors_dict.get(node, np.array([]))
            if len(neighbors) == 0:
                continue

            # Sample fanout neighbors
            k = min(self.config.fanout, len(neighbors))
            sampled = self.rng.choice(neighbors, size=k, replace=False)

            # Add susceptible neighbors to new infections
            for neighbor in sampled:
                if neighbor in S:
                    new_infections.add(neighbor)

        return new_infections


def run_parallel_simulations(
    adj: sparse.csr_matrix,
    node_to_cluster: np.ndarray,
    config,
    num_runs: int,
    n_workers: Optional[int] = None,
) -> List[Dict]:
    """Run multiple simulations in parallel.

    Args:
        adj: Sparse adjacency matrix
        node_to_cluster: Cluster assignment
        config: Configuration
        num_runs: Number of simulations to run
        n_workers: Number of worker processes

    Returns:
        List of metrics from each run
    """
    n_workers = n_workers or mp.cpu_count()

    def run_single(run_id: int) -> Dict:
        """Run a single simulation."""
        simulator = SimulatorMicroOptimized(adj, node_to_cluster, config, n_workers=1)

        # Select target and seeds
        initial_seeds = [run_id % node_to_cluster.shape[0]]
        target_cluster = (run_id + 1) % config.K
        target_nodes = np.where(node_to_cluster == target_cluster)[0]
        target_node = target_nodes[run_id % len(target_nodes)]

        states, metrics = simulator.run(initial_seeds, target_node)
        return metrics

    logger.info(f"Running {num_runs} simulations in parallel with {n_workers} workers")

    with mp.Pool(n_workers) as pool:
        results = pool.map(run_single, range(num_runs))

    return results

