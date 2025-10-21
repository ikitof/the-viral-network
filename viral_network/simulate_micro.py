"""Micro-scale (exact) simulation on explicit graph."""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Set, List
from dataclasses import dataclass, field
from loguru import logger

from viral_network.config import Config
from viral_network.graph_generation import sample_neighbors


@dataclass
class SimulationState:
    """State of a simulation at a given time step."""

    t: int
    S: Set[int] = field(default_factory=set)  # Susceptible
    I: Set[int] = field(default_factory=set)  # Infected (will transmit next)
    R: Set[int] = field(default_factory=set)  # Recovered (already transmitted)
    newly_infected: Set[int] = field(default_factory=set)  # Newly infected this step
    target_reached: bool = False
    target_reach_time: Optional[int] = None


class SimulatorMicro:
    """Micro-scale (exact) simulator on explicit graph."""

    def __init__(
        self,
        adj: sparse.csr_matrix,
        node_to_cluster: np.ndarray,
        config: Config,
    ):
        """
        Initialize micro simulator.

        Args:
            adj: Sparse adjacency matrix (CSR format)
            node_to_cluster: Array mapping node index to cluster
            config: Configuration object
        """
        self.adj = adj
        self.node_to_cluster = node_to_cluster
        self.config = config
        self.N = adj.shape[0]
        self.K = len(np.unique(node_to_cluster))
        self.rng = np.random.RandomState(config.seed)

        logger.info(
            f"Initialized SimulatorMicro: N={self.N}, K={self.K}, "
            f"fanout={config.fanout}, p_dropout={config.p_dropout}"
        )

    def run(
        self,
        initial_seeds: Optional[List[int]] = None,
        target_node: Optional[int] = None,
    ) -> Tuple[List[SimulationState], dict]:
        """
        Run the simulation.

        Args:
            initial_seeds: List of initial infected nodes (default: random in cluster 0)
            target_node: Target node to reach (default: random in different cluster)

        Returns:
            Tuple of (list of states, metrics dict)
        """
        # Initialize seeds
        if initial_seeds is None:
            cluster_id = self.config.initial_seeds.cluster_id or 0
            cluster_nodes = np.where(self.node_to_cluster == cluster_id)[0]
            initial_seeds = [self.rng.choice(cluster_nodes)]

        # Initialize target
        if target_node is None:
            source_cluster = self.node_to_cluster[initial_seeds[0]]
            if self.config.friend_target.must_cross_clusters:
                other_clusters = [c for c in range(self.K) if c != source_cluster]
                if other_clusters:
                    target_cluster = self.rng.choice(other_clusters)
                    cluster_nodes = np.where(self.node_to_cluster == target_cluster)[0]
                    target_node = self.rng.choice(cluster_nodes)
                else:
                    target_node = self.rng.choice(self.N)
            else:
                target_node = self.rng.choice(self.N)

        logger.info(
            f"Starting simulation: seeds={initial_seeds}, target={target_node} "
            f"(cluster {self.node_to_cluster[target_node]})"
        )

        # Initialize state
        state = SimulationState(t=0)
        state.S = set(range(self.N))
        state.I = set(initial_seeds)
        for node in initial_seeds:
            state.S.discard(node)

        states = [state]
        target_reached = False
        target_reach_time = None

        # Simulation loop
        for t in range(1, self.config.max_steps + 1):
            if not state.I:
                logger.info(f"Simulation ended at t={t}: no more infected nodes")
                break

            # Transmission step
            new_I = set()
            for node in state.I:
                # Dropout check
                if self.rng.rand() < self.config.p_dropout:
                    continue

                # Sample neighbors
                neighbors = sample_neighbors(
                    self.adj, node, self.config.fanout, self.rng
                )

                # Attempt transmission
                for neighbor in neighbors:
                    if neighbor in state.S:
                        new_I.add(neighbor)

            # Update state
            new_state = SimulationState(t=t)
            new_state.S = state.S - new_I
            new_state.I = new_I
            new_state.R = state.R | state.I
            new_state.newly_infected = new_I

            # Check if target reached (after transmission)
            if target_node in new_state.I or target_node in new_state.R:
                if not target_reached:
                    target_reached = True
                    target_reach_time = t
                    logger.info(f"Target reached at t={t}")

            new_state.target_reached = target_reached
            new_state.target_reach_time = target_reach_time

            states.append(new_state)
            state = new_state

        # Compute metrics
        metrics = self._compute_metrics(states, target_node, target_reach_time)

        return states, metrics

    def _compute_metrics(
        self, states: List[SimulationState], target_node: int, target_reach_time: Optional[int]
    ) -> dict:
        """Compute metrics from simulation states."""
        times = [s.t for s in states]
        I_counts = [len(s.I) for s in states]
        R_counts = [len(s.R) for s in states]
        cumulative_infected = [len(s.R) + len(s.I) for s in states]

        # Compute inter-cluster transmissions
        inter_cluster_transmissions = 0
        for state in states[1:]:
            for node in state.newly_infected:
                # Find which node in previous I transmitted to this node
                prev_state = states[state.t - 1]
                for source in prev_state.I:
                    neighbors = set(self.adj.getrow(source).nonzero()[1])
                    if node in neighbors:
                        if self.node_to_cluster[source] != self.node_to_cluster[node]:
                            inter_cluster_transmissions += 1
                        break

        metrics = {
            "target_reached": target_reach_time is not None,
            "target_reach_time": target_reach_time,
            "final_infected_count": cumulative_infected[-1] if states else 0,
            "max_concurrent_I": max(I_counts) if I_counts else 0,
            "total_steps": len(states) - 1,
            "inter_cluster_transmissions": inter_cluster_transmissions,
            "times": times,
            "I_counts": I_counts,
            "R_counts": R_counts,
            "cumulative_infected": cumulative_infected,
        }

        return metrics

