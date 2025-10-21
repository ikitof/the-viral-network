"""Macro-scale (approximate) simulation using cluster-level counts."""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from loguru import logger

from viral_network.config import Config
from viral_network.graph_generation import compute_mixing_matrix


@dataclass
class MacroState:
    """State of macro simulation at a given time step."""

    t: int
    S: np.ndarray = field(default_factory=lambda: np.array([]))  # S[k] = susceptible in cluster k
    I: np.ndarray = field(default_factory=lambda: np.array([]))  # I[k] = infected in cluster k
    R: np.ndarray = field(default_factory=lambda: np.array([]))  # R[k] = recovered in cluster k
    target_reached: bool = False
    target_reach_time: Optional[int] = None


class SimulatorMacro:
    """Macro-scale (approximate) simulator using cluster-level dynamics."""

    def __init__(
        self,
        cluster_sizes: np.ndarray,
        config: Config,
    ):
        """
        Initialize macro simulator.

        Args:
            cluster_sizes: Array of cluster sizes (sum = total population)
            config: Configuration object
        """
        self.cluster_sizes = cluster_sizes
        self.config = config
        self.K = len(cluster_sizes)
        self.N_total = cluster_sizes.sum()
        self.rng = np.random.RandomState(config.seed)

        # Compute mixing matrix
        self.mixing_matrix = compute_mixing_matrix(
            self.K,
            config.avg_degree,
            config.mixing.intra_strength,
            config.mixing.inter_floor,
        )

        logger.info(
            f"Initialized SimulatorMacro: K={self.K}, N_total={self.N_total}, "
            f"fanout={config.fanout}, p_dropout={config.p_dropout}"
        )

    def run(
        self,
        initial_cluster: int = 0,
        target_cluster: Optional[int] = None,
    ) -> Tuple[List[MacroState], dict]:
        """
        Run the macro simulation.

        Args:
            initial_cluster: Cluster to seed infection
            target_cluster: Cluster containing target (default: random other cluster)

        Returns:
            Tuple of (list of states, metrics dict)
        """
        if target_cluster is None:
            other_clusters = [c for c in range(self.K) if c != initial_cluster]
            target_cluster = self.rng.choice(other_clusters) if other_clusters else 0

        logger.info(
            f"Starting macro simulation: initial_cluster={initial_cluster}, "
            f"target_cluster={target_cluster}"
        )

        # Initialize state
        S = self.cluster_sizes.copy().astype(float)
        I = np.zeros(self.K, dtype=float)
        R = np.zeros(self.K, dtype=float)

        # Seed initial cluster
        I[initial_cluster] = 1.0
        S[initial_cluster] -= 1.0

        states = [MacroState(t=0, S=S.copy(), I=I.copy(), R=R.copy())]
        target_reached = False
        target_reach_time = None

        # Simulation loop
        for t in range(1, self.config.max_steps + 1):
            if I.sum() < 1e-6:
                logger.info(f"Simulation ended at t={t}: no more infected")
                break

            # Check if target reached
            if I[target_cluster] > 1e-6 or R[target_cluster] > 1e-6:
                if not target_reached:
                    target_reached = True
                    target_reach_time = t
                    logger.info(f"Target cluster reached at t={t}")

            # Transmission step
            new_I = np.zeros(self.K, dtype=float)

            for k in range(self.K):
                if I[k] < 1e-6:
                    continue

                # Expected transmissions from cluster k
                # Each infected attempts fanout transmissions with prob (1 - p_dropout)
                expected_attempts = I[k] * self.config.fanout * (1 - self.config.p_dropout)

                # Distribute attempts across clusters according to mixing matrix
                for ell in range(self.K):
                    # Probability that a contact from k goes to ell
                    p_contact = self.mixing_matrix[k, ell]

                    # Expected attempts to cluster ell
                    expected_attempts_to_ell = expected_attempts * p_contact

                    # Fraction of susceptible in cluster ell
                    if S[ell] > 0:
                        frac_susceptible = S[ell] / self.cluster_sizes[ell]
                    else:
                        frac_susceptible = 0.0

                    # Expected new infections (cap by available susceptible)
                    expected_new = expected_attempts_to_ell * frac_susceptible
                    expected_new = min(expected_new, S[ell])

                    new_I[ell] += expected_new

            # Stochastic update (Poisson sampling)
            new_I_stochastic = self.rng.poisson(new_I)

            # Update state
            new_I_stochastic = np.minimum(new_I_stochastic, S)  # Cap by susceptible
            S -= new_I_stochastic
            R += I
            I = new_I_stochastic.astype(float)

            new_state = MacroState(
                t=t,
                S=S.copy(),
                I=I.copy(),
                R=R.copy(),
                target_reached=target_reached,
                target_reach_time=target_reach_time,
            )
            states.append(new_state)

        # Compute metrics
        metrics = self._compute_metrics(states, target_cluster, target_reach_time)

        return states, metrics

    def _compute_metrics(
        self, states: List[MacroState], target_cluster: int, target_reach_time: Optional[int]
    ) -> dict:
        """Compute metrics from simulation states."""
        times = [s.t for s in states]
        I_totals = [s.I.sum() for s in states]
        R_totals = [s.R.sum() for s in states]
        cumulative_infected = [s.R.sum() + s.I.sum() for s in states]

        # Compute inter-cluster flow
        inter_cluster_flow = 0.0
        for state in states[1:]:
            prev_state = states[state.t - 1]
            for k in range(self.K):
                for ell in range(self.K):
                    if k != ell and prev_state.I[k] > 0:
                        # Estimate flow from k to ell
                        attempts = prev_state.I[k] * self.config.fanout * (1 - self.config.p_dropout)
                        flow = attempts * self.mixing_matrix[k, ell]
                        inter_cluster_flow += flow

        metrics = {
            "target_reached": target_reach_time is not None,
            "target_reach_time": target_reach_time,
            "final_infected_count": cumulative_infected[-1] if states else 0,
            "max_concurrent_I": max(I_totals) if I_totals else 0,
            "total_steps": len(states) - 1,
            "inter_cluster_flow": inter_cluster_flow,
            "times": times,
            "I_totals": I_totals,
            "R_totals": R_totals,
            "cumulative_infected": cumulative_infected,
        }

        return metrics

