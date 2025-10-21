"""Metrics collection and analysis."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class Metrics:
    """Container for simulation metrics."""

    target_reached: bool
    target_reach_time: Optional[int]
    final_infected_count: int
    max_concurrent_I: int
    total_steps: int
    inter_cluster_transmissions: int
    attack_rate: float  # Fraction of population infected
    R_eff: float  # Effective reproduction number
    die_out_rate: float  # Fraction of runs that die out before reaching target

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates metrics from simulations."""

    def __init__(self, N_total: int):
        """
        Initialize metrics collector.

        Args:
            N_total: Total population size
        """
        self.N_total = N_total
        self.runs: List[Dict] = []

    def add_run(self, metrics: dict) -> None:
        """Add metrics from a single run."""
        self.runs.append(metrics)

    def compute_aggregate_metrics(self) -> dict:
        """
        Compute aggregate metrics across all runs.

        Returns:
            Dictionary of aggregate metrics
        """
        if not self.runs:
            return {}

        target_reached_count = sum(1 for r in self.runs if r.get("target_reached", False))
        reach_times = [
            r["target_reach_time"]
            for r in self.runs
            if r.get("target_reached", False) and r.get("target_reach_time") is not None
        ]

        final_counts = [r.get("final_infected_count", 0) for r in self.runs]
        max_I_values = [r.get("max_concurrent_I", 0) for r in self.runs]

        aggregate = {
            "num_runs": len(self.runs),
            "target_reach_probability": target_reached_count / len(self.runs),
            "mean_reach_time": np.mean(reach_times) if reach_times else None,
            "median_reach_time": np.median(reach_times) if reach_times else None,
            "std_reach_time": np.std(reach_times) if reach_times else None,
            "min_reach_time": min(reach_times) if reach_times else None,
            "max_reach_time": max(reach_times) if reach_times else None,
            "mean_final_infected": np.mean(final_counts),
            "std_final_infected": np.std(final_counts),
            "mean_max_concurrent_I": np.mean(max_I_values),
            "die_out_rate": 1.0 - target_reached_count / len(self.runs),
        }

        return aggregate

    @staticmethod
    def compute_R_eff(I_counts: List[int]) -> float:
        """
        Compute effective reproduction number.

        R_eff ≈ mean(I[t+1] / I[t]) for t where I[t] > 0
        """
        ratios = []
        for t in range(len(I_counts) - 1):
            if I_counts[t] > 0:
                ratio = I_counts[t + 1] / I_counts[t]
                ratios.append(ratio)

        return np.mean(ratios) if ratios else 0.0

    @staticmethod
    def compute_attack_rate(final_infected: int, N_total: int) -> float:
        """Compute attack rate (fraction of population infected)."""
        return final_infected / N_total if N_total > 0 else 0.0

    @staticmethod
    def compute_frontier_size(
        I_counts_by_cluster: List[np.ndarray],
    ) -> List[int]:
        """
        Compute frontier size (number of clusters with active infection).

        Args:
            I_counts_by_cluster: List of I counts per cluster at each time step

        Returns:
            List of frontier sizes over time
        """
        frontier_sizes = []
        for I_cluster in I_counts_by_cluster:
            frontier = np.sum(I_cluster > 0)
            frontier_sizes.append(int(frontier))
        return frontier_sizes

