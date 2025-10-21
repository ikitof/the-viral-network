"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger


def plot_timeseries(
    times: List[int],
    I_counts: List[int],
    R_counts: List[int],
    cumulative: List[int],
    title: str = "Infection Dynamics",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot infection dynamics over time.

    Args:
        times: Time steps
        I_counts: Number of infected at each time
        R_counts: Number of recovered at each time
        cumulative: Cumulative infected at each time
        title: Plot title
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # I(t)
    axes[0, 0].plot(times, I_counts, "r-", linewidth=2, label="I(t)")
    axes[0, 0].set_xlabel("Time (steps)")
    axes[0, 0].set_ylabel("Number of Infected")
    axes[0, 0].set_title("Active Infections")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # R(t)
    axes[0, 1].plot(times, R_counts, "b-", linewidth=2, label="R(t)")
    axes[0, 1].set_xlabel("Time (steps)")
    axes[0, 1].set_ylabel("Number of Recovered")
    axes[0, 1].set_title("Cumulative Recovered")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Cumulative
    axes[1, 0].plot(times, cumulative, "g-", linewidth=2, label="Cumulative")
    axes[1, 0].set_xlabel("Time (steps)")
    axes[1, 0].set_ylabel("Cumulative Infected")
    axes[1, 0].set_title("Cumulative Infections")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # All together
    ax = axes[1, 1]
    ax.plot(times, I_counts, "r-", linewidth=2, label="I(t)", alpha=0.7)
    ax.plot(times, R_counts, "b-", linewidth=2, label="R(t)", alpha=0.7)
    ax.plot(times, cumulative, "g--", linewidth=2, label="Cumulative", alpha=0.7)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Count")
    ax.set_title("All Dynamics")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_inter_cluster_heatmap(
    inter_cluster_matrix: np.ndarray,
    title: str = "Inter-Cluster Transmission Heatmap",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot heatmap of inter-cluster transmissions.

    Args:
        inter_cluster_matrix: K x K matrix of inter-cluster flows
        title: Plot title
        output_path: Path to save figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=inter_cluster_matrix,
            colorscale="Viridis",
            text=inter_cluster_matrix,
            texttemplate="%{text:.1f}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Target Cluster",
        yaxis_title="Source Cluster",
        height=600,
        width=700,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Saved heatmap to {output_path}")
    else:
        fig.show()


def plot_reach_time_distribution(
    reach_times: List[int],
    title: str = "Distribution of Time to Reach Target",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot distribution of reach times across runs.

    Args:
        reach_times: List of reach times from multiple runs
        title: Plot title
        output_path: Path to save figure
    """
    fig = go.Figure(
        data=[
            go.Histogram(
                x=reach_times,
                nbinsx=max(10, len(set(reach_times))),
                name="Reach Time",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time to Reach Target (steps)",
        yaxis_title="Frequency",
        height=500,
        width=800,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Saved distribution plot to {output_path}")
    else:
        fig.show()


def plot_cluster_dynamics(
    times: List[int],
    I_by_cluster: np.ndarray,
    title: str = "Infection Dynamics by Cluster",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot infection dynamics per cluster over time.

    Args:
        times: Time steps
        I_by_cluster: Array of shape (num_clusters, num_timesteps)
        title: Plot title
        output_path: Path to save figure
    """
    fig = go.Figure()

    for k in range(I_by_cluster.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=I_by_cluster[k, :],
                mode="lines",
                name=f"Cluster {k}",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (steps)",
        yaxis_title="Number of Infected",
        height=600,
        width=1000,
        hovermode="x unified",
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Saved cluster dynamics plot to {output_path}")
    else:
        fig.show()

