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
    target_reach_time: Optional[int] = None,
    seed_cluster: Optional[int] = None,
    target_cluster: Optional[int] = None,
) -> None:
    """
    Plot infection dynamics over time with optional target annotations.

    Args:
        times: Time steps
        I_counts: Number of infected at each time
        R_counts: Number of recovered at each time
        cumulative: Cumulative infected at each time
        title: Plot title
        output_path: Path to save figure
        target_reach_time: If provided, draw a vertical line at this time
        seed_cluster: Optional seed cluster id for subtitle
        target_cluster: Optional target cluster id for subtitle
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

    # Optional annotation for target reach time
    if target_reach_time is not None:
        for a in axes.ravel():
            a.axvline(x=target_reach_time, color="k", linestyle=":", alpha=0.7)
        ax.text(
            target_reach_time,
            max(max(I_counts), max(R_counts), max(cumulative)) * 0.9,
            f"Target reached at t={target_reach_time}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

    subtitle = []
    if seed_cluster is not None:
        subtitle.append(f"seed cluster={seed_cluster}")
    if target_cluster is not None:
        subtitle.append(f"target cluster={target_cluster}")
    if subtitle:
        fig.suptitle(title + " (" + ", ".join(subtitle) + ")", fontsize=14, fontweight="bold")
    else:
        fig.suptitle(title, fontsize=14, fontweight="bold")

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
    seed_cluster: Optional[int] = None,
    target_cluster: Optional[int] = None,
    target_reach_time: Optional[int] = None,
    first_hit_times: Optional[List[Optional[int]]] = None,
) -> None:
    """
    Plot infection dynamics per cluster over time.

    Args:
        times: Time steps
        I_by_cluster: Array of shape (num_clusters, num_timesteps)
        title: Plot title
        output_path: Path to save figure
        seed_cluster: Optional; highlight this cluster
        target_cluster: Optional; highlight this cluster
        target_reach_time: Optional; vertical line when target was reached
    """
    fig = go.Figure()

    K, T = I_by_cluster.shape
    for k in range(K):
        style = {}
        if target_cluster is not None and k == target_cluster:
            style = dict(line=dict(width=3, color="#d62728"))  # red, thick
        elif seed_cluster is not None and k == seed_cluster:
            style = dict(line=dict(width=2.5, color="#2ca02c"))  # green
        fig.add_trace(
            go.Scatter(
                x=times,
                y=I_by_cluster[k, :],
                mode="lines",
                name=f"Cluster {k}",
                hovertemplate=f"t=%{{x}}<br>I=%{{y}}<extra>Cluster {k}</extra>",
                **style,
            )
        )
        # First hit marker per cluster
        if first_hit_times is not None:
            t_hit = first_hit_times[k] if k < len(first_hit_times) else None
            if t_hit is not None and isinstance(t_hit, int) and 0 <= t_hit < len(times):
                # find index of t_hit in times (times are integers 0..T)
                y_val = I_by_cluster[k, t_hit] if t_hit < I_by_cluster.shape[1] else None
                if y_val is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[times[t_hit]],
                            y=[y_val],
                            mode="markers",
                            marker=dict(symbol="triangle-up", size=8, color="#1f77b4"),
                            name=f"first-hit C{k}",
                            showlegend=False,
                            hovertemplate=f"first hit t=%{{x}}<extra>C{k}</extra>",
                        )
                    )

    if target_reach_time is not None:
        fig.add_vline(x=target_reach_time, line_dash="dot", line_color="black", opacity=0.7)

    subtitle = []
    if seed_cluster is not None:
        subtitle.append(f"seed={seed_cluster}")
    if target_cluster is not None:
        subtitle.append(f"target={target_cluster}")

    fig.update_layout(
        title=title + (" (" + ", ".join(subtitle) + ")" if subtitle else ""),
        xaxis_title="Time (steps)",
        yaxis_title="Number of Infected",
        height=650,
        width=1100,
        hovermode="x unified",
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Saved cluster dynamics plot to {output_path}")
    else:
        fig.show()



def plot_cluster_heatmap_over_time(
    times: List[int],
    I_by_cluster: np.ndarray,
    title: str = "Cluster Infection Heatmap",
    output_path: Optional[Path] = None,
) -> None:
    """
    Heatmap of infections per cluster over time.

    Args:
        times: Time steps (length T)
        I_by_cluster: Array shape (K, T)
        title: Plot title
        output_path: Path to save (HTML)
    """
    K, T = I_by_cluster.shape
    fig = go.Figure(
        data=go.Heatmap(
            z=I_by_cluster,
            x=times,
            y=[f"C{k}" for k in range(K)],
            colorscale="Viridis",
            colorbar=dict(title="I"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (steps)",
        yaxis_title="Cluster",
        height=700,
        width=950,
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Saved cluster heatmap to {output_path}")
    else:
        fig.show()

