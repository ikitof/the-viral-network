"""Command-line interface using Typer."""

import typer
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
import json
from loguru import logger

from viral_network.config import Config
from viral_network.graph_generation import generate_sbm_graph
from viral_network.simulate_micro import SimulatorMicro
from viral_network.simulate_micro_optimized import SimulatorMicroOptimized
from viral_network.simulate_macro import SimulatorMacro
from viral_network.targets import TargetSelector
from viral_network.metrics import MetricsCollector
from viral_network.viz import (
    plot_timeseries,
    plot_inter_cluster_heatmap,
    plot_reach_time_distribution,
)

app = typer.Typer(help="Viral Network Simulation CLI")


@app.command()
def run(
    mode: str = typer.Option("micro", help="Simulation mode: 'micro' or 'macro'"),
    N: int = typer.Option(200000, help="Number of nodes (micro mode)"),
    K: int = typer.Option(50, help="Number of clusters"),
    avg_degree: int = typer.Option(100, help="Average degree"),
    fanout: int = typer.Option(2, help="Fanout (neighbors to transmit to)"),
    p_dropout: float = typer.Option(0.15, help="Dropout probability"),
    max_steps: int = typer.Option(64, help="Maximum simulation steps"),
    runs: int = typer.Option(100, help="Number of runs (macro mode)"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: str = typer.Option("runs/exp001", help="Output directory"),
    micro_impl: str = typer.Option("optimized", help="Micro engine: 'optimized' or 'baseline'"),
    cluster_sizes: str = typer.Option("uniform", help="Cluster size distribution: 'uniform' or 'powerlaw'"),
    workers: Optional[int] = typer.Option(None, help="Workers for graph generation (default: CPU count)"),
) -> None:
    """Run a viral network simulation."""
    logger.info(f"Starting simulation: mode={mode}, N={N}, K={K}")

    # Create config
    config = Config(
        mode=mode,
        N=N,
        K=K,
        avg_degree=avg_degree,
        fanout=fanout,
        p_dropout=p_dropout,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
        output_dir=output_dir,
        cluster_sizes=cluster_sizes,
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_path / "config.json")
    logger.info(f"Saved config to {output_path / 'config.json'}")

    if mode == "micro":
        _run_micro(config, output_path, micro_impl, workers)
    elif mode == "macro":
        _run_macro(config, output_path)
    else:
        logger.error(f"Unknown mode: {mode}")
        raise ValueError(f"Unknown mode: {mode}")


def _run_micro(config: Config, output_path: Path, micro_impl: str, workers: Optional[int]) -> None:
    """Run micro-scale simulation (baseline or optimized)."""
    logger.info(f"Running micro-scale simulation (impl={micro_impl})...")

    # Generate graph
    adj, node_to_cluster, metadata = generate_sbm_graph(
        N=config.N,
        K=config.K,
        avg_degree=config.avg_degree,
        cluster_sizes=config.cluster_sizes,
        intra_strength=config.mixing.intra_strength,
        inter_floor=config.mixing.inter_floor,
        seed=config.seed,
        n_workers=workers,
    )

    # Save metadata
    with open(output_path / "graph_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create simulator
    if micro_impl.lower() in ("optimized", "fast"):
        simulator = SimulatorMicroOptimized(adj, node_to_cluster, config)
    elif micro_impl.lower() in ("baseline", "default"):
        simulator = SimulatorMicro(adj, node_to_cluster, config)
    else:
        raise ValueError(f"Unknown micro_impl: {micro_impl}")

    # Select target
    target_selector = TargetSelector(config, node_to_cluster, simulator.rng)
    initial_seeds = target_selector.select_initial_seeds()
    target_node = target_selector.select_target(initial_seeds[0])

    logger.info(f"Initial seeds: {initial_seeds}, Target: {target_node}")

    # Run simulation
    states, metrics = simulator.run(initial_seeds=initial_seeds, target_node=target_node)

    # Save metrics
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Simulation complete: target_reached={metrics['target_reached']}")

    # Plot
    plot_timeseries(
        metrics["times"],
        metrics["I_counts"],
        metrics["R_counts"],
        metrics["cumulative_infected"],
        title=f"Micro-Scale Simulation ({micro_impl})",
        output_path=output_path / "timeseries.png",
    )


def _run_macro(config: Config, output_path: Path) -> None:
    """Run macro-scale simulation."""
    logger.info("Running macro-scale simulation...")

    # Create cluster sizes
    cluster_sizes = np.ones(config.K, dtype=int) * (config.N // config.K)
    cluster_sizes[: config.N % config.K] += 1

    # Create metrics collector
    collector = MetricsCollector(config.N)

    reach_times = []

    for run_id in range(config.runs):
        logger.info(f"Run {run_id + 1}/{config.runs}")

        simulator = SimulatorMacro(cluster_sizes, config)
        states, metrics = simulator.run()

        collector.add_run(metrics)

        if metrics["target_reached"]:
            reach_times.append(metrics["target_reach_time"])

    # Compute aggregate metrics
    aggregate = collector.compute_aggregate_metrics()

    # Save results
    with open(output_path / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    logger.info(f"Aggregate metrics: {aggregate}")

    # Plot reach time distribution
    if reach_times:
        plot_reach_time_distribution(
            reach_times,
            output_path=output_path / "reach_time_distribution.html",
        )


@app.command()
def plot(
    run_id: str = typer.Option("runs/exp001", help="Run ID (output directory)"),
) -> None:
    """Generate plots from a completed run."""
    logger.info(f"Generating plots for run: {run_id}")

    run_path = Path(run_id)
    if not run_path.exists():
        logger.error(f"Run directory not found: {run_path}")
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    # Load metrics
    metrics_file = run_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        plot_timeseries(
            metrics["times"],
            metrics["I_counts"],
            metrics["R_counts"],
            metrics["cumulative_infected"],
            title=f"Simulation Results: {run_id}",
            output_path=run_path / "timeseries_regenerated.png",
        )
        logger.info("Plots generated successfully")
    else:
        logger.error(f"Metrics file not found: {metrics_file}")


if __name__ == "__main__":
    import numpy as np

    app()

