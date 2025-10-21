#!/usr/bin/env python
"""Simple demonstration of viral network simulation."""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_network.config import Config
from viral_network.graph_generation import generate_sbm_graph
from viral_network.simulate_micro import SimulatorMicro
from viral_network.simulate_macro import SimulatorMacro
from viral_network.targets import TargetSelector
from viral_network.metrics import MetricsCollector
from viral_network.viz import plot_timeseries, plot_reach_time_distribution


def demo_toy_linear_chain():
    """Demonstrate toy linear chain simulation."""
    print("\n" + "=" * 60)
    print("DEMO 1: Toy Linear Chain (6 nodes)")
    print("=" * 60)

    from scipy import sparse

    # Create linear chain: 0-1-2-3-4-5
    rows = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    cols = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))
    node_to_cluster = np.array([0, 0, 0, 1, 1, 1])

    config = Config.toy_linear_chain()
    simulator = SimulatorMicro(adj, node_to_cluster, config)
    states, metrics = simulator.run(initial_seeds=[0], target_node=5)

    print(f"Target reached: {metrics['target_reached']}")
    print(f"Time to reach: {metrics['target_reach_time']} steps")
    print(f"Final infected: {metrics['final_infected_count']} nodes")


def demo_micro_scale():
    """Demonstrate micro-scale simulation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Micro-Scale Simulation (N=5,000)")
    print("=" * 60)

    config = Config(
        mode="micro",
        N=5000,
        K=10,
        avg_degree=50,
        fanout=2,
        p_dropout=0.15,
        max_steps=32,
        seed=42,
    )

    # Generate graph
    adj, node_to_cluster, metadata = generate_sbm_graph(
        N=config.N,
        K=config.K,
        avg_degree=config.avg_degree,
        seed=config.seed,
    )

    print(f"Graph generated: {metadata['num_edges']} edges")
    print(f"Actual avg degree: {metadata['avg_degree_actual']:.2f}")

    # Run simulation
    simulator = SimulatorMicro(adj, node_to_cluster, config)
    target_selector = TargetSelector(config, node_to_cluster, simulator.rng)
    initial_seeds = target_selector.select_initial_seeds()
    target_node = target_selector.select_target(initial_seeds[0])

    states, metrics = simulator.run(initial_seeds=initial_seeds, target_node=target_node)

    print(f"Target reached: {metrics['target_reached']}")
    print(f"Time to reach: {metrics['target_reach_time']} steps")
    print(f"Final infected: {metrics['final_infected_count']} nodes")
    print(f"Attack rate: {metrics['final_infected_count'] / config.N:.2%}")
    print(f"Max concurrent I: {metrics['max_concurrent_I']}")

    # Plot
    output_dir = Path("runs/demo_micro")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_timeseries(
        metrics["times"],
        metrics["I_counts"],
        metrics["R_counts"],
        metrics["cumulative_infected"],
        title="Micro-Scale Simulation (N=5,000)",
        output_path=output_dir / "timeseries.png",
    )
    print(f"Plot saved to {output_dir / 'timeseries.png'}")


def demo_macro_scale():
    """Demonstrate macro-scale simulation with multiple runs."""
    print("\n" + "=" * 60)
    print("DEMO 3: Macro-Scale Simulation (K=20, 50 runs)")
    print("=" * 60)

    config = Config(
        mode="macro",
        K=20,
        avg_degree=100,
        fanout=2,
        p_dropout=0.15,
        max_steps=32,
        seed=42,
    )

    cluster_sizes = np.ones(config.K, dtype=int) * 1000

    # Run multiple simulations
    collector = MetricsCollector(cluster_sizes.sum())
    reach_times = []

    for run_id in range(50):
        simulator = SimulatorMacro(cluster_sizes, config)
        states, metrics = simulator.run()
        collector.add_run(metrics)

        if metrics["target_reached"]:
            reach_times.append(metrics["target_reach_time"])

    # Aggregate results
    aggregate = collector.compute_aggregate_metrics()

    print(f"Target reach probability: {aggregate['target_reach_probability']:.2%}")
    if aggregate['mean_reach_time'] is not None:
        print(f"Mean reach time: {aggregate['mean_reach_time']:.1f} steps")
        print(f"Median reach time: {aggregate['median_reach_time']:.1f} steps")
        print(f"Std reach time: {aggregate['std_reach_time']:.2f}")
    else:
        print("Mean reach time: N/A (target not reached in any run)")
    print(f"Die-out rate: {aggregate['die_out_rate']:.2%}")

    # Plot
    if reach_times:
        output_dir = Path("runs/demo_macro")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_reach_time_distribution(
            reach_times,
            title="Distribution of Time to Reach Target (50 runs)",
            output_path=output_dir / "reach_time_distribution.html",
        )
        print(f"Plot saved to {output_dir / 'reach_time_distribution.html'}")


def demo_parameter_sensitivity():
    """Demonstrate sensitivity to dropout probability."""
    print("\n" + "=" * 60)
    print("DEMO 4: Parameter Sensitivity (Dropout Probability)")
    print("=" * 60)

    cluster_sizes = np.ones(10, dtype=int) * 1000

    for p_dropout in [0.0, 0.1, 0.2, 0.3]:
        config = Config(
            mode="macro",
            K=10,
            avg_degree=100,
            fanout=2,
            p_dropout=p_dropout,
            max_steps=32,
            seed=42,
        )

        collector = MetricsCollector(cluster_sizes.sum())
        reach_times = []

        for _ in range(20):
            simulator = SimulatorMacro(cluster_sizes, config)
            states, metrics = simulator.run()
            collector.add_run(metrics)

            if metrics["target_reached"]:
                reach_times.append(metrics["target_reach_time"])

        aggregate = collector.compute_aggregate_metrics()

        mean_time_str = (
            f"{aggregate['mean_reach_time']:.1f}"
            if aggregate['mean_reach_time'] is not None
            else "N/A"
        )
        print(
            f"p_dropout={p_dropout:.1f}: "
            f"reach_prob={aggregate['target_reach_probability']:.2%}, "
            f"mean_time={mean_time_str}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VIRAL NETWORK SIMULATION - DEMO")
    print("=" * 60)

    demo_toy_linear_chain()
    demo_micro_scale()
    demo_macro_scale()
    demo_parameter_sensitivity()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

