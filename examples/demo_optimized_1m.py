#!/usr/bin/env python
"""Demo: Optimized simulation for N=1 million nodes.

This demonstrates the optimized implementation for large-scale simulations.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_network.config import Config
from viral_network.graph_generation_optimized import generate_sbm_graph_optimized
from viral_network.simulate_micro_optimized import SimulatorMicroOptimized


def main():
    """Run optimized simulation for N=1M."""
    
    print("\n" + "=" * 70)
    print("OPTIMIZED SIMULATION DEMO - N=1,000,000")
    print("=" * 70)
    
    # Configuration
    N = 1_000_000
    K = 200  # 200 clusters (countries)
    avg_degree = 100
    
    print(f"\nConfiguration:")
    print(f"  Nodes (N): {N:,}")
    print(f"  Clusters (K): {K}")
    print(f"  Average degree: {avg_degree}")
    print(f"  Fanout: 2")
    print(f"  Dropout probability: 0.15")
    
    # Step 1: Generate graph
    print(f"\n{'='*70}")
    print("STEP 1: Generate Optimized SBM Graph")
    print(f"{'='*70}")
    
    config = Config(
        mode="micro",
        N=N,
        K=K,
        fanout=2,
        p_dropout=0.15,
        max_steps=20,
        seed=42,
    )
    
    start_time = time.time()
    print(f"\nGenerating graph with {8} workers...")
    
    adj, node_to_cluster, metadata = generate_sbm_graph_optimized(
        N=N,
        K=K,
        avg_degree=avg_degree,
        seed=42,
        n_workers=8,
    )
    
    gen_time = time.time() - start_time
    
    print(f"\n✓ Graph generated in {gen_time:.2f}s")
    print(f"  Nodes: {adj.shape[0]:,}")
    print(f"  Edges: {metadata['num_edges']:,}")
    print(f"  Avg degree: {metadata['avg_degree_actual']:.2f}")
    print(f"  Clusters: {len(np.unique(node_to_cluster))}")
    print(f"  Sparsity: {metadata['num_edges'] / (N * N) * 100:.4f}%")
    
    # Step 2: Initialize simulator
    print(f"\n{'='*70}")
    print("STEP 2: Initialize Optimized Simulator")
    print(f"{'='*70}")
    
    start_time = time.time()
    simulator = SimulatorMicroOptimized(adj, node_to_cluster, config)
    init_time = time.time() - start_time
    
    print(f"\n✓ Simulator initialized in {init_time:.2f}s")
    
    # Step 3: Run simulation
    print(f"\n{'='*70}")
    print("STEP 3: Run Simulation")
    print(f"{'='*70}")
    
    # Select seeds and target
    initial_seeds = [0]  # Start from node 0
    target_cluster = K - 1  # Target last cluster
    target_nodes = np.where(node_to_cluster == target_cluster)[0]
    target_node = target_nodes[0]  # First node in target cluster
    
    print(f"\nSimulation parameters:")
    print(f"  Initial seeds: {initial_seeds}")
    print(f"  Seed cluster: {node_to_cluster[initial_seeds[0]]}")
    print(f"  Target node: {target_node}")
    print(f"  Target cluster: {target_cluster}")
    print(f"  Distance: {target_cluster} clusters")
    
    print(f"\nRunning simulation...")
    start_time = time.time()
    
    states, metrics = simulator.run(initial_seeds, target_node)
    
    sim_time = time.time() - start_time
    
    print(f"\n✓ Simulation completed in {sim_time:.2f}s")
    
    # Step 4: Display results
    print(f"\n{'='*70}")
    print("STEP 4: Results")
    print(f"{'='*70}")
    
    print(f"\nSimulation Results:")
    print(f"  Target reached: {metrics['target_reached']}")
    print(f"  Target reach time: {metrics['target_reach_time']}")
    print(f"  Final infected count: {metrics['final_infected_count']:,}")
    print(f"  Attack rate: {metrics['final_infected_count'] / N * 100:.2f}%")
    print(f"  Max concurrent infected: {metrics['max_concurrent_I']:,}")
    print(f"  Simulation steps: {len(states)}")
    
    # Step 5: Performance summary
    print(f"\n{'='*70}")
    print("STEP 5: Performance Summary")
    print(f"{'='*70}")
    
    total_time = gen_time + init_time + sim_time
    
    print(f"\nTiming Breakdown:")
    print(f"  Graph generation: {gen_time:.2f}s ({gen_time/total_time*100:.1f}%)")
    print(f"  Simulator init: {init_time:.2f}s ({init_time/total_time*100:.1f}%)")
    print(f"  Simulation: {sim_time:.2f}s ({sim_time/total_time*100:.1f}%)")
    print(f"  Total: {total_time:.2f}s")
    
    print(f"\nThroughput:")
    print(f"  Nodes processed: {N:,}")
    print(f"  Edges processed: {metadata['num_edges']:,}")
    print(f"  Nodes/second: {N / total_time:,.0f}")
    print(f"  Edges/second: {metadata['num_edges'] / total_time:,.0f}")
    
    # Step 6: Scaling estimate
    print(f"\n{'='*70}")
    print("STEP 6: Scaling Estimate to N=1 Billion")
    print(f"{'='*70}")
    
    # Estimate based on current performance
    scale_factor = 1_000_000_000 / N
    estimated_gen_time = gen_time * scale_factor
    estimated_sim_time = sim_time * scale_factor
    estimated_total_time = estimated_gen_time + estimated_sim_time
    
    print(f"\nEstimated Performance for N=1,000,000,000:")
    print(f"  Scale factor: {scale_factor:.0f}x")
    print(f"  Graph generation: ~{estimated_gen_time/3600:.1f} hours")
    print(f"  Simulation: ~{estimated_sim_time/3600:.1f} hours")
    print(f"  Total: ~{estimated_total_time/3600:.1f} hours")
    
    print(f"\nRecommendation:")
    print(f"  For N=1B, use MACRO-SCALE simulation instead (100-1000x faster)")
    print(f"  Macro-scale would complete in ~{estimated_total_time/3600/100:.2f} minutes")
    
    # Step 7: Infection timeline
    print(f"\n{'='*70}")
    print("STEP 7: Infection Timeline")
    print(f"{'='*70}")
    
    print(f"\nInfection progression:")
    print(f"{'Time':>6} {'Infected':>12} {'Recovered':>12} {'Susceptible':>12}")
    print("-" * 45)
    
    for i in range(0, len(states), max(1, len(states) // 10)):
        state = states[i]
        print(
            f"{state.t:>6} {state.I_count:>12,} {state.R_count:>12,} "
            f"{len(state.S):>12,}"
        )
    
    if len(states) > 0:
        state = states[-1]
        print(
            f"{state.t:>6} {state.I_count:>12,} {state.R_count:>12,} "
            f"{len(state.S):>12,}"
        )
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

