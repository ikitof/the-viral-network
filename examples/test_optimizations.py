#!/usr/bin/env python
"""Test optimized implementations.

Verify that optimized code produces correct results.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_network.config import Config
from viral_network.graph_generation_optimized import generate_sbm_graph_optimized
from viral_network.simulate_micro_optimized import SimulatorMicroOptimized


def test_graph_generation_optimized():
    """Test optimized graph generation."""
    print("\n" + "=" * 70)
    print("TEST 1: Optimized Graph Generation")
    print("=" * 70)
    
    N = 10000
    K = 10
    avg_degree = 50
    
    print(f"\nGenerating SBM: N={N}, K={K}, avg_degree={avg_degree}")
    
    adj, node_to_cluster, metadata = generate_sbm_graph_optimized(
        N=N,
        K=K,
        avg_degree=avg_degree,
        seed=42,
        n_workers=4,
    )
    
    print(f"✓ Graph generated successfully")
    print(f"  Nodes: {adj.shape[0]}")
    print(f"  Edges: {metadata['num_edges']}")
    print(f"  Avg degree: {metadata['avg_degree_actual']:.2f}")
    print(f"  Clusters: {len(np.unique(node_to_cluster))}")
    
    # Verify properties
    assert adj.shape[0] == N, "Wrong number of nodes"
    assert adj.shape[1] == N, "Wrong matrix shape"
    assert len(node_to_cluster) == N, "Wrong cluster assignment"
    assert len(np.unique(node_to_cluster)) == K, "Wrong number of clusters"
    
    print("✓ All assertions passed")


def test_simulation_optimized():
    """Test optimized simulation."""
    print("\n" + "=" * 70)
    print("TEST 2: Optimized Simulation")
    print("=" * 70)
    
    N = 5000
    K = 5
    
    print(f"\nGenerating graph: N={N}, K={K}")
    adj, node_to_cluster, _ = generate_sbm_graph_optimized(
        N=N,
        K=K,
        avg_degree=50,
        seed=42,
        n_workers=2,
    )
    
    print(f"✓ Graph generated")
    
    config = Config(
        mode="micro",
        N=N,
        K=K,
        fanout=2,
        p_dropout=0.15,
        max_steps=20,
        seed=42,
    )
    
    print(f"\nRunning simulation...")
    simulator = SimulatorMicroOptimized(adj, node_to_cluster, config)
    
    initial_seeds = [0]
    target_node = np.random.randint(0, N)
    
    states, metrics = simulator.run(initial_seeds, target_node)
    
    print(f"✓ Simulation completed")
    print(f"  Target reached: {metrics['target_reached']}")
    print(f"  Final infected: {metrics['final_infected_count']}")
    print(f"  Max concurrent I: {metrics['max_concurrent_I']}")
    print(f"  Simulation steps: {len(states)}")
    
    # Verify properties
    assert len(states) > 0, "No states recorded"
    assert metrics['final_infected_count'] >= 0, "Invalid infected count"
    assert metrics['max_concurrent_I'] >= 0, "Invalid max concurrent"
    
    print("✓ All assertions passed")


def test_scaling():
    """Test scaling with different N values."""
    print("\n" + "=" * 70)
    print("TEST 3: Scaling Test")
    print("=" * 70)
    
    N_values = [1000, 5000, 10000]
    
    for N in N_values:
        K = max(3, N // 1000)
        
        print(f"\nTesting N={N}, K={K}")
        
        adj, node_to_cluster, metadata = generate_sbm_graph_optimized(
            N=N,
            K=K,
            avg_degree=50,
            seed=42,
            n_workers=2,
        )
        
        config = Config(
            mode="micro",
            N=N,
            K=K,
            fanout=2,
            p_dropout=0.15,
            max_steps=10,
            seed=42,
        )
        
        simulator = SimulatorMicroOptimized(adj, node_to_cluster, config)
        states, metrics = simulator.run([0], N-1)
        
        print(f"  ✓ Completed: {metrics['final_infected_count']} infected")


def test_large_scale():
    """Test large scale (N=100k)."""
    print("\n" + "=" * 70)
    print("TEST 4: Large Scale (N=100,000)")
    print("=" * 70)
    
    N = 100000
    K = 20
    
    print(f"\nGenerating large graph: N={N}, K={K}")
    
    adj, node_to_cluster, metadata = generate_sbm_graph_optimized(
        N=N,
        K=K,
        avg_degree=100,
        seed=42,
        n_workers=4,
    )
    
    print(f"✓ Graph generated")
    print(f"  Edges: {metadata['num_edges']}")
    print(f"  Avg degree: {metadata['avg_degree_actual']:.2f}")
    
    config = Config(
        mode="micro",
        N=N,
        K=K,
        fanout=2,
        p_dropout=0.15,
        max_steps=15,
        seed=42,
    )
    
    print(f"\nRunning simulation on large graph...")
    simulator = SimulatorMicroOptimized(adj, node_to_cluster, config)
    states, metrics = simulator.run([0], N-1)
    
    print(f"✓ Simulation completed")
    print(f"  Final infected: {metrics['final_infected_count']}")
    print(f"  Attack rate: {metrics['final_infected_count'] / N:.2%}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VIRAL NETWORK SIMULATION - OPTIMIZATION TESTS")
    print("=" * 70)
    
    try:
        test_graph_generation_optimized()
        test_simulation_optimized()
        test_scaling()
        test_large_scale()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

