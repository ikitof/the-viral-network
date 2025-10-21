#!/usr/bin/env python
"""Benchmark optimized implementations against original.

This script compares performance of:
- Original vs optimized graph generation
- Original vs optimized simulation
- Scaling to N=1 billion
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_network.config import Config
from viral_network.benchmarks import PerformanceBenchmark


def benchmark_small_scale():
    """Benchmark on small scale (N=10k)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Small Scale (N=10,000)")
    print("=" * 70)
    
    config = Config(N=10000, K=10, fanout=2, p_dropout=0.15, seed=42)
    benchmark = PerformanceBenchmark()
    
    results = benchmark.compare_implementations(
        N=10000,
        K=10,
        avg_degree=50,
        config=config,
    )
    
    print("\nGraph Generation:")
    print(f"  Original: {results['graph_generation']['original']['time_seconds']:.2f}s")
    print(f"  Optimized: {results['graph_generation']['optimized']['time_seconds']:.2f}s")
    print(f"  Speedup: {results['graph_generation']['speedup']:.1f}x")
    
    print("\nSimulation:")
    print(f"  Original: {results['simulation']['original']['time_mean_seconds']:.2f}s")
    print(f"  Optimized: {results['simulation']['optimized']['time_mean_seconds']:.2f}s")
    print(f"  Speedup: {results['simulation']['speedup']:.1f}x")


def benchmark_medium_scale():
    """Benchmark on medium scale (N=100k)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Medium Scale (N=100,000)")
    print("=" * 70)
    
    config = Config(N=100000, K=20, fanout=2, p_dropout=0.15, seed=42)
    benchmark = PerformanceBenchmark()
    
    results = benchmark.compare_implementations(
        N=100000,
        K=20,
        avg_degree=100,
        config=config,
    )
    
    print("\nGraph Generation:")
    print(f"  Original: {results['graph_generation']['original']['time_seconds']:.2f}s")
    print(f"  Optimized: {results['graph_generation']['optimized']['time_seconds']:.2f}s")
    print(f"  Speedup: {results['graph_generation']['speedup']:.1f}x")
    
    print("\nSimulation:")
    print(f"  Original: {results['simulation']['original']['time_mean_seconds']:.2f}s")
    print(f"  Optimized: {results['simulation']['optimized']['time_mean_seconds']:.2f}s")
    print(f"  Speedup: {results['simulation']['speedup']:.1f}x")


def benchmark_large_scale():
    """Benchmark on large scale (N=1M)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Large Scale (N=1,000,000)")
    print("=" * 70)
    
    config = Config(N=1000000, K=50, fanout=2, p_dropout=0.15, seed=42)
    benchmark = PerformanceBenchmark()
    
    print("\nOptimized Graph Generation:")
    result = benchmark.benchmark_graph_generation(
        N=1000000,
        K=50,
        avg_degree=100,
        use_optimized=True,
    )
    print(f"  Time: {result['time_seconds']:.2f}s")
    print(f"  Memory: {result['memory_mb']:.1f}MB")
    print(f"  Edges: {result['edges']}")


def benchmark_billion_scale():
    """Benchmark on billion scale (N=1B) - optimized only."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Billion Scale (N=1,000,000,000) - OPTIMIZED ONLY")
    print("=" * 70)
    
    print("\nNote: This is a theoretical benchmark showing what's possible")
    print("with optimized code. Actual execution requires significant RAM.")
    
    # Estimate based on scaling
    N = 1_000_000_000
    K = 200
    avg_degree = 100
    
    # Estimate time based on smaller runs
    # Assuming O(N * avg_degree) complexity
    # For N=1M with 100 avg_degree: ~10 seconds
    # For N=1B: ~10,000 seconds = ~2.8 hours
    
    estimated_time = 10000  # seconds
    estimated_memory = 500_000  # MB = 500 GB (for full graph)
    
    print(f"\nEstimated Performance (N={N:,}):")
    print(f"  Graph generation time: ~{estimated_time/3600:.1f} hours")
    print(f"  Memory for full graph: ~{estimated_memory/1024:.0f} GB")
    print(f"  Simulation time: ~{estimated_time/3600:.1f} hours")
    
    print("\nOptimization Strategies for N=1B:")
    print("  1. Use streaming graph generation (avoid loading full graph)")
    print("  2. Use macro-scale simulation instead (100-1000x faster)")
    print("  3. Use GPU acceleration (CUDA) for 10-100x speedup")
    print("  4. Distribute across multiple machines")


def benchmark_scaling():
    """Benchmark scaling with different N values."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Scaling Analysis")
    print("=" * 70)
    
    benchmark = PerformanceBenchmark()
    
    N_values = [1000, 10000, 100000, 1000000]
    
    print("\nOptimized Graph Generation Scaling:")
    print(f"{'N':>12} {'Time (s)':>12} {'Memory (MB)':>15} {'Edges':>12}")
    print("-" * 52)
    
    for N in N_values:
        K = max(5, N // 10000)
        result = benchmark.benchmark_graph_generation(
            N=N,
            K=K,
            avg_degree=50,
            use_optimized=True,
        )
        print(
            f"{N:>12,} {result['time_seconds']:>12.2f} "
            f"{result['memory_mb']:>15.1f} {result['edges']:>12,}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VIRAL NETWORK SIMULATION - PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    try:
        benchmark_small_scale()
        benchmark_medium_scale()
        benchmark_large_scale()
        benchmark_billion_scale()
        benchmark_scaling()
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()

