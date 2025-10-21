#!/usr/bin/env python
"""Quick test of fast O(E) graph generation.

Verify that the new fast implementation produces correct results
with smaller N for quick validation.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from viral_network.graph_generation_optimized import (
    generate_sbm_graph_fast,
    generate_sbm_graph_optimized_original,
)


def test_fast_graph_small():
    """Test fast graph generation with N=10k (quick test)."""
    print("\n" + "=" * 70)
    print("QUICK TEST: Fast Graph Generation (N=10,000)")
    print("=" * 70)
    
    N = 10_000
    K = 5
    avg_degree = 20
    intra_strength = 0.95
    
    print(f"\nParameters:")
    print(f"  N: {N:,}")
    print(f"  K: {K}")
    print(f"  avg_degree: {avg_degree}")
    print(f"  intra_strength: {intra_strength}")
    
    # Generate graph
    print(f"\nGenerating graph with fast O(E) method...")
    start_time = time.time()
    
    adj, node_to_cluster, metadata = generate_sbm_graph_fast(
        N=N,
        K=K,
        avg_degree=avg_degree,
        intra_strength=intra_strength,
        seed=42,
        n_workers=2,
    )
    
    elapsed = time.time() - start_time
    
    print(f"✓ Graph generated in {elapsed:.2f}s")
    
    # Regression checks
    print(f"\nRegression checks:")
    
    # Check 1: Adjacency shape
    assert adj.shape == (N, N), f"Expected shape ({N}, {N}), got {adj.shape}"
    print(f"  ✓ Adjacency shape: {adj.shape}")
    
    # Check 2: Average degree
    actual_avg_degree = adj.nnz / N
    expected_avg_degree = avg_degree
    tolerance = 0.2 * expected_avg_degree
    
    print(f"  Expected avg_degree: {expected_avg_degree:.2f}")
    print(f"  Actual avg_degree: {actual_avg_degree:.2f}")
    print(f"  Tolerance: ±{tolerance:.2f}")
    
    assert abs(actual_avg_degree - expected_avg_degree) < tolerance, \
        f"Avg degree {actual_avg_degree:.2f} outside tolerance"
    print(f"  ✓ Average degree within tolerance")
    
    # Check 3: Cluster mapping
    assert len(node_to_cluster) == N, f"Expected {N} nodes, got {len(node_to_cluster)}"
    assert len(np.unique(node_to_cluster)) == K, f"Expected {K} clusters"
    print(f"  ✓ Cluster mapping: {len(node_to_cluster)} nodes, {len(np.unique(node_to_cluster))} clusters")
    
    # Check 4: Metadata
    assert "mu" in metadata, "Missing 'mu' in metadata"
    assert "method" in metadata, "Missing 'method' in metadata"
    print(f"  ✓ Metadata: mu={metadata['mu']:.4f}, method={metadata['method']}")
    
    print(f"\n✓ ALL CHECKS PASSED")
    return adj, node_to_cluster, metadata


def test_backward_compatibility():
    """Test that old API still works."""
    print("\n" + "=" * 70)
    print("TEST: Backward Compatibility")
    print("=" * 70)
    
    N = 5_000
    K = 5
    avg_degree = 20
    
    print(f"\nGenerating graph with N={N}, K={K}, avg_degree={avg_degree}")
    
    # This should use the fast version now
    from viral_network.graph_generation_optimized import generate_sbm_graph_optimized
    
    start = time.time()
    adj, node_to_cluster, metadata = generate_sbm_graph_optimized(
        N=N,
        K=K,
        avg_degree=avg_degree,
        seed=42,
        n_workers=2,
    )
    elapsed = time.time() - start
    
    print(f"✓ Graph generated in {elapsed:.2f}s")
    print(f"  Shape: {adj.shape}")
    print(f"  Edges: {adj.nnz}")
    print(f"  Avg degree: {2 * adj.nnz / N:.2f}")
    print(f"  Method: {metadata.get('method', 'unknown')}")
    
    assert adj.shape == (N, N)
    assert len(node_to_cluster) == N
    print(f"\n✓ BACKWARD COMPATIBILITY OK")


def test_comparison_small():
    """Compare fast vs original implementation (small N)."""
    print("\n" + "=" * 70)
    print("TEST: Performance Comparison (N=5,000)")
    print("=" * 70)
    
    N = 5_000
    K = 5
    avg_degree = 20
    
    print(f"\nParameters: N={N}, K={K}, avg_degree={avg_degree}")
    
    # Fast version
    print(f"\nFast version (O(E) with geometric skipping):")
    start = time.time()
    adj_fast, _, meta_fast = generate_sbm_graph_fast(
        N=N, K=K, avg_degree=avg_degree, seed=42, n_workers=2
    )
    time_fast = time.time() - start
    print(f"  Time: {time_fast:.2f}s")
    print(f"  Edges: {adj_fast.nnz}")
    print(f"  Avg degree: {2 * adj_fast.nnz / N:.2f}")
    
    # Original version
    print(f"\nOriginal version (O(N²) baseline):")
    start = time.time()
    adj_orig, _, meta_orig = generate_sbm_graph_optimized_original(
        N=N, K=K, avg_degree=avg_degree, seed=42, n_workers=2
    )
    time_orig = time.time() - start
    print(f"  Time: {time_orig:.2f}s")
    print(f"  Edges: {adj_orig.nnz}")
    print(f"  Avg degree: {2 * adj_orig.nnz / N:.2f}")
    
    # Comparison
    speedup = time_orig / time_fast
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"  Fast: {time_fast:.2f}s")
    print(f"  Original: {time_orig:.2f}s")
    
    # Check similarity
    edge_diff = abs(adj_fast.nnz - adj_orig.nnz)
    print(f"\nEdge count difference: {edge_diff} ({100*edge_diff/adj_orig.nnz:.1f}%)")
    
    print(f"\n✓ COMPARISON COMPLETE")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FAST GRAPH GENERATION - QUICK REGRESSION TESTS")
    print("=" * 70)
    
    try:
        test_fast_graph_small()
        test_backward_compatibility()
        test_comparison_small()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

