"""Optimized graph generation with multiprocessing and streaming.

Supports generation of massive graphs (N up to 1 billion) using:
- Multiprocessing for parallel edge generation
- Streaming to avoid loading entire graph in memory
- Efficient sparse matrix construction
- Block-wise geometric skipping with Numba JIT compilation
"""

import numpy as np
from scipy import sparse
from typing import Tuple, List, Optional
from loguru import logger
import multiprocessing as mp
from functools import partial

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ============================================================================
# True O(E) block-wise geometric skipping (Batagelj-Brandes)
# ============================================================================

# ---- Probabilities consistent with avg_degree & mixing ----

def _compute_block_probs(N: int, sizes: np.ndarray, avg_degree: float, intra_strength: float):
    mu = float(max(0.0, min(1.0, 1.0 - intra_strength)))  # fraction of edges outside cluster
    K = len(sizes)
    p_same = np.empty(K, dtype=np.float64)
    p_cross = np.empty(K, dtype=np.float64)
    for k, n_k in enumerate(sizes):
        n_k = int(n_k)
        p_same[k] = ((1.0 - mu) * avg_degree) / max(1, (n_k - 1))
        p_cross[k] = (mu * avg_degree) / max(1, (N - n_k))
    return p_same, p_cross, mu

# ---- Numba kernels: true geometric skipping per block ----
if HAS_NUMBA:
    @njit
    def _geom_skip_u(log1mp):
        u = np.random.random()
        return int(np.log(1.0 - u) / log1mp)

    @njit
    def _block_edges_same_numba(n, p, start_idx):
        m = n * (n - 1) // 2
        if p <= 0.0 or m == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        log1mp = np.log(1.0 - p)
        cap = max(1, int(1.1 * p * m) + 1)
        rows = np.empty(cap, np.int64)
        cols = np.empty(cap, np.int64)
        e = 0
        idx = -1
        while True:
            skip = _geom_skip_u(log1mp)
            idx += skip + 1
            if idx >= m:
                break
            # invert triangular index → (i,j)
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                cum = mid * (2 * n - mid - 1) // 2
                if idx < cum:
                    hi = mid
                else:
                    lo = mid + 1
            i = lo - 1
            cum_prev = i * (2 * n - i - 1) // 2 if i >= 0 else 0
            pos = idx - cum_prev
            i = max(0, i)
            if i >= n - 1:
                break
            j = i + 1 + pos
            if e == rows.size:
                new_cap = int(rows.size * 1.5) + 1
                r2 = np.empty(new_cap, np.int64); c2 = np.empty(new_cap, np.int64)
                r2[:e] = rows[:e]; c2[:e] = cols[:e]
                rows = r2; cols = c2
            rows[e] = start_idx + i
            cols[e] = start_idx + j
            e += 1
        return rows[:e], cols[:e]

    @njit
    def _block_edges_cross_numba(na, nb, p, start_a, start_b):
        m = na * nb
        if p <= 0.0 or m == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        log1mp = np.log(1.0 - p)
        cap = max(1, int(1.1 * p * m) + 1)
        rows = np.empty(cap, np.int64)
        cols = np.empty(cap, np.int64)
        e = 0
        idx = -1
        while True:
            skip = _geom_skip_u(log1mp)
            idx += skip + 1
            if idx >= m:
                break
            i_rel = idx // nb
            j_rel = idx % nb
            if e == rows.size:
                new_cap = int(rows.size * 1.5) + 1
                r2 = np.empty(new_cap, np.int64); c2 = np.empty(new_cap, np.int64)
                r2[:e] = rows[:e]; c2[:e] = cols[:e]
                rows = r2; cols = c2
            rows[e] = start_a + i_rel
            cols[e] = start_b + j_rel
            e += 1
        return rows[:e], cols[:e]
else:
    # Python fallbacks (slower) if Numba unavailable
    def _block_edges_same_numba(n, p, start_idx):
        rows = []
        cols = []
        if p <= 0.0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        # naive geometric skipping emulation
        m = n * (n - 1) // 2
        log1mp = np.log(1.0 - p)
        idx = -1
        while True:
            u = np.random.random()
            skip = int(np.log(1.0 - u) / log1mp)
            idx += skip + 1
            if idx >= m:
                break
            # invert triangular index
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                cum = mid * (2 * n - mid - 1) // 2
                if idx < cum:
                    hi = mid
                else:
                    lo = mid + 1
            i = lo - 1
            cum_prev = i * (2 * n - i - 1) // 2 if i >= 0 else 0
            pos = idx - cum_prev
            i = max(0, i)
            if i >= n - 1:
                break
            j = i + 1 + pos
            rows.append(start_idx + i)
            cols.append(start_idx + j)
        return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)

    def _block_edges_cross_numba(na, nb, p, start_a, start_b):
        rows = []
        cols = []
        if p <= 0.0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        m = na * nb
        log1mp = np.log(1.0 - p)
        idx = -1
        while True:
            u = np.random.random()
            skip = int(np.log(1.0 - u) / log1mp)
            idx += skip + 1
            if idx >= m:
                break
            i_rel = idx // nb
            j_rel = idx % nb
            rows.append(start_a + i_rel)
            cols.append(start_b + j_rel)
        return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


# ----------------------------------------------------------------------------
# Parallel block edge generation (jobs per cluster block)
# ----------------------------------------------------------------------------

def _run_block_job(args):
    kind, a, b, sizes, starts, p_same, p_cross, seed = args
    # per-job deterministic seed mixing to avoid collisions across processes
    s = (seed or 0) ^ (a * 1315423911) ^ (b * 2654435761)
    s &= 0xFFFFFFFF
    np.random.seed(int(s))
    if kind == 'same':
        r, c = _block_edges_same_numba(int(sizes[a]), min(1.0, float(p_same[a])), int(starts[a]))
    else:
        p_ab = 0.5 * (float(p_cross[a]) + float(p_cross[b]))
        r, c = _block_edges_cross_numba(int(sizes[a]), int(sizes[b]),
                                        min(1.0, p_ab), int(starts[a]), int(starts[b]))
    return r, c


def _generate_edges_blocks_parallel(N: int,
                                    sizes: np.ndarray,
                                    p_same: np.ndarray,
                                    p_cross: np.ndarray,
                                    n_workers: Optional[int],
                                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    K = len(sizes)
    starts = np.zeros(K, dtype=np.int64)
    for k in range(1, K):
        starts[k] = starts[k - 1] + int(sizes[k - 1])

    jobs = []
    for a in range(K):
        jobs.append(('same', a, a, sizes, starts, p_same, p_cross, seed))
        for b in range(a + 1, K):
            jobs.append(('cross', a, b, sizes, starts, p_same, p_cross, seed))

    logger.info(f"Submitting {len(jobs)} block jobs across {K} clusters using {n_workers or 1} workers")

    if n_workers and n_workers > 1:
        out = []
        with mp.Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_block_job, jobs, chunksize=4), start=1):
                out.append(res)
                if i % max(1, len(jobs)//10) == 0 or i == len(jobs):
                    logger.info(f"Blocks completed: {100.0 * i/len(jobs):.0f}% ({i}/{len(jobs)})")
    else:
        out = []
        for i, j in enumerate(jobs, start=1):
            out.append(_run_block_job(j))
            if i % max(1, len(jobs)//10) == 0 or i == len(jobs):
                logger.info(f"Blocks completed: {100.0 * i/len(jobs):.0f}% ({i}/{len(jobs)})")

    rows = np.concatenate([r for (r, _) in out]) if out else np.empty(0, np.int64)
    cols = np.concatenate([c for (_, c) in out]) if out else np.empty(0, np.int64)
    return rows, cols


def generate_sbm_graph_optimized(
    N: int,
    K: int,
    avg_degree: float,
    cluster_sizes: str = "uniform",
    intra_strength: float = 0.95,
    seed: Optional[int] = None,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000,
) -> Tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Generate SBM graph with optimizations for large N.

    Args:
        N: Number of nodes
        K: Number of clusters
        avg_degree: Average degree per node
        cluster_sizes: "uniform", "powerlaw", or "custom"
        intra_strength: Intra-cluster connection strength (0-1)
        seed: Random seed
        n_workers: Number of worker processes
        chunk_size: Size of chunks for parallel processing

    Returns:
        Tuple of (adjacency_matrix, node_to_cluster, metadata)
    """
    # Delegate to the O(E) fast implementation for backward compatibility
    return generate_sbm_graph_fast(
        N=N,
        K=K,
        avg_degree=avg_degree,
        cluster_sizes=cluster_sizes,
        intra_strength=intra_strength,
        seed=seed,
        n_workers=n_workers,
        chunk_size=chunk_size,
        block_size=1000,
    )


def generate_sbm_graph_fast(
    N: int,
    K: int,
    avg_degree: float,
    cluster_sizes: str = "uniform",
    intra_strength: float = 0.95,
    seed: Optional[int] = None,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000,
    block_size: int = 1000,
    inter_floor: float = 0.0,  # accepted for API compatibility; not used
) -> Tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Generate SBM graph with O(E) complexity using block-wise geometric skipping.

    This is a faster alternative to generate_sbm_graph_optimized using:
    - Block-wise processing for cache efficiency
    - Geometric skipping to avoid checking non-edges
    - Numba JIT compilation for tight loops

    Args:
        N: Number of nodes
        K: Number of clusters
        avg_degree: Average degree per node
        cluster_sizes: "uniform", "powerlaw", or "custom"
        intra_strength: Intra-cluster connection strength (0-1)
        seed: Random seed
        n_workers: Number of worker processes
        chunk_size: Size of chunks for parallel processing
        block_size: Size of blocks for geometric skipping

    Returns:
        Tuple of (adjacency_matrix, node_to_cluster, metadata)
    """
    if seed is not None:
        np.random.seed(seed)

    n_workers = n_workers or mp.cpu_count()

    logger.info(
        f"Generating fast SBM (O(E)): N={N}, K={K}, avg_degree={avg_degree}, "
        f"workers={n_workers}, block_size={block_size}"
    )

    # cluster sizes
    if cluster_sizes == "uniform":
        sizes = np.full(K, N // K, dtype=int)
        sizes[:N % K] += 1
    elif cluster_sizes == "powerlaw":
        raw = np.random.zipf(1.5, K).astype(np.float64)
        sizes = np.floor(raw / raw.sum() * N).astype(int)
        diff = N - sizes.sum()
        for i in range(abs(diff)):
            sizes[i % K] += 1 if diff > 0 else -1
        if sizes.min() < 1:
            raise ValueError("Powerlaw produced empty cluster; adjust parameters.")
    else:
        sizes = np.ones(K, dtype=int) * (N // K)
        sizes[:N % K] += 1

    # Create node-to-cluster mapping
    node_to_cluster = np.repeat(np.arange(K), sizes)

    # Probabilities per block from avg_degree & mixing
    p_same, p_cross, mu = _compute_block_probs(N, sizes, avg_degree, intra_strength)
    logger.info(f"Mixing fraction (mu): {mu:.4f}")

    # O(E) edges by block-wise geometric skipping
    rows, cols = _generate_edges_blocks_parallel(
        N=N, sizes=sizes, p_same=p_same, p_cross=p_cross,
        n_workers=n_workers, seed=seed or 0
    )

    # build CSR (symmetrize, zero diag)
    coo = sparse.coo_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(N, N))
    coo = coo + coo.T
    coo.setdiag(0)
    coo.eliminate_zeros()
    adj = coo.tocsr()

    num_edges = adj.nnz // 2
    avg_degree_actual = adj.nnz / N
    metadata = {
        "N": int(N),
        "K": int(K),
        "avg_degree_target": float(avg_degree),
        "intra_strength": float(intra_strength),
        "num_edges": int(num_edges),
        "avg_degree_actual": float(avg_degree_actual),
        "cluster_sizes": sizes.tolist(),
        "p_same": p_same.tolist(),
        "p_cross": p_cross.tolist(),
        "mu": float(mu),
        "method": "fast_O(E)_block_geometric",
    }
    logger.info(f"Generated graph: {num_edges} edges, avg_degree={avg_degree_actual:.2f}")

    return adj, node_to_cluster, metadata








# Alias for backward compatibility: use fast version by default
generate_sbm_graph_optimized_original = generate_sbm_graph_optimized
generate_sbm_graph_optimized = generate_sbm_graph_fast


def generate_sbm_graph_streaming(
    N: int,
    K: int,
    avg_degree: float,
    output_file: str,
    intra_strength: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Generate SBM graph with streaming to disk.

    For very large N, stream edges to disk to avoid memory issues.

    Args:
        N: Number of nodes
        K: Number of clusters
        avg_degree: Average degree per node
        output_file: File to stream edges to
        intra_strength: Intra-cluster connection strength
        seed: Random seed

    Returns:
        Tuple of (adjacency_matrix, node_to_cluster, metadata)
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Generating SBM with streaming: N={N}, K={K}, output={output_file}")

    # Generate cluster sizes
    sizes = np.full(K, N // K, dtype=int)
    sizes[:N % K] += 1

    # Create node-to-cluster mapping
    node_to_cluster = np.repeat(np.arange(K), sizes)

    # Compute mixing matrix
    p_intra = intra_strength
    p_inter = (avg_degree * (1 - intra_strength)) / (N - 1)

    # Stream edges to file
    edge_count = 0
    with open(output_file, 'w') as f:
        for i in range(N):
            cluster_i = node_to_cluster[i]

            for j in range(i + 1, N):
                cluster_j = node_to_cluster[j]
                p = p_intra if cluster_i == cluster_j else p_inter

                if np.random.random() < p:
                    f.write(f"{i} {j}\n")
                    edge_count += 1

            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i+1}/{N} nodes, {edge_count} edges so far")

    # Load edges from file into sparse matrix
    logger.info(f"Loading {edge_count} edges from file")
    rows, cols = [], []
    with open(output_file, 'r') as f:
        for line in f:
            i, j = map(int, line.strip().split())
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    metadata = {
        "num_edges": edge_count,
        "avg_degree_actual": 2 * edge_count / N,
        "cluster_sizes": sizes,
        "p_intra": p_intra,
        "p_inter": p_inter,
    }

    return adj, node_to_cluster, metadata

