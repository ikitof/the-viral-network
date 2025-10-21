"""Performance benchmarking utilities.

Compare original vs optimized implementations.
"""

import time
import numpy as np
from typing import Dict, Tuple
from loguru import logger
import psutil
import os


class PerformanceBenchmark:
    """Benchmark performance of simulations."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.results = {}
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_graph_generation(
        self,
        N: int,
        K: int,
        avg_degree: float,
        use_optimized: bool = True,
    ) -> Dict:
        """Benchmark graph generation.
        
        Args:
            N: Number of nodes
            K: Number of clusters
            avg_degree: Average degree
            use_optimized: Use optimized version
        
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking graph generation: N={N}, K={K}, optimized={use_optimized}")
        
        if use_optimized:
            from viral_network.graph_generation_optimized import generate_sbm_graph_optimized
            gen_func = generate_sbm_graph_optimized
        else:
            from viral_network.graph_generation import generate_sbm_graph
            gen_func = generate_sbm_graph
        
        mem_before = self.get_memory_usage()
        start_time = time.time()
        
        adj, node_to_cluster, metadata = gen_func(
            N=N,
            K=K,
            avg_degree=avg_degree,
            seed=42,
        )
        
        elapsed = time.time() - start_time
        mem_after = self.get_memory_usage()
        mem_used = mem_after - mem_before
        
        result = {
            "N": N,
            "K": K,
            "avg_degree": avg_degree,
            "time_seconds": elapsed,
            "memory_mb": mem_used,
            "edges": metadata["num_edges"],
            "avg_degree_actual": metadata["avg_degree_actual"],
            "optimized": use_optimized,
        }
        
        logger.info(
            f"Graph generation: {elapsed:.2f}s, {mem_used:.1f}MB, "
            f"{metadata['num_edges']} edges"
        )
        
        return result
    
    def benchmark_simulation(
        self,
        adj,
        node_to_cluster: np.ndarray,
        config,
        use_optimized: bool = True,
        num_runs: int = 1,
    ) -> Dict:
        """Benchmark simulation.
        
        Args:
            adj: Adjacency matrix
            node_to_cluster: Cluster assignment
            config: Configuration
            use_optimized: Use optimized version
            num_runs: Number of runs to average
        
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking simulation: N={adj.shape[0]}, optimized={use_optimized}")
        
        if use_optimized:
            from viral_network.simulate_micro_optimized import SimulatorMicroOptimized
            SimClass = SimulatorMicroOptimized
        else:
            from viral_network.simulate_micro import SimulatorMicro
            SimClass = SimulatorMicro
        
        times = []
        
        for run in range(num_runs):
            simulator = SimClass(adj, node_to_cluster, config)
            
            # Select random seeds and target
            initial_seeds = [np.random.randint(0, adj.shape[0])]
            target_node = np.random.randint(0, adj.shape[0])
            
            mem_before = self.get_memory_usage()
            start_time = time.time()
            
            states, metrics = simulator.run(initial_seeds, target_node)
            
            elapsed = time.time() - start_time
            mem_after = self.get_memory_usage()
            
            times.append(elapsed)
            
            logger.info(f"Run {run+1}/{num_runs}: {elapsed:.2f}s, memory delta: {mem_after - mem_before:.1f}MB")
        
        result = {
            "N": adj.shape[0],
            "K": len(np.unique(node_to_cluster)),
            "time_mean_seconds": np.mean(times),
            "time_std_seconds": np.std(times),
            "time_min_seconds": np.min(times),
            "time_max_seconds": np.max(times),
            "num_runs": num_runs,
            "optimized": use_optimized,
        }
        
        logger.info(
            f"Simulation: {result['time_mean_seconds']:.2f}s ± {result['time_std_seconds']:.2f}s"
        )
        
        return result
    
    def compare_implementations(
        self,
        N: int,
        K: int,
        avg_degree: float,
        config,
    ) -> Dict:
        """Compare original vs optimized implementations.
        
        Args:
            N: Number of nodes
            K: Number of clusters
            avg_degree: Average degree
            config: Configuration
        
        Returns:
            Comparison results
        """
        logger.info(f"Comparing implementations: N={N}, K={K}")
        
        # Benchmark graph generation
        logger.info("=== Graph Generation ===")
        result_gen_orig = self.benchmark_graph_generation(N, K, avg_degree, use_optimized=False)
        result_gen_opt = self.benchmark_graph_generation(N, K, avg_degree, use_optimized=True)
        
        speedup_gen = result_gen_orig["time_seconds"] / result_gen_opt["time_seconds"]
        logger.info(f"Graph generation speedup: {speedup_gen:.1f}x")
        
        # Benchmark simulation
        logger.info("=== Simulation ===")
        from viral_network.graph_generation_optimized import generate_sbm_graph_optimized
        adj, node_to_cluster, _ = generate_sbm_graph_optimized(N, K, avg_degree, seed=42)
        
        result_sim_orig = self.benchmark_simulation(adj, node_to_cluster, config, use_optimized=False)
        result_sim_opt = self.benchmark_simulation(adj, node_to_cluster, config, use_optimized=True)
        
        speedup_sim = result_sim_orig["time_mean_seconds"] / result_sim_opt["time_mean_seconds"]
        logger.info(f"Simulation speedup: {speedup_sim:.1f}x")
        
        return {
            "graph_generation": {
                "original": result_gen_orig,
                "optimized": result_gen_opt,
                "speedup": speedup_gen,
            },
            "simulation": {
                "original": result_sim_orig,
                "optimized": result_sim_opt,
                "speedup": speedup_sim,
            },
        }

