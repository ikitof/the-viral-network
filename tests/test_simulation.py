"""Tests for viral network simulations."""

import pytest
import numpy as np
from scipy import sparse

from viral_network.config import Config, InitialSeedsConfig
from viral_network.graph_generation import generate_sbm_graph, sample_neighbors
from viral_network.simulate_micro import SimulatorMicro
from viral_network.simulate_macro import SimulatorMacro
from viral_network.targets import TargetSelector


class TestGraphGeneration:
    """Test graph generation."""

    def test_sbm_generation_basic(self):
        """Test basic SBM generation."""
        N, K = 100, 3
        adj, node_to_cluster, metadata = generate_sbm_graph(
            N=N, K=K, avg_degree=10, seed=42
        )

        assert adj.shape == (N, N)
        assert len(node_to_cluster) == N
        assert len(np.unique(node_to_cluster)) == K
        assert metadata["N"] == N
        assert metadata["K"] == K

    def test_sbm_generation_sparse(self):
        """Test that SBM generates sparse matrix."""
        N, K = 1000, 10
        adj, _, _ = generate_sbm_graph(N=N, K=K, avg_degree=20, seed=42)

        # Check sparsity (with high intra-cluster prob, density can be higher)
        density = adj.nnz / (N * N)
        assert density < 0.5  # Should be reasonably sparse

    def test_sbm_cluster_assignment(self):
        """Test that nodes are correctly assigned to clusters."""
        N, K = 100, 5
        _, node_to_cluster, _ = generate_sbm_graph(N=N, K=K, avg_degree=10, seed=42)

        # All clusters should be represented
        unique_clusters = np.unique(node_to_cluster)
        assert len(unique_clusters) == K

    def test_sample_neighbors(self):
        """Test neighbor sampling."""
        # Create simple graph: 0-1-2-3-4
        rows = [0, 1, 1, 2, 2, 3, 3, 4]
        cols = [1, 0, 2, 1, 3, 2, 4, 3]
        adj = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(5, 5)
        )

        rng = np.random.RandomState(42)

        # Node 2 has neighbors [1, 3]
        neighbors = sample_neighbors(adj, 2, k=2, rng=rng)
        assert len(neighbors) == 2
        assert set(neighbors) == {1, 3}

        # Sample 1 neighbor
        neighbors = sample_neighbors(adj, 2, k=1, rng=rng)
        assert len(neighbors) == 1
        assert neighbors[0] in {1, 3}


class TestMicroSimulation:
    """Test micro-scale simulation."""

    def test_linear_chain_toy(self):
        """Test toy case: linear chain of 6 nodes."""
        # Create linear chain: 0-1-2-3-4-5
        rows = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
        cols = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
        adj = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(6, 6)
        )

        # Cluster assignment: 0-2 in cluster 0, 3-5 in cluster 1
        node_to_cluster = np.array([0, 0, 0, 1, 1, 1])

        config = Config(
            mode="micro",
            N=100,  # Minimum N is 1, but we use 100 for consistency
            K=2,
            fanout=1,
            p_dropout=0.0,
            max_steps=10,
            seed=42,
        )

        simulator = SimulatorMicro(adj, node_to_cluster, config)
        states, metrics = simulator.run(
            initial_seeds=[0], target_node=5
        )

        # With fanout=1 and no dropout, infection should propagate linearly
        # t=0: I={0}
        # t=1: I={1}
        # t=2: I={2}
        # t=3: I={3}
        # t=4: I={4}
        # t=5: I={5} <- target reached
        assert metrics["target_reached"]
        assert metrics["target_reach_time"] == 5

    def test_no_dropout_vs_dropout(self):
        """Test that dropout reduces reach probability."""
        N, K = 100, 2
        adj, node_to_cluster, _ = generate_sbm_graph(
            N=N, K=K, avg_degree=20, seed=42
        )

        config_no_dropout = Config(
            mode="micro",
            N=N,
            K=K,
            fanout=2,
            p_dropout=0.0,
            max_steps=20,
            seed=42,
        )

        config_with_dropout = Config(
            mode="micro",
            N=N,
            K=K,
            fanout=2,
            p_dropout=0.5,
            max_steps=20,
            seed=42,
        )

        sim1 = SimulatorMicro(adj, node_to_cluster, config_no_dropout)
        _, metrics1 = sim1.run()

        sim2 = SimulatorMicro(adj, node_to_cluster, config_with_dropout)
        _, metrics2 = sim2.run()

        # No dropout should reach more nodes
        assert metrics1["final_infected_count"] >= metrics2["final_infected_count"]

    def test_idempotence(self):
        """Test that same seed produces same results."""
        N, K = 100, 2
        adj, node_to_cluster, _ = generate_sbm_graph(
            N=N, K=K, avg_degree=15, seed=42
        )

        config = Config(
            mode="micro",
            N=N,
            K=K,
            fanout=2,
            p_dropout=0.1,
            max_steps=15,
            seed=42,
        )

        sim1 = SimulatorMicro(adj, node_to_cluster, config)
        _, metrics1 = sim1.run()

        sim2 = SimulatorMicro(adj, node_to_cluster, config)
        _, metrics2 = sim2.run()

        assert metrics1["target_reached"] == metrics2["target_reached"]
        assert metrics1["target_reach_time"] == metrics2["target_reach_time"]
        assert metrics1["final_infected_count"] == metrics2["final_infected_count"]

    def test_no_reinfection(self):
        """Test that recovered nodes never become infected again."""
        N, K = 100, 2
        adj, node_to_cluster, _ = generate_sbm_graph(
            N=N, K=K, avg_degree=15, seed=42
        )

        config = Config(
            mode="micro",
            N=N,
            K=K,
            fanout=2,
            p_dropout=0.1,
            max_steps=20,
            seed=42,
        )

        simulator = SimulatorMicro(adj, node_to_cluster, config)
        states, _ = simulator.run()

        # Check that R never decreases
        for i in range(1, len(states)):
            assert len(states[i].R) >= len(states[i - 1].R)

        # Check that S never increases
        for i in range(1, len(states)):
            assert len(states[i].S) <= len(states[i - 1].S)


class TestMacroSimulation:
    """Test macro-scale simulation."""

    def test_macro_basic(self):
        """Test basic macro simulation."""
        cluster_sizes = np.array([1000, 1000, 1000])
        config = Config(
            mode="macro",
            K=3,
            fanout=2,
            p_dropout=0.1,
            max_steps=20,
            seed=42,
        )

        simulator = SimulatorMacro(cluster_sizes, config)
        states, metrics = simulator.run()

        assert len(states) > 1
        assert metrics["final_infected_count"] >= 0

    def test_macro_idempotence(self):
        """Test that macro simulation is reproducible."""
        cluster_sizes = np.array([500, 500, 500])
        config = Config(
            mode="macro",
            K=3,
            fanout=2,
            p_dropout=0.1,
            max_steps=15,
            seed=42,
        )

        sim1 = SimulatorMacro(cluster_sizes, config)
        _, metrics1 = sim1.run()

        sim2 = SimulatorMacro(cluster_sizes, config)
        _, metrics2 = sim2.run()

        assert metrics1["target_reached"] == metrics2["target_reached"]
        assert metrics1["final_infected_count"] == metrics2["final_infected_count"]


class TestTargetSelector:
    """Test target selection."""

    def test_target_selection_other_cluster(self):
        """Test that target is selected from other cluster."""
        N, K = 100, 3
        _, node_to_cluster, _ = generate_sbm_graph(
            N=N, K=K, avg_degree=10, seed=42
        )

        config = Config(
            mode="micro",
            N=N,
            K=K,
            friend_target__must_cross_clusters=True,
            seed=42,
        )

        rng = np.random.RandomState(42)
        selector = TargetSelector(config, node_to_cluster, rng)

        source_node = 0
        source_cluster = node_to_cluster[source_node]

        for _ in range(10):
            target = selector.select_target(source_node)
            target_cluster = node_to_cluster[target]
            assert target_cluster != source_cluster

    def test_initial_seeds_selection(self):
        """Test initial seeds selection."""
        N, K = 100, 3
        _, node_to_cluster, _ = generate_sbm_graph(
            N=N, K=K, avg_degree=10, seed=42
        )

        config = Config(
            mode="micro",
            N=N,
            K=K,
            initial_seeds=InitialSeedsConfig(
                count=3,
                cluster_policy="home",
                cluster_id=0,
            ),
            seed=42,
        )

        rng = np.random.RandomState(42)
        selector = TargetSelector(config, node_to_cluster, rng)

        seeds = selector.select_initial_seeds()
        assert len(seeds) == 3
        for seed in seeds:
            assert node_to_cluster[seed] == 0


class TestConfig:
    """Test configuration."""

    def test_config_default_micro(self):
        """Test default micro config."""
        config = Config.default_micro()
        assert config.mode == "micro"
        assert config.N == 200000
        assert config.K == 50

    def test_config_default_macro(self):
        """Test default macro config."""
        config = Config.default_macro()
        assert config.mode == "macro"
        assert config.K == 200

    def test_config_toy_linear_chain(self):
        """Test toy linear chain config."""
        config = Config.toy_linear_chain()
        assert config.N == 6
        assert config.fanout == 1
        assert config.p_dropout == 0.0

    def test_config_save_load(self, tmp_path):
        """Test config save and load."""
        config = Config.default_micro()
        config_path = tmp_path / "config.json"

        config.save(config_path)
        loaded_config = Config.load(config_path)

        assert loaded_config.N == config.N
        assert loaded_config.K == config.K
        assert loaded_config.fanout == config.fanout

