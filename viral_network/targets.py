"""Target friend management and selection."""

import numpy as np
from typing import Optional, Tuple
from loguru import logger

from viral_network.config import Config


class TargetSelector:
    """Manages target friend selection."""

    def __init__(self, config: Config, node_to_cluster: np.ndarray, rng: np.random.RandomState):
        """
        Initialize target selector.

        Args:
            config: Configuration object
            node_to_cluster: Array mapping node to cluster
            rng: Random state
        """
        self.config = config
        self.node_to_cluster = node_to_cluster
        self.rng = rng
        self.N = len(node_to_cluster)
        self.K = len(np.unique(node_to_cluster))

    def select_target(self, source_node: int) -> int:
        """
        Select a target node.

        Args:
            source_node: Source node (seed)

        Returns:
            Target node index
        """
        if self.config.friend_target.select == "explicit":
            if self.config.friend_target.node_id is not None:
                return self.config.friend_target.node_id
            elif self.config.friend_target.cluster_id is not None:
                cluster_id = self.config.friend_target.cluster_id
                cluster_nodes = np.where(self.node_to_cluster == cluster_id)[0]
                if len(cluster_nodes) > 0:
                    return self.rng.choice(cluster_nodes)

        # Default: random_other_cluster
        source_cluster = self.node_to_cluster[source_node]

        if self.config.friend_target.must_cross_clusters:
            other_clusters = [c for c in range(self.K) if c != source_cluster]
            if not other_clusters:
                logger.warning("No other clusters available, selecting random node")
                return self.rng.choice(self.N)
            target_cluster = self.rng.choice(other_clusters)
        else:
            target_cluster = self.rng.choice(self.K)

        cluster_nodes = np.where(self.node_to_cluster == target_cluster)[0]
        if len(cluster_nodes) == 0:
            return self.rng.choice(self.N)

        return self.rng.choice(cluster_nodes)

    def select_initial_seeds(self) -> list:
        """
        Select initial seed nodes.

        Returns:
            List of initial seed node indices
        """
        count = self.config.initial_seeds.count
        cluster_id = self.config.initial_seeds.cluster_id or 0

        if self.config.initial_seeds.cluster_policy == "home":
            cluster_nodes = np.where(self.node_to_cluster == cluster_id)[0]
            if len(cluster_nodes) == 0:
                logger.warning(f"Cluster {cluster_id} is empty, using random nodes")
                return [self.rng.choice(self.N) for _ in range(count)]
            # Sample with replacement if count > len(cluster_nodes)
            return [self.rng.choice(cluster_nodes) for _ in range(count)]
        else:  # random
            return [self.rng.choice(self.N) for _ in range(count)]

