"""Configuration management for viral network simulations."""

from typing import Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import json
from datetime import datetime


class MixingConfig(BaseModel):
    """Configuration for inter/intra-cluster mixing probabilities."""

    intra_strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Probability of intra-cluster connections",
    )
    inter_floor: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum inter-cluster connection probability",
    )


class FriendTargetConfig(BaseModel):
    """Configuration for target friend selection."""

    select: Literal["random_other_cluster", "explicit"] = Field(
        default="random_other_cluster",
        description="How to select the target friend",
    )
    cluster_id: Optional[int] = Field(
        default=None, description="Explicit cluster ID (if select='explicit')"
    )
    node_id: Optional[int] = Field(
        default=None, description="Explicit node ID (if select='explicit')"
    )
    must_cross_clusters: bool = Field(
        default=True, description="Target must be in different cluster than source"
    )


class InitialSeedsConfig(BaseModel):
    """Configuration for initial seed selection."""

    type: Literal["single", "multi"] = Field(
        default="single", description="Single or multiple initial seeds"
    )
    cluster_policy: Literal["home", "random"] = Field(
        default="home", description="Which cluster to seed from"
    )
    count: int = Field(default=1, ge=1, description="Number of initial seeds")
    cluster_id: Optional[int] = Field(
        default=0, description="Cluster ID for 'home' policy"
    )


class Config(BaseModel):
    """Main configuration for viral network simulations."""

    # Random seed
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Simulation mode
    mode: Literal["micro", "macro"] = Field(
        default="micro", description="Simulation mode: 'micro' (exact) or 'macro' (approx)"
    )

    # Graph parameters
    N: int = Field(
        default=200000, ge=1, description="Number of nodes (micro mode)"
    )
    K: int = Field(default=50, ge=2, description="Number of clusters (countries)")
    cluster_sizes: Literal["uniform", "powerlaw", "custom"] = Field(
        default="powerlaw", description="Distribution of cluster sizes"
    )
    avg_degree: int = Field(
        default=100, ge=2, description="Average degree per node"
    )

    # Dynamics
    fanout: int = Field(
        default=2, ge=1, description="Number of neighbors to attempt transmission to"
    )
    p_dropout: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Probability of not transmitting"
    )
    max_steps: int = Field(
        default=64, ge=1, description="Maximum simulation steps"
    )

    # Mixing
    mixing: MixingConfig = Field(
        default_factory=MixingConfig, description="Inter/intra-cluster mixing config"
    )

    # Target
    friend_target: FriendTargetConfig = Field(
        default_factory=FriendTargetConfig, description="Target friend config"
    )

    # Initial seeds
    initial_seeds: InitialSeedsConfig = Field(
        default_factory=InitialSeedsConfig, description="Initial seeds config"
    )

    # Output
    output_dir: str = Field(
        default="runs/exp001", description="Output directory for results"
    )

    # Macro-specific
    runs: int = Field(
        default=100, ge=1, description="Number of runs for macro mode"
    )

    @field_validator("N")
    @classmethod
    def validate_N_for_mode(cls, v: int, info) -> int:
        """Validate N is reasonable for micro mode."""
        if info.data.get("mode") == "micro" and v > 7_000_000_000:
            raise ValueError("N should be <= 1,000,000 for micro mode")
        return v

    @field_validator("K")
    @classmethod
    def validate_K(cls, v: int) -> int:
        """Validate K is reasonable."""
        if v > 10000:
            raise ValueError("K should be <= 10,000")
        return v

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return self.model_dump_json(indent=2)

    def save(self, path: Path | str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        """Load config from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def default_micro(cls) -> "Config":
        """Create default micro-scale config."""
        return cls(
            mode="micro",
            N=200000,
            K=50,
            avg_degree=100,
            fanout=2,
            p_dropout=0.15,
            max_steps=64,
        )

    @classmethod
    def default_macro(cls) -> "Config":
        """Create default macro-scale config."""
        return cls(
            mode="macro",
            N=7_000_000_000,  # Symbolic
            K=200,
            avg_degree=100,
            fanout=2,
            p_dropout=0.2,
            max_steps=64,
            runs=200,
        )

    @classmethod
    def toy_linear_chain(cls) -> "Config":
        """Create toy config: linear chain of 6 nodes."""
        return cls(
            mode="micro",
            seed=42,
            N=6,
            K=2,
            cluster_sizes="custom",
            avg_degree=2,
            fanout=1,
            p_dropout=0.0,
            max_steps=10,
        )

