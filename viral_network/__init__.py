"""
Viral Network Simulation Package

A Python package for simulating message propagation in a global social network
with cluster structure (e.g., countries). Supports both micro-scale (exact)
and macro-scale (approximate) simulations.
"""

__version__ = "0.1.0"
__author__ = "Author"

from viral_network.config import Config
from viral_network.graph_generation import generate_sbm_graph
from viral_network.simulate_micro import SimulatorMicro
from viral_network.simulate_macro import SimulatorMacro
from viral_network.metrics import MetricsCollector

__all__ = [
    "Config",
    "generate_sbm_graph",
    "SimulatorMicro",
    "SimulatorMacro",
    "MetricsCollector",
]

