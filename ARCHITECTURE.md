# Architecture Overview

This document describes the architecture and design of the viral network simulation package.

## Project Structure

```
viral_network/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── graph_generation.py      # SBM graph generation
├── simulate_micro.py        # Micro-scale (exact) simulation
├── simulate_macro.py        # Macro-scale (approximate) simulation
├── targets.py               # Target friend selection
├── metrics.py               # Metrics collection and analysis
├── viz.py                   # Visualization utilities
└── cli.py                   # Command-line interface

tests/
├── __init__.py
└── test_simulation.py       # Comprehensive test suite

examples/
├── configs/                 # Example configuration files
│   ├── toy_linear_chain.json
│   ├── micro_200k.json
│   └── macro_world.json
├── notebooks/
│   └── demo.ipynb           # Interactive Jupyter notebook
└── simple_demo.py           # Simple Python demo script
```

## Core Modules

### config.py

Manages all configuration parameters using Pydantic for validation.

**Key Classes**:
- `Config`: Main configuration container
- `MixingConfig`: Inter/intra-cluster mixing probabilities
- `FriendTargetConfig`: Target friend selection strategy
- `InitialSeedsConfig`: Initial seed selection strategy

**Features**:
- JSON serialization/deserialization
- Validation with sensible defaults
- Pre-configured templates (default_micro, default_macro, toy_linear_chain)

### graph_generation.py

Generates Stochastic Block Model (SBM) graphs with cluster structure.

**Key Functions**:
- `compute_mixing_matrix()`: Compute inter/intra-cluster connection probabilities
- `generate_sbm_graph()`: Generate SBM graph with specified parameters
- `get_neighbors()`: Get neighbors from sparse adjacency matrix
- `sample_neighbors()`: Sample k distinct neighbors without replacement

**Features**:
- Sparse matrix representation (CSR format)
- Power-law cluster size distribution
- Configurable intra/inter-cluster connectivity

### simulate_micro.py

Exact simulation on explicit graph (N ≤ 200,000).

**Key Classes**:
- `SimulationState`: Dataclass representing state at time t
- `SimulatorMicro`: Main simulator class

**Algorithm**:
1. Initialize S (susceptible), I (infected), R (recovered)
2. For each time step:
   - Each I node samples fanout neighbors
   - With probability (1 - p_dropout), attempts transmission
   - S neighbors become I; I/R neighbors unchanged
   - All I nodes become R
3. Stop when target reached, I empty, or max_steps exceeded

**Features**:
- Exact node-level simulation
- Dropout probability support
- Target tracking
- Comprehensive metrics collection

### simulate_macro.py

Approximate cluster-level simulation for large populations.

**Key Classes**:
- `MacroState`: Dataclass for cluster-level state
- `SimulatorMacro`: Cluster-level simulator

**Algorithm**:
1. Track S_k, I_k, R_k per cluster k
2. For each time step:
   - Each I_k generates expected transmissions
   - Distribute across clusters via mixing matrix
   - Stochastic update (Poisson sampling)
   - Cap by available susceptible
3. Stop when target cluster reached or max_steps exceeded

**Features**:
- Cluster-level dynamics
- Mixing matrix for inter-cluster flow
- Stochastic updates
- 100-1000x faster than micro-scale

### targets.py

Manages target friend and initial seed selection.

**Key Classes**:
- `TargetSelector`: Selects targets and initial seeds

**Features**:
- Random target in different cluster
- Explicit target specification
- Configurable initial seed selection
- Support for single/multiple seeds

### metrics.py

Collects and aggregates simulation metrics.

**Key Classes**:
- `Metrics`: Dataclass for single-run metrics
- `MetricsCollector`: Aggregates metrics across runs

**Metrics Computed**:
- target_reached: Boolean
- target_reach_time: Time to reach target
- final_infected_count: Total infected
- max_concurrent_I: Peak concurrent infections
- attack_rate: Fraction of population infected
- R_eff: Effective reproduction number
- die_out_rate: Fraction of runs that die out

### viz.py

Visualization utilities using Matplotlib and Plotly.

**Key Functions**:
- `plot_timeseries()`: I(t), R(t), cumulative infections
- `plot_inter_cluster_heatmap()`: Inter-cluster transmission flows
- `plot_reach_time_distribution()`: Distribution of reach times
- `plot_cluster_dynamics()`: Per-cluster infection dynamics

### cli.py

Command-line interface using Typer.

**Commands**:
- `run`: Execute simulation with specified parameters
- `plot`: Generate plots from completed run

**Features**:
- Parameter validation
- Config saving
- Automatic output directory creation
- Logging integration

## Design Patterns

### State Management

- **Immutable states**: SimulationState and MacroState are immutable dataclasses
- **State history**: All states stored for analysis
- **Lazy evaluation**: Metrics computed after simulation

### Separation of Concerns

- **Graph generation**: Independent of simulation
- **Simulation**: Independent of visualization
- **Configuration**: Centralized, validated
- **Metrics**: Decoupled from simulation logic

### Extensibility

- **Config classes**: Easy to add new parameters
- **Simulator classes**: Can be subclassed for variants
- **Metrics**: Can add new metrics without changing simulation
- **Visualization**: Can add new plot types

## Data Flow

```
Config
  ↓
Graph Generation → Adjacency Matrix + Node-to-Cluster Mapping
  ↓
Target Selection → Initial Seeds + Target Node
  ↓
Simulation (Micro or Macro)
  ↓
States (List of SimulationState or MacroState)
  ↓
Metrics Computation
  ↓
Visualization
```

## Performance Considerations

### Micro-Scale

- **Bottleneck**: Graph generation (O(N * avg_degree))
- **Memory**: O(N + E) for sparse adjacency matrix
- **Optimization**: Use sparse matrices (CSR format)

### Macro-Scale

- **Bottleneck**: Stochastic updates (Poisson sampling)
- **Memory**: O(K) for cluster-level state
- **Optimization**: Vectorized NumPy operations

## Testing Strategy

### Unit Tests

- Graph generation correctness
- Simulation invariants (no reinfection, monotonic R)
- Configuration validation
- Metrics computation

### Integration Tests

- End-to-end micro simulation
- End-to-end macro simulation
- Reproducibility (same seed → same results)
- Idempotence

### Toy Cases

- Linear chain (6 nodes): Deterministic reach time
- Small SBM (K=3): Saturation behavior
- Parameter sensitivity: Dropout effects

## Future Enhancements

1. **Numba acceleration**: JIT compile hot loops
2. **Spatial constraints**: Geographic distance effects
3. **Heterogeneous nodes**: Variable transmission rates
4. **Temporal dynamics**: Time-varying network structure
5. **Web interface**: FastAPI + Plotly Dash
6. **Animation export**: GIF/MP4 generation
7. **Parallel runs**: Multi-processing for macro simulations
8. **GPU acceleration**: CUDA for large-scale simulations

## Dependencies

### Core

- **numpy**: Numerical computing
- **scipy**: Sparse matrices, scientific functions
- **networkx**: Graph algorithms (optional, for analysis)

### Visualization

- **matplotlib**: Static plots
- **plotly**: Interactive plots

### CLI

- **typer**: Command-line interface
- **pydantic**: Configuration validation

### Development

- **pytest**: Testing framework
- **mypy**: Type checking
- **black**: Code formatting
- **ruff**: Linting

## Code Quality

- **Type hints**: Full type annotations
- **Docstrings**: Comprehensive documentation
- **Tests**: 16 test cases covering core functionality
- **Logging**: Structured logging with loguru
- **Error handling**: Validation and graceful degradation

