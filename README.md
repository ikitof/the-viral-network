# Viral Network Simulation

A Python package for simulating message propagation in a global social network with cluster structure (e.g., countries). Supports both micro-scale (exact) and macro-scale (approximate) simulations.

## Features

- **Micro-scale simulation**: Exact simulation on explicit graph (N ≤ 200,000)
- **Macro-scale simulation**: Approximate cluster-level dynamics (N up to billions)
- **Stochastic Block Model (SBM)**: Realistic network generation with intra/inter-cluster structure
- **Configurable dynamics**: Fanout, dropout probability, maximum steps
- **Reproducible**: Full seed control for reproducibility
- **Metrics & visualization**: Comprehensive KPIs and plotting utilities
- **CLI interface**: Easy-to-use command-line interface

## Installation

### Using Poetry

```bash
poetry install
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Toy Example: Linear Chain

```python
from viral_network.config import Config
from viral_network.simulate_micro import SimulatorMicro
from scipy import sparse
import numpy as np

# Create linear chain: 0-1-2-3-4-5
rows = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
cols = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))
node_to_cluster = np.array([0, 0, 0, 1, 1, 1])

config = Config.toy_linear_chain()
simulator = SimulatorMicro(adj, node_to_cluster, config)
states, metrics = simulator.run(initial_seeds=[0], target_node=5)

print(f"Target reached: {metrics['target_reached']}")
print(f"Time to reach: {metrics['target_reach_time']}")
```

### Micro-Scale Simulation

```python
from viral_network.config import Config
from viral_network.graph_generation import generate_sbm_graph
from viral_network.simulate_micro import SimulatorMicro
from viral_network.targets import TargetSelector

# Create config
config = Config(
    mode="micro",
    N=200000,
    K=50,
    avg_degree=100,
    fanout=2,
    p_dropout=0.15,
    max_steps=64,
    seed=42,
)

# Generate graph
adj, node_to_cluster, metadata = generate_sbm_graph(
    N=config.N,
    K=config.K,
    avg_degree=config.avg_degree,
    seed=config.seed,
)

# Run simulation
simulator = SimulatorMicro(adj, node_to_cluster, config)
target_selector = TargetSelector(config, node_to_cluster, simulator.rng)
initial_seeds = target_selector.select_initial_seeds()
target_node = target_selector.select_target(initial_seeds[0])

states, metrics = simulator.run(initial_seeds=initial_seeds, target_node=target_node)

print(f"Target reached: {metrics['target_reached']}")
print(f"Final infected: {metrics['final_infected_count']}")
```

### Macro-Scale Simulation

```python
from viral_network.config import Config
from viral_network.simulate_macro import SimulatorMacro
from viral_network.metrics import MetricsCollector
import numpy as np

# Create config
config = Config(
    mode="macro",
    K=200,
    avg_degree=100,
    fanout=2,
    p_dropout=0.2,
    max_steps=64,
    seed=42,
)

# Create cluster sizes
cluster_sizes = np.ones(config.K, dtype=int) * (7_000_000_000 // config.K)

# Run multiple simulations
collector = MetricsCollector(cluster_sizes.sum())
for run_id in range(100):
    simulator = SimulatorMacro(cluster_sizes, config)
    states, metrics = simulator.run()
    collector.add_run(metrics)

# Aggregate results
aggregate = collector.compute_aggregate_metrics()
print(f"Target reach probability: {aggregate['target_reach_probability']}")
print(f"Mean reach time: {aggregate['mean_reach_time']}")
```

## CLI Usage

### Run Micro Simulation

```bash
python -m viral_network.cli run \
  --mode micro \
  --N 200000 \
  --K 50 \
  --avg-degree 100 \
  --fanout 2 \
  --p-dropout 0.15 \
  --max-steps 64 \
  --output-dir runs/exp001
```

### Run Macro Simulation

```bash
python -m viral_network.cli run \
  --mode macro \
  --K 200 \
  --avg-degree 100 \
  --fanout 2 \
  --p-dropout 0.2 \
  --max-steps 64 \
  --runs 200 \
  --output-dir runs/macro_world
```

### Generate Plots

```bash
python -m viral_network.cli plot --run-id runs/exp001
```

## Configuration

Configuration is managed via `Config` class. Key parameters:

- `seed`: Random seed for reproducibility
- `mode`: "micro" (exact) or "macro" (approximate)
- `N`: Number of nodes (micro mode)
- `K`: Number of clusters
- `avg_degree`: Average degree per node
- `fanout`: Number of neighbors to transmit to (default: 2)
- `p_dropout`: Probability of not transmitting (default: 0.15)
- `max_steps`: Maximum simulation steps (default: 64)
- `mixing.intra_strength`: Intra-cluster connection probability (default: 0.95)
- `mixing.inter_floor`: Inter-cluster connection probability (default: 0.05)

See `viral_network/config.py` for full configuration options.

## Simulation Dynamics

### States

- **S (Susceptible)**: Never received the message
- **I (Infected)**: Just received, will attempt transmission next step
- **R (Recovered)**: Already transmitted, will never transmit again

### Transmission Rules

1. Each infected node chooses exactly 2 distinct neighbors (fanout=2)
2. With probability `p_dropout`, the node doesn't transmit this step
3. Otherwise, it attempts to inform the chosen neighbors
4. Neighbors in state S become I; neighbors in I or R remain unchanged
5. All I nodes become R at the end of the step

### Termination

Simulation stops when:
- Target is reached, OR
- No more infected nodes (I is empty), OR
- Maximum steps reached

## Metrics

Key metrics computed:

- `target_reached`: Boolean, whether target was reached
- `target_reach_time`: Time step when target was reached
- `final_infected_count`: Total number of infected nodes
- `max_concurrent_I`: Maximum number of concurrent infected
- `inter_cluster_transmissions`: Number of cross-cluster transmissions
- `attack_rate`: Fraction of population infected
- `R_eff`: Effective reproduction number

## Visualization

The package provides several visualization utilities:

- `plot_timeseries()`: I(t), R(t), cumulative infections
- `plot_inter_cluster_heatmap()`: Inter-cluster transmission flows
- `plot_reach_time_distribution()`: Distribution of reach times
- `plot_cluster_dynamics()`: Per-cluster infection dynamics

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=viral_network --cov-report=html
```

## Examples

See `examples/` directory for:

- `notebooks/demo.ipynb`: Interactive Jupyter notebook with examples
- `configs/toy_linear_chain.json`: Toy configuration
- `configs/micro_200k.json`: Micro-scale configuration
- `configs/macro_world.json`: Macro-scale configuration

## Performance

### Micro Mode

- N=10,000: ~1 second
- N=100,000: ~10 seconds
- N=200,000: ~30 seconds

Memory usage: ~500 MB for N=200,000

### Macro Mode

- K=200, 100 runs: ~1 second
- K=500, 100 runs: ~2 seconds

Memory usage: Negligible

## Limitations

- Micro mode limited to N ≤ 1,000,000 due to memory constraints
- Macro mode is approximate (uses cluster-level counts)
- No spatial/geographic constraints (fully connected clusters)
- Assumes homogeneous transmission within clusters

## Future Work

- Numba acceleration for hot loops
- GIF/MP4 animation export
- Web interface (FastAPI + Plotly Dash)
- Spatial constraints and geographic distance
- Heterogeneous node properties

## License

MIT

## References

- Stochastic Block Models: https://en.wikipedia.org/wiki/Stochastic_block_model
- Branching processes: https://en.wikipedia.org/wiki/Branching_process
- Network epidemiology: https://en.wikipedia.org/wiki/Epidemic_model