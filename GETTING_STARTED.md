# Getting Started with Viral Network Simulation

## Installation

### Prerequisites

- Python 3.10+
- pip or poetry

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd the-viral-network

# Install in development mode
pip install -e .

# Or with all development dependencies
pip install -e ".[dev]"
```

## Your First Simulation

### 1. Using the Command Line

The simplest way to run a simulation:

```bash
# Run a small micro-scale simulation
viral-network run --mode micro --n 1000 --k 5 --avg-degree 20 \
  --fanout 2 --p-dropout 0.1 --output-dir runs/first_test

# Run a macro-scale simulation
viral-network run --mode macro --k 10 --avg-degree 50 \
  --fanout 2 --p-dropout 0.15 --runs 50 --output-dir runs/macro_test
```

Results will be saved to the output directory with:
- `config.json` - Configuration used
- `metrics.json` - Simulation results
- `timeseries.png` - Visualization (micro-scale)
- `reach_time_distribution.html` - Distribution plot (macro-scale)

### 2. Using Python API

Create a file `my_simulation.py`:

```python
from viral_network import Config, generate_sbm_graph, SimulatorMicro
from viral_network.targets import TargetSelector
from viral_network.viz import plot_timeseries

# Create configuration
config = Config(
    mode="micro",
    N=5000,
    K=10,
    avg_degree=50,
    fanout=2,
    p_dropout=0.15,
    max_steps=32,
    seed=42,
)

# Generate graph
adj, node_to_cluster, metadata = generate_sbm_graph(
    N=config.N,
    K=config.K,
    avg_degree=config.avg_degree,
    seed=config.seed,
)

print(f"Generated graph with {metadata['num_edges']} edges")

# Create simulator
simulator = SimulatorMicro(adj, node_to_cluster, config)

# Select target and seeds
target_selector = TargetSelector(config, node_to_cluster, simulator.rng)
initial_seeds = target_selector.select_initial_seeds()
target_node = target_selector.select_target(initial_seeds[0])

# Run simulation
states, metrics = simulator.run(
    initial_seeds=initial_seeds,
    target_node=target_node
)

# Print results
print(f"Target reached: {metrics['target_reached']}")
print(f"Time to reach: {metrics['target_reach_time']} steps")
print(f"Final infected: {metrics['final_infected_count']} nodes")
print(f"Attack rate: {metrics['final_infected_count'] / config.N:.2%}")

# Visualize
plot_timeseries(
    metrics["times"],
    metrics["I_counts"],
    metrics["R_counts"],
    metrics["cumulative_infected"],
    title="Viral Spread Simulation",
    output_path="simulation_plot.png",
)
```

Run it:
```bash
python my_simulation.py
```

### 3. Using the Demo Script

Run the included demo with multiple examples:

```bash
python examples/simple_demo.py
```

This will show:
- Toy linear chain (6 nodes)
- Micro-scale simulation (5,000 nodes)
- Macro-scale simulation (20 clusters, 50 runs)
- Parameter sensitivity analysis

## Understanding the Modes

### Micro-Scale (Exact)

Use when:
- You need exact node-level results
- N ≤ 200,000
- You have time for computation (seconds to minutes)

Example:
```bash
viral-network run --mode micro --n 50000 --k 20 --avg-degree 100 \
  --fanout 2 --p-dropout 0.15 --output-dir runs/micro
```

### Macro-Scale (Approximate)

Use when:
- You need fast results
- You're simulating large populations (billions)
- You want statistical aggregates
- You need to run many simulations

Example:
```bash
viral-network run --mode macro --k 200 --avg-degree 100 \
  --fanout 2 --p-dropout 0.15 --runs 500 --output-dir runs/macro
```

## Configuration

### Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `N` | 200,000 | 1+ | Number of nodes (micro-scale) |
| `K` | 50 | 1+ | Number of clusters |
| `avg_degree` | 100 | 1+ | Average connections per node |
| `fanout` | 2 | 1+ | Neighbors contacted per transmission |
| `p_dropout` | 0.15 | 0-1 | Probability of failed transmission |
| `max_steps` | 32 | 1+ | Maximum simulation steps |
| `runs` | 100 | 1+ | Number of runs (macro-scale) |

### Configuration Files

Load from JSON:

```python
from viral_network import Config

# Load from file
config = Config.load("examples/configs/micro_200k.json")

# Or create and save
config = Config(N=10000, K=10, fanout=2, p_dropout=0.15)
config.save("my_config.json")
```

## Interpreting Results

### Key Metrics

- **target_reached**: Boolean - did message reach target friend?
- **target_reach_time**: Steps to reach target (if reached)
- **final_infected_count**: Total nodes infected
- **attack_rate**: Fraction of population infected
- **max_concurrent_I**: Peak concurrent infections
- **R_eff**: Effective reproduction number
- **die_out_rate**: Fraction of runs that died out (macro-scale)

### Example Output

```
Target reached: True
Time to reach: 15 steps
Final infected: 3,430 nodes
Attack rate: 68.60%
Max concurrent I: 370
```

## Next Steps

1. **Explore parameters**: Try different dropout probabilities, fanout values
2. **Run sensitivity analysis**: See how parameters affect outcomes
3. **Compare modes**: Run same scenario in micro and macro modes
4. **Visualize results**: Check the generated plots
5. **Read documentation**: See ARCHITECTURE.md for technical details

## Troubleshooting

### "No such option" error

Make sure you're using lowercase parameter names:
```bash
# ✓ Correct
viral-network run --mode micro --n 1000

# ✗ Wrong
viral-network run --mode micro --N 1000
```

### Out of memory

For micro-scale simulations:
- Reduce N (number of nodes)
- Reduce avg_degree (connections per node)
- Use macro-scale instead

### Slow performance

For micro-scale:
- Reduce N or avg_degree
- Reduce max_steps
- Use macro-scale for large populations

For macro-scale:
- Reduce K (number of clusters)
- Reduce runs (number of simulations)

## Getting Help

- Check README.md for detailed documentation
- See ARCHITECTURE.md for technical design
- Review BENCHMARKS.md for performance info
- Run `viral-network run --help` for CLI options
- Check examples/ directory for sample code

## What's Next?

- Modify parameters and observe effects
- Create custom configurations
- Integrate into your research
- Extend with custom metrics
- Contribute improvements!

