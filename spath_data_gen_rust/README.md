# Fast Rust Implementation for Shortest Path Data Generation

This Rust module provides high-performance graph generation and shortest path calculations for the shortest path data generation pipeline. It can be used both as a Python extension module and as a standalone command-line tool.

## Building

### Prerequisites

1. Install Rust: https://rustup.rs/
2. For Python integration, install maturin (Python-Rust bridge):
   ```bash
   pip install maturin
   ```

### Build Python Extension Module

From the project root:
```bash
cd spath_data_gen_rust
maturin develop --release
```

Or to build in development mode (faster compilation, slower runtime):
```bash
maturin develop
```

### Build Standalone Binary

To build the standalone command-line tool:
```bash
cd spath_data_gen_rust
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release --bin spath_data_gen_rust
```

The binary will be located at `target/release/spath_data_gen_rust`.

## Usage

### As a Python Module

After building, you can use the fast Rust implementation from Python:

```python
from spath_data_gen.data_gen_len_rust import generate_dataset
generate_dataset()
```

### As a Standalone Command-Line Tool

The standalone binary can be run directly with command-line arguments:

```bash
# Using cargo run (development)
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo run --bin spath_data_gen_rust -- \
  -N 10000 \
  -p 0.4 \
  -s 0.2 \
  --min-nodes 5 \
  --max-nodes 20 \
  --seed 42 \
  -o output_data

# Using the built binary (release)
./target/release/spath_data_gen_rust --help
```

## Command-Line Arguments

### Required Arguments
None (all arguments have defaults)

### Optional Arguments

#### Dataset Generation
- **`-N, --num-datapoints <NUM>`**: Number of datapoints to generate (default: 500000)
- **`-s, --train-split <RATIO>`**: Train split ratio - fraction of data for training, remainder goes to validation (default: 0.1)
  - **Important**: This is the fraction that goes to **training**, not validation
  - Example: `0.1` means 10% train / 90% val, `0.8` means 80% train / 20% val
  - Must be between 0 and 1 (exclusive)

#### Graph Generation Parameters
- **`-p, --edge-probability <PROB>`**: Edge probability p for Erdős–Rényi G(n, p) distribution (default: 0.3)
  - Must be between 0 and 1 (inclusive)
- **`--min-nodes <N>`**: Minimum number of nodes in generated graphs (default: 1)
- **`--max-nodes <N>`**: Maximum number of nodes in generated graphs (default: 30)
- **`--min-weight <W>`**: Minimum edge weight (default: 1)
- **`--max-weight <W>`**: Maximum edge weight (default: 50)

#### In-Context Learning Parameters
- **`-e, --num-examples <N>`**: Number of example pairs (u_i, v_i) to include in each datapoint (default: 5)
- **`--max-rejection-attempts <N>`**: Maximum attempts for rejection sampling per pair (default: 100)
- **`--max-unique-attempts <N>`**: Maximum attempts to generate unique examples before regenerating graph (default: 500)

#### Output Configuration
- **`-o, --output-dir <DIR>`**: Output directory for train and val JSON files (default: "data")
- **`--train-output <FILE>`**: Train output filename (default: "train_len.json")
- **`--val-output <FILE>`**: Val output filename (default: "val_len.json")

#### Other Options
- **`--seed <SEED>`**: Random seed for reproducibility (default: 182)
- **`--print-progress-interval <N>`**: Print progress every N datapoints (default: 100)
- **`-h, --help`**: Print help message
- **`-V, --version`**: Print version

## Examples

### Generate a small test dataset
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo run --bin spath_data_gen_rust -- \
  -N 1000 \
  -s 0.2 \
  -o test_data
```

### Generate dataset with custom graph parameters
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo run --bin spath_data_gen_rust -- \
  -N 50000 \
  -p 0.5 \
  --min-nodes 10 \
  --max-nodes 25 \
  --min-weight 5 \
  --max-weight 100 \
  -s 0.15 \
  --seed 12345
```

### Generate dataset with more in-context examples
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo run --bin spath_data_gen_rust -- \
  -N 100000 \
  -e 10 \
  -s 0.1 \
  -o data_with_more_examples
```

## Output Format

The tool generates two JSON files:
- **Train file**: Contains training datapoints in instruction format
- **Val file**: Contains validation datapoints in instruction format

Each datapoint in the JSON files has the following structure:
```json
{
  "input": "<graph>...\nShortest path length:\nu1 v1 -> length1\n...\nu_q v_q ->",
  "output": "length"
}
```

The input contains:
1. A `<graph>` section with vertices and edge weights
2. Example pairs in format "u v -> length"
3. A query pair "u v ->" (without answer)

The output is the shortest path length for the query pair.

## Train/Val Split Clarification

The `--train-split` parameter specifies the **fraction of data that goes to training**:

- `--train-split 0.1` → 10% train, 90% validation
- `--train-split 0.2` → 20% train, 80% validation  
- `--train-split 0.8` → 80% train, 20% validation
- `--train-split 0.9` → 90% train, 10% validation

The split is calculated as:
- Train: first `train_split * total_datapoints` datapoints
- Val: remaining datapoints

Datapoints are shuffled before splitting to ensure random distribution.

## Performance

The Rust implementation is significantly faster than the Python/NetworkX version:
- Graph generation: ~10-50x faster
- Shortest path calculations: ~5-20x faster
- Overall data generation: ~5-15x faster depending on graph size

## Implementation Details

- Uses Dijkstra's algorithm implemented in Rust for shortest path calculations
- Efficient adjacency list representation
- Thread-safe random number generation
- Optimized for release builds with LTO (Link Time Optimization)
- Generates Erdős–Rényi random graphs with weighted edges
- Ensures unique example pairs and query pairs within each datapoint
- Deduplicates datapoints using hash-based checking

## Troubleshooting

### Python Version Compatibility

If you encounter PyO3 version errors when building the Python extension, you can:
1. Use the standalone binary instead (doesn't require Python)
2. Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` environment variable
3. Update PyO3 to a newer version that supports your Python version

### Building the Binary

When building the standalone binary, you may need to set:
```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo build --release --bin spath_data_gen_rust
```

This allows the build to proceed even if PyO3 detects a newer Python version than officially supported.
