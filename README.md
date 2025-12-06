# Spath

## Setup

1. **Create conda environment:**
   ```bash
   conda env create -f spath-env.yaml
   conda activate spath
   ```

2. **Build Rust extension (optional, for faster data generation):**
   ```bash
   cd spath_data_gen_rust
   maturin develop --release
   cd ..
   ```

3. **Generate training data:**
   ```bash
   cd spath_data_gen
   python3 data_gen_len.py
   # Or use the Rust-accelerated version:
   python3 data_gen_len_rust.py
   ```

4. **Run training:**
   ```bash
   bash sft-len.sh
   ```

Training logs to wandb and saves checkpoints to `Models/`.

## Data Generation Usage

### Python Scripts

Both `data_gen_len.py` and `data_gen_len_rust.py` support command-line arguments:

#### Basic Usage
```bash
# Generate dataset with default parameters
python3 data_gen_len.py

# Generate dataset with custom parameters
python3 data_gen_len.py \
  --seed 42 \
  --samples-per-stage 5000 \
  --min-nodes 5 \
  --max-nodes 30 \
  --num-examples 5 \
  --train-split 0.1
```

#### Common Arguments

**Dataset Configuration:**
- `--seed <N>`: Random seed (default: 182)
- `--train-output-file <FILE>`: Training output file (default: "data/train_len.json")
- `--val-output-file <FILE>`: Validation output file (default: "data/val_len.json")
- `--train-split <RATIO>`: Fraction for training, remainder for validation (default: 0.1)
- `--use-curriculum` / `--no-curriculum`: Enable/disable curriculum learning (default: enabled)
- `--num-datapoints <N>`: Total datapoints (only used with `--no-curriculum`, default: 500000)

**Graph Generation:**
- `--edge-probability <P>`: Erdős–Rényi edge probability (default: 0.3)
- `--min-weight <W>`: Minimum edge weight (default: 1)
- `--max-weight <W>`: Maximum edge weight cap (default: 50)

**Curriculum Learning:**
- `--samples-per-stage <N>`: Samples per curriculum stage (default: 3000)
- `--min-nodes <N>`: Minimum number of nodes (default: 5)
- `--max-nodes <N>`: Maximum number of nodes (default: 30)
- `--rehearsal-ratio <R>`: Ratio of rehearsal samples from previous stages (default: 0.1)

**In-Context Learning:**
- `--num-examples <N>`: Number of example pairs per datapoint (default: 5)
- `--max-rejection-attempts <N>`: Max attempts for rejection sampling (default: 100)
- `--max-unique-attempts <N>`: Max attempts for unique examples (default: 500)

**Other:**
- `--print-progress-interval <N>`: Print progress every N datapoints (default: 100)

#### Examples

```bash
# Generate small test dataset
python3 data_gen_len.py --samples-per-stage 100 --max-nodes 10

# Generate dataset with custom curriculum range
python3 data_gen_len.py --min-nodes 10 --max-nodes 20 --samples-per-stage 2000

# Generate random dataset (no curriculum)
python3 data_gen_len.py --no-curriculum --num-datapoints 10000

# Generate with more examples per datapoint
python3 data_gen_len.py --num-examples 10 --samples-per-stage 5000
```

### Rust Standalone Binary

The Rust implementation can also be used as a standalone binary:

```bash
cd spath_data_gen_rust

# Build the binary
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release --bin spath_data_gen_rust

# Run with arguments
./target/release/spath_data_gen_rust \
  --samples-per-stage 3000 \
  --min-nodes 5 \
  --max-nodes 30 \
  --num-examples 5 \
  --train-split 0.1 \
  --seed 182
```

See `spath_data_gen_rust/README.md` for detailed Rust binary usage.

### Help

View all available options:
```bash
python3 data_gen_len.py --help
python3 data_gen_len_rust.py --help
```

