# Spath

## Setup

1. **Create conda environment:**
   ```bash
   conda env create -f spath-env.yaml
   conda activate spath
   ```

2. **Generate training data:**
   ```bash
   cd spath_data_gen
   python3 data_gen.py
   python3 data_viz.py
   ```

3. **Run training:**
   ```bash
   bash sft.sh
   ```

Training logs to wandb and saves checkpoints to `Models/`.

