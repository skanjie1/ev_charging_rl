# EV Charging RL — How to Run

## Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies
pip install numpy matplotlib
```

## Run

```bash
# Step 1: Train agents (~30 seconds)
python3 train.py

# Step 2: Run experiments (~3-5 minutes)
python3 experiments.py

# Step 3: Generate plots
python3 visualizations.py
```

All figures will be saved in `results/figures/`.
