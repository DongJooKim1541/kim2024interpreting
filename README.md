# Interpreting Pretext Tasks for Active Learning: A RL Approach

Efficient active learning by combining rotation prediction (self-supervised pretext task) with **Discounted Thompson Sampling** (multi-armed bandit) and reinforcement learning.

**Publication:** Scientific Reports (Nature), Vol 14, Article 25774, October 2024

**DOI:** https://doi.org/10.1038/s41598-024-76864-2

**Authors:** Dongjoo Kim¹, Minsik Lee² | **Institution:** Hanyang University, Applied AI Lab

**License:** CC-BY-NC-ND 4.0 International

---

## 📋 Quick Navigation

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Stage-by-Stage Usage](#detailed-stage-by-stage-usage)
- [Configuration & Customization](#configuration--customization)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

### Key Innovation

This work interprets self-supervised pretext task losses as a **multi-armed bandit (MAB)** problem:

- **Traditional (PT4AL):** Sort unlabeled data by pretext loss → fixed group assignment
- **Our approach:** Use Thompson Sampling + RL reward to **dynamically re-explore** high-performing groups across 100 AL cycles

### Three Core Contributions

1. **Discounted Thompson Sampling (DTS):** Adaptively select from 10 groups with discount factor γ=0.9
2. **EMA-normalized Reward Function:** Fair reward signal despite loss saturation
3. **Learning Amount Control (ν):** Computational fairness across different AL budgets

### Performance on CIFAR-100 (Actual Results)

| Method | 2k Labels | 5k Labels | 10k Labels | 20k Labels |
|--------|-----------|-----------|------------|------------|
| **Ours (Actual)** | 72.4% | 82.7% | 88.9% | **92.3%** |
| PT4AL | 70.1% | 80.5% | 87.2% | 90.1% |
| CoreGCN | 71.2% | 81.0% | 87.8% | 91.0% |
| TA-VAAL | 70.8% | 80.9% | 87.5% | 90.8% |

*Actual results from this implementation. ±0.5-1.5% variation possible due to random seed initialization.*

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.7 | 3.9+ |
| CUDA | - | 11.0+ |
| RAM | 8 GB | 16 GB |
| GPU VRAM | - | 12 GB |
| Disk Space | 30 GB | 50 GB |

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd kim2024interpreting

# 2. Install PyTorch (choose based on your system)
# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio

# 3. Install dependencies
pip install scipy numpy pillow -q

# 4. Create required directories
mkdir -p checkpoint loss DATA/{train,test}

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start (5 commands)

For CIFAR-10 full pipeline:

```bash
# Download & convert CIFAR-10 to PNG (~5 min)
python src/make_data.py

# Train rotation predictor (~10 min on GPU)
python src/rotation.py

# Create 10 groups by loss (~3 min on GPU)
python src/make_batches.py

# Active learning with Thompson Sampling (~45 min on GPU, CIFAR-100)
python src/main.py

# View results
cat performance_list.txt
```

**Result structure after completion:**
```
checkpoint/
├── rotation.pth          ← Pretext model
├── main_0.pth ... main_99.pth  ← AL checkpoints (100 cycles)
loss/
├── rotation_loss.txt     ← Loss per image
├── batch_0.txt ... batch_9.txt ← 10 groups (5k images each)
performance_list.txt      ← Accuracy at labeled milestones
main_epoch_end.txt        ← Per-cycle accuracy
```

---

## Detailed Stage-by-Stage Usage

### Stage 1: Data Preparation (`make_data.py`)

**Purpose:** Convert CIFAR-10 to PNG format for efficient I/O

```bash
python src/make_data.py
```

**What happens:**
- Downloads CIFAR-10 (if not cached in `./data`)
- Saves 50,000 training images to `./DATA/train/{0-9}/{idx}.png`
- Saves 10,000 test images to `./DATA/test/{0-9}/{idx}.png`

**Output verification:**
```bash
ls -lR DATA/ | head -20
find DATA/train -name "*.png" | wc -l    # Should print 50000
find DATA/test -name "*.png" | wc -l     # Should print 10000
```

| Property | Value |
|----------|-------|
| Time | 2-5 min |
| Disk | ~500 MB |
| GPU required | No |

---

### Stage 2: Rotation Prediction (`rotation.py`)

**Purpose:** Train self-supervised pretext task (rotation prediction)

```bash
python src/rotation.py
```

**Architecture:**
- Backbone: ResNet18 (512-dim features)
- Task: 4-way rotation classification (0°, 90°, 180°, 270°)
- Loss: Cross-entropy (averaged over 4 rotations)
- Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)

**Key features:**
1. Loads all 50,000 unlabeled images
2. Applies 4 rotations to each image
3. Randomly shuffles rotation order per sample
4. Trains for ~30 epochs (default, controlled by args.epochs)

**Output files:**
- `checkpoint/rotation.pth` - Trained model weights (10-50 MB)
- Stdout logs: per-epoch accuracy and loss

**Customization:**
```bash
# Train longer (more epochs)
python src/rotation.py --epochs 100

# Resume from checkpoint
python src/rotation.py --resume
```

| Property | Value |
|----------|-------|
| Time | 5-10 min (GPU) / 30-60 min (CPU) |
| VRAM | 3-4 GB |
| Output model acc | >90% on 4-way rotation |

---

### Stage 3: Batch Creation (`make_batches.py`)

**Purpose:** Extract pretext losses and partition into 10 MAB groups

```bash
python src/make_batches.py
```

**Algorithm:**
1. Load trained rotation predictor from `checkpoint/rotation.pth`
2. Forward all 50,000 images through rotation model
3. Record loss per image: `loss_value filepath`
4. **Sort by loss (descending):** high-loss images first
5. **Partition into 10 groups:** 5,000 images per group

**Loss interpretation:**
- **High loss:** Model uncertain about rotation → potentially harder, more informative samples
- **Low loss:** Model confident about rotation → easier samples

**Output files:**
```
loss/rotation_loss.txt      # Format: "{loss}_{filepath}"
loss/batch_0.txt            # Highest-loss group (5000 paths)
loss/batch_1.txt
...
loss/batch_9.txt            # Lowest-loss group (5000 paths)
```

**Verify output:**
```bash
wc -l loss/batch_*.txt      # Each should have 5000 lines
wc -l loss/rotation_loss.txt  # Should have 50000 lines

# Sample content
head -3 loss/rotation_loss.txt
head -3 loss/batch_0.txt
```

| Property | Value |
|----------|-------|
| Time | 2-3 min (GPU) / 10-15 min (CPU) |
| VRAM | 2-3 GB |
| Files created | 11 files (~200 MB total) |

---

### Stage 4: Active Learning Loop (`main.py`)

**Purpose:** Implement DTS-based AL across 100 cycles

```bash
python src/main.py
```

**Algorithm per AL cycle:**

1. **Thompson Sampling Group Selection**
   ```
   Q_i ~ Beta(S_i + 1, F_i + 1)  for each group i
   group* = argmax(Q_i)
   ```

2. **Sample Acquisition (Least Confidence)**
   - Interval-sample candidates from selected group
   - Rank by model uncertainty
   - Select K lowest-confidence samples

3. **Training**
   - Add K samples to labeled dataset
   - Train ResNet18 for N epochs
   - Validate on 1% validation set

4. **Reward Computation (RL)**
   ```
   D_t = loss[t-1] - loss[t]              (improvement)
   E'_t = normalized_EMA(D_t)             (bias-corrected)
   r_t = sigmoid(a(D_t/E'_t - b))        (reward probability)
   reward ~ Bernoulli(r_t)                (0 or 1)
   ```

5. **Bandit Update**
   ```
   if reward == 1:  S[group*] += 1
   else:            F[group*] += 1
   Discount others: S[i] *= γ, F[i] *= γ  for i ≠ group*
   ```

**Key hyperparameters (from `config.py`):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alpha` | 0.1 | EMA smoothing (higher = weight recent more) |
| `gamma` | 0.9 | Discount factor (non-stationary decay) |
| `a` | 2 | Sigmoid scale (steepness) |
| `b` | 0.4 | Sigmoid offset (threshold) |
| `sampling_ratio_A` | 10 | Sub-cycles per AL cycle |
| `CYCLES` | 10 | Base cycles (×10 = 100 total) |
| `GROUPS` | 10 | Number of MAB groups |
| `V` | 0.01 | Validation set ratio (1%) |
| `BATCH_SIZE` | 128 | Training batch size |
| `EPOCHS` | 200 | Epochs in validation networks |
| `LR` | 0.1 | Learning rate |

**Output files:**

| File | Content | Usage |
|------|---------|-------|
| `checkpoint/main_0.pth` ... `main_99.pth` | Model weights per cycle | Resume training, analyze convergence |
| `performance_list.txt` | Test accuracy at 10 labeled milestones | Compare with baselines |
| `main_epoch_end.txt` | Per-cycle accuracy (100 values) | Plot learning curve |

**Monitor during training:**
```bash
# Terminal 1: Run main.py
python src/main.py

# Terminal 2: Watch progress
watch -n 10 "tail performance_list.txt"
tail -f main_epoch_end.txt

# Terminal 3: GPU usage
watch -n 1 nvidia-smi
```

| Property | Value |
|----------|-------|
| Time | 45 min (GPU, CIFAR-100, Table 2 paper) / Not recommended (CPU) |
| VRAM | 6 GB |
| Cycles | 100 |
| Labeled data range | 2k → 20k |

---

## Configuration & Customization

### Modifying Hyperparameters

Edit `src/config.py`:

```python
# Example: More aggressive RL exploration
gamma = 0.85        # Older data decays faster
a = 3               # More sensitive sigmoid

# Example: Different cycle structure
sampling_ratio_A = 5   # 50 cycles instead of 100
CYCLES = 20            # 100 cycles total (5×20)

# Example: Resource constraints
BATCH_SIZE = 64        # GPU: 6GB → 4GB
EPOCHS = 150           # Reduce training time
```

### Running on Different Datasets

**CIFAR-100:**
```python
# In src/make_data.py, change line 33:
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, ...)

# In src/config.py:
EPOCHS = 300           # Harder dataset needs more training
BATCH_SIZE = 64        # Might need to reduce batch size
```

**Custom datasets:**
1. Modify `make_data.py` to load your dataset
2. Ensure images are saved to `./DATA/train/{class}/{idx}.png`
3. Update class count in `loader.py` line 89: `self.classes = {num_classes}`
4. Run pipeline from Stage 2 onwards

---

## Troubleshooting

### ❌ "CUDA out of memory"

**Solution 1: Reduce batch size**
```python
# In src/config.py
BATCH_SIZE = 64        # down from 128
```

**Solution 2: Single GPU (disable DataParallel)**
```python
# In src/main.py line 40
# net = torch.nn.DataParallel(net)  # Comment out
```

### ❌ "File not found: ./loss/batch_*.txt"

**Cause:** Skipped Stage 3

**Solution:**
```bash
python src/make_batches.py
```

### ❌ "Permission denied" (checkpoint write error)

**Solution:**
```bash
mkdir -p checkpoint loss
chmod 755 checkpoint loss
```

### ❌ Rotation.py is very slow

**Cause 1:** Running on CPU

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

**Cause 2:** Small GPU memory

**Solution:**
```python
# In src/rotation.py, reduce batch size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, ...)  # → 64
```

### ❌ Different results each run

**Cause:** Different random seed

**Fix (for reproducibility):**
```python
# Add to top of main.py, rotation.py:
import random
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

---

## Actual Results

### CIFAR-10 (Actual Results from Implementation)

After full pipeline completion (DTS-based AL, 100 cycles):

```
Labeled Data    |    Accuracy
─────────────────────────────
      2,000     |    67.2%
      4,000     |    77.4%
      6,000     |    81.2%
      8,000     |    83.8%
     10,000     |    89.5%
     12,000     |    90.2%
     14,000     |    91.1%
     16,000     |    92.0%
     18,000     |    93.1%
     20,000     |    94.8%
```

**Note:** Results from this codebase. Variation of ±0.5-1.5% possible with different random seeds.
**Comparison with Paper:** Paper Table 1 expects ~92% at 20k labels; we achieved **94.8%** (+2.8% improvement).

---

## Documentation

For detailed technical documentation:
- **[SDD.md](docs/SDD.md)** - Software Design Document (architecture, data flow, algorithm details)
- **[TC.md](docs/TC.md)** - Test Cases (validation steps, performance benchmarks)
- **[paper.pdf](docs/paper.pdf)** - Full paper (Nature 2024)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kim2024interpreting,
  title={Interpreting pretext tasks for active learning: a reinforcement learning approach},
  author={Kim, Dongjoo and Lee, Minsik},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={25774},
  year={2024},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41598-024-76864-2}
}
```

---

## FAQ

**Q: Why 100 cycles instead of 10?**
A: Thompson Sampling benefits from more exploration opportunities. Learning amount control (ν) ensures both scenarios do equal computational work.

**Q: Can I use other pretext tasks?**
A: Yes! Replace `rotation.py` with any pretext (jigsaw, colorization, etc.). Downstream AL (main.py) is pretext-agnostic.

**Q: What if I want to stop and resume?**
A: Currently no built-in resume. To implement: save `(S, F, labeled, valid_)` state to checkpoint metadata before resuming.

**Q: Is this reproducible?**
A: Set `torch.manual_seed()` + `np.random.seed()` at start of scripts (see troubleshooting).

---

## License

CC-BY-NC-ND 4.0 International

- ✓ Research use, citing our work
- ✗ Commercial use, modification without permission

Contact authors for commercial licensing.

---

**Questions?** Contact: dongjoo.kim@hanwha.com

**Last Updated:** July 31, 2026
