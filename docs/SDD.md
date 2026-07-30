# Software Design Document (SDD)
## Interpreting Pretext Tasks for Active Learning: A RL Approach

### 1. System Overview

This document describes the software architecture for implementing Discounted Thompson Sampling (DTS) based Active Learning with Self-Supervised pretext tasks.

**Publication Reference:**
- Title: "Interpreting Pretext Tasks for Active Learning: A Reinforcement Learning Approach"
- Authors: Dongjoo Kim, Minsik Lee
- Venue: Scientific Reports (Nature), Vol 14, Article 25774, October 2024

---

### 2. System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  make_data  │ ───► │  rotation.py │ ───► │make_batches  │ ───► │   main.py    │
│             │      │              │      │              │      │              │
│ CIFAR→PNG   │      │ ResNet18     │      │ Group by     │      │ Thompson     │
│             │      │ 4-way        │      │ loss (top10) │      │ Sampling+RL  │
└─────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                            │                     │                      │
                            ▼                     ▼                      ▼
                      loss/rotation_loss.txt  loss/batch_*.txt  checkpoint/main_*.pth
                                                                 performance_list.txt
```

---

### 3. Module Descriptions

#### 3.1 **make_data.py** - Data Preparation
**Purpose:** Convert CIFAR-10 dataset to PNG format for efficient I/O

**Input:**
- CIFAR-10 dataset (downloaded from torchvision)

**Output:**
- `./DATA/train/{class}/{idx}.png` - 50,000 training images
- `./DATA/test/{class}/{idx}.png` - 10,000 test images

**Key Function:**
- `save_dataset`: Custom PyTorch Dataset wrapper that saves PIL images to disk

**Hyperparameters:**
- None (only directory structure management)

---

#### 3.2 **rotation.py** - Rotation Prediction (Pretext Task)
**Purpose:** Learn image representations using rotation prediction without labels

**Architecture:**
- Backbone: ResNet18
- Output layer: 4-way classifier (0°, 90°, 180°, 270°)
- Loss: Cross-entropy loss averaged over 4 rotations

**Input:**
- `./DATA/train/*/*.png` (all 50k unlabeled training images)

**Output:**
- `./checkpoint/rotation.pth` - trained rotation predictor weights
- Training stdout: per-epoch loss/accuracy

**Key Process:**
1. Load images and apply 4 rotations
2. Forward pass through 4-head network
3. Average cross-entropy loss across 4 rotation classes
4. Backprop and SGD update

**Hyperparameters:**
- LR: 0.1
- Momentum: 0.9
- Weight decay: 5e-4
- Epochs: Determined at runtime (typically 30-100)
- Batch size: 256

---

#### 3.3 **make_batches.py** - Group Creation by Loss
**Purpose:** Extract pretext task loss and partition unlabeled pool into 10 groups

**Algorithm:**
1. Load trained rotation predictor
2. Forward all 50k images through rotation model
3. Record loss value per image: `loss_value_filepath`
4. Sort images by loss (descending - high loss first)
5. Partition into 10 groups of 5000 images each

**Input:**
- `./checkpoint/rotation.pth` (pretext model)
- `./DATA/train/*/*.png` (all 50k images)

**Output:**
- `./loss/rotation_loss.txt` - Format: `{loss}_{filepath}\n`
- `./loss/batch_{0-9}.txt` - Each group contains 5000 image paths

**Key Function:**
- `test()`: Evaluate rotation model on all images, compute and log loss per image

**Note:** Groups ordered by descending loss → difficult/ambiguous images first

---

#### 3.4 **main.py** - Active Learning Loop with Thompson Sampling
**Purpose:** Implement DTS-based AL: adaptively select groups and train classifier

**Core Algorithm (per AL cycle):**

1. **Group Selection (Thompson Sampling):**
   - Sample Q-value for each group i: `Q_i ~ Beta(S_i + 1, F_i + 1)`
   - Select best group: `argmax(Q_i)`

2. **Sample Acquisition:**
   - Interval sampling from selected group (every 10th sample)
   - Least-confidence sampling: select K lowest-confidence samples
   - Add to labeled dataset

3. **Model Training:**
   - Train ResNet18 classifier on labeled data for N epochs
   - Validate on held-out validation set
   - Track epoch-wise loss improvement

4. **Reward Computation:**
   ```
   D_t = loss_{t-1} - loss_t                    (loss improvement)
   E_t = α·D_{t-1} + (1-α)·E_{t-1}            (EMA term)
   E'_t = E_t / (1 - (1-α)^{t-1})            (bias correction)
   r_t = σ(a(D_t/E'_t - b))                  (sigmoid-normalized reward)
   reward ~ Bernoulli(r_t)
   ```

5. **Bandit Update:**
   ```
   if reward == 1:
       S[group] = γ·S[group] + 1
   else:
       F[group] = γ·F[group] + 1
   
   for all i ≠ group:  (discount non-selected)
       S[i] = γ·S[i]
       F[i] = γ·F[i]
   ```

**Input:**
- `./loss/batch_*.txt` (10 groups of image paths)
- Previous checkpoint (if cycle > 0)

**Output:**
- `./checkpoint/main_{0-99}.pth` (100 AL cycles)
- `./performance_list.txt` (test accuracy at labeled milestones)
- `./main_epoch_end.txt` (per-cycle test accuracy)

**Key Hyperparameters:**
| Parameter | Value | Role |
|-----------|-------|------|
| α (alpha) | 0.1 | EMA smoothing parameter |
| γ (gamma) | 0.9 | Discount factor (non-stationary) |
| a | 2 | Sigmoid scale for reward |
| b | 0.4 | Sigmoid threshold for reward |
| GROUPS | 10 | Number of groups |
| V | 0.01 | Validation set ratio |
| sampling_ratio_A | 10 | Sub-cycles per AL cycle |
| CYCLES | 10 | Base AL cycles (multiplied by sampling_ratio_A) |
| EPOCHS | 200 | Epochs per cycle in separate validation network |

---

### 4. Data Flow & Variables

#### Learning Amount (ν) Concept
The paper introduces **learning amount** to fairly compare different AL strategies:
- Baseline (Scenario B): 10 cycles, larger batches, fewer epochs each
- Proposed (Scenario A): 100 cycles (10×10), smaller batches, more epochs each
- Both scenarios maintain equal total **learning amount**: ν

**Key Functions (refactored):**

1. **compute_per_cycle_labeling_data(labeling_data_per_cycle, num_sub_cycles)**
   - Input: [2000] × 10 cycles, 10 sub-cycles
   - Output: [200, 200, ..., 200] (100 cycles)
   - Logic: Distribute per-cycle samples evenly across sub-cycles + handle remainder

2. **compute_cumulative_labeling_data(per_cycle)**
   - Input: [200, 200, ..., 200] (per-cycle)
   - Output: [200, 400, 600, ..., 20000] (cumulative)
   - Logic: Running sum of per-cycle counts

3. **compute_learning_amount_A(learning_amount_B, labeling_data_cycle_A, per_cycle_data)**
   - Input: Baseline amounts, cumulative labeled counts, per-cycle data
   - Output: [learning_amt_0, learning_amt_1, ..., learning_amt_9]
   - Logic: Sum of (iterations × cumulative_data) per base cycle

4. **get_learning_amount_A()**
   - Orchestration function: calls all 3 compute functions
   - Returns: (learning_amount_A, per_cycle, cumulative)

5. **adjust_labeling_for_validation(labeling_data_cycle_A, per_cycle, valid_count)**
   - Purpose: Subtract validation samples from labeled budget
   - Logic: If validation > first cycle, spread subtraction across multiple cycles
   - Returns: Adjusted (cumulative, per_cycle)

**Calculation Flow:**
```
Input: labeling_data_per_cycle = [2000] × 10
         ↓
compute_per_cycle_labeling_data()
         ↓
per_cycle = [200] × 100
         ↓
compute_cumulative_labeling_data()
         ↓
cumulative = [200, 400, ..., 20000]
         ↓
compute_learning_amount_A()
         ↓
learning_amount_A ≈ learning_amount_B (fairness)
         ↓
adjust_labeling_for_validation()
         ↓
Final: cumulative_adjusted, per_cycle_adjusted
```

#### Validation Set Creation
- **Size:** 1% of total AL budget split across 10 groups
- **Selection:** Interval sampling from each group (highest-loss region)
- **Purpose:** Compute rewards + early stopping during training
- **Exclusion:** Never included in labeled dataset

---

### 5. Thompson Sampling Details

**Beta Distribution (MAB):**
- State: (S_i, F_i) = (success count, failure count) for group i
- Sampling: `Q_i ~ Beta(S_i + 1, F_i + 1)` (conjugate prior)
- Selection: `argmax(Q_i)` (greedy exploitation)
- Discount: γ = 0.9 (prioritize recent successes)

**Bernoulli Trial:**
- Reward probability: `r_t = σ(2(D_t/E'_t - 0.4))`
- Trial outcome: `reward ~ Bernoulli(r_t) ∈ {0, 1}`
- Update: S/F incremented only if group was selected

**Interpretation:**
- Group with high recent success rate → high probability of selection
- DTS decay prevents stale information → adapts to non-stationary AL

---

### 6. Training Loop Details

**Per-cycle Training:**
1. Load previous checkpoint (if exists)
2. Create DataLoaders: trainset, validset (from labeled + valid_)
3. For each epoch:
   - Train step: forward, backward, SGD update
   - Valid step: evaluate on validation set
   - Check best accuracy → save checkpoint
4. Early stopping condition (optional): if epoch == total_epoch-1 and no improvement

**Performance Checkpoint:**
- Separate network initialized at each labeled data milestone
- Trained for 200 epochs to assess true dataset quality
- Result logged in `performance_list.txt`

---

### 7. Key Implementation Notes

**Critical Functions:**

| Function | Purpose | Returns |
|----------|---------|---------|
| `get_learning_amount_A()` | Compute sub-cycle sample counts & learning amounts | (amounts, per_cycle, cumulative) |
| `get_epoch_cycles_A()` | Adjust epochs per sub-cycle to match baseline learning amount | adjusted_epochs |
| `group_search_sampling_data()` | Thompson sampling → group selection → confident samples | (group, updated_params) |
| `get_labels()` | Least-confidence sampling: top-K lowest confidence scores | selected_samples |
| `get_ema()` | Exponential moving average with bias correction | (EMA, EMA_bias_corrected) |
| `calculate_update_reward()` | Bernoulli trial + S/F/discount update | (success_list, failure_list) |

**Interval Sampling:**
- Use: Every 10th sample for uncertainty computation (cost reduction)
- Condition: If selected group has limited unsampled data
- Purpose: Trade-off between computational cost and coverage

**Validation Data:**
- Created before main AL loop: `sampling_valid()`
- Interval sampled from batch_*.txt files (1% each group)
- Excluded from labeled dataset throughout training

---

### 8. Output Files & Interpretation

| File | Format | Meaning |
|------|--------|---------|
| `performance_list.txt` | Comma-separated floats | Test accuracy at each labeled milestone |
| `checkpoint/main_*.pth` | PyTorch state_dict | Trained classifier at each cycle |
| `loss/rotation_loss.txt` | `{loss}_{path}\n` | Pretext loss per image |
| `loss/batch_*.txt` | One filepath per line | Grouped image paths (descending loss) |
| `main_epoch_end.txt` | List format `[acc, acc, ...]` | Per-cycle test accuracy across 100 cycles |

---

### 9. Relationship to Paper

**Paper Section → Code Mapping:**

| Paper Concept | Code Implementation |
|--------------|------------------|
| Algorithm 1 (DTS) | `group_search_sampling_data()` + `calculate_update_reward()` |
| Reward Function (Eq. 3) | `get_ema()` + `calculate_update_reward()` |
| Learning Amount (ν) | `get_learning_amount_A()` + `get_epoch_cycles_A()` |
| Validation Set (1%) | `sampling_valid()` |
| Least-confidence sampling | `get_labels()` using softmax probabilities |
| Discount Thompson Sampling | `beta.rvs()` with S/F decay (γ=0.9) |

---

### 10. Configuration

All hyperparameters are defined in `config.py`:
```python
alpha = 0.1
gamma = 0.9
a = 2
b = 0.4
sampling_ratio_A = 10
GROUPS = 10
V = 0.01
EPOCHS = 200
CYCLES = 10
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
```

These values are fixed across all 5 datasets (CIFAR-10/100, SVHN, Caltech-101, ImageNet-64) as reported in the paper.

