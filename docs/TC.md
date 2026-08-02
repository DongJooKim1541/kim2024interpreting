# Test Cases (TC) Document
## Interpreting Pretext Tasks for Active Learning

**Publication Reference:**
- Title: "Interpreting Pretext Tasks for Active Learning: A Reinforcement Learning Approach"
- Authors: Dongjoo Kim, Minsik Lee
- Venue: Scientific Reports (Nature), Vol 14, Article 25774, October 2024
- DOI: https://doi.org/10.1038/s41598-024-76864-2

---

## 1. Test Setup Overview

This document defines test cases for validating the DTS-based AL implementation against the paper's methodology.

**Test Environment:**
- Dataset: CIFAR-10 (50k train, 10k test)
- Device: CUDA (or CPU fallback)
- Python: 3.7+
- Key Libraries: torch, torchvision, scipy

---

## 2. Unit Test Cases (Refactored Functions)

### 2.0 Test Case: Learning Amount Refactored Functions

**Test ID:** TC-000-LEARNING-REFACTORED

**Objective:** Validate refactored learning amount computation functions

#### Sub-test: compute_per_cycle_labeling_data()

**Input:**
```python
labeling_data_per_cycle = [2000] × 10
num_sub_cycles = 10
```

**Expected Output:**
```python
per_cycle = [200] × 100  # 10 base cycles × 10 sub-cycles
```

**Validation:**
```python
assert len(per_cycle) == 100
assert sum(per_cycle) == 20000
assert all(x == 200 for x in per_cycle)
```

**Success Criterion:**
- Length: exactly 100 elements ✓
- Sum: equals 20,000 ✓
- Distribution: uniform (no remainder handling needed for even split) ✓

#### Sub-test: compute_cumulative_labeling_data()

**Input:**
```python
per_cycle = [200] × 100
```

**Expected Output:**
```python
cumulative = [200, 400, 600, ..., 20000]  # 100 elements
```

**Validation:**
```python
assert len(cumulative) == 100
assert cumulative[0] == 200
assert cumulative[-1] == 20000
assert all(cumulative[i] < cumulative[i+1] for i in range(99))  # Strictly increasing
```

**Success Criterion:**
- Length: 100 ✓
- First element: 200 ✓
- Last element: 20,000 ✓
- Monotonically increasing ✓

#### Sub-test: compute_learning_amount_A()

**Input:**
```python
learning_amount_B = [312500, 625000, 937500, ...]  # Baseline (10 cycles)
labeling_data_cycle_A = [200, 400, 600, ...]  # Cumulative (100 cycles)
per_cycle_data = [200] × 100
```

**Expected Output:**
```python
learning_amount_A ≈ [learning_amount_B[0], learning_amount_B[1], ...]
# Fairness: sum(learning_amount_A) ≈ sum(learning_amount_B)
```

**Validation:**
```python
assert len(learning_amount_A) == 10  # One per base cycle
total_A = sum(learning_amount_A)
total_B = sum(learning_amount_B)
assert 0.95 <= total_A / total_B <= 1.05  # Within 5% fairness tolerance
```

**Success Criterion:**
- Length: 10 base cycles ✓
- Fairness ratio: 0.95-1.05 ✓

#### Sub-test: adjust_labeling_for_validation()

**Scenario A: Validation < first cycle size**

**Input:**
```python
labeling_data_cycle_A = [200, 400, 600, ...]
per_cycle = [200] × 100
valid_count = 100  # 1% of 10,000 labeled budget
```

**Expected Output:**
```python
adjusted_cumulative = [100, 300, 500, ...]  # First element reduced by 100
adjusted_per_cycle = [100, 200, 200, ...]  # First element reduced by 100
```

**Validation:**
```python
assert adjusted_cumulative[0] == 200 - 100
assert adjusted_per_cycle[0] == 200 - 100
assert sum(adjusted_per_cycle) == 20000 - 100
```

**Scenario B: Validation > first few cycles**

**Input:**
```python
labeling_data_cycle_A = [200, 400, 600, ...]
per_cycle = [200] × 100
valid_count = 600  # Spans 3 cycles worth
```

**Expected Output:**
```python
adjusted_cumulative = [0, 0, 0, 600, 800, ...]  # First 3 zeroed, 4th reduced
adjusted_per_cycle = [0, 0, 0, 0, 200, ...]    # First 3 zeroed, 4th reduced
```

**Validation:**
```python
delete_cycles = 600 // 200  # = 3
assert adjusted_cumulative[0:3] == [0, 0, 0]
assert adjusted_cumulative[3] == 600 - (600 % 200)
```

**Success Criterion:**
- Both scenarios handle validation subtraction ✓
- Total labeled: reduced by valid_count ✓

---

## 2.1 Module-Level Test Cases (Original + Refactored)

### 2.1 Test Case: Data Preparation (make_data.py - Refactored)

**Test ID:** TC-001-DATA-REFACTORED

**Objective:** Validate refactored data preparation functions

#### Sub-test: prepare_directories()

**Purpose:** Create required directory structure atomically

**Input:** None (side effect: creates directories)

**Expected Output:**
```
./DATA/
├── train/
└── test/
```

**Validation:**
```python
prepare_directories()
assert os.path.isdir('./DATA')
assert os.path.isdir('./DATA/train')
assert os.path.isdir('./DATA/test')
```

**Success Criterion:**
- All 3 directories exist ✓
- Idempotent (can be called multiple times safely) ✓

#### Sub-test: convert_cifar_to_png()

**Purpose:** Iterate dataset and save all images via wrapper

**Input:**
```python
train_dataset = save_dataset(trainset, split='train')  # 50,000 items
test_dataset = save_dataset(testset, split='test')     # 10,000 items
```

**Expected Output:**
- 50,000 PNG files in ./DATA/train/{0-9}/
- 10,000 PNG files in ./DATA/test/{0-9}/

**Validation:**
```python
convert_cifar_to_png(train_dataset, test_dataset)
assert len(glob.glob('./DATA/train/*/*.png')) == 50000
assert len(glob.glob('./DATA/test/*/*.png')) == 10000
for class_id in range(10):
    assert len(glob.glob(f'./DATA/train/{class_id}/*.png')) == 5000
    assert len(glob.glob(f'./DATA/test/{class_id}/*.png')) == 1000
```

**Success Criterion:**
- Training images: 50,000 ✓
- Test images: 10,000 ✓
- Per-class distribution: 5000 train, 1000 test per class ✓

---

### 2.2 Test Case: Batch Creation (make_batches.py - Refactored)

**Test ID:** TC-003-BATCHES-REFACTORED

**Objective:** Validate refactored loss parsing and group creation

#### Sub-test: parse_loss_file()

**Input File:** loss/rotation_loss.txt
```
1.234_./DATA/train/3/4567.png
0.956_./DATA/train/5/8901.png
1.567_./DATA/train/2/3456.png
...
```

**Expected Output:**
```python
loss_values = ['1.234', '0.956', '1.567', ...]  (str format, 50,000 items)
image_paths = ['./DATA/train/3/4567.png', ...]  (50,000 items)
```

**Validation:**
```python
loss_values, image_paths = parse_loss_file('loss/rotation_loss.txt')
assert len(loss_values) == 50000
assert len(image_paths) == 50000
assert all(isinstance(loss, str) for loss in loss_values)
assert all(path.endswith('.png') for path in image_paths)
assert all('_' not in path for path in image_paths)  # Correctly split
```

**Success Criterion:**
- Both lists have 50,000 items ✓
- Loss values are strings (preserves precision) ✓
- Image paths are cleaned (no '_' prefix) ✓

#### Sub-test: create_groups()

**Input:**
```python
loss_values = ['1.234', '0.956', '1.567', ...]  (50,000)
image_paths = ['./DATA/train/3/4567.png', ...]  (50,000)
num_groups = 10
samples_per_group = 5000
```

**Expected Behavior:**
1. Convert loss_values to float
2. Sort by loss (descending)
3. Partition into 10 groups
4. Write each group to loss/batch_{i}.txt

**Expected Output Files:**
```
loss/batch_0.txt  (5000 lines, highest loss images)
loss/batch_1.txt  (5000 lines)
...
loss/batch_9.txt  (5000 lines, lowest loss images)
```

**Validation:**
```python
create_groups(loss_values, image_paths, num_groups=10, samples_per_group=5000)

# Check file existence and line count
for i in range(10):
    filepath = f'loss/batch_{i}.txt'
    assert os.path.exists(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 5000

# Check sorting: average loss decreases across groups
avg_losses = []
for i in range(10):
    with open(f'loss/batch_{i}.txt', 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    # Find corresponding losses from original loss_values
    indices = [image_paths.index(p) for p in paths]
    losses = [float(loss_values[idx]) for idx in indices]
    avg_losses.append(np.mean(losses))

assert all(avg_losses[i] >= avg_losses[i+1] for i in range(9))  # Descending
```

**Success Criterion:**
- 10 group files created ✓
- Each has exactly 5000 lines ✓
- Average loss decreases across groups (descending sort) ✓
- No duplicate paths across groups ✓

---

### 2.3 Test Case: Data Preparation (make_data.py)

**Test ID:** TC-001-DATA

**Objective:** Validate CIFAR-10 → PNG conversion preserves data integrity

**Preconditions:**
- Internet connection (for CIFAR-10 download)
- Disk space: ~500 MB

**Input:**
- CIFAR-10 dataset via torchvision.datasets.CIFAR10

**Expected Output:**
- Directory structure:
  ```
  ./DATA/
  ├── train/
  │   ├── 0/ (1000 images)
  │   ├── 1/ (1000 images)
  │   └── ... (10 classes)
  └── test/
      ├── 0/ (1000 images)
      └── ... (10 classes)
  ```
- Total: 50,000 training + 10,000 test PNG files
- Image format: RGB PIL Image (32×32)

**Validation Steps:**
1. Count files: `len(train) == 50000` ✓
2. Class distribution: `all(len(class_dir) == 5000 for class in 0-9)` ✓
3. Image format check: All files readable by PIL ✓
4. Image shape: All images are 32×32 pixels ✓

**Success Criterion:**
```python
assert len(glob.glob('./DATA/train/*/*.png')) == 50000
assert len(glob.glob('./DATA/test/*/*.png')) == 10000
```

---

### 2.4 Test Case: Rotation Predictor Training (rotation.py - Fixed)

**Test ID:** TC-002-ROTATION-FIXED

**Objective:** Validate rotation prediction with corrected transform

**Change Note:** trainset now uses transform_train (not transform_test)

#### Key Validation: Data Augmentation Applied

**Input:**
```python
trainset = RotationLoader(is_train=True, transform=transform_train)
# transform_train includes: RandomCrop(32, 4), RandomHorizontalFlip(), ToTensor(), Normalize()
```

**Expected Output:**
- Training images have augmentations (random crops, flips)
- Model sees diverse transformations per epoch
- Rotation accuracy > 90%

**Validation Steps:**
1. Verify transform_train is used:
   ```python
   assert trainset.transform == transform_train
   ```

2. Check augmentation application:
   ```python
   # Sample batch should show variation per epoch
   img1, _, _, _, _, _, _, _ = trainset[0]  # First sample
   img2, _, _, _, _, _, _, _ = trainset[0]  # Same sample, different augmentation
   assert not torch.allclose(img1, img2)  # Should differ due to augmentation
   ```

3. Verify model training improvement:
   ```python
   # After training for 30 epochs
   checkpoint = torch.load('./checkpoint/rotation.pth')
   assert checkpoint['acc'] > 90.0  # Rotation prediction accuracy
   ```

**Success Criterion:**
- transform_train applied to trainset ✓
- Augmentation produces variation ✓
- Final accuracy > 90% ✓

---

### 2.3 Test Case: Rotation Predictor Training (rotation.py - Original)

**Test ID:** TC-002-ROTATION

**Objective:** Validate rotation prediction model trains successfully

**Preconditions:**
- `make_data.py` completed successfully
- GPU available or CPU mode enabled

**Input:**
- 50,000 images from `./DATA/train/`

**Expected Output:**
- Checkpoint file: `./checkpoint/rotation.pth`
- Model performance: >90% accuracy on rotation classification (4-way)

**Key Metrics:**
| Metric | Expected Range |
|--------|-----------------|
| Training loss (final epoch) | < 0.1 |
| Accuracy on training set | > 90% |
| Checkpoint file size | 10-50 MB |

**Validation Steps:**
1. Load checkpoint: `torch.load('./checkpoint/rotation.pth')['net']` ✓
2. Model inference: Forward 100 random images → 4×softmax outputs ✓
3. Loss computation: CrossEntropyLoss on 4-way classification ✓

**Success Criterion:**
```python
# Model can be loaded and inferred
checkpoint = torch.load('./checkpoint/rotation.pth')
assert 'net' in checkpoint
assert 'acc' in checkpoint
assert checkpoint['acc'] > 90.0  # Accuracy > 90%
```

---

### 2.3 Test Case: Batch Creation by Loss (make_batches.py)

**Test ID:** TC-003-BATCHES

**Objective:** Validate loss extraction and group partitioning

**Preconditions:**
- `rotation.py` completed successfully
- Checkpoint `./checkpoint/rotation.pth` exists

**Input:**
- Rotation model weights
- 50,000 images from training set

**Expected Output:**
- Loss file: `./loss/rotation_loss.txt` (50,000 lines)
- 10 batch files: `./loss/batch_0.txt` ... `./loss/batch_9.txt` (5,000 lines each)

**File Format Validation:**
```
rotation_loss.txt:  {float}_{filepath}\n
batch_*.txt:        {filepath}\n
```

**Key Statistics:**
| Metric | Expected |
|--------|----------|
| Lines in rotation_loss.txt | 50,000 |
| Lines per batch file | 5,000 ± 0 |
| Loss range | [0, ~4.0] |
| Groups sorted by loss | descending (high → low) |

**Validation Steps:**
1. Count lines in each file ✓
2. Parse loss values: all convertible to float ✓
3. Verify sorting: first batch has higher avg loss than last batch ✓
4. No duplicate filepaths across groups ✓

**Success Criterion:**
```python
for i in range(10):
    with open(f'./loss/batch_{i}.txt', 'r') as f:
        lines = f.readlines()
    assert len(lines) == 5000
    assert all(line.endswith('\n') for line in lines)

# Verify no overlaps
all_paths = set()
for i in range(10):
    with open(f'./loss/batch_{i}.txt', 'r') as f:
        paths = {line.strip() for line in f.readlines()}
    assert len(all_paths & paths) == 0  # No overlap
    all_paths |= paths

assert len(all_paths) == 50000  # All images covered
```

---

### 2.4 Test Case: Learning Amount Calculation (main.py)

**Test ID:** TC-004-LEARNING-AMOUNT

**Objective:** Validate learning amount fairness between scenarios

**Preconditions:**
- Configuration: CYCLES=10, sampling_ratio_A=10, BATCH_SIZE=128

**Input:**
- labeled_data_per_AL_cycle: [2000, 4000, ..., 20000]
- labeling_data_per_cycle: [2000] × 10

**Expected Behavior:**
```
Scenario B (baseline): 10 cycles × (50k / 10) ≈ 15.84M iterations
Scenario A (proposed): 100 cycles, adjusted epochs ≈ 15.84M iterations
```

**Key Invariants:**
1. `len(labeling_data_cycle_A_per_cycle) == 100` (10 × sampling_ratio_A)
2. `sum(labeling_data_cycle_A_per_cycle) == 20000` (total labeled)
3. `learning_amount_per_cycle_A ≈ learning_amount_per_cycle_B` (fairness)
4. Early cycles have more epochs than late cycles

**Validation Steps:**
1. Call `get_learning_amount_A()` → verify output shapes ✓
2. Call `get_learning_amount_B()` → baseline amounts ✓
3. Compare: `sum(learning_amount_per_cycle_A) ≈ sum(learning_amount_per_cycle_B)` ✓
4. Epochs per cycle: `epoch_total_cycle_A` should be decreasing trend ✓

**Success Criterion:**
```python
learning_amount_A, per_cycle_A, cumul_A = get_learning_amount_A()
learning_amount_B = get_learning_amount_B()

assert len(per_cycle_A) == 100
assert sum(per_cycle_A) == 20000
assert abs(sum(learning_amount_A) - sum(learning_amount_B)) / sum(learning_amount_B) < 0.05
# Learning amounts within 5% difference
```

---

### 2.5 Test Case: Validation Set Creation (main.py)

**Test ID:** TC-005-VALIDATION

**Objective:** Validate 1% validation set is properly sampled

**Preconditions:**
- Batch files created (TC-003)
- Configuration: V=0.01, GROUPS=10

**Input:**
- 10 batch files, each 5,000 images
- Validation ratio: 1%

**Expected Output:**
- Validation set size: ~500 samples (1% of 50,000)
- Evenly distributed: ~50 samples per group

**Validation Steps:**
1. Call `sampling_valid(labeled_data_per_AL_cycle)` ✓
2. Length check: `len(valid_) ≈ 500` ✓
3. Distribution: `all(45 ≤ len(valid_) // GROUPS ≤ 55)` ✓
4. Format: All paths end with `.png` ✓

**Success Criterion:**
```python
valid_ = sampling_valid(labeled_data_per_AL_cycle)
assert 450 <= len(valid_) <= 550  # ~500 ± 10%
assert all('.png' in path for path in valid_)
```

---

## 3. Integration Test Cases

### 3.0 Test Case: Learning Amount Pipeline (End-to-End Refactored)

**Test ID:** TC-INT-000

**Objective:** Validate complete learning amount computation flow with all refactored functions

**Input Setup:**
```python
from config import *
labeled_data_per_AL_cycle = list(range(2000, 20001, 2000))  # [2000, 4000, ..., 20000]
labeling_data_per_cycle = [2000] * sampling_ratio_A         # [2000] × 10

learning_amount_per_cycle_B = get_learning_amount_B()
```

**Step 1: Compute per-cycle distribution**
```python
per_cycle = compute_per_cycle_labeling_data(labeling_data_per_cycle, sampling_ratio_A)
# Expected: [200] × 100

assert len(per_cycle) == 100
assert sum(per_cycle) == 20000
```

**Step 2: Compute cumulative distribution**
```python
cumulative = compute_cumulative_labeling_data(per_cycle)
# Expected: [200, 400, 600, ..., 20000]

assert len(cumulative) == 100
assert cumulative[-1] == 20000
assert cumulative == sorted(cumulative)  # Strictly increasing
```

**Step 3: Compute learning amounts**
```python
learning_amount_A = compute_learning_amount_A(learning_amount_per_cycle_B, cumulative, per_cycle)
# Expected: [learning_amt_0, learning_amt_1, ..., learning_amt_9]

assert len(learning_amount_A) == CYCLES
total_A = sum(learning_amount_A)
total_B = sum(learning_amount_per_cycle_B)
fairness_ratio = total_A / total_B

assert 0.95 <= fairness_ratio <= 1.05, f"Fairness check failed: {fairness_ratio}"
```

**Step 4: Adjust for validation set**
```python
valid_ = sampling_valid(labeled_data_per_AL_cycle)  # ~200 samples
adjusted_cumulative, adjusted_per_cycle = adjust_labeling_for_validation(cumulative, per_cycle, valid_)

assert sum(adjusted_per_cycle) == 20000 - len(valid_)
assert adjusted_cumulative[-1] == 20000 - len(valid_)
```

**Step 5: Verify epoch computation**
```python
epoch_total_cycle_A = get_epoch_cycles_A(learning_amount_per_cycle_B, learning_amount_A)
# Expected: [epoch_0, epoch_1, ..., epoch_99]

assert len(epoch_total_cycle_A) == 100
assert all(e > 0 for e in epoch_total_cycle_A)
```

**Success Criterion:**
- All 5 steps execute without error ✓
- Fairness: 0.95 ≤ A/B ≤ 1.05 ✓
- Cumulative monotonically increasing ✓
- Total labeled: 20,000 (or adjusted for validation) ✓
- Epochs per cycle: all positive ✓

**Paper Alignment:**
- Scenario A vs B learning amounts: Equal (within 5%) ✓
- Learning amount formula (ν): Correctly implemented ✓

---

### 3.1 Test Case: Thompson Sampling Selection

**Test ID:** TC-INT-001

**Objective:** Validate Thompson Sampling group selection logic

**Input:**
- Beta parameters: (S, F) for 10 groups
- Sample Beta distributions multiple times

**Expected Behavior:**
1. High (S, F) groups have higher selection probability
2. Q-values vary across samples (stochasticity)
3. Discount update: γ=0.9 decay works correctly

**Validation Steps:**
```python
success_list = [10, 5, 3, 1, 0] + [0]*5
failure_list = [0, 0, 0, 0, 0] + [1]*5

Q_list = [beta.rvs(success_list[i]+1, failure_list[i]+1) for i in range(10)]
selected_group = np.argmax(Q_list)

# Group 0 (10 successes) should rarely be outselected
# Repeat 100 times → Group 0 selected ~90% of time (stochastic)
```

**Success Criterion:**
```python
group_counts = [0] * 10
for _ in range(100):
    Q_list = [beta.rvs(success_list[i]+1, failure_list[i]+1) for i in range(10)]
    group_counts[np.argmax(Q_list)] += 1

# Group 0 (highest S) should be selected most frequently
assert group_counts[0] > sum(group_counts[1:])
```

---

### 3.2 Test Case: Reward Function & Bernoulli Update

**Test ID:** TC-INT-002

**Objective:** Validate EMA + sigmoid reward mapping

**Input:**
- Initial EMA: 0
- Loss sequence: [2.0, 1.9, 1.85, 1.83, 1.82, ...] (improving)
- Parameters: α=0.1, a=2, b=0.4

**Expected Behavior:**
1. EMA should smoothly follow loss improvements
2. Reward probability should increase as improvement ratio rises
3. Bernoulli trials should be ~50% success early, ~20% late

**Validation Steps:**
1. Compute EMA for 10 cycles → verify smooth curve ✓
2. Compute reward probability via sigmoid ✓
3. Sample Bernoulli 1000× per cycle → verify probability distribution ✓

**Success Criterion:**
```python
losses = [2.0, 1.9, 1.85, 1.83, 1.82, 1.81, 1.80, 1.80, 1.80, 1.80]
ema, ema_prime = 0, 0
reward_probs = []

for t in range(1, len(losses)):
    D_t = losses[t-1] - losses[t]
    ema, ema_prime = get_ema(alpha=0.1, last_ema=ema, last_loss_diff=D_t, time_step=t)
    
    reward_prob = sigmoid_func(2 * (D_t / ema_prime - 0.4))
    reward_probs.append(reward_prob)
    
    # Early cycles: reward_prob > 0.5
    # Late cycles: reward_prob < 0.5 (diminishing improvement)

assert reward_probs[0] > 0.5  # First improvement should be rewarded
# Reward probability should generally decrease as loss saturates
```

---

### 3.3 Test Case: Least-Confidence Sampling

**Test ID:** TC-INT-003

**Objective:** Validate uncertain sample selection

**Input:**
- 100 test samples
- Model predictions: logits for 10 classes
- Target: select 10 lowest-confidence samples

**Expected Output:**
- 10 sample paths with lowest softmax max-probability

**Validation Steps:**
1. Compute softmax probabilities ✓
2. Find indices of lowest max-prob ✓
3. Verify selected samples are distinct ✓
4. Confidence scores should be in ascending order ✓

**Success Criterion:**
```python
# Mock predictions: [confident, confident, uncertain, ...]
mock_probs = [0.95, 0.92, 0.45, 0.88, 0.32, ...]

selected_indices = np.argsort(mock_probs)[:10]
selected_probs = [mock_probs[i] for i in selected_indices]

assert all(selected_probs[i] <= selected_probs[i+1] for i in range(9))
assert selected_probs[0] < 0.5  # Most uncertain < 0.5 confidence
```

---

### 3.4 Test Case: End-to-End AL Cycle (Single Iteration)

**Test ID:** TC-INT-004

**Objective:** Validate one complete AL cycle

**Preconditions:**
- All batch files created
- Initial labeled dataset empty

**Input:**
- Cycle 0, Group selection via Thompson Sampling
- Sample 100 images from selected group
- Train for 2 epochs (short test)

**Expected Output:**
- Model checkpoint saved
- Training loss decreases monotonically
- Validation accuracy > random (>10% for CIFAR-10)

**Validation Steps:**
1. Initialize model ✓
2. Sample labeled data ✓
3. Train 2 epochs → check loss decreases ✓
4. Validate → accuracy > 10% ✓
5. Checkpoint saved ✓

**Success Criterion:**
```python
# After 1 cycle of training
assert os.path.exists('./checkpoint/main_0.pth')

checkpoint = torch.load('./checkpoint/main_0.pth')
assert checkpoint['acc'] > 10.0  # Better than random (10%)

# Loss should have decreased
assert losses[1] < losses[0]  # epoch_loss improving
```

---

## 4. Validation Against Paper

### 4.1 Algorithm 1 Verification

**Test ID:** TC-PAPER-001

**Objective:** Verify implementation matches published algorithm

**Steps:**
1. ✓ Thompson Sampling: Lines 166-171 of main.py
2. ✓ Reward function: Lines 352-356 (EMA), 359-361 (sigmoid)
3. ✓ Bernoulli update: Line 375 in main.py
4. ✓ Discount decay: Lines 378-386 in main.py
5. ✓ Group revisit: `group_list_empty` mechanism (line 187)

**Validation Method:**
```
Cross-reference each step of Algorithm 1 with corresponding code section
Ensure parameter names match (α, γ, a, b)
Verify mathematical formulas match Equations 1-4 in paper
```

---

### 4.2 Experimental Setup Verification

**Test ID:** TC-PAPER-002

**Objective:** Confirm CIFAR-10 test matches paper Scenario A

**Conditions:**
- Backbone: ResNet18 ✓
- Pretext task: Rotation prediction (0°, 90°, 180°, 270°) ✓
- AL budget: 2k → 20k (10 milestones) ✓
- Uncertainty sampling: Least confidence ✓
- Learning amount control: ν equation (Section 3.4 in paper) ✓

**Expected Results (Paper Table 1):**
| Labeled Data | Expected Acc | Implementation Range |
|--------------|--------------|----------------------|
| 2,000 | ~65% | 60-70% |
| 10,000 | ~85% | 80-90% |
| 20,000 | ~92% | 88-95% |

*Note: Exact values depend on random seed initialization*

---

### 4.3 Hyperparameter Validation

**Test ID:** TC-PAPER-003

**Objective:** Verify config.py matches paper Table 3

**Paper Table 3 → config.py Mapping:**

| Paper Param | Symbol | config.py | Expected |
|------------|--------|-----------|----------|
| EMA parameter | α | alpha | 0.1 |
| Discount factor | γ | gamma | 0.9 |
| Sigmoid scale | a | a | 2 |
| Sigmoid offset | b | b | 0.4 |
| Groups | n_G | GROUPS | 10 |
| Validation ratio | V | V | 0.01 |
| Sub-cycles | k | sampling_ratio_A | 10 |
| Cycles | n_c | CYCLES | 10 |
| Batch size | B | BATCH_SIZE | 128 |

**Validation:**
```python
from config import *

assert alpha == 0.1
assert gamma == 0.9
assert a == 2
assert b == 0.4
assert GROUPS == 10
assert V == 0.01
assert sampling_ratio_A == 10
assert CYCLES == 10
assert BATCH_SIZE == 128
```

---

## 4.5 Test Case: Loader Refactoring (loader.py - Simplified)

**Test ID:** TC-INT-LOADER

**Objective:** Validate RotationLoader after removal of redundant branching

**Change:** Removed is_train=0 vs else branching (both used same path)

**Input:**
```python
loader = RotationLoader(is_train=True, transform=transform_test)
# Note: is_train parameter now only controls train vs test behavior
# In current implementation: always loads from './DATA/train/*/*.png'
```

**Validation:**
```python
assert len(loader) == 50000  # Training set size
images = loader[0]
assert len(images) == 8  # 4 rotated images + 4 rotation labels
assert images[4] in [0, 1, 2, 3]  # rotation label
```

**Success Criterion:**
- Loader returns correct number of samples ✓
- Rotation labels are valid (0-3) ✓
- No redundant branching ✓

---

## 5. Regression Test Cases

### 5.1 Check: Output Files Exist

**Test ID:** TC-REG-001

**Objective:** Ensure all output artifacts are created

**After full pipeline:**
```python
assert os.path.exists('./DATA/train') and len(glob.glob('./DATA/train/*/*.png')) == 50000
assert os.path.exists('./checkpoint/rotation.pth')
assert os.path.exists('./loss/rotation_loss.txt')
assert all(os.path.exists(f'./loss/batch_{i}.txt') for i in range(10))
assert os.path.exists('./checkpoint/main_99.pth')  # 100 cycles (0-99)
assert os.path.exists('./performance_list.txt')
```

---

### 5.2 Check: No Data Leakage

**Test ID:** TC-REG-002

**Objective:** Validate set separation

**Conditions:**
- Validation set ⊆ Unlabeled pool
- Labeled set ⊆ Unlabeled pool (disjoint from valid)
- Test set ⊥ Train/Unlabeled

```python
assert all(path in unlabeled_pool for path in valid_)
assert all(path in unlabeled_pool for path in labeled)
assert len(set(valid_) & set(labeled)) == 0  # Disjoint
assert test_set not in unlabeled_pool  # No test leakage
```

---

## 6. Performance Benchmarks (Expected)

| Component | Time (CPU) | Time (GPU) | Memory |
|-----------|-----------|-----------|---------|
| make_data.py | 2-5 min | N/A | 500 MB |
| rotation.py | 30-60 min | 5-10 min | 4 GB |
| make_batches.py | 10-15 min | 2-3 min | 2 GB |
| main.py (100 cycles) | N/A | 5-10 hours | 6 GB |

---

## 7. Test Execution Priority

**Priority 1 (Critical - Refactored Components):**
- **TC-000-LEARNING-REFACTORED** (4 compute functions + adjust)
- **TC-001-DATA-REFACTORED** (prepare_directories, convert_cifar_to_png)
- **TC-003-BATCHES-REFACTORED** (parse_loss_file, create_groups)
- **TC-002-ROTATION-FIXED** (transform_train usage verification)
- **TC-INT-000** (end-to-end learning amount pipeline)
- **TC-INT-001, TC-INT-002** (Thompson Sampling, rewards)
- **TC-PAPER-003** (hyperparameters)

**Priority 2 (Important - Integration):**
- **TC-INT-LOADER** (refactored RotationLoader)
- **TC-INT-003, TC-INT-004** (uncertainty sampling, full cycle)
- **TC-004, TC-005** (validation set, original learning amount)
- **TC-PAPER-001, TC-PAPER-002** (algorithm match, results)

**Priority 3 (Regression):**
- **TC-REG-001, TC-REG-002** (outputs, data separation)

**Recommended Test Sequence:**

```
Session 1: Unit Tests (Base validation)
├── TC-000-LEARNING-REFACTORED
├── TC-001-DATA-REFACTORED
├── TC-003-BATCHES-REFACTORED
├── TC-002-ROTATION-FIXED
└── TC-INT-LOADER

Session 2: Integration Tests (Function interoperability)
├── TC-INT-000 (Learning Amount Pipeline)
├── TC-INT-001 (Thompson Sampling)
├── TC-INT-002 (Reward Function)
└── TC-INT-003 (Least-Confidence Sampling)

Session 3: System & Paper Alignment
├── TC-INT-004 (Full AL Cycle)
├── TC-PAPER-001 (Algorithm 1 Verification)
├── TC-PAPER-002 (Experimental Setup)
└── TC-PAPER-003 (Hyperparameter Validation)

Session 4: Regression Tests
├── TC-REG-001 (Output Files)
└── TC-REG-002 (Data Leakage Prevention)
```

**Total Test Coverage:**
- Unit tests: 5 functions × 3-4 sub-cases = 15-20 assertions
- Integration tests: 5 pipelines × 5-6 steps = 25-30 assertions
- System tests: 4 validations = 20+ assertions
- Regression tests: 2 checks = 10+ assertions
- **Total: 70-80+ test assertions**

