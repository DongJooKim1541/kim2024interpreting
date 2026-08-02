"""Configuration with environment variable support"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Active Learning parameters
alpha = float(os.getenv("ALPHA", "0.1"))
gamma = float(os.getenv("GAMMA", "0.9"))
a = float(os.getenv("REWARD_SCALE", "2"))
b = float(os.getenv("REWARD_THRESHOLD", "0.4"))

sampling_ratio_A = int(os.getenv("SAMPLING_RATIO", "10"))
GROUPS = int(os.getenv("GROUPS", "10"))
V = float(os.getenv("VALIDATION_PERCENT", "0.01"))

# Training parameters
EPOCHS = int(os.getenv("EPOCHS", "200"))
CYCLES = int(os.getenv("CYCLES", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
LR = float(os.getenv("LR", "0.1"))
MOMENTUM = float(os.getenv("MOMENTUM", "0.9"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "5e-4"))

# Data and checkpoint paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "DATA")))
LOSS_DIR = Path(os.getenv("LOSS_DIR", str(PROJECT_ROOT / "loss")))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoint")))

def ensure_directories_exist() -> None:
    """Create required directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

