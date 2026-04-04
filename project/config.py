# Configuration file for the project
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

WANDB_PROJECT = "tooth-detection-project"
WANDB_NAME = "eda-exploration"