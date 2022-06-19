import os
from pathlib import Path

ROOT_PATH = Path.cwd().parent
iter_changes = "fresh_rolling_train"  # label for changes in this run iteration
OUTPUT_PATH = ROOT_PATH / "dataset" / iter_changes
Path.mkdir(OUTPUT_PATH, exist_ok=True)
