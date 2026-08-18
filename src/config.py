"""Shared configuration values for the maintenance data workflow.

Keeping project paths and reproducibility settings in one place makes the
synthetic-data generator and notebooks easier to reuse without hard-coded
machine-specific paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "maintenance_work_orders.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "maintenance_ml.csv"

RANDOM_SEED = 42
