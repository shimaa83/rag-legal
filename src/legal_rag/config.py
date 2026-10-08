from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ============================================================
# Input / Output Files
# ============================================================

RAW_JSON_PATH = RAW_DATA_DIR / "civil_code.json"

PROCESSED_JSON_PATH = (
    PROCESSED_DATA_DIR / "civil_code.json"
)


# ============================================================
# Arabic Text Processing
# ============================================================

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"

ENGLISH_DIGITS = "0123456789"


# ============================================================
# Legal Code
# ============================================================

# Articles 54–80 were repealed.
REPEALED_ARTICLES = set(range(54, 81))