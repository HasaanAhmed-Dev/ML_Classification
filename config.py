"""
config.py - the ONLY file you normally need to edit.

Every setting has a safe default, so you can run the project without changing
anything. Change a value here only if you want to customise the behaviour.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Folders (relative to this file, so the project works from any location)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------
# Dataset selection
# --------------------------------------------------------------------------
# None  -> use the single .csv file found inside the "data" folder.
# "name.csv" -> use that specific file (needed only if data/ has several CSVs).
CSV_FILE_NAME = "sample_10k.csv"   # lets other CSV files sit in data/ without confusing main.py

# None -> the program detects the target column automatically.
# "column_name" -> the program uses exactly this column as the target.
TARGET_COLUMN = "Label"   # MedSec-25 setting. Use None for other datasets (auto-detect).

# Columns you want to ignore completely (they will be logged as
# "User-configured exclusion"). Example: EXCLUDE_COLUMNS = ["notes", "url"]
EXCLUDE_COLUMNS = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"]  # MedSec-25 setting (see README, section 13). Use [] for other datasets.

# --------------------------------------------------------------------------
# Train / test split and models
# --------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42

USE_CLASS_WEIGHT = True        # True -> class_weight="balanced" for both models
LR_MAX_ITER = 1000             # Logistic Regression iterations
N_ESTIMATORS = 200             # Random Forest trees
RF_MIN_SAMPLES_LEAF = 2        # mild regularisation for Random Forest
RF_MAX_DEPTH = None            # None = unlimited; e.g. 25 for smaller models
MODEL_COMPRESSION = 3          # joblib compression (0-9). Higher = smaller file

# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
TOP_FEATURES = 20              # number of features shown in importance plots

# --------------------------------------------------------------------------
# Automatic target detection
# --------------------------------------------------------------------------
MAX_TARGET_CLASSES = 20        # a target candidate may have at most this many classes

# --------------------------------------------------------------------------
# Class handling
# --------------------------------------------------------------------------
RARE_CLASS_FRACTION = 0.01     # classes below 1% of rows are reported as "rare"
REMOVE_SINGLETON_CLASSES = True  # classes with <2 rows cannot be stratified:
                                 # True  -> remove those rows (documented)
                                 # False -> keep them and split WITHOUT stratification

# --------------------------------------------------------------------------
# Identifier / cardinality rules
# --------------------------------------------------------------------------
ID_UNIQUE_RATIO = 0.95         # "nearly unique" threshold used as ONE identifier signal
HIGH_CARDINALITY_THRESHOLD = 30   # categorical columns with more unique values than this
                                  # are encoded with rare categories grouped together
MAX_ONEHOT_CATEGORIES = 30     # max one-hot columns for a high-cardinality feature
RARE_CATEGORY_MIN_FREQUENCY = 0.01  # categories rarer than 1% are grouped as "infrequent"
EXTREME_CARDINALITY_COUNT = 1000    # categorical with more unique values -> dropped
EXTREME_CARDINALITY_RATIO = 0.50    # ... or unique ratio above this (and high cardinality)

# --------------------------------------------------------------------------
# Data-leakage detection thresholds
# --------------------------------------------------------------------------
LEAKAGE_BALANCED_ACCURACY = 0.99   # one feature alone predicts target this well -> flag
LEAKAGE_CORRELATION = 0.95         # |correlation| with a binary target above this -> flag
LEAKAGE_WARNING_BALANCED_ACCURACY = 0.75  # softer warning: one feature alone is a strong predictor
