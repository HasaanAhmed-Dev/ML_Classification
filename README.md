# ML Classification Project (Logistic Regression vs Random Forest)

A reusable, ready-to-run machine-learning project for **any CSV classification dataset**.
You put one CSV file in the `data/` folder and run `python main.py`. The program inspects the
file, finds the target column, cleans the data, trains **Logistic Regression** and
**Random Forest**, and writes a full set of results (metrics, plots, reports) to `results/`.

Nothing in the code depends on a particular dataset: column names, types, class names and
cleaning decisions all come from inspecting your CSV.

---

## 1. Quick start (Windows + VS Code)

1. Install **Python 3.9+** from python.org (tick *"Add Python to PATH"*).
2. Unzip `ML_Classification_Project.zip`.
3. In VS Code: **File -> Open Folder...** and choose the `ML_Classification_Project` folder.
4. **Copy your CSV into the `data/` folder** (`ML_Classification_Project/data/your_file.csv`).
   Keep only one CSV there (or set `CSV_FILE_NAME` in `config.py`).
5. Open a terminal in VS Code (**Terminal -> New Terminal**) and run:

```powershell
python -m venv .venv
```

Activate the environment:

- PowerShell: `.venv\Scripts\Activate.ps1`
- CMD: `.venv\Scripts\activate`

> If PowerShell blocks the script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (or use CMD instead).
> When VS Code asks "use this environment?", choose **Yes**.

```powershell
pip install -r requirements.txt
python main.py
```

Results appear in `results/` and trained models in `models/`.

---

## 2. What the project does (in order)

| Step | What happens |
|------|--------------|
| 1 | Load the CSV (tries several encodings and delimiters) |
| 2 | Inspect it: shape, dtypes, duplicates, missing values, unique values, numeric summary |
| 3 | Detect the target column |
| 4 | Clean the data and log every removed column with a reason |
| 5 | Detect feature types (numeric, boolean, categorical, datetime) |
| 6 | Analyse class distribution and handle classes that are too small |
| 7 | Split 80/20 (`random_state=42`, stratified) **before** any preprocessing |
| 8 | Run leakage checks on the training data |
| 9 | Train both models (preprocessing is inside each Pipeline) |
| 10 | Evaluate, plot, compare, and save everything |

## 3. Target detection

1. If `TARGET_COLUMN` in `config.py` is set, that column is used.
2. Otherwise every column is scored and the highest score wins. Ties go to the column that
   appears later in the file. Columns are rejected if they are empty, constant, continuous,
   have more than `MAX_TARGET_CLASSES` (20) values, have too many unique values for the number
   of rows, or look like an identifier/date by name.

| Signal | Score |
|--------|-------|
| text/boolean column | +1.5 |
| integer column with few values | +1.0 |
| 2 classes / 3-10 classes / more | +1.0 / +0.7 / +0.2 |
| name contains a word like target, label, class, outcome, status, churn... | +2.0 |
| last column in the file | +2.0 |
| class balance (0 to 1) | +0 to +1 |
| a class with fewer than 2 rows | -0.5 |

The top 5 candidates and the selected target are printed before training.
**This is a heuristic, not magic.** If the wrong column is chosen (for example when the true label is
in the middle of the file and another small categorical column looks more "target-like"),
set `TARGET_COLUMN = "your_column"` in `config.py` and run again.

## 4. Cleaning, missing values and feature detection

Safe, general rules only (nothing dataset-specific):

- Whitespace is stripped; blank cells and `?` become missing; `inf` becomes missing.
- Text columns that are at least 90% numeric are converted to numbers (invalid entries become missing).
- Empty rows, empty columns, rows with a missing target, and duplicate rows are removed.
- Constant columns are removed.
- Every removed column is saved with its reason in `results/removed_features.csv`.
- Missing numeric values: `SimpleImputer(strategy="median")`. Missing categorical/boolean values:
  `SimpleImputer(strategy="most_frequent")`. Both are fitted **on training data only**.
- **Boolean** columns (true/false, yes/no, y/n, t/f) are converted to 0/1.
- **Datetime** columns (text that looks like dates) are replaced by year, month, day, day-of-week
  (and hour if it varies). Components that never change are dropped.
- **Numeric** columns are scaled with `StandardScaler` for Logistic Regression only.

### Identifier detection
A column is removed as identifier-like only when **several signals agree**, for example:
an ID-like name (`id`, `customer_id`, `ref`...) with mostly unique values; sequential integers;
a `prefix + digits` pattern; or nearly unique free-text. A continuous numeric column with many
unique values is **not** removed just for that. Columns with weak signals are kept and listed in the log.
Duplicate detection runs after identifiers are removed, so the same record stored under two
different ids is also caught.

### Categorical encoding and high cardinality
- Up to `HIGH_CARDINALITY_THRESHOLD` (30) categories: `OneHotEncoder(handle_unknown="ignore")`.
- More categories: one-hot encoding with rare categories grouped into an "infrequent" column
  (`MAX_ONEHOT_CATEGORIES`, `RARE_CATEGORY_MIN_FREQUENCY`), learned on training data only.
- Extremely high cardinality (over 1000 values, or more than 50% of rows are distinct) is removed
  and logged.

## 5. Class distribution and rare classes

`results/class_distribution.csv/.png` show counts and percentages. Rare classes (under 1%) are
reported but **never** removed. A class with only 1 row cannot be stratified (the split cannot
put one row in both train and test), so if `REMOVE_SINGLETON_CLASSES = True` only those rows
are removed and documented (dataset summary and leakage report); if `False` they are kept and
the split is not stratified.

## 6. Train/test split and data-leakage prevention

- 80% / 20%, `random_state=42`, stratified when possible. The split happens first.
- Imputers, scalers and encoders are inside the Pipeline, fitted only on the training split.
- No hyperparameter tuning is done, so the test set is never used for decisions.
- The target is never a feature.
- Suspicious features are **flagged, not removed**: names containing the target name, a single
  feature that predicts the target almost perfectly, near-perfect correlation with a binary target.
  See `results/data_leakage_report.txt`. Very high scores usually deserve a look at this file.

## 7. Models

- **Logistic Regression**: `max_iter=1000`, `class_weight="balanced"`, scaled numeric features.
- **Random Forest**: `n_estimators=200`, `random_state=42`, `class_weight="balanced"`,
  `n_jobs=-1`, `min_samples_leaf=2` (mild regularisation).
- Class imbalance is handled with `class_weight="balanced"` (set `USE_CLASS_WEIGHT = False` to turn off).

## 8. Evaluation outputs

- **Metrics**: accuracy, macro and weighted precision/recall/F1 (`zero_division=0`). In the terminal,
  "Precision", "Recall" are the *weighted* averages.
- **Confusion matrices**: counts and row-normalised side by side, with class labels.
- **ROC/AUC** from `predict_proba`: binary problems use one ROC curve (positive class = the class sorted last);
  multiclass problems use One-vs-Rest per class plus a macro average. Classes absent from the test set
  are skipped (AUC is undefined for them). Results are in `auc_results.csv`.
- **Feature importance**: Random Forest uses impurity importance (can favour features with many
  distinct values). Logistic Regression uses the mean of absolute coefficients over classes
  (for two classes, simply the absolute coefficient).
- **Overfitting analysis**: train vs test accuracy and weighted F1, with gaps. Wording is cautious:
  *relatively small / moderate / substantial gap / possible overfitting*. A train score above the test
  score is normal and is not called overfitting by itself.
- **Model comparison**: models are **not** ranked by accuracy alone. Macro F1 is the primary criterion
  (it treats every class equally), with weighted F1, AUC and gaps as supporting evidence.

## 9. Generated files

`results/`: `dataset_summary.txt`, `feature_summary.csv`, `missing_values.csv/.png`, `removed_features.csv`,
`preprocessing_decisions.txt`, `class_distribution.csv/.png`, `data_leakage_report.txt`,
`classification_report_*.txt`, `*_confusion_matrix.png`, `*_roc_curve.png`, `auc_results.csv`,
`*_feature_importance.csv/.png`, `overfitting_analysis.csv`, `model_comparison.csv/.png`,
`per_class_f1.png` (multiclass only), `final_results.txt`.

`models/`: `logistic_regression_model.joblib`, `random_forest_model.joblib`, `model_metadata.joblib`
(class names, feature list, target name).

Use a saved model:

```python
import joblib, pandas as pd
model = joblib.load("models/random_forest_model.joblib")
meta = joblib.load("models/model_metadata.joblib")
new_rows = pd.read_csv("new_data.csv")[meta["feature_columns"]]   # same cleaned column types as training
codes = model.predict(new_rows)
labels = [meta["class_names"][c] for c in codes]
```

## 10. Configuration (`config.py`)

`TARGET_COLUMN`, `CSV_FILE_NAME`, `EXCLUDE_COLUMNS`, `TEST_SIZE`, `RANDOM_STATE`, `USE_CLASS_WEIGHT`,
`N_ESTIMATORS`, `TOP_FEATURES`, and the cleaning thresholds, all documented inside the file.

## 11. Large files and speed

A 40 MB CSV (about 300,000 rows) took roughly 4 minutes in testing, almost all of it Random Forest training,
and produced a Random Forest model of about 165 MB. To make it faster/smaller, lower `N_ESTIMATORS`, set
`RF_MAX_DEPTH = 20`, or raise `RF_MIN_SAMPLES_LEAF` in `config.py`.

## 12. Troubleshooting

| Message | Fix |
|---------|-----|
| `No CSV file found` | Put the CSV in the `data` folder |
| `Several CSV files found` | Keep one, or set `CSV_FILE_NAME` |
| `TARGET_COLUMN ... is not a column` | Check spelling against the column list shown |
| Wrong target auto-detected | Set `TARGET_COLUMN` |
| Scores near 100% | Read `data_leakage_report.txt`; exclude suspicious columns via `EXCLUDE_COLUMNS` |
| `pip` or `python` not found | Reinstall Python with "Add to PATH", or use `py -m venv .venv` |

---

## 13. MedSec-25 / paper comparison (extra)

This project was also used to compare against Hananto et al., "Characterization of Network Traffic
Features for Intrusion Detection in IoMT Cybersecurity" (INFORMATIK 22(2), 2026).

**IMPORTANT - capture-time leakage in MedSec-25.** Each attack stage was recorded in a different
session, so the `Timestamp` column (and to a lesser degree IP addresses and ports) almost identifies the class.
With these columns included, every model scores about 99.9-100% and the top features are timestamp parts.
`config.py` therefore excludes `Src IP`, `Dst IP`, `Src Port`, `Dst Port` and `Timestamp`
(`Flow ID` is removed automatically). The leakage report now also flags every date/time-derived feature
and softer "strong single-feature predictor" cases.

Steps (10,000-flow stratified sample as in the paper):

1. Put the sample CSV in `data/` (only one CSV there).
2. `python main.py` - the standard Logistic Regression vs Random Forest project.
3. `python paper_comparison.py` - adds XGBoost, MCC, AUPRC, ANOVA/MI rankings, Kruskal-Wallis,
   repeated splits, the compact "MI intersect ANOVA" feature core, and the leakage demonstration.
   Outputs: `results/paper_comparison/` (CSV tables, `summary.txt`, 3 figures).

4. `python reproduce_paper.py` - tries to reproduce the paper's sample statistics (class mix, max MI,
   max ANOVA F) with several sampling methods. Needs the FULL MedSec-25 CSV in the folder `full_data/`
   (not in `data/`). Output: `results/paper_comparison/reproduction_attempt.csv`.
5. `python time_aware_experiment.py` - our extension: time-aware split (first 80% of each class's
   timeline for training, last 20% for testing; Timestamp is used only to order rows, never as a feature) and
   Exfiltration threshold tuning (threshold chosen on a validation block, never on the test block).
   Needs the 10,000-flow sample (with its `Timestamp` column) in `data/`.
   Output: `results/paper_comparison/time_aware/`.

---

## 14. Full-stack deployment: Flask API + Angular app + GitHub CI/CD

Follows the "ML Model -> Flask -> Angular -> GitHub CI/CD" lab manual, applied to THIS project.
The model is the Random Forest from `main.py`; the form has one number box per model feature (65 for MedSec-25).

**Extra tools:** Node.js LTS (24 recommended; Angular 22 needs 22.22.3 or newer), Angular CLI
(`npm install -g @angular/cli`), Git, and in `.venv`: `pip install flask requests pytest`.

**One-time setup (project root, `.venv` active)**

```powershell
python main.py                           # trains the models (sample CSV in data/)
python backend/package_model.py          # -> backend/models/model_bundle.joblib
python backend/make_payload.py           # -> sample flows, test payload, feature schema
python backend/make_requirements.py      # -> backend/requirements.txt pinned to YOUR versions
python -m pytest backend -v              # API tests (structure test passes after the frontend exists)
ng new frontend --routing --style=css --skip-git --defaults     # answer N to the usage-data question
python setup_frontend.py                 # fills the Angular app with our form (65 fields)
```

**Daily use: three terminals**

| Terminal | Folder | Command | Keep open |
|---|---|---|---|
| 1 Flask | project root | `python backend/app.py` | yes |
| 2 Angular | `frontend` | `ng serve --open --proxy-config proxy.conf.json` | yes |
| 3 commands | project root | git, pytest, curl | free |

Open http://localhost:4200, click an example flow, then "Run local prediction".

**GitHub:** `git init`, commit, create an EMPTY repository on github.com, `git remote add origin ...`, `git branch -M main`,
`git push -u origin main`. Then Settings -> Pages -> Source -> **GitHub Actions**. The workflows in `.github/workflows/`
run the tests/build (`ci.yml`) and publish the static Angular page (`pages.yml`).

**Limits (state these in the report):** GitHub Pages hosts only the static Angular page. It cannot run Python,
so the public page shows stored results of the example flows; live predictions need Flask running locally.
The model is a research prototype trained on a public dataset with a random split.
