# Operations & MLOps Documentation — Group G054 (MP025)

**Project:** Fishing Boat Catch Prediction (`catch_kg`)  
**Client:** Lucknow Fisherfolk Cooperative  
**Course:** SCSE3040 MLOps Mini Project  
**Group ID:** G054 | **Problem ID:** MP025  
**Operations / MLOps Owner:** Abhibhav Manhas (`S24CSEU2780`)

---

## 1. Assigned Responsibilities

This document covers the Operations / MLOps components implemented and integrated into the existing repository:

1. **GitHub Actions CI** (`.github/workflows/ci.yml`) — Automated test execution on every push and pull request.
2. **Kubernetes Manifests** (`k8s/deployment.yaml`, `k8s/service.yaml`) — Declarative deployment (`mp025-api`, 3 replicas) and internal `ClusterIP` service (`mp025-svc`).
3. **Data Drift Monitoring (PSI)** (`src/mp025/monitoring.py`) — 10-bin Population Stability Index comparing cleaned `data/train.csv` against `data/later.csv`.
4. **Fairness / Group Performance Monitoring (MAE)** (`src/mp025/monitoring.py`) — Harbour-wise MAE evaluation (`North Jetty`, `South Jetty`, `Lagoon`) using the v1 model.
5. **Model Retraining (v2)** (`src/mp025/monitoring.py`) — Chronological 70/30 split of `later.csv`, retraining v2 on `train_cleaned + later_train (70%)`.
6. **Model Evaluation & Promotion Rule** (`src/mp025/monitoring.py`) — Fair comparison of v1 vs. v2 on the held-out 30% `later_check` split, enforcing the mandatory 12% MAE reduction rule (`v2_MAE <= v1_MAE * 0.88`), and updating MLflow `@champion` alias.
7. **DVC Data Versioning** (`data/later.csv.dvc`, `.dvc/config`) — Tracking `later.csv` with DVC and pushing to the configured local DVC remote.
8. **Automated Tests & Reporting** (`tests/test_monitoring.py`, `reports/monitoring_report.json`, `findings.yaml`, `predictions/holdout_predictions.csv`).

---

## 2. Integration with Teammates' Code

Teammates' existing modules were preserved without overwriting:
- `src/mp025/data_clean.py`: Reused `clean_dataset()` and `load_config()` so drift monitoring and v2 retraining apply the exact same cleaning pipeline (removing 46 duplicates, fixing North Jetty `trip_hours` unit mixup by factor 60, normalizing `boat` spelling variants, dropping leakage column `kg_sold_at_auction`).
- `src/mp025/features.py`: Reused `get_feature_target_split()` and `build_pipeline()` so v2 uses the identical preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`) and feature columns without target or leakage columns.
- `src/mp025/train.py` & `app/main.py`: Preserved v1 training and FastAPI serving logic (`GET /health`, `POST /predict`).

### MLflow Artifact Location Fix
The original `mlflow.db` experiment (`mp025_fishing_boat_catch`, `experiment_id=1`) had `artifact_location` pointing to a developer-specific drive (`file:G:/Github-projects/MLOPS/mlruns/1`), which caused `FileNotFoundError` when logging new runs locally. We updated `artifact_location` in `mlflow.db` to point to the local `mlruns/1` directory in the repository, and loaded v1 from `models/model.joblib` for comparison so v1 remains intact and auditable.

---

## 3. CI/CD Architecture (GitHub Actions)

**File:** `.github/workflows/ci.yml`

- **Triggers:** `on: [push, pull_request]` (uses the `on:` key properly, not `true:`).
- **Runner:** `ubuntu-latest`
- **Python Version:** `3.12` via `actions/setup-python@v5` with `pip` caching.
- **Dependencies:** Installed from `requirements.txt`.
- **Test Execution:** Runs `pytest -v`. Any failing test fails the CI job.
- **CI Independence:** All unit tests in `tests/test_monitoring.py` and teammates' tests use synthetic DataFrames, mocks, or YAML parsing — they do **not** require raw CSV files, secrets, or a live Kubernetes cluster.

### Verifying CI on GitHub
1. Commit and push the branch to GitHub (`git push origin <branch>`).
2. Open the repository on GitHub and click the **Actions** tab.
3. Confirm the workflow run triggered by the push passes with a green checkmark across all 70 tests.

---

## 4. Kubernetes Deployment & Service

**Files:** `k8s/deployment.yaml`, `k8s/service.yaml`

### Deployment (`mp025-api`)
- **Replicas:** `3` — provides high availability, zero-downtime rolling updates, and load distribution across pods.
- **Image:** `mp025-api:1.0`
- **Container Port:** `8000` (FastAPI / Uvicorn)
- **Pod Labels & Selector:** `app: mp025-api`
- **Resource Requests & Limits:**
  - Requests: `256Mi` memory, `100m` CPU
  - Limits: `512Mi` memory, `500m` CPU
- **Health Probes:**
  - **Readiness Probe:** `HTTP GET /health` on port `8000`, `initialDelaySeconds: 5`, `periodSeconds: 10`, `timeoutSeconds: 3`, `failureThreshold: 3`. Controls whether the Pod receives Service traffic.
  - **Liveness Probe:** `HTTP GET /health` on port `8000`, `initialDelaySeconds: 15`, `periodSeconds: 20`, `timeoutSeconds: 3`, `failureThreshold: 3`. Longer initial delay than readiness ensures the app finishes loading the model before Kubernetes considers restarting it.

### Service (`mp025-svc`)
- **Type:** `ClusterIP` (internal cluster routing; avoids exposing unnecessary external load balancers).
- **Port:** `80` -> **TargetPort:** `8000`
- **Selector:** `app: mp025-api` (matches the Deployment's Pod template labels).

### Traffic Flow
Client / Ingress -> `mp025-svc` (`ClusterIP`, port `80`) -> Service endpoints (healthy Ready Pods matching `app: mp025-api`) -> Container port `8000` (FastAPI `app/main.py`).

---

## 5. Data Drift Monitoring (PSI)

Population Stability Index (PSI) measures how much a feature's distribution has shifted between the cleaned reference training set (`data/processed/train_cleaned.csv`, 1000 rows) and the later production batch (`data/later.csv`, 300 rows).

### Formula & Binning
$$\text{PSI} = \sum_{i=1}^{10} (\text{actual}\%_i - \text{expected}\%_i) \times \ln\left(\frac{\text{actual}\%_i}{\text{expected}\%_i}\right)$$

- **Binning:** Exactly `10` equal-width bins computed from the reference (`train_cleaned`) distribution (`min` to `max`). Bin edges are fixed on the reference distribution and reused on `later.csv`.
- **Stability Guards:** Missing values (`NaN`) are dropped consistently; a small smoothing constant $\epsilon = 10^{-6}$ is added to bin counts before normalization to prevent division by zero or $\ln(0)$.
- **Excluded Columns:** `record_id`, `date`, categorical columns (`boat`, `harbour`), and target `catch_kg`.

### Computed PSI Results

| Feature | PSI Value | Interpretation |
| :--- | :---: | :--- |
| `trip_hours` | `0.029426` | Stable (< 0.1) |
| `distance_from_shore_km` | **`8.618333`** | **Severe Drift (> 0.25)** |
| `crew` | `0.027228` | Stable (< 0.1) |
| `wave_height_m` | `0.054356` | Stable (< 0.1) |
| `water_temp_c` | `0.038976` | Stable (< 0.1) |
| `net_length_m` | `0.033900` | Stable (< 0.1) |

- **Most Drifted Feature:** `distance_from_shore_km` (`8.618333`) — in `train_cleaned.csv`, `distance_from_shore_km` ranges from `2.0` to `59.9` km, whereas in `later.csv` it shifts outward to `33.4`–`91.0` km.

---

## 6. Fairness / Group Performance Monitoring (MAE)

To assess operational fairness across landing harbours, the existing v1 model (`models/model.joblib`, Ridge Regression) was evaluated on `data/later.csv` without target leakage, computing Mean Absolute Error per harbour:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

*Note: Group-wise MAE measures predictive error parity across harbours; it does not by itself establish complete algorithmic or demographic fairness.*

### Computed Harbour-Wise MAE (v1 on `later.csv`)

| Harbour | MAE (`catch_kg`) |
| :--- | :---: |
| `North Jetty` | `45.8467` |
| **`South Jetty`** | **`196.1579`** |
| `Lagoon` | `40.8651` |

- **Worst-Served Group:** `South Jetty` (`196.1579` kg MAE).

---

## 7. Model Retraining (v2) & Chronological Split

To adapt to the shifted `later.csv` distribution without data leakage:

1. `data/later.csv` (300 rows) is sorted chronologically by `date`.
2. It is split strictly by time:
   - **First 70% (`later_train`):** 210 rows — combined with `train_cleaned.csv` (1000 rows) to form 1210 training rows.
   - **Final 30% (`later_check`):** 90 rows — held out completely during training and used solely as the common evaluation benchmark.
3. **Model Architecture:** `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)` wrapped in the same `build_pipeline()` preprocessor from `src/mp025/features.py` (matching the best-performing non-linear architecture from v1's model search).
4. **v1 Preservation:** v1 remains registered as version `1` in MLflow (`run_id: 1ed21384971d487ca5d6db4c2d859bf3`) for full auditability.

---

## 8. Model Evaluation & Promotion Decision

Both v1 and v2 are evaluated on the exact same 90-row `later_check` split:

- **v1 MAE on `later_check`:** `84.5629`
- **v2 MAE on `later_check`:** `46.3200`
- **Mandatory Promotion Threshold (12% reduction):**
  $$\text{v2 MAE} \le \text{v1 MAE} \times 0.88 \implies 46.3200 \le 84.5629 \times 0.88 = 74.4154$$
- **Result:** `46.3200 <= 74.4154` (**45.22% MAE reduction**, well beyond the 12% requirement).
- **Decision:** **`promote`**

### MLflow Registry & Artifacts Updated
- **Tracking URI:** `sqlite:///mlflow.db`
- **Experiment:** `mp025_fishing_boat_catch`
- **Registered Model:** `fishing_boat_catch_champion`
  - **Version 1:** Original Ridge Regression v1 (`run_id: 1ed21384971d487ca5d6db4c2d859bf3`)
  - **Version 2 (`@champion`):** Promoted GradientBoostingRegressor v2 (`run_id: 37ebe6bee16a4cb7aae2722919aef596`)
- **Saved Outputs:**
  - `findings.yaml` (all `monitoring` fields updated)
  - `reports/monitoring_report.json`
  - `predictions/holdout_predictions.csv` (250 rows: `record_id,prediction`)
  - `models/model.joblib` (updated with promoted champion v2)

---

## 9. How to Run Everything

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run the full test suite (70 tests)
pytest -v

# 3. Run the monitoring, retraining, promotion, and holdout prediction pipeline
python -m src.mp025.monitoring params.yaml

# 4. Validate Kubernetes manifests (requires kubectl)
kubectl apply --dry-run=client -f k8s/
```

---

## 10. Troubleshooting, Limitations & Future Work

- **Troubleshooting — MLflow `G:/` artifact path:** If moving `mlflow.db` across machines, ensure the experiment `artifact_location` in SQLite matches the local `mlruns/` path.
- **Troubleshooting — CI without CSVs:** `tests/test_monitoring.py` uses synthetic DataFrames so CI passes cleanly even when raw CSVs are gitignored or tracked via DVC.
- **Limitations & Future Work:**
  - PSI currently uses equal-width bins on `[min, max]` of the reference distribution; quantile-based binning or clipping out-of-range values to `[-inf, +inf]` could improve sensitivity when future batches fall completely outside the historical range.
  - Harbour-wise MAE highlights disparity (`South Jetty` = `196.16` vs `Lagoon` = `40.87`), warranting post-promotion group-wise evaluation on v2 and potential harbour-specific calibration or features.
  - Automated scheduled retraining (e.g., via GitHub Actions cron or DVC pipelines) can trigger retraining automatically whenever PSI exceeds `0.25`.
