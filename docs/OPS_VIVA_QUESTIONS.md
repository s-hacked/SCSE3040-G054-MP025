# Technical Viva Q&A — Operations / MLOps (Group G054, MP025)

**Project:** Fishing Boat Catch Prediction (`catch_kg`)  
**Owner:** Abhibhav Manhas (`S24CSEU2780`)

---

## Section A: GitHub Actions CI & Testing

**Q1. What does `.github/workflows/ci.yml` do?**  
It automatically runs on every `push` and `pull_request` on an `ubuntu-latest` runner, sets up Python `3.12`, installs dependencies from `requirements.txt`, and executes `pytest -v`. If any test fails, the workflow fails and blocks bad merges.

**Q2. Why must the workflow start with `on:` and not `true:`?**  
In YAML 1.1, an unquoted `on` is parsed as boolean `true` by some YAML parsers if serialized carelessly. GitHub Actions requires the literal key `on:` to recognize workflow triggers. Our CI test `test_ci_starts_with_on_key_not_true` explicitly verifies `on` is present and `True` is not.

**Q3. Why don't CI tests depend on `data/train.csv` or `data/later.csv`?**  
Raw datasets may be large, private, or tracked externally via DVC. Unit tests in `tests/test_monitoring.py` generate small deterministic synthetic DataFrames and use mocks so CI runs fast, deterministically, and without needing data files or secrets.

**Q4. How many tests are in the suite and what do they cover?**  
70 tests total: 22 teammates' tests (`test_api.py`, `test_cleaning.py`, `test_features.py`, `test_model.py`) plus 48 operations tests in `tests/test_monitoring.py` covering PSI, harbour MAE, chronological splitting, the 12% promotion rule, v1/v2 evaluation, Kubernetes YAML validation, GitHub Actions YAML validation, and `findings.yaml` updates.

**Q5. How do you verify CI passed after pushing?**  
Push the branch to GitHub (`git push`), open the repository's **Actions** tab, select the latest workflow run, and verify the green checkmark and `70 passed` output in the `pytest -v` step.

---

## Section B: Docker & Kubernetes

**Q6. What is the difference between a Pod and a Deployment in Kubernetes?**  
A **Pod** is the smallest deployable unit (wrapping one or more containers, e.g., our FastAPI container). Pods are ephemeral—if a Pod crashes or a node dies, it is not recreated on its own. A **Deployment** manages a ReplicaSet of identical Pods, ensuring the desired number of replicas (`3`) stay running and enabling rolling updates and rollbacks.

**Q7. What is the difference between a Deployment and a Service?**  
A **Deployment** manages Pod lifecycle and scaling, where each Pod gets an ephemeral IP address. A **Service** (`mp025-svc`) provides a stable virtual IP (`ClusterIP`) and DNS name, load-balancing traffic across all Ready Pods whose labels match its selector (`app: mp025-api`).

**Q8. Why are 3 replicas configured in `k8s/deployment.yaml`?**  
Three replicas provide high availability (if one Pod restarts or fails, two remain serving), load distribution across requests, and zero-downtime rolling updates.

**Q9. What is the difference between a readiness probe and a liveness probe?**  
- **Readiness probe (`GET /health`, `initialDelaySeconds: 5`):** Determines whether a Pod is ready to receive traffic from the Service. If it fails, Kubernetes removes the Pod's IP from Service endpoints without restarting the container.
- **Liveness probe (`GET /health`, `initialDelaySeconds: 15`):** Determines whether the container is alive or deadlocked. If it fails repeatedly (`failureThreshold: 3`), Kubernetes restarts the container.

**Q10. Why must liveness `initialDelaySeconds` (15s) be longer than readiness `initialDelaySeconds` (5s)?**  
During startup, Python/FastAPI needs time to import libraries and load `models/model.joblib`. If liveness fires too early, Kubernetes could kill and restart the container in a crash loop before it finishes booting. Readiness can check earlier so traffic is simply held back until `/health` returns `200`.

**Q11. Why is `mp025-svc` of type `ClusterIP` on port `80` targeting port `8000`?**  
`ClusterIP` exposes the API internally inside the cluster without provisioning an unnecessary cloud load balancer. Clients/Ingress call standard HTTP port `80`, and the Service forwards traffic to container port `8000` where Uvicorn listens.

**Q12. How do we validate Kubernetes manifests when `kubectl` is not installed?**  
We parse `k8s/deployment.yaml` and `k8s/service.yaml` with PyYAML in `TestKubernetesYAML` (`tests/test_monitoring.py`), asserting exact names (`mp025-api`, `mp025-svc`), replicas (`3`), image (`mp025-api:1.0`), ports (`80` -> `8000`), probe paths/delays, and matching `app: mp025-api` labels. When `kubectl` is available, we run `kubectl apply --dry-run=client -f k8s/`.

---

## Section C: Data Drift Monitoring (PSI)

**Q13. What is Population Stability Index (PSI) and what is its formula?**  
PSI measures how much a numeric feature's distribution has shifted between a reference dataset (`train_cleaned`) and a comparison dataset (`later.csv`):
$$\text{PSI} = \sum_{i=1}^{10} (\text{actual}\%_i - \text{expected}\%_i) \times \ln\left(\frac{\text{actual}\%_i}{\text{expected}\%_i}\right)$$

**Q14. What are standard PSI thresholds?**  
- `PSI < 0.1`: No significant drift (stable)
- `0.1 <= PSI < 0.25`: Moderate drift (investigate)
- `PSI >= 0.25`: Significant drift (retraining strongly recommended)

**Q15. How does binning work in `compute_psi_single`?**  
We compute `10` equal-width bins (`11` bin edges) from the minimum to maximum of the cleaned reference (`train_cleaned`) distribution, then apply those exact same bin edges to `later.csv`. Fixing edges on the reference ensures apples-to-apples comparison.

**Q16. How are empty bins and missing values handled?**  
`NaN` values are dropped consistently via `.dropna()`. Empty bins are guarded by adding `epsilon = 1e-6` to every bin count before converting to proportions, preventing division by zero and `ln(0)`.

**Q17. Why did we clean `train.csv` using `clean_dataset()` before computing PSI?**  
The raw `train.csv` had 46 duplicate rows, a `60x` unit mixup on `trip_hours` for `North Jetty`, and `boat` spelling variants. Computing PSI on raw data would compare corrupted training distributions against `later.csv`, producing false drift signals.

**Q18. Which feature drifted the most in MP025 and what was its PSI?**  
`distance_from_shore_km` drifted the most with `PSI = 8.618333` (severe drift). In `train_cleaned.csv` it ranged from `2.0` to `59.9` km, whereas in `later.csv` it shifted outward to `33.4`–`91.0` km. All other five numeric features had `PSI < 0.06`.

**Q19. What is the difference between data drift and concept drift?**  
- **Data drift (covariate shift):** Input feature distribution $P(X)$ changes (e.g., boats fishing much farther offshore in `later.csv`).
- **Concept drift:** The relationship $P(Y \mid X)$ between features and `catch_kg` changes over time.

**Q20. Why are `record_id`, `date`, `boat`, `harbour`, and `catch_kg` excluded from numeric PSI?**  
`record_id` is an identifier, `date` is a timestamp, `boat` and `harbour` are categorical strings (not numeric), and `catch_kg` is the target variable (not an input feature).

---

## Section D: Fairness / Group Performance Monitoring (MAE)

**Q21. How is fairness monitored in MP025?**  
By computing Mean Absolute Error ($\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$) separately for each of the three harbours (`North Jetty`, `South Jetty`, `Lagoon`) using the v1 model on `later.csv`.

**Q22. Does low harbour-wise MAE disparity prove complete algorithmic fairness?**  
No. Group-wise MAE checks predictive error parity across harbours only; it does not establish full demographic, socioeconomic, or causal fairness.

**Q23. What were the v1 MAE values by harbour and which group was worst served?**  
- `North Jetty`: `45.8467`
- `South Jetty`: **`196.1579`** (worst-served group)
- `Lagoon`: `40.8651`

**Q24. How do you ensure no target leakage when running v1 inference on `later.csv`?**  
We pass `later.csv` through `get_feature_target_split(later_df, drop_leakage=True)` from `src/mp025/features.py`, which separates `catch_kg` as `y` and drops leakage/non-feature columns before calling `model.predict(X)`.

---

## Section E: Model Retraining (v2) & Chronological Split

**Q25. Why do we split `later.csv` chronologically (70/30) instead of randomly?**  
Fishing trip data is temporal. A random split would leak future seasonal/operational patterns into training and overestimate evaluation performance. Sorting by `date` and taking the first 70% (`210` rows) for training and the final 30% (`90` rows) for evaluation mirrors real-world deployment.

**Q26. What data is v2 trained on?**  
Cleaned `train.csv` (`1000` rows) concatenated with the first 70% of `later.csv` (`210` rows) = `1210` training rows. The final 30% (`90` rows) is strictly held out.

**Q27. Why wasn't v1 overwritten during retraining?**  
V1 (`run_id: 1ed21384971d487ca5d6db4c2d859bf3`) must remain preserved in MLflow as version `1` for baseline comparison, rollback capability, and auditability.

**Q28. What model architecture is used for v2 and why?**  
`GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)` inside `build_pipeline()` from `src/mp025/features.py`—reusing the identical preprocessing pipeline and the non-linear tree ensemble architecture from v1's model search, which handles the drifted non-linear offshore distance relationship far better than linear Ridge Regression.

**Q29. How is preprocessing protected against data leakage during v2 training?**  
`build_pipeline()` wraps `SimpleImputer`, `StandardScaler`, and `OneHotEncoder(handle_unknown="ignore")` inside a scikit-learn `Pipeline`, which fits scalers/imputers **only** on `X_train_v2` during `.fit()` and transforms `X_check` without refitting.

---

## Section F: Model Promotion & MLflow Registry

**Q30. What is the mandatory client promotion rule?**  
Promote v2 **only if**:
$$\text{v2 MAE} \le \text{v1 MAE} \times 0.88$$
That is, v2 must achieve at least a **12% reduction** in MAE on the exact same `later_check` evaluation set.

**Q31. What were the actual v1 and v2 MAE values on `later_check`, and was v2 promoted?**  
- `v1 MAE on check`: `84.5629`
- `Required threshold (84.5629 * 0.88)`: `74.4154`
- `v2 MAE on check`: `46.3200`
- Since `46.3200 <= 74.4154` (a 45.22% MAE reduction), the decision is **`promote`**.

**Q32. How does MLflow manage model versions and the `champion` alias?**  
Using `mlflow.set_tracking_uri("sqlite:///mlflow.db")`, v2 is logged and registered under `fishing_boat_catch_champion` as version `2`. `MlflowClient().set_registered_model_alias("fishing_boat_catch_champion", "champion", "2")` moves the `@champion` alias to version `2` while keeping version `1` intact.

**Q33. What happens if v2 fails the 12% threshold?**  
`apply_promotion_rule()` returns `"keep_v1"`. Version `1` retains the `@champion` alias, `models/model.joblib` is not overwritten, and `findings.yaml` records `decision: keep_v1`.

**Q34. What MLflow issue did you encounter and how did you fix it?**  
The existing `mlflow.db` had `experiments.artifact_location` set to `file:G:/Github-projects/MLOPS/mlruns/1` from another team member's machine, causing `FileNotFoundError` on `mlflow.sklearn.log_model`. We updated `artifact_location` in `mlflow.db` to the local `mlruns/1` directory and loaded v1 from `models/model.joblib` for comparison.

---

## Section G: DVC, Reproducibility & Artifacts

**Q35. How is DVC connected in this repository?**  
DVC (`3.67.1`) is initialized in `.dvc/`, `data/later.csv` is tracked via `data/later.csv.dvc` (MD5 hash stored in Git while the file is ignored via `data/.gitignore`), and a local DVC remote is configured and populated with `dvc push`.

**Q36. Why do we track `later.csv` with DVC instead of committing large data directly to Git?**  
Git is designed for source code, not large or evolving datasets. DVC stores lightweight `.dvc` pointer metadata in Git while versioning the actual data blobs in the DVC cache/remote, keeping repository history clean and reproducible.

**Q37. Where are the monitoring results and holdout predictions saved?**  
- `findings.yaml`: Updated `monitoring` section (`psi`, `most_drifted_feature`, `mae_by_group`, `worst_served_group`, `mae_v1_on_check`, `mae_v2_on_check`, `decision`).
- `reports/monitoring_report.json`: Complete JSON summary including `champion_version_after: 2`.
- `predictions/holdout_predictions.csv`: 250 holdout predictions (`record_id,prediction`) generated by the promoted champion model.

**Q38. Why does `app/main.py` continue to work seamlessly after v2 promotion?**  
Because `models/model.joblib` was updated with the promoted v2 scikit-learn `Pipeline` (which accepts the same raw feature columns as v1) and `mlflow` `@champion` points to version `2`.

**Q39. How can someone reproduce the entire monitoring + retraining run from scratch?**  
Run `python -m src.mp025.monitoring params.yaml` inside the `.venv` environment. All random seeds (`random_state=42`), bin edges, and chronological splits are deterministic.

**Q40. What if `later.csv` has missing values or unseen boat/harbour categories?**  
Numeric `NaN`s in PSI are dropped cleanly; in model inference/training, `SimpleImputer(strategy="median")` imputes missing numeric values and `OneHotEncoder(handle_unknown="ignore")` safely encodes unseen categories without crashing.
